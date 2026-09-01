"""Внутренняя баллистика: основная задача выстрела.

Система уравнений — классическая (Серебряков, «Внутренняя баллистика ствольных
систем и пороховых ракет», гл. 5; Corner, гл. 5):

  закон горения (Вьель)      dz/dt = u1 * p^nu / e1
  газообразование            psi   = psi(z)                (функция формы)
  движение снаряда           phi * m * dv/dt = S * (p_дно - p_сопр)
  путь                       dl/dt = v
  уравнение Резаля           p * W = f*omega*psi - theta*(phi*m*v^2/2 + Q_ст)
  свободный объём            W = W0 - omega/delta*(1-psi) - alpha*omega*psi + S*l

где
  phi = K + omega/(3m) — коэффициент фиктивности массы: треть массы заряда
        «едет» вместе со снарядом (приближение Лагранжа), K учитывает трение;
  theta = k - 1;
  Q_ст  — теплоотдача в стенки (считается, если включена).

Распределение давления по каналу берётся лагранжево (параболическое):

    p(x) = p_дно * [1 + omega/(2*phi*m) * (1 - (x/l)^2)]
    p_кн  = p_дно * (1 + omega/(2*phi*m))
    p_ср  = p_дно * (1 + omega/(3*phi*m))

Именно p_кн нагружает казённик, p_дно разгоняет пулю, а p_ср входит в
уравнение Резаля. Путать их — классический способ получить «прочный» ствол,
который разрывает на третьем выстреле.

Интегратор — Кэш-Карп RK45 с контролем шага: артиллерийский выстрел
(20 мс) и винтовочный (1.2 мс) считаются одинаково устойчиво и быстро.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .propellant import T_REF_POWDER, Propellant

# --- коэффициенты Кэша-Карпа -------------------------------------------------
_A = (0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8)
_B = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (3 / 10, -9 / 10, 6 / 5),
    (-11 / 54, 5 / 2, -70 / 27, 35 / 27),
    (1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096),
)
_C5 = (37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771)
_C4 = (2825 / 27648, 0.0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4)


@dataclass
class Primer:
    """Капсюль-воспламенитель.

    Даёт стартовое давление и определяет надёжность/равномерность
    воспламенения (через неё — разброс скоростей).
    """

    name: str = "Стандартный винтовочный"
    charge_mass: float = 25e-6      # кг воспламенительного состава
    force: float = 0.30e6           # Дж/кг, сила состава
    ignition_delay: float = 0.15e-3  # с
    consistency: float = 0.985      # 0..1, влияет на разброс

    def gas_energy(self) -> float:
        return self.force * self.charge_mass


@dataclass
class Charge:
    """Метательный заряд."""

    propellant: Propellant
    mass: float                     # кг
    temperature: float = T_REF_POWDER


@dataclass
class GunSystem:
    """Ствольная система в терминах, которые нужны основной задаче.

    Всё, что связано с реальной геометрией ствола и патрона, приводится
    сюда адаптером (см. simulation.build_system).
    """

    bore_area: float                # S, м^2 — площадь канала с учётом нарезов
    chamber_volume: float           # W0, м^3 — объём каморы за снарядом
    travel: float                   # l_д, м — путь снаряда до дульного среза
    projectile_mass: float          # m, кг
    bore_diameter: float            # d, м (для теплообмена)
    shot_start_pressure: float = 30e6   # p0, Па — давление форсирования
    friction_coefficient: float = 1.02  # K в phi = K + omega/(3m)
    engraving_length: float = 6e-3      # м, длина врезания в нарезы
    engraving_pressure: float = 0.0     # Па, доп. сопротивление врезания
    bore_friction_pressure: float = 3e6  # Па, трение пояска по каналу
    twist: float = 0.0                  # м/оборот (0 = гладкий ствол)
    wall_temperature: float = 350.0     # К, начальная температура канала


@dataclass
class InteriorOptions:
    heat_loss: bool = True
    rtol: float = 1e-6
    dt_init: float = 1e-8
    dt_max: float = 5e-5
    max_steps: int = 200000
    sample_stride: int = 1           # прореживание истории
    # Потолок давления: выше него считать бессмысленно — ствол уже разорван,
    # а интегратор на таких режимах вязнет в микроскопических шагах. Расчёт
    # прерывается и помечается предупреждением.
    pressure_ceiling: float = 3.0e9


@dataclass
class InteriorResult:
    """Результат выстрела в стволе."""

    muzzle_velocity: float = 0.0
    p_max_breech: float = 0.0
    p_max_mean: float = 0.0
    p_max_base: float = 0.0
    x_at_pmax: float = 0.0
    t_at_pmax: float = 0.0
    time_muzzle: float = 0.0
    psi_muzzle: float = 1.0
    z_muzzle: float = 0.0
    burnout_travel: float | None = None
    max_gas_temp: float = 0.0
    muzzle_pressure: float = 0.0
    stuck: bool = False                # снаряд не вышел из ствола
    stuck_travel: float = 0.0          # где именно застрял, м
    heat_to_barrel: float = 0.0        # Дж
    thermal_efficiency: float = 0.0
    recoil_impulse: float = 0.0        # Н*с
    spin_rate: float = 0.0             # рад/с на дульном срезе
    loading_density: float = 0.0       # кг/м^3
    # история: списки одинаковой длины
    t: list[float] = field(default_factory=list)
    l: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)
    p_breech: list[float] = field(default_factory=list)
    p_mean: list[float] = field(default_factory=list)
    p_base: list[float] = field(default_factory=list)
    psi: list[float] = field(default_factory=list)
    t_gas: list[float] = field(default_factory=list)
    heat_flux: list[float] = field(default_factory=list)   # Вт/м^2 у казны
    warnings: list[str] = field(default_factory=list)

    def pressure_at_station(self, x: float) -> float:
        """Максимальное давление, которое видела точка канала на расстоянии x
        от начала пути снаряда. Это и есть нагрузка на стенку в этом сечении.
        """
        best = 0.0
        for i, li in enumerate(self.l):
            if li < x:
                continue
            pb = self.p_breech[i]
            pd = self.p_base[i]
            if li <= 0.0:
                p = pb
            else:
                ratio = min(x / li, 1.0)
                p = pd + (pb - pd) * (1.0 - ratio * ratio)
            if p > best:
                best = p
        return best

    def pressure_envelope(self, n: int = 40) -> list[tuple[float, float]]:
        """Огибающая максимальных давлений вдоль пути снаряда."""
        if not self.l:
            return []
        travel = self.l[-1]
        out = []
        for i in range(n + 1):
            x = travel * i / n
            out.append((x, self.pressure_at_station(x)))
        return out


# --- теплофизика пороховых газов --------------------------------------------

def gas_viscosity(t: float) -> float:
    """Динамическая вязкость пороховых газов, Па*с (степенная аппроксимация)."""
    return 1.8e-5 * (max(t, 200.0) / 300.0) ** 0.672


GAS_PRANDTL = 0.75


def _heat_transfer_coefficient(rho: float, u: float, d: float, t_gas: float,
                               cp: float) -> float:
    """Коэффициент теплоотдачи по Диттусу-Бёльтеру: Nu = 0.023 Re^0.8 Pr^0.4.

    Для ствола это стандартная инженерная оценка (Corner, гл. 9): течение
    турбулентное с Re порядка 10^6, вход в канал — как в трубу.
    """
    mu = gas_viscosity(t_gas)
    re = rho * max(abs(u), 1.0) * d / mu
    k_gas = mu * cp / GAS_PRANDTL
    nu = 0.023 * re ** 0.8 * GAS_PRANDTL ** 0.4
    return nu * k_gas / d


# --- решатель ----------------------------------------------------------------

def solve_interior(system: GunSystem, charge: Charge,
                   primer: Primer | None = None,
                   options: InteriorOptions | None = None) -> InteriorResult:
    """Решает основную задачу внутренней баллистики."""
    opt = options or InteriorOptions()
    primer = primer or Primer()
    prop = charge.propellant
    form = prop.grain.form

    omega = charge.mass
    m = system.projectile_mass
    s = system.bore_area
    w0 = system.chamber_volume
    delta = prop.density
    alpha = prop.covolume
    f = prop.force_at(charge.temperature)
    theta = prop.theta
    e1 = prop.web
    phi = system.friction_coefficient + omega / (3.0 * m)
    lagrange = omega / (2.0 * phi * m)
    lagrange_mean = omega / (3.0 * phi * m)
    # удельная теплоёмкость газов при постоянном давлении, Дж/(кг*К)
    cv_spec = prop.chem.cv_specific
    cp_spec = cv_spec + prop.chem.gas_constant

    res = InteriorResult()
    res.loading_density = omega / w0

    if omega / w0 > prop.grain.bulk_density:
        res.warnings.append(
            f"Плотность заряжания {omega / w0:.0f} кг/м3 выше насыпной "
            f"({prop.grain.bulk_density:.0f}): заряд физически не влезет "
            "в гильзу без прессования.")
    if omega / w0 > 0.95 * delta:
        res.warnings.append("Плотность заряжания близка к плотности пороха — "
                            "расчёт за пределами применимости модели.")

    primer_energy = primer.gas_energy()

    def free_volume(psi: float, l: float) -> float:
        return (w0 - omega * (1.0 - psi) / delta - alpha * omega * psi + s * l)

    def state_pressure(z: float, l: float, v: float, q_wall: float
                       ) -> tuple[float, float, float, float, float]:
        """Возвращает (p_ср, p_дно, p_кн, psi, T_газа)."""
        psi = form.psi(z)
        w = free_volume(psi, l)
        if w <= 1e-12:
            w = 1e-12
        energy = (f * omega * psi + primer_energy
                  - theta * (0.5 * phi * m * v * v + q_wall))
        p_mean = max(energy / w, 0.0)
        p_base = p_mean / (1.0 + lagrange_mean)
        p_breech = p_base * (1.0 + lagrange)
        mass_gas = omega * psi + primer.charge_mass
        if mass_gas > 1e-12 and prop.chem.gas_constant > 0.0:
            t_gas = p_mean * w / (mass_gas * prop.chem.gas_constant)
        else:
            t_gas = prop.flame_temp
        return p_mean, p_base, p_breech, psi, min(t_gas, prop.flame_temp * 1.05)

    def resistance(l: float) -> float:
        """Сопротивление движению, приведённое к давлению."""
        r = system.bore_friction_pressure
        if system.engraving_pressure > 0.0 and l < system.engraving_length:
            r += system.engraving_pressure * (1.0 - l / system.engraving_length)
        return r

    heat_state = {"q_flux": 0.0}

    def derivatives(t: float, y: list[float]) -> list[float]:
        z, l, v, q_wall = y
        p_mean, p_base, p_breech, psi, t_gas = state_pressure(z, l, v, q_wall)

        # горение
        if z < form.z_k:
            dz = prop.burn_rate(p_mean, charge.temperature, z) / e1
        else:
            dz = 0.0

        # движение: до давления форсирования снаряд стоит
        moving = (l > 0.0) or (p_base >= system.shot_start_pressure)
        if moving:
            drive = p_base - resistance(l)
            dv = s * drive / (phi * m)
            if l <= 0.0 and dv < 0.0:
                dv = 0.0
            dl = v
        else:
            dv = 0.0
            dl = 0.0

        # теплоотдача в стенки
        dq = 0.0
        if opt.heat_loss:
            w = free_volume(psi, l)
            mass_gas = omega * psi + primer.charge_mass
            rho_gas = mass_gas / max(w, 1e-12)
            l0 = w0 / s
            area = math.pi * system.bore_diameter * (l0 + l)
            # средний по каналу поток — для энергетического баланса
            u_mean = 0.5 * v
            h = _heat_transfer_coefficient(rho_gas, max(u_mean, 5.0),
                                           system.bore_diameter, t_gas, cp_spec)
            dt_film = max(t_gas - system.wall_temperature, 0.0)
            dq = h * dt_film * area
            # поток именно у начала нарезов — он и определяет выгорание.
            # По Лагранжу скорость газа линейна: u(x) = v * x / (l0 + l),
            # начало нарезов сидит в приведённой координате x = l0.
            u_throat = v * l0 / (l0 + l) if (l0 + l) > 0.0 else v
            h_th = _heat_transfer_coefficient(rho_gas, max(u_throat, 5.0),
                                              system.bore_diameter, t_gas,
                                              cp_spec)
            heat_state["q_flux"] = h_th * dt_film
        return [dz, dl, dv, dq]

    # --- интегрирование -------------------------------------------------------
    y = [0.0, 0.0, 0.0, 0.0]
    y_prev = list(y)
    t_prev = 0.0
    t = 0.0
    dt = opt.dt_init
    steps = 0
    burnout_recorded = False
    impulse = 0.0
    prev_pb = 0.0

    def track(t: float, y: list[float]) -> tuple[float, float, float, float, float]:
        """Обновляет экстремумы. Вызывается на КАЖДОМ принятом шаге.

        Отслеживание максимумов принципиально отделено от записи истории:
        пик давления острый (доли миллисекунды), и если искать его только
        по прореженной истории, он просто проваливается между точками —
        расчёт тихо занижает p_max в полтора раза.
        """
        z, l, v, q = y
        p_mean, p_base, p_breech, psi, t_gas = state_pressure(z, l, v, q)
        if p_breech > res.p_max_breech:
            res.p_max_breech = p_breech
            res.x_at_pmax = l
            res.t_at_pmax = t
        res.p_max_mean = max(res.p_max_mean, p_mean)
        res.p_max_base = max(res.p_max_base, p_base)
        res.max_gas_temp = max(res.max_gas_temp, t_gas)
        return p_mean, p_base, p_breech, psi, t_gas

    def record(t: float, y: list[float]) -> None:
        """Дописывает точку в историю (может прореживаться)."""
        p_mean, p_base, p_breech, psi, t_gas = track(t, y)
        res.t.append(t)
        res.l.append(y[1])
        res.v.append(y[2])
        res.p_mean.append(p_mean)
        res.p_base.append(p_base)
        res.p_breech.append(p_breech)
        res.psi.append(psi)
        res.t_gas.append(t_gas)
        res.heat_flux.append(heat_state["q_flux"])

    record(t, y)

    while steps < opt.max_steps:
        steps += 1
        # ограничиваем шаг, чтобы не проскочить дульный срез
        if y[2] > 1.0:
            dt = min(dt, 0.25 * max(system.travel - y[1], 1e-6) / y[2])
        dt = min(dt, opt.dt_max)

        k = []
        for i in range(6):
            yi = list(y)
            for j in range(i):
                bij = _B[i][j]
                if bij:
                    for n in range(4):
                        yi[n] += dt * bij * k[j][n]
            k.append(derivatives(t + _A[i] * dt, yi))

        y5 = [y[n] + dt * sum(_C5[i] * k[i][n] for i in range(6)) for n in range(4)]
        y4 = [y[n] + dt * sum(_C4[i] * k[i][n] for i in range(6)) for n in range(4)]

        scale = [max(abs(y5[0]), 1e-3), max(abs(y5[1]), 1e-4),
                 max(abs(y5[2]), 1.0), max(abs(y5[3]), 1.0)]
        err = max(abs(y5[n] - y4[n]) / scale[n] for n in range(4)) / opt.rtol

        if err > 1.0 and dt > 1e-12:
            dt *= max(0.2, 0.9 * err ** -0.25)
            continue

        # шаг принят
        y_prev, t_prev = list(y), t
        pb_mid = state_pressure(*y5[:3], y5[3])[2]
        impulse += s * 0.5 * (prev_pb + pb_mid) * dt
        prev_pb = pb_mid
        t += dt
        y = y5
        if y[0] < 0.0:
            y[0] = 0.0
        dt *= min(5.0, 0.9 * err ** -0.2) if err > 0.0 else 5.0

        if not burnout_recorded and y[0] >= form.z_k:
            burnout_recorded = True
            res.burnout_travel = y[1]

        track(t, y)
        if steps % opt.sample_stride == 0:
            record(t, y)

        if y[1] >= system.travel:
            break

        # Затяжной выстрел: заряд догорел, а энергии снаряда уже не хватит,
        # чтобы продавить оставшийся путь против трения. Дальше он только
        # тормозит — считать нечего, пуля остаётся в стволе. Это не сбой
        # расчёта, а штатно моделируемый отказ: следующий выстрел по
        # застрявшей пуле разрывает ствол.
        if y[0] >= form.z_k:
            remaining = system.travel - y[1]
            kinetic = 0.5 * phi * m * y[2] * y[2]
            work_needed = resistance(y[1]) * s * remaining
            _, p_base_now, _, _, _ = state_pressure(y[0], y[1], y[2], y[3])
            if kinetic < work_needed and p_base_now <= resistance(y[1]):
                res.stuck = True
                res.stuck_travel = y[1]
                res.warnings.append(
                    f"Снаряд застрял в стволе на {y[1] * 1e3:.0f} мм из "
                    f"{system.travel * 1e3:.0f}: заряда не хватило. "
                    "Стрелять следующим патроном нельзя — разорвёт ствол.")
                break

        if res.p_max_breech > opt.pressure_ceiling:
            res.warnings.append(
                f"Давление превысило расчётный потолок "
                f"{opt.pressure_ceiling / 1e6:.0f} МПа — ствол разрушен, "
                "интегрирование прекращено.")
            break

    # доводим ровно до дульного среза линейной интерполяцией по последнему шагу
    if y[1] > system.travel and y[1] > y_prev[1]:
        frac = (system.travel - y_prev[1]) / (y[1] - y_prev[1])
        y = [y_prev[n] + frac * (y[n] - y_prev[n]) for n in range(4)]
        t = t_prev + frac * (t - t_prev)
        y[1] = system.travel

    record(t, y)

    res.muzzle_velocity = y[2]
    res.time_muzzle = t
    res.z_muzzle = y[0]
    res.psi_muzzle = form.psi(y[0])
    res.heat_to_barrel = y[3]
    res.muzzle_pressure = res.p_base[-1]
    res.recoil_impulse = impulse
    if system.twist > 0.0:
        res.spin_rate = 2.0 * math.pi * y[2] / system.twist
    potential = f * omega / theta
    res.thermal_efficiency = (0.5 * m * y[2] ** 2) / potential if potential > 0 else 0.0

    if steps >= opt.max_steps:
        res.warnings.append("Достигнут лимит шагов интегрирования — "
                            "результат может быть недостоверен.")
    if res.psi_muzzle < 0.985:
        res.warnings.append(
            f"На дульном срезе сгорело {100 * res.psi_muzzle:.1f}% заряда: "
            "часть пороха выброшена несгоревшей, дульное пламя, потеря энергии "
            "и рост разброса скоростей.")
    if res.burnout_travel is not None and res.burnout_travel < 0.25 * system.travel:
        res.warnings.append(
            "Порох догорает в первой четверти ствола: пик давления острый, "
            "ствол изнашивается быстрее нужного. Возьмите более медленный порох.")
    if res.muzzle_pressure > 0.35 * res.p_max_breech:
        res.warnings.append(
            "Высокое дульное давление: сильная дульная волна и заметное "
            "последействие газов на отдачу.")
    return res


def aftereffect(result: InteriorResult, system: GunSystem, charge: Charge,
                beta: float = 1.30) -> tuple[float, float]:
    """Период последействия газов: прибавка скорости и импульса отдачи.

    Приближение Слухоцкого: после вылета снаряда газы ещё некоторое время
    давят на дно, добавляя (1..3)% скорости, а на ствол — заметный импульс.
    """
    m = system.projectile_mass
    omega = charge.mass
    dv = result.muzzle_pressure * system.bore_area * 0.0015 / m
    dv = min(dv, 0.03 * result.muzzle_velocity)
    impulse = result.recoil_impulse + beta * omega * result.muzzle_velocity
    return dv, impulse
