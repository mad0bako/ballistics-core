"""Ствол: геометрия, нарезы, прочность, жёсткость, масса.

Прочность считается по толстостенному цилиндру (задача Ламе) с критерием
Мизеса, потому что для ствола отношение b/a редко бывает меньше 1.5 и
тонкостенные формулы врут в разы.

Напряжения на внутренней поверхности канала при внутреннем давлении p
(наружное — атмосферное, торцы закрыты затвором):

    sigma_theta =  p (b^2 + a^2) / (b^2 - a^2)     окружное (растяжение)
    sigma_r     = -p                               радиальное (сжатие)
    sigma_z     =  p a^2 / (b^2 - a^2)             осевое

Эквивалентное по Мизесу на внутренней поверхности сводится к красивому:

    sigma_экв = sqrt(3) * p * b^2 / (b^2 - a^2)

откуда предел упругости стенки:

    p_упр = sigma_т * (b^2 - a^2) / (sqrt(3) * b^2)

Полное пластическое (разрывное) давление:

    p_разр = (2/sqrt(3)) * sigma_т * ln(b/a)

Автофретирование (предварительный наклёп внутреннего слоя до радиуса rho)
поднимает предел упругости до

    p_авт = (2 sigma_т / sqrt(3)) * [ ln(rho/a) + (b^2 - rho^2) / (2 b^2) ]

Это ровно то, чем артиллерия получает лёгкий ствол на 400 МПа.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .materials import Metal, STEEL_4140


@dataclass
class Rifling:
    """Нарезы."""

    grooves: int = 6
    groove_depth: float = 0.10e-3      # м (глубина нареза)
    land_ratio: float = 0.5            # доля периметра, занятая полями
    twist: float = 0.305               # м/оборот (шаг нарезов)
    gain_twist_start: float | None = None  # прогрессивная крутизна: шаг у казны

    def twist_at(self, x: float, length: float) -> float:
        """Шаг нарезов в сечении x (для прогрессивной нарезки)."""
        if self.gain_twist_start is None or length <= 0.0:
            return self.twist
        frac = min(max(x / length, 0.0), 1.0)
        return self.gain_twist_start + (self.twist - self.gain_twist_start) * frac

    @property
    def twist_calibers(self) -> float:
        return self.twist


@dataclass
class BarrelProfile:
    """Наружный контур: список (x от казённого среза, наружный диаметр)."""

    stations: list[tuple[float, float]]

    def outer_diameter(self, x: float) -> float:
        st = self.stations
        if x <= st[0][0]:
            return st[0][1]
        if x >= st[-1][0]:
            return st[-1][1]
        for i in range(1, len(st)):
            x0, d0 = st[i - 1]
            x1, d1 = st[i]
            if x <= x1:
                f = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
                return d0 + f * (d1 - d0)
        return st[-1][1]

    @staticmethod
    def tapered(length: float, d_breech: float, d_muzzle: float,
                chamber_length: float = 0.0,
                d_chamber: float | None = None) -> "BarrelProfile":
        """Типовой контур: цилиндр под патронник, дальше конус к дулу."""
        d_chamber = d_chamber if d_chamber is not None else d_breech
        st = [(0.0, d_chamber)]
        if chamber_length > 0.0:
            st.append((chamber_length, d_chamber))
            st.append((chamber_length * 1.15, d_breech))
        st.append((length, d_muzzle))
        return BarrelProfile(st)

    @staticmethod
    def cylindrical(length: float, outer_diameter: float) -> "BarrelProfile":
        return BarrelProfile([(0.0, outer_diameter), (length, outer_diameter)])


@dataclass
class StressStation:
    """Результат прочностного расчёта в одном сечении."""

    x: float
    pressure: float
    outer_diameter: float
    inner_diameter: float
    wall_thickness: float
    hoop_stress: float
    von_mises: float
    yield_strength: float
    safety_factor: float           # по началу текучести на поверхности канала
    elastic_limit_pressure: float
    burst_pressure: float
    burst_safety_factor: float     # по полному разрушению стенки
    hoop_strain: float          # относительное расширение канала
    radial_expansion: float     # м, упругое раздутие радиуса канала


@dataclass
class Chamber:
    """Патронник (камора)."""

    volume: float                 # м^3, полный объём каморы
    length: float                 # м, длина патронника
    mouth_diameter: float         # м, диаметр у дульца
    base_diameter: float          # м, диаметр у донца
    freebore: float = 1.5e-3      # м, свободный ход до нарезов
    leade_angle: float = math.radians(1.5)


@dataclass
class Barrel:
    """Ствол целиком."""

    material: Metal = field(default_factory=lambda: STEEL_4140)
    bore_diameter: float = 7.62e-3       # м, по полям
    length: float = 0.610                # м, полная длина от казённого среза
    rifling: Rifling = field(default_factory=Rifling)
    chamber: Chamber | None = None
    profile: BarrelProfile | None = None
    lining: Metal | None = None
    lining_thickness: float = 25e-6
    autofrettage: float = 0.0            # доля стенки, доведённая до пластики
    surface_finish_ra: float = 0.4e-6    # м, шероховатость канала
    temperature: float = 293.15          # К, текущая температура стенки

    def __post_init__(self) -> None:
        if self.profile is None:
            d_out = max(self.bore_diameter * 3.2, self.bore_diameter + 12e-3)
            self.profile = BarrelProfile.tapered(
                self.length, d_out, max(self.bore_diameter * 2.2,
                                        self.bore_diameter + 6e-3),
                chamber_length=self.chamber.length if self.chamber else 0.0,
                d_chamber=d_out * 1.12)

    # --- геометрия канала ----------------------------------------------------
    @property
    def groove_diameter(self) -> float:
        return self.bore_diameter + 2.0 * self.rifling.groove_depth

    @property
    def bore_area(self) -> float:
        """Площадь поперечного сечения канала с учётом нарезов.

        S = площадь по полям + суммарная площадь канавок. Это та площадь,
        на которую давят газы; брать просто pi*d^2/4 по полям — занизить
        силу на 2-4%, а по дну нарезов — завысить.
        """
        d = self.bore_diameter
        n = self.rifling.grooves
        if n <= 0:
            return 0.25 * math.pi * d * d
        groove_width = (1.0 - self.rifling.land_ratio) * math.pi * d / n
        return (0.25 * math.pi * d * d
                + n * groove_width * self.rifling.groove_depth)

    @property
    def effective_diameter(self) -> float:
        """Диаметр круга той же площади — для теплообмена и аэродинамики."""
        return math.sqrt(4.0 * self.bore_area / math.pi)

    @property
    def travel(self) -> float:
        """Путь снаряда: от начала нарезов до дульного среза."""
        if self.chamber is None:
            return self.length
        return max(self.length - self.chamber.length - self.chamber.freebore,
                   1e-3)

    @property
    def calibers(self) -> float:
        return self.length / self.bore_diameter

    @property
    def twist_calibers(self) -> float:
        """Шаг нарезов в калибрах — привычная мера (например, 12 клб)."""
        return self.rifling.twist / self.bore_diameter

    # --- масса и стоимость ---------------------------------------------------
    @property
    def mass(self) -> float:
        n = 200
        total = 0.0
        for i in range(n):
            x = self.length * (i + 0.5) / n
            do = self.profile.outer_diameter(x)
            di = (self.chamber.base_diameter
                  if (self.chamber and x < self.chamber.length)
                  else self.groove_diameter)
            area = 0.25 * math.pi * (do * do - di * di)
            total += max(area, 0.0) * self.length / n
        return total * self.material.density

    @property
    def cost(self) -> float:
        blank = self.mass * 1.9 * self.material.cost_per_kg
        machining = 40.0 * self.material.machinability * (1.0 + self.calibers / 40.0)
        rifling_cost = 25.0 * self.rifling.grooves / 6.0
        lining_cost = 0.0
        if self.lining is not None:
            area = math.pi * self.bore_diameter * self.length
            lining_cost = (area * self.lining_thickness
                           * self.lining.density * self.lining.cost_per_kg + 60.0)
        return blank + machining + rifling_cost + lining_cost

    # --- прочность -----------------------------------------------------------
    def _inner_radius(self, x: float) -> float:
        if self.chamber is not None and x < self.chamber.length:
            return 0.5 * self.chamber.base_diameter
        return 0.5 * self.groove_diameter

    def elastic_limit_pressure(self, x: float,
                               temperature: float | None = None) -> float:
        a = self._inner_radius(x)
        b = 0.5 * self.profile.outer_diameter(x)
        if b <= a:
            return 0.0
        sy = self.material.yield_at(temperature or self.temperature)
        p_el = sy * (b * b - a * a) / (math.sqrt(3.0) * b * b)
        if self.autofrettage > 0.0:
            rho = a + min(max(self.autofrettage, 0.0), 1.0) * (b - a)
            p_af = (2.0 * sy / math.sqrt(3.0)) * (
                math.log(rho / a) + (b * b - rho * rho) / (2.0 * b * b))
            p_el = max(p_el, p_af)
        return p_el

    def burst_pressure(self, x: float,
                       temperature: float | None = None) -> float:
        """Давление полного пластического течения стенки (разрыв)."""
        a = self._inner_radius(x)
        b = 0.5 * self.profile.outer_diameter(x)
        if b <= a:
            return 0.0
        su = self.material.ultimate_strength
        scale = self.material.yield_at(temperature or self.temperature) / \
            self.material.yield_strength
        return (2.0 / math.sqrt(3.0)) * su * scale * math.log(b / a)

    def stress_at(self, x: float, pressure: float,
                  temperature: float | None = None) -> StressStation:
        a = self._inner_radius(x)
        b = 0.5 * self.profile.outer_diameter(x)
        b = max(b, a * 1.0001)
        k2 = b * b / (b * b - a * a)
        hoop = pressure * (b * b + a * a) / (b * b - a * a)
        vm = math.sqrt(3.0) * pressure * k2
        sy = self.material.yield_at(temperature or self.temperature)
        p_el = self.elastic_limit_pressure(x, temperature)
        p_burst = self.burst_pressure(x, temperature)
        e = self.material.youngs_modulus
        nu = self.material.poisson
        radial = pressure * a / e * ((b * b + a * a) / (b * b - a * a) + nu)
        return StressStation(
            x=x, pressure=pressure, outer_diameter=2.0 * b,
            inner_diameter=2.0 * a, wall_thickness=b - a,
            hoop_stress=hoop, von_mises=vm, yield_strength=sy,
            safety_factor=(sy / vm if vm > 0.0 else float("inf")),
            elastic_limit_pressure=p_el,
            burst_pressure=p_burst,
            burst_safety_factor=(p_burst / pressure if pressure > 0.0
                                 else float("inf")),
            hoop_strain=radial / a, radial_expansion=radial,
        )

    def analyse(self, envelope: list[tuple[float, float]],
                chamber_pressure: float | None = None,
                temperature: float | None = None) -> list[StressStation]:
        """Прочностной расчёт по всей длине.

        envelope — огибающая давлений (x от начала пути снаряда, p).
        Сечения патронника нагружаются максимальным казённым давлением.
        """
        out: list[StressStation] = []
        offset = (self.chamber.length + self.chamber.freebore) if self.chamber else 0.0
        p_breech = chamber_pressure if chamber_pressure is not None else (
            envelope[0][1] if envelope else 0.0)
        if offset > 0.0:
            for i in range(6):
                x = offset * i / 5.0
                out.append(self.stress_at(x, p_breech, temperature))
        for x_travel, p in envelope:
            out.append(self.stress_at(offset + x_travel, p, temperature))
        return out

    def min_safety_factor(self, envelope: list[tuple[float, float]],
                          chamber_pressure: float | None = None,
                          temperature: float | None = None,
                          criterion: str = "yield"
                          ) -> tuple[float, StressStation]:
        """Наименьший запас прочности по всей длине.

        criterion='yield' — по началу текучести на поверхности канала;
        criterion='burst' — по разрушению стенки.

        Важно понимать разницу. У боевых нарезных стволов запас по текучести
        на самой поверхности канала штатно бывает около единицы: первые
        выстрелы слегка наклёпывают внутренний слой, ствол сам себя
        автофретирует и дальше работает упруго. Аварийным считается
        исчерпание запаса по РАЗРУШЕНИЮ, и его держат не ниже 1.5-2.
        """
        stations = self.analyse(envelope, chamber_pressure, temperature)
        key = ((lambda s: s.burst_safety_factor) if criterion == "burst"
               else (lambda s: s.safety_factor))
        worst = min(stations, key=key)
        return key(worst), worst

    # --- жёсткость и колебания ----------------------------------------------
    def second_moment(self, x: float) -> float:
        do = self.profile.outer_diameter(x)
        di = self._inner_radius(x) * 2.0
        return math.pi * (do ** 4 - di ** 4) / 64.0

    @property
    def muzzle_droop(self) -> float:
        """Прогиб дульного среза под собственным весом (консоль), м.

        Считаем численно как консольную балку переменного сечения.
        Прогиб входит в бюджет рассеивания: чем мягче ствол, тем сильнее
        он «играет» при выстреле и при нагреве.
        """
        n = 100
        dx = self.length / n
        e = self.material.youngs_modulus
        # интегрируем dv/dx дважды: M(x) от распределённого веса
        theta = 0.0
        defl = 0.0
        for i in range(n):
            x = (i + 0.5) * dx
            do = self.profile.outer_diameter(x)
            di = self._inner_radius(x) * 2.0
            w_area = 0.25 * math.pi * (do * do - di * di)
            # момент в сечении x от веса участка правее
            m = 0.0
            for j in range(i, n):
                xj = (j + 0.5) * dx
                doj = self.profile.outer_diameter(xj)
                dij = self._inner_radius(xj) * 2.0
                aj = 0.25 * math.pi * (doj * doj - dij * dij)
                m += aj * self.material.density * 9.80665 * dx * (xj - x)
            curvature = m / (e * max(self.second_moment(x), 1e-12))
            theta += curvature * dx
            defl += theta * dx
            _ = w_area
        return defl

    @property
    def first_mode_frequency(self) -> float:
        """Первая собственная частота консольного ствола, Гц.

        f1 = (1.875^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A)) — приближение
        балки постоянного сечения по среднему.
        """
        n = 40
        ei = 0.0
        ra = 0.0
        for i in range(n):
            x = self.length * (i + 0.5) / n
            ei += self.second_moment(x)
            do = self.profile.outer_diameter(x)
            di = self._inner_radius(x) * 2.0
            ra += 0.25 * math.pi * (do * do - di * di)
        ei = self.material.youngs_modulus * ei / n
        ra = self.material.density * ra / n
        if ra <= 0.0 or self.length <= 0.0:
            return 0.0
        return (1.875 ** 2 / (2.0 * math.pi * self.length ** 2)) * math.sqrt(ei / ra)

    # --- нарезы --------------------------------------------------------------
    def rifling_torque(self, projectile_inertia: float, velocity: float,
                       acceleration: float) -> float:
        """Момент на ведущем пояске, Н*м: M = I * dw/dt, w = 2*pi*v/twist."""
        if self.rifling.twist <= 0.0:
            return 0.0
        return projectile_inertia * 2.0 * math.pi * acceleration / self.rifling.twist

    def land_shear_stress(self, torque: float) -> float:
        """Касательное напряжение в полях нарезов от момента закрутки."""
        n = self.rifling.grooves
        if n <= 0 or self.rifling.twist <= 0.0:
            return 0.0
        r = 0.5 * self.bore_diameter
        bearing_len = 8.0 * self.rifling.groove_depth  # длина контакта пояска
        area = n * bearing_len * self.rifling.groove_depth
        force = torque / r
        return force / max(area, 1e-12)

    def heat_capacity(self) -> float:
        """Теплоёмкость ствола, Дж/К — определяет нагрев при стрельбе очередью."""
        return self.mass * self.material.specific_heat

    def describe(self) -> str:
        return (
            f"Ствол {self.bore_diameter * 1e3:.2f} мм, длина "
            f"{self.length * 1e3:.0f} мм ({self.calibers:.0f} клб), "
            f"{self.material.name}\n"
            f"  нарезы: {self.rifling.grooves} шт, глубина "
            f"{self.rifling.groove_depth * 1e3:.3f} мм, шаг "
            f"{self.rifling.twist * 1e3:.0f} мм ({self.twist_calibers:.1f} клб)\n"
            f"  S = {self.bore_area * 1e6:.2f} мм2, масса {self.mass:.2f} кг, "
            f"частота 1-й моды {self.first_mode_frequency:.0f} Гц, "
            f"прогиб дула {self.muzzle_droop * 1e3:.2f} мм"
        )


# --- обратные задачи ---------------------------------------------------------

def required_outer_diameter(inner_diameter: float, pressure: float,
                            material: Metal, safety_factor: float = 1.4,
                            autofrettage: float = 0.0,
                            temperature: float = 293.15) -> float:
    """Минимальный наружный диаметр стенки под заданное давление.

    Из sigma_экв = sqrt(3) p b^2/(b^2-a^2) <= sigma_т / n получаем

        b^2 / (b^2 - a^2) <= sigma_т / (sqrt(3) * n * p)
        =>  b = a * sqrt(K / (K - 1)),   K = sigma_т / (sqrt(3) n p)

    Если K <= 1, никакая толщина не спасает: нужен другой материал,
    автофретирование или меньшее давление.
    """
    a = 0.5 * inner_diameter
    sy = material.yield_at(temperature)
    k = sy / (math.sqrt(3.0) * safety_factor * max(pressure, 1.0))
    if autofrettage > 0.0:
        # автофретирование даёт выигрыш; решаем численно по p_авт
        lo, hi = a * 1.001, a * 20.0
        target = safety_factor * pressure
        for _ in range(200):
            b = 0.5 * (lo + hi)
            rho = a + autofrettage * (b - a)
            p_af = (2.0 * sy / math.sqrt(3.0)) * (
                math.log(rho / a) + (b * b - rho * rho) / (2.0 * b * b))
            p_el = sy * (b * b - a * a) / (math.sqrt(3.0) * b * b)
            if max(p_af, p_el) >= target:
                hi = b
            else:
                lo = b
        return 2.0 * hi
    if k <= 1.0:
        return float("inf")
    return 2.0 * a * math.sqrt(k / (k - 1.0))


def gyroscopic_twist_miller(bullet_diameter: float, bullet_length: float,
                            bullet_mass: float, velocity: float = 823.0,
                            stability: float = 1.5) -> float:
    """Требуемый шаг нарезов по правилу Миллера (Miller twist rule).

    Формула эмпирическая и в дюймах/гранах, поэтому внутри переводим:

        Sg = 30 m / (t^2 * d^3 * L * (1 + L^2))

    где m — масса в гранах, d — калибр в дюймах, L — длина пули в калибрах,
    t — шаг в калибрах. Плюс поправка на скорость (V/2800)^(1/3).
    Возвращает шаг нарезов в метрах на оборот.
    """
    d_in = bullet_diameter / 0.0254
    m_gr = bullet_mass / 64.79891e-6
    l_cal = bullet_length / bullet_diameter
    v_fps = velocity / 0.3048
    sg_target = stability / ((v_fps / 2800.0) ** (1.0 / 3.0))
    t_sq = 30.0 * m_gr / (sg_target * d_in ** 3 * l_cal * (1.0 + l_cal * l_cal))
    if t_sq <= 0.0:
        return 0.0
    t_cal = math.sqrt(t_sq)
    return t_cal * bullet_diameter


def miller_stability(bullet_diameter: float, bullet_length: float,
                     bullet_mass: float, twist: float, velocity: float,
                     air_density_ratio: float = 1.0) -> float:
    """Коэффициент гироскопической устойчивости Sg по Миллеру."""
    if twist <= 0.0:
        return 0.0
    d_in = bullet_diameter / 0.0254
    m_gr = bullet_mass / 64.79891e-6
    l_cal = bullet_length / bullet_diameter
    t_cal = twist / bullet_diameter
    v_fps = velocity / 0.3048
    sg = 30.0 * m_gr / (t_cal ** 2 * d_in ** 3 * l_cal * (1.0 + l_cal ** 2))
    return sg * (v_fps / 2800.0) ** (1.0 / 3.0) / max(air_density_ratio, 1e-6)
