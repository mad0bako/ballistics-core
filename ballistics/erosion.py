"""Износ и нагрев ствола: сколько выстрелов он проживёт.

Три независимых механизма убивают ствол, и модель считает все три.

1. ТЕПЛОВОЙ/ХИМИЧЕСКИЙ ИЗНОС НАЧАЛА НАРЕЗОВ (главный).
   За выстрел поверхность канала получает тепловой удар: за ~1 мс её
   температура подскакивает на 700-1100 К, после чего тепло уходит в толщу.
   Расчёт температуры поверхности — точное решение для полубесконечного
   тела с переменным тепловым потоком (интеграл Дюамеля):

       T_s(t) = T_0 + 1/sqrt(pi*k*rho*c) * INT_0^t q(tau)/sqrt(t-tau) dtau

   Скорость уноса металла подчиняется аррениусовской зависимости
   (Lawton B., "Thermal and chemical effects on gun barrel wear", 2001):

       w = A * INT exp(-E_a / (R * T_s(t))) dt

   Химия газов входит множителем: окислительные газы (CO2, H2O) выжигают
   железо, восстановительные (CO, H2) науглероживают поверхность и делают
   «белый слой», который потом выкрашивается. Поэтому кислородный баланс
   пороха — рычаг ресурса ствола, а не абстракция.

2. МАЛОЦИКЛОВАЯ УСТАЛОСТЬ КАЗЁННОЙ ЧАСТИ.
   Каждый выстрел — цикл нагружения с размахом деформации канала.
   Число циклов до трещины — по Мэнсону-Коффину.

3. МЕХАНИЧЕСКИЙ ИЗНОС ПОЛЕЙ НАРЕЗОВ ведущим пояском.

Результат — ресурс в выстрелах и деградация баллистики по мере настрела.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .barrel import Barrel
from .interior import InteriorResult
from .materials import Metal

# Калибровочные константы модели износа.
#
# EROSION_B — эмпирическая «температура активации» уноса металла по Лоутону.
# Это НЕ химическая энергия активации отдельной реакции: корреляция обобщает
# окисление, науглероживание, оплавление и смыв размягчённого слоя разом,
# поэтому наклон получается заметно положе чисто химического (B*R ~ 53 кДж/моль
# против 200+ у отдельной реакции). Проверка на разумность: между стволом с
# пиком 700 К и стволом с пиком 1400 К корреляция даёт разницу ресурса ~100
# раз, что и наблюдается в жизни (от 500 выстрелов у горячего магнума до
# 50 тыс. у холодного малоимпульсного).
#
# EROSION_A подобрана так, чтобы ствол под винтовочный патрон среднего
# давления (.308-класс, T1 ~ 2900 К, p ~ 450 МПа) давал ~0.02 мкм/выстрел,
# то есть браковочные 0.24 мм начала нарезов примерно за 12 тыс. выстрелов.
EROSION_A = 4.21e-2            # м/с
EROSION_B = 6400.0             # К
THROAT_LIMIT_FRACTION = 0.032  # доля калибра: браковочный износ начала нарезов
BORE_WEAR_RATIO = 0.18         # износ канала относительно износа начала нарезов


@dataclass
class ThermalResponse:
    """Тепловой отклик поверхности канала на один выстрел."""

    peak_surface_temp: float       # К
    time_at_peak: float            # с
    heat_input: float              # Дж/м^2 за выстрел
    bulk_temp_rise: float          # К, нагрев всей массы ствола
    surface_history: list[tuple[float, float]] = field(default_factory=list)


def surface_temperature(interior: InteriorResult, material: Metal,
                        initial_temp: float = 300.0,
                        tail_time: float = 0.02,
                        effusivity_override: float | None = None
                        ) -> ThermalResponse:
    """Температура поверхности канала у начала нарезов по интегралу Дюамеля.

    Используется история теплового потока q(t) из решения внутренней задачи.
    После вылета снаряда поток спадает — учитываем «хвост» экспонентой.
    """
    ts = interior.t
    qs = interior.heat_flux
    if len(ts) < 3:
        return ThermalResponse(initial_temp, 0.0, 0.0, 0.0, [])

    # добавляем период последействия: поток падает экспоненциально
    t_end = ts[-1]
    q_end = qs[-1]
    ext_t = list(ts)
    ext_q = list(qs)
    n_tail = 40
    for i in range(1, n_tail + 1):
        tt = t_end + tail_time * i / n_tail
        ext_t.append(tt)
        ext_q.append(q_end * math.exp(-(tt - t_end) / (0.25 * tail_time)))

    effusivity = (effusivity_override if effusivity_override
                  else material.thermal_effusivity)
    coef = 1.0 / (math.sqrt(math.pi) * effusivity)

    history: list[tuple[float, float]] = []
    peak = initial_temp
    t_peak = 0.0
    heat_input = 0.0
    for i in range(1, len(ext_t)):
        t_now = ext_t[i]
        acc = 0.0
        for j in range(1, i + 1):
            dt = ext_t[j] - ext_t[j - 1]
            if dt <= 0.0:
                continue
            q_avg = 0.5 * (ext_q[j] + ext_q[j - 1])
            tau = 0.5 * (ext_t[j] + ext_t[j - 1])
            denom = math.sqrt(max(t_now - tau, 1e-9))
            acc += q_avg * dt / denom
        temp = initial_temp + coef * acc
        history.append((t_now, temp))
        if temp > peak:
            peak = temp
            t_peak = t_now
    for i in range(1, len(ext_t)):
        heat_input += 0.5 * (ext_q[i] + ext_q[i - 1]) * (ext_t[i] - ext_t[i - 1])

    return ThermalResponse(peak_surface_temp=peak, time_at_peak=t_peak,
                           heat_input=heat_input, bulk_temp_rise=0.0,
                           surface_history=history)


@dataclass
class WearResult:
    """Износ за один выстрел и прогноз ресурса."""

    throat_wear_per_shot: float        # м на выстрел (радиальный унос)
    bore_wear_per_shot: float          # м на выстрел
    peak_surface_temp: float           # К
    barrel_temp_rise: float            # К за выстрел (нагрев всей массы)
    chemical_factor: float
    material_factor: float
    barrel_life: int                   # выстрелов до браковочного износа
    fatigue_life: int                  # циклов до трещины в казённой части
    limiting_factor: str
    notes: list[str] = field(default_factory=list)


def chemical_wear_factor(oxidizing_ratio: float, flame_temp: float) -> float:
    """Множитель износа от химии пороховых газов.

    Окислительные газы (высокое отношение CO2+H2O к CO+H2) прямо окисляют
    железо; сильно восстановительные науглероживают поверхность, образуя
    хрупкий «белый слой». Минимум износа — около слабо восстановительного
    состава, что и объясняет живучесть трёхосновных порохов.
    """
    ox = max(oxidizing_ratio, 0.01)
    oxidation = 0.55 + 0.9 * ox
    carburizing = 0.25 / (0.35 + ox)
    hot = 1.0 + 0.35 * max(0.0, (flame_temp - 2900.0) / 500.0)
    return (oxidation + carburizing) * hot


def effective_effusivity(barrel: Barrel, shot_time: float) -> float:
    """Тепловая активность поверхности sqrt(k*rho*c) с учётом покрытия.

    За время выстрела тепло проникает вглубь примерно на sqrt(a*t) — для
    стали это ~110 мкм за 1 мс. Гальванический хром в 20-30 мкм тоньше этой
    глубины и теплофизически почти «прозрачен»: температуру поверхности
    задаёт подложка. Вставка из стеллита в 1-2 мм, наоборот, работает как
    самостоятельное тело. Между этими крайностями отклик composite-слоя
    меняется плавно, поэтому берём взвешенную по глубине проникновения
    комбинацию, а не переключатель.
    """
    if barrel.lining is None or barrel.lining_thickness <= 0.0:
        return barrel.material.thermal_effusivity
    penetration = math.sqrt(barrel.material.thermal_diffusivity
                            * max(shot_time, 1e-5))
    w = min(barrel.lining_thickness / max(penetration, 1e-9), 1.0)
    return (w * barrel.lining.thermal_effusivity
            + (1.0 - w) * barrel.material.thermal_effusivity)


def analyse_wear(barrel: Barrel, interior: InteriorResult,
                 oxidizing_ratio: float, flame_temp: float,
                 initial_temp: float = 300.0,
                 shots_per_minute: float = 0.0) -> WearResult:
    """Полный расчёт износа за выстрел и ресурса ствола."""
    shot_time = max(interior.time_muzzle, 1e-4)
    effusivity = effective_effusivity(barrel, shot_time)
    surface_material = barrel.lining or barrel.material
    thermal = surface_temperature(interior, barrel.material, initial_temp,
                                  effusivity_override=effusivity)

    chem = chemical_wear_factor(oxidizing_ratio, flame_temp)
    # стойкость определяет тот материал, который реально лежит на поверхности
    mat = surface_material.erosion_factor

    # интеграл выдержки по истории температуры поверхности
    integral = 0.0
    hist = thermal.surface_history
    for i in range(1, len(hist)):
        t0, temp0 = hist[i - 1]
        t1, temp1 = hist[i]
        tm = 0.5 * (temp0 + temp1)
        if tm < 400.0:
            continue
        integral += math.exp(-EROSION_B / tm) * (t1 - t0)

    # механическая составляющая: скребущее действие газов и пояска
    mech = 1.0 + 0.55 * (interior.p_max_breech / 400e6) ** 1.5

    base_rate = EROSION_A * integral * chem * mech
    throat = base_rate * mat
    bore = throat * BORE_WEAR_RATIO

    # нагрев всей массы ствола за выстрел
    heat_capacity = barrel.heat_capacity()
    dt_bulk = (interior.heat_to_barrel / heat_capacity
               if heat_capacity > 0.0 else 0.0)

    limit = THROAT_LIMIT_FRACTION * barrel.bore_diameter

    # Ресурс в две фазы. Пока покрытие цело, унос идёт по его стойкости;
    # как только оно съедено насквозь — обнажается подложка и дальше ствол
    # горит с её скоростью. Именно поэтому хромирование даёт выигрыш в разы,
    # а не в десятки раз: слой в 25 мкм конечен.
    if throat <= 1e-15:
        life = 10 ** 6
    elif barrel.lining is not None and barrel.lining_thickness > 0.0:
        n_lining = barrel.lining_thickness / throat
        remaining = max(limit - barrel.lining_thickness, 0.0)
        base_wear = base_rate * barrel.material.erosion_factor
        n_base = remaining / base_wear if base_wear > 1e-15 else 10 ** 6
        life = int(min(n_lining + n_base, 1e6))
    else:
        life = int(limit / throat)

    fatigue = _fatigue_life(barrel, interior)

    notes: list[str] = []
    if thermal.peak_surface_temp > surface_material.melting_point * 0.85:
        notes.append(
            f"Пик температуры поверхности {thermal.peak_surface_temp:.0f} К — "
            "поверхностный слой оплавляется, износ пойдёт лавинообразно.")
    if chem > 1.6:
        notes.append("Химически агрессивные газы: сместите кислородный баланс "
                     "в слабо восстановительную область или добавьте охладитель.")
    if shots_per_minute > 0.0 and dt_bulk * shots_per_minute > 300.0:
        notes.append(
            f"При {shots_per_minute:.0f} выстр/мин ствол греется на "
            f"{dt_bulk * shots_per_minute:.0f} К/мин — нужен перерыв "
            "или сменный ствол.")

    limiting = "выгорание начала нарезов" if life <= fatigue else \
        "усталость казённой части"
    return WearResult(
        throat_wear_per_shot=throat, bore_wear_per_shot=bore,
        peak_surface_temp=thermal.peak_surface_temp, barrel_temp_rise=dt_bulk,
        chemical_factor=chem, material_factor=mat,
        barrel_life=min(life, fatigue), fatigue_life=fatigue,
        limiting_factor=limiting, notes=notes)


def _fatigue_life(barrel: Barrel, interior: InteriorResult) -> int:
    """Малоцикловая усталость казённой части (Мэнсон-Коффин).

        eps_a = sigma_f'/E * (2N)^b + eps_f' * (2N)^c

    Для конструкционных сталей берём типовые sigma_f' = 1.5*sigma_в,
    eps_f' = 0.5, b = -0.09, c = -0.6. Решаем относительно N численно.
    """
    st = barrel.stress_at(0.0, interior.p_max_breech)
    eps_a = 0.5 * st.hoop_strain
    if eps_a <= 1e-9:
        return 10 ** 6
    e = barrel.material.youngs_modulus
    sf = 1.5 * barrel.material.ultimate_strength
    ef = 0.5
    lo, hi = 1.0, 1e9
    for _ in range(200):
        n = math.sqrt(lo * hi)
        val = sf / e * (2.0 * n) ** -0.09 + ef * (2.0 * n) ** -0.6
        if val > eps_a:
            lo = n
        else:
            hi = n
    return int(min(math.sqrt(lo * hi), 1e6))


@dataclass
class BarrelCondition:
    """Накопленное состояние ствола — то, что живёт между выстрелами."""

    rounds_fired: int = 0
    throat_erosion: float = 0.0        # м, радиальный унос начала нарезов
    bore_enlargement: float = 0.0      # м, радиальный износ канала
    temperature: float = 293.15        # К
    fatigue_damage: float = 0.0        # 0..1 по правилу Майнера
    copper_fouling: float = 0.0        # 0..1, омеднение

    def apply_shot(self, wear: WearResult, cooling_time: float = 0.0,
                   ambient: float = 293.15,
                   cooling_constant: float = 0.02) -> None:
        """Один выстрел: добавить износ, нагреть, потом остыть."""
        self.rounds_fired += 1
        self.throat_erosion += wear.throat_wear_per_shot
        self.bore_enlargement += wear.bore_wear_per_shot
        self.temperature += wear.barrel_temp_rise
        if wear.fatigue_life > 0:
            self.fatigue_damage += 1.0 / wear.fatigue_life
        self.copper_fouling = min(1.0, self.copper_fouling + 0.0004)
        if cooling_time > 0.0:
            self.cool(cooling_time, ambient, cooling_constant)

    def cool(self, seconds: float, ambient: float = 293.15,
             cooling_constant: float = 0.02) -> None:
        """Ньютоновское остывание (конвекция + излучение упрощённо)."""
        self.temperature = ambient + (self.temperature - ambient) * math.exp(
            -cooling_constant * seconds)

    def health(self, barrel: Barrel) -> float:
        """0..1: сколько ресурса осталось."""
        limit = THROAT_LIMIT_FRACTION * barrel.bore_diameter
        by_wear = 1.0 - min(self.throat_erosion / limit, 1.0)
        by_fatigue = 1.0 - min(self.fatigue_damage, 1.0)
        return max(0.0, min(by_wear, by_fatigue))

    def velocity_loss(self, barrel: Barrel) -> float:
        """Относительная потеря дульной скорости от износа.

        Выгоревшее начало нарезов удлиняет свободный ход и увеличивает
        объём каморы: снаряд врезается позже, давление форсирования падает,
        часть газов прорывается вперёд.
        """
        limit = THROAT_LIMIT_FRACTION * barrel.bore_diameter
        frac = min(self.throat_erosion / limit, 2.0)
        return 0.035 * frac + 0.02 * frac * frac

    def dispersion_growth(self, barrel: Barrel) -> float:
        """Множитель к угловому рассеиванию от износа и омеднения."""
        limit = THROAT_LIMIT_FRACTION * barrel.bore_diameter
        frac = min(self.throat_erosion / limit, 2.0)
        return 1.0 + 0.9 * frac ** 1.6 + 0.25 * self.copper_fouling

    def status(self, barrel: Barrel) -> str:
        h = self.health(barrel)
        if h > 0.75:
            state = "как новый"
        elif h > 0.45:
            state = "рабочий, начало нарезов подсело"
        elif h > 0.15:
            state = "изношен, кучность ушла"
        elif h > 0.0:
            state = "на грани браковки"
        else:
            state = "выбраковка"
        return (f"{self.rounds_fired} выстр., износ начала нарезов "
                f"{self.throat_erosion * 1e6:.0f} мкм, ресурс {100 * h:.0f}% "
                f"({state}), T = {self.temperature - 273.15:.0f} C")
