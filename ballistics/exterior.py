"""Внешняя баллистика: атмосфера, закон сопротивления, траектория.

Модель — точечная (3-DOF) с полным набором значимых сил:

    m * dV/dt = -0.5 * rho * |V_возд| * V_возд * S * Cd(M)  +  m*g
                - 2*m*(Omega x V)            (Кориолис)

плюс деривация (боковой уход от гироскопической прецессии), учитываемая
эмпирикой Литца, — строго она требует 6-DOF, а для игрового ядра важен
факт и порядок величины.

Закон сопротивления — стандартные функции G1/G7 (BRL), пересчитанные
форм-фактором конкретной пули:

    Cd_пули(M) = i * Cd_эталона(M)

G7 (эталон — 10-калиберная секущая оживальная головная часть с запоясковым
конусом) описывает современные остроконечные пули заметно точнее, чем G1
(эталон 1881 года, тупой, с плоским дном). Для тупоконечных и пистолетных
пуль G1 ближе.

Атмосфера — МСА с поправкой на влажность (влажный воздух ЛЕГЧЕ сухого,
потому что молярная масса водяного пара 18 против 29 у воздуха).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .units import (GAMMA_AIR, G0, OMEGA_EARTH, P_STD, R_AIR, R_VAPOR,
                    T_STD)

# --- стандартные функции сопротивления --------------------------------------
# Cd относительно миделя эталонного снаряда, по числу Маха.

G7_TABLE: tuple[tuple[float, float], ...] = (
    (0.00, 0.1198), (0.05, 0.1197), (0.30, 0.1194), (0.60, 0.1204),
    (0.70, 0.1223), (0.80, 0.1275), (0.85, 0.1339), (0.875, 0.1387),
    (0.90, 0.1454), (0.925, 0.1560), (0.95, 0.1702), (0.975, 0.1881),
    (1.00, 0.2079), (1.025, 0.2278), (1.05, 0.2413), (1.075, 0.2497),
    (1.10, 0.2547), (1.15, 0.2580), (1.20, 0.2573), (1.30, 0.2519),
    (1.40, 0.2450), (1.50, 0.2377), (1.60, 0.2306), (1.80, 0.2174),
    (2.00, 0.2050), (2.20, 0.1935), (2.50, 0.1787), (3.00, 0.1583),
    (3.50, 0.1443), (4.00, 0.1338), (5.00, 0.1183),
)

G1_TABLE: tuple[tuple[float, float], ...] = (
    (0.00, 0.2629), (0.05, 0.2558), (0.10, 0.2487), (0.20, 0.2413),
    (0.30, 0.2344), (0.40, 0.2278), (0.50, 0.2214), (0.60, 0.2155),
    (0.70, 0.2104), (0.725, 0.2278), (0.75, 0.2481), (0.775, 0.2670),
    (0.80, 0.2874), (0.825, 0.3084), (0.85, 0.3323), (0.875, 0.3583),
    (0.90, 0.3839), (0.925, 0.4055), (0.95, 0.4270), (0.975, 0.4477),
    (1.00, 0.4784), (1.025, 0.5285), (1.05, 0.5476), (1.075, 0.5559),
    (1.10, 0.5570), (1.125, 0.5533), (1.15, 0.5488), (1.20, 0.5376),
    (1.25, 0.5273), (1.30, 0.5178), (1.40, 0.5008), (1.50, 0.4869),
    (1.60, 0.4758), (1.80, 0.4580), (2.00, 0.4415), (2.20, 0.4260),
    (2.50, 0.4050), (3.00, 0.3765), (3.50, 0.3551), (4.00, 0.3386),
    (5.00, 0.3128),
)


def _interp(table: tuple[tuple[float, float], ...], mach: float) -> float:
    if mach <= table[0][0]:
        return table[0][1]
    if mach >= table[-1][0]:
        return table[-1][1]
    lo, hi = 0, len(table) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if table[mid][0] <= mach:
            lo = mid
        else:
            hi = mid
    m0, c0 = table[lo]
    m1, c1 = table[hi]
    return c0 + (c1 - c0) * (mach - m0) / (m1 - m0)


@dataclass
class DragModel:
    """Закон сопротивления пули."""

    standard: str = "G7"                 # G7 | G1 | custom
    form_factor: float = 1.0
    custom_table: tuple[tuple[float, float], ...] | None = None

    def cd(self, mach: float) -> float:
        if self.custom_table is not None:
            return _interp(self.custom_table, mach)
        table = G1_TABLE if self.standard.upper() == "G1" else G7_TABLE
        return self.form_factor * _interp(table, mach)


# --- атмосфера ---------------------------------------------------------------

@dataclass
class Atmosphere:
    """Стандартная атмосфера с поправками на реальные условия."""

    sea_level_pressure: float = P_STD        # Па (приведённое к уровню моря)
    sea_level_temperature: float = T_STD     # К
    lapse_rate: float = 0.0065               # К/м
    humidity: float = 0.0                    # 0..1 относительная влажность
    altitude: float = 0.0                    # м, высота огневой позиции

    def temperature(self, h: float = 0.0) -> float:
        return self.sea_level_temperature - self.lapse_rate * (self.altitude + h)

    def pressure(self, h: float = 0.0) -> float:
        t = self.temperature(h)
        exponent = G0 / (self.lapse_rate * R_AIR)
        ratio = t / self.sea_level_temperature
        return self.sea_level_pressure * max(ratio, 1e-6) ** exponent

    @staticmethod
    def saturation_pressure(t_k: float) -> float:
        """Давление насыщенного пара по Магнусу, Па."""
        t_c = t_k - 273.15
        return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))

    def density(self, h: float = 0.0) -> float:
        t = self.temperature(h)
        p = self.pressure(h)
        pv = self.humidity * self.saturation_pressure(t)
        pv = min(pv, 0.95 * p)
        pd = p - pv
        return pd / (R_AIR * t) + pv / (R_VAPOR * t)

    def sound_speed(self, h: float = 0.0) -> float:
        return math.sqrt(GAMMA_AIR * R_AIR * self.temperature(h))

    @property
    def density_ratio(self) -> float:
        """rho / rho_МСА — входит в оценку устойчивости."""
        std = P_STD / (R_AIR * T_STD)
        return self.density(0.0) / std


@dataclass
class Wind:
    """Ветер. speed — модуль, direction — откуда дует, рад от направления
    стрельбы по часовой стрелке (0 = встречный, pi/2 = справа)."""

    speed: float = 0.0
    direction: float = 0.0
    vertical: float = 0.0

    def vector(self) -> tuple[float, float, float]:
        # ветер «откуда» -> вектор скорости воздуха «куда»
        vx = -self.speed * math.cos(self.direction)
        vz = -self.speed * math.sin(self.direction)
        return vx, self.vertical, vz


# --- траектория --------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    t: float
    x: float          # дальность, м
    y: float          # превышение над горизонтом оружия, м
    z: float          # боковое смещение (вправо +), м
    vx: float
    vy: float
    vz: float

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)


@dataclass
class TrajectoryResult:
    points: list[TrajectoryPoint] = field(default_factory=list)
    mass: float = 0.0
    reached_target: bool = False
    impact_range: float = 0.0
    impact_velocity: float = 0.0
    impact_angle: float = 0.0        # рад, ниже горизонта
    time_of_flight: float = 0.0
    max_ordinate: float = 0.0
    max_ordinate_range: float = 0.0
    transonic_range: float | None = None
    subsonic_range: float | None = None
    warnings: list[str] = field(default_factory=list)

    def at_range(self, x: float) -> TrajectoryPoint | None:
        pts = self.points
        if not pts or x < pts[0].x or x > pts[-1].x:
            return None
        for i in range(1, len(pts)):
            if pts[i].x >= x:
                p0, p1 = pts[i - 1], pts[i]
                f = (x - p0.x) / (p1.x - p0.x) if p1.x > p0.x else 0.0
                return TrajectoryPoint(
                    t=p0.t + f * (p1.t - p0.t), x=x,
                    y=p0.y + f * (p1.y - p0.y), z=p0.z + f * (p1.z - p0.z),
                    vx=p0.vx + f * (p1.vx - p0.vx),
                    vy=p0.vy + f * (p1.vy - p0.vy),
                    vz=p0.vz + f * (p1.vz - p0.vz))
        return pts[-1]

    def energy_at(self, x: float) -> float:
        p = self.at_range(x)
        return 0.5 * self.mass * p.speed ** 2 if p else 0.0

    def drop_at(self, x: float) -> float:
        p = self.at_range(x)
        return p.y if p else float("nan")


@dataclass
class ShotConditions:
    velocity: float                    # м/с, дульная
    launch_angle: float = 0.0          # рад, угол бросания
    azimuth: float = 0.0               # рад, от севера по часовой (для Кориолиса)
    latitude: float = math.radians(50.0)
    sight_height: float = 0.0          # м, высота линии прицеливания над осью
    spin_rate: float = 0.0             # рад/с на дульном срезе
    stability: float = 0.0             # Sg (для деривации)
    coriolis: bool = True
    spin_drift: bool = True


def solve_trajectory(mass: float, diameter: float, drag: DragModel,
                     shot: ShotConditions,
                     atmosphere: Atmosphere | None = None,
                     wind: Wind | None = None,
                     max_range: float = 3000.0,
                     ground_level: float = -1e9,
                     dt: float = 1e-3,
                     max_time: float = 200.0) -> TrajectoryResult:
    """Интегрирует траекторию методом Рунге-Кутты 4-го порядка."""
    atm = atmosphere or Atmosphere()
    wnd = wind or Wind()
    wx, wy, wz = wnd.vector()
    area = 0.25 * math.pi * diameter * diameter

    lat = shot.latitude
    az = shot.azimuth
    om_x = OMEGA_EARTH * math.cos(lat) * math.cos(az)
    om_y = OMEGA_EARTH * math.sin(lat)
    om_z = -OMEGA_EARTH * math.cos(lat) * math.sin(az)

    res = TrajectoryResult(mass=mass)

    def deriv(state: list[float]) -> list[float]:
        x, y, z, vx, vy, vz = state
        avx, avy, avz = vx - wx, vy - wy, vz - wz
        speed = math.sqrt(avx * avx + avy * avy + avz * avz)
        if speed < 1e-6:
            drag_acc = (0.0, 0.0, 0.0)
        else:
            rho = atm.density(y)
            a_snd = atm.sound_speed(y)
            cd = drag.cd(speed / a_snd)
            k = 0.5 * rho * speed * area * cd / mass
            drag_acc = (-k * avx, -k * avy, -k * avz)
        ax, ay, az_ = drag_acc
        ay -= G0
        if shot.coriolis:
            ax += -2.0 * (om_y * vz - om_z * vy)
            ay += -2.0 * (om_z * vx - om_x * vz)
            az_ += -2.0 * (om_x * vy - om_y * vx)
        return [vx, vy, vz, ax, ay, az_]

    v0 = shot.velocity
    state = [0.0, 0.0, 0.0,
             v0 * math.cos(shot.launch_angle),
             v0 * math.sin(shot.launch_angle),
             0.0]
    t = 0.0
    res.points.append(TrajectoryPoint(t, *state[:3], *state[3:]))
    steps = 0
    max_steps = int(max_time / dt) + 10
    prev = list(state)

    while steps < max_steps:
        steps += 1
        k1 = deriv(state)
        s2 = [state[i] + 0.5 * dt * k1[i] for i in range(6)]
        k2 = deriv(s2)
        s3 = [state[i] + 0.5 * dt * k2[i] for i in range(6)]
        k3 = deriv(s3)
        s4 = [state[i] + dt * k3[i] for i in range(6)]
        k4 = deriv(s4)
        prev = list(state)
        state = [state[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                 for i in range(6)]
        t += dt

        if state[1] < ground_level and prev[1] >= ground_level:
            f = (prev[1] - ground_level) / max(prev[1] - state[1], 1e-12)
            state = [prev[i] + f * (state[i] - prev[i]) for i in range(6)]
            t = t - dt + f * dt
            res.points.append(TrajectoryPoint(t, *state[:3], *state[3:]))
            break

        res.points.append(TrajectoryPoint(t, *state[:3], *state[3:]))

        if state[0] >= max_range:
            break
        if state[3] <= 0.0:
            res.warnings.append("Снаряд потерял горизонтальную скорость.")
            break

    # деривация: боковой уход от гироскопической прецессии (эмпирика Литца)
    if shot.spin_drift and shot.stability > 0.0:
        sign = 1.0  # правые нарезы уводят вправо
        for p in res.points:
            p.z += sign * 0.0254 * 1.25 * (shot.stability + 1.2) * p.t ** 1.83

    pts = res.points
    last = pts[-1]
    res.time_of_flight = last.t
    res.impact_range = last.x
    res.impact_velocity = last.speed
    res.impact_angle = math.atan2(-last.vy, last.vx)
    res.reached_target = last.x >= max_range - 1e-6

    top = max(pts, key=lambda p: p.y)
    res.max_ordinate = top.y
    res.max_ordinate_range = top.x

    for p in pts:
        mach = p.speed / atm.sound_speed(p.y)
        if res.transonic_range is None and mach < 1.2:
            res.transonic_range = p.x
        if res.subsonic_range is None and mach < 1.0:
            res.subsonic_range = p.x
            break
    return res


# --- обратные задачи ---------------------------------------------------------

def angle_for_target(mass: float, diameter: float, drag: DragModel,
                     shot: ShotConditions, target_range: float,
                     target_height: float = 0.0,
                     atmosphere: Atmosphere | None = None,
                     wind: Wind | None = None,
                     high_angle: bool = False,
                     tol: float = 1e-3) -> float | None:
    """Подбирает угол бросания, чтобы попасть в точку (дальность, высота).

    Возвращает угол в радианах или None, если цель недостижима.
    Для навесной траектории (high_angle=True) ищется вторая, крутая ветвь.
    """
    def miss(angle: float) -> float:
        s = ShotConditions(**{**shot.__dict__, "launch_angle": angle})
        r = solve_trajectory(mass, diameter, drag, s, atmosphere, wind,
                             max_range=target_range, dt=1e-3)
        p = r.at_range(target_range)
        if p is None:
            return -1e9
        return p.y - target_height

    lo, hi = math.radians(-5.0), math.radians(89.0)
    peak = max_range_angle(mass, diameter, drag, shot, atmosphere, wind)
    if high_angle:
        lo, hi = peak, math.radians(89.0)
    else:
        hi = peak
    f_lo, f_hi = miss(lo), miss(hi)
    if f_lo * f_hi > 0.0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = miss(mid)
        if abs(fm) < tol:
            return mid
        if f_lo * fm <= 0.0:
            hi, f_hi = mid, fm
        else:
            lo, f_lo = mid, fm
    return 0.5 * (lo + hi)


def max_range_angle(mass: float, diameter: float, drag: DragModel,
                    shot: ShotConditions,
                    atmosphere: Atmosphere | None = None,
                    wind: Wind | None = None) -> float:
    """Угол наибольшей дальности (золотое сечение).

    Для реальных снарядов он заметно меньше 45 градусов — сопротивление
    воздуха «съедает» пологие участки навесной ветви.
    """
    def reach(angle: float) -> float:
        s = ShotConditions(**{**shot.__dict__, "launch_angle": angle})
        r = solve_trajectory(mass, diameter, drag, s, atmosphere, wind,
                             max_range=1e9, ground_level=0.0, dt=2e-3)
        return r.impact_range

    lo, hi = math.radians(5.0), math.radians(75.0)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = reach(a), reach(b)
    for _ in range(30):
        if fa < fb:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = reach(b)
        else:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = reach(a)
        if hi - lo < 1e-4:
            break
    return 0.5 * (lo + hi)


def maximum_range(mass: float, diameter: float, drag: DragModel,
                  shot: ShotConditions,
                  atmosphere: Atmosphere | None = None,
                  wind: Wind | None = None) -> tuple[float, float]:
    """(максимальная дальность, оптимальный угол)."""
    angle = max_range_angle(mass, diameter, drag, shot, atmosphere, wind)
    s = ShotConditions(**{**shot.__dict__, "launch_angle": angle})
    r = solve_trajectory(mass, diameter, drag, s, atmosphere, wind,
                         max_range=1e9, ground_level=0.0, dt=2e-3)
    return r.impact_range, angle


def zero_angle(mass: float, diameter: float, drag: DragModel,
               shot: ShotConditions, zero_range: float,
               atmosphere: Atmosphere | None = None) -> float | None:
    """Угол возвышения для пристрелки на заданную дальность
    с учётом высоты линии прицеливания над осью канала."""
    return angle_for_target(mass, diameter, drag, shot, zero_range,
                            target_height=shot.sight_height,
                            atmosphere=atmosphere)


# --- устойчивость ------------------------------------------------------------

def gyroscopic_stability(spin_rate: float, velocity: float, diameter: float,
                         axial_inertia: float, transverse_inertia: float,
                         cm_alpha: float, air_density: float) -> float:
    """Коэффициент гироскопической устойчивости (аэродинамическая форма).

        Sg = Ix^2 * p^2 / (2 * rho * S * d * Iy * V^2 * Cm_alpha)

    Sg > 1 — пуля устойчива; на практике целятся в 1.4...2.0: при Sg чуть
    больше единицы пуля «летит боком» на нисходящей ветви, при Sg > 2.5
    избыточная закрутка усиливает деривацию и чувствительность к ветру.
    """
    area = 0.25 * math.pi * diameter * diameter
    den = (2.0 * air_density * area * diameter * transverse_inertia
           * velocity * velocity * cm_alpha)
    if den <= 0.0:
        return 0.0
    return (axial_inertia ** 2) * (spin_rate ** 2) / den


def overturning_moment_slope(length_calibers: float, mach: float) -> float:
    """Оценка Cm_alpha (производной опрокидывающего момента).

    Для остроконечных пуль Cm_alpha растёт с удлинением и слабо зависит
    от числа Маха в сверхзвуке (McCoy, гл. 2): в первом приближении
    Cm_alpha ~ 2 + 0.9*(L/d - 3), с ростом на трансзвуке.
    """
    base = 2.0 + 0.9 * max(length_calibers - 3.0, 0.0)
    if mach < 1.0:
        return base * 1.25
    if mach < 1.4:
        return base * (1.25 - 0.25 * (mach - 1.0) / 0.4)
    return base
