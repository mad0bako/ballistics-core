"""Допуски, разброс и кучность: почему одинаковые патроны летят по-разному.

Кучность в этой модели не «характеристика оружия», а СУММА конкретных
физических вкладов, каждый из которых считается отдельно:

1. РАЗБРОС ДУЛЬНОЙ СКОРОСТИ -> вертикальный разброс на дальности.
   Источники: навеска, свод зерна (разные партии), объём гильзы,
   глубина посадки, температура заряда. Прогоняются методом Монте-Карло
   через настоящую внутреннюю баллистику, а не через «плюс-минус 10 м/с».

2. БОКОВОЙ СБРОС ОТ ДИСБАЛАНСА ПУЛИ (lateral throw-off).
   Если центр масс смещён с оси вращения на eps, то в момент вылета пуля
   получает поперечную скорость eps*omega, и угол сброса равен

       theta = eps * omega / V

   Это самый недооценённый вклад: смещение ЦМ на 10 мкм при 6000 рад/с
   и 800 м/с даёт 0.26 МОА — больше, чем весь остальной бюджет.

3. НАЧАЛЬНЫЙ УГОЛ НУТАЦИИ от несоосности пули и канала — через
   аэродинамический подскок (aerodynamic jump).

4. КОЛЕБАНИЯ СТВОЛА: момент вылета попадает в разную фазу изгибных
   колебаний, потому что время выстрела гуляет вместе со скоростью.

5. ИЗНОС: выгоревшее начало нарезов ухудшает всё сразу.

Вклады складываются квадратично (некоррелированные случайные величины).
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

from .cartridge import Cartridge, build_system
from .exterior import DragModel, ShotConditions, solve_trajectory
from .interior import Charge, InteriorOptions, solve_interior
from .rng import Rng
from .simulation import Environment, Weapon
from .units import moa_from_rad

MC_OPTIONS = InteriorOptions(rtol=3e-6, sample_stride=8)


@dataclass
class Tolerances:
    """Технологические допуски (сигмы, а не поля допуска)."""

    charge_mass_sd: float = 0.02e-3        # кг, разброс навески
    web_sd_relative: float = 0.008         # относительный разброс свода зерна
    burn_rate_sd_relative: float = 0.006   # разброс партии пороха
    bullet_mass_sd_relative: float = 0.002
    case_capacity_sd_relative: float = 0.006
    seating_depth_sd: float = 0.05e-3      # м
    powder_temp_sd: float = 1.0            # К
    shot_start_sd_relative: float = 0.05   # разброс давления форсирования
    # геометрия, влияющая на кучность напрямую
    cg_offset: float = 4e-6                # м, смещение ЦМ пули с оси
    bullet_runout: float = 25e-6           # м, биение пули в патроне
    crown_quality: float = 1.0             # 1.0 = идеальный дульный срез
    bedding_quality: float = 1.0           # 1.0 = идеальная вывеска ствола

    @staticmethod
    def factory_match() -> "Tolerances":
        return Tolerances(charge_mass_sd=0.013e-3, web_sd_relative=0.005,
                          burn_rate_sd_relative=0.004,
                          bullet_mass_sd_relative=0.0012,
                          case_capacity_sd_relative=0.004,
                          seating_depth_sd=0.025e-3, cg_offset=2.0e-6,
                          bullet_runout=12e-6)

    @staticmethod
    def military() -> "Tolerances":
        return Tolerances(charge_mass_sd=0.035e-3, web_sd_relative=0.012,
                          burn_rate_sd_relative=0.010,
                          bullet_mass_sd_relative=0.004,
                          case_capacity_sd_relative=0.010,
                          seating_depth_sd=0.10e-3, cg_offset=8e-6,
                          bullet_runout=50e-6)

    @staticmethod
    def garage() -> "Tolerances":
        return Tolerances(charge_mass_sd=0.09e-3, web_sd_relative=0.030,
                          burn_rate_sd_relative=0.025,
                          bullet_mass_sd_relative=0.012,
                          case_capacity_sd_relative=0.020,
                          seating_depth_sd=0.25e-3, cg_offset=25e-6,
                          bullet_runout=120e-6, crown_quality=0.6,
                          bedding_quality=0.6)


@dataclass
class DispersionResult:
    """Бюджет рассеивания и статистика скоростей."""

    velocity_mean: float = 0.0
    velocity_sd: float = 0.0
    velocity_es: float = 0.0
    pressure_mean: float = 0.0      # среднебаллистическое (как мерит датчик)
    pressure_sd: float = 0.0
    pressure_p99: float = 0.0
    breech_p99: float = 0.0         # казённое: им нагружается затвор
    samples: int = 0
    range_m: float = 0.0
    vertical_moa: float = 0.0
    horizontal_moa: float = 0.0
    group_moa: float = 0.0
    contributions: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"v0 = {self.velocity_mean:.1f} м/с, SD = {self.velocity_sd:.1f}, "
            f"ES = {self.velocity_es:.1f} ({self.samples} выстрелов)",
            f"p_ср: среднее {self.pressure_mean / 1e6:.0f} МПа, "
            f"SD = {self.pressure_sd / 1e6:.1f}, 99-й перцентиль "
            f"{self.pressure_p99 / 1e6:.0f} МПа (казённое "
            f"{self.breech_p99 / 1e6:.0f})",
            f"рассеивание на {self.range_m:.0f} м: {self.group_moa:.2f} МОА "
            f"(верт. {self.vertical_moa:.2f}, гор. {self.horizontal_moa:.2f})",
            "вклады, МОА:",
        ]
        for k, v in sorted(self.contributions.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:32s} {v:.3f}")
        lines.extend(f"! {w}" for w in self.warnings)
        return "\n".join(lines)


def monte_carlo(weapon: Weapon, cartridge: Cartridge, tol: Tolerances, *,
                shots: int = 60, environment: Environment | None = None,
                target_range: float = 300.0,
                seed: int = 12345) -> DispersionResult:
    """Прогон партии патронов через полную физику выстрела.

    seed — номер партии боеприпаса: одно и то же зерно даёт ту же партию
    с точностью до выстрела, на любой платформе.
    """
    env = environment or Environment()
    rng = Rng(seed if seed is not None else 0)
    res = DispersionResult(samples=shots, range_m=target_range)

    velocities: list[float] = []
    pressures: list[float] = []
    breech: list[float] = []
    base_capacity = cartridge.case.capacity

    for _ in range(shots):
        cart = deepcopy(cartridge)
        mass = max(cartridge.charge.mass
                   + rng.gauss(0.0, tol.charge_mass_sd), 1e-9)
        prop = deepcopy(cartridge.charge.propellant)
        prop.u1 *= max(1.0 + rng.gauss(0.0, tol.burn_rate_sd_relative), 0.1)
        web_factor = max(1.0 + rng.gauss(0.0, tol.web_sd_relative), 0.1)
        prop.grain.shape = _scale_shape(prop.grain.shape, web_factor)
        prop.grain.__post_init__()

        cart.case.capacity_override = base_capacity * max(
            1.0 + rng.gauss(0.0, tol.case_capacity_sd_relative), 0.5)
        cart.seating_depth = max(
            (cartridge.seating_depth or 0.0)
            + rng.gauss(0.0, tol.seating_depth_sd), 0.0)
        temp = env.powder_temperature + rng.gauss(0.0, tol.powder_temp_sd)
        cart.charge = Charge(prop, mass, temp)

        system = build_system(cart, weapon.barrel)
        system.shot_start_pressure *= max(
            1.0 + rng.gauss(0.0, tol.shot_start_sd_relative), 0.2)
        m_factor = max(1.0 + rng.gauss(0.0, tol.bullet_mass_sd_relative), 0.5)
        system.projectile_mass *= m_factor

        r = solve_interior(system, cart.charge, cart.primer, MC_OPTIONS)
        velocities.append(r.muzzle_velocity)
        # для сравнения с паспортом гильзы нужно СРЕДНЕЕ давление: именно его
        # показывает датчик в стенке, и именно к нему относятся нормы
        pressures.append(r.p_max_mean)
        breech.append(r.p_max_breech)

    velocities.sort()
    n = len(velocities)
    res.velocity_mean = sum(velocities) / n
    res.velocity_sd = math.sqrt(
        sum((v - res.velocity_mean) ** 2 for v in velocities) / max(n - 1, 1))
    res.velocity_es = velocities[-1] - velocities[0]

    pressures.sort()
    res.pressure_mean = sum(pressures) / n
    res.pressure_sd = math.sqrt(
        sum((p - res.pressure_mean) ** 2 for p in pressures) / max(n - 1, 1))
    res.pressure_p99 = pressures[min(n - 1, int(0.99 * n))]
    breech.sort()
    res.breech_p99 = breech[min(n - 1, int(0.99 * n))]

    _dispersion_budget(res, weapon, cartridge, tol, env, target_range)

    limit = cartridge.case.max_pressure
    if res.pressure_p99 > 1.03 * limit:
        over = 100.0 * (res.pressure_p99 / limit - 1.0)
        res.warnings.append(
            f"99-й перцентиль давления {res.pressure_p99 / 1e6:.0f} МПа — на "
            f"{over:.0f}% выше паспортных {limit / 1e6:.0f}: при таком разбросе "
            "часть партии выйдет за предел. Уменьшите навеску или ужесточите "
            "допуск на неё.")
    return res


def _scale_shape(shape, factor: float):
    """Масштабирует все линейные размеры зерна (имитация разброса свода)."""
    new = deepcopy(shape)
    for attr in ("diameter", "thickness", "length", "width",
                 "outer_diameter", "inner_diameter", "perf_diameter"):
        if hasattr(new, attr):
            setattr(new, attr, getattr(new, attr) * factor)
    return new


def _dispersion_budget(res: DispersionResult, weapon: Weapon,
                       cartridge: Cartridge, tol: Tolerances,
                       env: Environment, target_range: float) -> None:
    """Раскладывает угловое рассеивание по физическим источникам."""
    proj = cartridge.projectile
    barrel = weapon.barrel
    drag = DragModel("G7", proj.form_factor_g7)
    v0 = res.velocity_mean

    # 1. вертикаль от разброса скорости: численная производная dy/dv
    def drop_at(v: float) -> float:
        shot = ShotConditions(velocity=v, launch_angle=math.radians(0.5),
                              latitude=env.latitude)
        tr = solve_trajectory(proj.mass, proj.geometry.diameter, drag, shot,
                              env.atmosphere, env.wind,
                              max_range=target_range, dt=2e-3)
        p = tr.at_range(target_range)
        return p.y if p else 0.0

    dv = max(res.velocity_sd, 0.5)
    dydv = (drop_at(v0 + dv) - drop_at(v0 - dv)) / (2.0 * dv)
    sigma_vert = abs(dydv) * res.velocity_sd
    moa_velocity = moa_from_rad(sigma_vert / max(target_range, 1e-6))

    # 2. боковой сброс от дисбаланса пули: theta = eps * omega / V
    spin = (2.0 * math.pi * v0 / barrel.rifling.twist
            if barrel.rifling.twist > 0.0 else 0.0)
    moa_imbalance = moa_from_rad(tol.cg_offset * spin / max(v0, 1.0))

    # 3. аэродинамический подскок от начального угла нутации.
    # Несоосность пули в патроне даёт угол atan(runout / длина ведущей части),
    # а подскок составляет лишь малую долю этого угла: пуля не летит под
    # углом нутации, она вокруг него прецессирует, и в сторону уходит только
    # усреднённый по обороту снос.
    #
    # Коэффициент откалиброван по известной практике снаряжения: биение
    # 0.003 дюйма (76 мкм) стоит около 0.35 МОА. Прямая пропорция углу
    # нутации давала бы 3 МОА, то есть завышала бы вклад на порядок и
    # забивала собой весь остальной бюджет рассеивания.
    yaw0 = math.atan2(tol.bullet_runout,
                      max(proj.geometry.bearing_length, 1e-4))
    jump_gain = 0.014 * proj.geometry.length_calibers / 4.0
    moa_jump = moa_from_rad(yaw0 * jump_gain / max(tol.crown_quality, 0.1))

    # 4. колебания ствола: время вылета гуляет вместе со скоростью и
    # попадает в разную фазу изгибных колебаний дульного среза
    t_exit = barrel.travel / max(0.5 * v0, 1.0)
    sigma_t = t_exit * res.velocity_sd / max(v0, 1.0)
    omega_b = 2.0 * math.pi * barrel.first_mode_frequency
    amp = barrel.muzzle_droop / max(tol.bedding_quality, 0.1)
    moa_whip = moa_from_rad(amp * omega_b * sigma_t / max(barrel.length, 1e-3))

    # 5. износ ствола
    wear_factor = weapon.condition.dispersion_growth(barrel)

    vert = math.sqrt(moa_velocity ** 2 + moa_jump ** 2 + moa_whip ** 2)
    horiz = math.sqrt(moa_imbalance ** 2 + moa_jump ** 2 + moa_whip ** 2)
    res.vertical_moa = vert * wear_factor
    res.horizontal_moa = horiz * wear_factor
    res.group_moa = math.sqrt(res.vertical_moa ** 2
                              + res.horizontal_moa ** 2)
    res.contributions = {
        "разброс скорости (вертикаль)": moa_velocity * wear_factor,
        "дисбаланс пули (боковой сброс)": moa_imbalance * wear_factor,
        "несоосность/подскок": moa_jump * wear_factor,
        "колебания ствола": moa_whip * wear_factor,
        "множитель износа ствола": wear_factor,
    }
