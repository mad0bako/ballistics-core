"""Сквозной выстрел: патрон + ствол + условия -> полный отчёт.

Этот модуль ничего не считает сам — он сшивает пять решателей в одну
цепочку и переводит их результаты на язык, на котором разговаривает игра:

    термохимия -> внутренняя баллистика -> прочность ствола -> износ
                                       \\-> внешняя баллистика -> попадание

Здесь же живёт состояние оружия между выстрелами: нагрев, накопленный
износ, усталость. Именно оно превращает набор формул в игровой цикл.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .barrel import Barrel, StressStation, miller_stability
from .cartridge import Cartridge, build_system
from .erosion import BarrelCondition, WearResult, analyse_wear
from .exterior import (Atmosphere, DragModel, ShotConditions, TrajectoryResult,
                       Wind, gyroscopic_stability, maximum_range,
                       overturning_moment_slope, solve_trajectory)
from .interior import (InteriorOptions, InteriorResult, aftereffect,
                       solve_interior)
from .units import moa_from_rad


@dataclass
class Weapon:
    """Оружие: ствол плюс всё, что к нему прикручено."""

    name: str
    barrel: Barrel
    total_mass: float = 4.0                 # кг, для расчёта отдачи
    condition: BarrelCondition = field(default_factory=BarrelCondition)
    muzzle_device_efficiency: float = 0.0   # 0..0.6, дульный тормоз
    magazine_length: float | None = None
    sight_height: float = 40e-3


@dataclass
class Environment:
    """Условия стрельбы."""

    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    wind: Wind = field(default_factory=Wind)
    latitude: float = math.radians(50.0)
    azimuth: float = 0.0
    powder_temperature: float = 294.15
    ambient_temperature: float = 293.15


@dataclass
class ShotReport:
    """Всё, что известно про один выстрел."""

    interior: InteriorResult
    wear: WearResult
    stress: list[StressStation]
    trajectory: TrajectoryResult | None
    muzzle_velocity: float
    muzzle_energy: float
    peak_pressure_breech: float
    min_safety_yield: float
    min_safety_burst: float
    worst_station: StressStation | None
    stability: float
    stability_miller: float
    recoil_energy: float
    recoil_impulse: float
    barrel_life: int
    barrel_temp_rise: float
    cost_per_shot: float
    verdicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return self.min_safety_burst >= 1.5 and not any(
            v.startswith("ОПАСНО") for v in self.verdicts)

    def summary(self) -> str:
        lines = [
            f"v0 = {self.muzzle_velocity:.0f} м/с, дульная энергия "
            f"{self.muzzle_energy:.0f} Дж",
            f"p_кн_max = {self.peak_pressure_breech / 1e6:.0f} МПа, "
            f"запас прочности: текучесть {self.min_safety_yield:.2f}, "
            f"разрушение {self.min_safety_burst:.2f}",
            f"Sg = {self.stability:.2f} (Миллер {self.stability_miller:.2f}), "
            f"отдача {self.recoil_energy:.1f} Дж",
            f"ресурс ствола {self.barrel_life} выстр., нагрев "
            f"{self.barrel_temp_rise:.1f} К/выстр, цена выстрела "
            f"{self.cost_per_shot:.2f}",
        ]
        if self.trajectory is not None:
            tr = self.trajectory
            lines.append(
                f"дальность {tr.impact_range:.0f} м, ВП {tr.time_of_flight:.2f} с, "
                f"скорость у цели {tr.impact_velocity:.0f} м/с, энергия "
                f"{0.5 * tr.mass * tr.impact_velocity ** 2:.0f} Дж")
        lines.extend(self.verdicts)
        lines.extend(f"! {w}" for w in self.warnings)
        return "\n".join(lines)


def _drag_model(cartridge: Cartridge, standard: str = "G7") -> DragModel:
    proj = cartridge.projectile
    i = (proj.form_factor_g1 if standard.upper() == "G1"
         else proj.form_factor_g7)
    return DragModel(standard=standard, form_factor=i)


def fire(weapon: Weapon, cartridge: Cartridge,
         environment: Environment | None = None, *,
         target_range: float | None = None,
         launch_angle: float = 0.0,
         trajectory: bool = True,
         interior_options: InteriorOptions | None = None) -> ShotReport:
    """Один выстрел со всеми последствиями."""
    env = environment or Environment()
    barrel = weapon.barrel
    cond = weapon.condition

    # состояние ствола влияет на баллистику: выгоревшее начало нарезов
    # увеличивает свободный ход и снижает давление форсирования
    barrel.temperature = cond.temperature
    cartridge.charge.temperature = env.powder_temperature

    system = build_system(cartridge, barrel)
    if cond.throat_erosion > 0.0:
        system.chamber_volume += (0.25 * math.pi * barrel.groove_diameter ** 2
                                  * cond.throat_erosion * 12.0)
        system.shot_start_pressure *= max(
            1.0 - 2.5 * cond.throat_erosion / barrel.bore_diameter, 0.35)

    interior = solve_interior(system, cartridge.charge, cartridge.primer,
                              interior_options)

    # прочность
    envelope = interior.pressure_envelope(30)
    stations = barrel.analyse(envelope, interior.p_max_breech, cond.temperature)
    worst_y = min(stations, key=lambda s: s.safety_factor)
    worst_b = min(stations, key=lambda s: s.burst_safety_factor)

    # износ
    chem = cartridge.charge.propellant.chem
    wear = analyse_wear(barrel, interior, chem.oxidizing_ratio,
                        chem.flame_temp, initial_temp=cond.temperature)

    # отдача
    _, impulse = aftereffect(interior, system, cartridge.charge)
    impulse *= (1.0 - weapon.muzzle_device_efficiency)
    recoil_energy = impulse ** 2 / (2.0 * max(weapon.total_mass, 1e-3))

    # устойчивость
    proj = cartridge.projectile
    v0 = interior.muzzle_velocity * (1.0 - cond.velocity_loss(barrel))
    mach0 = v0 / env.atmosphere.sound_speed(0.0)
    cma = overturning_moment_slope(proj.geometry.length_calibers, mach0)
    sg = gyroscopic_stability(interior.spin_rate, v0, proj.geometry.diameter,
                              proj.axial_inertia, proj.transverse_inertia,
                              cma, env.atmosphere.density(0.0))
    sg_miller = miller_stability(proj.geometry.diameter,
                                 proj.geometry.total_length, proj.mass,
                                 barrel.rifling.twist, v0,
                                 1.0 / max(env.atmosphere.density_ratio, 1e-6))

    # внешняя баллистика
    traj = None
    if trajectory:
        shot = ShotConditions(velocity=v0, launch_angle=launch_angle,
                              azimuth=env.azimuth, latitude=env.latitude,
                              sight_height=weapon.sight_height,
                              spin_rate=interior.spin_rate, stability=sg)
        traj = solve_trajectory(
            proj.mass, proj.geometry.diameter, _drag_model(cartridge), shot,
            env.atmosphere, env.wind,
            max_range=target_range if target_range else 3000.0,
            ground_level=-1e9 if target_range else 0.0)

    verdicts: list[str] = []
    if worst_b.burst_safety_factor < 1.0:
        verdicts.append(
            f"ОПАСНО: разрыв ствола. Запас по разрушению "
            f"{worst_b.burst_safety_factor:.2f} в сечении "
            f"x = {worst_b.x * 1e3:.0f} мм (стенка "
            f"{worst_b.wall_thickness * 1e3:.1f} мм).")
    elif worst_b.burst_safety_factor < 1.5:
        verdicts.append(
            f"ОПАСНО: запас по разрушению всего "
            f"{worst_b.burst_safety_factor:.2f} — ниже принятого минимума 1.5.")
    if worst_y.safety_factor < 0.8:
        verdicts.append(
            f"Канал раздувается пластически (запас по текучести "
            f"{worst_y.safety_factor:.2f}): патронник поплывёт, гильзы "
            "начнёт закусывать.")
    # Паспортный предел — это максимальное СРЕДНЕЕ давление партии, и на
    # него всегда закладывают допуск: превышение на процент-другой лежит
    # внутри разброса, а вот на десять процентов — уже перезаряд.
    p_limit = cartridge.case.max_pressure
    if interior.p_max_mean > 1.12 * p_limit:
        verdicts.append(
            f"ОПАСНО: перезаряд. {interior.p_max_mean / 1e6:.0f} МПа против "
            f"паспортных {p_limit / 1e6:.0f} — гильзу раздует, возможен "
            "прорыв газов через капсюльное гнездо.")
    elif interior.p_max_mean > 1.03 * p_limit:
        verdicts.append(
            f"Давление {interior.p_max_mean / 1e6:.0f} МПа выше паспортных "
            f"{p_limit / 1e6:.0f}: за пределом допуска, гильзы будут туго "
            "экстрагироваться.")
    if sg < 1.2:
        verdicts.append(
            f"Sg = {sg:.2f}: пуля неустойчива, будет кувыркаться. "
            "Нужен более крутой шаг нарезов или более короткая пуля.")
    elif sg > 2.5 and proj.geometry.length_calibers >= 3.0:
        verdicts.append(
            f"Sg = {sg:.2f}: закрутка с большим запасом. Пуля полетит, но "
            "вырастут деривация и чувствительность к боковому ветру.")

    # Перекрут опасен не «большим Sg» — у короткой пистолетной пули Sg = 5...8
    # это норма. Опасно центробежное напряжение в оболочке: вращающееся
    # кольцо радиуса r с угловой скоростью w растянуто напряжением
    # sigma = rho * w^2 * r^2. Тонкая оболочка на крутых нарезах рвётся
    # прямо в воздухе, и именно это ограничивает шаг снизу.
    jacket = proj.driving_band
    hoop = (jacket.density * interior.spin_rate ** 2
            * (0.5 * proj.geometry.diameter) ** 2)
    if hoop > jacket.ultimate_strength:
        verdicts.append(
            f"ОПАСНО: центробежное напряжение в оболочке "
            f"{hoop / 1e6:.0f} МПа против предела прочности "
            f"{jacket.ultimate_strength / 1e6:.0f} МПа — пулю разорвёт "
            "в полёте. Нужен более пологий шаг нарезов.")
    elif hoop > 0.55 * jacket.ultimate_strength:
        verdicts.append(
            f"Центробежное напряжение в оболочке {hoop / 1e6:.0f} МПа — "
            f"{100 * hoop / jacket.ultimate_strength:.0f}% от предела "
            "прочности. Тонкая оболочка на такой закрутке живёт недолго.")

    check = cartridge.check(magazine_length=weapon.magazine_length)
    warnings = list(interior.warnings) + list(wear.notes) + list(check.messages)
    if traj is not None:
        warnings.extend(traj.warnings)

    return ShotReport(
        interior=interior, wear=wear, stress=stations, trajectory=traj,
        muzzle_velocity=v0,
        muzzle_energy=0.5 * proj.mass * v0 ** 2,
        peak_pressure_breech=interior.p_max_breech,
        min_safety_yield=worst_y.safety_factor,
        min_safety_burst=worst_b.burst_safety_factor,
        worst_station=worst_b, stability=sg, stability_miller=sg_miller,
        recoil_energy=recoil_energy, recoil_impulse=impulse,
        barrel_life=wear.barrel_life, barrel_temp_rise=wear.barrel_temp_rise,
        cost_per_shot=cartridge.cost, verdicts=verdicts, warnings=warnings)


@dataclass
class SeriesResult:
    """Итог серии выстрелов: как оружие деградирует по ходу стрельбы."""

    shots: int
    velocities: list[float] = field(default_factory=list)
    pressures: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    throat_erosion: list[float] = field(default_factory=list)
    final_condition: BarrelCondition | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def velocity_drop(self) -> float:
        if len(self.velocities) < 2:
            return 0.0
        return self.velocities[0] - self.velocities[-1]


def fire_series(weapon: Weapon, cartridge: Cartridge, shots: int,
                environment: Environment | None = None, *,
                rate_per_minute: float = 60.0,
                sample_every: int = 1) -> SeriesResult:
    """Серия выстрелов с накоплением нагрева и износа.

    Считать внутреннюю баллистику на каждом выстреле дорого, поэтому
    пересчёт делается на выборочных выстрелах, а между ними износ и нагрев
    экстраполируются линейно — этого достаточно, чтобы поймать и просадку
    скорости от выгорания, и разогрев в очереди.
    """
    env = environment or Environment()
    cond = weapon.condition
    res = SeriesResult(shots=shots, final_condition=cond)
    interval = 60.0 / max(rate_per_minute, 1e-6)

    last: ShotReport | None = None
    for i in range(shots):
        if i % sample_every == 0 or last is None:
            last = fire(weapon, cartridge, env, trajectory=False)
            res.velocities.append(last.muzzle_velocity)
            res.pressures.append(last.peak_pressure_breech)
            res.temperatures.append(cond.temperature)
            res.throat_erosion.append(cond.throat_erosion)
        cond.apply_shot(last.wear, cooling_time=interval,
                        ambient=env.ambient_temperature)
        if cond.temperature > 900.0 and "перегрев" not in " ".join(res.notes):
            res.notes.append(
                f"Ствол перегрет ({cond.temperature - 273.15:.0f} C) на "
                f"{i + 1}-м выстреле: прочность просела, износ ускорился, "
                "возможен самопроизвольный выстрел от нагрева патрона.")
        if cond.health(weapon.barrel) <= 0.0:
            res.notes.append(f"Ствол выбракован на {i + 1}-м выстреле.")
            res.shots = i + 1
            break
    return res


def effective_range(weapon: Weapon, cartridge: Cartridge,
                    environment: Environment | None = None, *,
                    min_energy: float = 0.0,
                    min_velocity: float = 0.0,
                    require_supersonic: bool = False) -> float:
    """Дальность, на которой снаряд ещё удовлетворяет критерию поражения.

    Это и есть «долетает до цели» в строгом смысле: не «куда упадёт»,
    а «где ещё сохраняет заданную энергию/скорость/сверхзвук».
    """
    env = environment or Environment()
    report = fire(weapon, cartridge, env, trajectory=False)
    proj = cartridge.projectile
    shot = ShotConditions(velocity=report.muzzle_velocity,
                          launch_angle=math.radians(2.0),
                          azimuth=env.azimuth, latitude=env.latitude,
                          spin_rate=report.interior.spin_rate,
                          stability=report.stability)
    traj = solve_trajectory(proj.mass, proj.geometry.diameter,
                            _drag_model(cartridge), shot, env.atmosphere,
                            env.wind, max_range=1e9, ground_level=-2000.0,
                            dt=2e-3)
    best = 0.0
    for p in traj.points:
        v = p.speed
        e = 0.5 * proj.mass * v * v
        mach = v / env.atmosphere.sound_speed(p.y)
        if e < min_energy or v < min_velocity:
            break
        if require_supersonic and mach < 1.0:
            break
        best = p.x
    return best
