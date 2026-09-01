"""Обратные задачи: подбор навески, ствола, нарезов, рецептуры.

Прямая задача («что будет, если...») решается модулями физики. Здесь —
обратная («сколько надо, чтобы...»), и это ровно то, чем занят оружейник:

    подобрать навеску под заданное давление или скорость
    подобрать длину ствола под заданную скорость
    подобрать толщину стенки под заданный запас прочности
    подобрать шаг нарезов под заданную устойчивость
    подобрать состав пороха под заданный ресурс ствола

Все солверы монотонны по своему параметру, поэтому используется дихотомия:
она надёжнее Ньютона на разрывных функциях (а функции здесь разрывны —
скажем, при переходе через «заряд не влезает в гильзу»).
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

from .barrel import (Barrel, gyroscopic_twist_miller, required_outer_diameter)
from .cartridge import Cartridge, build_system
from .erosion import analyse_wear
from .exterior import Atmosphere, DragModel, ShotConditions, solve_trajectory
from .interior import Charge, InteriorOptions, solve_interior
from .materials import Metal
from .propellant import Propellant
from .simulation import Environment, ShotReport, Weapon, fire
from .thermochem import INGREDIENTS, compute_thermochemistry

FAST_OPTIONS = InteriorOptions(rtol=3e-6, sample_stride=4)


def _bisect(func, lo: float, hi: float, target: float,
            tol: float = 1e-4, iters: int = 60) -> float | None:
    """Дихотомия для монотонно возрастающей func."""
    f_lo = func(lo) - target
    f_hi = func(hi) - target
    if f_lo > 0.0:
        return None
    if f_hi < 0.0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = func(mid) - target
        if abs(fm) <= tol * max(abs(target), 1.0):
            return mid
        if fm < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- навеска -----------------------------------------------------------------

@dataclass
class ChargeSolution:
    charge_mass: float
    muzzle_velocity: float
    peak_pressure: float
    burnt_fraction: float
    fill_ratio: float
    feasible: bool
    reason: str = ""


def _evaluate_charge(cartridge: Cartridge, barrel: Barrel, mass: float,
                     temperature: float | None = None, system=None):
    """Считает выстрел с заданной навеской.

    Постановка задачи (объём каморы, площадь канала, путь, давление
    форсирования) от массы заряда НЕ зависит, поэтому system строится один
    раз и переиспользуется. Глубокое копирование патрона на каждой итерации
    дихотомии тянуло за собой всю термохимию пороха и съедало на порядок
    больше времени, чем сам расчёт выстрела.
    """
    charge = Charge(cartridge.charge.propellant, mass,
                    temperature if temperature is not None
                    else cartridge.charge.temperature)
    if system is None:
        system = build_system(cartridge, barrel)
    res = solve_interior(system, charge, cartridge.primer, FAST_OPTIONS)
    return charge, res


def _max_loadable_charge(cartridge: Cartridge) -> float:
    """Сколько пороха физически влезает в камору.

    Верхняя граница поиска должна быть насыпной, а не истинной плотностью:
    заряд плотнее насыпного в гильзу не засыпать, а считать выстрел на
    заведомо невозможной навеске — только вязнуть в интеграторе на
    гигапаскалях, которых в природе не бывает.
    """
    bulk = cartridge.charge.propellant.grain.bulk_density
    return 1.15 * bulk * cartridge.chamber_volume()


def _fill_ratio(cartridge: Cartridge, mass: float) -> float:
    """Заполнение каморы навеской по насыпной плотности, без копирования."""
    bulk = cartridge.charge.propellant.grain.bulk_density
    return (mass / bulk) / cartridge.chamber_volume()


def charge_for_pressure(cartridge: Cartridge, barrel: Barrel,
                        pressure_limit: float, *,
                        metric: str = "mean",
                        lo: float = 1e-5, hi: float | None = None
                        ) -> ChargeSolution:
    """Навеска, при которой давление равно заданному пределу.

    metric='mean' — по среднебаллистическому давлению: именно его показывает
    крешерный или пьезодатчик в стенке гильзы, и именно к нему относятся
    паспортные пределы SAAMI/CIP. metric='breech' — по казённому, которое
    выше на несколько процентов и нагружает затвор.
    """
    if hi is None:
        hi = _max_loadable_charge(cartridge)
    system = build_system(cartridge, barrel)
    cache: dict[float, tuple] = {}

    def pressure(m: float) -> float:
        if m not in cache:
            cache[m] = _evaluate_charge(cartridge, barrel, m, system=system)
        res = cache[m][1]
        return res.p_max_mean if metric == "mean" else res.p_max_breech

    m = _bisect(pressure, lo, hi, pressure_limit, tol=1e-3)
    if m is None:
        return ChargeSolution(0.0, 0.0, 0.0, 0.0, 0.0, False,
                              "Предел давления недостижим в этом диапазоне "
                              "навесок: смените порох или ёмкость гильзы.")
    _, res = cache.get(m) or _evaluate_charge(cartridge, barrel, m,
                                              system=system)
    fill = _fill_ratio(cartridge, m)
    return ChargeSolution(m, res.muzzle_velocity, res.p_max_breech,
                          res.psi_muzzle, fill, fill <= 1.0,
                          "" if fill <= 1.0
                          else "Навеска не помещается в гильзу.")


def charge_for_velocity(cartridge: Cartridge, barrel: Barrel,
                        target_velocity: float, *,
                        pressure_limit: float | None = None,
                        lo: float = 1e-5, hi: float | None = None
                        ) -> ChargeSolution:
    """Навеска под заданную дульную скорость, с проверкой по давлению."""
    if hi is None:
        hi = _max_loadable_charge(cartridge)
    system = build_system(cartridge, barrel)
    cache: dict[float, tuple] = {}

    def velocity(m: float) -> float:
        if m not in cache:
            cache[m] = _evaluate_charge(cartridge, barrel, m, system=system)
        return cache[m][1].muzzle_velocity

    m = _bisect(velocity, lo, hi, target_velocity, tol=1e-4)
    if m is None:
        return ChargeSolution(0.0, 0.0, 0.0, 0.0, 0.0, False,
                              f"Скорость {target_velocity:.0f} м/с недостижима: "
                              "не хватает ёмкости гильзы или энергии пороха.")
    _, res = cache.get(m) or _evaluate_charge(cartridge, barrel, m,
                                              system=system)
    fill = _fill_ratio(cartridge, m)
    ok = fill <= 1.0
    reason = "" if ok else "Навеска не помещается в гильзу."
    if pressure_limit is not None and res.p_max_breech > pressure_limit:
        ok = False
        reason = (f"Скорость достигается только при "
                  f"{res.p_max_breech / 1e6:.0f} МПа против предела "
                  f"{pressure_limit / 1e6:.0f} МПа. Нужен более медленный "
                  "порох, более длинный ствол или более лёгкая пуля.")
    return ChargeSolution(m, res.muzzle_velocity, res.p_max_breech,
                          res.psi_muzzle, fill, ok, reason)


def charge_for_range(weapon: Weapon, cartridge: Cartridge,
                     target_range: float, *,
                     min_impact_energy: float = 0.0,
                     pressure_limit: float | None = None,
                     environment: Environment | None = None,
                     launch_angle: float = math.radians(1.0)
                     ) -> ChargeSolution:
    """Навеска, при которой снаряд доносит заданную энергию на дальность.

    Внешняя задача входит в цикл: подбираем дульную скорость так, чтобы на
    нужной дистанции осталась требуемая энергия, затем находим навеску.
    """
    env = environment or Environment()
    proj = cartridge.projectile
    drag = DragModel("G7", proj.form_factor_g7)

    def impact_energy(v0: float) -> float:
        shot = ShotConditions(velocity=v0, launch_angle=launch_angle,
                              latitude=env.latitude, azimuth=env.azimuth)
        tr = solve_trajectory(proj.mass, proj.geometry.diameter, drag, shot,
                              env.atmosphere, env.wind,
                              max_range=target_range, dt=2e-3)
        p = tr.at_range(target_range)
        if p is None:
            return 0.0
        return 0.5 * proj.mass * p.speed ** 2

    v_needed = _bisect(impact_energy, 50.0, 2500.0,
                       max(min_impact_energy, 1e-6), tol=1e-3)
    if v_needed is None:
        return ChargeSolution(0.0, 0.0, 0.0, 0.0, 0.0, False,
                              f"Даже на 2500 м/с пуля не доносит "
                              f"{min_impact_energy:.0f} Дж на {target_range:.0f} м. "
                              "Нужна пуля с лучшим баллистическим коэффициентом "
                              "или больший калибр.")
    return charge_for_velocity(cartridge, weapon.barrel, v_needed,
                               pressure_limit=pressure_limit)


# --- подбор пороха -----------------------------------------------------------

@dataclass
class WebSolution:
    """Оптимальный свод зерна под конкретную систему."""

    web_scale: float          # во сколько раз масштабировать зерно
    web: float                # м, получившийся свод e1
    charge_mass: float
    muzzle_velocity: float
    peak_pressure: float
    burnt_fraction: float
    fill_ratio: float
    propellant: Propellant


def optimal_grain_web(cartridge: Cartridge, barrel: Barrel,
                      pressure_limit: float, *,
                      scale_lo: float = 0.35, scale_hi: float = 3.0,
                      steps: int = 16, metric: str = "mean"
                      ) -> WebSolution | None:
    """Ищет свод зерна, дающий наибольшую скорость при заданном давлении.

    Это центральный компромисс внутренней баллистики. Мелкий свод — порох
    сгорает рано, давление уходит в острый пик, и предел достигается на
    малой навеске: скорость низкая, ствол горит. Крупный свод — можно
    засыпать много, но порох не успевает догореть до дульного среза, и
    часть заряда вылетает несгоревшей. Оптимум лежит между, и найти его
    можно только перебором: аналитического выражения у него нет.

    Для каждого свода навеска подбирается под предел давления, так что
    сравнение идёт честно — при одинаковой нагрузке на ствол.
    """
    base = cartridge.charge.propellant
    best: WebSolution | None = None

    for i in range(steps):
        scale = scale_lo * (scale_hi / scale_lo) ** (i / (steps - 1))
        prop = deepcopy(base)
        prop.grain.shape = _scale_grain(prop.grain.shape, scale)
        prop.grain.__post_init__()
        cart = deepcopy(cartridge)
        cart.charge = Charge(prop, cartridge.charge.mass,
                             cartridge.charge.temperature)
        sol = charge_for_pressure(cart, barrel, pressure_limit, metric=metric)
        if sol.charge_mass <= 0.0 or sol.fill_ratio > 1.0:
            continue
        if best is None or sol.muzzle_velocity > best.muzzle_velocity:
            best = WebSolution(
                web_scale=scale, web=prop.web, charge_mass=sol.charge_mass,
                muzzle_velocity=sol.muzzle_velocity,
                peak_pressure=sol.peak_pressure,
                burnt_fraction=sol.burnt_fraction,
                fill_ratio=sol.fill_ratio, propellant=prop)
    return best


def web_for_charge_and_pressure(cartridge: Cartridge, barrel: Barrel,
                                charge_mass: float, pressure_limit: float, *,
                                metric: str = "mean",
                                scale_lo: float = 0.2, scale_hi: float = 6.0,
                                iters: int = 40) -> WebSolution | None:
    """Свод зерна, при котором ЗАДАННАЯ навеска даёт заданное давление.

    Обратная задача к обычной: навеска и предел давления известны (так
    задаётся реальный патрон), а подобрать надо порох. Именно так и
    рассуждает оружейник — «под эту гильзу и эту пулю нужен порох вот
    такой быстроты».

    Давление монотонно убывает с ростом свода: крупное зерно горит дольше,
    пик размазывается. Поэтому дихотомия идёт по масштабу зерна.
    """
    base = cartridge.charge.propellant

    def pressure_at(scale: float) -> tuple[float, Propellant, object]:
        prop = deepcopy(base)
        prop.grain.shape = _scale_grain(prop.grain.shape, scale)
        prop.grain.__post_init__()
        cart = deepcopy(cartridge)
        cart.charge = Charge(prop, charge_mass, cartridge.charge.temperature)
        res = solve_interior(build_system(cart, barrel), cart.charge,
                             cart.primer, FAST_OPTIONS)
        p = res.p_max_mean if metric == "mean" else res.p_max_breech
        return p, prop, res

    p_lo, _, _ = pressure_at(scale_lo)     # мелкое зерно -> высокое давление
    p_hi, _, _ = pressure_at(scale_hi)     # крупное -> низкое
    if p_lo < pressure_limit or p_hi > pressure_limit:
        return None

    lo, hi = scale_lo, scale_hi
    prop = res = None
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        p, prop, res = pressure_at(mid)
        if abs(p - pressure_limit) <= 1e-3 * pressure_limit:
            break
        if p > pressure_limit:
            lo = mid
        else:
            hi = mid
    if prop is None or res is None:
        return None
    cart = deepcopy(cartridge)
    cart.charge = Charge(prop, charge_mass, cartridge.charge.temperature)
    return WebSolution(
        web_scale=math.sqrt(lo * hi), web=prop.web, charge_mass=charge_mass,
        muzzle_velocity=res.muzzle_velocity, peak_pressure=res.p_max_breech,
        burnt_fraction=res.psi_muzzle, fill_ratio=cart.fill_ratio,
        propellant=prop)


def _scale_grain(shape, factor: float):
    """Масштабирует все линейные размеры зерна."""
    new = deepcopy(shape)
    for attr in ("diameter", "thickness", "length", "width",
                 "outer_diameter", "inner_diameter", "perf_diameter"):
        if hasattr(new, attr):
            setattr(new, attr, getattr(new, attr) * factor)
    return new


# --- ствол -------------------------------------------------------------------

def barrel_length_for_velocity(cartridge: Cartridge, barrel: Barrel,
                               target_velocity: float, *,
                               lo: float = 0.05, hi: float = 3.0
                               ) -> float | None:
    """Длина ствола, дающая заданную скорость при неизменной навеске."""
    def velocity(length: float) -> float:
        b = deepcopy(barrel)
        b.length = length
        b.profile = None
        b.__post_init__()
        system = build_system(cartridge, b)
        return solve_interior(system, cartridge.charge, cartridge.primer,
                              FAST_OPTIONS).muzzle_velocity

    return _bisect(velocity, lo, hi, target_velocity, tol=1e-4, iters=40)


def wall_for_pressure(bore_diameter: float, pressure: float, material: Metal,
                      *, burst_safety: float = 2.0,
                      autofrettage: float = 0.0,
                      temperature: float = 293.15) -> dict[str, float]:
    """Толщина стенки под давление с заданным запасом по РАЗРУШЕНИЮ.

    Из p_разр = (2/sqrt(3)) * sigma_в * ln(b/a) >= n * p следует

        b = a * exp( sqrt(3) * n * p / (2 * sigma_в) )

    Дополнительно считаем толщину по началу текучести — она больше, но её
    превышение для ствола не авария, а штатное самоавтофретирование.
    """
    a = 0.5 * bore_diameter
    su = material.ultimate_strength * (material.yield_at(temperature)
                                       / material.yield_strength)
    b_burst = a * math.exp(math.sqrt(3.0) * burst_safety * pressure / (2.0 * su))
    d_yield = required_outer_diameter(bore_diameter, pressure, material,
                                      safety_factor=1.0,
                                      autofrettage=autofrettage,
                                      temperature=temperature)
    return {
        "outer_diameter_burst": 2.0 * b_burst,
        "wall_burst": b_burst - a,
        "outer_diameter_elastic": d_yield,
        "wall_elastic": (0.5 * d_yield - a) if math.isfinite(d_yield)
        else float("inf"),
        "recommended_outer_diameter": max(2.0 * b_burst, min(d_yield, 8.0 * a)),
    }


def twist_for_stability(cartridge: Cartridge, muzzle_velocity: float,
                        target_sg: float = 1.6,
                        atmosphere: Atmosphere | None = None) -> float:
    """Шаг нарезов под заданный запас гироскопической устойчивости."""
    proj = cartridge.projectile
    return gyroscopic_twist_miller(proj.geometry.diameter,
                                   proj.geometry.total_length, proj.mass,
                                   muzzle_velocity, target_sg)


# --- рецептура пороха --------------------------------------------------------

@dataclass
class RecipeSolution:
    composition: dict[str, float]
    force: float
    flame_temp: float
    oxygen_balance: float
    oxidizing_ratio: float
    score: float
    notes: list[str] = field(default_factory=list)


def tune_oxidizer_balance(base: dict[str, float], *,
                          oxidizer: str = "Нитрат калия",
                          coolant: str = "Нитрогуанидин (охладитель)",
                          target_force: float | None = None,
                          max_flame_temp: float | None = None,
                          steps: int = 21) -> list[RecipeSolution]:
    """Перебор соотношения окислителя и охладителя в составе.

    Это тот самый рычаг «кислородный баланс против ресурса ствола»:
    добавляя окислитель, поднимаем силу пороха и температуру пламени
    (быстрее пуля, быстрее выгорает ствол); добавляя охладитель, наоборот.
    Возвращает всю сетку вариантов — решение, где встать, за игроком.
    """
    out: list[RecipeSolution] = []
    active = {k: v for k, v in base.items() if v > 0.0}
    total = sum(active.values())
    active = {k: v / total for k, v in active.items()}

    for i in range(steps):
        frac = 0.35 * i / (steps - 1)          # доля «добавки» 0...35%
        for additive in (oxidizer, coolant):
            if additive not in INGREDIENTS:
                continue
            mix = {k: v * (1.0 - frac) for k, v in active.items()}
            mix[additive] = mix.get(additive, 0.0) + frac
            try:
                chem = compute_thermochemistry(mix)
            except ValueError:
                continue
            notes = list(chem.warnings)
            score = chem.force / 1e6 - 0.35 * max(
                0.0, (chem.flame_temp - 2800.0) / 1000.0)
            if target_force is not None:
                score = -abs(chem.force - target_force) / 1e6
            if max_flame_temp is not None and chem.flame_temp > max_flame_temp:
                score -= 5.0
                notes.append(
                    f"T1 = {chem.flame_temp:.0f} К выше заданного потолка "
                    f"{max_flame_temp:.0f} К.")
            out.append(RecipeSolution(
                composition=mix, force=chem.force,
                flame_temp=chem.flame_temp,
                oxygen_balance=chem.oxygen_balance,
                oxidizing_ratio=chem.oxidizing_ratio,
                score=score, notes=notes))
            if frac == 0.0:
                break
    out.sort(key=lambda r: -r.score)
    return out


# --- комплексная оптимизация -------------------------------------------------

@dataclass
class LoadRequirement:
    """Техзадание на патрон."""

    target_range: float = 600.0
    min_impact_energy: float = 400.0
    max_pressure: float | None = None
    min_burst_safety: float = 1.8
    min_barrel_life: int = 3000
    min_stability: float = 1.4
    max_recoil_energy: float | None = None
    # Недогоревший заряд — это не «чуть меньше энергии», а выброшенный
    # в воздух порох, дульное пламя и разброс скоростей от выстрела к
    # выстрелу. Ниже этого порога вариант отбраковывается.
    min_burnt_fraction: float = 0.95


@dataclass
class LoadCandidate:
    propellant: Propellant
    charge_mass: float
    report: ShotReport
    effective_range: float
    score: float
    failures: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return not self.failures


def optimize_load(weapon: Weapon, cartridge: Cartridge,
                  library: dict[str, Propellant],
                  requirement: LoadRequirement,
                  environment: Environment | None = None
                  ) -> list[LoadCandidate]:
    """Перебирает пороха и для каждого подбирает навеску под ТЗ.

    Для каждого пороха берётся максимальная навеска, укладывающаяся
    одновременно в предел давления и в объём гильзы, — а дальше проверяется,
    хватает ли её на дальность и не убивает ли она ствол.
    Результат отсортирован: сначала подходящие, внутри — по ресурсу ствола.
    """
    env = environment or Environment()
    p_limit = requirement.max_pressure or cartridge.case.max_pressure
    out: list[LoadCandidate] = []

    for prop in library.values():
        cart = deepcopy(cartridge)
        cart.charge = Charge(prop, cartridge.charge.mass,
                             env.powder_temperature)
        sol = charge_for_pressure(cart, weapon.barrel, p_limit)
        if not sol.feasible or sol.charge_mass <= 0.0:
            # давление упирается в объём гильзы раньше предела — берём
            # максимум, который влезает
            bulk = prop.grain.bulk_density
            m_max = 0.995 * bulk * cart.chamber_volume()
            if m_max <= 0.0:
                continue
            cart.charge = Charge(prop, m_max, env.powder_temperature)
        else:
            cart.charge = Charge(prop, sol.charge_mass, env.powder_temperature)

        weapon_copy = deepcopy(weapon)
        report = fire(weapon_copy, cart, env, trajectory=False)

        proj = cart.projectile
        drag = DragModel("G7", proj.form_factor_g7)
        shot = ShotConditions(velocity=report.muzzle_velocity,
                              launch_angle=math.radians(1.0),
                              latitude=env.latitude, azimuth=env.azimuth)
        tr = solve_trajectory(proj.mass, proj.geometry.diameter, drag, shot,
                              env.atmosphere, env.wind,
                              max_range=requirement.target_range, dt=2e-3)
        pt = tr.at_range(requirement.target_range)
        e_impact = 0.5 * proj.mass * pt.speed ** 2 if pt else 0.0

        failures: list[str] = []
        if e_impact < requirement.min_impact_energy:
            failures.append(
                f"на {requirement.target_range:.0f} м остаётся {e_impact:.0f} Дж "
                f"при требуемых {requirement.min_impact_energy:.0f}")
        if report.min_safety_burst < requirement.min_burst_safety:
            failures.append(
                f"запас по разрушению {report.min_safety_burst:.2f} < "
                f"{requirement.min_burst_safety:.2f}")
        if report.barrel_life < requirement.min_barrel_life:
            failures.append(
                f"ресурс ствола {report.barrel_life} < "
                f"{requirement.min_barrel_life}")
        if report.stability < requirement.min_stability:
            failures.append(f"Sg = {report.stability:.2f} < "
                            f"{requirement.min_stability:.2f}")
        if (requirement.max_recoil_energy is not None
                and report.recoil_energy > requirement.max_recoil_energy):
            failures.append(f"отдача {report.recoil_energy:.1f} Дж > "
                            f"{requirement.max_recoil_energy:.1f} Дж")
        if report.interior.psi_muzzle < requirement.min_burnt_fraction:
            failures.append(
                f"сгорает лишь {100 * report.interior.psi_muzzle:.0f}% заряда "
                f"при требуемых {100 * requirement.min_burnt_fraction:.0f}% — "
                "порох слишком медленный для этой системы")
        if report.interior.stuck:
            failures.append("снаряд не выходит из ствола")

        # Ресурс важен, но он не должен перевешивать всё: между стволом на
        # 200 тыс. выстрелов при негодной баллистике и стволом на 15 тыс.,
        # который выполняет задачу, выбирать надо второй. Поэтому ресурс
        # входит логарифмом и с насыщением на десятикратном запасе.
        life_score = min(math.log10(max(report.barrel_life, 1)
                                    / max(requirement.min_barrel_life, 1)),
                         1.0)
        energy_score = e_impact / max(requirement.min_impact_energy, 1.0)
        score = (1.5 * life_score + energy_score
                 + 2.0 * report.interior.psi_muzzle - 3.0 * len(failures))
        out.append(LoadCandidate(prop, cart.charge.mass, report,
                                 tr.impact_range, score, failures))

    out.sort(key=lambda c: (not c.acceptable, -c.score))
    return out
