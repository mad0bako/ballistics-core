"""Регрессионные тесты: модель должна сходиться с учебниками и паспортами.

Это не тесты «код не падает». Каждый тест — сверка расчёта с независимо
известной величиной: таблицей Серебрякова, справочником по порохам,
паспортом патрона или опубликованной таблицей стрельбы. Если калибровка
где-то поедет, отвалится именно тот тест, который держит соответствующий
кусок физики.

Запуск:  python -m pytest tests -q      (или python tests/test_physics.py)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ballistics import presets                                    # noqa: E402
from ballistics.barrel import (Barrel, Chamber, Rifling,           # noqa: E402
                               required_outer_diameter)
from ballistics.cartridge import STANDARD_CASES, build_system     # noqa: E402
from ballistics.erosion import analyse_wear, surface_temperature   # noqa: E402
from ballistics.exterior import (Atmosphere, DragModel,            # noqa: E402
                                 ShotConditions, maximum_range,
                                 solve_trajectory)
from ballistics.grain import (MultiPerf, Sphere, Tube,             # noqa: E402
                              seven_perf, stick_powder)
from ballistics.interior import (Charge, InteriorOptions,          # noqa: E402
                                 solve_interior)
from ballistics.materials import STEEL_4140                        # noqa: E402
from ballistics.simulation import fire                             # noqa: E402
from ballistics.thermochem import compute_thermochemistry          # noqa: E402
from ballistics.units import RHO_STD                               # noqa: E402


def approx(value: float, expected: float, rel: float) -> bool:
    return abs(value - expected) <= rel * abs(expected)


# --- термохимия --------------------------------------------------------------

SINGLE_BASE = {
    "Нитроцеллюлоза 13.25% N (пироксилин №1)": 0.955,
    "Дифениламин (стабилизатор)": 0.010,
    "Дибутилфталат (флегматизатор)": 0.030,
    "Графит (антистатик, полировка)": 0.005,
}
DOUBLE_BASE = {
    "Нитроцеллюлоза 13.25% N (пироксилин №1)": 0.590,
    "Нитроглицерин": 0.360,
    "Централит-2 (стабилизатор)": 0.045,
    "Графит (антистатик, полировка)": 0.005,
}


def test_single_base_matches_handbook():
    """Пироксилиновый порох: f, T1, k, коволюм, число молей газа."""
    c = compute_thermochemistry(SINGLE_BASE)
    assert 0.95e6 <= c.force <= 1.00e6, c.force
    assert 2800 <= c.flame_temp <= 3000, c.flame_temp
    assert 1.21 <= c.gamma <= 1.26, c.gamma
    assert 0.95e-3 <= c.covolume <= 1.10e-3, c.covolume
    assert 38.0 <= c.gas_moles <= 43.0, c.gas_moles
    # теплота взрывчатого превращения 800-950 кал/г
    assert 3.3e6 <= c.heat_of_explosion <= 4.0e6, c.heat_of_explosion
    # кислородный баланс пироксилина заметно отрицательный
    assert -45.0 <= c.oxygen_balance <= -25.0, c.oxygen_balance


def test_double_base_is_hotter_and_stronger():
    """Добавление нитроглицерина обязано поднять и силу, и температуру."""
    sb = compute_thermochemistry(SINGLE_BASE)
    db = compute_thermochemistry(DOUBLE_BASE)
    assert db.force > sb.force
    assert db.flame_temp > sb.flame_temp
    assert 3200 <= db.flame_temp <= 3400, db.flame_temp
    # справочная вилка для двухосновных с ~36% НГ: 1.00...1.15 МДж/кг
    assert 1.00e6 <= db.force <= 1.20e6, db.force
    # более горячий порох даёт более окислительные газы
    assert db.oxidizing_ratio > sb.oxidizing_ratio


def test_oxidizer_shifts_oxygen_balance():
    """Нитрат калия обязан двигать кислородный баланс вверх."""
    base = compute_thermochemistry(SINGLE_BASE)
    mix = {k: v * 0.8 for k, v in SINGLE_BASE.items()}
    mix["Нитрат калия"] = 0.2
    rich = compute_thermochemistry(mix)
    assert rich.oxygen_balance > base.oxygen_balance
    assert rich.oxidizing_ratio > base.oxidizing_ratio


def test_coolant_lowers_flame_temperature():
    """Нитрогуанидин — охладитель, обязан снижать T1."""
    base = compute_thermochemistry(SINGLE_BASE)
    mix = {k: v * 0.7 for k, v in SINGLE_BASE.items()}
    mix["Нитрогуанидин (охладитель)"] = 0.3
    cool = compute_thermochemistry(mix)
    assert cool.flame_temp < base.flame_temp


# --- функция формы -----------------------------------------------------------

def test_sphere_form_function_is_exact():
    """Шар: psi = 1-(1-z)^3, то есть kappa=3, lambda=-1, mu=1/3 точно."""
    f = Sphere(0.6e-3).form_function()
    assert approx(f.kappa, 3.0, 1e-9)
    assert approx(f.lam, -1.0, 1e-9)
    assert approx(f.mu, 1.0 / 3.0, 1e-9)
    assert approx(f.psi(1.0), 1.0, 1e-9)


def test_tube_burns_almost_neutrally():
    """Трубка: внутренняя поверхность растёт как убывает внешняя."""
    f = Tube(1.2e-3, 0.3e-3, 10e-3).form_function()
    assert approx(f.kappa, 1.0, 0.10)
    assert abs(f.lam) < 0.10
    assert abs(f.mu) < 1e-9
    assert 0.85 < f.progressivity < 1.05


def test_seven_perf_is_progressive():
    """Семиканальное зерно: kappa<1, поверхность растёт, есть догорание."""
    g = seven_perf(10.0, 1.0, 12.0)
    f = g.form
    assert 0.70 <= f.kappa <= 0.85, f.kappa
    assert 0.10 <= f.lam <= 0.20, f.lam
    assert f.mu < 0.0
    assert 0.82 <= f.psi_1 <= 0.90, f.psi_1
    assert f.progressivity > 1.0
    assert approx(f.psi(f.z_k), 1.0, 1e-9)


def test_multiperf_web_formula_reduces_to_tube():
    """Формула свода при m=0 обязана вырождаться в трубчатую (D-d)/4."""
    shape = MultiPerf(10e-3, 1e-3, 12e-3, perforations=1)
    assert approx(shape.web(), (10e-3 - 1e-3) / 4.0, 1e-9)


def test_kappa_equals_surface_times_web_over_volume():
    """Независимая проверка: kappa = S1*e1/V1 по определению."""
    for shape in (Sphere(0.5e-3), Tube(1.2e-3, 0.3e-3, 8e-3),
                  MultiPerf(10e-3, 1.0e-3, 12e-3, 7)):
        f = shape.form_function()
        assert approx(f.kappa, shape.check_kappa(), 1e-6), shape.name


# --- внутренняя баллистика ---------------------------------------------------

def test_peak_pressure_independent_of_history_thinning():
    """Прореживание истории не имеет права менять найденный пик давления.

    Регрессия на реальную ошибку: пик острый, и если искать максимум
    только по сохранённым точкам, он проваливается между ними.
    """
    weapon, cart = presets.rifle_762x51()
    system = build_system(cart, weapon.barrel)
    fine = solve_interior(system, cart.charge, cart.primer,
                          InteriorOptions(sample_stride=1))
    coarse = solve_interior(system, cart.charge, cart.primer,
                            InteriorOptions(sample_stride=16))
    assert approx(coarse.p_max_breech, fine.p_max_breech, 0.02)
    assert approx(coarse.muzzle_velocity, fine.muzzle_velocity, 0.005)


def test_squib_load_is_detected():
    """Ничтожная навеска обязана оставить пулю в стволе, а не «выстрелить».

    Заодно это защита от зависания: без обнаружения затяжного выстрела
    интегратор молотил все 200 тыс. шагов, и подбор навески замедлялся
    в полсотни раз.
    """
    import time
    weapon, cart = presets.rifle_762x51()
    system = build_system(cart, weapon.barrel)
    tiny = Charge(cart.charge.propellant, 0.05e-3)
    start = time.time()
    r = solve_interior(system, tiny, cart.primer)
    elapsed = time.time() - start
    assert r.stuck, "пуля обязана застрять"
    assert r.muzzle_velocity < 50.0, r.muzzle_velocity
    assert elapsed < 1.0, f"расчёт затяжного выстрела занял {elapsed:.1f} с"
    assert any("застрял" in w for w in r.warnings)


def test_charge_solver_is_fast_enough_for_interactive_use():
    """Подбор навески обязан укладываться в интерактивное время."""
    import time
    from ballistics.design import charge_for_pressure
    weapon, cart = presets.rifle_762x51()
    start = time.time()
    charge_for_pressure(cart, weapon.barrel, 415e6)
    assert time.time() - start < 2.0


def test_lagrange_pressure_ordering():
    """p_кн > p_ср > p_дно — иначе где-то перепутаны формулы Лагранжа."""
    weapon, cart = presets.rifle_762x51()
    r = solve_interior(build_system(cart, weapon.barrel), cart.charge,
                       cart.primer)
    assert r.p_max_breech > r.p_max_mean > r.p_max_base
    # для стрелкового оружия разница невелика: omega/m мало
    assert r.p_max_breech / r.p_max_base < 1.20


def test_thermal_efficiency_in_physical_range():
    """КПД ствольной системы — 20-35%, всё остальное подозрительно."""
    for factory in (presets.rifle_762x51, presets.carbine_556x45,
                    presets.sniper_338lm):
        weapon, cart = factory()
        r = solve_interior(build_system(cart, weapon.barrel), cart.charge,
                           cart.primer)
        assert 0.15 <= r.thermal_efficiency <= 0.40, (
            factory.__name__, r.thermal_efficiency)


def test_slower_powder_lowers_peak_pressure():
    """При равной навеске крупный свод обязан снижать пик давления."""
    from copy import deepcopy
    weapon, cart = presets.rifle_762x51()
    fast = solve_interior(build_system(cart, weapon.barrel), cart.charge,
                          cart.primer)
    slow_cart = deepcopy(cart)
    prop = slow_cart.charge.propellant
    prop.grain.shape.diameter *= 1.6
    prop.grain.__post_init__()
    slow = solve_interior(build_system(slow_cart, weapon.barrel),
                          slow_cart.charge, slow_cart.primer)
    assert slow.p_max_breech < fast.p_max_breech


def test_hot_powder_raises_pressure():
    """Жаркий день — выше давление: термочувствительность скорости горения."""
    weapon, cart = presets.rifle_762x51()
    cold = Charge(cart.charge.propellant, cart.charge.mass, 253.15)
    hot = Charge(cart.charge.propellant, cart.charge.mass, 323.15)
    system = build_system(cart, weapon.barrel)
    p_cold = solve_interior(system, cold, cart.primer).p_max_breech
    p_hot = solve_interior(system, hot, cart.primer).p_max_breech
    assert p_hot > p_cold * 1.05


# --- гильза и камора ---------------------------------------------------------

def test_case_capacities_match_published():
    """Объём гильзы из чертежа обязан сойтись с паспортной ёмкостью."""
    for key in ("9x19", "5.56x45", "7.62x51", ".338LM"):
        case = STANDARD_CASES[key]()
        computed = case.geometry.internal_volume()
        assert approx(computed, case.capacity, 0.10), (key, computed,
                                                       case.capacity)


# --- ствол -------------------------------------------------------------------

def test_lame_matches_closed_form():
    """Обратная задача по стенке обязана быть согласована с прямой."""
    d = 7.62e-3
    p = 400e6
    outer = required_outer_diameter(d, p, STEEL_4140, safety_factor=1.0)
    barrel = Barrel(material=STEEL_4140, bore_diameter=d, length=0.5)
    barrel.profile.stations = [(0.0, outer), (0.5, outer)]
    st = barrel.stress_at(0.25, p)
    assert approx(st.safety_factor, 1.0, 0.02)


def test_autofrettage_raises_elastic_limit():
    """Автофретирование обязано поднимать упругий предел стенки."""
    ch = Chamber(1e-4, 0.05, 0.02, 0.02)
    plain = Barrel(bore_diameter=0.152, length=4.0, chamber=ch)
    treated = Barrel(bore_diameter=0.152, length=4.0, chamber=ch,
                     autofrettage=0.6)
    assert (treated.elastic_limit_pressure(2.0)
            > plain.elastic_limit_pressure(2.0) * 1.15)


def test_thicker_wall_is_stronger():
    for d_out in (20e-3, 30e-3, 45e-3):
        b = Barrel(bore_diameter=7.62e-3, length=0.5)
        b.profile.stations = [(0.0, d_out), (0.5, d_out)]
        st = b.stress_at(0.25, 400e6)
        assert st.burst_safety_factor > 0.0
    thin = Barrel(bore_diameter=7.62e-3, length=0.5)
    thin.profile.stations = [(0.0, 16e-3), (0.5, 16e-3)]
    thick = Barrel(bore_diameter=7.62e-3, length=0.5)
    thick.profile.stations = [(0.0, 40e-3), (0.5, 40e-3)]
    assert (thick.stress_at(0.25, 400e6).burst_safety_factor
            > thin.stress_at(0.25, 400e6).burst_safety_factor)


# --- пуля --------------------------------------------------------------------

def test_bullet_masses_match_catalogue():
    """Масса пули считается из обводов и материалов, а не назначается."""
    for factory, expected in ((presets.bullet_762_168, 10.886e-3),
                              (presets.bullet_556_62, 4.02e-3),
                              (presets.bullet_9mm_124, 8.03e-3),
                              (presets.bullet_338_250, 16.2e-3)):
        proj = factory()
        assert approx(proj.mass, expected, 0.05), (factory.__name__,
                                                   proj.mass, expected)


def test_ballistic_coefficient_matches_catalogue():
    """BC(G7) матчевой .308 — 0.224 фнт/дюйм2."""
    proj = presets.bullet_762_168()
    assert approx(proj.bc_g7_imperial, 0.224, 0.08), proj.bc_g7_imperial


def test_boattail_beats_flat_base():
    """Запоясковый конус обязан улучшать форм-фактор."""
    from ballistics.projectile import flat_base_round_nose, spitzer_boattail
    sharp = spitzer_boattail(7.823e-3, 31e-3)
    blunt = flat_base_round_nose(7.823e-3, 31e-3)
    assert sharp.form_factor_g7 < blunt.form_factor_g7


# --- внешняя баллистика ------------------------------------------------------

def test_standard_atmosphere():
    atm = Atmosphere()
    assert approx(atm.density(0.0), RHO_STD, 1e-6)
    assert approx(atm.sound_speed(0.0), 340.3, 0.01)
    # с высотой плотность падает
    assert atm.density(3000.0) < 0.78 * atm.density(0.0)


def test_humid_air_is_lighter():
    """Влажный воздух легче сухого: молярная масса пара 18 против 29."""
    dry = Atmosphere(humidity=0.0, sea_level_temperature=303.15)
    wet = Atmosphere(humidity=1.0, sea_level_temperature=303.15)
    assert wet.density(0.0) < dry.density(0.0)


def test_m118lr_trajectory_matches_published_table():
    """Сверка с опубликованной таблицей: .308 175 гран, 790 м/с, BC7=0.243.

    Паспорт на 1000 м: время полёта 1.73 с, скорость ~420 м/с,
    понижение относительно линии бросания 11.9-12.1 м.
    """
    m, d = 11.34e-3, 7.823e-3
    bc7 = 0.243 * 0.45359237 / 0.0254 ** 2
    drag = DragModel("G7", m / (bc7 * d * d))
    shot = ShotConditions(velocity=790.0, launch_angle=0.0, stability=1.8,
                          spin_drift=False, coriolis=False)
    tr = solve_trajectory(m, d, drag, shot, Atmosphere(), max_range=1000.0)
    p = tr.at_range(1000.0)
    assert approx(p.t, 1.73, 0.03), p.t
    assert approx(p.speed, 420.0, 0.05), p.speed
    assert approx(-p.y, 12.0, 0.06), p.y


def test_max_range_angle_below_45_degrees():
    """С сопротивлением воздуха оптимальный угол всегда меньше 45 градусов."""
    m, d = 11.34e-3, 7.823e-3
    drag = DragModel("G7", 1.1)
    shot = ShotConditions(velocity=790.0, spin_drift=False, coriolis=False)
    rng, angle = maximum_range(m, d, drag, shot, Atmosphere())
    assert math.radians(25.0) < angle < math.radians(45.0), math.degrees(angle)
    assert 3500.0 < rng < 6500.0, rng


def test_drag_peaks_at_transonic():
    """Обе стандартные функции обязаны иметь пик сопротивления у M~1.1."""
    for std in ("G1", "G7"):
        model = DragModel(std, 1.0)
        peak_mach = max((m * 0.01 for m in range(10, 500)),
                        key=lambda mm: model.cd(mm))
        assert 1.0 < peak_mach < 1.3, (std, peak_mach)
        assert model.cd(peak_mach) > 1.5 * model.cd(0.5)


# --- износ -------------------------------------------------------------------

def test_bore_surface_temperature_physical():
    """Пик температуры поверхности канала — сотни кельвинов выше начальной,
    но заведомо ниже температуры газов."""
    weapon, cart = presets.rifle_762x51()
    r = solve_interior(build_system(cart, weapon.barrel), cart.charge,
                       cart.primer)
    th = surface_temperature(r, STEEL_4140, initial_temp=300.0)
    assert 600.0 < th.peak_surface_temp < 1600.0, th.peak_surface_temp
    assert th.peak_surface_temp < r.max_gas_temp


def test_barrel_life_order_of_magnitude():
    """Ресурс винтовочного ствола — десятки тысяч, а не сотни и не миллионы."""
    weapon, cart = presets.rifle_762x51()
    report = fire(weapon, cart, trajectory=False)
    assert 4_000 <= report.barrel_life <= 40_000, report.barrel_life


def test_hotter_propellant_kills_barrel_faster():
    """Более горячий порох обязан резать ресурс ствола."""
    from copy import deepcopy
    weapon, cart = presets.rifle_762x51()
    cool = fire(deepcopy(weapon), cart, trajectory=False)
    hot_cart = deepcopy(cart)
    hot_cart.charge.propellant = presets.library()[
        "Двухосновный высокоэнергетический"]
    hot = fire(deepcopy(weapon), hot_cart, trajectory=False)
    assert hot.wear.throat_wear_per_shot > cool.wear.throat_wear_per_shot


def test_lining_extends_barrel_life():
    """Хромирование канала обязано увеличивать ресурс в разы, но не в сотни."""
    from copy import deepcopy
    from ballistics.materials import CHROME_LINING
    weapon, cart = presets.rifle_762x51()
    plain = fire(deepcopy(weapon), cart, trajectory=False)
    lined_weapon = deepcopy(weapon)
    lined_weapon.barrel.lining = CHROME_LINING
    lined_weapon.barrel.lining_thickness = 25e-6
    lined = fire(lined_weapon, cart, trajectory=False)
    ratio = lined.barrel_life / plain.barrel_life
    assert 1.2 <= ratio <= 5.0, ratio


def test_wear_accumulates_and_degrades():
    """Настрел обязан снижать ресурс, скорость и кучность."""
    from ballistics.simulation import fire_series
    weapon, cart = presets.carbine_556x45()
    before = weapon.condition.health(weapon.barrel)
    series = fire_series(weapon, cart, 120, rate_per_minute=200.0,
                         sample_every=40)
    after = weapon.condition.health(weapon.barrel)
    assert after < before
    assert weapon.condition.temperature > 293.15
    assert weapon.condition.throat_erosion > 0.0
    assert series.shots == 120


# --- сквозной выстрел --------------------------------------------------------

def test_presets_run_and_are_safe():
    """Все пресеты обязаны считаться и не разрывать ствол."""
    for name, factory in presets.ALL_PRESETS.items():
        weapon, cart = factory()
        r = fire(weapon, cart, trajectory=False)
        assert r.muzzle_velocity > 0.0, name
        assert r.min_safety_burst > 1.0, (name, r.min_safety_burst)
        assert r.peak_pressure_breech > 0.0, name


def test_presets_match_passport_data():
    """Пресеты обязаны воспроизводить паспортные пары навеска/скорость/давление.

    Допуски разные, потому что модель разной точности в разных режимах:
    на умеренной плотности заряжания она даёт единицы процентов, на
    предельной (5.56, .338) уезжает сильнее — это задокументировано.
    Тест фиксирует достигнутое, чтобы калибровка не разъехалась молча.
    """
    checks = {
        # патрон: (v паспорт, допуск v, p паспорт МПа, допуск p)
        "9x19": (360.0, 0.03, 235.0, 0.05),
        "7.62x51": (810.0, 0.03, 415.0, 0.05),
        "152mm": (655.0, 0.03, 300.0, 0.06),
        "5.56x45": (940.0, 0.09, 430.0, 0.10),
        ".338LM": (890.0, 0.06, 420.0, 0.05),
    }
    for name, (vref, vtol, pref, ptol) in checks.items():
        weapon, cart = presets.ALL_PRESETS[name]()
        r = fire(weapon, cart, trajectory=False)
        assert approx(r.muzzle_velocity, vref, vtol), (
            name, "скорость", r.muzzle_velocity, vref)
        assert approx(r.interior.p_max_mean / 1e6, pref, ptol), (
            name, "давление", r.interior.p_max_mean / 1e6, pref)


def test_progressive_grain_lowers_peak_at_equal_velocity():
    """Прогрессивное зерно обязано давать ту же скорость при меньшем пике.

    Это ключевое утверждение всей конструкции: за счёт формы зерна можно
    купить ресурс ствола, ничего не теряя в баллистике.
    """
    from ballistics.grain import cord_powder, seven_perf
    from ballistics.propellant import SINGLE_BASE, make_propellant
    weapon, cart = presets.sniper_338lm()
    degressive = make_propellant("цилиндр", SINGLE_BASE,
                                 lambda rho: cord_powder(1.60, 2.6, rho))
    progressive = make_propellant("7-канальное", SINGLE_BASE,
                                  lambda rho: seven_perf(4.0, 0.40, 5.0, rho))
    assert degressive.grain.form.progressivity < 1.0
    assert progressive.grain.form.progressivity > 1.0

    from copy import deepcopy
    results = {}
    for label, prop in (("deg", degressive), ("prog", progressive)):
        c = deepcopy(cart)
        c.charge = Charge(prop, 6.30e-3)
        results[label] = solve_interior(build_system(c, weapon.barrel),
                                        c.charge, c.primer)
    # скорости сопоставимы, а пик у прогрессивного заметно ниже
    assert approx(results["prog"].muzzle_velocity,
                  results["deg"].muzzle_velocity, 0.05)
    assert results["prog"].p_max_mean < 0.85 * results["deg"].p_max_mean


def test_driving_band_not_shell_body():
    """Врезание считается по ведущему пояску, а не по корпусу снаряда."""
    from ballistics.materials import COPPER
    from ballistics.projectile import artillery_shell
    shell = artillery_shell(0.152, 0.66, 43.56)
    assert shell.driving_band is COPPER
    assert shell.core_material is not COPPER


def test_overspin_bursts_thin_jacket():
    """Слишком крутой шаг обязан разорвать оболочку центробежными силами."""
    from copy import deepcopy
    weapon, cart = presets.rifle_762x51()
    w = deepcopy(weapon)
    w.barrel.rifling.twist = 0.055      # заведомо запредельно крутой шаг
    r = fire(w, cart, trajectory=False)
    assert any("центробежное" in v.lower() for v in r.verdicts), r.verdicts
    assert not r.safe


def test_effective_range_monotone_in_energy_threshold():
    """Чем выше требуемая энергия у цели, тем меньше эффективная дальность."""
    from ballistics.simulation import effective_range
    weapon, cart = presets.rifle_762x51()
    far = effective_range(weapon, cart, min_energy=200.0)
    near = effective_range(weapon, cart, min_energy=1500.0)
    assert near < far


# --- допуски -----------------------------------------------------------------

def test_tighter_tolerances_shrink_dispersion():
    """Матчевые допуски обязаны давать меньший разброс, чем гаражные."""
    from ballistics.tolerances import Tolerances, monte_carlo
    weapon, cart = presets.rifle_762x51()
    match = monte_carlo(weapon, cart, Tolerances.factory_match(), shots=25,
                        target_range=300.0)
    rough = monte_carlo(weapon, cart, Tolerances.garage(), shots=25,
                        target_range=300.0)
    assert match.velocity_sd < rough.velocity_sd
    assert match.group_moa < rough.group_moa
    assert 0.05 < match.group_moa < 5.0, match.group_moa


# --- обратные задачи ---------------------------------------------------------

def test_charge_solver_hits_requested_pressure():
    from ballistics.design import charge_for_pressure
    weapon, cart = presets.rifle_762x51()
    sol = charge_for_pressure(cart, weapon.barrel, 380e6, metric="mean")
    assert sol.charge_mass > 0.0
    r = solve_interior(
        build_system(_with_charge(cart, sol.charge_mass), weapon.barrel),
        Charge(cart.charge.propellant, sol.charge_mass), cart.primer)
    assert approx(r.p_max_mean, 380e6, 0.02), r.p_max_mean


def test_barrel_length_solver_is_monotone():
    from ballistics.design import barrel_length_for_velocity
    weapon, cart = presets.rifle_762x51()
    base = fire(weapon, cart, trajectory=False).muzzle_velocity
    length = barrel_length_for_velocity(cart, weapon.barrel, base * 0.92)
    assert length is not None
    assert length < weapon.barrel.length


def test_wall_solver_gives_requested_margin():
    from ballistics.design import wall_for_pressure
    res = wall_for_pressure(7.62e-3, 400e6, STEEL_4140, burst_safety=2.0)
    b = Barrel(material=STEEL_4140, bore_diameter=7.62e-3, length=0.5,
               rifling=Rifling(grooves=0, groove_depth=0.0))
    d = res["outer_diameter_burst"]
    b.profile.stations = [(0.0, d), (0.5, d)]
    st = b.stress_at(0.25, 400e6)
    assert approx(st.burst_safety_factor, 2.0, 0.03), st.burst_safety_factor


def _with_charge(cartridge, mass: float):
    from copy import deepcopy
    c = deepcopy(cartridge)
    c.charge = Charge(cartridge.charge.propellant, mass,
                      cartridge.charge.temperature)
    return c


# --- мастерская --------------------------------------------------------------

def test_workshop_quality_maps_to_tolerances():
    """Лучше мастерская — жёстче допуски. Это связка игры с физикой."""
    from ballistics.workshop import (arsenal_workshop, garage_workshop,
                                     gunsmith_shop)
    garage = garage_workshop().tolerances()
    shop = gunsmith_shop().tolerances()
    arsenal = arsenal_workshop().tolerances()
    assert garage.charge_mass_sd > shop.charge_mass_sd > arsenal.charge_mass_sd
    assert garage.cg_offset > shop.cg_offset > arsenal.cg_offset


def test_garage_cannot_chrome_plate():
    """Без гальванической ванны хромировать канал нечем."""
    from ballistics.materials import CHROME_LINING
    from ballistics.workshop import garage_workshop
    barrel, log = garage_workshop().make_barrel(
        STEEL_4140, 7.62e-3, 0.6, Rifling(), lining=CHROME_LINING)
    assert barrel is None
    assert any("ванн" in line for line in log)


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  ПРОВАЛ {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ОШИБКА {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} из {len(tests)} пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
