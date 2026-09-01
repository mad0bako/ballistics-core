"""Демонстрация ядра: пять сценариев, которые и составляют игровой цикл.

    python examples/demo.py            все сценарии
    python examples/demo.py 3          только третий
"""
from __future__ import annotations

import math
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ballistics import presets
from ballistics.barrel import Barrel, Chamber, Rifling
from ballistics.cartridge import build_system
from ballistics.design import (LoadRequirement, charge_for_pressure,
                               optimal_grain_web, optimize_load,
                               tune_oxidizer_balance, twist_for_stability,
                               wall_for_pressure)
from ballistics.erosion import chemical_wear_factor
from ballistics.exterior import Atmosphere, Wind
from ballistics.interior import Charge, solve_interior
from ballistics.materials import (CHROME_LINING, STEEL_4140, STEEL_CRMOV,
                                  STEEL_MILD)
from ballistics.propellant import SINGLE_BASE, make_propellant
from ballistics.grain import cord_powder
from ballistics.simulation import Environment, effective_range, fire, fire_series
from ballistics.thermochem import compute_thermochemistry
from ballistics.tolerances import Tolerances, monte_carlo
from ballistics.units import moa_from_rad
from ballistics.workshop import (arsenal_workshop, garage_workshop,
                                 gunsmith_shop)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --- 1. Порох: что даёт кислородный баланс -----------------------------------

def demo_powder() -> None:
    rule("1. РЕЦЕПТУРА ПОРОХА: чем платят за силу заряда")
    print("Базовый пироксилиновый состав и что с ним делают добавки.\n")
    print(f"{'состав':38s} {'f, кДж/кг':>10s} {'T1, К':>7s} {'k':>6s} "
          f"{'КБ, %':>7s} {'окисл.':>7s} {'износ':>6s}")
    variants = [
        ("базовый одноосновный", dict(SINGLE_BASE)),
        ("+10% нитрата калия (окислитель)",
         {**{k: v * 0.9 for k, v in SINGLE_BASE.items()}, "Нитрат калия": 0.10}),
        ("+25% нитрата калия",
         {**{k: v * 0.75 for k, v in SINGLE_BASE.items()}, "Нитрат калия": 0.25}),
        ("+30% нитроглицерина",
         {**{k: v * 0.7 for k, v in SINGLE_BASE.items()}, "Нитроглицерин": 0.30}),
        ("+25% нитрогуанидина (охладитель)",
         {**{k: v * 0.75 for k, v in SINGLE_BASE.items()},
          "Нитрогуанидин (охладитель)": 0.25}),
        ("+15% оксамида (охладитель)",
         {**{k: v * 0.85 for k, v in SINGLE_BASE.items()},
          "Оксамид (охладитель)": 0.15}),
    ]
    for label, mix in variants:
        c = compute_thermochemistry(mix)
        wear = chemical_wear_factor(c.oxidizing_ratio, c.flame_temp)
        print(f"{label:38s} {c.force / 1e3:10.0f} {c.flame_temp:7.0f} "
              f"{c.gamma:6.3f} {c.oxygen_balance:+7.1f} "
              f"{c.oxidizing_ratio:7.2f} {wear:6.2f}")
    print("\nВидно главный компромисс: окислитель поднимает силу заряда, но\n"
          "вместе с ней температуру пламени и агрессивность газов — то есть\n"
          "покупает скорость за ресурс ствола. Охладитель делает обратное.")

    print("\nПеребор соотношения окислителя и охладителя (лучшие пять):")
    grid = tune_oxidizer_balance(SINGLE_BASE, max_flame_temp=3100.0)
    for sol in grid[:5]:
        add = {k: v for k, v in sol.composition.items()
               if k in ("Нитрат калия", "Нитрогуанидин (охладитель)")}
        label = ", ".join(f"{k.split()[0]} {100 * v:.0f}%" for k, v in add.items())
        print(f"   {label or 'без добавок':28s} f={sol.force / 1e3:6.0f} "
              f"T1={sol.flame_temp:6.0f} КБ={sol.oxygen_balance:+6.1f}")


# --- 2. Навеска: сила, давление и ствол ---------------------------------------

def demo_charge() -> None:
    rule("2. НАВЕСКА: сколько пороха, чтобы хватило силы и не убило ствол")
    weapon, cart = presets.rifle_762x51()
    print(f"{weapon.name}, {cart.name}\n{cart.describe()}\n")
    print(f"{'навеска, г':>10s} {'v0, м/с':>8s} {'p_ср, МПа':>10s} "
          f"{'p_кн, МПа':>10s} {'psi_дул':>8s} {'запас разр.':>12s} "
          f"{'ресурс':>9s} {'дальн. 400Дж':>13s}")
    for grams in (2.4, 2.7, 2.9, 3.1, 3.3):
        c = deepcopy(cart)
        c.charge = Charge(cart.charge.propellant, grams * 1e-3)
        w = deepcopy(weapon)
        r = fire(w, c, trajectory=False)
        rng = effective_range(deepcopy(weapon), c, min_energy=400.0)
        flag = "  <-- опасно" if r.min_safety_burst < 1.5 else ""
        print(f"{grams:10.2f} {r.muzzle_velocity:8.0f} "
              f"{r.interior.p_max_mean / 1e6:10.0f} "
              f"{r.peak_pressure_breech / 1e6:10.0f} "
              f"{r.interior.psi_muzzle:8.3f} {r.min_safety_burst:12.2f} "
              f"{r.barrel_life:9d} {rng:11.0f} м{flag}")

    print("\nПодбор под паспортный предел 415 МПа:")
    sol = charge_for_pressure(cart, weapon.barrel, 415e6, metric="mean")
    print(f"   навеска {sol.charge_mass * 1e3:.3f} г -> "
          f"{sol.muzzle_velocity:.0f} м/с, заполнение гильзы "
          f"{100 * sol.fill_ratio:.0f}%, сгорело {100 * sol.burnt_fraction:.1f}%")

    print("\nПодбор оптимального свода зерна при том же пределе давления:")
    best = optimal_grain_web(cart, weapon.barrel, 415e6, steps=10)
    if best is not None:
        print(f"   свод 2e1 = {2e3 * best.web:.3f} мм даёт "
              f"{best.muzzle_velocity:.0f} м/с при навеске "
              f"{best.charge_mass * 1e3:.3f} г "
              f"(сгорает {100 * best.burnt_fraction:.1f}%)")
        print("   Мелкий свод упирается в давление на малой навеске, крупный —\n"
              "   не успевает сгореть. Оптимум лежит строго между.")


# --- 3. Ствол: длина, стенка, нарезы ------------------------------------------

def demo_barrel() -> None:
    rule("3. СТВОЛ: длина под скорость, стенка под давление, шаг под пулю")
    weapon, cart = presets.rifle_762x51()

    print("Длина ствола против скорости (навеска неизменна):")
    print(f"{'длина, мм':>10s} {'v0, м/с':>8s} {'p_дул, МПа':>11s} "
          f"{'psi_дул':>8s} {'масса, кг':>10s}")
    for length in (0.406, 0.508, 0.610, 0.711, 0.813):
        w = deepcopy(weapon)
        w.barrel.length = length
        w.barrel.profile = None
        w.barrel.__post_init__()
        r = fire(w, cart, trajectory=False)
        print(f"{length * 1e3:10.0f} {r.muzzle_velocity:8.0f} "
              f"{r.interior.muzzle_pressure / 1e6:11.0f} "
              f"{r.interior.psi_muzzle:8.3f} {w.barrel.mass:10.2f}")
    print("Каждые следующие 100 мм дают всё меньше: газ уже расширился, а\n"
          "масса и дульное давление растут линейно.")

    print("\nТолщина стенки под 415 МПа с запасом по разрушению 2.0:")
    print(f"{'материал':46s} {'D нар., мм':>11s} {'стенка, мм':>11s}")
    for mat in (STEEL_MILD, STEEL_4140, STEEL_CRMOV):
        res = wall_for_pressure(7.62e-3, 415e6, mat, burst_safety=2.0)
        print(f"{mat.name:46s} {res['outer_diameter_burst'] * 1e3:11.1f} "
              f"{res['wall_burst'] * 1e3:11.1f}")
    af = wall_for_pressure(0.152, 300e6, STEEL_CRMOV, burst_safety=2.0,
                           autofrettage=0.6)
    print(f"\n152-мм ствол под 300 МПа: наружный диаметр по разрушению "
          f"{af['outer_diameter_burst'] * 1e3:.0f} мм,\n"
          f"по упругости с автофретированием 60% — "
          f"{af['outer_diameter_elastic'] * 1e3:.0f} мм.")

    print("\nШаг нарезов под устойчивость пули:")
    for sg in (1.3, 1.5, 2.0):
        twist = twist_for_stability(cart, 800.0, sg)
        print(f"   Sg = {sg:.1f}  ->  шаг {twist * 1e3:.0f} мм "
              f"({twist / cart.projectile.geometry.diameter:.1f} клб)")
    print("Проверка на реальных шагах:")
    for twist in (0.254, 0.305, 0.356, 0.420):
        w = deepcopy(weapon)
        w.barrel.rifling.twist = twist
        r = fire(w, cart, trajectory=False)
        mark = "  неустойчива!" if r.stability < 1.2 else ""
        print(f"   шаг {twist * 1e3:3.0f} мм: Sg = {r.stability:5.2f}, "
              f"закрутка {r.interior.spin_rate / (2 * math.pi):6.0f} об/с{mark}")


# --- 4. Износ ствола ----------------------------------------------------------

def demo_wear() -> None:
    rule("4. ИЗНОС: чем платит ствол за каждый выстрел")
    weapon, cart = presets.rifle_762x51()

    print("Влияние пороха на ресурс при одинаковом пределе давления 415 МПа:")
    print(f"{'порох':36s} {'T1, К':>6s} {'v0':>6s} {'T пов., К':>10s} "
          f"{'хим.':>6s} {'нм/выстр':>9s} {'ресурс':>8s}")
    for pname in ("Винтовочный сферический", "Двухосновный высокоэнергетический",
                  "Трёхосновный (щадящий ствол)", "Охлаждённый оксамидом"):
        c = deepcopy(cart)
        c.charge = Charge(presets.library()[pname], cart.charge.mass)
        sol = charge_for_pressure(c, weapon.barrel, 415e6, metric="mean")
        if sol.charge_mass <= 0:
            continue
        c.charge = Charge(presets.library()[pname], sol.charge_mass)
        r = fire(deepcopy(weapon), c, trajectory=False)
        print(f"{pname[:36]:36s} {c.charge.propellant.flame_temp:6.0f} "
              f"{r.muzzle_velocity:6.0f} {r.wear.peak_surface_temp:10.0f} "
              f"{r.wear.chemical_factor:6.2f} "
              f"{r.wear.throat_wear_per_shot * 1e9:9.2f} {r.barrel_life:8d}")

    print("\nВлияние покрытия канала:")
    for label, lining, thick in (("без покрытия", None, 0.0),
                                 ("хром 25 мкм", CHROME_LINING, 25e-6),
                                 ("хром 50 мкм", CHROME_LINING, 50e-6)):
        w = deepcopy(weapon)
        w.barrel.lining = lining
        w.barrel.lining_thickness = thick
        r = fire(w, cart, trajectory=False)
        print(f"   {label:14s} ресурс {r.barrel_life:7d} выстрелов, "
              f"стоимость ствола {w.barrel.cost:.0f}")

    print("\nСерия из 400 выстрелов очередями (200 выстр/мин):")
    w, c = presets.carbine_556x45()
    series = fire_series(w, c, 400, rate_per_minute=200.0, sample_every=100)
    for i, (v, t, e) in enumerate(zip(series.velocities, series.temperatures,
                                      series.throat_erosion)):
        print(f"   выстрел {i * 100 + 1:4d}: v0 = {v:.0f} м/с, "
              f"ствол {t - 273.15:5.0f} C, износ {e * 1e6:5.1f} мкм")
    print(f"   {w.condition.status(w.barrel)}")
    for note in series.notes:
        print(f"   ! {note}")


# --- 5. Кучность и мастерская -------------------------------------------------

def demo_workshop() -> None:
    rule("5. МАСТЕРСКАЯ: как станок превращается в кучность")
    weapon, cart = presets.rifle_762x51()

    for label, shop in (("гараж", garage_workshop()),
                        ("мастерская", gunsmith_shop()),
                        ("арсенал", arsenal_workshop())):
        tol = shop.tolerances()
        res = monte_carlo(weapon, cart, tol, shots=40, target_range=300.0)
        print(f"\n--- {label} ({shop.smith.name}, навык "
              f"{shop.smith.skill:.2f}) ---")
        print(f"   навеска +-{tol.charge_mass_sd * 1e6:.0f} мг, "
              f"дисбаланс пули {tol.cg_offset * 1e6:.1f} мкм, "
              f"биение {tol.bullet_runout * 1e6:.0f} мкм")
        print(f"   v0 = {res.velocity_mean:.0f} м/с, SD = "
              f"{res.velocity_sd:.1f}, ES = {res.velocity_es:.1f}")
        print(f"   рассеивание на 300 м: {res.group_moa:.2f} МОА")
        for k, v in sorted(res.contributions.items(), key=lambda kv: -kv[1]):
            if k.startswith("множитель"):
                continue
            print(f"      {k:32s} {v:.3f} МОА")
        for wmsg in res.warnings:
            print(f"   ! {wmsg}")

    print("\nПопытка изготовить ствол в гараже:")
    barrel, log = garage_workshop().make_barrel(
        STEEL_4140, 7.62e-3, 0.610, Rifling(4, 0.1015e-3, 0.5, 0.305),
        Chamber(3.7e-6, 51.9e-3, 8.8e-3, 11.5e-3), lining=CHROME_LINING)
    for line in log:
        print(f"   {line}")
    print("\nТо же в арсенале:")
    shop = arsenal_workshop()
    barrel, log = shop.make_barrel(
        STEEL_CRMOV, 7.62e-3, 0.610, Rifling(4, 0.1015e-3, 0.5, 0.305),
        Chamber(3.7e-6, 51.9e-3, 8.8e-3, 11.5e-3), lining=CHROME_LINING,
        autofrettage=0.5)
    for line in log:
        print(f"   {line}")
    if barrel is not None:
        print(f"   трудоёмкость {shop.barrel_hours(barrel):.1f} ч")


# --- 6. Полный цикл: ТЗ -> изделие -------------------------------------------

def demo_full_cycle() -> None:
    rule("6. ПОЛНЫЙ ЦИКЛ: техзадание -> подбор -> отчёт по выстрелу")
    weapon, cart = presets.rifle_762x51()
    req = LoadRequirement(target_range=800.0, min_impact_energy=600.0,
                          min_burst_safety=1.8, min_barrel_life=5000,
                          min_stability=1.4)
    print(f"ТЗ: на {req.target_range:.0f} м донести {req.min_impact_energy:.0f} Дж, "
          f"запас по разрушению >= {req.min_burst_safety}, "
          f"ресурс >= {req.min_barrel_life}, Sg >= {req.min_stability}\n")
    candidates = optimize_load(weapon, cart, presets.library(), req)
    print(f"{'порох':34s} {'навеска':>9s} {'v0':>6s} {'p_кн':>7s} "
          f"{'ресурс':>8s} {'вердикт':>10s}")
    for cand in candidates[:6]:
        verdict = "годится" if cand.acceptable else "не годится"
        print(f"{cand.propellant.name[:34]:34s} "
              f"{cand.charge_mass * 1e3:8.2f}г {cand.report.muzzle_velocity:6.0f} "
              f"{cand.report.peak_pressure_breech / 1e6:6.0f}  "
              f"{cand.report.barrel_life:8d} {verdict:>10s}")
        for fail in cand.failures[:2]:
            print(f"      - {fail}")

    winner = next((c for c in candidates if c.acceptable), candidates[0])
    print(f"\nВыбран: {winner.propellant.name}, навеска "
          f"{winner.charge_mass * 1e3:.2f} г\n")
    final = deepcopy(cart)
    final.charge = Charge(winner.propellant, winner.charge_mass)
    report = fire(deepcopy(weapon), final, target_range=800.0,
                  launch_angle=math.radians(0.55))
    print(report.summary())

    print("\nТраектория:")
    tr = report.trajectory
    if tr is not None:
        print(f"{'дальность':>10s} {'время, с':>9s} {'скорость':>9s} "
              f"{'энергия':>9s} {'падение, м':>11s} {'снос, см':>9s}")
        for rng in (100, 200, 400, 600, 800):
            p = tr.at_range(float(rng))
            if p is None:
                continue
            print(f"{rng:10d} {p.t:9.3f} {p.speed:9.0f} "
                  f"{0.5 * final.projectile.mass * p.speed ** 2:9.0f} "
                  f"{p.y:11.2f} {p.z * 100:9.1f}")

    print("\nТо же в других условиях:")
    for label, env in (
            ("МСА, штиль", Environment()),
            ("мороз -25 C", Environment(
                atmosphere=Atmosphere(sea_level_temperature=248.15),
                powder_temperature=248.15)),
            ("жара +45 C", Environment(
                atmosphere=Atmosphere(sea_level_temperature=318.15),
                powder_temperature=318.15)),
            ("горы 2500 м", Environment(
                atmosphere=Atmosphere(altitude=2500.0))),
            ("ветер 8 м/с справа", Environment(
                wind=Wind(speed=8.0, direction=math.pi / 2)))):
        r = fire(deepcopy(weapon), deepcopy(final), env, target_range=800.0,
                 launch_angle=math.radians(0.55))
        p = r.trajectory.at_range(800.0) if r.trajectory else None
        if p is None:
            continue
        print(f"   {label:20s} v0={r.muzzle_velocity:6.0f} "
              f"p={r.peak_pressure_breech / 1e6:5.0f} МПа  "
              f"на 800 м: падение {p.y:6.2f} м, снос {p.z * 100:6.1f} см, "
              f"скорость {p.speed:.0f} м/с")


DEMOS = [demo_powder, demo_charge, demo_barrel, demo_wear, demo_workshop,
         demo_full_cycle]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEMOS[int(sys.argv[1]) - 1]()
    else:
        for d in DEMOS:
            d()
