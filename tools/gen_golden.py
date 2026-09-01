"""Генератор эталонных векторов для проверки порта на TypeScript.

Питоновское ядро проверено по учебникам и паспортам. Порт обязан
воспроизводить его число в число — иначе вся эта валидация не переносится
вместе с кодом. Скрипт прогоняет каждый модуль на фиксированном наборе
входов и складывает пары вход-выход в JSON, который затем читает
TypeScript-тест.

    python tools/gen_golden.py            -> game/test/golden.json
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ballistics import presets                                     # noqa: E402
from ballistics.barrel import (Barrel, Chamber, Rifling,            # noqa: E402
                               gyroscopic_twist_miller,
                               miller_stability,
                               required_outer_diameter)
from ballistics.cartridge import (STANDARD_CASES, build_system)     # noqa: E402
from ballistics.erosion import analyse_wear, surface_temperature    # noqa: E402
from ballistics.exterior import (Atmosphere, DragModel,             # noqa: E402
                                 ShotConditions, maximum_range,
                                 solve_trajectory)
from ballistics.grain import (Cord, Flake, MultiPerf, Sphere,       # noqa: E402
                              Tube)
from ballistics.interior import (Charge, InteriorOptions,           # noqa: E402
                                 solve_interior)
from ballistics.materials import (BARREL_STEELS, STEEL_4140,        # noqa: E402
                                  STEEL_CRMOV, STEEL_MILD)
from ballistics.propellant import (COOL_SINGLE_BASE, DOUBLE_BASE,   # noqa: E402
                                   SINGLE_BASE, TRIPLE_BASE,
                                   default_library)
from ballistics.rng import Rng                                      # noqa: E402
from ballistics.simulation import fire                              # noqa: E402
from ballistics.thermochem import compute_thermochemistry           # noqa: E402
from ballistics.tolerances import Tolerances, monte_carlo           # noqa: E402


def r(x: float, digits: int = 9):
    """Округление, чтобы JSON не распухал от мусорных знаков.

    Бесконечность в JSON не представима, поэтому кодируется строкой:
    «толщина стенки бесконечна» — осмысленный ответ задачи (материал
    не держит такое давление ни при какой толщине), и терять его нельзя.
    """
    if x != x:
        return "nan"
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    if x == 0.0:
        return 0.0
    # Двенадцать значащих цифр независимо от запрошенного: меньше нельзя.
    # При девяти относительная погрешность округления сама достигает 1e-9 и
    # съедает весь допуск сравнения — тест начинает ловить не ошибки порта,
    # а собственное округление эталона.
    digits = max(digits, 12)
    return round(x, max(0, digits - int(math.floor(math.log10(abs(x)))) - 1))


def gen_rng() -> dict:
    a = Rng(12345)
    b = Rng(12345)
    c = Rng(1)
    return {
        "seed": 12345,
        "uint32": [a.next_uint32() for _ in range(8)],
        "gauss": [r(b.gauss()) for _ in range(8)],
        "uniform_seed1": [r(c.random()) for _ in range(8)],
        "fork": Rng(12345).fork(3).next_uint32(),
    }


def gen_thermochem() -> list[dict]:
    mixes = {
        "single_base": SINGLE_BASE,
        "double_base": DOUBLE_BASE,
        "triple_base": TRIPLE_BASE,
        "cool_single_base": COOL_SINGLE_BASE,
        "with_oxidizer": {**{k: v * 0.75 for k, v in SINGLE_BASE.items()},
                          "Нитрат калия": 0.25},
        "oxygen_rich": {"Нитроглицерин": 0.60, "Нитрат калия": 0.40},
        "carbon_rich": {"Нитроцеллюлоза 12.0% N (коллоксилин)": 0.70,
                        "Дибутилфталат (флегматизатор)": 0.30},
    }
    out = []
    for name, mix in mixes.items():
        c = compute_thermochemistry(mix)
        out.append({
            "name": name, "mix": mix,
            "out": {
                "force": r(c.force), "flame_temp": r(c.flame_temp),
                "gamma": r(c.gamma), "covolume": r(c.covolume),
                "heat_of_explosion": r(c.heat_of_explosion),
                "gas_moles": r(c.gas_moles), "density": r(c.density),
                "oxygen_balance": r(c.oxygen_balance),
                "oxidizing_ratio": r(c.oxidizing_ratio),
                "gas_constant": r(c.gas_constant),
                "cv_specific": r(c.cv_specific),
                "soot_fraction": r(c.soot_fraction),
                "composition": {k: r(v) for k, v in c.gas_composition.items()},
            },
        })
    return out


def gen_grain() -> list[dict]:
    shapes = [
        ("sphere", Sphere(0.6e-3), {"diameter": 0.6e-3}),
        ("sphere_big", Sphere(1.5e-3), {"diameter": 1.5e-3}),
        ("flake", Flake(0.15e-3, 1.5e-3, 1.5e-3),
         {"thickness": 0.15e-3, "length": 1.5e-3, "width": 1.5e-3}),
        ("cord", Cord(0.82e-3, 1.5e-3),
         {"diameter": 0.82e-3, "length": 1.5e-3}),
        ("tube", Tube(1.2e-3, 0.3e-3, 10e-3),
         {"outer_diameter": 1.2e-3, "inner_diameter": 0.3e-3, "length": 10e-3}),
        ("perf7", MultiPerf(10e-3, 1.0e-3, 12e-3, 7),
         {"outer_diameter": 10e-3, "perf_diameter": 1.0e-3,
          "length": 12e-3, "perforations": 7}),
        ("perf19", MultiPerf(20e-3, 1.2e-3, 25e-3, 19),
         {"outer_diameter": 20e-3, "perf_diameter": 1.2e-3,
          "length": 25e-3, "perforations": 19}),
        ("perf1_is_tube", MultiPerf(10e-3, 1e-3, 12e-3, 1),
         {"outer_diameter": 10e-3, "perf_diameter": 1e-3,
          "length": 12e-3, "perforations": 1}),
    ]
    out = []
    for name, shape, params in shapes:
        f = shape.form_function()
        out.append({
            "name": name, "kind": type(shape).__name__.lower(),
            "params": {k: r(v) if isinstance(v, float) else v
                       for k, v in params.items()},
            "out": {
                "web": r(shape.web()), "volume": r(shape.volume(), 12),
                "surface": r(shape.initial_surface(), 12),
                "kappa": r(f.kappa), "lam": r(f.lam), "mu": r(f.mu),
                "z_k": r(f.z_k), "chi_s": r(f.chi_s), "lam_s": r(f.lam_s),
                "psi_1": r(f.psi_1),
                "psi_samples": [r(f.psi(z / 10.0)) for z in range(0, 16)],
                "dpsi_samples": [r(f.dpsi_dz(z / 10.0)) for z in range(0, 16)],
            },
        })
    return out


def gen_projectile() -> list[dict]:
    items = {
        "b762_168": presets.bullet_762_168(),
        "b556_62": presets.bullet_556_62(),
        "b9mm_124": presets.bullet_9mm_124(),
        "b338_250": presets.bullet_338_250(),
    }
    out = []
    for name, p in items.items():
        g = p.geometry
        out.append({
            "name": name,
            "out": {
                "mass": r(p.mass, 10), "cg": r(p.center_of_gravity, 10),
                "ix": r(p.axial_inertia, 10), "iy": r(p.transverse_inertia, 10),
                "i7": r(p.form_factor_g7), "bc_g7": r(p.bc_g7),
                "sectional_density": r(p.sectional_density),
                "total_length": r(g.total_length), "volume": r(g.volume(), 12),
                "wetted_area": r(g.wetted_area(), 10),
                "radius_samples": [r(g.radius(g.total_length * i / 10.0), 10)
                                   for i in range(11)],
            },
        })
    return out


def gen_cartridge() -> list[dict]:
    out = []
    for key, factory in STANDARD_CASES.items():
        case = factory()
        out.append({
            "name": key,
            "out": {
                "internal_volume": r(case.geometry.internal_volume(), 10),
                "capacity": r(case.capacity, 10),
            },
        })
    return out


def gen_barrel() -> list[dict]:
    out = []
    for name, mat in (("4140", STEEL_4140), ("crmov", STEEL_CRMOV),
                      ("mild", STEEL_MILD)):
        for p in (200e6, 400e6, 600e6):
            out.append({
                "kind": "required_outer_diameter", "material": name,
                "pressure": p, "safety_factor": 1.5,
                "out": r(required_outer_diameter(7.62e-3, p, mat, 1.5)),
            })
        for t in (293.15, 573.15, 873.15):
            out.append({"kind": "yield_at", "material": name,
                        "temperature": t, "out": r(mat.yield_at(t))})
    b = Barrel(material=STEEL_4140, bore_diameter=7.62e-3, length=0.610,
               rifling=Rifling(4, 0.1015e-3, 0.5, 0.305),
               chamber=Chamber(3.7e-6, 51.9e-3, 8.8e-3, 11.5e-3, 1.5e-3))
    st = b.stress_at(0.3, 400e6)
    out.append({
        "kind": "barrel_762", "out": {
            "bore_area": r(b.bore_area, 10), "mass": r(b.mass),
            "travel": r(b.travel), "groove_diameter": r(b.groove_diameter),
            "effective_diameter": r(b.effective_diameter),
            "first_mode": r(b.first_mode_frequency),
            "muzzle_droop": r(b.muzzle_droop, 8),
            "hoop": r(st.hoop_stress), "von_mises": r(st.von_mises),
            "safety": r(st.safety_factor),
            "burst_safety": r(st.burst_safety_factor),
            "elastic_limit": r(b.elastic_limit_pressure(0.3)),
            "burst": r(b.burst_pressure(0.3)),
            "profile": [[r(x), r(b.profile.outer_diameter(x))]
                        for x in (0.0, 0.05, 0.1, 0.3, 0.61)],
        },
    })
    out.append({
        "kind": "miller", "out": {
            "twist_for_sg15": r(gyroscopic_twist_miller(
                7.823e-3, 30.9e-3, 10.9e-3, 800.0, 1.5)),
            "sg_at_305": r(miller_stability(
                7.823e-3, 30.9e-3, 10.9e-3, 0.305, 800.0)),
        },
    })
    return out


def gen_exterior() -> dict:
    atm = Atmosphere()
    humid = Atmosphere(humidity=0.8, sea_level_temperature=303.15)
    high = Atmosphere(altitude=2500.0)
    m, d = 11.34e-3, 7.823e-3
    bc7 = 0.243 * 0.45359237 / 0.0254 ** 2
    drag = DragModel("G7", m / (bc7 * d * d))
    shot = ShotConditions(velocity=790.0, launch_angle=0.0, stability=1.8,
                          spin_drift=False, coriolis=False)
    tr = solve_trajectory(m, d, drag, shot, atm, max_range=1000.0)
    shot_c = ShotConditions(velocity=790.0, launch_angle=math.radians(1.0),
                            azimuth=math.radians(90.0),
                            latitude=math.radians(55.0), stability=1.8,
                            spin_drift=True, coriolis=True)
    tr_c = solve_trajectory(m, d, drag, shot_c, atm, max_range=1000.0)
    mr, angle = maximum_range(m, d, drag, shot, atm)
    machs = [0.3, 0.7, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0]
    return {
        "atmosphere": [
            {"kind": "std", "h": h,
             "density": r(atm.density(h)), "pressure": r(atm.pressure(h)),
             "temperature": r(atm.temperature(h)),
             "sound_speed": r(atm.sound_speed(h))}
            for h in (0.0, 1000.0, 3000.0, 8000.0)
        ] + [
            {"kind": "humid", "h": 0.0, "density": r(humid.density(0.0)),
             "pressure": r(humid.pressure(0.0)),
             "temperature": r(humid.temperature(0.0)),
             "sound_speed": r(humid.sound_speed(0.0))},
            {"kind": "altitude2500", "h": 0.0, "density": r(high.density(0.0)),
             "pressure": r(high.pressure(0.0)),
             "temperature": r(high.temperature(0.0)),
             "sound_speed": r(high.sound_speed(0.0))},
        ],
        "drag": {
            "g1": [[mm, r(DragModel("G1", 1.0).cd(mm))] for mm in machs],
            "g7": [[mm, r(DragModel("G7", 1.0).cd(mm))] for mm in machs],
        },
        "trajectory": {
            "form_factor": r(drag.form_factor), "mass": m, "diameter": d,
            "velocity": 790.0,
            "plain": [
                {"range": rr,
                 "t": r(tr.at_range(float(rr)).t),
                 "y": r(tr.at_range(float(rr)).y),
                 "v": r(tr.at_range(float(rr)).speed)}
                for rr in (100, 300, 500, 800, 1000)
            ],
            "with_coriolis": [
                {"range": rr,
                 "t": r(tr_c.at_range(float(rr)).t),
                 "y": r(tr_c.at_range(float(rr)).y),
                 "z": r(tr_c.at_range(float(rr)).z),
                 "v": r(tr_c.at_range(float(rr)).speed)}
                for rr in (300, 1000)
            ],
            "max_range": r(mr, 6), "max_range_angle": r(angle),
        },
    }


def gen_interior() -> list[dict]:
    out = []
    lib = default_library()
    for name, factory in presets.ALL_PRESETS.items():
        weapon, cart = factory()
        system = build_system(cart, weapon.barrel)
        base = cart.charge.mass
        for scale in (0.85, 1.0, 1.12):
            charge = Charge(cart.charge.propellant, base * scale)
            res = solve_interior(system, charge, cart.primer)
            out.append({
                "preset": name, "charge_scale": scale,
                "charge_mass": r(base * scale, 10),
                "powder": cart.charge.propellant.name,
                "primer": {
                    "charge_mass": r(cart.primer.charge_mass, 12),
                    "force": r(cart.primer.force),
                },
                "system": {
                    "bore_area": r(system.bore_area),
                    "chamber_volume": r(system.chamber_volume),
                    "travel": r(system.travel),
                    "projectile_mass": r(system.projectile_mass),
                    "bore_diameter": r(system.bore_diameter),
                    "shot_start_pressure": r(system.shot_start_pressure),
                    "friction_coefficient": r(system.friction_coefficient),
                    "engraving_length": r(system.engraving_length),
                    "engraving_pressure": r(system.engraving_pressure),
                    "bore_friction_pressure": r(system.bore_friction_pressure),
                    "twist": r(system.twist),
                    "wall_temperature": r(system.wall_temperature),
                },
                "out": {
                    "muzzle_velocity": r(res.muzzle_velocity, 7),
                    "p_max_breech": r(res.p_max_breech, 7),
                    "p_max_mean": r(res.p_max_mean, 7),
                    "p_max_base": r(res.p_max_base, 7),
                    "psi_muzzle": r(res.psi_muzzle, 7),
                    "time_muzzle": r(res.time_muzzle, 7),
                    "x_at_pmax": r(res.x_at_pmax),
                    "t_at_pmax": r(res.t_at_pmax),
                    "spin_rate": r(res.spin_rate, 7),
                    "heat_to_barrel": r(res.heat_to_barrel, 7),
                    "thermal_efficiency": r(res.thermal_efficiency, 7),
                    "recoil_impulse": r(res.recoil_impulse, 7),
                    "stuck": res.stuck,
                },
            })
    # затяжной выстрел
    weapon, cart = presets.rifle_762x51()
    system = build_system(cart, weapon.barrel)
    res = solve_interior(system, Charge(cart.charge.propellant, 0.05e-3),
                         cart.primer)
    out.append({
        "preset": "7.62x51_squib", "charge_scale": 0.0175,
        "charge_mass": 0.05e-3,
        "powder": cart.charge.propellant.name,
        "primer": {"charge_mass": r(cart.primer.charge_mass, 12),
                   "force": r(cart.primer.force)},
        "system": {
            "bore_area": r(system.bore_area),
            "chamber_volume": r(system.chamber_volume),
            "travel": r(system.travel),
            "projectile_mass": r(system.projectile_mass),
            "bore_diameter": r(system.bore_diameter),
            "shot_start_pressure": r(system.shot_start_pressure),
            "friction_coefficient": r(system.friction_coefficient),
            "engraving_length": r(system.engraving_length),
            "engraving_pressure": r(system.engraving_pressure),
            "bore_friction_pressure": r(system.bore_friction_pressure),
            "twist": r(system.twist),
            "wall_temperature": r(system.wall_temperature),
        },
        "out": {"stuck": res.stuck, "muzzle_velocity": r(res.muzzle_velocity),
                "stuck_travel": r(res.stuck_travel)}})
    _ = lib
    return out


def gen_interior_converged() -> list[dict]:
    """Те же выстрелы, но с ужатой точностью интегратора.

    На рабочем rtol = 1e-6 адаптивный шаг сам по себе даёт погрешность
    порядка 1e-6, и сравнивать порт с оригиналом жёстче этого уровня
    бессмысленно. Зато в пределе rtol -> 0 обе реализации обязаны сойтись
    к одному числу — вот это и есть настоящая проверка переноса.
    """
    out = []
    opts = InteriorOptions(rtol=1e-10)
    for name, factory in presets.ALL_PRESETS.items():
        weapon, cart = factory()
        system = build_system(cart, weapon.barrel)
        res = solve_interior(system, cart.charge, cart.primer, opts)
        out.append({
            "preset": name, "rtol": 1e-10,
            "powder": cart.charge.propellant.name,
            "charge_mass": r(cart.charge.mass),
            "primer": {"charge_mass": r(cart.primer.charge_mass, 12),
                       "force": r(cart.primer.force)},
            "system": {
                "bore_area": r(system.bore_area),
                "chamber_volume": r(system.chamber_volume),
                "travel": r(system.travel),
                "projectile_mass": r(system.projectile_mass),
                "bore_diameter": r(system.bore_diameter),
                "shot_start_pressure": r(system.shot_start_pressure),
                "friction_coefficient": r(system.friction_coefficient),
                "engraving_length": r(system.engraving_length),
                "engraving_pressure": r(system.engraving_pressure),
                "bore_friction_pressure": r(system.bore_friction_pressure),
                "twist": r(system.twist),
                "wall_temperature": r(system.wall_temperature),
            },
            "out": {
                "muzzle_velocity": r(res.muzzle_velocity),
                "p_max_breech": r(res.p_max_breech),
                "psi_muzzle": r(res.psi_muzzle),
                "time_muzzle": r(res.time_muzzle),
            },
        })
    return out


def gen_erosion() -> list[dict]:
    out = []
    for name, factory in presets.ALL_PRESETS.items():
        weapon, cart = factory()
        system = build_system(cart, weapon.barrel)
        res = solve_interior(system, cart.charge, cart.primer)
        chem = cart.charge.propellant.chem
        th = surface_temperature(res, weapon.barrel.material)
        w = analyse_wear(weapon.barrel, res, chem.oxidizing_ratio,
                         chem.flame_temp)
        out.append({
            "preset": name,
            "out": {
                "peak_surface_temp": r(th.peak_surface_temp, 7),
                "heat_input": r(th.heat_input, 7),
                "throat_wear": r(w.throat_wear_per_shot, 7),
                "chemical_factor": r(w.chemical_factor, 7),
                "barrel_life": w.barrel_life,
                "fatigue_life": w.fatigue_life,
                "barrel_temp_rise": r(w.barrel_temp_rise, 7),
            },
        })
    return out


def gen_tolerances() -> list[dict]:
    out = []
    weapon, cart = presets.rifle_762x51()
    for label, tol in (("match", Tolerances.factory_match()),
                       ("military", Tolerances.military()),
                       ("garage", Tolerances.garage())):
        res = monte_carlo(weapon, cart, tol, shots=25, target_range=300.0,
                          seed=12345)
        out.append({
            "tolerance": label, "shots": 25, "seed": 12345,
            "out": {
                "velocity_mean": r(res.velocity_mean, 7),
                "velocity_sd": r(res.velocity_sd, 7),
                "velocity_es": r(res.velocity_es, 7),
                "pressure_mean": r(res.pressure_mean, 7),
                "pressure_p99": r(res.pressure_p99, 7),
                "group_moa": r(res.group_moa, 7),
                "vertical_moa": r(res.vertical_moa, 7),
                "horizontal_moa": r(res.horizontal_moa, 7),
                "contributions": {k: r(v, 7)
                                  for k, v in res.contributions.items()},
            },
        })
    return out


def gen_presets() -> list[dict]:
    out = []
    for name, factory in presets.ALL_PRESETS.items():
        weapon, cart = factory()
        rep = fire(weapon, cart, trajectory=False)
        out.append({
            "preset": name,
            "out": {
                "muzzle_velocity": r(rep.muzzle_velocity, 7),
                "muzzle_energy": r(rep.muzzle_energy, 7),
                "p_max_breech": r(rep.peak_pressure_breech, 7),
                "p_max_mean": r(rep.interior.p_max_mean, 7),
                "stability": r(rep.stability, 7),
                "stability_miller": r(rep.stability_miller, 7),
                "min_safety_yield": r(rep.min_safety_yield, 7),
                "min_safety_burst": r(rep.min_safety_burst, 7),
                "recoil_energy": r(rep.recoil_energy, 7),
                "barrel_life": rep.barrel_life,
                "verdict_count": len(rep.verdicts),
                "safe": rep.safe,
            },
        })
    return out


def main() -> None:
    data = {
        "generated_by": "ballistics core (Python), tools/gen_golden.py",
        "note": "Порт на TypeScript обязан воспроизводить эти числа.",
        "rng": gen_rng(),
        "thermochem": gen_thermochem(),
        "grain": gen_grain(),
        "projectile": gen_projectile(),
        "cartridge": gen_cartridge(),
        "barrel": gen_barrel(),
        "exterior": gen_exterior(),
        "interior": gen_interior(),
        "interior_converged": gen_interior_converged(),
        "erosion": gen_erosion(),
        "tolerances": gen_tolerances(),
        "presets": gen_presets(),
    }
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(os.path.dirname(root), "game", "test")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "golden.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    size = os.path.getsize(path)
    counts = {k: (len(v) if isinstance(v, list) else 1)
              for k, v in data.items() if k not in ("generated_by", "note")}
    print(f"записано {path} ({size / 1024:.0f} КБ)")
    for k, v in counts.items():
        print(f"   {k:12s} {v} векторов")


if __name__ == "__main__":
    main()
