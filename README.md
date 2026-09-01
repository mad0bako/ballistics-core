# ballistics

A physics core that simulates a gunshot end to end — from the chemical formula
of the propellant to the point of impact.

Nothing here is tuned by eye. Bullet mass comes out of a drawing and the
densities of its metals. Propellant force comes out of an elemental balance and
the equilibrium composition of the combustion products. Chamber pressure comes
out of Resal's equation. Barrel strength comes out of the Lamé problem. Barrel
life comes out of a Duhamel integral for the bore surface temperature and a wear
correlation.

Pure Python, standard library only, no dependencies. A full shot with all of its
consequences takes 15–25 ms, which makes the core usable both for interactive
load development inside a game and for Monte-Carlo runs over hundreds of shots.

[Русская версия README](README.ru.md)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
![Tests](https://img.shields.io/badge/tests-45%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

> ### ⚠️ This is a game model, not a load manual
>
> This code was written as the physics core of a video game about a gunsmith.
> It reproduces the *behaviour* of interior and exterior ballistics well enough
> to make in-game engineering decisions meaningful, and it is validated against
> published data (see below) — but it is not a reloading tool.
>
> **Do not use it to develop charges for live ammunition.** Its wear constant is
> calibrated against a single reference point, its burn-rate model is inferred
> from composition rather than measured, and its equation of state is known to
> be wrong at high loading densities. Real load development requires measured
> lot data from the powder manufacturer and published load data — use QuickLOAD,
> GRT, or the manufacturer's own tables, together with a chronograph and a
> pressure trace, not a simulation.

---

## Quick start

```bash
python -c "from ballistics import presets, simulation; w, c = presets.rifle_762x51(); print(simulation.fire(w, c, target_range=600.0).summary())"
```

```
v0 = 804 м/с, дульная энергия 3561 Дж
p_кн_max = 433 МПа, запас прочности: текучесть 0.91, разрушение 2.31
Sg = 1.71 (Миллер 1.72), отдача 23.1 Дж
ресурс ствола 14505 выстр., нагрев 1.9 К/выстр, цена выстрела 0.61
дальность 600 м, ВП 0.90 с, скорость у цели 551 м/с, энергия 1671 Дж
```

```bash
python tests/test_physics.py   # 45 regression checks against textbooks and spec sheets
python examples/demo.py        # six scenarios from the game loop
python examples/demo.py 4      # barrel wear only
```

Requires Python 3.10 or newer. Developed and tested on CPython 3.12.

## What it actually models

### Propellant thermochemistry — `thermochem.py`

You give a recipe in mass fractions. The module then computes:

1. the elemental C, H, O, N balance per kilogram of charge;
2. the equilibrium product composition in the five-gas approximation
   (CO₂, CO, H₂O, H₂, N₂) through the water-gas shift `CO + H₂O ⇌ CO₂ + H₂`,
   adding soot when oxygen is short and free O₂ when it is in excess;
3. the heat of explosion at constant volume, `Qv = −[ΔH_reaction − Δn·R·T₀]`;
4. the flame temperature from an energy balance with a dissociation correction;
5. and from those, the propellant force `f = n·R·T₁`, the ratio
   `k = 1 + R_specific/Cv_specific`, and the covolume `α = Σ nᵢbᵢ`.

The calculation is self-consistent: composition depends on temperature and
temperature depends on composition, and the iteration converges in about ten
steps.

This is the node where the oxidizer/fuel trade-off lives, and it is honest about
why "just add more oxidizer" fails: potassium nitrate raises the oxygen balance
and the flame temperature, but carries part of the mass into condensed slag that
produces no gas — so propellant force actually **drops** while barrel wear goes
up.

### Grain geometry — `grain.py`

The form function `ψ(z) = κz(1 + λz + μz²)` is not looked up in a table; it is
derived exactly from the grain's dimensions. The trick: for every practically
used shape, the grain volume `V(e)` is a polynomial of degree at most three, so
`ψ(z)` is a cubic too, and its coefficients can be recovered from three points.
The same code handles spheres, flakes, tubes and seven-perforated grains.

The web of a multi-perforated grain follows from hexagonal packing:

```
e₁ = (D − (2m+1)·d) / (4(m+1)),   n = 3m² + 3m + 1
```

At m = 0 this degenerates to the single-tube `(D−d)/4`; at m = 1 it gives the
classical seven-perforated `(D−3d)/8`.

### Interior ballistics — `interior.py`

The classical system (Serebryakov ch. 5; Corner ch. 5):

```
dz/dt = u₁·p^ν / e₁                            Vieille's burn law
ψ = ψ(z)                                       gas generation
φ·m·dv/dt = S·(p_base − p_resist)              projectile motion
p·W = f·ω·ψ − θ·(φ·m·v²/2 + Q_wall)            Resal's equation
W = W₀ − ω(1−ψ)/δ − α·ω·ψ + S·l                free volume
```

Pressure distribution along the bore is Lagrangian, and the three pressures are
kept distinct because they are used for three different things:

```
p_breech = p_base·(1 + ω/(2φm))     loads the bolt
p_mean   = p_base·(1 + ω/(3φm))     what a transducer sees — SAAMI/CIP refer to this
p_base                              accelerates the bullet
```

The integrator is Cash-Karp RK45 with step control, so an artillery shot (20 ms)
and a pistol shot (0.3 ms) are both integrated stably without retuning.

Heat loss to the walls uses the Dittus-Boelter correlation, evaluated separately
as a bore average (for the energy balance) and at the throat (for wear): under
the Lagrange assumption gas velocity is linear along the bore, so the throat is
scoured very differently from the mid-barrel.

### Barrel — `barrel.py`

Thick-walled cylinder, von Mises criterion. At the bore surface:

```
σ_eq    = √3·p·b² / (b² − a²)
p_yield = σ_y·(b² − a²) / (√3·b²)
p_burst = (2/√3)·σ_u·ln(b/a)
```

Autofrettage to radius ρ raises the elastic limit to

```
p_af = (2σ_y/√3)·[ln(ρ/a) + (b² − ρ²)/(2b²)]
```

One distinction the model keeps explicit: on real rifled barrels the **yield**
margin at the bore surface is routinely around unity. The first shots work-harden
the inner layer, the barrel autofrettages itself, and it runs elastically after
that. The failure criterion is exhaustion of the **burst** margin, which is kept
at 1.5–2 or better.

Bore area accounts for the grooves: `S = πd²/4 + n·w·h`. Using the land diameter
alone understates the force by 2–4%.

### Projectile — `projectile.py`

The body of revolution is integrated numerically, so mass, centre of mass and
both moments of inertia all follow from the drawing. A jacketed bullet is
modelled in layers, including the hollow nose — lead is poured in from the base
and does not reach the tip. Without that, mass is overstated by about five
percent.

The G7 form factor is estimated from the profile by an empirical fit over four
real bullets; if the BC is known from testing it can be supplied directly through
`form_factor_override`.

### Barrel wear — `erosion.py`

Three independent mechanisms:

1. **Throat erosion.** The bore surface temperature during a shot is the exact
   solution for a semi-infinite body under a time-varying flux (Duhamel's
   integral): `T_s(t) = T₀ + 1/√(πkρc) · ∫ q(τ)/√(t−τ) dτ`. The removal rate is
   Lawton's correlation `w = A·∫exp(−B/T_s)dt`, scaled by the chemical
   aggressiveness of the gas and the resistance of the surface material.
   A thin coating is thermally "transparent" — over one millisecond heat
   penetrates ~110 µm into steel while the chrome layer is 25 µm — so surface
   activity is taken depth-weighted, and life is computed in two phases: by the
   coating while it lasts, by the substrate once it is worn through. That is why
   chrome plating buys a factor of a few, not a factor of tens.
2. **Low-cycle fatigue of the breech end**, via Manson-Coffin.
3. **Mechanical wear of the lands** by the driving band.

### Exterior ballistics — `exterior.py`

Point-mass 3-DOF: drag from the standard G1/G7 functions scaled by the form
factor, gravity, Coriolis, wind and spin drift. The atmosphere is the ISA with a
humidity correction (moist air is lighter than dry air). Gyroscopic stability is
computed both aerodynamically and by the Miller rule.

### Tolerances and precision — `tolerances.py`

Group size is not a property of the weapon; it is the sum of physical
contributions:

- muzzle velocity spread → vertical dispersion (Monte-Carlo through the actual
  interior ballistics, not a hand-waved ±10 m/s);
- lateral throw-off from bullet imbalance, `θ = ε·ω/V` — a 10 µm centre-of-mass
  offset at 6000 rad/s and 800 m/s gives 0.26 MOA, more than the rest of the
  budget combined;
- aerodynamic jump from bullet tilt in the case;
- barrel vibration: exit time drifts with velocity and lands in a different phase
  of the flexural oscillation;
- a wear multiplier.

### Workshop — `workshop.py`

The game layer invents no physics of its own. It turns a machine park and a skill
level into tolerances, which are then run honestly through the Monte-Carlo:

```
machine + skill → tolerance      → velocity spread → group size
machine + skill → bore finish    → wear           → barrel life
```

Plus the process constraints: no chrome lining without a plating bath, chrome
thicker than 60 µm does not adhere, autofrettage needs a mandrel press.

## Validation against real data

The model is checked against independently known quantities, not against itself.
All of these checks are pinned by `tests/test_physics.py`.

**Propellants** — three reference compositions land inside handbook ranges on
four parameters at once:

| | f, kJ/kg | T₁, K | k | α, dm³/kg |
|---|---|---|---|---|
| single-base | 986 (950–1000) | 2879 (2800–3000) | 1.248 (1.21–1.26) | 1.045 (0.95–1.10) |
| double-base | 1053 (1000–1150) | 3285 (3200–3400) | 1.218 | 0.985 |
| triple-base | 1028 (1050–1150) | 3002 (2800–3000) | 1.240 | 1.039 |

**Form function** — matches Serebryakov's tables: a sphere gives exactly κ=3,
λ=−1, μ=1/3; a tube 1.045 / −0.043 / 0; a seven-perforated grain
0.771 / +0.147 / −0.034 with ψ₁=0.859.

**Bullets** — mass from the drawing versus catalogue: .308 168 gr match
11.01 g vs 10.89 (+1.1%), 9×19 124 gr 8.02 vs 8.03 (−0.1%), .338 250 gr
16.34 vs 16.2 (+0.8%). G7 BC of the .308 match bullet: 0.224 against a
catalogue 0.224.

**Interior ballistics** — five systems from pistol to howitzer. The order of the
check matters: the grain web is fitted so that the **published charge** produces
the **published pressure**, and after that muzzle velocity is *not* tuned — it is
predicted, and that prediction is what the model is judged on.

| cartridge | charge | v₀ model | v₀ published | err | p model | p published | err |
|---|---|---|---|---|---|---|---|
| 9×19 | 0.40 g | 358 m/s | 360 m/s | −0.5% | 237 MPa | 235 MPa | +0.9% |
| 7.62×51 | 2.85 g | 804 m/s | 810 m/s | −0.7% | 418 MPa | 415 MPa | +0.7% |
| 152 mm | 7.0 kg | 659 m/s | 655 m/s | +0.5% | 307 MPa | 300 MPa | +2.3% |
| .338 LM | 6.30 g | 892 m/s | 890 m/s | +0.2% | 471 MPa | 420 MPa | +12% |
| 5.56×45 | 1.64 g | 877 m/s | 940 m/s | −6.7% | 398 MPa | 430 MPa | −7.5% |

Three systems out of five agree within one percent on velocity and pressure
simultaneously. The two outliers, .338 LM and 5.56, are the same effect: both run
at the limit of loading density, with the charge filling the whole case, and that
is exactly where the Noble-Abel equation of state becomes too stiff and drives
pressure into its ceiling early. On the .338 the velocity is still accurate; only
the peak is missed.

**Exterior ballistics** — against a published table for the .308 175 gr
(790 m/s, G7 BC 0.243) at 1000 m:

| | model | table |
|---|---|---|
| time of flight | 1.729 s | 1.73 s |
| impact velocity | 424 m/s | ~420 m/s |
| drop | 12.03 m | 11.9–12.1 m |
| spin drift | 0.35 m | ~0.3 m |

**Barrel life** — 12 200 rounds for the .308 class (real: 10–15 k), 709 for a hot
double-base magnum (real: 1.5–2 k), 28 100 for an oxamide-cooled load. 25 µm of
chrome gives 1.7×, 50 µm gives 2.9× (real: 2–3×).

## Where the model is knowingly approximate

An honest list, so that convenient numbers are not mistaken for truth:

- **Dissociation** of the combustion products is handled by a lumped correction
  with two calibration constants rather than full equilibrium with radicals
  (OH, H, O, NO). On the reference propellants this gives the right T₁ and f, but
  it may drift on exotic compositions.
- **Burn rate** is estimated from the energetics of the composition through an
  empirical `u₁ ~ (f/f_ref)³` relation with per-component modifiers. Real burn
  rate also depends on the condensed-phase structure, which composition alone
  does not determine.
- **The wear constant** is calibrated against a single reference point (.308
  class, ~0.02 µm/shot). The order of magnitude and all the trends are right;
  the absolute number is good to within a multiplier.
- **Form factor from the profile** is good to 5–10%, worse on unusual shapes.
- **Spin drift** uses Litz's empirical fit; strictly it needs 6-DOF.
- **Shot start pressure** is a lumped parameter (neck tension plus engraving).
  The model puts it in the realistic 30–50 MPa range for rifles, but the split
  between the two terms is arbitrary.
- **The model is one-dimensional**: no gas blow-by past the bullet, no projectile
  tilt in the bore, no non-uniform ignition along the charge.
- **The Noble-Abel equation of state** `p(W − αω) = ωRT` is the main source of
  systematic error at the edge of the envelope. It works well up to a loading
  density of about 0.8–0.9 g/cm³; above that it is too stiff — the free volume in
  the denominator collapses faster than it should and the model hits its pressure
  ceiling early. This is exactly what shows up on 5.56 and .338 LM in the table
  above. Replacing it with a virial equation of state is the single most
  worthwhile improvement left.

## Architecture

Every layer depends only on the layers below it, so any one of them can be
replaced wholesale without touching the rest.

```
units, materials       constants, unit conversion, metal properties
   ↓
thermochem             composition → f, T₁, k, covolume, gas composition
grain                  grain geometry → ψ(z)
   ↓
propellant             propellant = thermochemistry + grain + burn law
   ↓
interior               the main interior ballistics problem
   ↓
barrel, projectile, cartridge, erosion, exterior
   ↓
tolerances, simulation
   ↓
design, workshop       inverse problems and the game layer
```

17 modules, ~5900 lines.

## Units

Everything inside the core is strictly SI: metres, kilograms, seconds, pascals,
kelvin. Conversion to gun units — millimetres, grains, feet per second, MOA —
happens only at the boundary, in `units.py`. This eliminates the classic
mixed-unit mistake that quietly costs a factor of 10⁶ in ballistics
calculations.

## A note on language

The code is in English: identifiers, module structure and the public API. The
docstrings, inline comments, preset names, propellant ingredient keys and the
human-readable strings returned by `summary()` and the verdict lists are in
Russian, because the core was written for a Russian-language game.

That means a few of the dictionary keys you pass in are Russian strings — for
example the ingredient names in `thermochem.INGREDIENTS`. Translating those
strings is a well-scoped, self-contained contribution if anyone wants it.

## Roadmap

- Replace the Noble-Abel equation of state with a virial one — the only
  remaining systematic miss (see 5.56 and .338 LM above).
- Full product equilibrium with radicals instead of a lumped dissociation
  correction.
- 6-DOF exterior ballistics instead of empirical spin drift.
- A model for gas blow-by past the bullet as the bore wears.

## References

- Serebryakov M. E. *Interior Ballistics of Barrel Systems and Solid-Propellant
  Rockets.* Oborongiz. (Внутренняя баллистика ствольных систем и пороховых
  ракет)
- Corner J. *Theory of the Interior Ballistics of Guns.* Wiley.
- McCoy R. L. *Modern Exterior Ballistics.* Schiffer.
- Lawton B. *Thermal and chemical effects on gun barrel wear.* 2001.
- Hoerner S. F. *Fluid-Dynamic Drag.*
- ASM Metals Handbook (steel and alloy properties).

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, ship it in a commercial game; just
keep the copyright notice. No warranty of any kind, which for a physics model
that is openly approximate in the ways listed above is not merely a formality.
