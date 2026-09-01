"""Патрон: гильза, посадка пули, объём каморы, проверки сборки.

Объём каморы W0 — вход в основную задачу внутренней баллистики, и считать
его надо честно:

    W0 = V_гильзы - V_пули_внутри_гильзы + V_свободного_хода

Ошибка в W0 на 5% сдвигает максимальное давление примерно на 10-12% —
поэтому здесь геометрия, а не «примерно столько-то гран воды».
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .grain import GrainDesign
from .interior import Charge, GunSystem, Primer
from .materials import BRASS_CARTRIDGE, Metal
from .projectile import Projectile
from .units import h2o_grains_to_m3, m3_to_h2o_grains


@dataclass
class CaseGeometry:
    """Геометрия гильзы (бутылочной или цилиндрической), метры."""

    base_diameter: float             # диаметр у донца (над проточкой)
    shoulder_diameter: float         # диаметр у ската
    neck_diameter: float             # наружный диаметр дульца
    shoulder_position: float         # расстояние от донца до начала ската
    shoulder_length: float           # длина ската
    neck_length: float               # длина дульца
    case_length: float               # полная длина гильзы
    web_thickness: float = 5.0e-3    # толщина сплошного дна
    wall_thickness_base: float = 1.05e-3
    wall_thickness_neck: float = 0.36e-3
    rim_diameter: float | None = None
    body_taper: float = 0.0          # конусность корпуса (на всю длину), м

    def outer_radius(self, x: float) -> float:
        """Наружный радиус на расстоянии x от донца."""
        if x <= self.shoulder_position:
            frac = x / max(self.shoulder_position, 1e-9)
            d = self.base_diameter - self.body_taper * frac
            if self.shoulder_diameter > 0.0:
                d = min(d, max(self.base_diameter, self.shoulder_diameter))
            return 0.5 * d
        if x <= self.shoulder_position + self.shoulder_length:
            f = (x - self.shoulder_position) / max(self.shoulder_length, 1e-9)
            d0 = self.base_diameter - self.body_taper
            return 0.5 * (d0 + (self.neck_diameter - d0) * f)
        return 0.5 * self.neck_diameter

    def wall_thickness(self, x: float) -> float:
        frac = min(max((x - self.web_thickness)
                       / max(self.case_length - self.web_thickness, 1e-9), 0.0), 1.0)
        return (self.wall_thickness_base
                + (self.wall_thickness_neck - self.wall_thickness_base) * frac)

    def inner_radius(self, x: float) -> float:
        return max(self.outer_radius(x) - self.wall_thickness(x), 0.0)

    def internal_volume(self, n: int = 600) -> float:
        """Внутренний объём гильзы до среза дульца, м^3."""
        x0 = self.web_thickness
        x1 = self.case_length
        dx = (x1 - x0) / n
        v = 0.0
        for i in range(n):
            r = self.inner_radius(x0 + (i + 0.5) * dx)
            v += math.pi * r * r * dx
        return v

    def inner_radius_at_mouth(self) -> float:
        return self.inner_radius(self.case_length - 1e-6)


@dataclass
class Case:
    """Гильза как изделие."""

    name: str
    geometry: CaseGeometry
    material: Metal = field(default_factory=lambda: BRASS_CARTRIDGE)
    capacity_override: float | None = None    # м^3, если известна из замера
    neck_tension: float = 0.05e-3             # м, натяг дульца на пуле
    max_pressure: float = 415e6               # Па, паспортный предел (SAAMI/CIP)
    proof_pressure_factor: float = 1.30

    @property
    def capacity(self) -> float:
        return (self.capacity_override if self.capacity_override is not None
                else self.geometry.internal_volume())

    @property
    def capacity_grains_h2o(self) -> float:
        return m3_to_h2o_grains(self.capacity)

    @property
    def proof_pressure(self) -> float:
        return self.max_pressure * self.proof_pressure_factor

    def bullet_pull_force(self, projectile: Projectile,
                          seated_depth: float) -> float:
        """Усилие извлечения пули из дульца.

        Дульце — тонкостенное кольцо, натянутое на пулю. Натяг neck_tension
        задаётся по диаметру, значит радиальный натяг вдвое меньше.
        Окружное напряжение в кольце sigma = E * delta_r / r, а контактное
        давление на пулю получается из равновесия кольца:

            p_конт = sigma * t / r = E * delta_r * t / r^2

        Ограничение принципиально не в том, что sigma упирается в предел
        текучести: как только кольцо потекло, контактное давление уже не
        растёт и фиксируется на давлении пластического смятия

            p_пред = sigma_т * t / r

        которое в r/t раз (то есть примерно в 10 раз) МЕНЬШЕ самого предела
        текучести. Спутать эти две величины — значит завысить усилие
        распатронивания на порядок и получить давление форсирования 80 МПа
        вместо реальных 30-40.
        """
        g = self.geometry
        r = 0.5 * projectile.geometry.diameter
        t = g.wall_thickness_neck
        delta_r = 0.5 * self.neck_tension
        elastic_p = self.material.youngs_modulus * delta_r * t / (r * r)
        plastic_p = self.material.yield_strength * t / r
        contact_p = min(elastic_p, plastic_p)
        length = min(seated_depth, g.neck_length)
        area = 2.0 * math.pi * r * max(length, 0.0)
        return contact_p * area * 0.22   # коэффициент трения латунь/томпак


@dataclass
class AssemblyCheck:
    ok: bool
    messages: list[str] = field(default_factory=list)
    fill_ratio: float = 0.0
    loading_density: float = 0.0


@dataclass
class Cartridge:
    """Собранный патрон."""

    name: str
    case: Case
    projectile: Projectile
    charge: Charge
    primer: Primer = field(default_factory=Primer)
    coal: float | None = None          # м, длина патрона; None -> подобрать
    seating_depth: float | None = None  # м, глубина посадки пули в гильзу

    def __post_init__(self) -> None:
        if self.seating_depth is None:
            if self.coal is not None:
                # COAL = длина гильзы + выступающая часть пули
                self.seating_depth = (self.projectile.geometry.total_length
                                      - (self.coal - self.case.geometry.case_length))
            else:
                # по умолчанию сажаем пулю на длину дульца
                self.seating_depth = 0.85 * self.case.geometry.neck_length
        if self.coal is None:
            self.coal = (self.case.geometry.case_length
                         + self.projectile.geometry.total_length
                         - self.seating_depth)

    @property
    def seated_volume(self) -> float:
        """Объём части пули, находящейся внутри гильзы."""
        g = self.projectile.geometry
        depth = min(max(self.seating_depth or 0.0, 0.0), g.total_length)
        x_start = g.total_length - depth
        n = 200
        dx = depth / n if n else 0.0
        v = 0.0
        for i in range(n):
            r = g.radius(x_start + (i + 0.5) * dx)
            v += math.pi * r * r * dx
        return v

    def chamber_volume(self, freebore_volume: float = 0.0) -> float:
        """W0 — объём каморы за донцем пули."""
        return max(self.case.capacity - self.seated_volume + freebore_volume,
                   1e-9)

    @property
    def loading_density(self) -> float:
        """Плотность заряжания, кг/м^3."""
        return self.charge.mass / self.chamber_volume()

    @property
    def fill_ratio(self) -> float:
        """Насколько заряд заполняет камору по насыпному объёму."""
        bulk = self.charge.propellant.grain.bulk_density
        return (self.charge.mass / bulk) / self.chamber_volume()

    @property
    def mass(self) -> float:
        g = self.case.geometry
        n = 200
        dx = g.case_length / n
        vol = 0.0
        for i in range(n):
            x = (i + 0.5) * dx
            ro = g.outer_radius(x)
            ri = g.inner_radius(x) if x > g.web_thickness else 0.0
            vol += math.pi * (ro * ro - ri * ri) * dx
        case_mass = vol * self.case.material.density
        return (case_mass + self.projectile.mass + self.charge.mass
                + self.primer.charge_mass)

    @property
    def cost(self) -> float:
        g = self.case.geometry
        case_mass = self.mass - self.projectile.mass - self.charge.mass
        powder_cost = 0.0
        for ing_name, frac in self.charge.propellant.composition.items():
            from .thermochem import INGREDIENTS
            powder_cost += (frac * self.charge.mass
                            * INGREDIENTS[ing_name].cost_per_kg)
        _ = g
        return (case_mass * self.case.material.cost_per_kg * 2.2
                + self.projectile.cost + powder_cost + 0.08)

    def check(self, magazine_length: float | None = None,
              chamber_freebore: float | None = None) -> AssemblyCheck:
        """Проверки сборки, которые обычно и ловят кустарный патрон."""
        msgs: list[str] = []
        ok = True
        g = self.case.geometry
        pg = self.projectile.geometry

        fill = self.fill_ratio
        if fill > 1.0:
            ok = False
            msgs.append(
                f"Заряд не влезает: нужно {100 * fill:.0f}% объёма каморы "
                "по насыпной плотности. Нужен более плотный порох, "
                "прессование или гильза большей ёмкости.")
        elif fill > 0.98:
            msgs.append("Компрессионная посадка: заряд сжимается пулей.")
        elif fill < 0.70:
            msgs.append(
                f"Заряд занимает {100 * fill:.0f}% каморы: в частично "
                "заполненной гильзе воспламенение неровное, разброс скоростей "
                "вырастет, возможны затяжные выстрелы.")

        if (self.seating_depth or 0.0) > g.neck_length + g.shoulder_length:
            msgs.append("Донце пули ушло ниже ската — при выстреле дульце "
                        "может не отпустить пулю равномерно.")
        if (self.seating_depth or 0.0) < 0.5 * pg.diameter:
            ok = False
            msgs.append("Пуля сидит слишком мелко: не держится в дульце.")

        if magazine_length is not None and (self.coal or 0.0) > magazine_length:
            ok = False
            msgs.append(
                f"Длина патрона {1e3 * (self.coal or 0):.2f} мм больше "
                f"магазина ({1e3 * magazine_length:.2f} мм).")

        if chamber_freebore is not None:
            jump = chamber_freebore
            if jump < 0.0:
                ok = False
                msgs.append("Пуля упирается в нарезы: давление подскочит.")

        pull = self.case.bullet_pull_force(self.projectile,
                                           self.seating_depth or 0.0)
        if pull < 150.0:
            msgs.append(f"Усилие распатронивания {pull:.0f} Н — маловато, "
                        "пуля может сдвинуться от отдачи в магазине.")

        return AssemblyCheck(ok=ok, messages=msgs, fill_ratio=fill,
                             loading_density=self.loading_density)

    def describe(self) -> str:
        return (
            f"Патрон {self.name}\n"
            f"  гильза {self.case.name}: ёмкость "
            f"{self.case.capacity_grains_h2o:.1f} гран H2O "
            f"({self.case.capacity * 1e6:.2f} см3)\n"
            f"  камора W0 = {self.chamber_volume() * 1e6:.3f} см3, "
            f"заряд {self.charge.mass * 1e3:.2f} г, плотность заряжания "
            f"{self.loading_density:.0f} кг/м3, заполнение "
            f"{100 * self.fill_ratio:.0f}%\n"
            f"  длина патрона {1e3 * (self.coal or 0):.2f} мм, "
            f"масса {self.mass * 1e3:.1f} г, цена {self.cost:.2f}"
        )


def build_system(cartridge: Cartridge, barrel, *,
                 shot_start_pressure: float | None = None) -> GunSystem:
    """Адаптер: патрон + ствол -> постановка задачи внутренней баллистики."""
    ch = barrel.chamber
    freebore_volume = 0.0
    if ch is not None:
        freebore_volume = 0.25 * math.pi * barrel.groove_diameter ** 2 * ch.freebore
    w0 = cartridge.chamber_volume(freebore_volume)

    # давление форсирования: врезание пояска + натяг дульца
    if shot_start_pressure is None:
        area = barrel.bore_area
        pull = cartridge.case.bullet_pull_force(cartridge.projectile,
                                                cartridge.seating_depth or 0.0)
        engrave = _engraving_pressure(cartridge.projectile, barrel)
        shot_start_pressure = pull / area + 0.35 * engrave

    return GunSystem(
        bore_area=barrel.bore_area,
        chamber_volume=w0,
        travel=barrel.travel,
        projectile_mass=cartridge.projectile.mass,
        bore_diameter=barrel.effective_diameter,
        shot_start_pressure=shot_start_pressure,
        engraving_pressure=_engraving_pressure(cartridge.projectile, barrel),
        engraving_length=max(cartridge.projectile.geometry.bearing_length, 2e-3),
        twist=barrel.rifling.twist,
        wall_temperature=barrel.temperature,
    )


def _engraving_pressure(projectile: Projectile, barrel) -> float:
    """Давление, нужное чтобы продавить оболочку через нарезы.

    Металл оболочки выдавливается в канавки: работа пластической деформации
    на объём вытесняемого металла, отнесённая к пути врезания.
    """
    jacket = projectile.driving_band
    n = barrel.rifling.grooves
    depth = barrel.rifling.groove_depth
    d = barrel.bore_diameter
    groove_width = (1.0 - barrel.rifling.land_ratio) * math.pi * d / max(n, 1)
    displaced_area = n * groove_width * depth
    # напряжение течения ~ 2.5*sigma_т при стеснённой осадке
    flow = 2.5 * jacket.yield_strength
    return flow * displaced_area / barrel.bore_area


# --- готовые гильзы ----------------------------------------------------------

def case_9x19() -> Case:
    geom = CaseGeometry(
        base_diameter=9.93e-3, shoulder_diameter=9.93e-3,
        neck_diameter=9.65e-3, shoulder_position=17.0e-3,
        shoulder_length=0.5e-3, neck_length=1.5e-3, case_length=19.15e-3,
        web_thickness=3.6e-3, wall_thickness_base=0.85e-3,
        wall_thickness_neck=0.30e-3, body_taper=0.20e-3)
    return Case("9x19 Parabellum", geom,
                capacity_override=h2o_grains_to_m3(13.3),
                max_pressure=235e6, neck_tension=0.04e-3)


def case_556x45() -> Case:
    geom = CaseGeometry(
        base_diameter=9.58e-3, shoulder_diameter=9.00e-3,
        neck_diameter=6.43e-3, shoulder_position=35.0e-3,
        shoulder_length=3.0e-3, neck_length=6.0e-3, case_length=44.70e-3,
        web_thickness=4.4e-3, wall_thickness_base=0.95e-3,
        wall_thickness_neck=0.30e-3, body_taper=0.35e-3)
    return Case("5.56x45 NATO", geom,
                capacity_override=h2o_grains_to_m3(28.5),
                max_pressure=430e6, neck_tension=0.04e-3)


def case_762x51() -> Case:
    geom = CaseGeometry(
        base_diameter=11.96e-3, shoulder_diameter=11.20e-3,
        neck_diameter=8.71e-3, shoulder_position=40.7e-3,
        shoulder_length=2.7e-3, neck_length=7.7e-3, case_length=51.18e-3,
        web_thickness=5.0e-3, wall_thickness_base=1.05e-3,
        wall_thickness_neck=0.36e-3, body_taper=0.42e-3)
    return Case("7.62x51 NATO (.308 Win)", geom,
                capacity_override=h2o_grains_to_m3(56.0),
                max_pressure=415e6)


def case_338lm() -> Case:
    geom = CaseGeometry(
        base_diameter=14.93e-3, shoulder_diameter=14.20e-3,
        neck_diameter=9.63e-3, shoulder_position=54.0e-3,
        shoulder_length=4.5e-3, neck_length=10.0e-3, case_length=69.20e-3,
        web_thickness=6.0e-3, wall_thickness_base=1.30e-3,
        wall_thickness_neck=0.43e-3, body_taper=0.40e-3)
    return Case(".338 Lapua Magnum", geom,
                capacity_override=h2o_grains_to_m3(114.0),
                max_pressure=420e6)


def case_152mm() -> Case:
    """Гильзовое заряжание 152-мм гаубицы (переменный заряд)."""
    geom = CaseGeometry(
        base_diameter=167e-3, shoulder_diameter=167e-3,
        neck_diameter=157e-3, shoulder_position=520e-3,
        shoulder_length=20e-3, neck_length=60e-3, case_length=600e-3,
        web_thickness=25e-3, wall_thickness_base=6.0e-3,
        wall_thickness_neck=2.5e-3, body_taper=4e-3)
    return Case("152-мм гильза", geom, capacity_override=13.6e-3,
                max_pressure=320e6, proof_pressure_factor=1.15)


STANDARD_CASES = {
    "9x19": case_9x19,
    "5.56x45": case_556x45,
    "7.62x51": case_762x51,
    ".338LM": case_338lm,
    "152mm": case_152mm,
}
