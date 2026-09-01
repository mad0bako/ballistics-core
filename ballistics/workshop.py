"""Мастерская: игровой слой поверх физики.

Физика отвечает на вопрос «что будет». Мастерская отвечает на вопрос
«что ты вообще способен изготовить». Она не выдумывает новых законов —
она превращает станочный парк и навык оружейника в те самые допуски,
которые дальше честно прогоняются через Монте-Карло, и в стоимость/время.

Ключевая связка игры:

    станок + навык  ->  допуск  ->  разброс скоростей  ->  кучность
    станок + навык  ->  чистота канала  ->  износ  ->  ресурс ствола
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .barrel import Barrel, Chamber, Rifling
from .grain import DeterrentCoating, GrainDesign
from .materials import ALL_MATERIALS, Metal
from .propellant import Propellant, make_propellant
from .thermochem import INGREDIENTS
from .tolerances import Tolerances


@dataclass
class MachineTool:
    """Станок или приспособление."""

    name: str
    precision: float          # достижимый допуск, м (СКО)
    surface_finish: float     # достижимая шероховатость Ra, м
    speed: float = 1.0        # относительная производительность
    cost: float = 1000.0
    max_length: float = 1.0   # м, наибольшая обрабатываемая длина
    max_diameter: float = 0.1

    def can_make(self, length: float, diameter: float) -> bool:
        return length <= self.max_length and diameter <= self.max_diameter


TOOLS: dict[str, MachineTool] = {
    "Напильник и дрель": MachineTool(
        "Напильник и дрель", precision=120e-6, surface_finish=6.3e-6,
        speed=0.25, cost=150.0, max_length=0.8, max_diameter=0.05),
    "Токарный станок бытовой": MachineTool(
        "Токарный станок бытовой", precision=35e-6, surface_finish=1.6e-6,
        speed=1.0, cost=4000.0, max_length=1.0, max_diameter=0.15),
    "Токарный станок прецизионный": MachineTool(
        "Токарный станок прецизионный", precision=8e-6, surface_finish=0.4e-6,
        speed=1.4, cost=45000.0, max_length=1.5, max_diameter=0.30),
    "Глубокого сверления": MachineTool(
        "Глубокого сверления", precision=12e-6, surface_finish=0.8e-6,
        speed=1.2, cost=90000.0, max_length=2.5, max_diameter=0.20),
    "Дорнирующий пресс": MachineTool(
        "Дорнирующий пресс", precision=5e-6, surface_finish=0.2e-6,
        speed=3.0, cost=250000.0, max_length=1.2, max_diameter=0.06),
    "Хонинговальный": MachineTool(
        "Хонинговальный", precision=4e-6, surface_finish=0.15e-6,
        speed=0.8, cost=30000.0, max_length=1.5, max_diameter=0.20),
    "Гальваническая ванна": MachineTool(
        "Гальваническая ванна", precision=5e-6, surface_finish=0.5e-6,
        speed=0.5, cost=15000.0, max_length=2.0, max_diameter=0.30),
    "Весы аптечные": MachineTool(
        "Весы аптечные", precision=6.5e-6, surface_finish=0.0,
        speed=0.3, cost=800.0),
    "Дозатор объёмный": MachineTool(
        "Дозатор объёмный", precision=45e-6, surface_finish=0.0,
        speed=3.0, cost=200.0),
    "Дозатор весовой автоматический": MachineTool(
        "Дозатор весовой автоматический", precision=13e-6, surface_finish=0.0,
        speed=2.0, cost=6000.0),
}


@dataclass
class Gunsmith:
    """Оружейник: навык переводится в достижимую долю паспортной точности."""

    name: str = "подмастерье"
    skill: float = 0.5          # 0..1
    fatigue: float = 0.0        # 0..1, растёт при спешке

    @property
    def precision_multiplier(self) -> float:
        """Во сколько раз реальный допуск хуже паспортного для станка.

        Новичок на прецизионном станке всё равно делает грубо, мастер на
        бытовом выжимает почти паспорт — но не лучше него.
        """
        base = 3.0 - 2.2 * min(max(self.skill, 0.0), 1.0)
        return base * (1.0 + 0.6 * min(max(self.fatigue, 0.0), 1.0))

    @property
    def speed_multiplier(self) -> float:
        return 0.5 + 0.9 * self.skill


@dataclass
class Workshop:
    """Мастерская: набор станков плюс мастер."""

    tools: dict[str, MachineTool] = field(default_factory=dict)
    smith: Gunsmith = field(default_factory=Gunsmith)

    def has(self, tool: str) -> bool:
        return tool in self.tools

    def best(self, *names: str) -> MachineTool | None:
        avail = [self.tools[n] for n in names if n in self.tools]
        return min(avail, key=lambda t: t.precision) if avail else None

    # --- изготовление ствола -------------------------------------------------
    def make_barrel(self, material: Metal, bore_diameter: float, length: float,
                    rifling: Rifling, chamber: Chamber | None = None, *,
                    lining: Metal | None = None,
                    lining_thickness: float = 25e-6,
                    autofrettage: float = 0.0,
                    outer_diameter: float | None = None
                    ) -> tuple[Barrel | None, list[str]]:
        """Пробует изготовить ствол. Возвращает (ствол, протокол)."""
        log: list[str] = []
        drill = self.best("Глубокого сверления", "Токарный станок прецизионный",
                          "Токарный станок бытовой", "Напильник и дрель")
        if drill is None:
            return None, ["Нечем сверлить канал."]
        if not drill.can_make(length, outer_diameter or bore_diameter * 3.5):
            return None, [f"{drill.name}: ствол длиной {length * 1e3:.0f} мм "
                          "не помещается на станке."]

        rifler = self.best("Дорнирующий пресс", "Токарный станок прецизионный",
                           "Токарный станок бытовой")
        if rifler is None:
            return None, ["Нечем нарезать нарезы."]

        finisher = self.best("Хонинговальный", "Дорнирующий пресс",
                             "Токарный станок прецизионный")
        mult = self.smith.precision_multiplier
        bore_tol = drill.precision * mult
        finish = ((finisher or drill).surface_finish
                  * (0.7 + 0.6 * (1.0 - self.smith.skill)))

        log.append(f"Сверление канала: {drill.name}, допуск "
                   f"{bore_tol * 1e6:.1f} мкм")
        log.append(f"Нарезка: {rifler.name}")
        log.append(f"Финиш канала: Ra = {finish * 1e6:.2f} мкм")

        if material.machinability > 1.2 and self.smith.skill < 0.4:
            log.append(f"{material.name} тяжело обрабатывается — "
                       "поверхность вышла хуже расчётной.")
            finish *= 1.6

        if lining is not None:
            if not self.has("Гальваническая ванна") and lining.name.startswith("Хром"):
                return None, log + ["Хромирование канала требует "
                                    "гальванической ванны."]
            if lining.name.startswith("Хром") and lining_thickness > 60e-6:
                log.append(
                    f"Слой хрома {lining_thickness * 1e6:.0f} мкм технологически "
                    "не держится: толстое покрытие растрескивается и "
                    "отслаивается. Ограничено 60 мкм.")
                lining_thickness = 60e-6

        if autofrettage > 0.0 and not self.has("Дорнирующий пресс"):
            log.append("Автофретирование требует дорнирующего пресса — "
                       "пропущено.")
            autofrettage = 0.0

        from .barrel import BarrelProfile
        profile = None
        if outer_diameter is not None:
            profile = BarrelProfile.tapered(
                length, outer_diameter, outer_diameter * 0.75,
                chamber_length=chamber.length if chamber else 0.0,
                d_chamber=outer_diameter)

        barrel = Barrel(material=material, bore_diameter=bore_diameter,
                        length=length, rifling=rifling, chamber=chamber,
                        profile=profile, lining=lining,
                        lining_thickness=lining_thickness,
                        autofrettage=autofrettage, surface_finish_ra=finish)
        log.append(f"Готово: масса {barrel.mass:.2f} кг, "
                   f"стоимость {self.barrel_cost(barrel, drill):.0f}")
        return barrel, log

    def barrel_cost(self, barrel: Barrel, tool: MachineTool | None = None) -> float:
        tool = tool or self.best(*TOOLS.keys()) or TOOLS["Токарный станок бытовой"]
        base = barrel.cost
        labour = (30.0 * barrel.length / (tool.speed * self.smith.speed_multiplier))
        return base + labour

    def barrel_hours(self, barrel: Barrel) -> float:
        tool = self.best("Дорнирующий пресс", "Глубокого сверления",
                         "Токарный станок прецизионный",
                         "Токарный станок бытовой", "Напильник и дрель")
        speed = (tool.speed if tool else 0.2) * self.smith.speed_multiplier
        return (2.0 + 6.0 * barrel.length) / max(speed, 0.05)

    # --- изготовление боеприпаса ---------------------------------------------
    def mix_powder(self, name: str, composition: dict[str, float],
                   grain_factory, *,
                   coating: DeterrentCoating | None = None
                   ) -> tuple[Propellant | None, list[str]]:
        """Смешивает порох по рецепту, проверяя выполнимость."""
        log: list[str] = []
        hazard = 0.0
        for ing_name, frac in composition.items():
            if ing_name not in INGREDIENTS:
                return None, [f"Неизвестный компонент: {ing_name}"]
            hazard = max(hazard, INGREDIENTS[ing_name].hazard * frac)
        risk = hazard * (1.2 - self.smith.skill)
        if risk > 0.55:
            log.append(
                f"Опасная работа (риск {risk:.2f}): в составе есть компоненты, "
                "которые при таком навыке смешивать нельзя.")
            return None, log
        if risk > 0.30:
            log.append(f"Работа на грани (риск {risk:.2f}): не спешить.")

        prop = make_propellant(name, composition, grain_factory,
                               coating=coating)
        # разброс партии определяется точностью мастерской
        prop.lot_variation = 0.004 * self.smith.precision_multiplier
        log.append(prop.describe())
        return prop, log

    def tolerances(self, *, hand_weighed: bool = True) -> Tolerances:
        """Какие допуски эта мастерская реально выдерживает."""
        mult = self.smith.precision_multiplier
        scale = mult / 1.8         # 1.8 = множитель уверенного среднего мастера

        dispenser = None
        if hand_weighed:
            dispenser = self.best("Весы аптечные",
                                  "Дозатор весовой автоматический",
                                  "Дозатор объёмный")
        else:
            dispenser = self.best("Дозатор объёмный",
                                  "Дозатор весовой автоматический")
        charge_sd = (dispenser.precision if dispenser else 120e-6) * mult

        lathe = self.best("Токарный станок прецизионный",
                          "Дорнирующий пресс", "Токарный станок бытовой",
                          "Напильник и дрель")
        geom = (lathe.precision if lathe else 150e-6) * mult

        return Tolerances(
            charge_mass_sd=charge_sd,
            web_sd_relative=0.006 * scale,
            burn_rate_sd_relative=0.005 * scale,
            bullet_mass_sd_relative=0.0015 * scale,
            case_capacity_sd_relative=0.005 * scale,
            seating_depth_sd=geom * 2.0,
            shot_start_sd_relative=0.04 * scale,
            cg_offset=geom * 0.35,
            bullet_runout=geom * 2.5,
            crown_quality=min(1.0, 0.45 + 0.55 * self.smith.skill),
            bedding_quality=min(1.0, 0.45 + 0.55 * self.smith.skill),
        )


# --- готовые мастерские ------------------------------------------------------

def garage_workshop() -> Workshop:
    return Workshop(
        tools={n: TOOLS[n] for n in ("Напильник и дрель",
                                     "Токарный станок бытовой",
                                     "Дозатор объёмный")},
        smith=Gunsmith("самоучка", skill=0.25))


def gunsmith_shop() -> Workshop:
    return Workshop(
        tools={n: TOOLS[n] for n in ("Токарный станок бытовой",
                                     "Токарный станок прецизионный",
                                     "Хонинговальный", "Весы аптечные",
                                     "Дозатор весовой автоматический")},
        smith=Gunsmith("мастер", skill=0.7))


def arsenal_workshop() -> Workshop:
    return Workshop(
        tools=dict(TOOLS),
        smith=Gunsmith("технолог арсенала", skill=0.95))
