"""Материалы: стволные стали, металлы пуль/гильз, покрытия канала.

Данные справочные (ASM Metals Handbook; Lawton B., "Thermal and chemical
effects on gun barrel wear", 2001). Прочность дана при 293 К плюс модель
разупрочнения с температурой:

    sigma_y(T) = sigma_y_20 * softening(T)

Это принципиально: у прогретого ствола предел текучести падает, и запас
прочности на 300-м выстреле очереди совсем не тот, что на первом.
"""
from __future__ import annotations

from dataclasses import dataclass

from .units import KSI, MPA


@dataclass(frozen=True)
class Metal:
    """Конструкционный материал."""

    name: str
    density: float                 # кг/м^3
    yield_strength: float          # Па, предел текучести при 293 К
    ultimate_strength: float       # Па, предел прочности
    youngs_modulus: float          # Па
    poisson: float
    # теплофизика (нужна для прогрева канала и расчёта износа)
    thermal_conductivity: float    # Вт/(м*К)
    specific_heat: float           # Дж/(кг*К)
    melting_point: float           # К
    thermal_expansion: float       # 1/К
    # доля sigma_y, остающаяся при 500 C
    hot_strength_500c: float = 0.75
    # эрозионная стойкость: множитель к скорости уноса (1.0 = ствольная сталь)
    erosion_factor: float = 1.0
    # игровой слой: сложность обработки и цена
    machinability: float = 1.0
    cost_per_kg: float = 5.0

    @property
    def thermal_diffusivity(self) -> float:
        """a = k / (rho * c), м^2/с."""
        return self.thermal_conductivity / (self.density * self.specific_heat)

    @property
    def thermal_effusivity(self) -> float:
        """sqrt(k*rho*c) — определяет температуру поверхности при тепловом ударе."""
        return (self.thermal_conductivity * self.density * self.specific_heat) ** 0.5

    def yield_at(self, temp_k: float) -> float:
        """Кусочно-линейная аппроксимация падения sigma_y с температурой.

        До 100 C — полка; далее линейно к hot_strength_500c при 500 C;
        далее обвал к нулю у 0.85*T_melt.
        """
        t_c = temp_k - 273.15
        if t_c <= 100.0:
            return self.yield_strength
        if t_c <= 500.0:
            frac = 1.0 + (self.hot_strength_500c - 1.0) * (t_c - 100.0) / 400.0
            return self.yield_strength * frac
        t_zero = 0.85 * (self.melting_point - 273.15)
        if t_c >= t_zero:
            return 0.02 * self.yield_strength
        frac = self.hot_strength_500c * (t_zero - t_c) / (t_zero - 500.0)
        return self.yield_strength * max(frac, 0.02)


# --- ствольные стали ---------------------------------------------------------
STEEL_4140 = Metal(
    name="4140 CrMo (аналог 40ХМ), закалка+отпуск 30 HRC",
    density=7850.0, yield_strength=120 * KSI, ultimate_strength=145 * KSI,
    youngs_modulus=205e9, poisson=0.29,
    thermal_conductivity=42.6, specific_heat=473.0, melting_point=1700.0,
    thermal_expansion=12.3e-6, hot_strength_500c=0.72,
    erosion_factor=1.0, machinability=1.0, cost_per_kg=6.0,
)

STEEL_4150 = Metal(
    name="4150 CrMo (пулемётные стволы), 32 HRC",
    density=7850.0, yield_strength=128 * KSI, ultimate_strength=155 * KSI,
    youngs_modulus=205e9, poisson=0.29,
    thermal_conductivity=42.0, specific_heat=473.0, melting_point=1700.0,
    thermal_expansion=12.3e-6, hot_strength_500c=0.76,
    erosion_factor=0.95, machinability=0.9, cost_per_kg=7.5,
)

STEEL_416R = Metal(
    name="416R нержавеющая (матчевые стволы)",
    density=7750.0, yield_strength=112 * KSI, ultimate_strength=138 * KSI,
    youngs_modulus=200e9, poisson=0.28,
    thermal_conductivity=24.9, specific_heat=460.0, melting_point=1720.0,
    thermal_expansion=9.9e-6, hot_strength_500c=0.68,
    erosion_factor=1.15, machinability=1.25, cost_per_kg=12.0,
)

STEEL_CRMOV = Metal(
    name="CrMoV артиллерийская (аналог ОХН3МФА)",
    density=7850.0, yield_strength=150 * KSI, ultimate_strength=175 * KSI,
    youngs_modulus=210e9, poisson=0.30,
    thermal_conductivity=38.0, specific_heat=470.0, melting_point=1710.0,
    thermal_expansion=12.0e-6, hot_strength_500c=0.82,
    erosion_factor=0.85, machinability=0.7, cost_per_kg=18.0,
)

STEEL_MILD = Metal(
    name="Ст3 (кустарный ствол)",
    density=7850.0, yield_strength=250 * MPA, ultimate_strength=400 * MPA,
    youngs_modulus=200e9, poisson=0.29,
    thermal_conductivity=50.0, specific_heat=486.0, melting_point=1750.0,
    thermal_expansion=12.0e-6, hot_strength_500c=0.55,
    erosion_factor=1.8, machinability=1.6, cost_per_kg=1.5,
)

# --- покрытия и футеровка канала --------------------------------------------
CHROME_LINING = Metal(
    name="Хромовое покрытие канала",
    density=7190.0, yield_strength=500 * MPA, ultimate_strength=700 * MPA,
    youngs_modulus=279e9, poisson=0.21,
    thermal_conductivity=93.9, specific_heat=449.0, melting_point=2180.0,
    thermal_expansion=6.2e-6, hot_strength_500c=0.90,
    erosion_factor=0.35, machinability=0.5, cost_per_kg=45.0,
)

STELLITE_21 = Metal(
    name="Стеллит 21 (вставка в казённой части)",
    density=8300.0, yield_strength=485 * MPA, ultimate_strength=725 * MPA,
    youngs_modulus=248e9, poisson=0.30,
    thermal_conductivity=14.8, specific_heat=420.0, melting_point=1613.0,
    thermal_expansion=14.6e-6, hot_strength_500c=0.95,
    erosion_factor=0.22, machinability=0.25, cost_per_kg=180.0,
)

# --- металлы пуль и гильз ----------------------------------------------------
LEAD = Metal(
    name="Свинец сурьмянистый 3% Sb",
    density=11050.0, yield_strength=12 * MPA, ultimate_strength=20 * MPA,
    youngs_modulus=16e9, poisson=0.44,
    thermal_conductivity=35.0, specific_heat=129.0, melting_point=600.0,
    thermal_expansion=29e-6, cost_per_kg=3.0,
)

COPPER = Metal(
    name="Медь М1 (монолитная пуля)",
    density=8960.0, yield_strength=70 * MPA, ultimate_strength=220 * MPA,
    youngs_modulus=117e9, poisson=0.34,
    thermal_conductivity=401.0, specific_heat=385.0, melting_point=1358.0,
    thermal_expansion=16.5e-6, cost_per_kg=12.0,
)

GILDING_METAL = Metal(
    name="Томпак Л90 (оболочка, ведущий поясок)",
    density=8860.0, yield_strength=105 * MPA, ultimate_strength=310 * MPA,
    youngs_modulus=115e9, poisson=0.34,
    thermal_conductivity=180.0, specific_heat=380.0, melting_point=1300.0,
    thermal_expansion=18.4e-6, cost_per_kg=14.0,
)

BRASS_CARTRIDGE = Metal(
    name="Латунь Л70 (гильза)",
    density=8530.0, yield_strength=310 * MPA, ultimate_strength=440 * MPA,
    youngs_modulus=110e9, poisson=0.35,
    thermal_conductivity=120.0, specific_heat=380.0, melting_point=1230.0,
    thermal_expansion=19.9e-6, cost_per_kg=11.0,
)

STEEL_JACKET = Metal(
    name="Сталь плакированная (биметалл)",
    density=7800.0, yield_strength=250 * MPA, ultimate_strength=380 * MPA,
    youngs_modulus=200e9, poisson=0.29,
    thermal_conductivity=52.0, specific_heat=470.0, melting_point=1750.0,
    thermal_expansion=12.0e-6, cost_per_kg=2.5,
)

TUNGSTEN_CARBIDE = Metal(
    name="ВК8 (бронебойный сердечник)",
    density=14900.0, yield_strength=2000 * MPA, ultimate_strength=3400 * MPA,
    youngs_modulus=600e9, poisson=0.22,
    thermal_conductivity=85.0, specific_heat=240.0, melting_point=3140.0,
    thermal_expansion=5.2e-6, machinability=0.1, cost_per_kg=320.0,
)

BARREL_STEELS = {m.name: m for m in (STEEL_4140, STEEL_4150, STEEL_416R,
                                     STEEL_CRMOV, STEEL_MILD)}
LININGS = {m.name: m for m in (CHROME_LINING, STELLITE_21)}
BULLET_METALS = {m.name: m for m in (LEAD, COPPER, GILDING_METAL,
                                     STEEL_JACKET, TUNGSTEN_CARBIDE)}
CASE_METALS = {m.name: m for m in (BRASS_CARTRIDGE, STEEL_JACKET)}

ALL_MATERIALS: dict[str, Metal] = {}
for _group in (BARREL_STEELS, LININGS, BULLET_METALS, CASE_METALS):
    ALL_MATERIALS.update(_group)
