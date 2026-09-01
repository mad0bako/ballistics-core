"""Термохимия порохового заряда: состав -> сила пороха, температура, k, коволюм.

Это тот узел, который превращает «намешал компонентов» в конкретные числа
внутренней баллистики. Схема расчёта классическая (Серебряков, гл. 2;
Corner J. "Theory of the Interior Ballistics of Guns", гл. 3):

1. Из брутто-состава считаем элементный баланс C, H, O, N на 1 кг заряда.
2. Находим равновесный состав продуктов в приближении «пяти газов»
   (CO2, CO, H2O, H2, N2) по реакции водяного газа
        CO + H2O <-> CO2 + H2,  Kp(T) = (CO2*H2)/(CO*H2O)
   При сильном недостатке кислорода добавляется сажа C(тв),
   при избытке — свободный O2.
3. Теплота взрывчатого превращения при постоянном объёме
        Qv = -[ dH_реакции - dn_газ * R * T0 ]
4. Температура горения T1 — из условия, что Qv целиком идёт на нагрев
   продуктов от 298 К до T1 при V = const (плюс поправка на диссоциацию).
5. Отсюда:
        f     = n_газ * R * T1        (сила пороха, Дж/кг)
        k     = 1 + R_уд / Cv_уд      (показатель адиабаты)
        alpha = SUM n_i * b_i         (коволюм, м^3/кг)

Диссоциация (CO2 -> CO + O, H2O -> OH + H, ...) учтена сосредоточенной
поправкой: выше DISSOC_ONSET часть энергии уходит в развал молекул. Без неё
модель завышает T1 на 350-600 К. Коэффициент откалиброван по двум реперным
порохам, и на них модель даёт:

    одноосновный (96% НЦ 13.25% N + флегматизатор):
        f = 986 кДж/кг, T1 = 2879 К, k = 1.248, alpha = 1.045 дм3/кг,
        Qv = 3493 кДж/кг, n = 41.2 моль/кг
    двухосновный (60/36 НЦ/НГ):
        f = 1053 кДж/кг, T1 = 3285 К, k = 1.218, alpha = 0.985 дм3/кг
    трёхосновный (с нитрогуанидином):
        f = 1028 кДж/кг, T1 = 3002 К, k = 1.240

Справочные значения для этих типов: f = 0.95...1.20 МДж/кг, T1 = 2800...3400 К,
k = 1.21...1.26, alpha = 0.95...1.10 дм3/кг. Модель попадает в вилку по всем
четырём параметрам одновременно, что и требуется для дальнейшего расчёта.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .units import R_UNIVERSAL

T0 = 298.15  # К, стандартная температура

# --- термодинамика газообразных продуктов -----------------------------------
# Cv(T) = a + b*T, Дж/(моль*К) — линейная аппроксимация JANAF на 1500...3600 К.
# hf — стандартная энтальпия образования, Дж/моль.
# b_cov — мольный коволюм, м^3/моль (ван-дер-ваальсова оценка).


@dataclass(frozen=True)
class GasSpecies:
    name: str
    molar_mass: float   # кг/моль
    hf: float           # Дж/моль
    cv_a: float
    cv_b: float
    b_cov: float

    def cv(self, t: float) -> float:
        return self.cv_a + self.cv_b * t

    def sensible_energy(self, t: float) -> float:
        """Интеграл Cv dT от 298 К до T, Дж/моль."""
        return self.cv_a * (t - T0) + 0.5 * self.cv_b * (t * t - T0 * T0)


CO2 = GasSpecies("CO2", 0.044010, -393510.0, 25.70, 9.40e-3, 4.27e-5)
CO_ = GasSpecies("CO", 0.028010, -110530.0, 19.70, 2.60e-3, 3.95e-5)
H2O = GasSpecies("H2O", 0.018015, -241820.0, 22.86, 8.19e-3, 3.05e-5)
H2_ = GasSpecies("H2", 0.002016, 0.0, 19.80, 2.10e-3, 2.66e-5)
N2_ = GasSpecies("N2", 0.028014, 0.0, 19.50, 2.50e-3, 3.86e-5)
O2_ = GasSpecies("O2", 0.031999, 0.0, 21.00, 3.00e-3, 3.19e-5)
C_S = GasSpecies("C(s)", 0.012011, 0.0, 16.00, 2.00e-3, 0.0)  # сажа: газа не даёт

GAS_ORDER = (CO2, CO_, H2O, H2_, N2_, O2_)

# Калибровочные константы модели
COVOLUME_CAL = 0.70          # коэффициент к сумме мольных коволюмов
DISSOC_ONSET = 1800.0        # К, порог заметной диссоциации
DISSOC_COEF = 0.011          # Дж/(моль*К^2)
T_MAX = 4600.0               # физический потолок для итераций


def water_gas_kp(t: float) -> float:
    """Константа равновесия реакции водяного газа.

    ln Kp = 4229/T - 3.864 — подгонка к JANAF на 1000...3000 К
    (Kp безразмерна, реакция идёт без изменения числа молей).
    """
    return math.exp(4229.0 / t - 3.864)


def dissociation_sink(t: float) -> float:
    """Энергия, «съедаемая» диссоциацией, Дж на моль газа."""
    if t <= DISSOC_ONSET:
        return 0.0
    return DISSOC_COEF * (t - DISSOC_ONSET) ** 2


# --- компоненты заряда -------------------------------------------------------


@dataclass(frozen=True)
class Residue:
    """Конденсированный остаток минерального окислителя (шлак)."""

    name: str
    molar_mass: float   # кг/моль
    hf: float           # Дж/моль
    cv: float           # Дж/(моль*К), считаем постоянной


K2O = Residue("K2O", 0.09420, -363200.0, 85.0)
BAO = Residue("BaO", 0.15330, -548000.0, 47.0)
PBO = Residue("PbO", 0.22320, -219000.0, 46.0)


@dataclass(frozen=True)
class Ingredient:
    """Компонент порохового состава.

    c, h, n, o — число атомов на моль вещества, уходящих в ГАЗОВЫЙ баланс.
    Для минеральных окислителей часть кислорода связана в шлаке и в газовый
    баланс не попадает: она вычтена из o и учтена в residue/residue_moles.
    """

    name: str
    c: float
    h: float
    n: float
    o: float
    molar_mass: float          # кг/моль
    hf: float                  # Дж/моль, энтальпия образования
    density: float             # кг/м^3
    role: str = "energetic"    # energetic | oxidizer | binder | coolant |
    #                            plasticizer | stabilizer | deterrent | inert
    residue: Residue | None = None
    residue_moles: float = 0.0
    # модификаторы скорости горения (мультипликативные, по массовой доле)
    burn_rate_mod: float = 1.0
    # модификатор показателя nu в законе Вьеля
    exponent_mod: float = 0.0
    cost_per_kg: float = 10.0
    hazard: float = 0.3        # игровой слой: опасность обращения 0..1

    @property
    def oxygen_balance(self) -> float:
        """Кислородный баланс, % масс. (по CO2 и H2O)."""
        excess_o = self.o - 2.0 * self.c - 0.5 * self.h
        return 100.0 * excess_o * 0.015999 / self.molar_mass


# Справочник компонентов. Энтальпии образования — сводки по
# Meyer R. "Explosives" и Военно-техническому справочнику по порохам.
INGREDIENTS: dict[str, Ingredient] = {}


def _reg(ing: Ingredient) -> Ingredient:
    INGREDIENTS[ing.name] = ing
    return ing


NC_1325 = _reg(Ingredient(
    name="Нитроцеллюлоза 13.25% N (пироксилин №1)",
    c=6.0, h=7.30, n=2.70, o=10.40, molar_mass=0.28363, hf=-680600.0,
    density=1660.0, role="energetic", burn_rate_mod=1.0,
    cost_per_kg=25.0, hazard=0.6,
))

NC_1200 = _reg(Ingredient(
    name="Нитроцеллюлоза 12.0% N (коллоксилин)",
    c=6.0, h=7.70, n=2.30, o=9.60, molar_mass=0.26565, hf=-677400.0,
    density=1650.0, role="energetic", burn_rate_mod=0.86,
    cost_per_kg=22.0, hazard=0.5,
))

NG = _reg(Ingredient(
    name="Нитроглицерин",
    c=3.0, h=5.0, n=3.0, o=9.0, molar_mass=0.22709, hf=-370900.0,
    density=1600.0, role="energetic", burn_rate_mod=1.65, exponent_mod=0.10,
    cost_per_kg=40.0, hazard=0.95,
))

DEGDN = _reg(Ingredient(
    name="ДЭГДН (диэтиленгликольдинитрат)",
    c=4.0, h=8.0, n=2.0, o=7.0, molar_mass=0.19612, hf=-414000.0,
    density=1380.0, role="energetic", burn_rate_mod=1.30, exponent_mod=0.05,
    cost_per_kg=45.0, hazard=0.7,
))

RDX = _reg(Ingredient(
    name="Гексоген (RDX)",
    c=3.0, h=6.0, n=6.0, o=6.0, molar_mass=0.22212, hf=70700.0,
    density=1820.0, role="energetic", burn_rate_mod=1.9, exponent_mod=0.12,
    cost_per_kg=90.0, hazard=1.0,
))

NQ = _reg(Ingredient(
    name="Нитрогуанидин (охладитель)",
    c=1.0, h=4.0, n=4.0, o=2.0, molar_mass=0.10407, hf=-92500.0,
    density=1710.0, role="coolant", burn_rate_mod=0.75,
    cost_per_kg=35.0, hazard=0.4,
))

OXAMIDE = _reg(Ingredient(
    name="Оксамид (охладитель)",
    c=2.0, h=4.0, n=2.0, o=2.0, molar_mass=0.08807, hf=-505000.0,
    density=1670.0, role="coolant", burn_rate_mod=0.55,
    cost_per_kg=30.0, hazard=0.1,
))

KNO3 = _reg(Ingredient(
    name="Нитрат калия",
    # KNO3 -> 0.5 K2O(тв) + 0.5 N2 + 1.25 O2: в газ уходит N=1, O=2.5
    c=0.0, h=0.0, n=1.0, o=2.50, molar_mass=0.10110, hf=-494600.0,
    density=2110.0, role="oxidizer", residue=K2O, residue_moles=0.5,
    burn_rate_mod=1.10, cost_per_kg=4.0, hazard=0.2,
))

BA_NO3_2 = _reg(Ingredient(
    name="Нитрат бария",
    c=0.0, h=0.0, n=2.0, o=5.0, molar_mass=0.26134, hf=-992000.0,
    density=3240.0, role="oxidizer", residue=BAO, residue_moles=1.0,
    burn_rate_mod=1.05, cost_per_kg=8.0, hazard=0.3,
))

DBP = _reg(Ingredient(
    name="Дибутилфталат (флегматизатор)",
    c=16.0, h=22.0, n=0.0, o=4.0, molar_mass=0.27834, hf=-843900.0,
    density=1045.0, role="deterrent", burn_rate_mod=0.45,
    exponent_mod=-0.04, cost_per_kg=15.0, hazard=0.05,
))

CENTRALITE = _reg(Ingredient(
    name="Централит-2 (стабилизатор)",
    c=17.0, h=20.0, n=2.0, o=1.0, molar_mass=0.26836, hf=-135000.0,
    density=1120.0, role="stabilizer", burn_rate_mod=0.60,
    exponent_mod=-0.02, cost_per_kg=28.0, hazard=0.05,
))

DPA = _reg(Ingredient(
    name="Дифениламин (стабилизатор)",
    c=12.0, h=11.0, n=1.0, o=0.0, molar_mass=0.16922, hf=130200.0,
    density=1160.0, role="stabilizer", burn_rate_mod=0.70,
    cost_per_kg=20.0, hazard=0.1,
))

CAMPHOR = _reg(Ingredient(
    name="Камфора (пластификатор)",
    c=10.0, h=16.0, n=0.0, o=1.0, molar_mass=0.15223, hf=-320000.0,
    density=990.0, role="plasticizer", burn_rate_mod=0.65,
    cost_per_kg=18.0, hazard=0.05,
))

GRAPHITE = _reg(Ingredient(
    name="Графит (антистатик, полировка)",
    c=1.0, h=0.0, n=0.0, o=0.0, molar_mass=0.012011, hf=0.0,
    density=2200.0, role="inert", burn_rate_mod=0.9,
    cost_per_kg=6.0, hazard=0.0,
))

K2SO4_FLASH = _reg(Ingredient(
    name="Сульфат калия (пламегаситель)",
    # K2SO4 считаем инертной солью: в газ ничего не отдаёт
    c=0.0, h=0.0, n=0.0, o=0.0, molar_mass=0.17426, hf=-1437700.0,
    density=2660.0, role="inert", residue=Residue("K2SO4", 0.17426, -1437700.0, 130.0),
    residue_moles=1.0, burn_rate_mod=0.9, cost_per_kg=5.0, hazard=0.0,
))


# --- результат расчёта -------------------------------------------------------


@dataclass
class Thermochemistry:
    """Термохимические свойства заряда (всё на 1 кг пороха)."""

    force: float                   # f, Дж/кг — сила пороха
    flame_temp: float              # T1, К
    covolume: float                # alpha, м^3/кг
    gamma: float                   # k = Cp/Cv
    heat_of_explosion: float       # Qv, Дж/кг
    gas_moles: float               # n, моль/кг
    gas_composition: dict[str, float] = field(default_factory=dict)  # моль/кг
    condensed: dict[str, float] = field(default_factory=dict)        # моль/кг
    oxygen_balance: float = 0.0    # % масс.
    mean_molar_mass: float = 0.0   # кг/моль
    density: float = 1600.0        # кг/м^3, плотность спрессованного пороха
    gas_constant: float = 0.0      # R_уд, Дж/(кг*К)
    cv_specific: float = 0.0       # Дж/(кг*К)
    soot_fraction: float = 0.0     # доля углерода, ушедшего в сажу
    warnings: list[str] = field(default_factory=list)

    @property
    def theta(self) -> float:
        """theta = k - 1 — рабочий параметр уравнения Резаля."""
        return self.gamma - 1.0

    @property
    def oxidizing_ratio(self) -> float:
        """(CO2 + H2O + 2*O2) / (CO + H2) — окислительность пороховых газов.

        Прямо управляет химической составляющей износа канала: окислительные
        газы жгут сталь, восстановительные — науглероживают поверхностный слой.
        """
        ox = (self.gas_composition.get("CO2", 0.0)
              + self.gas_composition.get("H2O", 0.0)
              + 2.0 * self.gas_composition.get("O2", 0.0))
        red = (self.gas_composition.get("CO", 0.0)
               + self.gas_composition.get("H2", 0.0))
        return ox / max(red, 1e-9)


# --- ядро расчёта ------------------------------------------------------------


def _equilibrium(c: float, h: float, o: float, n: float, t: float
                 ) -> tuple[dict[str, float], float]:
    """Равновесный состав продуктов при температуре t.

    Возвращает (состав моль/кг, число молей сажи).
    """
    comp = {"CO2": 0.0, "CO": 0.0, "H2O": 0.0, "H2": 0.0,
            "N2": 0.5 * n, "O2": 0.0}
    o_needed_full = 2.0 * c + 0.5 * h

    if o >= o_needed_full:
        # избыток кислорода: полное окисление плюс свободный O2
        comp["CO2"] = c
        comp["H2O"] = 0.5 * h
        comp["O2"] = 0.5 * (o - o_needed_full)
        return comp, 0.0

    if o <= c:
        # кислорода не хватает даже на весь CO — выпадает сажа,
        # весь водород остаётся в виде H2
        comp["CO"] = o
        comp["H2"] = 0.5 * h
        return comp, c - o

    # основной случай: решаем равновесие водяного газа по x = n(CO2)
    kp = water_gas_kp(t)
    lo = max(0.0, o - c - 0.5 * h)
    hi = min(c, o - c)
    if hi <= lo:
        x = max(lo, 0.0)
    else:
        def resid(x: float) -> float:
            n_co = c - x
            n_h2o = o - c - x
            n_h2 = 0.5 * h - n_h2o
            return x * n_h2 - kp * n_co * n_h2o

        a, b = lo, hi
        fa = resid(a)
        fb = resid(b)
        if fa > 0.0:
            x = a
        elif fb < 0.0:
            x = b
        else:
            for _ in range(80):
                x = 0.5 * (a + b)
                if resid(x) > 0.0:
                    b = x
                else:
                    a = x
            x = 0.5 * (a + b)

    comp["CO2"] = x
    comp["CO"] = c - x
    comp["H2O"] = o - c - x
    comp["H2"] = 0.5 * h - (o - c - x)
    for key in comp:
        if comp[key] < 0.0:
            comp[key] = 0.0
    return comp, 0.0


def compute_thermochemistry(mix: dict[str, float] | dict[Ingredient, float],
                            *, dissociation: bool = True) -> Thermochemistry:
    """Считает термохимию заряда по массовым долям компонентов.

    mix: {имя_или_Ingredient: массовая доля}. Доли нормируются автоматически.
    """
    items: list[tuple[Ingredient, float]] = []
    for key, frac in mix.items():
        ing = key if isinstance(key, Ingredient) else INGREDIENTS[key]
        if frac > 0.0:
            items.append((ing, float(frac)))
    if not items:
        raise ValueError("пустой состав заряда")

    total = sum(f for _, f in items)
    items = [(ing, f / total) for ing, f in items]

    # элементный баланс на 1 кг
    c = h = o = n = 0.0
    hf_react = 0.0            # Дж/кг
    residues: dict[str, float] = {}
    residue_hf = 0.0
    residue_cv = 0.0
    vol = 0.0                 # м^3/кг, для плотности смеси
    warnings: list[str] = []

    for ing, frac in items:
        moles = frac / ing.molar_mass       # моль/кг смеси
        c += moles * ing.c
        h += moles * ing.h
        o += moles * ing.o
        n += moles * ing.n
        hf_react += moles * ing.hf
        vol += frac / ing.density
        if ing.residue is not None and ing.residue_moles > 0.0:
            rm = moles * ing.residue_moles
            residues[ing.residue.name] = residues.get(ing.residue.name, 0.0) + rm
            residue_hf += rm * ing.residue.hf
            residue_cv += rm * ing.residue.cv

    density = 1.0 / vol if vol > 0.0 else 1600.0
    ob = 100.0 * (o - 2.0 * c - 0.5 * h) * 0.015999

    # самосогласованная итерация: состав зависит от T, T зависит от состава
    t = 3000.0
    comp: dict[str, float] = {}
    soot = 0.0
    qv = 0.0
    for _ in range(60):
        comp, soot = _equilibrium(c, h, o, n, t)

        hf_prod = (comp["CO2"] * CO2.hf + comp["CO"] * CO_.hf
                   + comp["H2O"] * H2O.hf + soot * C_S.hf) + residue_hf
        n_gas = sum(comp.values())
        dh = hf_prod - hf_react
        du = dh - n_gas * R_UNIVERSAL * T0     # переход к V = const
        qv = -du

        # температура из энергетического баланса
        def energy(tt: float) -> float:
            e = (comp["CO2"] * CO2.sensible_energy(tt)
                 + comp["CO"] * CO_.sensible_energy(tt)
                 + comp["H2O"] * H2O.sensible_energy(tt)
                 + comp["H2"] * H2_.sensible_energy(tt)
                 + comp["N2"] * N2_.sensible_energy(tt)
                 + comp["O2"] * O2_.sensible_energy(tt)
                 + soot * C_S.sensible_energy(tt)
                 + residue_cv * (tt - T0))
            if dissociation:
                e += n_gas * dissociation_sink(tt)
            return e

        if qv <= 0.0:
            t_new = T0
        else:
            lo, hi = T0, T_MAX
            if energy(hi) < qv:
                t_new = hi
            else:
                for _ in range(70):
                    mid = 0.5 * (lo + hi)
                    if energy(mid) < qv:
                        lo = mid
                    else:
                        hi = mid
                t_new = 0.5 * (lo + hi)

        if abs(t_new - t) < 0.05:
            t = t_new
            break
        t += 0.6 * (t_new - t)   # релаксация, чтобы итерация не раскачивалась

    n_gas = sum(comp.values())
    if n_gas <= 0.0:
        raise ValueError("состав не даёт газообразных продуктов")

    r_spec = n_gas * R_UNIVERSAL
    cv_spec = (comp["CO2"] * CO2.cv(t) + comp["CO"] * CO_.cv(t)
               + comp["H2O"] * H2O.cv(t) + comp["H2"] * H2_.cv(t)
               + comp["N2"] * N2_.cv(t) + comp["O2"] * O2_.cv(t)
               + soot * C_S.cv(t) + residue_cv)
    gamma = 1.0 + r_spec / cv_spec

    covolume = COVOLUME_CAL * (
        comp["CO2"] * CO2.b_cov + comp["CO"] * CO_.b_cov
        + comp["H2O"] * H2O.b_cov + comp["H2"] * H2_.b_cov
        + comp["N2"] * N2_.b_cov + comp["O2"] * O2_.b_cov
    )

    mass_gas = (comp["CO2"] * CO2.molar_mass + comp["CO"] * CO_.molar_mass
                + comp["H2O"] * H2O.molar_mass + comp["H2"] * H2_.molar_mass
                + comp["N2"] * N2_.molar_mass + comp["O2"] * O2_.molar_mass)
    mean_m = mass_gas / n_gas

    force = n_gas * R_UNIVERSAL * t

    if soot > 0.0:
        warnings.append(
            f"Резкий недостаток кислорода: {soot:.1f} моль/кг сажи — "
            "закопчённый канал, потеря энергии, сильное дульное пламя.")
    if ob > 5.0:
        warnings.append(
            f"Кислородный баланс +{ob:.1f}%: избыток окислителя, "
            "свободный O2 в газах — ускоренное выгорание канала.")
    if t > 3400.0:
        warnings.append(
            f"T1 = {t:.0f} К: очень горячий заряд, ресурс ствола просядет резко.")
    if t < 2200.0:
        warnings.append(
            f"T1 = {t:.0f} К: заряд холодный и слабый, возможны затяжные "
            "выстрелы и неполное сгорание.")

    return Thermochemistry(
        force=force, flame_temp=t, covolume=covolume, gamma=gamma,
        heat_of_explosion=qv, gas_moles=n_gas,
        gas_composition={key: val for key, val in comp.items() if val > 1e-9},
        condensed=dict(residues), oxygen_balance=ob, mean_molar_mass=mean_m,
        density=density, gas_constant=r_spec, cv_specific=cv_spec,
        soot_fraction=(soot / c if c > 0.0 else 0.0), warnings=warnings,
    )


def burn_rate_params(mix: dict[str, float] | dict[Ingredient, float],
                     chem: Thermochemistry) -> tuple[float, float, float]:
    """Оценка коэффициентов закона горения Вьеля  u = u1 * p^nu.

    Возвращает (u1 [м/(с*Па^nu)], nu, beta_T [1/К] — термочувствительность).

    Физическая основа: скорость горения растёт с температурой пламени
    (тепловой поток в к-фазу), падает от флегматизаторов и инертных добавок.
    Масштаб откалиброван по пироксилиновому пороху: при p = 300 МПа
    линейная скорость горения ~0.40 м/с.
    """
    items: list[tuple[Ingredient, float]] = []
    for key, frac in mix.items():
        ing = key if isinstance(key, Ingredient) else INGREDIENTS[key]
        if frac > 0.0:
            items.append((ing, float(frac)))
    total = sum(f for _, f in items)
    items = [(ing, f / total) for ing, f in items]

    mod = 1.0
    nu = 0.82
    for ing, frac in items:
        mod *= ing.burn_rate_mod ** frac
        nu += ing.exponent_mod * frac
    nu = min(max(nu, 0.60), 1.05)

    # энергетическая часть: u ~ (f / f_ref)^3
    f_ref = 970e3
    energy_factor = (chem.force / f_ref) ** 3.0
    u1_ref = 0.40 / (3.0e8 ** 0.82)     # м/(с*Па^nu) для реперного пороха
    u1 = u1_ref * mod * energy_factor * (3.0e8 ** (0.82 - nu))

    # термочувствительность скорости горения: холодные составы чувствительнее
    beta_t = 0.0035 * (2900.0 / max(chem.flame_temp, 1500.0))
    return u1, nu, beta_t
