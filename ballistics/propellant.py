"""Порох как изделие: рецептура + зерно + закон горения.

Связывает термохимию (что горит) с геометрией (как горит) и даёт
единственную функцию, которая нужна внутренней баллистике:

    burn_rate(p, T_заряда) -> линейная скорость горения, м/с
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .grain import (DeterrentCoating, GrainDesign, ball_powder, cord_powder,
                    flake_powder, seven_perf, stick_powder)
from .thermochem import (Ingredient, Thermochemistry, burn_rate_params,
                         compute_thermochemistry)

T_REF_POWDER = 294.15   # К (21 C) — реперная температура заряда


@dataclass
class Propellant:
    """Готовый порох."""

    name: str
    composition: dict[str, float]
    chem: Thermochemistry
    grain: GrainDesign
    u1: float                       # м/(с*Па^nu)
    nu: float                       # показатель в законе Вьеля
    beta_t: float = 0.0025          # термочувствительность скорости, 1/К
    lot_variation: float = 0.005    # относительный разброс партии (сигма)

    # --- свойства, которые дальше нужны внутренней баллистике ----------------
    @property
    def force(self) -> float:
        return self.chem.force

    @property
    def covolume(self) -> float:
        return self.chem.covolume

    @property
    def density(self) -> float:
        return self.chem.density

    @property
    def theta(self) -> float:
        return self.chem.theta

    @property
    def flame_temp(self) -> float:
        return self.chem.flame_temp

    @property
    def web(self) -> float:
        return self.grain.web

    def burn_rate(self, pressure: float, temperature: float = T_REF_POWDER,
                  z: float = 0.0) -> float:
        """Закон Вьеля с поправками на температуру заряда и флегматизацию."""
        if pressure <= 0.0:
            return 0.0
        temp_factor = 1.0 + self.beta_t * (temperature - T_REF_POWDER)
        return (self.u1 * pressure ** self.nu * temp_factor
                * self.grain.rate_factor(z))

    def force_at(self, temperature: float = T_REF_POWDER) -> float:
        """Сила пороха слабо растёт с начальной температурой заряда."""
        return self.force * (1.0 + 1.2e-4 * (temperature - T_REF_POWDER))

    @property
    def relative_quickness(self) -> float:
        """Относительная быстрота: чем меньше свод и больше u1, тем быстрее.

        Условная величина для сравнения порохов между собой
        (1.0 ~ винтовочный порох среднего горения).
        """
        ref_u1, ref_web = 4.65e-8, 0.35e-3
        return (self.u1 / ref_u1) * (ref_web / max(self.web, 1e-9))

    def describe(self) -> str:
        c = self.chem
        return (f"{self.name}\n"
                f"  f = {c.force / 1e3:.0f} кДж/кг, T1 = {c.flame_temp:.0f} К, "
                f"k = {c.gamma:.3f}, alpha = {c.covolume * 1e3:.3f} дм3/кг, "
                f"КБ = {c.oxygen_balance:+.1f}%\n"
                f"  {self.grain.describe()}\n"
                f"  u1 = {self.u1:.3e} м/(с*Па^nu), nu = {self.nu:.2f}, "
                f"быстрота {self.relative_quickness:.2f}")


def make_propellant(name: str, composition: dict[str, float] | dict[Ingredient, float],
                    grain_factory, *, coating: DeterrentCoating | None = None,
                    burn_rate_scale: float = 1.0) -> Propellant:
    """Собирает порох из рецептуры и формы зерна.

    grain_factory — вызываемое с одним аргументом (плотность), возвращает
    GrainDesign. Плотность берётся из термохимического расчёта смеси, так что
    геометрия зерна автоматически согласована с реальной плотностью состава.
    """
    chem = compute_thermochemistry(composition)
    grain = grain_factory(chem.density)
    if coating is not None:
        grain.coating = coating
    u1, nu, beta_t = burn_rate_params(composition, chem)
    norm: dict[str, float] = {}
    for key, frac in composition.items():
        norm[key if isinstance(key, str) else key.name] = float(frac)
    total = sum(norm.values())
    norm = {k: v / total for k, v in norm.items()}
    return Propellant(name=name, composition=norm, chem=chem, grain=grain,
                      u1=u1 * burn_rate_scale, nu=nu, beta_t=beta_t)


# --- рецептуры-пресеты -------------------------------------------------------

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

TRIPLE_BASE = {
    "Нитроцеллюлоза 13.25% N (пироксилин №1)": 0.280,
    "Нитроглицерин": 0.225,
    "Нитрогуанидин (охладитель)": 0.470,
    "Централит-2 (стабилизатор)": 0.025,
}

COOL_SINGLE_BASE = {
    "Нитроцеллюлоза 13.25% N (пироксилин №1)": 0.855,
    "Оксамид (охладитель)": 0.100,
    "Дифениламин (стабилизатор)": 0.010,
    "Дибутилфталат (флегматизатор)": 0.035,
}


def default_library() -> dict[str, Propellant]:
    """Библиотека порохов, перекрывающая практический диапазон быстроты.

    Геометрия зёрен взята близкой к реальной: пистолетные — тонкая пластина,
    винтовочные — сплошной экструдированный цилиндр (свод e1 = D/2),
    сферические — шар с флегматизирующим покрытием, артиллерийские —
    семиканальное зерно. Именно свод, а не «марка», задаёт быстроту.
    """
    lib: dict[str, Propellant] = {}

    lib["Пистолетный быстрый (пластина)"] = make_propellant(
        "Пистолетный быстрый (пластина)", SINGLE_BASE,
        lambda rho: flake_powder(0.10, 0.85, rho))

    lib["Пистолетный средний (пластина)"] = make_propellant(
        "Пистолетный средний (пластина)", SINGLE_BASE,
        lambda rho: flake_powder(0.22, 1.0, rho))

    lib["Промежуточный (сферический)"] = make_propellant(
        "Промежуточный (сферический)", SINGLE_BASE,
        lambda rho: ball_powder(0.62, rho),
        coating=DeterrentCoating(0.35, 0.35))

    lib["Винтовочный сферический"] = make_propellant(
        "Винтовочный сферический", SINGLE_BASE,
        lambda rho: ball_powder(0.80, rho),
        coating=DeterrentCoating(0.30, 0.40))

    lib["Винтовочный экструдированный"] = make_propellant(
        "Винтовочный экструдированный", SINGLE_BASE,
        lambda rho: cord_powder(0.82, 1.5, rho))

    lib["Магнум медленный"] = make_propellant(
        "Магнум медленный", SINGLE_BASE,
        lambda rho: cord_powder(1.15, 2.0, rho))

    lib["Сверхмедленный (большая ёмкость)"] = make_propellant(
        "Сверхмедленный (большая ёмкость)", SINGLE_BASE,
        lambda rho: cord_powder(1.60, 2.6, rho))

    lib["Двухосновный высокоэнергетический"] = make_propellant(
        "Двухосновный высокоэнергетический", DOUBLE_BASE,
        lambda rho: cord_powder(1.05, 1.8, rho))

    lib["Трёхосновный (щадящий ствол)"] = make_propellant(
        "Трёхосновный (щадящий ствол)", TRIPLE_BASE,
        lambda rho: cord_powder(1.00, 1.8, rho))

    lib["Охлаждённый оксамидом"] = make_propellant(
        "Охлаждённый оксамидом", COOL_SINGLE_BASE,
        lambda rho: cord_powder(0.90, 1.6, rho))

    lib["Артиллерийский семиканальный"] = make_propellant(
        "Артиллерийский семиканальный", SINGLE_BASE,
        lambda rho: seven_perf(11.0, 1.0, 14.0, rho))

    lib["Артиллерийский трубчатый"] = make_propellant(
        "Артиллерийский трубчатый", SINGLE_BASE,
        lambda rho: stick_powder(9.0, 1.2, 300.0, rho))

    # --- пороха, спроектированные под конкретные патроны ---------------------
    # Своды подобраны решателем design.web_for_charge_and_pressure: при
    # паспортной навеске порох обязан давать паспортное давление. Дульная
    # скорость после этого уже не подгоняется — она предсказывается, и
    # именно по ней модель проверяется (см. README, раздел сверки).

    lib["Пистолетный под 9x19"] = make_propellant(
        "Пистолетный под 9x19", SINGLE_BASE,
        lambda rho: flake_powder(0.156, 1.32, rho))

    lib["Промежуточный под 5.56x45"] = make_propellant(
        "Промежуточный под 5.56x45", SINGLE_BASE,
        lambda rho: ball_powder(0.706, rho),
        coating=DeterrentCoating(0.35, 0.35))

    lib["Винтовочный под 7.62x51"] = make_propellant(
        "Винтовочный под 7.62x51", SINGLE_BASE,
        lambda rho: ball_powder(0.767, rho),
        coating=DeterrentCoating(0.30, 0.40))

    # Магнуму нужна прогрессивная геометрия: дегрессивное зерно упирается
    # в предел давления задолго до того, как успеет сгореть, и большая
    # гильза остаётся недоиспользованной.
    lib["Магнум прогрессивный под .338"] = make_propellant(
        "Магнум прогрессивный под .338", SINGLE_BASE,
        lambda rho: seven_perf(4.0, 0.40, 5.0, rho))

    lib["Артиллерийский под 152 мм"] = make_propellant(
        "Артиллерийский под 152 мм", SINGLE_BASE,
        lambda rho: stick_powder(7.32, 0.98, 244.0, rho))

    return lib
