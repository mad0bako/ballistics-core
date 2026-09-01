"""Геометрия порохового зерна и закон газообразования (функция формы).

Основа — геометрический закон горения: зерно горит параллельными слоями с
одинаковой линейной скоростью по всей поверхности. Тогда доля сгоревшей массы

    psi(z) = kappa * z * (1 + lambda*z + mu*z*z),   z = e / e1

где e — глубина сгоревшего свода, e1 — половина минимального свода (для
z = 1 первичная форма исчерпана). Коэффициенты kappa, lambda, mu у Серебрякова
даны таблицами для типовых зёрен; здесь они считаются ТОЧНО из размеров.

Приём: объём зерна V(e) для всех практических форм — полином не выше третьей
степени по e. Значит psi(z) = 1 - V(z*e1)/V(0) — тоже кубика без свободного
члена, и её коэффициенты однозначно восстанавливаются по трём точкам
z = 1/3, 2/3, 1. Это избавляет от ручной алгебры и одинаково работает для
шара, ленты, трубки и семиканального зерна.

Для дегрессивно распадающихся зёрен (многоканальных) после z = 1 остаются
«косточки» — фаза догорания остатков:

    psi_s(z) = chi_s * z + lambda_s * z^2,   1 <= z <= z_k

с условиями psi_s(1) = psi(1) и psi_s(z_k) = 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class FormFunction:
    """Коэффициенты закона газообразования конкретного зерна."""

    kappa: float
    lam: float
    mu: float
    z_k: float = 1.0          # z, при котором зерно сгорает полностью
    chi_s: float = 0.0        # коэффициенты фазы догорания остатков
    lam_s: float = 0.0
    psi_1: float = 1.0        # доля, сгоревшая к концу первичной фазы

    def psi(self, z: float) -> float:
        """Доля сгоревшей массы при относительной глубине свода z."""
        if z <= 0.0:
            return 0.0
        if z >= self.z_k:
            return 1.0
        if z <= 1.0:
            return self.kappa * z * (1.0 + self.lam * z + self.mu * z * z)
        return min(1.0, self.chi_s * z + self.lam_s * z * z)

    def dpsi_dz(self, z: float) -> float:
        """Производная — пропорциональна текущей поверхности горения."""
        if z <= 0.0 or z >= self.z_k:
            return 0.0
        if z <= 1.0:
            return self.kappa * (1.0 + 2.0 * self.lam * z + 3.0 * self.mu * z * z)
        return self.chi_s + 2.0 * self.lam_s * z

    @property
    def progressivity(self) -> float:
        """S(z=1)/S(z=0): >1 — прогрессивное горение, <1 — дегрессивное.

        Прогрессивный порох даёт более пологую кривую давления: та же дульная
        скорость при меньшем p_max, то есть меньший износ ствола.
        """
        s0 = self.dpsi_dz(1e-9)
        s1 = self.dpsi_dz(0.999)
        return s1 / s0 if s0 > 0.0 else 0.0


@dataclass
class DeterrentCoating:
    """Флегматизирующее покрытие: поверхность зерна горит медленнее.

    depth_fraction — глубина проникновения флегматизатора в долях свода e1;
    surface_factor — во сколько раз замедлена скорость на самой поверхности.
    Это основной способ сделать прогрессивным сферический (дегрессивный) порох.
    """

    depth_fraction: float = 0.25
    surface_factor: float = 0.45

    def rate_factor(self, z: float) -> float:
        if self.depth_fraction <= 0.0 or z >= self.depth_fraction:
            return 1.0
        frac = z / self.depth_fraction
        return self.surface_factor + (1.0 - self.surface_factor) * frac


class GrainShape:
    """Базовый класс формы зерна.

    Наследник обязан задать web (e1), volume_at(e) и surface_at(e=0).
    """

    name = "abstract"

    def web(self) -> float:
        raise NotImplementedError

    def volume_at(self, e: float) -> float:
        raise NotImplementedError

    def initial_surface(self) -> float:
        raise NotImplementedError

    def sliver_z_k(self) -> float:
        return 1.0

    # --- общая машинерия -----------------------------------------------------
    def volume(self) -> float:
        return self.volume_at(0.0)

    def form_function(self) -> FormFunction:
        """Точное восстановление kappa, lambda, mu по трём точкам."""
        e1 = self.web()
        v0 = self.volume_at(0.0)
        if v0 <= 0.0 or e1 <= 0.0:
            raise ValueError(f"вырожденная геометрия зерна: {self.name}")

        zs = (1.0 / 3.0, 2.0 / 3.0, 1.0)
        psis = [1.0 - max(self.volume_at(z * e1), 0.0) / v0 for z in zs]

        # решаем 3x3: psi = k*z + kl*z^2 + km*z^3
        rows = [[z, z * z, z ** 3, p] for z, p in zip(zs, psis)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(rows[r][col]))
            rows[col], rows[piv] = rows[piv], rows[col]
            pv = rows[col][col]
            rows[col] = [val / pv for val in rows[col]]
            for r in range(3):
                if r != col and rows[r][col] != 0.0:
                    fac = rows[r][col]
                    rows[r] = [a - fac * b for a, b in zip(rows[r], rows[col])]
        k, kl, km = rows[0][3], rows[1][3], rows[2][3]

        kappa = k
        lam = kl / k if k != 0.0 else 0.0
        mu = km / k if k != 0.0 else 0.0

        z_k = self.sliver_z_k()
        psi_1 = kappa * (1.0 + lam + mu)
        chi_s = lam_s = 0.0
        if z_k > 1.0 + 1e-9 and psi_1 < 1.0:
            den = z_k * z_k - z_k
            chi_s = (psi_1 * z_k * z_k - 1.0) / den
            lam_s = (1.0 - psi_1 * z_k) / den
        else:
            z_k = 1.0
            psi_1 = 1.0

        return FormFunction(kappa=kappa, lam=lam, mu=mu, z_k=z_k,
                            chi_s=chi_s, lam_s=lam_s, psi_1=psi_1)

    def check_kappa(self) -> float:
        """kappa должна равняться S1*e1/V1 — независимая проверка геометрии."""
        return self.initial_surface() * self.web() / self.volume()


@dataclass
class Sphere(GrainShape):
    """Сферическое зерно (шаровой/сферический порох). Резко дегрессивное."""

    diameter: float          # м
    name: str = field(default="сферическое", init=False)

    def web(self) -> float:
        return 0.5 * self.diameter

    def volume_at(self, e: float) -> float:
        r = 0.5 * self.diameter - e
        return (4.0 / 3.0) * math.pi * r ** 3 if r > 0.0 else 0.0

    def initial_surface(self) -> float:
        return math.pi * self.diameter ** 2


@dataclass
class Flake(GrainShape):
    """Пластинчатое (чешуйчатое) зерно."""

    thickness: float         # м, полный свод (e1 = thickness/2)
    length: float
    width: float
    name: str = field(default="пластинчатое", init=False)

    def web(self) -> float:
        return 0.5 * min(self.thickness, self.length, self.width)

    def volume_at(self, e: float) -> float:
        a = self.length - 2.0 * e
        b = self.width - 2.0 * e
        c = self.thickness - 2.0 * e
        if a <= 0.0 or b <= 0.0 or c <= 0.0:
            return 0.0
        return a * b * c

    def initial_surface(self) -> float:
        a, b, c = self.length, self.width, self.thickness
        return 2.0 * (a * b + b * c + a * c)


@dataclass
class Cord(GrainShape):
    """Цилиндрическое зерно без канала (шнур, «зёрнышко» без перфорации)."""

    diameter: float
    length: float
    name: str = field(default="цилиндрическое", init=False)

    def web(self) -> float:
        return 0.5 * min(self.diameter, self.length)

    def volume_at(self, e: float) -> float:
        d = self.diameter - 2.0 * e
        l = self.length - 2.0 * e
        if d <= 0.0 or l <= 0.0:
            return 0.0
        return 0.25 * math.pi * d * d * l

    def initial_surface(self) -> float:
        d, l = self.diameter, self.length
        return 0.5 * math.pi * d * d + math.pi * d * l


@dataclass
class Tube(GrainShape):
    """Трубчатое зерно (один канал). Практически нейтральное горение.

    Внутренняя поверхность растёт ровно настолько, насколько убывает внешняя:
    (D-2e)^2 - (d+2e)^2 = (D+d)(D-d-4e) — линейно по e.
    """

    outer_diameter: float
    inner_diameter: float
    length: float
    name: str = field(default="трубчатое", init=False)

    def web(self) -> float:
        return 0.25 * (self.outer_diameter - self.inner_diameter)

    def volume_at(self, e: float) -> float:
        do = self.outer_diameter - 2.0 * e
        di = self.inner_diameter + 2.0 * e
        l = self.length - 2.0 * e
        if do <= di or l <= 0.0:
            return 0.0
        return 0.25 * math.pi * (do * do - di * di) * l

    def initial_surface(self) -> float:
        do, di, l = self.outer_diameter, self.inner_diameter, self.length
        return (0.5 * math.pi * (do * do - di * di)
                + math.pi * (do + di) * l)


@dataclass
class MultiPerf(GrainShape):
    """Многоканальное зерно (7 или 19 каналов) — прогрессивное горение.

    Свод берётся как перемычка между соседними каналами. Для канонического
    семиканального зерна с равными перемычками e1 = (D - 3d) / 4.
    После распада зерна догорают «косточки» (фаза до z_k).
    """

    outer_diameter: float
    perf_diameter: float
    length: float
    perforations: int = 7
    z_k_override: float | None = None
    name: str = field(default="многоканальное", init=False)

    @property
    def rings(self) -> int:
        """Число колец гексагональной укладки: n = 3m^2 + 3m + 1."""
        m = 0
        while 3 * m * m + 3 * m + 1 < self.perforations:
            m += 1
        return m

    def web(self) -> float:
        """e1 = (D - (2m+1)*d) / (4*(m+1)).

        Вывод: каналы стоят гексагонально с шагом s, наружное кольцо на
        радиусе m*s. Равенство перемычек (внутренней и наружной) даёт
        s = (D + d) / (2(m+1)), а свод — половину перемычки e1 = (s - d)/2.
        При m = 0 формула вырождается в трубку: e1 = (D - d)/4.
        """
        m = self.rings
        return (self.outer_diameter - (2 * m + 1) * self.perf_diameter) / (
            4.0 * (m + 1))

    def volume_at(self, e: float) -> float:
        do = self.outer_diameter - 2.0 * e
        dp = self.perf_diameter + 2.0 * e
        l = self.length - 2.0 * e
        if do <= 0.0 or l <= 0.0:
            return 0.0
        area = 0.25 * math.pi * (do * do - self.perforations * dp * dp)
        return max(area, 0.0) * l

    def initial_surface(self) -> float:
        do, dp, l, n = (self.outer_diameter, self.perf_diameter,
                        self.length, self.perforations)
        ends = 2.0 * 0.25 * math.pi * (do * do - n * dp * dp)
        return ends + math.pi * do * l + n * math.pi * dp * l

    def geometric_z_k(self) -> float:
        """Оценка z_k по геометрии внутренней «косточки».

        Остаток между тремя соседними каналами исчезает, когда фронт из
        каналов доходит до центра криволинейного треугольника:
            e_k = s/sqrt(3) - d/2,   z_k = e_k / e1.
        Даёт нижнюю границу: наружные косточки живут дольше внутренних.
        """
        m = max(self.rings, 1)
        s = (self.outer_diameter + self.perf_diameter) / (2.0 * (m + 1))
        e_k = s / math.sqrt(3.0) - 0.5 * self.perf_diameter
        e1 = self.web()
        return max(1.0, e_k / e1) if e1 > 0.0 else 1.0

    def sliver_z_k(self) -> float:
        if self.z_k_override is not None:
            return self.z_k_override
        # Табличные значения Серебрякова: они больше чисто геометрической
        # оценки, потому что наружные косточки догорают дольше внутренних.
        table = {7: 1.53, 19: 1.60}
        return table.get(self.perforations, max(1.25, self.geometric_z_k()))


@dataclass
class GrainDesign:
    """Зерно целиком: форма + покрытие + плотность вещества."""

    shape: GrainShape
    density: float                                  # кг/м^3 (из термохимии)
    coating: DeterrentCoating | None = None
    packing_fraction: float = 0.60                  # насыпная / истинная

    def __post_init__(self) -> None:
        self._form = self.shape.form_function()

    @property
    def form(self) -> FormFunction:
        return self._form

    @property
    def web(self) -> float:
        return self.shape.web()

    @property
    def grain_mass(self) -> float:
        return self.shape.volume() * self.density

    @property
    def bulk_density(self) -> float:
        """Гравиметрическая плотность, кг/м^3 — предел засыпки в гильзу."""
        return self.density * self.packing_fraction

    def grain_count(self, charge_mass: float) -> float:
        return charge_mass / max(self.grain_mass, 1e-15)

    def rate_factor(self, z: float) -> float:
        return self.coating.rate_factor(z) if self.coating else 1.0

    def describe(self) -> str:
        f = self.form
        return (f"{self.shape.name}: свод 2e1 = {2e3 * self.web:.3f} мм, "
                f"kappa={f.kappa:.3f} lambda={f.lam:+.3f} mu={f.mu:+.3f} "
                f"z_k={f.z_k:.2f}, прогрессивность S1/S0 = "
                f"{f.progressivity:.2f}, масса зерна {self.grain_mass * 1e6:.3f} мг")


# --- типовые зёрна (удобные пресеты) ----------------------------------------


def ball_powder(diameter_mm: float, density: float = 1600.0,
                coating: DeterrentCoating | None = None) -> GrainDesign:
    return GrainDesign(Sphere(diameter_mm * 1e-3), density,
                       coating or DeterrentCoating(), packing_fraction=0.62)


def stick_powder(od_mm: float, id_mm: float, length_mm: float,
                 density: float = 1600.0) -> GrainDesign:
    return GrainDesign(Tube(od_mm * 1e-3, id_mm * 1e-3, length_mm * 1e-3),
                       density, packing_fraction=0.55)


def cord_powder(diameter_mm: float, length_mm: float,
                density: float = 1600.0) -> GrainDesign:
    """Сплошной экструдированный цилиндр — самая частая форма винтовочного
    пороха (IMR/Varget-подобные). Свод e1 = D/2, горение дегрессивное."""
    return GrainDesign(Cord(diameter_mm * 1e-3, length_mm * 1e-3),
                       density, packing_fraction=0.57)


def flake_powder(thickness_mm: float, size_mm: float,
                 density: float = 1600.0) -> GrainDesign:
    return GrainDesign(Flake(thickness_mm * 1e-3, size_mm * 1e-3, size_mm * 1e-3),
                       density, packing_fraction=0.50)


def seven_perf(od_mm: float, perf_mm: float, length_mm: float,
               density: float = 1600.0) -> GrainDesign:
    return GrainDesign(MultiPerf(od_mm * 1e-3, perf_mm * 1e-3, length_mm * 1e-3, 7),
                       density, packing_fraction=0.58)
