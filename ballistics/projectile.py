"""Пуля/снаряд: геометрия тела вращения, масса, моменты инерции, форм-фактор.

Всё считается численным интегрированием по профилю r(x), поэтому любая
геометрия — оживало, конус, цилиндр, запоясковый конус — обрабатывается
одинаково, и масса не «назначается», а получается из чертежа и материалов.

Оживало (ogive) строится как касательная дуга радиуса R:

    R = (L_н^2 + r0^2) / (2 * r0)         (касательное оживало)
    y(x) = sqrt(R^2 - (L_н - x)^2) - (R - r0)

Для секущего (secant) оживала радиус задаётся напрямую, R > R_касат.

Форм-фактор i7 (отношение сопротивления пули к эталону G7) оценивается
эмпирической зависимостью от длины головной части, площадки на носике и
запоясковой части. Коэффициенты подобраны по четырём реальным пулям
(SMK 168gr i7=1.13, Berger VLD 175gr i7=1.00, M855 i7=1.17, тупоконечная
безоживальная i7=2.6); точность оценки — порядка 5-10%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .materials import GILDING_METAL, LEAD, Metal


@dataclass
class BulletGeometry:
    """Профиль пули в метрах. Начало координат — острие."""

    diameter: float                 # калибр по ведущей части, м
    nose_length: float              # длина головной части, м
    bearing_length: float           # длина ведущей (цилиндрической) части, м
    boattail_length: float = 0.0    # длина запоясковой части, м
    boattail_angle: float = math.radians(8.0)
    meplat_diameter: float = 0.0    # диаметр площадки на носике, м
    ogive_radius: float | None = None   # м; None -> касательное оживало
    hollow_point_depth: float = 0.0
    hollow_point_diameter: float = 0.0

    @property
    def total_length(self) -> float:
        return self.nose_length + self.bearing_length + self.boattail_length

    @property
    def base_diameter(self) -> float:
        d = self.diameter - 2.0 * self.boattail_length * math.tan(self.boattail_angle)
        return max(d, 0.3 * self.diameter)

    @property
    def length_calibers(self) -> float:
        return self.total_length / self.diameter

    @property
    def nose_calibers(self) -> float:
        return self.nose_length / self.diameter

    @property
    def meplat_ratio(self) -> float:
        return self.meplat_diameter / self.diameter

    @property
    def tangent_ogive_radius(self) -> float:
        r0 = 0.5 * self.diameter
        return (self.nose_length ** 2 + r0 * r0) / (2.0 * r0)

    def radius(self, x: float) -> float:
        """Радиус тела вращения на расстоянии x от острия."""
        r0 = 0.5 * self.diameter
        if x <= 0.0:
            return 0.5 * self.meplat_diameter
        if x < self.nose_length:
            big_r = self.ogive_radius or self.tangent_ogive_radius
            # центр дуги лежит на расстоянии (big_r - r0) от оси
            offset = big_r - r0
            dx = self.nose_length - x
            val = big_r * big_r - dx * dx
            y = math.sqrt(max(val, 0.0)) - offset
            y = max(y, 0.5 * self.meplat_diameter)
            return min(y, r0)
        if x <= self.nose_length + self.bearing_length:
            return r0
        bt = x - self.nose_length - self.bearing_length
        if bt >= self.boattail_length:
            return 0.5 * self.base_diameter
        return r0 - bt * math.tan(self.boattail_angle)

    # --- интегральные характеристики -----------------------------------------
    def volume(self, n: int = 400) -> float:
        dx = self.total_length / n
        v = 0.0
        for i in range(n):
            r = self.radius((i + 0.5) * dx)
            v += math.pi * r * r * dx
        v -= self.hollow_point_volume
        return v

    @property
    def hollow_point_volume(self) -> float:
        if self.hollow_point_depth <= 0.0:
            return 0.0
        r = 0.5 * self.hollow_point_diameter
        return (1.0 / 3.0) * math.pi * r * r * self.hollow_point_depth

    def wetted_area(self, n: int = 400) -> float:
        """Площадь боковой поверхности (нужна для трения о воздух)."""
        dx = self.total_length / n
        area = 0.0
        prev = self.radius(0.0)
        for i in range(1, n + 1):
            r = self.radius(i * dx)
            slant = math.hypot(dx, r - prev)
            area += math.pi * (r + prev) * slant
            prev = r
        return area

    @property
    def frontal_area(self) -> float:
        return 0.25 * math.pi * self.diameter ** 2


@dataclass
class Projectile:
    """Пуля: геометрия + конструкция (сердечник/оболочка)."""

    geometry: BulletGeometry
    core_material: Metal = field(default_factory=lambda: LEAD)
    jacket_material: Metal | None = field(default_factory=lambda: GILDING_METAL)
    jacket_thickness: float = 0.55e-3
    name: str = "пуля"
    penetrator_material: Metal | None = None
    penetrator_fraction: float = 0.0     # доля объёма под сердечник
    mass_override: float | None = None   # кг; для снарядов со сложной начинкой
    nose_void_length: float | None = None   # м, полая часть носика (см. ниже)
    form_factor_override: float | None = None   # i7, если известен из отстрела
    band_material: Metal | None = None   # металл ведущего пояска (см. ниже)

    @property
    def driving_band(self) -> Metal:
        """Металл, который врезается в нарезы.

        У пули это оболочка, у артиллерийского снаряда — медный ведущий
        поясок, а вовсе не сталь корпуса: считать врезание по прочности
        корпуса — значит завысить давление форсирования в десятки раз.
        """
        if self.band_material is not None:
            return self.band_material
        return self.jacket_material or self.core_material

    @property
    def _void_length(self) -> float:
        """Длина полой части носика.

        У оболочечной пули свинец засыпается со стороны дна, и до самого
        острия он не доходит: кончик оболочки остаётся пустым. Игнорировать
        эту пустоту — значит завысить массу пули процентов на пять и
        сдвинуть центр масс вперёд. По умолчанию берём 45% длины оживала —
        это соответствует обмерам типовых остроконечных оболочечных пуль.
        """
        if self.nose_void_length is not None:
            return self.nose_void_length
        if self.jacket_material is None or self.jacket_thickness <= 0.0:
            return 0.0
        return 0.45 * self.geometry.nose_length

    # --- масса и распределение массы -----------------------------------------
    def _mass_profile(self, n: int = 400) -> list[tuple[float, float, float]]:
        """[(x, dm, dV)] — распределение массы по длине."""
        g = self.geometry
        dx = g.total_length / n
        void = self._void_length
        out = []
        for i in range(n):
            x = (i + 0.5) * dx
            r = g.radius(x)
            area = math.pi * r * r
            if self.jacket_material is not None and self.jacket_thickness > 0.0:
                r_in = max(r - self.jacket_thickness, 0.0)
                a_core = math.pi * r_in * r_in
                a_jacket = area - a_core
                core_density = (0.0 if x < void
                                else self.core_material.density)
                rho = ((a_core * core_density
                        + a_jacket * self.jacket_material.density) / area
                       if area > 0.0 else 0.0)
            else:
                rho = self.core_material.density
            if self.penetrator_material is not None and self.penetrator_fraction > 0.0:
                rho = (rho * (1.0 - self.penetrator_fraction)
                       + self.penetrator_material.density * self.penetrator_fraction)
            dv = area * dx
            out.append((x, rho * dv, dv))
        return out

    @property
    def mass(self) -> float:
        if self.mass_override is not None:
            return self.mass_override
        prof = self._mass_profile()
        m = sum(dm for _, dm, _ in prof)
        g = self.geometry
        if g.hollow_point_depth > 0.0:
            m -= g.hollow_point_volume * self.core_material.density
        return m

    @property
    def _mass_scale(self) -> float:
        """Поправка моментов инерции, если масса задана напрямую."""
        if self.mass_override is None:
            return 1.0
        prof = self._mass_profile()
        m_geom = sum(dm for _, dm, _ in prof)
        return self.mass_override / m_geom if m_geom > 0.0 else 1.0

    @property
    def center_of_gravity(self) -> float:
        prof = self._mass_profile()
        m = sum(dm for _, dm, _ in prof)
        if m <= 0.0:
            return 0.0
        return sum(x * dm for x, dm, _ in prof) / m

    @property
    def axial_inertia(self) -> float:
        """Ix, кг*м^2 — момент инерции относительно оси симметрии."""
        g = self.geometry
        n = 400
        dx = g.total_length / n
        prof = self._mass_profile(n)
        ix = 0.0
        for (x, dm, dv) in prof:
            r = g.radius(x)
            ix += 0.5 * dm * r * r
            _ = dv
        return ix * self._mass_scale

    @property
    def transverse_inertia(self) -> float:
        """Iy, кг*м^2 — относительно поперечной оси через центр масс."""
        g = self.geometry
        n = 400
        prof = self._mass_profile(n)
        cg = self.center_of_gravity
        iy = 0.0
        for (x, dm, _) in prof:
            r = g.radius(x)
            iy += dm * (0.25 * r * r + (x - cg) ** 2)
        return iy * self._mass_scale

    @property
    def sectional_density(self) -> float:
        """Поперечная нагрузка m/d^2, кг/м^2."""
        return self.mass / self.geometry.diameter ** 2

    # --- аэродинамика --------------------------------------------------------
    @property
    def form_factor_g7(self) -> float:
        """Форм-фактор относительно эталона G7.

        Если пуля отстреляна и её BC известен, надёжнее задать i7 напрямую
        через form_factor_override: оценка по обводам даёт 5-10% и на
        нетиповых формах может ошибаться сильнее.
        """
        if self.form_factor_override is not None:
            return self.form_factor_override
        g = self.geometry
        ln = max(g.nose_calibers, 0.3)
        base = 1.703 * ln ** -0.72
        meplat = 1.0 + 0.5 * g.meplat_ratio
        bt_cal = g.boattail_length / g.diameter
        bt_eff = min(bt_cal / 0.6, 1.0)
        angle_eff = 1.0 - abs(math.degrees(g.boattail_angle) - 8.0) / 45.0
        bt = 1.0 + 0.30 * (1.0 - bt_eff * max(angle_eff, 0.4))
        short_body = 1.0 + 0.06 * max(0.0, 3.0 - g.length_calibers)
        return base * meplat * bt * short_body

    @property
    def form_factor_g1(self) -> float:
        """G1 — более тупой эталон, поэтому i1 примерно вдвое меньше i7."""
        return self.form_factor_g7 * 0.512

    @property
    def bc_g7(self) -> float:
        """Баллистический коэффициент по G7, кг/м^2."""
        return self.mass / (self.form_factor_g7 * self.geometry.diameter ** 2)

    @property
    def bc_g7_imperial(self) -> float:
        """Тот же BC в привычных фунт/дюйм^2."""
        return self.bc_g7 * 0.0254 ** 2 / 0.45359237

    @property
    def bc_g1_imperial(self) -> float:
        return (self.mass / (self.form_factor_g1 * self.geometry.diameter ** 2)
                * 0.0254 ** 2 / 0.45359237)

    @property
    def cost(self) -> float:
        prof = self._mass_profile()
        vol = sum(dv for _, _, dv in prof)
        jacket_vol = 0.0
        if self.jacket_material is not None:
            jacket_vol = self.geometry.wetted_area() * self.jacket_thickness
        core_vol = max(vol - jacket_vol, 0.0)
        cost = core_vol * self.core_material.density * self.core_material.cost_per_kg
        if self.jacket_material is not None:
            cost += (jacket_vol * self.jacket_material.density
                     * self.jacket_material.cost_per_kg)
        return cost * 1.35 + 0.02   # передел + оснастка

    def describe(self) -> str:
        g = self.geometry
        return (f"{self.name}: калибр {g.diameter * 1e3:.2f} мм, длина "
                f"{g.total_length * 1e3:.2f} мм ({g.length_calibers:.2f} клб), "
                f"масса {self.mass * 1e3:.2f} г ({self.mass / 64.79891e-6:.1f} гран)\n"
                f"  оживало {g.nose_calibers:.2f} клб, площадка "
                f"{g.meplat_ratio:.3f} клб, запоясковая часть "
                f"{g.boattail_length / g.diameter:.2f} клб\n"
                f"  i7 = {self.form_factor_g7:.3f}, BC(G7) = "
                f"{self.bc_g7_imperial:.3f} фнт/дюйм2, поперечная нагрузка "
                f"{self.sectional_density:.1f} кг/м2\n"
                f"  Ix = {self.axial_inertia * 1e9:.3f} г*мм2, Iy = "
                f"{self.transverse_inertia * 1e9:.1f} г*мм2, ЦМ "
                f"{100 * self.center_of_gravity / g.total_length:.1f}% длины")


# --- готовые конструкции -----------------------------------------------------

def spitzer_boattail(diameter: float, total_length: float, *,
                     nose_calibers: float = 2.2,
                     boattail_calibers: float = 0.55,
                     meplat_calibers: float = 0.18,
                     jacket_thickness: float = 0.55e-3,
                     name: str = "остроконечная с запоясковым конусом"
                     ) -> Projectile:
    nose = nose_calibers * diameter
    bt = boattail_calibers * diameter
    bearing = max(total_length - nose - bt, 0.2 * diameter)
    geom = BulletGeometry(diameter=diameter, nose_length=nose,
                          bearing_length=bearing, boattail_length=bt,
                          meplat_diameter=meplat_calibers * diameter)
    return Projectile(geom, jacket_thickness=jacket_thickness, name=name)


def flat_base_round_nose(diameter: float, total_length: float,
                         name: str = "тупоконечная плоскодонная") -> Projectile:
    nose = 0.9 * diameter
    geom = BulletGeometry(diameter=diameter, nose_length=nose,
                          bearing_length=total_length - nose,
                          boattail_length=0.0,
                          meplat_diameter=0.12 * diameter)
    return Projectile(geom, jacket_thickness=0.6e-3, name=name)


def artillery_shell(diameter: float, total_length: float, mass: float,
                    name: str = "осколочно-фугасный снаряд") -> Projectile:
    """Снаряд задаётся массой напрямую: начинка неоднородна (корпус, ВВ,
    взрыватель), считать её послойно смысла нет. Геометрия нужна для
    аэродинамики, материал корпуса — для расчёта центробежной прочности
    на нарезах."""
    nose = 2.4 * diameter
    bt = 0.6 * diameter
    geom = BulletGeometry(diameter=diameter, nose_length=nose,
                          bearing_length=max(total_length - nose - bt, 0.5 * diameter),
                          boattail_length=bt,
                          meplat_diameter=0.10 * diameter)
    from .materials import COPPER, STEEL_4140
    return Projectile(geom, core_material=STEEL_4140, jacket_material=None,
                      band_material=COPPER, name=name, mass_override=mass)
