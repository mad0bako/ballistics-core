"""Физические константы и перевод единиц.

Внутри ядра ВСЁ считается строго в СИ:
    длина  — м,  масса — кг,  время — с,  давление — Па,
    температура — К,  энергия — Дж,  сила пороха f — Дж/кг,
    коволюм alpha — м^3/кг.

Перевод в «оружейные» единицы (мм, граны, м/с, МПа, дюймы) — только на границе,
в слое представления. Это избавляет от классической ошибки смешения систем,
из-за которой в баллистических расчётах регулярно теряется множитель 1e6.
"""
from __future__ import annotations

import math

# --- фундаментальные ---------------------------------------------------------
R_UNIVERSAL = 8.314462618          # Дж/(моль*К)
G0 = 9.80665                       # м/с^2, стандартное ускорение свободного падения
T_STD = 288.15                     # К, МСА на уровне моря
P_STD = 101325.0                   # Па
OMEGA_EARTH = 7.292115e-5          # рад/с, угловая скорость Земли
R_EARTH = 6.371e6                  # м

# --- воздух ------------------------------------------------------------------
R_AIR = 287.0528                   # Дж/(кг*К), сухой воздух
R_VAPOR = 461.495                  # Дж/(кг*К), водяной пар
GAMMA_AIR = 1.400
RHO_STD = P_STD / (R_AIR * T_STD)  # ~1.2250 кг/м^3

# --- множители перевода (умножить, чтобы получить СИ) -----------------------
MM = 1e-3
CM = 1e-2
INCH = 0.0254
FOOT = 0.3048
YARD = 0.9144

GRAM = 1e-3
GRAIN = 64.79891e-6                # 1 гран = 64.79891 мг
POUND = 0.45359237
OUNCE = POUND / 16.0

MPA = 1e6
BAR = 1e5
PSI = 6894.757293168
KSI = 1000.0 * PSI
ATM = 101325.0

JOULE = 1.0
FTLB = 1.3558179483314004
CAL = 4.184
KCAL_PER_MOL = 4184.0              # -> Дж/моль

LITRE = 1e-3                       # м^3
CM3 = 1e-6
MM3 = 1e-9
DM3_PER_KG = 1e-3                  # дм^3/кг -> м^3/кг (коволюм)

CELSIUS_OFFSET = 273.15


def c_to_k(t_c: float) -> float:
    return t_c + CELSIUS_OFFSET


def k_to_c(t_k: float) -> float:
    return t_k - CELSIUS_OFFSET


def f_to_k(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0 + CELSIUS_OFFSET


def fps(v_ms: float) -> float:
    """м/с -> фут/с (для отображения)."""
    return v_ms / FOOT


def ms_from_fps(v_fps: float) -> float:
    return v_fps * FOOT


def grains(m_kg: float) -> float:
    """кг -> граны (для отображения навески/массы пули)."""
    return m_kg / GRAIN


def kg_from_grains(m_gr: float) -> float:
    return m_gr * GRAIN


def mpa(p_pa: float) -> float:
    return p_pa / MPA


def psi(p_pa: float) -> float:
    return p_pa / PSI


def h2o_grains_to_m3(cap_grains_h2o: float) -> float:
    """Ёмкость гильзы в «гранах воды» -> м^3.

    Стандарт релоадинга: объём меряют по массе дистиллированной воды при ~20 C
    (rho = 998.2 кг/м^3).
    """
    return cap_grains_h2o * GRAIN / 998.2


def m3_to_h2o_grains(v_m3: float) -> float:
    return v_m3 * 998.2 / GRAIN


def moa_from_rad(rad: float) -> float:
    return rad * 180.0 * 60.0 / math.pi


def rad_from_moa(moa: float) -> float:
    return moa * math.pi / (180.0 * 60.0)


def mil_from_rad(rad: float) -> float:
    """Тысячные (NATO mil = 1/6400 круга) — берём угловую миллирадиану."""
    return rad * 1000.0
