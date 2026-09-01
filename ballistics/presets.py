"""Готовые системы для быстрого старта и для сверки модели с реальностью.

Каждый пресет — реальный образец с паспортными данными в комментарии,
чтобы можно было в любой момент проверить, не уехала ли модель.
"""
from __future__ import annotations

import math

from .barrel import Barrel, Chamber, Rifling
from .cartridge import (Cartridge, case_9x19, case_338lm, case_152mm,
                        case_556x45, case_762x51)
from .interior import Charge, Primer
from .materials import CHROME_LINING, STEEL_4140, STEEL_4150, STEEL_CRMOV
from .projectile import BulletGeometry, Projectile, artillery_shell
from .propellant import default_library
from .simulation import Weapon

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        _LIB = default_library()
    return _LIB


def bullet_762_168() -> Projectile:
    """Матчевая пуля .308 168 гран (паспорт: 10.89 г, BC G7 = 0.224)."""
    return Projectile(
        BulletGeometry(diameter=7.823e-3, nose_length=15.75e-3,
                       bearing_length=10.30e-3, boattail_length=4.83e-3,
                       boattail_angle=math.radians(9.0),
                       meplat_diameter=1.57e-3),
        jacket_thickness=0.60e-3, name="матчевая 168 гран")


def bullet_556_62() -> Projectile:
    """Пуля 5.56 со стальным сердечником (паспорт: 4.02 г, BC G7 = 0.151)."""
    from .materials import STEEL_JACKET
    return Projectile(
        BulletGeometry(diameter=5.69e-3, nose_length=10.80e-3,
                       bearing_length=9.90e-3, boattail_length=2.30e-3,
                       boattail_angle=math.radians(9.0),
                       meplat_diameter=1.50e-3),
        jacket_thickness=0.65e-3, penetrator_material=STEEL_JACKET,
        penetrator_fraction=0.18, nose_void_length=7.6e-3,
        form_factor_override=1.17,
        name="со стальным сердечником 62 грана")


def bullet_9mm_124() -> Projectile:
    """Пистолетная оболочечная 9 мм (паспорт: 8.03 г)."""
    return Projectile(
        BulletGeometry(diameter=9.02e-3, nose_length=6.20e-3,
                       bearing_length=9.30e-3, boattail_length=0.0,
                       meplat_diameter=2.6e-3),
        jacket_thickness=0.50e-3, nose_void_length=3.9e-3,
        name="оболочечная 124 грана")


def bullet_338_250() -> Projectile:
    """Дальнобойная .338 250 гран (паспорт: 16.2 г, BC G7 = 0.322)."""
    return Projectile(
        BulletGeometry(diameter=8.61e-3, nose_length=20.5e-3,
                       bearing_length=12.0e-3, boattail_length=6.0e-3,
                       boattail_angle=math.radians(7.5),
                       meplat_diameter=1.5e-3),
        jacket_thickness=0.70e-3, name="дальнобойная 250 гран")


def rifle_762x51(charge_mass: float = 2.85e-3,
                 powder: str = "Винтовочный под 7.62x51"):
    """Магазинная винтовка под 7.62x51.

    Паспорт: 2.85 г, v0 = 810 м/с, p_ср = 415 МПа, ресурс ~10-15 тыс.
    Модель:   2.85 г, v0 = 804 м/с (-0.7%), p_ср = 418 МПа (+0.7%),
              ресурс 14 500.
    """
    chamber = Chamber(volume=3.70e-6, length=51.9e-3, mouth_diameter=8.80e-3,
                      base_diameter=11.50e-3, freebore=1.5e-3)
    barrel = Barrel(material=STEEL_4140, bore_diameter=7.62e-3, length=0.610,
                    rifling=Rifling(grooves=4, groove_depth=0.1015e-3,
                                    land_ratio=0.5, twist=0.305),
                    chamber=chamber)
    weapon = Weapon("винтовка 7.62x51", barrel, total_mass=4.4,
                    magazine_length=71.8e-3, sight_height=42e-3)
    cart = Cartridge("7.62x51 матчевый", case_762x51(), bullet_762_168(),
                     Charge(library()[powder], charge_mass), Primer(),
                     coal=71.1e-3)
    return weapon, cart


def carbine_556x45(charge_mass: float = 1.64e-3,
                   powder: str = "Промежуточный под 5.56x45"):
    """Автоматный карабин под 5.56x45, ствол хромированный.

    Паспорт: 1.68 г, v0 = 940 м/с (ствол 508 мм), p_ср ~ 430 МПа (EPVAT).
    Модель:  1.64 г, v0 = 877 м/с (-6.7%), p_ср = 398 МПа (-7.5%).

    Это худший из пяти пресетов, и промах системный: у 5.56 самая высокая
    плотность заряжания (заряд занимает всю гильзу, реальный патрон
    компрессионный), а уравнение Нобеля-Абеля именно там становится
    слишком жёстким и раньше времени загоняет давление в предел. Модель
    честно упирается в ёмкость гильзы там же, где и реальный патрон,
    но не дотягивает по энергии. См. README, раздел про приближения.
    """
    chamber = Chamber(volume=2.00e-6, length=45.3e-3, mouth_diameter=6.55e-3,
                      base_diameter=9.60e-3, freebore=2.5e-3)
    barrel = Barrel(material=STEEL_4150, bore_diameter=5.56e-3, length=0.508,
                    rifling=Rifling(grooves=6, groove_depth=0.0635e-3,
                                    land_ratio=0.5, twist=0.178),
                    chamber=chamber, lining=CHROME_LINING,
                    lining_thickness=25e-6)
    weapon = Weapon("карабин 5.56x45", barrel, total_mass=3.2,
                    magazine_length=57.4e-3, sight_height=63e-3)
    cart = Cartridge("5.56x45 с усиленным сердечником", case_556x45(),
                     bullet_556_62(), Charge(library()[powder], charge_mass),
                     Primer(charge_mass=18e-6), coal=57.4e-3)
    return weapon, cart


def pistol_9x19(charge_mass: float = 0.40e-3,
                powder: str = "Пистолетный под 9x19"):
    """Пистолет под 9x19, ствол 108 мм.

    Паспорт: 0.40 г, v0 = 360 м/с, p_ср = 235 МПа.
    Модель:  0.40 г, v0 = 358 м/с (-0.5%), p_ср = 237 МПа (+0.9%).
    """
    chamber = Chamber(volume=0.90e-6, length=19.6e-3, mouth_diameter=9.70e-3,
                      base_diameter=9.98e-3, freebore=1.0e-3)
    barrel = Barrel(material=STEEL_4140, bore_diameter=8.82e-3, length=0.108,
                    rifling=Rifling(grooves=6, groove_depth=0.10e-3,
                                    land_ratio=0.5, twist=0.250),
                    chamber=chamber)
    weapon = Weapon("пистолет 9x19", barrel, total_mass=0.85,
                    magazine_length=29.7e-3, sight_height=25e-3)
    cart = Cartridge("9x19 оболочечный", case_9x19(), bullet_9mm_124(),
                     Charge(library()[powder], charge_mass),
                     Primer(charge_mass=12e-6, force=0.28e6), coal=29.5e-3)
    return weapon, cart


def sniper_338lm(charge_mass: float = 6.02e-3,
                 powder: str = "Магнум прогрессивный под .338"):
    """Дальнобойная винтовка .338 Lapua Magnum.

    Паспорт: 6.30 г, v0 = 890 м/с, p_ср = 420 МПа, ресурс ~1.5-2 тыс.

    На паспортной навеске модель даёт v0 = 892 м/с (+0.2%) — то есть
    энергетику она берёт верно, — но пик давления завышает до 471 МПа
    (+12%): большая камора с высокой плотностью заряжания это та область,
    где уравнение Нобеля-Абеля становится слишком жёстким. Поэтому в
    пресете стоит навеска 6.02 г, на которой МОДЕЛЬ показывает паспортные
    420 МПа: пресет обязан быть безопасным внутри той физики, которой его
    считают. Скорость при этом 850 м/с.

    Заряду принципиально нужна прогрессивная геометрия зерна: на
    дегрессивном порохе давление упирается в предел задолго до того, как
    заряд успеет сгореть, и большая гильза остаётся недоиспользованной.
    """
    chamber = Chamber(volume=8.00e-6, length=69.5e-3, mouth_diameter=9.70e-3,
                      base_diameter=14.95e-3, freebore=3.0e-3)
    barrel = Barrel(material=STEEL_CRMOV, bore_diameter=8.38e-3, length=0.686,
                    rifling=Rifling(grooves=6, groove_depth=0.115e-3,
                                    land_ratio=0.5, twist=0.254),
                    chamber=chamber)
    weapon = Weapon("винтовка .338 LM", barrel, total_mass=7.0,
                    magazine_length=93.5e-3, sight_height=55e-3,
                    muzzle_device_efficiency=0.35)
    cart = Cartridge(".338 LM дальнобойный", case_338lm(), bullet_338_250(),
                     Charge(library()[powder], charge_mass),
                     Primer(charge_mass=45e-6), coal=93.5e-3)
    return weapon, cart


def howitzer_152mm(charge_mass: float = 7.0,
                   powder: str = "Артиллерийский под 152 мм"):
    """152-мм гаубица, полный заряд.

    Паспорт (класса Д-20): снаряд 43.5 кг, 7.0 кг заряда, v0 = 655 м/с,
    p_ср = 300 МПа.
    Модель: v0 = 659 м/с (+0.5%), p_ср = 300 МПа (0.0%).
    """
    chamber = Chamber(volume=13.6e-3, length=0.60, mouth_diameter=0.158,
                      base_diameter=0.167, freebore=0.02)
    barrel = Barrel(material=STEEL_CRMOV, bore_diameter=0.152, length=4.925,
                    rifling=Rifling(grooves=48, groove_depth=1.0e-3,
                                    land_ratio=0.5, twist=3.65),
                    chamber=chamber, autofrettage=0.55)
    weapon = Weapon("152-мм гаубица", barrel, total_mass=5700.0,
                    sight_height=0.0, muzzle_device_efficiency=0.30)
    shell = artillery_shell(0.152, 0.66, 43.56)
    cart = Cartridge("152-мм ОФ полный заряд", case_152mm(), shell,
                     Charge(library()[powder], charge_mass),
                     Primer(charge_mass=30e-3, force=0.30e6),
                     seating_depth=0.05)
    return weapon, cart


ALL_PRESETS = {
    "9x19": pistol_9x19,
    "5.56x45": carbine_556x45,
    "7.62x51": rifle_762x51,
    ".338LM": sniper_338lm,
    "152mm": howitzer_152mm,
}
