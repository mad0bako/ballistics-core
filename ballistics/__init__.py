"""ballistics — физическое ядро игры про оружейника.

Слои, снизу вверх (каждый зависит только от нижних):

    units, materials          константы, единицы, свойства металлов
    thermochem                состав заряда -> f, T1, k, коволюм, состав газов
    grain                     геометрия зерна -> функция формы psi(z)
    propellant                порох = термохимия + зерно + закон горения
    interior                  основная задача внутренней баллистики
    barrel                    геометрия, прочность (Ламе), жёсткость, нарезы
    projectile                пуля: обводы, масса, моменты инерции, i7/BC
    cartridge                 гильза, посадка, объём каморы, сборка
    erosion                   прогрев канала, износ, ресурс ствола
    exterior                  атмосфера, Cd(M), 3-DOF траектория, устойчивость
    tolerances                Монте-Карло: допуски -> разброс -> кучность
    simulation                сквозной выстрел и серия выстрелов
    design                    обратные задачи: подбор навески, ствола, состава
    workshop                  игровой слой: станки, навык, изготовление

Быстрый старт:

    from ballistics import presets
    weapon, cartridge = presets.rifle_762x51()
    report = simulation.fire(weapon, cartridge, target_range=600.0)
    print(report.summary())
"""

from . import (barrel, cartridge, design, erosion, exterior, grain, interior,
               materials, presets, projectile, propellant, simulation,
               thermochem, tolerances, units, workshop)

__all__ = [
    "units", "materials", "thermochem", "grain", "propellant", "interior",
    "barrel", "projectile", "cartridge", "erosion", "exterior", "tolerances",
    "simulation", "design", "workshop", "presets",
]

__version__ = "1.0.0"
