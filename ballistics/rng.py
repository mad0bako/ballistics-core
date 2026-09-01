"""Детерминированный генератор случайных чисел.

Штатный random Питона использовать нельзя по двум причинам.

Первая — портируемость. Ядро переносится на TypeScript, и Монте-Карло
обязано давать там ПОБИТОВО те же числа, иначе эталонные векторы не
проверяют ничего. Mersenne Twister воспроизводить в JS — мучение,
а splitmix32 умещается в пять строк на любом языке.

Вторая — игровая. Партия боеприпаса должна быть воспроизводимой: если
игрок отстрелял серию из партии №184, то повторный отстрел той же партии
обязан дать те же выстрелы. Партия — это просто зерно генератора.

Алгоритм splitmix32: счётчик с золотым шагом плюс лавинное перемешивание.
Все операции — в 32-битной беззнаковой арифметике, чтобы в JS их можно
было повторить через Math.imul.
"""
from __future__ import annotations

import math

MASK32 = 0xFFFFFFFF
GOLDEN32 = 0x9E3779B9
MIX_A = 0x21F0AAAD
MIX_B = 0x735A2D97
TWO_POW_32 = 4294967296.0


class Rng:
    """Воспроизводимый поток случайных чисел."""

    __slots__ = ("state", "_spare")

    def __init__(self, seed: int = 0) -> None:
        self.state = seed & MASK32
        self._spare: float | None = None

    def next_uint32(self) -> int:
        self.state = (self.state + GOLDEN32) & MASK32
        z = self.state
        z = ((z ^ (z >> 16)) * MIX_A) & MASK32
        z = ((z ^ (z >> 15)) * MIX_B) & MASK32
        return (z ^ (z >> 15)) & MASK32

    def random(self) -> float:
        """Равномерное [0, 1)."""
        return self.next_uint32() / TWO_POW_32

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Нормальное распределение по Боксу-Мюллеру.

        Пара значений генерируется разом, второе придерживается до
        следующего вызова — так расходуется ровно два числа на два
        нормальных, и порядок потребления одинаков в любой реализации.
        """
        if self._spare is not None:
            value = self._spare
            self._spare = None
            return mu + sigma * value
        u1 = self.random()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = self.random()
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        self._spare = radius * math.sin(angle)
        return mu + sigma * (radius * math.cos(angle))

    def fork(self, tag: int) -> "Rng":
        """Независимый поток, выведенный из текущего.

        Нужен, чтобы разные источники разброса (навеска, свод, биение)
        не перемешивались: добавление нового источника не должно менять
        уже сгенерированные значения остальных.
        """
        return Rng((self.state ^ ((tag * GOLDEN32) & MASK32)) & MASK32)
