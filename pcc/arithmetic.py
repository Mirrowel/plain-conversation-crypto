"""Fixed-width arithmetic coder used by the neural carrier experiment."""

from __future__ import annotations

import math

ARITHMETIC_TOTAL = 32768
HALF = 0x80000000
QUARTER = 0x40000000
THREE_QUARTER = 0xC0000000


def frequency_table(scores: list[float], visible_lengths: list[int], temperature: float = 1.0, length_bias: float = 0.1) -> list[int]:
    if len(scores) < 2 or len(scores) >= ARITHMETIC_TOTAL:
        raise ValueError("invalid arithmetic candidate count")
    maximum = max(scores)
    weights = [
        math.exp((score - maximum) / temperature - length_bias * float(length))
        for score, length in zip(scores, visible_lengths)
    ]
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError("invalid arithmetic model score")
    available = ARITHMETIC_TOTAL - len(scores)
    counts = []
    fractions = []
    used = 0
    total = sum(weights)
    for weight in weights:
        exact = weight / total * available
        base = math.floor(exact)
        counts.append(base + 1)
        fractions.append(exact - base)
        used += base + 1
    order = sorted(range(len(scores)), key=lambda index: (-fractions[index], index))
    index = 0
    while used < ARITHMETIC_TOTAL:
        counts[order[index % len(order)]] += 1
        used += 1
        index += 1
    cumulative = [0]
    for count in counts:
        cumulative.append(cumulative[-1] + count)
    return cumulative


class Encoder:
    def __init__(self, emit):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.pending = 0
        self.emit = emit

    def symbol(self, symbol: int, cumulative: list[int]) -> None:
        range_size = self.high - self.low + 1
        self.high = self.low + range_size * cumulative[symbol + 1] // ARITHMETIC_TOTAL - 1
        self.low = self.low + range_size * cumulative[symbol] // ARITHMETIC_TOTAL
        while True:
            if self.high < HALF:
                self._output(0)
            elif self.low >= HALF:
                self._output(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF

    def _output(self, bit: int) -> None:
        self.emit(bit)
        for _ in range(self.pending):
            self.emit(bit ^ 1)
        self.pending = 0


class Decoder:
    def __init__(self, read):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.code = 0
        self.read = read
        for _ in range(32):
            self.code = (self.code << 1) | read()

    def symbol(self, cumulative: list[int]) -> int:
        range_size = self.high - self.low + 1
        value = ((self.code - self.low + 1) * ARITHMETIC_TOTAL - 1) // range_size
        symbol = 0
        while cumulative[symbol + 1] <= value:
            symbol += 1
        self.high = self.low + range_size * cumulative[symbol + 1] // ARITHMETIC_TOTAL - 1
        self.low = self.low + range_size * cumulative[symbol] // ARITHMETIC_TOTAL
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF
            self.code = ((self.code << 1) | self.read()) & 0xFFFFFFFF
        return symbol
