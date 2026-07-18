import random
import unittest

from pcc.arithmetic import ARITHMETIC_TOTAL, Decoder, Encoder, frequency_table


class ArithmeticTests(unittest.TestCase):
    def test_frequency_table_is_complete_and_positive(self):
        cumulative = frequency_table([3.0, 2.0, 0.5, -1.0], [1, 2, 3, 4])
        self.assertEqual(cumulative[0], 0)
        self.assertEqual(cumulative[-1], ARITHMETIC_TOTAL)
        self.assertTrue(all(right > left for left, right in zip(cumulative, cumulative[1:])))

    def test_variable_tables_recover_the_source_bit_prefix(self):
        rng = random.Random(7281)
        source = [rng.randrange(2) for _ in range(192)]
        tables = [
            frequency_table([4.0, 2.0, 1.0, 0.0], [1, 1, 2, 3]),
            frequency_table([1.0, 0.8, 0.1], [1, 3, 2]),
        ]
        read_offset = 0
        confirmed = 0
        desynchronized = False

        def source_bit(offset):
            if offset < len(source):
                return source[offset]
            return 1 if (offset - len(source)) % 2 == 0 else 0

        def read():
            nonlocal read_offset
            bit = source_bit(read_offset)
            read_offset += 1
            return bit

        def confirm(bit):
            nonlocal confirmed, desynchronized
            desynchronized |= bit != source_bit(confirmed)
            confirmed += 1

        decoder = Decoder(read)
        verifier = Encoder(confirm)
        symbols = []
        while confirmed < len(source):
            table = tables[len(symbols) % len(tables)]
            symbol = decoder.symbol(table)
            verifier.symbol(symbol, table)
            symbols.append(symbol)
            self.assertFalse(desynchronized)
            self.assertLess(len(symbols), 4096)

        recovered = []
        encoder = Encoder(recovered.append)
        for index, symbol in enumerate(symbols):
            encoder.symbol(symbol, tables[index % len(tables)])
        self.assertEqual(recovered[: len(source)], source)


if __name__ == "__main__":
    unittest.main()
