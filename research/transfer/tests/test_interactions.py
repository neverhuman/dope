import unittest

from research.transfer.stress.interactions import (
    permute_columns,
    product_target,
    sign_gate_target,
    signed_bits,
    weak_sum,
)


class InteractionStressTests(unittest.TestCase):
    def test_mechanisms_follow_the_columns_and_hide_the_code(self):
        rows = 35
        left = [1.0 if row % 5 == 0 else -1.0 for row in range(rows)]
        right = [1.0 if row % 7 == 0 else -1.0 for row in range(rows)]
        self.assertEqual(product_target(left, right), [left[row] * right[row] for row in range(rows)])
        gate = sign_gate_target(left, right)
        self.assertEqual(sum(gate), 1)
        self.assertEqual(gate[0], 1.0)
        columns = [signed_bits(32, salt) for salt in range(6)]
        summed = weak_sum(columns, 4)
        self.assertEqual(summed[3], sum(column[3] for column in columns[:4]))
        moved = permute_columns(columns, [2, 0, 1, 3, 4, 5])
        self.assertIs(moved[0], columns[2])
        with self.assertRaises(ValueError) as raised:
            product_target(left, right[:-1])
        self.assertEqual(str(raised.exception), "invalid_shape")
        self.assertNotIn(str(rows), str(raised.exception))
