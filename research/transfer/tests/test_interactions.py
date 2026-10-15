import unittest

from research.transfer.stress.interactions import (
    DEVELOPMENT_STRESS,
    development_conditional_interaction,
    development_continuous_product,
    development_diffuse_additive,
    development_rare_gate,
    development_xor_parity,
    permute_columns,
    product_target,
    sign_gate_target,
    signed_bits,
    weak_sum,
)


def _dot(left, right):
    return sum(left[row] * right[row] for row in range(len(left)))


def _same_or_negated(left, right):
    return left == right or tuple(-value for value in left) == tuple(right)


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

    def test_historical_signed_bits_keep_their_values(self):
        self.assertEqual(signed_bits(4, 0), [-1.0, 1.0, 1.0, 1.0])
        self.assertEqual(signed_bits(4, 1), [-1.0, -1.0, -1.0, -1.0])
        self.assertEqual(len(set(signed_bits(4, 1))), 1)

    def test_balanced_parity_has_no_marginal_and_a_real_interaction(self):
        report = development_xor_parity(32, 4, (2, 0), seed=0, signal=1.0, noise=0.0)
        self.assertEqual(report["format"], "dope-interaction-stress-development")
        self.assertEqual(report["version"], 1)
        self.assertFalse(report["qualifies_transfer"])
        self.assertFalse(report["feeds_selection"])
        self.assertFalse(report["formal_dp"])
        self.assertFalse(report["hipaa"])
        self.assertNotIn("selected", report)
        left = report["columns"][2]
        right = report["columns"][0]
        product = tuple(left[row] * right[row] for row in range(32))
        self.assertEqual(report["target"], product)
        self.assertEqual(_dot(left, report["target"]), 0.0)
        self.assertEqual(_dot(right, report["target"]), 0.0)
        self.assertEqual(_dot(product, report["target"]), 32.0)
        nuisance = report["columns"][1]
        self.assertEqual(_dot(nuisance, report["target"]), 0.0)
        self.assertFalse(_same_or_negated(nuisance, left))
        self.assertFalse(_same_or_negated(nuisance, right))
        parity = development_xor_parity(32, 5, (0, 1, 3), seed=2, signal=1.0, noise=0.0)
        factors = [parity["columns"][index] for index in (0, 1, 3)]
        triple = tuple(factors[0][row] * factors[1][row] * factors[2][row] for row in range(32))
        self.assertEqual(parity["target"], triple)
        for factor in factors:
            self.assertEqual(_dot(factor, parity["target"]), 0.0)
        pair = tuple(factors[0][row] * factors[1][row] for row in range(32))
        self.assertEqual(_dot(pair, parity["target"]), 0.0)
        self.assertEqual(_dot(triple, parity["target"]), 32.0)

    def test_controls_preserve_the_transformed_mechanism(self):
        base = development_xor_parity(
            32, 4, (0, 1), seed=0, signal=2.0, noise=0.0, scale=3.0, offset=-1.0,
        )
        moved = development_xor_parity(
            32, 4, (0, 1), seed=0, signal=2.0, noise=0.0, scale=3.0, offset=-1.0,
            order=(2, 0, 1, 3),
        )
        product = [base["columns"][0][row] * base["columns"][1][row] for row in range(32)]
        self.assertEqual(base["target"], tuple(3.0 * (2.0 * value) - 1.0 for value in product))
        self.assertEqual(moved["target"], base["target"])
        self.assertEqual(moved["columns"][0], base["columns"][2])
        other = development_xor_parity(32, 4, (0, 1), seed=1, signal=2.0, noise=0.0, scale=3.0, offset=-1.0)
        self.assertEqual(other["target"], base["target"])
        self.assertNotEqual(other["columns"], base["columns"])
        repeated = development_xor_parity(
            32, 4, (0, 1), seed=1, signal=2.0, noise=0.0, scale=3.0, offset=-1.0,
        )
        self.assertEqual(repeated, other)
        noisy = development_xor_parity(32, 4, (0, 1), seed=4, signal=1.0, noise=0.25, scale=2.0, offset=0.0)
        clean = development_xor_parity(32, 4, (0, 1), seed=4, signal=1.0, noise=0.0, scale=2.0, offset=0.0)
        self.assertNotEqual(noisy["target"], clean["target"])
        for noisy_value, clean_value in zip(noisy["target"], clean["target"]):
            self.assertLessEqual(abs(noisy_value - clean_value), 0.5)
        constant = development_continuous_product(
            16, 3, (0, 2), seed=5, signal=0.0, noise=0.0, scale=4.0, offset=1.5,
        )
        self.assertEqual(constant["target"], tuple([1.5] * 16))

    def test_product_rare_diffuse_and_conditional_arrays(self):
        product = development_continuous_product(
            16, 4, (3, 1), seed=6, signal=2.0, noise=0.0, scale=1.0, offset=0.5,
        )
        raw = [
            product["columns"][3][row] * product["columns"][1][row] for row in range(16)
        ]
        self.assertEqual(product["target"], tuple(2.0 * value + 0.5 for value in raw))
        for column in product["columns"]:
            self.assertEqual(len(set(column)), 16)
            self.assertTrue(all(value == value and abs(value) != float("inf") for value in column))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(_same_or_negated(product["columns"][left], product["columns"][right]))
        rare = development_rare_gate(32, 5, (4, 1, 2, 0), seed=0, signal=1.0, noise=0.0)
        events = [row for row, value in enumerate(rare["target"]) if value == 1.0]
        self.assertEqual(len(events), 2)
        for row in range(32):
            positive = all(rare["columns"][index][row] > 0.0 for index in (4, 1, 2, 0))
            self.assertEqual(rare["target"][row], 1.0 if positive else 0.0)
        nuisance = rare["columns"][3]
        for index in (4, 1, 2, 0):
            self.assertFalse(_same_or_negated(nuisance, rare["columns"][index]))
        diffuse = development_diffuse_additive(32, 6, (1, 3, 5), seed=0, signal=0.25, noise=0.0)
        dots = [_dot(diffuse["columns"][index], diffuse["target"]) for index in (1, 3, 5)]
        self.assertEqual(dots, [8.0, 8.0, 8.0])
        for index in (0, 2, 4):
            self.assertEqual(_dot(diffuse["columns"][index], diffuse["target"]), 0.0)
        conditional = development_conditional_interaction(
            32, 5, (0, 2, 4), seed=3, signal=1.0, noise=0.0,
        )
        regime = conditional["columns"][4]
        self.assertGreater(sum(1 for value in regime if value > 0.0), 0)
        saw_zero = False
        for row in range(32):
            product_value = conditional["columns"][0][row] * conditional["columns"][2][row]
            if regime[row] > 0.0:
                self.assertEqual(conditional["target"][row], product_value)
            else:
                self.assertEqual(conditional["target"][row], 0.0)
                saw_zero = True
        self.assertTrue(saw_zero)
        self.assertNotEqual(
            conditional["target"],
            tuple(conditional["columns"][0][row] * conditional["columns"][2][row] for row in range(32)),
        )

    def test_development_shapes_fail_before_allocation(self):
        self.assertEqual(DEVELOPMENT_STRESS["cell_budget"], 4096)
        self.assertEqual(DEVELOPMENT_STRESS["version"], 1)
        allowed = development_xor_parity(256, 16, (0, 1), seed=0, noise=0.0)
        self.assertEqual(len(allowed["columns"]), 16)
        self.assertEqual(len(allowed["columns"][0]), 256)
        cases = (
            lambda: development_xor_parity(True, 4, (0, 1), seed=0),
            lambda: development_xor_parity(288, 16, (0, 1), seed=0),
            lambda: development_xor_parity(10**6, 10**6, (0, 1), seed=0),
            lambda: development_continuous_product(16, 3, (0, 1), seed=0, signal=float("nan")),
            lambda: development_rare_gate(32, 5, (0, 1, 2, 3), seed=0, noise=-1.0),
            lambda: development_diffuse_additive(32, 4, (0, 1), seed=-1),
            lambda: development_conditional_interaction(32, 4, (0, 1, 1), seed=0),
            lambda: development_xor_parity(32, 4, (0, 1), seed=0, order=(0, 0, 1, 2)),
            lambda: development_xor_parity(12, 4, (0, 1), seed=0),
        )
        for case in cases:
            with self.assertRaises(ValueError) as raised:
                case()
            self.assertEqual(str(raised.exception), "invalid_shape")
            self.assertNotIn("1000000", str(raised.exception))
            self.assertNotIn("nan", str(raised.exception))
