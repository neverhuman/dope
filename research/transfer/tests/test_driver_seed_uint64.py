"""Synthetic stream/identity regressions; no vendor models or scientific claim."""
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch
import numpy as np

from research.transfer.drivers import permutation_importance
from research.transfer.pilot_learning_contract import DriverPlan, identity

PIN = 'a' * 64


def plan(seed):
    return DriverPlan(((PIN, (0,)),), 4, seed, PIN)


def draws(seed):
    observed = []
    def loss(columns):
        observed.append(columns)
        return 0.0
    permutation_importance(loss, (tuple(range(16)),), {PIN: (0,)},
                           learner_id='synthetic', query_population=PIN,
                           seed=seed, repeats=4)
    return observed


class DriverSeedTests(unittest.TestCase):
    def test_zero_max_and_above_safe_integer_decode_exactly(self):
        for value in (0, (1 << 53) + 1, (1 << 64) - 1):
            with self.subTest(value=value):
                encoded = f'0x{value:016x}'
                value_plan = plan(encoded)
                self.assertEqual(value_plan.native_seed, value)
                self.assertEqual(value_plan.seed, encoded)
                self.assertEqual(asdict(value_plan)['seed'], encoded)

    def test_legacy_safe_integer_wire_and_identity_unchanged(self):
        for seed in (0, 17, (1 << 53) - 1):
            value = plan(seed)
            self.assertEqual(value.native_seed, seed)
            self.assertEqual(value.sha256, identity({
                'groups': ((PIN, (0,)),), 'repeats': 4,
                'seed': seed, 'protocol_sha256': PIN}))
            self.assertEqual(draws(value.native_seed), draws(plan(f'0x{seed:016x}').native_seed))

    def test_bad_encodings_and_lossy_numeric_inputs_refuse(self):
        invalid = (True, False, 1.0, -1, 1 << 53, 1 << 64,
                   '0x0', '0X0000000000000000', '0xFFFFFFFFFFFFFFFF',
                   '0x10000000000000000', '0x000000000000000g',
                   ' 0x0000000000000000', '0x0000000000000000\n', None)
        for seed in invalid:
            with self.subTest(kind=type(seed).__name__):
                with self.assertRaises(ValueError):
                    plan(seed)

    def test_native_rng_receives_full_uint64_not_float_or_low_bits(self):
        for seed in ((1 << 53) + 1, (1 << 64) - 1):
            native = plan(f'0x{seed:016x}').native_seed
            original_rng = np.random.default_rng
            with patch('research.transfer.drivers.np.random.default_rng', wraps=original_rng) as factory:
                actual = draws(native)
                factory.assert_called_once_with(seed)
            expected_rng = original_rng(seed)
            expected = [(tuple(range(16)),)]
            expected += [(tuple(int(i) for i in expected_rng.permutation(16)),) for _ in range(4)]
            self.assertEqual(actual, expected)

    def test_exact_stream_remains_bound_to_operation_identity(self):
        left = plan('0x0020000000000001')
        right = plan('0x0020000000000002')
        self.assertNotEqual(left.sha256, right.sha256)
        for value in (left, right):
            key = identity({'fit': PIN, 'query': PIN, 'labels': PIN,
                            'driver_plan': asdict(value)})
            self.assertEqual(key, identity({'fit': PIN, 'query': PIN, 'labels': PIN,
                                           'driver_plan': asdict(plan(value.seed))}))
        self.assertNotEqual(draws(left.native_seed), draws(right.native_seed))

    def test_direct_permutation_rejects_outside_uint64_before_loss(self):
        loss = unittest.mock.Mock(return_value=0.0)
        for seed in (True, 1.5, -1, 1 << 64, '0xffffffffffffffff'):
            with self.assertRaises(ValueError):
                permutation_importance(loss, ((0, 1),), {PIN: (0,)},
                                       learner_id='synthetic', query_population=PIN, seed=seed)
        loss.assert_not_called()

    def test_prospective_and_executed_operation_keys_match_for_max_stream(self):
        from research.transfer.tests import test_pilot_physical_operations as fixtures
        from research.transfer.pilot_learning_contract import LEARNERS
        from research.transfer.pilot_learning_reports import execute_streamed
        from research.transfer.pilot_physical_operations import reserve_learning_operations
        helper = fixtures.PhysicalOperationTests('test_full_streamed_fixture_matches_prospective_keys_and_physical_aliases')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        execution = helper.execution(replace(fixtures.make_plan(), model_retention='stream_drivers_then_release'))
        labels = fixtures.QueryLabels(execution.plan.query.row_ids, (1.0, 2.0), fixtures.PIN)
        drivers = DriverPlan(((fixtures.PIN, (0,)),), 2, '0xffffffffffffffff', fixtures.PIN)
        wall = {role: {'prepare': 10000, 'predict': 10000} for role in (*LEARNERS, 'null_reference')}
        with helper.journal() as journal:
            helper.begin_learners(journal, 2000000)
            guard = reserve_learning_operations(execution, journal, fixtures.CELL, wall,
                                                driver_plan=drivers, labels=labels)
            self.assertEqual(guard.graph.document()['driver_plan_sha256'], drivers.sha256)
            execution, actual = execute_streamed(execution, labels, drivers)
            self.assertTrue(all(c['status'] == 'completed' for c in execution.cases))
            self.assertTrue(all(n['status'] == 'completed' for n in guard.settlement()))
            self.assertTrue(actual['records'])
            for record in actual['records'].values():
                self.assertEqual(record['driver_plan']['seed'], drivers.seed)
                self.assertTrue(all(c['permutation_seed'] == drivers.seed for c in record['calls']))
            journal.settle_operations(fixtures.CELL, 'learners', guard)
            journal.finish(fixtures.CELL, 'learners', 'completed', observed_wall_ms=100,
                           peak_ram_bytes=1024, output_sha256=fixtures.PIN)
