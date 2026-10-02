"""Hand-calculated marginal controls and sealed native-objective contracts."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import aim_native


EVIDENCE = Path(__file__).resolve().parents[3] / 'target'


class AimNativeTests(unittest.TestCase):
    def fixture(self, root, real, generated):
        worker = root / 'worker'; worker.mkdir()
        projection = worker / 'projection.json'
        projection.write_text(json.dumps({'task': 'regression', 'output_features': 1}))
        valid = worker / 'validation.csv'; valid.write_text(real)
        manifest = worker / 'worker-manifest.json'
        manifest.write_text(json.dumps({'projection_sha256': aim_native.sha(projection),
                                       'projected_hashes': {'validation': aim_native.sha(valid)}}))
        sample = root / 'sample.csv'; sample.write_text(generated)
        return worker, sample, aim_native.sha(manifest), aim_native.sha(sample)

    def test_equal_distributions_with_different_row_counts_have_zero_error(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            root = Path(directory)
            args = self.fixture(root, '0,0\n1,1\n', '1,1\n0,0\n0,0\n1,1\n')
            with patch.object(aim_native, 'SCRATCH', root): result = aim_native.measure(*args, 4)
            self.assertEqual(result['value'], 0)
            self.assertEqual(result['workload_pairs'], 1)
            self.assertFalse(result['formal_dp_claim'])
            self.assertIsNone(result['mfs_v2'])

    def test_disjoint_joint_distributions_have_unit_tv_despite_equal_marginals(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            root = Path(directory)
            args = self.fixture(root, '0,0\n1,1\n', '0,1\n1,0\n')
            with patch.object(aim_native, 'SCRATCH', root): result = aim_native.measure(*args, 1)
            self.assertEqual(result['value'], 1)

    def test_all_pairs_including_target_are_equal_weighted(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            root = Path(directory); a = root / 'a.csv'; b = root / 'b.csv'
            a.write_text('0,0,0\n1,1,0\n'); b.write_text('0,1,0\n1,0,0\n')
            left, n = aim_native.histograms(a, 3); right, m = aim_native.histograms(b, 3)
            errors = [0.5 * sum(abs(x / n - y / m) for x, y in zip(p, q))
                      for p, q in zip(left, right)]
            self.assertEqual(errors, [1, 0, 0])
            self.assertAlmostEqual(sum(errors) / len(errors), 1 / 3)

    def test_epsilon_and_seed_cannot_be_tuned_across_budgets(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            root = Path(directory); args = self.fixture(root, '0,0\n', '0,0\n')
            with patch.object(aim_native, 'SCRATCH', root):
                for epsilon in (True, 0, 2, '1'):
                    with self.assertRaises(ValueError): aim_native.measure(*args, epsilon)
                with self.assertRaises(ValueError): aim_native.measure(*args, 4, sample_seed=211)

    def test_changed_inputs_and_evaluator_paths_reject(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            root = Path(directory); args = self.fixture(root, '0,0\n', '0,0\n')
            with patch.object(aim_native, 'SCRATCH', root):
                args[1].write_text('1,1\n')
                with self.assertRaises(ValueError): aim_native.measure(*args, 1)
                evaluator = root / 'evaluator'; evaluator.mkdir()
                forbidden = evaluator / 'renamed.csv'; forbidden.write_text('0,0\n')
                with self.assertRaises(ValueError): aim_native.safe(forbidden)
                test = root / 'test.csv'; test.write_text('0,0\n')
                with self.assertRaises(ValueError): aim_native.safe(test)

    def test_nonfinite_input_errors_do_not_echo_source_values(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            root = Path(directory); args = self.fixture(root, 'nan,0\n', '0,0\n')
            with patch.object(aim_native, 'SCRATCH', root):
                with self.assertRaisesRegex(ValueError, '^invalid AIM native numeric table$'):
                    aim_native.measure(*args, 10)

    def test_native_inventory_is_bound_and_final_admission_stays_closed(self):
        root = Path(__file__).resolve().parents[1]
        lock = json.loads((root / 'methods.lock.json').read_text())
        method = lock['methods']['AIM']; objective = method['native_objective']
        self.assertEqual(objective['implementation_sha256'],
                         hashlib.sha256((root / 'aim_native.py').read_bytes()).hexdigest())
        self.assertEqual(objective['direction'], 'minimize')
        self.assertTrue(objective['epsilon_fixed_per_cell'])
        self.assertEqual(objective['sample_seed'], 101)
        self.assertFalse(objective['formal_dp_claim'])
        self.assertEqual(method['status'], 'pilot_locked')
        self.assertFalse(lock['complete'])
        self.assertFalse(lock['frozen_for_final_evaluation'])


if __name__ == '__main__': unittest.main()
