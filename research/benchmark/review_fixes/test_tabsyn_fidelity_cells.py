"""Published fidelity formula properties on toy values only."""
import unittest
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch
from research.benchmark.review_fixes import tabsyn_fidelity_cells as producer
import numpy as np
from research.benchmark.review_fixes.tabsyn_fidelity_cells import fidelity


class FidelityTests(unittest.TestCase):
    def test_equal_distributions_and_correlations(self):
        table = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        self.assertEqual(fidelity(table, table[::-1]),
                         {'marginal_ks_mean': 0., 'pair_correlation_fidelity': 1.})

    def test_equal_marginals_can_have_opposite_dependence(self):
        table = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        synthetic = table.copy(); synthetic[:, 1] = table[::-1, 1]
        self.assertEqual(fidelity(table, synthetic),
                         {'marginal_ks_mean': 0., 'pair_correlation_fidelity': 0.})

    def test_shifted_marginals_preserve_dependence(self):
        table = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
        self.assertEqual(fidelity(table, table + 10),
                         {'marginal_ks_mean': 1., 'pair_correlation_fidelity': 1.})

    def test_invalid_width_or_nonfinite_values_refused(self):
        table = np.array([[0., 0.], [1., 1.]])
        for synthetic in (table[:, :1], np.full((2, 2), np.nan)):
            with self.assertRaisesRegex(ValueError, 'dimensions or finite'):
                fidelity(table, synthetic)


class ProducerBindingTests(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target/fidelity-binding-tests'
        target.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(prefix='fidelity-opaque-toy-', dir=target)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dataset = 'aa00000000000001'
        self.workers = self.root / 'toy-workers'
        self.samples = self.root / 'toy-samples'
        self.validation = self.workers / self.dataset / 'validation.csv'
        self.sample = self.samples / self.dataset / 'sample-101-1n.csv'
        self.validation.parent.mkdir(parents=True)
        self.sample.parent.mkdir(parents=True)
        self.validation.write_bytes(b'0,0\n1,1\n2,2\n3,3\n')
        self.sample.write_bytes(b'x,y\n0,0\n1,1\n2,2\n3,3\n')
        self.implementation = self.root / 'toy_expanded.py'
        self.implementation.write_bytes(b"def c2st(a,b,c):\n return {'status':'ok','value':0.5,'rows_per_class':4}\ndef alpha_beta(a,b):\n return {'status':'ok','value':{'alpha_precision':0.2,'beta_recall':0.3}}\n")
        self.cell = {'dataset': self.dataset, 'size': 1, 'sample_seed': 101,
                     'synthetic_sha256': hashlib.sha256(self.sample.read_bytes()).hexdigest(),
                     'rows': {'validation': 4, 'synthetic': 4}}
        self.declaration = {'validation_sha256_by_dataset': {
            self.dataset: hashlib.sha256(self.validation.read_bytes()).hexdigest()},
            'expanded_implementation_sha256': hashlib.sha256(self.implementation.read_bytes()).hexdigest()}
        self.patches = [patch.object(producer, 'WORKERS', self.workers),
                        patch.object(producer, 'SAMPLES', self.samples)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def score(self):
        return producer.score_cell((self.cell, self.declaration, self.implementation))

    def test_authenticated_positive_no_real_model(self):
        result = self.score()
        self.assertEqual((result['marginal_ks_mean'], result['pair_correlation_fidelity']), (0., 1.))
        self.assertEqual((result['c2st_status'], result['c2st_catboost_auc'], result['c2st_rows_per_class']), ('ok', .5, 4))
        self.assertEqual((result['alpha_beta_status'], result['alpha_precision'], result['beta_recall']), ('ok', .2, .3))

    def test_unavailable_diagnostics_remain_null_with_reason(self):
        raw = b"def c2st(a,b,c):\n return {'status':'unavailable','value':None,'reason':'toy_group_absence'}\ndef alpha_beta(a,b):\n return {'status':'unavailable','value':None,'reason':'negative_author_alpha_beta_delta'}\n"
        self.implementation.write_bytes(raw)
        self.declaration['expanded_implementation_sha256'] = hashlib.sha256(raw).hexdigest()
        result = self.score()
        self.assertEqual(result['c2st_status'], 'unavailable')
        self.assertEqual(result['c2st_reason'], 'toy_group_absence')
        self.assertIsNone(result['c2st_catboost_auc'])
        self.assertIsNone(result['c2st_rows_per_class'])
        self.assertEqual(result['alpha_beta_status'], 'unavailable')
        self.assertIsNone(result['alpha_precision'])
        self.assertIsNone(result['beta_recall'])
        self.assertEqual(result['alpha_beta_reason'], 'negative_author_alpha_beta_delta')

    def test_validation_substitution_before_decode(self):
        self.validation.write_bytes(b'opaque replacement, not rows')
        with patch.object(np, 'loadtxt', side_effect=AssertionError('decode before auth')) as decode:
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                self.score()
        decode.assert_not_called()

    def test_sample_substitution_before_sample_decode(self):
        self.sample.write_bytes(b'opaque replacement, not rows')
        with patch.object(np, 'loadtxt', return_value=np.zeros((4, 2))) as decode:
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                self.score()
        self.assertEqual(decode.call_count, 1)  # Only authenticated toy validation.

    def test_implementation_substitution_refused(self):
        self.implementation.write_bytes(b'raise AssertionError("untrusted code")\n')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.score()

    def test_captured_implementation_does_not_reread_path(self):
        verify = producer.authenticated

        def capture_then_replace(path, expected):
            raw = verify(path, expected)
            if path == self.implementation:
                path.write_bytes(b'raise AssertionError("late substituted code")\n')
            return raw

        with patch.object(producer, 'authenticated', side_effect=capture_then_replace):
            result = self.score()
        self.assertEqual(result['c2st_catboost_auc'], .5)

    def test_bound_row_count_mismatch_refused(self):
        self.cell['rows']['validation'] = 3
        with self.assertRaisesRegex(ValueError, 'row counts differ'):
            self.score()

    def test_partition_ancestor_symlink_refused_before_decode(self):
        alias = self.workers / self.dataset
        moved = self.root / 'outside-worker'
        alias.rename(moved)
        alias.symlink_to(moved, target_is_directory=True)
        with patch.object(np, 'loadtxt', side_effect=AssertionError('redirected read')) as decode:
            with self.assertRaisesRegex(ValueError, 'redirected fidelity partition'):
                self.score()
        decode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
