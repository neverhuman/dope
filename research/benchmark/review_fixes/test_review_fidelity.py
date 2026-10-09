"""Scalar fidelity status/cohort negatives; no rows, models, hosts or jobs."""
import hashlib
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from docs.whitepaper.scripts import review_fidelity as review


class FidelityReducerTests(unittest.TestCase):
    def setUp(self):
        datasets = ['aa00000000000001', 'bb00000000000002']
        self.utility, self.diagnostics = [], []
        self.docs = {
            review.RECEIPTS: {'fidelity_stored': []},  # Historical format has no official flag.
            review.PRIVACY: {'summaries': []},  # The pinned legacy aggregate lacks this flag too.
            review.EXPANDED: {'official_tests_opened': False, 'cells': []},
            review.FOLLOWON: {'official_tests_opened': False, 'rows': []},
            review.RETAINED: {'official_tests_opened': False, 'cells': []},
            review.NAMES: {'official_tests_opened': False, 'rows': [
                {'dataset': d, 'display_name': 'plain_' + d} for d in datasets]},
        }
        for d in datasets:
            for size in (1, 4):
                for seed in (101, 211, 307):
                    base = {'dataset': d, 'size': size, 'sample_seed': seed,
                            'official_tests_opened': False, 'formal_dp': False,
                            'status': 'ok', 'synthetic_sha256': 'a' * 64}
                    self.utility.append({**base, 'rows': {'validation': 5, 'synthetic': size * 20, 'train': 20}})
                    self.diagnostics.append({
                        **base, 'validation_sha256': 'b' * 64,
                        'marginal_ks_mean': .2, 'pair_correlation_fidelity': .8,
                        'c2st_catboost_auc': .5, 'c2st_status': 'ok',
                        'c2st_rows_per_class': 5, 'c2st_reason': None,
                        'alpha_beta_status': 'ok', 'alpha_beta_reason': None,
                        'alpha_precision': .3, 'beta_recall': .4})
        self.declaration = {'official_tests_opened': False, 'formal_dp': False,
                            'expanded_implementation_sha256': 'e' * 64,
                            'runtime': {'numpy': 'toy-version'},
                            'validation_sha256_by_dataset': {d: 'b' * 64 for d in datasets}}

    def inputs(self):
        data = {name: json.dumps(doc, sort_keys=True).encode() for name, doc in self.docs.items()}
        data[review.TABSYN] = b''.join(json.dumps(c, sort_keys=True).encode() + b'\n' for c in self.utility)
        data[review.DIAGNOSTICS] = b''.join(json.dumps(c, sort_keys=True).encode() + b'\n' for c in self.diagnostics)
        declaration = {**self.declaration,
                       'scalar_ledger_sha256': hashlib.sha256(data[review.TABSYN]).hexdigest(),
                       'validation_pin_source': {'sha256': hashlib.sha256(data[review.FOLLOWON]).hexdigest()}}
        data[review.DECLARATION] = json.dumps(declaration, sort_keys=True).encode()
        data[review.META] = json.dumps({
            'format': 'dope-wave2-tabsyn-fidelity-cells-v1',
            'official_tests_opened': False, 'formal_dp': False,
            'cell_count': len(self.diagnostics), 'producer_sha256': review.PRODUCER_SHA256,
            'declaration_sha256': hashlib.sha256(data[review.DECLARATION]).hexdigest(),
            'ledger_sha256': hashlib.sha256(data[review.DIAGNOSTICS]).hexdigest(),
            'expanded_implementation_sha256': declaration['expanded_implementation_sha256'],
            'numpy': 'toy-version'}).encode()
        return data

    def test_positive_full_scalar_grid_and_legacy_receipt_format(self):
        result = review.payload_from_inputs(self.inputs())
        self.assertEqual([r['n'] for r in result['fidelity']], [2] * 4)
        self.assertEqual([r['n'] for r in result['detectors']], [2] * 2)
        self.assertEqual([r['n'] for r in result['alpha_beta']], [2] * 4)
        fidelity, alpha = review.render(result)
        self.assertIn('Logistic AUC & CatBoost AUC', fidelity)
        self.assertIn('common numeric', alpha)

    def test_wrong_detector_class_n_refused(self):
        self.diagnostics[0]['c2st_rows_per_class'] = 6
        with self.assertRaisesRegex(ValueError, 'class-n differs'):
            review.payload_from_inputs(self.inputs())

    def test_wrong_sample_and_validation_hash_refused(self):
        for key in ('synthetic_sha256', 'validation_sha256'):
            with self.subTest(key=key):
                previous = self.diagnostics[0][key]
                self.diagnostics[0][key] = 'f' * 64
                with self.assertRaisesRegex(ValueError, 'hash differs'):
                    review.payload_from_inputs(self.inputs())
                self.diagnostics[0][key] = previous

    def test_missing_or_duplicate_diagnostic_cell_refused(self):
        for cells in (self.diagnostics[:-1], self.diagnostics[:-1] + [self.diagnostics[0]]):
            saved = self.diagnostics
            self.diagnostics = cells
            with self.assertRaisesRegex(ValueError, 'cohort differs|identity is duplicated'):
                review.payload_from_inputs(self.inputs())
            self.diagnostics = saved

    def test_unavailable_detector_with_a_value_refused(self):
        self.diagnostics[0].update(c2st_status='unavailable', c2st_reason='toy_no_groups')
        with self.assertRaisesRegex(ValueError, 'unavailable diagnostic contains a result'):
            review.payload_from_inputs(self.inputs())

    def test_unavailable_alpha_with_a_value_refused(self):
        self.diagnostics[0].update(alpha_beta_status='unavailable', alpha_beta_reason='negative_author_alpha_beta_delta')
        with self.assertRaisesRegex(ValueError, 'unavailable diagnostic contains a result'):
            review.payload_from_inputs(self.inputs())

    def test_all_unavailable_detector_remains_null_summary(self):
        for c in self.diagnostics:
            c.update(c2st_status='unavailable', c2st_reason='toy_no_groups',
                     c2st_catboost_auc=None, c2st_rows_per_class=None)
        result = review.payload_from_inputs(self.inputs())
        self.assertEqual([r['n'] for r in result['detectors']], [0, 0])
        self.assertTrue(all(r['median'] is None for r in result['detectors']))

    def test_all_unavailable_alpha_preserves_planned_cohort_rows(self):
        for c in self.diagnostics:
            c.update(alpha_beta_status='unavailable', alpha_beta_reason='negative_author_alpha_beta_delta',
                     alpha_precision=None, beta_recall=None)
        result = review.payload_from_inputs(self.inputs())
        self.assertEqual(len(result['alpha_beta']), 4)
        self.assertEqual([r['n'] for r in result['alpha_beta']], [0] * 4)
        self.assertTrue(all(r['median'] is None for r in result['alpha_beta']))
        _, table = review.render(result)
        self.assertEqual(table.count('n=0'), 4)

    def test_logistic_and_grouped_catboost_results_remain_distinct(self):
        self.docs[review.RECEIPTS]['fidelity_stored'] = [{
            'method': 'DOPE', 'configuration': 'features12_steps2048', 'size': 1,
            'metric': 'c2st', 'n': 2, 'median': .1, 'lo': .08, 'hi': .12}]
        result = review.payload_from_inputs(self.inputs())
        table, _ = review.render(result)
        self.assertIn('Logistic AUC & CatBoost AUC', table)
        self.assertIn('0.1000', table)
        self.assertIn('0.5000', table)

    def test_retained_subset_and_common_numeric_alpha_cohorts_remain_distinct(self):
        self.docs[review.RETAINED]['cells'] = [{
            'dataset': c['dataset'], 'sample_seed': c['sample_seed'],
            'row_multiplier': 1, 'fit_seed': 11,
            'metrics': {'alpha_beta': {'status': 'ok', 'value': {
                'alpha_precision': .7, 'beta_recall': .8}}}}
            for c in self.diagnostics if c['size'] == 1]
        result = review.payload_from_inputs(self.inputs())
        common = [r for r in result['alpha_beta'] if r['cohort'] == 'common numeric']
        retained = [r for r in result['alpha_beta'] if r['cohort'] == 'retained subset']
        self.assertEqual(len(common), 4)
        self.assertEqual(len(retained), 2)
        self.assertEqual([r['median'] for r in retained], [.7, .8])
        _, table = review.render(result)
        self.assertIn('common numeric', table)
        self.assertIn('retained subset', table)

    def test_producer_runtime_and_metadata_cannot_be_replaced(self):
        for field, value in [('producer_sha256', 'f' * 64), ('numpy', 'changed'),
                             ('official_tests_opened', 0), ('ledger_sha256', 'f' * 64),
                             ('cell_count', True)]:
            data = self.inputs(); meta = json.loads(data[review.META]); meta[field] = value
            data[review.META] = json.dumps(meta).encode()
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                review.payload_from_inputs(data)

    def test_changed_input_refuses_before_decode_or_panel_write(self):
        target = Path(__file__).resolve().parents[3] / 'target/fidelity-input-tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory); data = self.inputs()
            for name, raw in data.items():
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}
            (root / review.DIAGNOSTICS).write_bytes(b'changed source')
            with patch.object(review, 'REPO', root), patch.object(review, 'SOURCE_SHA256', pins), \
                    patch('sys.argv', ['review_fidelity.py', '--write-panel']), patch.object(review.json, 'loads') as decode:
                with self.assertRaisesRegex(ValueError, 'digest differs before decoding'):
                    review.main()
                decode.assert_not_called()
            self.assertFalse((root / review.PANEL).exists())



if __name__ == "__main__":
    unittest.main()
