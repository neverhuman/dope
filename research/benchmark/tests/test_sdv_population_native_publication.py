"""Opaque metadata and local bytes only; no benchmark data or dependency init."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import publish_sdv_population_native as m
from research.benchmark import publish_sdv_population_native_manifest as manifest
from research.benchmark.tests import test_sdv_native_continuation as fixtures


class PublicationControls(unittest.TestCase):
    def setUp(self):
        f = fixtures.ContinuationFixture()
        f.setUp()
        self.lock, self.report = f.lock, f.report
        for r in self.report['trials']:
            r['artifact_bytes'] = 1000 + r['trial_index']
        self.report['new_trial_status_counts'] = {'ok': 704}
        self.report['cost'] = {
            'new_operation_seconds': 7040.0, 'prior_native_trial_wall_seconds': 960.0,
            'previous_failed_operation_seconds': 601.0, 'coordinator_wall_seconds': 9000.0,
            'energy_attributable_to_job': None}
        self.terminals = {(r['dataset'], r['method'], r['trial_index']):
            'ok' if r['kind'] == 'new_native_trial' else 'immutable_prior_reuse'
            for r in self.report['trials']}
        self.select()

    def select(self):
        self.report['default_native_cells'] = m.accounting.select(self.report['trials'],
            self.lock['native_objectives'], [(f'opaque-{i:03d}', method)
                for i in range(100) for method in ('CTGAN', 'TVAE')], self.report['previous_failed_trials'])
        for r in self.report['previous_failed_trials']:
            r.update(receipt_path='/opaque/failed.json', receipt_sha256='c'*64, partial_artifact_bytes=19)

    def call(self):
        return m.assemble(self.lock, self.report, self.terminals)

    def test_complete_denominator_and_null_claims(self):
        r = self.call()
        self.assertEqual((len(r['trials']), len(r['cells'])), (800, 200))
        self.assertEqual(r['cell_status_counts'], {'selected': 200})
        self.assertFalse(r['production_certified'])
        self.assertIsNone(r['ptf_v1'])
        self.assertEqual(len(r['previous_failed_trials']), 2)

    def test_unstarted_default_stays_missing_and_in_denominator(self):
        r = self.report['trials'][96]
        r.update(status='failed', native_kpi=None, native_selection_eligible=False,
                 common_samples_complete=False, complete_fitted_artifact=False, artifact_bytes=0)
        self.terminals[(r['dataset'], r['method'], r['trial_index'])] = 'deadline_unstarted'
        self.report['new_trial_status_counts'] = {'ok': 703, 'failed': 1}
        self.select()
        out = self.call()
        self.assertEqual(out['new_terminal_operation_status_counts']['deadline_unstarted'], 1)
        cell = next(c for c in out['cells'] if (c['dataset'], c['method']) == (r['dataset'], r['method']))
        self.assertIsNone(cell['default_native_kpi'])
        self.assertEqual(out['within_method_summaries'][0]['paired_default_selected_cells'], 99)

    def test_post_native_sample_failure_remains_selection_eligible(self):
        r = self.report['trials'][96]
        r.update(status='failed', common_samples_complete=False)
        self.report['new_trial_status_counts'] = {'ok': 703, 'failed': 1}
        self.terminals[(r['dataset'], r['method'], r['trial_index'])] = 'transport_or_prelaunch_failure'
        self.select()
        self.assertEqual(self.call()['cell_status_counts'], {'selected': 200})

    def test_negative_native_values_not_clipped(self):
        for r in self.report['trials']:
            r['native_kpi'].update(value=-2.0, components={'LinearRegression': -1.0, 'MLPRegressor': -3.0})
        self.select()
        self.assertEqual(self.call()['within_method_summaries'][0]['paired_default_median_native_r2'], -2.0)

    def test_config_tie_uses_charged_bytes_without_sampling_preference(self):
        self.assertTrue(all(c['native_selected_trial_index'] == 0 for c in self.call()['cells']))

    def test_stale_selected_trial_rejects(self):
        self.report['default_native_cells'][0]['native_selected_trial']['artifact_bytes'] += 1
        with self.assertRaises(ValueError): self.call()

    def test_missing_or_duplicate_trial_rejects(self):
        original = copy.deepcopy(self.report)
        self.report['trials'].pop()
        with self.assertRaises(ValueError): self.call()
        self.report = original
        self.report['trials'][-1] = copy.deepcopy(self.report['trials'][0])
        with self.assertRaises(ValueError): self.call()

    def test_incomplete_ledger_rejects(self):
        self.report['new_trials_closed'] = 703
        with self.assertRaises(ValueError): self.call()

    def test_cost_undercharge_rejects(self):
        self.report['cost']['new_operation_seconds'] -= 1
        with self.assertRaises(ValueError): self.call()

    def test_shared_score_or_sampling_selection_claim_rejects(self):
        for key, value in (('mfs_v2', .99), ('shared_kpi_used_for_selection', True),
                           ('sampling_success_used_for_selection', True)):
            original = self.report[key]
            self.report[key] = value
            with self.assertRaises(ValueError): self.call()
            self.report[key] = original

    def test_tables_regenerate_all_cells_and_trial_zero(self):
        r = self.call()
        self.assertEqual(m.table(r), m.table(json.loads(json.dumps(r))))
        self.assertEqual(len(m.table(r).splitlines()), 201)
        self.assertIn(',0,1000,1000,', m.table(r))

    def test_digest_subclass_rejected_before_verifier_or_file_reads(self):
        class Forged(str):
            pass
        with patch.object(m, 'bound_json') as read, patch.object(m.continuation, 'build') as init:
            for values in ((Forged('a'*64), 'b'*64), ('a'*64, Forged('b'*64)), ('A'*64, 'b'*64)):
                with self.assertRaises(ValueError): m.build(*values)
            read.assert_not_called()
            init.assert_not_called()

    def test_wrong_anchor_rejects_before_runtime(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        with tempfile.TemporaryDirectory(dir=target) as temp:
            root = Path(temp) / 'round'
            root.mkdir()
            (root / 'receipt-lock-v1.json').write_text('{}')
            with patch.object(m.predecessor, 'ROOT', root), patch.object(m.continuation, 'build') as init:
                with self.assertRaisesRegex(ValueError, 'reference changed'): m.build('a'*64, 'b'*64)
                init.assert_not_called()

    def test_bound_json_uses_digest_of_parsed_bytes_and_rejects_aliases(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        with tempfile.TemporaryDirectory(dir=target) as temp:
            root = Path(temp) / 'round'
            root.mkdir()
            p = root / 'owned.json'
            p.write_bytes(b'{"opaque": true}')
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(m.predecessor, 'ROOT', root):
                self.assertEqual(m.bound_json(p, h), {'opaque': True})
                alias = root / 'alias.json'
                alias.symlink_to(p)
                with self.assertRaises(ValueError): m.bound_json(alias, h)

    def test_offline_regeneration_rejects_wrong_digest_before_rendering(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        with tempfile.TemporaryDirectory(dir=target) as temp:
            p = Path(temp) / (m.NAME + '.json')
            p.write_text('{}')
            with patch.object(manifest.figure, 'render') as render:
                with self.assertRaisesRegex(ValueError, 'digest differs'):
                    manifest.generate(Path(temp), 'a'*64)
                render.assert_not_called()


if __name__ == '__main__':
    unittest.main()
