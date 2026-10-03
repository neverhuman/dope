"""Synthetic metadata controls only; no measurements, rows, models or ML."""
from pathlib import Path
import copy
import json
import unittest

from research.benchmark.manifest import digest
from research.benchmark import sdv_native_continuation as m
from unittest.mock import patch
import tempfile


def fixture():
    workers = {f'opaque-{i:03d}': {'path': f'/opaque-workers/{i:03d}',
        'files': {'validation.csv': 'a'*64}} for i in range(100)}
    objective = {'name': 'sdmetrics_mean_regression_r2', 'direction': 'maximize',
                 'implementation_sha256': 'b'*64}
    configurations = [{'epochs': 300, 'embedding_dim': width} for width in (128, 64, 256, 512)]
    lock = {'trial_cap_per_method_dataset': 8,
            'total_seconds_cap_per_method_dataset': 43200,
            'whole_operation_timeout_seconds': 600,
            'adapter_sha256': 'b'*64, 'native_metric_seed': 1729,
            'native_objectives': {method: {'default_config': configurations[0],
                'configurations': copy.deepcopy(configurations), 'native_objective': objective}
                for method in ('CTGAN', 'TVAE')},
            'jobs': [], 'logical_trials': [], 'previous_failed_trials': []}
    for i, (dataset, worker) in enumerate(workers.items()):
        for method in ('CTGAN', 'TVAE'):
            for trial in range(4):
                identity = {'dataset': dataset, 'method': method, 'trial_index': trial}
                config = configurations[trial]
                if i < 12:
                    logical = identity | {'kind': 'immutable_prior_native_trial_reuse', 'config': config}
                else:
                    job = identity | {'config': config, 'worker': worker}
                    lock['jobs'].append(job)
                    logical = identity | {'kind': 'new_native_trial', 'physical_job_sha256': digest(job)}
                lock['logical_trials'].append(logical)
    for trial in range(2):
        job = next(j for j in lock['jobs'] if j['dataset'] == 'opaque-012'
                   and j['method'] == 'CTGAN' and j['trial_index'] == trial)
        lock['previous_failed_trials'].append({'dataset': 'opaque-012', 'method': 'CTGAN',
            'trial_index': trial, 'source_job_sha256': digest(job),
            'failed_operation_seconds': 300.0 + trial})
    return lock, workers


class ContinuationFixture:
    def setUp(self):
        self.lock, self.workers = fixture()
        jobs = {digest(job): job for job in self.lock['jobs']}
        rows = []
        for item in self.lock['logical_trials']:
            is_new = item['kind'] == 'new_native_trial'
            job = jobs[item['physical_job_sha256']] if is_new else item
            dataset, method = item['dataset'], item['method']
            metric = {'objective': 'sdmetrics_mean_regression_r2', 'direction': 'maximize',
                'partition': 'validation', 'implementation_sha256': self.lock['adapter_sha256'],
                'validation_sha256': self.workers[dataset]['files']['validation.csv'], 'seed': 1729,
                'components': {'LinearRegression': 0.25, 'MLPRegressor': 0.75}, 'value': 0.5}
            rows.append({'dataset': dataset, 'method': method, 'trial_index': item['trial_index'],
                'config': job['config'], 'kind': item['kind'], 'fit_seed': 11, 'status': 'ok',
                'job_sha256': item.get('physical_job_sha256'), 'operation_seconds': 10.0,
                'complete_fitted_artifact': True, 'native_selection_eligible': True,
                'common_samples_complete': is_new, 'native_kpi': metric,
                'receipt_path': '/opaque-synthetic-fixture/receipt.json', 'receipt_sha256': 'd'*64})
        self.report = {'complete_native_matrix': True, 'datasets': 100, 'new_trials_closed': 704,
            'new_trials_planned': 704, 'logical_tuning_trials': 800, 'prior_trials_reused': 96,
            'official_tests_opened': False, 'shared_kpi_used_for_selection': False,
            'sampling_success_used_for_selection': False, 'mfs_v2': None, 'ptf_v1': None,
            'release_safe': None, 'superiority': None, 'trials': rows,
            'previous_failed_trials': [dict(row, status='failed') for row in self.lock['previous_failed_trials']]}
        self.pair = (self.lock['previous_failed_trials'][0]['dataset'], 'CTGAN')
        self.rows = [r for r in rows if (r['dataset'], r['method']) == self.pair]
        assert len(self.rows) == 4 and all(r['kind'] == 'new_native_trial' for r in self.rows)

    def call(self):
        return m.plan(self.lock, self.report, self.workers)

    def cell(self, plan):
        return next(r for r in plan['method_dataset_cells'] if (r['dataset'], r['method']) == self.pair)

    def failed_native(self, fitted=False):
        for row in self.rows:
            row.update(status='failed', complete_fitted_artifact=fitted, native_selection_eligible=False,
                       common_samples_complete=False, native_kpi=None)



class ContinuationControls(ContinuationFixture, unittest.TestCase):
    def test_complete_synthetic_ledger_stays_unadmitted_and_preserves_all_groups(self):
        result = self.call()
        self.assertEqual(len(result['method_dataset_cells']), 200)
        self.assertEqual(result['datasets'], 100)
        self.assertFalse(result['execution_admitted'])
        self.assertEqual(result['physical_jobs_launched'], 0)
        self.assertIsNone(result['mfs_v2'])

    def test_incomplete_preview_rejects_before_missing_trial_field_read(self):
        preview = {'complete_native_matrix': False}
        with self.assertRaisesRegex(ValueError, 'complete predecessor'): m.plan(self.lock, preview, self.workers)

    def test_missing_trial_rejects(self):
        self.report['trials'].pop()
        with self.assertRaises(ValueError): self.call()

    def test_duplicate_trial_rejects(self):
        self.report['trials'][-1] = copy.deepcopy(self.report['trials'][0])
        with self.assertRaises(ValueError): self.call()

    def test_old_failures_count_toward_eight_and_default_first_order(self):
        self.failed_native()
        self.report['trials'].reverse()
        cell = self.cell(self.call())
        self.assertEqual(cell['previous_attempts_charged'], 6)
        self.assertEqual([r['trial_index'] for r in cell['native_retry_previews']], [0, 1])
        self.assertEqual([r['trial_index'] for r in cell['budget_unavailable_trials']], [2, 3])
        self.assertEqual(cell['maximum_additional_seconds_reserved'], 1200)
        self.assertTrue(all(r['new_fit_needed'] for r in cell['native_retry_previews']))

    def test_retry_preserves_full_fit_identity_and_prior_receipt(self):
        self.failed_native()
        for retry in self.cell(self.call())['native_retry_previews']:
            row = next(r for r in self.rows if r['trial_index'] == retry['trial_index'])
            self.assertEqual(retry['fit_job_sha256'], row['job_sha256'])
            self.assertEqual(retry['prior_receipt_sha256'], row['receipt_sha256'])
            self.assertEqual(retry['continuation_attempt'], 2)

    def test_complete_fit_reused_for_native_only_retry(self):
        self.failed_native(fitted=True)
        retries = self.cell(self.call())['native_retry_previews']
        self.assertTrue(all(not r['new_fit_needed'] for r in retries))
        self.assertTrue(all(r['operation'] == 'native_on_complete_fitted_artifact' for r in retries))

    def test_sampling_failure_keeps_native_eligibility_without_refit(self):
        for row in self.rows: row.update(status='failed', common_samples_complete=False)
        cell = self.cell(self.call())
        self.assertEqual(cell['native_retry_previews'], [])
        self.assertEqual(len(cell['native_evidence_reused_without_refit']), 4)
        self.assertTrue(all(not r['new_fit_needed'] and not r['sampling_success_used_for_selection']
                            for r in cell['native_evidence_reused_without_refit']))

    def test_remaining_cost_under600_blocks_all_retry_operations(self):
        self.failed_native()
        historical = sum(r['failed_operation_seconds'] for r in self.report['previous_failed_trials'])
        for row in self.rows: row['operation_seconds'] = (43200 - historical - 599) / 4
        cell = self.cell(self.call())
        self.assertEqual(cell['native_retry_previews'], [])
        self.assertEqual(len(cell['budget_unavailable_trials']), 4)

    def test_prior_cost_already_over12_hours_rejects(self):
        self.rows[0]['operation_seconds'] = 43200
        with self.assertRaises(ValueError): self.call()

    def test_omitted_historical_failure_rejects(self):
        self.report['previous_failed_trials'].pop()
        with self.assertRaises(ValueError): self.call()

    def test_duplicate_historical_failure_rejects(self):
        self.report['previous_failed_trials'][1] = copy.deepcopy(self.report['previous_failed_trials'][0])
        with self.assertRaises(ValueError): self.call()

    def test_native_kpi_on_partial_fit_rejects(self):
        self.rows[0]['complete_fitted_artifact'] = False
        with self.assertRaises(ValueError): self.call()

    def test_config_drift_rejects(self):
        self.rows[0]['config'] = {'unfrozen': True}
        with self.assertRaises(ValueError): self.call()

    def test_shared_kpi_substitution_rejects(self):
        self.rows[0]['native_kpi']['objective'] = 'mfs_v2'
        with self.assertRaises(ValueError): self.call()

    def test_native_implementation_or_validation_drift_rejects(self):
        self.rows[0]['native_kpi']['implementation_sha256'] = 'e'*64
        with self.assertRaises(ValueError): self.call()

    def test_nan_cost_rejects(self):
        self.rows[0]['operation_seconds'] = float('nan')
        with self.assertRaises(ValueError): self.call()

    def test_positive_gated_score_rejects(self):
        self.report['mfs_v2'] = 0.99
        with self.assertRaises(ValueError): self.call()

    def test_prior_only_lineage_cannot_be_dropped_from_matched_workers(self):
        prior_only = next(r['dataset'] for r in self.report['trials']
                          if r['dataset'] not in {j['dataset'] for j in self.lock['jobs']})
        del self.workers[prior_only]
        with self.assertRaises(ValueError): self.call()

    def test_immutable_prior_native_trial_cannot_be_retrained(self):
        row = next(r for r in self.report['trials'] if r['kind'] == 'immutable_prior_native_trial_reuse')
        row.update(native_selection_eligible=False, native_kpi=None, common_samples_complete=False)
        with self.assertRaises(ValueError): self.call()

    def test_attempt_cap_cannot_be_increased(self):
        self.lock['trial_cap_per_method_dataset'] = 9
        with self.assertRaises(ValueError): self.call()

    def test_whole_operation_cap_cannot_be_increased(self):
        self.lock['whole_operation_timeout_seconds'] = 601
        with self.assertRaises(ValueError): self.call()

    def test_negative_native_r2_remains_unclipped_and_eligible(self):
        self.rows[0]['native_kpi'].update(components={'LinearRegression': -1.0, 'MLPRegressor': -3.0}, value=-2.0)
        cell = self.cell(self.call())
        self.assertEqual(next(r['native_value'] for r in cell['native_evidence_reused_without_refit']
                              if r['trial_index'] == self.rows[0]['trial_index']), -2.0)

    def test_native_only_retries_do_not_reserve_gpu(self):
        self.failed_native(fitted=True)
        self.assertTrue(all(not r['requires_gpu'] and not r['native_scoring_requires_gpu']
                            for r in self.cell(self.call())['native_retry_previews']))

    def test_fit_retries_still_require_gpu_admission(self):
        self.failed_native()
        self.assertTrue(all(r['requires_gpu'] for r in self.cell(self.call())['native_retry_previews']))

    def test_common_sampling_is_cpu_and_native_winner_is_deferred(self):
        for cell in self.call()['method_dataset_cells']:
            self.assertFalse(cell['common_sampling_requires_gpu'])
            self.assertTrue(cell['winner_selection_deferred_until_native_retries_close'])

    def test_default_first_retry_cannot_assume_changed_default(self):
        self.lock['native_objectives']['CTGAN']['default_config'] = {'unfrozen': True}
        with self.assertRaises(ValueError): self.call()

    def test_grid_mismatch_rejects_even_if_physical_identity_matches(self):
        self.lock['native_objectives']['CTGAN']['configurations'][0] = {'unfrozen': True}
        with self.assertRaises(ValueError): self.call()

    def test_boolean_trial_index_rejects(self):
        self.rows[0]['trial_index'] = False
        with self.assertRaises(ValueError): self.call()

    def test_boolean_elapsed_cost_rejects(self):
        self.rows[0]['operation_seconds'] = True
        with self.assertRaises(ValueError): self.call()

    def test_wrong_digest_rejects_before_runtime_or_report_reads(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        with tempfile.TemporaryDirectory(dir=target) as folder:
            root = Path(folder) / 'round'
            root.mkdir()
            (root / 'round.lock.json').write_bytes(b'opaque')
            with patch.object(m.predecessor, 'ROOT', root), patch.object(m.predecessor, 'declared_runtime') as initializer:
                with self.assertRaisesRegex(ValueError, 'digest changed'): m.build('c'*64)
                initializer.assert_not_called()

    def test_unfrozen_digest_rejects_without_file_reads(self):
        with patch.object(m.accounting, 'evidence') as evidence:
            for value in (None, '', 'C'*64, 'c'*63, True):
                with self.assertRaises(ValueError): m.build(value)
            evidence.assert_not_called()


class FrozenCustodyControls(ContinuationFixture, unittest.TestCase):
    """Opaque metadata wrapper fixture; runtime/closure have their own tests."""

    def setUp(self):
        super().setUp()
        target = Path(__file__).resolve().parents[3] / 'target'
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'round'
        self.root.mkdir()
        self.matched = self.base / 'dope-s3-population-research-v2' / 'round.lock.json'
        self.matched.parent.mkdir()
        self.matched.write_text(json.dumps({'jobs': [{'dataset': d, 'worker': w}
                                                    for d, w in self.workers.items()]}))
        self.lock['frozen_references'] = {str(self.matched): m.sha256(self.matched)}
        self.round_path = self.root / 'round.lock.json'
        self.round_path.write_text(json.dumps(self.lock))
        self.round_sha = m.sha256(self.round_path)
        trial_path = self.root / 'opaque-trial-receipt.json'
        trial_path.write_text('{"opaque_synthetic_fixture": true}')
        for row in self.report['trials']:
            row['receipt_path'], row['receipt_sha256'] = str(trial_path), m.sha256(trial_path)
        self.report['round_sha256'] = self.round_sha
        self.report['cost'] = {'coordinator_wall_seconds': 1234.0}
        self.report_path = self.root / 'reconciliation-v1.json'
        self.report_path.write_text(json.dumps(self.report))
        self.anchor = {'format': 'dope-sdv-population-native-receipt-lock', 'version': 1,
            'round_sha256': self.round_sha, 'official_tests_opened': False,
            'mfs_v2': None, 'ptf_v1': None,
            'refs': {str(p): m.sha256(p) for p in (self.round_path, self.matched, trial_path)},
            'reconciliation_sha256': m.sha256(self.report_path)}
        self.anchor_path = self.root / 'receipt-lock-v1.json'
        self.anchor_path.write_text(json.dumps(self.anchor))
        self.anchor_sha = m.sha256(self.anchor_path)

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        with patch.object(m.predecessor, 'ROOT', self.root), \
             patch.object(m.predecessor, 'ROUND_SHA', self.round_sha), \
             patch.object(m.predecessor, 'declared_runtime') as runtime, \
             patch.object(m.accounting, 'worker_files') as workers, \
             patch.object(m.accounting, 'closure', return_value=1234.0) as closure:
            result = m.build(self.anchor_sha)
            runtime.assert_called_once()
            closure.assert_called_once()
            self.assertEqual(workers.call_count, 100)
            return result

    def test_wrapper_keeps_external_anchor_and_all100_workers(self):
        result = self.build()
        self.assertEqual(result['predecessor_receipt_lock_sha256'], self.anchor_sha)
        self.assertEqual(result['predecessor_round_sha256'], self.round_sha)
        self.assertFalse(result['execution_admitted'])

    def test_rewritten_anchor_rejects(self):
        self.anchor['mfs_v2'] = 0.99
        self.anchor_path.write_text(json.dumps(self.anchor))
        with self.assertRaisesRegex(ValueError, 'digest changed'): self.build()

    def test_rewritten_report_rejects_before_planning_or_runtime(self):
        self.report['trials'][0]['native_kpi']['value'] = 0.99
        self.report_path.write_text(json.dumps(self.report))
        with patch.object(m, 'plan') as planner:
            with self.assertRaisesRegex(ValueError, 'digest changed'): self.build()
            planner.assert_not_called()

    def test_missing_matched_anchor_rejects(self):
        self.anchor['refs'].pop(str(self.matched))
        self.anchor_path.write_text(json.dumps(self.anchor))
        self.anchor_sha = m.sha256(self.anchor_path)
        with self.assertRaisesRegex(ValueError, 'matched worker'): self.build()

    def test_trial_receipt_reference_missing_rejects(self):
        self.anchor['refs'].pop(self.report['trials'][0]['receipt_path'])
        self.anchor_path.write_text(json.dumps(self.anchor))
        self.anchor_sha = m.sha256(self.anchor_path)
        with self.assertRaisesRegex(ValueError, 'trial receipt'): self.build()

    def test_referenced_file_drift_rejects_before_report_read(self):
        path = Path(self.report['trials'][0]['receipt_path'])
        path.write_bytes(b'rewritten fixture')
        with self.assertRaisesRegex(ValueError, 'digest changed'): self.build()

    def test_referenced_directory_alias_rejects(self):
        directory = self.matched.parent
        real = self.base / 'real-matched'
        directory.rename(real)
        directory.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'outside worker scope'): self.build()


if __name__ == '__main__': unittest.main()
