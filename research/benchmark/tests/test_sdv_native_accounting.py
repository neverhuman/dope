"""Hermetic native selection and failed-operation accounting controls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import sdv_native_accounting as m
from research.benchmark import reconcile_sdv_population as reconcile


class NativeAccounting(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        target.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'round'
        self.worker = self.base / 'worker'
        self.worker.mkdir()
        for name in ('train.csv', 'validation.csv', 'projection.json', 'row-group-assignments.json', 'worker-manifest.json'):
            (self.worker / name).write_bytes(b'opaque fixture')
        objective = {'name': 'sdmetrics_mean_regression_r2', 'direction': 'maximize',
                     'implementation_sha256': 'adapter'}
        self.objective = objective
        self.job = {'dataset': 'fixture', 'method': 'CTGAN', 'trial_index': 0, 'config': {'embedding_dim': 128},
                    'final': False, 'fit_seed': 11, 'track': 'common-numeric', 'native_objective': objective,
                    'worker': {'path': str(self.worker), 'train_rows': 80,
                               'files': {p.name: m.sha256(p) for p in self.worker.iterdir()}}}
        self.lock = {'round_sha256': 'round', 'allowed_gpu_hosts': ['xbabe1'], 'whole_operation_timeout_seconds': 600,
                     'gpu_vram_limit_mib': 16384, 'adapter_sha256': 'adapter', 'native_metric_seed': 1729,
                     'cpu_slots': {'xbabe1': list(range(16, 32))},
                     'roles': {'gpu_worker': {'ram_bytes': 24 * 2**30, 'scratch_bytes': 5000000000}},
                     'common_sample_schedule': [{'name': n, 'multiplier': z, 'seed': s} for n, z, s in
                         [('n-seed101', 1, 101), ('n-seed211', 1, 211), ('n-seed307', 1, 307),
                          ('4n-seed101', 4, 101), ('4n-seed211', 4, 211), ('4n-seed307', 4, 307), ('repeat', 1, 101)]]}
        self.parent = self.root / 'attempts' / m.digest(self.job) / 'attempt-0001'
        self.parent.mkdir(parents=True)
        artifact = self.parent / 'artifact'
        artifact.mkdir()
        for name in ('model.json', 'model.pt', 'projection.json'):
            (artifact / name).write_bytes(b'opaque fixture')
        self.flags = {'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None}
        self.identity = self.flags | {'status': 'ok', 'job_sha256': m.digest(self.job),
                                     'round_sha256': 'round', 'host': 'xbabe1'}
        inventory = [{'path': p.name, 'sha256': m.sha256(p), 'bytes': p.stat().st_size} for p in sorted(artifact.iterdir())]
        self.put('fit.json', self.identity | {'gpu_fit_required': True, 'source_adapter_unchanged': True,
             'projection_bytes_included': True, 'training_rows': 80, 'peak_torch_allocated_bytes': 100,
             'peak_torch_reserved_bytes': 200, 'artifact_inventory': inventory, 'artifact_bytes': sum(r['bytes'] for r in inventory)})
        (self.parent / 'native-sample.csv').write_bytes(b'opaque sample')
        self.metric = self.identity | {'objective': objective['name'], 'name': objective['name'],
             'direction': 'maximize', 'implementation_sha256': 'adapter', 'partition': 'validation', 'seed': 1729,
             'validation_sha256': self.job['worker']['files']['validation.csv'],
             'synthetic_sha256': m.sha256(self.parent / 'native-sample.csv'),
             'value': -0.5, 'components': {'LinearRegression': -0.25, 'MLPRegressor': -0.75},
             'shared_kpi_used_for_selection': False}
        self.put('native.json', self.metric)
        samples = []
        for schedule in self.lock['common_sample_schedule']:
            name = schedule['name']
            (self.parent / (name + '.csv')).write_bytes(b'opaque sample')
            h = m.sha256(self.parent / (name + '.csv'))
            sample = self.identity | {'schedule': schedule, 'sha256': h, 'sample_sha256': h,
                                     'rows': 80 * schedule['multiplier']}
            self.put(name + '.json', sample)
            samples.append(sample)
        self.put('sample.json', self.identity | {'samples': samples, 'sample_replay_exact': True})
        self.operations = []
        for mode in ('fit', 'native', 'sample'):
            self.put(mode + '.request.json', {'job': self.job, 'artifact': str(artifact),
                                               'receipt': str(self.parent / (mode + '.json'))}
                     | ({'schedule': self.lock['common_sample_schedule']} if mode == 'sample' else {}))
            (self.parent / (mode + '.worker.log')).write_bytes(b'opaque worker log')
            (self.parent / (mode + '.transport.log')).write_bytes(b'opaque transport log')
            request_sha = m.sha256(self.parent / (mode + '.request.json'))
            self.put(mode + '.xbabe1.admission-0001.json', {'admitted': True, 'blockers': [],
                'round_sha256': 'round', 'host': 'xbabe1', 'official_tests_opened': False,
                'requested': self.lock['roles']['gpu_worker'] | {'host': 'xbabe1', 'requires_gpu': True,
                                                               'cpu_slot': self.lock['cpu_slots']['xbabe1']}})
            self.put(mode + '.monitor.json', self.flags | {'operation': mode, 'status': 'ok', 'exit_code': 0,
                'round_sha256': 'round', 'host': 'xbabe1', 'foreign_processes_signaled': False,
                'energy_attributable_to_job': None, 'request_sha256': request_sha,
                'worker_log_sha256': m.sha256(self.parent / (mode + '.worker.log')),
                'timeout_seconds': 600, 'elapsed_seconds': 12, 'peak_gpu_used_mib': 1000,
                'peak_resident_bytes': 2**30, 'gpu_process_observed': True})
            op = {'status': 'ok', 'exit_code': 0, 'new_operation_started': True, 'elapsed_seconds': 14,
                'host': 'xbabe1', 'timeout_seconds': 600, 'request_sha256': request_sha, 'official_tests_opened': False,
                'transport_log_sha256': m.sha256(self.parent / (mode + '.transport.log')),
                'admission_sha256': m.sha256(self.parent / (mode + '.xbabe1.admission-0001.json'))}
            self.put(mode + '.operation.json', op)
            self.operations.append(op)
        self.receipt = self.flags | {'job': self.job, 'round_sha256': 'round', 'attempt': 1, 'status': 'ok',
                                    'counts_as_dope_win': False, 'operations': self.operations}
        self.refresh()

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, value):
        (self.parent / name).write_text(json.dumps(value, sort_keys=True))

    def refresh(self):
        self.receipt['evidence_files'] = {p.name: m.sha256(p) for p in self.parent.iterdir()
                                        if p.is_file() and p.name != 'receipt.json'}
        self.put('receipt.json', self.receipt)

    def call(self):
        return m.trial(self.job, self.root, self.lock, {})

    def test_success_preserves_negative_kpi_and_complete_byte_charge(self):
        row = self.call()
        self.assertEqual(row['native_kpi']['value'], -0.5)
        self.assertEqual(row['operation_seconds'], 42)
        self.assertTrue(row['common_samples_complete'])

    def test_drift_or_added_artifact_alias_rejected(self):
        (self.parent / 'native.json').write_text('{}')
        with self.assertRaises(ValueError): self.call()
        self.put('native.json', self.metric)
        self.refresh()
        (self.parent / 'artifact' / 'alias').symlink_to(self.worker, target_is_directory=True)
        with self.assertRaises(ValueError): self.call()

    def test_failed_sample_keeps_native_selection_evidence_and_cost(self):
        self.operations[2]['status'] = 'timeout'
        self.operations[2]['exit_code'] = 1
        self.operations[2]['elapsed_seconds'] = 602
        self.put('sample.operation.json', self.operations[2])
        monitor = m.read(self.parent / 'sample.monitor.json')
        monitor.update(status='timeout', exit_code=-15, elapsed_seconds=601)
        self.put('sample.monitor.json', monitor)
        self.receipt['status'] = 'failed'
        self.refresh()
        row = self.call()
        self.assertTrue(row['native_selection_eligible'])
        self.assertFalse(row['common_samples_complete'])
        self.assertEqual(row['operation_seconds'], 630)

    def test_understated_transport_cost_rejected_for_success_and_timeout(self):
        for index, mode in enumerate(('fit', 'native', 'sample')):
            original = self.operations[index].copy()
            self.operations[index]['elapsed_seconds'] = 0
            self.put(mode + '.operation.json', self.operations[index])
            self.refresh()
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'undercharges'):
                self.call()
            self.operations[index] = original
            self.put(mode + '.operation.json', original)
        self.operations[2].update(status='timeout', exit_code=1, elapsed_seconds=0)
        self.put('sample.operation.json', self.operations[2])
        monitor = m.read(self.parent / 'sample.monitor.json')
        self.put('sample.monitor.json', monitor | {'status': 'timeout', 'exit_code': -15, 'elapsed_seconds': 601})
        self.receipt['status'] = 'failed'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'undercharges'): self.call()

    def test_invalid_failed_monitor_cost_rejected_and_clock_tolerance_never_undercharges(self):
        self.operations[2].update(status='failed', exit_code=1)
        self.put('sample.operation.json', self.operations[2])
        monitor = m.read(self.parent / 'sample.monitor.json') | {'status': 'failed', 'exit_code': 1}
        self.receipt['status'] = 'failed'
        for value in (True, -1, float('nan'), float('inf')):
            self.put('sample.monitor.json', monitor | {'elapsed_seconds': value})
            self.refresh()
            with self.subTest(value=value), self.assertRaises(ValueError): self.call()
        measured = self.operations[2]['elapsed_seconds'] + m.ELAPSED_TOLERANCE_SECONDS / 2
        self.put('sample.monitor.json', monitor | {'elapsed_seconds': measured})
        self.refresh()
        self.assertEqual(self.call()['operation_seconds'], 28 + measured)

    def test_success_rejects_failed_admission_quota_and_gpu_receipts(self):
        for mode, field, value in [('fit', 'gpu_process_observed', False), ('fit', 'peak_gpu_used_mib', 16385),
                                   ('native', 'peak_resident_bytes', 25 * 2**30), ('sample', 'elapsed_seconds', 601)]:
            name = mode + '.monitor.json'
            original = m.read(self.parent / name)
            self.put(name, original | {field: value})
            self.refresh()
            with self.subTest(field=field), self.assertRaises(ValueError): self.call()
            self.put(name, original)
        admission = m.read(self.parent / 'fit.xbabe1.admission-0001.json')
        self.put('fit.xbabe1.admission-0001.json', admission | {'admitted': False})
        self.operations[0]['admission_sha256'] = m.sha256(self.parent / 'fit.xbabe1.admission-0001.json')
        self.put('fit.operation.json', self.operations[0])
        self.refresh()
        with self.assertRaises(ValueError): self.call()

    def test_invalid_native_aggregation_bool_and_shared_selection_rejected(self):
        for change in [{'value': True}, {'value': float('nan')}, {'value': 0},
                       {'shared_kpi_used_for_selection': True}, {'validation_sha256': 'changed'},
                       {'implementation_sha256': 'changed'}]:
            self.put('native.json', self.metric | change)
            self.refresh()
            with self.subTest(change=change), self.assertRaises(ValueError): self.call()

    def test_worker_drift_or_official_test_admission_rejected(self):
        (self.worker / 'train.csv').write_bytes(b'changed fixture')
        with self.assertRaises(ValueError): self.call()
        (self.worker / 'train.csv').write_bytes(b'opaque fixture')
        (self.worker / 'test.csv').write_bytes(b'opaque sealed fixture')
        with self.assertRaises(ValueError): self.call()

    def test_projection_omission_and_uncharged_nested_model_rejected(self):
        fit = m.read(self.parent / 'fit.json')
        self.put('fit.json', fit | {'artifact_bytes': fit['artifact_bytes'] - 1})
        self.refresh()
        with self.assertRaises(ValueError): self.call()
        self.put('fit.json', fit)
        self.refresh()
        (self.parent / 'hidden-model').mkdir()
        (self.parent / 'hidden-model' / 'weight.bin').write_bytes(b'uncharged fixture')
        with self.assertRaises(ValueError): self.call()

    def test_partial_failed_fit_still_charges_its_bytes_and_compute(self):
        for path in self.parent.iterdir():
            if path.is_file() and not path.name.startswith('fit.') and path.name != 'receipt.json':
                path.unlink()
        self.operations[:] = [self.operations[0] | {'status': 'failed', 'exit_code': 1}]
        self.put('fit.operation.json', self.operations[0])
        monitor = m.read(self.parent / 'fit.monitor.json')
        self.put('fit.monitor.json', monitor | {'status': 'failed', 'exit_code': 1})
        self.receipt['status'] = 'failed'
        self.refresh()
        row = self.call()
        self.assertEqual(row['operation_seconds'], 14)
        self.assertEqual(row['artifact_bytes'], 42)
        self.assertEqual(len(row['partial_artifact_inventory']), 3)
        self.assertFalse(row['native_selection_eligible'])

    def test_deadline_unstarted_preserves_native_kpi_without_fake_cost(self):
        for suffix in ('.operation.json', '.monitor.json', '.worker.log', '.json', '.transport.log'):
            (self.parent / ('sample' + suffix)).unlink()
        for row in self.lock['common_sample_schedule']:
            for suffix in ('.csv', '.json'):
                path = self.parent / (row['name'] + suffix)
                if path.exists(): path.unlink()
        self.operations[2] = {'status': 'deadline_unstarted', 'new_operation_started': False, 'elapsed_seconds': 0}
        self.receipt['status'] = 'failed'
        self.refresh()
        row = self.call()
        self.assertEqual(row['operation_seconds'], 28)
        self.assertTrue(row['native_selection_eligible'])
        self.assertFalse(row['common_samples_complete'])

    def test_unstarted_operation_cannot_hide_execution_evidence(self):
        (self.parent / 'sample.operation.json').unlink()
        self.operations[2] = {'status': 'deadline_unstarted', 'new_operation_started': False, 'elapsed_seconds': 0}
        self.receipt['status'] = 'failed'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'execution evidence'): self.call()

    def test_each_phase_owned_output_rejected_for_unstarted_operations(self):
        outputs = {'fit': ['artifact/model.pt', 'artifact/projection.json'],
                   'native': ['native-sample.csv'],
                   'sample': [row['name'] + suffix for row in self.lock['common_sample_schedule']
                              for suffix in ('.csv', '.json')]}
        parent = self.base / 'unstarted-control'
        parent.mkdir()
        row = {'status': 'deadline_unstarted', 'new_operation_started': False, 'elapsed_seconds': 0}
        for mode, names in outputs.items():
            for name in names:
                path = parent / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'opaque execution output')
                with self.subTest(mode=mode, name=name), self.assertRaisesRegex(ValueError, 'execution evidence'):
                    m.operation(mode, row, parent, self.lock, self.base, {})
                path.unlink()
                if path.parent != parent: path.parent.rmdir()
            (parent / (mode + '.request.json')).write_text('{}')
            (parent / (mode + '.xbabe1.admission-0001.json')).write_text('{}')
            self.assertEqual(m.operation(mode, row, parent, self.lock, self.base, {}), 0)

    def test_omitted_later_phase_cannot_hide_execution_outputs(self):
        for mode in ('native', 'sample'):
            (self.parent / (mode + '.operation.json')).unlink()
        self.operations[:] = [self.operations[0] | {'status': 'failed', 'exit_code': 1}]
        self.put('fit.operation.json', self.operations[0])
        self.receipt['status'] = 'failed'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'execution evidence'): self.call()

    def test_clean_closure_requires_the_actual_supervisor_format_and_all_jobs(self):
        row = self.call()
        lock = self.lock | {'jobs': [self.job]}
        (self.root / 'coordinator.log').write_bytes(b'opaque coordinator log')
        values = {'completion.json': self.flags | {'jobs': 1, 'round_sha256': 'round'},
            'coordinator-exit.json': self.flags | {'exit_code': 0, 'round_sha256': 'round',
                'elapsed_seconds': 50, 'log_sha256': m.sha256(self.root / 'coordinator.log')},
            'supervisor-launch.json': {'pid': 99999991, 'supervisor_pid': 99999992, 'round_sha256': 'round'}}
        for name, value in values.items():
            (self.root / name).write_text(json.dumps(value))
        self.assertEqual(m.closure(self.root, lock, [row], {}), 50)
        with self.assertRaises(ValueError): m.closure(self.root, lock, [], {})
        with self.assertRaises(ValueError): m.closure(self.root, lock, [row, row], {})
        for name, change in [('coordinator-exit.json', {'exit_code': -15}),
                             ('completion.json', {'jobs': 2}), ('supervisor-launch.json', {'pid': os.getpid()})]:
            (self.root / name).write_text(json.dumps(values[name] | change))
            with self.subTest(name=name), self.assertRaises(ValueError): m.closure(self.root, lock, [row], {})
            (self.root / name).write_text(json.dumps(values[name]))
        (self.root / 'coordinator.log').write_bytes(b'changed log')
        with self.assertRaises(ValueError): m.closure(self.root, lock, [row], {})

    def test_extra_physical_attempt_rejected_even_with_complete_receipts(self):
        row = self.call()
        lock = self.lock | {'jobs': [self.job]}
        extra = self.parent.parent / 'attempt-0002'
        extra.mkdir()
        (extra / 'receipt.json').write_text(json.dumps(self.receipt))
        (extra / 'partial-model.pt').write_bytes(b'uncharged fixture')
        with self.assertRaisesRegex(ValueError, 'physical attempt'): self.call()
        with self.assertRaisesRegex(ValueError, 'unaccounted attempt'):
            m.closure(self.root, lock, [row], {})

    def test_orphan_job_directory_alias_or_file_rejected(self):
        directory = self.root / 'attempts'
        for kind in ('directory', 'alias', 'file'):
            extra = directory / 'unexpected-job'
            if kind == 'directory': extra.mkdir()
            elif kind == 'alias': extra.symlink_to(self.parent.parent, target_is_directory=True)
            else: extra.write_bytes(b'unaccounted fixture')
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'unexpected jobs'):
                m.attempt_inventory(self.root, [self.job])
            if kind == 'directory': extra.rmdir()
            else: extra.unlink()
        alias = self.parent.parent / 'attempt-alias'
        alias.symlink_to(self.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'unaccounted attempt'):
            m.attempt_inventory(self.root, [self.job], complete=True)

    def test_pending_job_inventory_allowed_only_for_preview(self):
        extra_job = self.job | {'trial_index': 1}
        m.attempt_inventory(self.root, [self.job, extra_job])
        with self.assertRaisesRegex(ValueError, 'inventory'):
            m.attempt_inventory(self.root, [self.job, extra_job], complete=True)
        pending = self.root / 'attempts' / m.digest(extra_job)
        pending.mkdir()
        m.attempt_inventory(self.root, [self.job, extra_job])
        with self.assertRaisesRegex(ValueError, 'unaccounted attempt'):
            m.attempt_inventory(self.root, [self.job, extra_job], complete=True)

    def prior(self):
        path = self.parent / 'historical-receipt.json'
        (self.parent / 'sample.csv').write_bytes(b'opaque sample')
        fit = {'status': 'ok', 'exit_code': 0, 'peak_device_used_mib': 1000, 'wall_seconds': 12,
               'child': {'peak_torch_allocated_bytes': 100, 'peak_torch_reserved_bytes': 200}}
        self.put('native.receipt.json', {'status': 'ok', 'exit_code': 0, 'child': self.metric, 'wall_seconds': 2})
        job = {'dataset': 'fixture', 'method': 'CTGAN', 'trial': 0, 'config': self.job['config'], 'fit_seed': 11,
               'sample_seed': 101, 'size_multiplier': 1, 'train_rows': 80}
        for name, key in [('train.csv', 'train_sha256'), ('validation.csv', 'validation_sha256'),
                          ('projection.json', 'projection_sha256')]:
            job[key] = self.job['worker']['files'][name]
        receipt = {'status': 'ok', 'validation_only': True, 'round_sha256': m.PRIOR_ROUND_SHA,
                   'mfs_v2': None, 'ptf_v1': None, 'job': job, 'fit': fit, 'wall_seconds': 20,
                   'native_kpi': self.metric, 'sample': {'status': 'ok', 'exit_code': 0,
                   'child': {'sha256': self.metric['synthetic_sha256']}},
                   'sampling_repeated_sha256': self.metric['synthetic_sha256'],
                   'artifact_inventory': m.read(self.parent / 'fit.json')['artifact_inventory'], 'artifact_bytes': 42,
                   'evidence_files': {'native.receipt.json': m.sha256(self.parent / 'native.receipt.json')}}
        path.write_text(json.dumps(receipt))
        row = {'dataset': 'fixture', 'method': 'CTGAN', 'trial_index': 0, 'config': self.job['config'],
               'kind': 'immutable_prior_native_trial_reuse', 'historical_runtime_closure_upgraded': False,
               'new_gpu_fit_started': False, 'counts_as_dope_win': False, 'receipt_path': str(path),
               'receipt_sha256': m.sha256(path), 'artifact_path': str(self.parent / 'artifact'),
               'artifact_bytes': 42, 'native_kpi': self.metric}
        lock = self.lock | {'native_objectives': {'CTGAN': {'native_objective': self.objective}}}
        return path, row, lock

    def test_prior_receipt_anchor_and_unupgraded_runtime_scope(self):
        path, row, lock = self.prior()
        verified = m.prior_trial(row, lock, self.job['worker'], self.base, {})
        self.assertEqual(verified['operation_seconds'], 20)
        self.assertFalse(verified['historical_runtime_closure_upgraded'])
        self.assertFalse(verified['common_samples_complete'])
        path.write_text('{}')
        with self.assertRaises(ValueError): m.prior_trial(row, lock, self.job['worker'], self.base, {})

    def test_prior_native_operation_or_wrong_worker_cannot_be_reused(self):
        path, row, lock = self.prior()
        receipt = m.read(path)
        receipt['round_sha256'] = 'wrong round'
        path.write_text(json.dumps(receipt))
        row['receipt_sha256'] = m.sha256(path)
        with self.assertRaises(ValueError): m.prior_trial(row, lock, self.job['worker'], self.base, {})
        receipt['round_sha256'] = m.PRIOR_ROUND_SHA
        self.put('native.receipt.json', {'status': 'failed', 'exit_code': 1, 'child': self.metric, 'wall_seconds': 2})
        receipt['evidence_files']['native.receipt.json'] = m.sha256(self.parent / 'native.receipt.json')
        path.write_text(json.dumps(receipt)); row['receipt_sha256'] = m.sha256(path)
        with self.assertRaises(ValueError): m.prior_trial(row, lock, self.job['worker'], self.base, {})

    def selection(self, rows, failures=None):
        methods = {'CTGAN': {'configurations': [{'embedding_dim': x} for x in (128, 64, 32, 16)],
                             'default_config': {'embedding_dim': 128}}}
        return m.select(rows, methods, [('fixture', 'CTGAN')], failures or [])[0]

    def rows(self):
        row = self.call()
        return [row | {'trial_index': i, 'config': {'embedding_dim': x}, 'artifact_bytes': 100 + i,
                       'native_kpi': row['native_kpi'] | {'value': float(i)}}
                for i, x in enumerate((128, 64, 32, 16))]

    def test_native_winner_independent_of_sampling_success_or_shared_metric(self):
        rows = self.rows()
        rows[3].update(status='failed', common_samples_complete=False, shared_retention=-100)
        selected = self.selection(rows)
        self.assertEqual(selected['native_selected_trial']['trial_index'], 3)
        self.assertEqual(selected['default_trial']['trial_index'], 0)
        self.assertFalse(selected['sampling_success_used_for_selection'])

    def test_incomplete_duplicate_grid_cannot_be_selected(self):
        rows = self.rows()
        self.assertIsNone(self.selection(rows[:3])['native_selected_trial'])
        with self.assertRaises(ValueError): self.selection(rows + [rows[0]])
        rows[0]['config'] = {'embedding_dim': 999}
        with self.assertRaises(ValueError): self.selection(rows)

    def test_failed_trials_and_time_are_charged(self):
        rows = self.rows()
        failure = {'dataset': 'fixture', 'method': 'CTGAN', 'failed_operation_seconds': 623.7}
        selected = self.selection(rows, [failure, failure])
        self.assertEqual(selected['tuning_trials_with_previous_failures'], 6)
        self.assertAlmostEqual(selected['operation_seconds_with_previous_failures'], 1415.4)
        with self.assertRaises(ValueError): self.selection(rows, [failure] * 5)
        rows[0]['operation_seconds'] = 43200
        with self.assertRaises(ValueError): self.selection(rows, [failure])

    def test_ties_follow_bytes_then_config_hash(self):
        rows = self.rows()
        for r in rows: r['native_kpi']['value'] = -0.5
        rows[0]['artifact_bytes'] = 1000
        rows[1]['artifact_bytes'] = rows[2]['artifact_bytes'] = 50
        expected = min(rows[1:3], key=lambda r: m.digest(r['config']))
        self.assertEqual(self.selection(rows)['native_selected_trial']['trial_index'], expected['trial_index'])

    def runtime_fixture(self):
        source = self.root / 'source'
        source.mkdir()
        code = ("from pathlib import Path\n"
                "Path(__file__).parents[1].joinpath('initialized').write_text('fixture')\n"
                "def check(expected): assert expected == 'round'\n"
                "def verify_runtime(lock): return {'files': {}}\n")
        (source / 'common.py').write_text(code)
        lock = {'source_files': {str(source / 'common.py'): m.sha256(source / 'common.py')}}
        return source, lock

    def test_frozen_runtime_source_drift_or_added_bytecode_rejected_before_initializer(self):
        source, lock = self.runtime_fixture()
        with patch.object(reconcile, 'ROOT', self.root), patch.object(reconcile, 'ROUND_SHA', 'round'), \
                patch('sys.dont_write_bytecode', True):
            original = (source / 'common.py').read_text()
            (source / 'common.py').write_text(original + '# drift\n')
            with self.assertRaises(ValueError): reconcile.declared_runtime(lock, {})
            self.assertFalse((self.root / 'initialized').exists())
            (source / 'common.py').write_text(original)
            (source / '__pycache__').mkdir()
            (source / '__pycache__' / 'common.cpython-312.pyc').write_bytes(b'uninventoried cache')
            with self.assertRaises(ValueError): reconcile.declared_runtime(lock, {})
            self.assertFalse((self.root / 'initialized').exists())

    def test_bytecode_guard_rejects_writes_and_positive_probe_leaves_source_unchanged(self):
        source, lock = self.runtime_fixture()
        with patch.object(reconcile, 'ROOT', self.root), patch.object(reconcile, 'ROUND_SHA', 'round'):
            with patch('sys.dont_write_bytecode', False), self.assertRaises(ValueError):
                reconcile.declared_runtime(lock, {})
            self.assertFalse((self.root / 'initialized').exists())
            with patch('sys.dont_write_bytecode', True): reconcile.declared_runtime(lock, {})
            self.assertTrue((self.root / 'initialized').exists())
            self.assertEqual({p.name for p in source.iterdir()}, {'common.py'})

    def test_optimized_interpreters_rejected_before_frozen_initializer(self):
        source, lock = self.runtime_fixture()
        code = """import json, sys
from pathlib import Path
from research.benchmark import reconcile_sdv_population as r
r.ROOT = Path(sys.argv[1])
r.ROUND_SHA = 'round'
try:
    r.declared_runtime(json.loads(sys.argv[2]), {})
except ValueError as exc:
    if str(exc) != 'native closure requires Python optimization disabled':
        raise
else:
    raise RuntimeError('optimized runtime verification accepted')
"""
        (source / 'runtime-drift.pyc').write_bytes(b'uninventoried executable cache')
        for flags, optimization in ((['-O'], ''), (['-OO'], ''), ([], '1'), ([], '2')):
            environment = dict(os.environ, PYTHONOPTIMIZE=optimization)
            result = subprocess.run([sys.executable, '-B', *flags, '-c', code, str(self.root), json.dumps(lock)],
                                    cwd=Path(__file__).resolve().parents[3], env=environment,
                                    capture_output=True, text=True, timeout=30)
            with self.subTest(flags=flags, optimization=optimization):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse((self.root / 'initialized').exists())


if __name__ == '__main__':
    unittest.main()
