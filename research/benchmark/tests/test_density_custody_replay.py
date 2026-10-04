"""Opaque file and timer controls; no models, CSV rows or dependencies execute."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import density_custody_replay as m


class TemporaryEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3] / 'target')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.refs = {}
        self.patcher = patch.object(m.owned, 'BASE', self.base)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        h = hashlib.sha256(data).hexdigest()
        self.refs[str(path)] = h
        return h

    def json(self, path, value):
        return self.put(path, json.dumps(value, sort_keys=True, allow_nan=False).encode())


class NativeFileControls(TemporaryEvidence):
    def setUp(self):
        super().setUp()
        self.root = self.base / 'native'
        self.report = dict(default_native_cells=[], trials=[])
        for i in range(100):
            dataset = f'opaque-{i}'
            worker = self.base / 'workers' / dataset
            projection = dict(task='regression', fit_partition='train', version=1, output_features=12)
            files = {name: self.put(worker / name, b'opaque bytes, never parsed')
                     for name in ('train.csv', 'validation.csv', 'row-group-assignments.json')}
            files['projection.json'] = self.json(worker / 'projection.json', projection)
            split = dict(train='a'*64, validation='b'*64)
            files['worker-manifest.json'] = self.json(worker / 'worker-manifest.json',
                dict(dataset_id=dataset, train_rows=4, projected_hashes=dict(train=files['train.csv'],
                    validation=files['validation.csv']), projection_sha256=files['projection.json'],
                    split_hashes=split))
            worker_row = dict(dataset=dataset, path=str(worker), files=files, train_rows=4,
                recorded_official_test_digest_read_as_metadata_only=True, projected_features=12,
                raw_training_derived_split_hashes=split)
            attempts = []
            for index in range(11):
                attempt = self.base / 'trials' / dataset / str(index) / 'attempt.json'
                artifact = attempt.parent / 'artifact'
                model_sha = self.put(artifact / 'model.json', b'opaque model, never deserialized')
                projection_sha = self.put(artifact / 'projection.json', (worker / 'projection.json').read_bytes())
                inventory = [dict(path=name, sha256=h, bytes=(artifact / name).stat().st_size)
                             for name, h in (('model.json', model_sha), ('projection.json', projection_sha))]
                raw = dict(artifact_inventory=inventory, artifact_bytes=sum(r['bytes'] for r in inventory),
                           artifact_sha256=model_sha,
                           identity=dict(train_sha256=files['train.csv'], validation_sha256=files['validation.csv']))
                h = self.json(attempt, raw)
                self.report['trials'].append(dict(dataset=dataset, attempt_receipt_path=str(attempt),
                                                  attempt_receipt_sha256=h))
                attempts.append((attempt, artifact, raw))
            for index in range(6):
                # Ninety-eight identical default/native pairs share an artifact.
                choice = 4 if i < 98 and index == 5 else index
                attempt, artifact, raw = attempts[choice]
                self.report['default_native_cells'].append(dict(dataset=dataset, worker=worker_row,
                    artifact_path=str(artifact), fit_attempt_receipt_path=str(attempt),
                    artifact_inventory=raw['artifact_inventory'], artifact_bytes=raw['artifact_bytes'],
                    model_sha256=raw['artifact_sha256']))
        self.patchers = [patch.object(m.native, 'ROOT', self.root),
                         patch.object(m.native, 'replay_native_metadata')]
        for p in self.patchers:
            p.start(); self.addCleanup(p.stop)
        self.save()

    def save(self):
        report_sha = self.json(self.root / 'reconciliation-v1.json', self.report)
        anchor_path = self.root / 'receipt-lock-v1.json'
        self.refs.pop(str(anchor_path), None)
        anchor_sha = self.json(anchor_path, dict(reconciliation_sha256=report_sha, refs=dict(self.refs)))
        for key, value in (('NATIVE', anchor_sha), ('NATIVE_REPORT', report_sha)):
            p = patch.object(m.owned, key, value)
            p.start(); self.addCleanup(p.stop)

    def test_all_trials_workers_and_charged_artifacts(self):
        value = m.replay_native_files()
        self.assertEqual((value['workers_verified'], value['default_native_cells_verified'],
                          value['distinct_artifacts_verified'], value['native_trial_artifacts_verified']),
                         (100, 600, 502, 1100))
        for field in ('rows_parsed', 'models_deserialized', 'frozen_code_initialized', 'publication_admitted'):
            self.assertIs(value[field], False)

    def test_unselected_trial_charge_is_verified(self):
        trial = self.report['trials'][10]
        path = Path(trial['attempt_receipt_path'])
        raw = json.loads(path.read_bytes()); raw['artifact_bytes'] += 1
        trial['attempt_receipt_sha256'] = self.json(path, raw)
        self.save()
        with self.assertRaisesRegex(ValueError, 'charge'): m.replay_native_files()

    def test_trial_is_bound_to_verified_worker_partition(self):
        trial = self.report['trials'][10]
        path = Path(trial['attempt_receipt_path'])
        raw = json.loads(path.read_bytes()); raw['identity']['train_sha256'] = 'e'*64
        trial['attempt_receipt_sha256'] = self.json(path, raw)
        self.save()
        with self.assertRaisesRegex(ValueError, 'partition'): m.replay_native_files()

    def test_charge_projection_manifest_and_attempt_bindings_rejected(self):
        cell = self.report['default_native_cells'][0]
        for field, value in (('artifact_bytes', float(cell['artifact_bytes'])),
                             ('model_sha256', 'e'*64), ('fit_attempt_receipt_path', str(self.base / 'wrong.json'))):
            old = cell[field]; cell[field] = value; self.save()
            with self.subTest(field=field), self.assertRaises(ValueError): m.replay_native_files()
            cell[field] = old
        worker = cell['worker']; path = Path(worker['path']) / 'worker-manifest.json'
        manifest = json.loads(path.read_bytes()); manifest['train_rows'] = 4.
        worker['files']['worker-manifest.json'] = self.json(path, manifest)
        self.save()
        with self.assertRaisesRegex(ValueError, 'manifest'): m.replay_native_files()

    def test_extra_worker_directory_artifact_alias_and_digest_drift(self):
        cell = self.report['default_native_cells'][0]
        extra = Path(cell['worker']['path']) / 'extra'; extra.mkdir()
        with self.assertRaises(ValueError): m.replay_native_files()
        extra.rmdir()
        artifact = Path(cell['artifact_path'])
        alias = artifact / 'alias'; alias.symlink_to(artifact, target_is_directory=True)
        with self.assertRaises(ValueError): m.replay_native_files()
        alias.unlink()
        (artifact / 'model.json').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'): m.replay_native_files()


def operation_fixture(status='ok', outer=5., inner=4., cap=600):
    op = dict(status=status, exit_code=0 if status == 'ok' else 124, host='xbabe2',
              elapsed_seconds=outer, timeout_seconds=cap, new_operation_started=True,
              official_tests_opened=False, foreign_processes_signaled=False)
    receipt = dict(status=status, operation=op, round_sha256=m.owned.ROUND, evidence_files={})
    marker = dict(timeout_seconds=cap, round_sha256=m.owned.ROUND, start_monotonic=1.)
    return receipt, outer, marker, dict(elapsed_seconds=inner) if inner is not None else None, copy.deepcopy(op)


class CostControls(unittest.TestCase):
    def test_outer_worker_timers_are_not_added(self):
        value = m.execution_cost(*operation_fixture())
        self.assertEqual(value['operation_seconds'], 5.)
        self.assertEqual(value['worker_inner_seconds'], 4.)
        self.assertFalse(value['publication_admitted'])

    def test_fractional_deadline_cap_and_shutdown_cost_remain_measured(self):
        value = m.execution_cost(*operation_fixture(outer=1., inner=.3, cap=.5))
        self.assertEqual(value['operation_seconds'], 1.)
        value = m.execution_cost(*operation_fixture('timeout', outer=640., inner=None))
        self.assertEqual((value['operation_seconds'], value['raw_status']), (640., 'timeout'))

    def test_timeout_ram_and_postmetric_failures_remain_raw(self):
        for status in ('failed', 'timeout', 'ram_cap_exceeded'):
            value = m.execution_cost(*operation_fixture(status, inner=None))
            self.assertEqual(value['raw_status'], status)
        values = list(operation_fixture()); values[0]['status'] = 'prelaunch_or_metric_failure'
        self.assertEqual(m.execution_cost(*values)['raw_status'], 'prelaunch_or_metric_failure')
        values = list(operation_fixture('prelaunch_or_metric_failure', inner=None)); values[4] = None
        self.assertEqual(m.execution_cost(*values)['operation_seconds'], 5.)

    def test_unstarted_requires_zero_cost_and_no_execution_evidence(self):
        receipt = dict(status='deadline_unstarted', operation=dict(new_operation_started=False,
                       status='deadline_unstarted', elapsed_seconds=0), evidence_files={'request.json':'a'*64})
        self.assertEqual(m.execution_cost(receipt, 0., None, None)['operation_seconds'], 0.)
        for changes, cost, marker, batch in (({}, 1., None, None), ({'status':'ok'}, 0., None, None),
            ({'evidence_files':{'worker.log':'a'*64}}, 0., None, None), ({}, 0., {}, None), ({}, 0., None, {})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.execution_cost(receipt | changes, cost, marker, batch)

    def test_nonfinite_bool_overflow_and_numeric_alias_costs_rejected(self):
        for value in (True, float('nan'), float('inf'), -1., 10**400, 5):
            values = list(operation_fixture()); values[1] = value
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError): m.execution_cost(*values)
        for key, value in (('new_operation_started', 1), ('timeout_seconds', True),
                            ('timeout_seconds', 601), ('host', 'xbabe0'), ('official_tests_opened', 0),
                            ('foreign_processes_signaled', 0), ('exit_code', False), ('status', 'timeout')):
            values = list(operation_fixture()); values[0]['operation'][key] = value
            values[4] = copy.deepcopy(values[0]['operation'])
            with self.subTest(key=key), self.assertRaises(ValueError): m.execution_cost(*values)

    def test_missing_changed_marker_alias_and_inner_undercharge_rejected(self):
        for marker in (None, dict(timeout_seconds=600., round_sha256=m.owned.ROUND, start_monotonic=1.),
                       dict(timeout_seconds=600, round_sha256='e'*64, start_monotonic=1.)):
            values = list(operation_fixture()); values[2] = marker
            with self.subTest(marker=marker), self.assertRaises(ValueError): m.execution_cost(*values)
        for inner in (True, float('nan'), -1., 0., 5.0000001, 601.):
            values = list(operation_fixture()); values[3]['elapsed_seconds'] = inner
            with self.subTest(inner=inner), self.assertRaises(ValueError): m.execution_cost(*values)
        values = list(operation_fixture()); values[4]['elapsed_seconds'] = 6.
        with self.assertRaises(ValueError): m.execution_cost(*values)
        values = list(operation_fixture()); values[4] = None
        with self.assertRaises(ValueError): m.execution_cost(*values)

    def test_closed_input_gate_runs_before_native_or_cost_replay(self):
        with patch.object(m.owned, 'load_closed_inputs', side_effect=ValueError('external closure denied')), \
             patch.object(m, 'replay_native_files') as native:
            with self.assertRaisesRegex(ValueError, 'external closure denied'):
                m.replay_closed_costs('a'*64, 'b'*64)
            native.assert_not_called()


class ClosedCostControls(TemporaryEvidence):
    def setUp(self):
        super().setUp()
        self.root = self.base / 'closed'
        self.jobs, self.batches = [], []
        role = dict(cpu_slot=list(range(80, 96)), host='xbabe2', ram_bytes=1 << 34,
                    requires_gpu=False, scratch_bytes=10_000_000_000)
        self.lock = dict(jobs=self.jobs, runtime_lock_files={'opaque-runtime':'a'*64},
                         roles={'coordinator':role})
        self.report = dict(batches=self.batches, new_operation_seconds=2510.,
                           native_trial_operation_seconds=1100.)
        self.receipts = []
        for i in range(502):
            job = dict(dataset=f'opaque-{i}', fit_seed=11, final=False)
            key = m.owned.digest_identity(job)
            self.jobs.append(job)
            out = self.root / 'attempts' / key / 'attempt-0001'
            values = list(operation_fixture()); receipt, outer, marker, batch, op = values
            request_sha = self.json(out / 'request.json', dict(job=job, round_sha256=m.owned.ROUND))
            log_sha = self.put(out / 'worker.log', b'opaque log, never parsed')
            admission = dict(admitted=True, blockers=[], round_sha256=m.owned.ROUND,
                             host='xbabe2', requested=role | dict(pid=43210))
            admission_sha = self.json(out / 'admission-0001.json', admission)
            op.update(runtime_lock_files=self.lock['runtime_lock_files'], request_sha256=request_sha,
                      worker_log_sha256=log_sha, admission_sha256=admission_sha,
                      peak_resident_bytes_including_coordinator=4096)
            receipt.update(job=job, operation=op)
            batch['job_sha256'] = key
            for name, value in (('operation.json', op), ('operation-started.json', marker), ('batch.json', batch)):
                self.json(out / name, value)
            receipt['evidence_files'] = {p.name: self.refs[str(p)] for p in out.iterdir()}
            h = self.json(out / 'receipt.json', receipt)
            self.receipts.append((out, receipt))
            self.batches.append(dict(physical_job_sha256=key, operation_seconds=outer, status='ok', receipt_sha256=h))
        self.closed = dict(lock=self.lock, report=self.report,
                           native_report=dict(native_trial_operation_seconds=1100.))
        self.patchers = [patch.object(m.owned, 'ROOT', self.root),
                         patch.object(m.owned, 'load_closed_inputs', return_value=self.closed),
                         patch.object(m, 'replay_native_files', return_value={'opaque_file_gate':True})]
        for p in self.patchers:
            p.start(); self.addCleanup(p.stop)
        self.save()

    def save(self):
        path = self.root / 'receipt-lock-v1.json'
        self.refs.pop(str(path), None)
        self.anchor_sha = self.json(path, dict(refs=dict(self.refs)))

    def call(self):
        return m.replay_closed_costs(self.anchor_sha, 'a'*64)

    def test_all_physical_costs_and_prior_fit_cost_once(self):
        # Closure/native gates are independently tested elsewhere; this fixture
        # isolates anchored operation replay after those mandatory gates.
        value = self.call()
        self.assertEqual((value['physical_batches'], value['operation_seconds']), (502, 2510.))
        self.assertEqual(value['prior_native_trial_operation_seconds'], 1100.)
        self.assertFalse(value['prior_fit_cost_multiplied_by_logical_reuse'])
        self.assertFalse(value['publication_admitted'])
        self.assertIsNone(value['ptf_v1'])

    def test_undercharge_and_prior_reuse_multiplication_rejected(self):
        for field, value in (('new_operation_seconds', 2509.), ('native_trial_operation_seconds', 6600.)):
            old = self.report[field]; self.report[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'totals'): self.call()
            self.report[field] = old

    def test_request_numeric_alias_and_runtime_binding_rejected(self):
        out, receipt = self.receipts[0]
        request = json.loads((out / 'request.json').read_bytes())
        request['job']['fit_seed'] = 11.
        self.json(out / 'request.json', request); self.save()
        with self.assertRaisesRegex(ValueError, 'identity'): self.call()
        request['job']['fit_seed'] = 11
        self.json(out / 'request.json', request)
        receipt['operation']['runtime_lock_files'] = {'changed':'e'*64}
        self.json(out / 'operation.json', receipt['operation'])
        self.json(out / 'receipt.json', receipt); self.save()
        with self.assertRaisesRegex(ValueError, 'reference'): self.call()

    def test_admitted_resource_alias_and_memory_overrun_rejected(self):
        out, receipt = self.receipts[0]
        path = out / 'admission-0001.json'; admission = json.loads(path.read_bytes())
        admission['requested']['ram_bytes'] = float(1 << 34)
        h = self.json(path, admission)
        receipt['operation']['admission_sha256'] = h
        receipt['evidence_files'][path.name] = h
        self.json(out / 'operation.json', receipt['operation'])
        self.json(out / 'receipt.json', receipt); self.save()
        with self.assertRaisesRegex(ValueError, 'admission'): self.call()
        admission['requested']['ram_bytes'] = 1 << 34
        h = self.json(path, admission)
        receipt['operation']['admission_sha256'] = h
        receipt['evidence_files'][path.name] = h
        receipt['operation']['peak_resident_bytes_including_coordinator'] = (1 << 34) + 1
        self.json(out / 'operation.json', receipt['operation'])
        self.json(out / 'receipt.json', receipt); self.save()
        with self.assertRaisesRegex(ValueError, 'memory cap'): self.call()

    def test_actual_receipt_digest_drift_rejected(self):
        (self.receipts[0][0] / 'receipt.json').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'): self.call()


if __name__ == '__main__':
    unittest.main(verbosity=2)
