"""Hermetic pre-initialization controls for existing SDV sample evaluation."""
from pathlib import Path
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from research.benchmark import sdv_frozen_sample_validation as worker


class FrozenSampleValidation(unittest.TestCase):
    def setUp(self):
        target = Path.cwd() / 'target'
        target.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'round'
        self.parent = self.base / 'parent'
        self.inputs = self.base / 'inputs'
        self.out = self.root / 'attempts' / 'job' / 'attempt-0001'
        self.source = self.parent / 'attempts' / 'original' / 'attempt-0001'
        for p in (self.root / 'source', self.parent / 'source', self.inputs, self.out, self.source / 'artifact'):
            p.mkdir(parents=True)
        self.marker = self.base / 'initialized'
        self.write(self.root / 'source' / 'shared_runtime.py',
                   f"from pathlib import Path\ndef verify():\n Path({str(self.marker)!r}).write_text('initialized')\n return {{'metric_sha256': 'metric'}}\n", raw=True)
        self.write(self.parent / 'source' / 'adapter.py', '# immutable original source\n', raw=True)
        files = {}
        for name in ('train.csv', 'validation.csv', 'projection.json', 'row-group-assignments.json', 'worker-manifest.json'):
            self.write(self.inputs / name, '{}\n', raw=True)
            files[name] = worker.sha(self.inputs / name)
        inputs = {'path': str(self.inputs), 'files': files, 'train_rows': 2}
        original = {'final': False, 'fit_seed': 11, 'method': 'CTGAN', 'dataset': 'dataset', 'worker': inputs}
        self.original_sha = worker.digest(original)
        inventory = []
        for name in ('model.json', 'model.pt', 'projection.json'):
            p = self.source / 'artifact' / name
            self.write(p, '{}\n', raw=True)
            inventory.append({'path': name, 'sha256': worker.sha(p), 'bytes': p.stat().st_size})
        fit = {'status': 'ok', 'job_sha256': self.original_sha, 'artifact_bytes': 9,
               'projection_bytes_included': True, 'artifact_inventory': inventory}
        self.write(self.source / 'fit.json', fit)
        samples = []
        for size in (1, 4):
            for seed in (101, 211, 307):
                name = ('n' if size == 1 else '4n') + f'-seed{seed}'
                self.write(self.source / (name + '.csv'), '0,0\n', raw=True)
                samples.append({'schedule': {'multiplier': size, 'name': name, 'seed': seed},
                                'rows': 2 * size, 'sample_sha256': worker.sha(self.source / (name + '.csv'))})
        self.write(self.source / 'repeat.csv', '0,0\n', raw=True)
        repeat = samples[0] | {'schedule': {'multiplier': 1, 'name': 'repeat', 'seed': 101}}
        self.write(self.source / 'sample.json', {'status': 'ok', 'sample_replay_exact': True,
                   'job_sha256': self.original_sha, 'samples': samples + [repeat]})
        source_files = {str(p): worker.sha(p) for p in (self.parent / 'source').iterdir()}
        self.write(self.parent / 'round.lock.json', {'source_files': source_files})
        self.parent_sha = worker.sha(self.parent / 'round.lock.json')
        receipt = {'status': 'ok', 'job': original, 'round_sha256': self.parent_sha,
                   'evidence_files': {p.name: worker.sha(p) for p in self.source.iterdir() if p.is_file()}}
        self.write(self.source / 'receipt.json', receipt)
        trial = {'job_sha256': self.original_sha, 'receipt_sha256': worker.sha(self.source / 'receipt.json'),
                 'common_samples_complete': True}
        report = {'complete_native_matrix': True, 'new_trials_closed': 704,
                  'shared_kpi_used_for_selection': False, 'sampling_success_used_for_selection': False,
                  'default_native_cells': [{'dataset': 'dataset', 'method': 'CTGAN', 'all_four_trials_closed': True,
                                           'default_trial': trial, 'native_selected_trial': trial}]}
        self.write(self.parent / 'reconciliation-v1.json', report)
        refs = {str(p): worker.sha(p) for root in (self.inputs, self.parent) for p in root.rglob('*') if p.is_file()}
        self.write(self.parent / 'receipt-lock-v1.json', {'refs': refs, 'round_sha256': self.parent_sha,
                   'reconciliation_sha256': worker.sha(self.parent / 'reconciliation-v1.json')})
        self.job = {'dataset': 'dataset', 'method': 'CTGAN', 'fit_seed': 11, 'final': False, 'track': 'common-numeric',
                    'worker': inputs, 'original_job_sha256': self.original_sha,
                    'native_receipt_path': str(self.source / 'receipt.json'),
                    'native_receipt_sha256': worker.sha(self.source / 'receipt.json'),
                    'fit_receipt_sha256': worker.sha(self.source / 'fit.json'),
                    'sample_receipt_sha256': worker.sha(self.source / 'sample.json'),
                    'artifact_path': str(self.source / 'artifact'), 'artifact_bytes': 9,
                    'projection_bytes_included': True, 'samples': samples,
                    'selection_bindings': ['author_default', 'native_selected']}
        self.lock = {'jobs': [self.job], 'source_files': {str(p): worker.sha(p) for p in (self.root / 'source').iterdir()},
                     'gpu_operations_enabled': False, 'new_generator_fits_started': 0, 'native_selection_changed': False,
                     'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None,
                     'parent_receipt_lock_path': str(self.parent / 'receipt-lock-v1.json'),
                     'parent_receipt_lock_sha256': worker.sha(self.parent / 'receipt-lock-v1.json'),
                     'parent_reconciliation_path': str(self.parent / 'reconciliation-v1.json'),
                     'parent_reconciliation_sha256': worker.sha(self.parent / 'reconciliation-v1.json'),
                     'runtime_lock_files': {}, 'metric_source_sha256': 'metric'}
        self.refresh()
        self.environment = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @staticmethod
    def write(path, value, raw=False):
        Path(path).write_text(value if raw else json.dumps(value, sort_keys=True, allow_nan=False))

    def refresh(self, request_job=None):
        self.write(self.root / 'round.lock.json', self.lock)
        self.expected = worker.sha(self.root / 'round.lock.json')
        self.request = self.out / 'request.json'
        self.write(self.request, {'job': self.job if request_job is None else request_job, 'round_sha256': self.expected})

    def reject_before_initializer(self):
        with self.assertRaises((ValueError, FileNotFoundError, KeyError)):
            worker.prepare(self.root, self.expected, self.request)
        self.assertFalse(self.marker.exists())

    def test_positive_existing_samples_only(self):
        _, job, _, _, runtime = worker.prepare(self.root, self.expected, self.request)
        self.assertEqual(job, self.job)
        self.assertEqual(runtime['metric_sha256'], 'metric')
        self.assertTrue(self.marker.exists())

    def test_cpu_environment_checked_first(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            self.reject_before_initializer()

    def test_numeric_identity_aliases_rejected(self):
        for field, value in [('fit_seed', 11.0), ('final', 0)]:
            with self.subTest(field=field):
                job = copy.deepcopy(self.job)
                job[field] = value
                self.refresh(job)
                self.reject_before_initializer()

    def test_changed_parent_receipt_rejected(self):
        self.write(self.source / 'receipt.json', {})
        self.reject_before_initializer()

    def test_rewritten_parent_lock_rejected(self):
        self.write(self.parent / 'receipt-lock-v1.json', {})
        self.reject_before_initializer()

    def test_changed_sample_rejected(self):
        self.write(self.source / '4n-seed101.csv', '1,1\n', raw=True)
        self.reject_before_initializer()

    def test_added_artifact_rejected(self):
        self.write(self.source / 'artifact' / 'uncharged', 'extra', raw=True)
        self.reject_before_initializer()

    def test_directory_alias_rejected(self):
        (self.root / 'source' / 'alias').symlink_to(self.root / 'source', target_is_directory=True)
        self.reject_before_initializer()

    def test_wrong_native_winner_rejected(self):
        self.job['selection_bindings'] = ['invalid_shared_winner']
        self.refresh()
        self.reject_before_initializer()

    def test_added_test_file_rejected(self):
        self.write(self.inputs / 'test.csv', 'sealed', raw=True)
        self.reject_before_initializer()

    def test_changed_runtime_pin_rejected(self):
        p = self.base / 'runtime.lock.json'
        self.write(p, {})
        self.lock['runtime_lock_files'] = {str(p): '0' * 64}
        self.refresh()
        self.reject_before_initializer()

    def test_final_rehash_charged_before_success(self):
        self.job['metric_replay_required'] = False
        runtime = {'metric_source': self.base / 'metric.py', 'expected_versions': {}}
        lock = self.lock | {'metric_batch_timeout_seconds': 600}
        sequence = []

        def prepare(*args):
            sequence.append('integrity')
            return lock, self.job, self.inputs, self.source, runtime

        def measure(train, validation, sample, task, seed):
            sequence.append('measure')
            return {'dependencies': {}, 'mfs_v2': None, 'gate_profile_complete': False,
                    'utility': dict.fromkeys(('linear', 'catboost', 'mlp')),
                    'rows': {'synthetic': 8 if sample.name.startswith('4n') else 2}}

        with patch.object(worker, 'prepare', side_effect=prepare), \
                patch.object(worker, 'load', return_value=SimpleNamespace(measure=measure)), \
                patch.object(worker.time, 'monotonic', side_effect=[0, 601]), \
                patch.object(worker.sys, 'argv', ['worker', str(self.root), self.expected, str(self.request)]):
            with self.assertRaisesRegex(ValueError, 'whole batch deadline exceeded'):
                worker.main()
        self.assertEqual(sequence, ['integrity'] + ['measure'] * 6 + ['integrity'])
        self.assertFalse((self.out / 'batch.json').exists())


if __name__ == '__main__':
    unittest.main()
