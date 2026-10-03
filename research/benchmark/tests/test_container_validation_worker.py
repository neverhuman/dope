"""Hermetic worker custody controls; no real models, dependencies or binary runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from research.benchmark import container_validation_worker as w
from research.benchmark import research_container as codec


def sha(body):
    return hashlib.sha256(body).hexdigest()


class WorkerControls(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        target.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='container-worker-control-', dir=target)
        self.base = Path(self.temp.name)
        self.root = self.base / 'round'
        self.source = self.root / 'source'
        self.source.mkdir(parents=True)
        self.out = self.root / 'attempts' / 'opaque-job' / 'attempt-0001'
        self.out.mkdir(parents=True, mode=0o700)
        self.worker = self.base / 'worker'
        self.worker.mkdir()
        self.model = b'opaque model member' * 200
        self.projection = json.dumps({'task': 'regression'}).encode()
        for name in ('train.csv', 'validation.csv', 'row-group-assignments.json', 'worker-manifest.json'):
            (self.worker / name).write_bytes(b'opaque fixture')
        (self.worker / 'projection.json').write_bytes(self.projection)
        self.original = self.base / 'original.dpk'
        self.original.write_bytes(self.model)
        self.blob = codec.encode(self.model, self.projection, 9)
        self.artifact = self.base / 'compressed.artifact'
        self.artifact.write_bytes(self.blob)
        self.parent = self.base / 'parent-receipts.json'
        self.parent.write_text(json.dumps({'refs': {str(self.original): sha(self.model)}}))
        self.marker = self.base / 'initializer-sentinel'
        self.metric = self.base / 'metric.py'
        self.metric.write_text('''def measure(train, validation, sample, task, seed):
    rows = int(sample.read_text().split(':')[0])
    return {'dependencies': {}, 'rows': {'synthetic': rows}, 'mfs_v2': None,
            'gate_profile_complete': False, 'utility': {'linear': {}, 'catboost': {}, 'mlp': {}}}
''')
        repo = Path(w.__file__).parent
        for name in ('research_container.py', 'research_container_members.py'):
            shutil.copyfile(repo / name, self.source / name)
        shutil.copyfile(w.__file__, self.source / 'entry.py')
        (self.source / 'common.py').write_text('def check(expected):\n    pass\n')
        runtime = {'metric_source': str(self.metric), 'metric_sha256': w.sha(self.metric), 'expected_versions': {}}
        (self.source / 'shared_runtime.py').write_text('from pathlib import Path\n'
            f'Path({str(self.marker)!r}).write_text("initialized")\n'
            f'def verify():\n    return {runtime!r}\n')
        (self.source / 'native_runtime.py').write_text('def verify_runtime(lock):\n'
            '    return {"torch_library_directory": "opaque-frozen-library-directory"}\n')
        self.job = {'final': False, 'track': 'common-numeric', 'split': 'official_training_derived_validation',
            'fit_seed': 11, 'sample_seeds': [101, 211, 307], 'size_multipliers': [1, 4],
            'original_generator_binary_sha256': '0' * 64,
            'worker': {'path': str(self.worker), 'train_rows': 7,
                       'files': {p.name: w.sha(p) for p in self.worker.iterdir()}},
            'compressed_artifact_path': str(self.artifact), 'compressed_artifact_sha256': sha(self.blob),
            'compressed_artifact_bytes': len(self.blob), 'expected_original_model_sha256': sha(self.model),
            'expected_original_projection_sha256': sha(self.projection),
            'original_model_path_for_sampling_replay': str(self.original)}
        self.lock = {'execution_admitted': True, 'execution_lock_frozen': True,
            'clean_sdv_predecessor_closed': True, 'density_priority_satisfied': True,
            'official_tests_opened': False, 'gpu_operations_enabled': False, 'global_family_selected': False,
            'gate_profile_complete': False, 'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None,
            'source_files': {str(p): w.sha(p) for p in self.source.iterdir()},
            'jobs': [self.job], 'frozen_references': {}, 'fit_receipt_lock_path': str(self.parent),
            'fit_receipt_lock_sha256': w.sha(self.parent), 'runtime_lock_files': {},
            'metric_source_sha256': w.sha(self.metric), 'gpu_binary_sha256': '0' * 64,
            'native_library_directory': 'opaque-frozen-library-directory'}
        self.request = self.out / 'request.json'
        self.host = patch.object(w.socket, 'gethostname', return_value='xbabe2')
        self.host.start()
        self.bytecode = patch.object(sys, 'dont_write_bytecode', True)
        self.bytecode.start()
        self.env = patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH='opaque-frozen-library-directory')
        self.env.start()
        self.freeze()

    def tearDown(self):
        self.env.stop()
        self.host.stop()
        self.bytecode.stop()
        self.clear_modules()
        self.temp.cleanup()

    def clear_modules(self):
        for name in list(sys.modules):
            if name in ('common', 'shared_runtime', 'native_runtime', 'population_shared_metric') or name.startswith('_dope_container_validation'):
                del sys.modules[name]

    def freeze(self):
        parent = self.out.parent
        identity = parent.parent / w.digest(self.job)
        if parent != identity:
            parent.rename(identity)
            self.out = identity / self.out.name
            self.request = self.out / 'request.json'
        lock = self.root / 'round.lock.json'
        lock.write_text(json.dumps(self.lock))
        self.expected = w.sha(lock)
        self.request.write_text(json.dumps({'round_sha256': self.expected, 'job': self.job}))

    def reject_before_initializer(self):
        with self.assertRaises(ValueError), patch.object(w, 'sample') as sample:
            w.run(self.root, self.expected, self.request)
        self.assertFalse(self.marker.exists())
        self.assertFalse((self.out / 'decoded-members').exists())
        sample.assert_not_called()

    def test_unadmitted_or_unclosed_predecessor_never_initializes(self):
        for key in ('execution_admitted', 'execution_lock_frozen', 'clean_sdv_predecessor_closed', 'density_priority_satisfied'):
            for value in (False, 1, 'true'):
                with self.subTest(key=key, value=value):
                    self.lock[key] = value
                    self.freeze()
                    self.reject_before_initializer()
            self.lock[key] = True

    def test_final_truthy_values_never_initialize(self):
        for value in (True, 1, 'true', None):
            with self.subTest(value=value):
                self.job['final'] = value
                self.freeze()
                self.reject_before_initializer()

    def test_equal_valued_request_cannot_substitute_an_unfrozen_digest(self):
        self.lock['jobs'] = [json.loads(json.dumps(self.job))]
        self.job['fit_seed'] = 11.0
        self.assertEqual(self.job, self.lock['jobs'][0])
        self.assertNotEqual(w.digest(self.job), w.digest(self.lock['jobs'][0]))
        self.freeze()
        self.reject_before_initializer()

    def test_invalid_cpu_environment_rejects_before_any_runtime_initializer(self):
        for change in ({'CUDA_VISIBLE_DEVICES': 'opaque-invalid-device'},
                       {'LD_LIBRARY_PATH': 'opaque-invalid-library-directory'}):
            with self.subTest(change=change), patch.dict(os.environ, change):
                self.reject_before_initializer()

    def test_manifest_and_source_drift_never_initialize(self):
        with (self.root / 'round.lock.json').open('a') as stream:
            stream.write(' ')
        self.reject_before_initializer()
        self.freeze()
        with (self.source / 'shared_runtime.py').open('a') as stream:
            stream.write('\n# drift\n')
        self.reject_before_initializer()

    def test_added_file_or_directory_alias_rejected_before_initializer(self):
        for name, target in (('added.py', self.metric), ('empty-directory', self.worker)):
            path = self.source / name
            path.symlink_to(target)
            self.reject_before_initializer()
            path.unlink()
        added = self.source / 'extra.py'
        added.write_text('raise RuntimeError("must not initialize")')
        self.reject_before_initializer()

    def test_parent_or_worker_drift_rejected_before_initializer(self):
        self.parent.write_text(json.dumps({'refs': {}}))
        self.reject_before_initializer()
        self.parent.write_text(json.dumps({'refs': {str(self.original): sha(self.model)}}))
        (self.worker / 'train.csv').write_bytes(b'changed opaque input')
        self.reject_before_initializer()

    def test_container_or_external_member_drift_rejected_before_initializer(self):
        self.artifact.write_bytes(self.blob + b'changed')
        self.reject_before_initializer()
        self.clear_modules()
        forged = codec.encode(self.model + b'changed', self.projection, 9)
        self.artifact.write_bytes(forged)
        self.job['compressed_artifact_sha256'] = sha(forged)
        self.job['compressed_artifact_bytes'] = len(forged)
        self.freeze()
        self.reject_before_initializer()

    def test_xbabe0_is_rejected_even_with_frozen_admission(self):
        with patch.object(w.socket, 'gethostname', return_value='xbabe0'):
            self.reject_before_initializer()

    def test_bytecode_writes_are_disabled_before_source_initializers(self):
        with patch.object(sys, 'dont_write_bytecode', False):
            self.reject_before_initializer()

    def test_retry_identity_and_private_directory_mode_precede_initializer(self):
        parent = self.out.parent
        parent.rename(parent.with_name('changed-job-identity'))
        self.out = parent.with_name('changed-job-identity') / self.out.name
        self.request = self.out / 'request.json'
        self.reject_before_initializer()
        self.freeze()
        self.out.chmod(0o755)
        self.reject_before_initializer()

    def test_runtime_lock_drift_never_initializes(self):
        runtime = self.base / 'runtime-inventory.json'
        runtime.write_text('{}')
        self.lock['runtime_lock_files'] = {str(runtime): w.sha(runtime)}
        self.freeze()
        runtime.write_text('{"changed":true}')
        self.reject_before_initializer()

    def test_failed_native_guard_or_spent_whole_batch_budget_never_executes(self):
        native = types.SimpleNamespace(verify_runtime=lambda lock: (_ for _ in ()).throw(ValueError('opaque runtime drift')))
        with patch.object(w.subprocess, 'run') as invoke, self.assertRaises(ValueError):
            w.sample(self.lock, native, self.original, 7, 101, self.out / 'new.csv', time.monotonic())
        invoke.assert_not_called()
        native = types.SimpleNamespace(verify_runtime=lambda lock: {})
        with patch.object(w.subprocess, 'run') as invoke, self.assertRaises(ValueError):
            w.sample(self.lock, native, self.original, 7, 101, self.out / 'new.csv', time.monotonic() - 601)
        invoke.assert_not_called()

    def test_sampler_consumes_owned_members_and_replays_all_six_cells(self):
        calls = []
        def fake_sample(lock, native, kernel, rows, seed, output, start):
            self.assertEqual(kernel.read_bytes(), self.model)
            calls.append((kernel, rows, seed, output))
            output.write_text(f'{rows}:{seed}')
        with patch.object(w, 'sample', side_effect=fake_sample):
            w.run(self.root, self.expected, self.request)
        self.assertEqual(len(calls), 13)
        owned = self.out / 'decoded-members' / 'model.dpk'
        self.assertEqual(sum(p == owned for p, _, _, _ in calls), 7)
        self.assertEqual(sum(p == self.original for p, _, _, _ in calls), 6)
        self.assertEqual((owned.parent / 'projection.json').read_bytes(), self.projection)
        value = json.loads((self.out / 'batch.json').read_text())
        self.assertEqual(value['artifact_bytes'], len(self.blob))
        self.assertEqual(value['artifact_sha256'], sha(self.blob))
        self.assertEqual(len(value['samples']), 6)
        self.assertEqual(value['job_sha256'], w.digest(self.job))
        self.assertEqual(value['round_sha256'], self.expected)
        self.assertEqual(value['request_sha256'], w.sha(self.request))
        self.assertEqual(sum(row['metric_replay'] == 'exact' for row in value['samples']), 1)
        self.assertTrue(value['original_kernel_replays_exact'])
        self.assertEqual(value['new_generator_fits_started'], 0)
        self.assertIsNone(value['mfs_v2'])
        self.assertIsNone(value['ptf_v1'])
        self.assertFalse(value['official_tests_opened'])
        self.assertFalse(value['gate_profile_complete'])

    def test_original_replay_mismatch_never_writes_success_batch(self):
        def fake_sample(lock, native, kernel, rows, seed, output, start):
            output.write_text('changed' if kernel == self.original else f'{rows}:{seed}')
        with patch.object(w, 'sample', side_effect=fake_sample), self.assertRaises(ValueError):
            w.run(self.root, self.expected, self.request)
        self.assertFalse((self.out / 'batch.json').exists())
        self.assertFalse(list(self.out.glob('*.metric.json')))

    def test_request_changed_during_sampling_never_writes_success_batch(self):
        def fake_sample(lock, native, kernel, rows, seed, output, start):
            output.write_text(f'{rows}:{seed}')
            self.request.write_text('{"changed":true}')
        with patch.object(w, 'sample', side_effect=fake_sample), self.assertRaises(ValueError):
            w.run(self.root, self.expected, self.request)
        self.assertFalse((self.out / 'batch.json').exists())
        self.assertEqual(len(list(self.out.glob('*.metric.json'))), 6)

    def test_metric_replay_drift_preserves_partial_evidence_without_success(self):
        self.metric.write_text('''count = 0
def measure(train, validation, sample, task, seed):
    global count
    count += 1
    rows = int(sample.read_text().split(':')[0])
    return {'dependencies': {}, 'rows': {'synthetic': rows}, 'mfs_v2': None,
            'gate_profile_complete': False, 'utility': {'linear': {'opaque': count}, 'catboost': {}, 'mlp': {}}}
''')
        old = self.lock['metric_source_sha256']
        self.lock['metric_source_sha256'] = w.sha(self.metric)
        path = self.source / 'shared_runtime.py'
        # The hermetic guard returns this fixture identity, with no real runtime.
        path.write_text(path.read_text().replace(old, w.sha(self.metric)))
        self.lock['source_files'][str(path)] = w.sha(path)
        self.freeze()
        def fake_sample(lock, native, kernel, rows, seed, output, start):
            output.write_text(f'{rows}:{seed}')
        with patch.object(w, 'sample', side_effect=fake_sample), self.assertRaises(ValueError):
            w.run(self.root, self.expected, self.request)
        self.assertFalse((self.out / 'batch.json').exists())
        self.assertEqual(len(list(self.out.glob('*.metric.json'))), 3)

    def test_deadline_expiry_during_final_checks_prevents_success_batch(self):
        clock = {'now': 0.0}
        original_sha = w.sha
        def checked_sha(path):
            value = original_sha(path)
            if Path(path) == self.request:
                clock['now'] = 601.0
            return value
        def fake_sample(lock, native, kernel, rows, seed, output, start):
            output.write_text(f'{rows}:{seed}')
        with patch.object(w, 'sample', side_effect=fake_sample), patch.object(w, 'sha', side_effect=checked_sha), \
             patch.object(w.time, 'monotonic', side_effect=lambda: clock['now']), \
             self.assertRaisesRegex(TimeoutError, '^compressed research batch deadline exhausted$'):
            w.run(self.root, self.expected, self.request)
        self.assertFalse((self.out / 'batch.json').exists())
        self.assertEqual(len(list(self.out.glob('*.metric.json'))), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
