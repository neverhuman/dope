"""Opaque CPU reuse controls only; no models, data, ML or GPU execution."""
from pathlib import Path
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import sdv_cpu_reuse as m


class FakeAdapter:
    def __init__(self, path, worker):
        self.__file__ = str(path)
        self.worker = worker
        self.samples = []
        self.metrics = []
        self.mutate = False
        self.fail_metric = False

    def sample(self, artifact, rows, seed, output):
        self.samples.append((rows, seed))
        output.write_bytes(b'opaque generated fixture, not measured data')
        if self.mutate:
            (artifact / 'model.pt').write_bytes(b'changed opaque artifact')
        return {'rows': rows, 'sha256': m.sha(output)}

    def efficacy(self, sample, validation, seed):
        self.metrics.append(seed)
        if self.fail_metric:
            raise RuntimeError('opaque injected metric failure')
        return {'objective': 'sdmetrics_mean_regression_r2', 'direction': 'maximize',
            'partition': 'validation', 'seed': seed, 'implementation_sha256': m.sha(self.__file__),
            'validation_sha256': m.sha(validation), 'synthetic_sha256': m.sha(sample),
            'components': {'LinearRegression': -1.0, 'MLPRegressor': -3.0}, 'value': -2.0,
            'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None}

    def fit(self, *args, **kwargs):
        raise AssertionError('CPU reuse must never refit')


class CPUReuseControls(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.base = Path(self.temp.name).resolve()
        self.worker = self.base / 'worker'
        self.worker.mkdir()
        for name in ('train.csv', 'validation.csv', 'projection.json', 'row-group-assignments.json', 'worker-manifest.json'):
            (self.worker / name).write_bytes(b'opaque synthetic fixture only')
        self.output = self.base / 'output'
        self.output.mkdir()
        adapter_path = self.base / 'adapter.py'
        adapter_path.write_bytes(b'opaque source marker only')
        self.adapter = FakeAdapter(adapter_path, self.worker)
        self.worker_spec = {'path': str(self.worker), 'train_rows': 8,
            'files': {p.name: m.sha(p) for p in self.worker.iterdir()}}
        self.job = {'method': 'CTGAN', 'final': False, 'fit_seed': 11,
            'dp_budget': None, 'track': 'common-numeric', 'worker': self.worker_spec,
            'native_objective': {'name': 'sdmetrics_mean_regression_r2', 'direction': 'maximize',
                'implementation_sha256': m.sha(adapter_path)}}
        self.root = self.base / 'closed-native-v2'
        self.artifact = self.root / 'attempts' / m.digest(self.job) / 'attempt-0001' / 'artifact'
        self.artifact.mkdir(parents=True)
        for name in ('model.pt', 'model.json', 'projection.json'):
            (self.artifact / name).write_bytes(b'opaque synthetic fixture only')
        self.rows = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': m.sha(p)}
                     for p in sorted(self.artifact.iterdir())]
        self.fit = {'status': 'ok', 'job_sha256': m.digest(self.job), 'round_sha256': m.ROUND_SHA,
                    'artifact_inventory': self.rows, 'artifact_bytes': sum(r['bytes'] for r in self.rows),
                    'source_adapter_unchanged': True, 'gpu_fit_required': True, 'projection_bytes_included': True,
                    'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None}
        self.root_patch = patch.object(m, 'ROOT', self.root)
        self.base_patch = patch.object(m, 'BASE', self.base)
        self.root_patch.start()
        self.base_patch.start()
        self.env = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'LOKY_MAX_CPU_COUNT': '1'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.root_patch.stop()
        self.base_patch.stop()
        self.temp.cleanup()

    def call(self, action='native', inherited=None):
        return m.cpu_reuse(self.adapter, self.job, self.fit, self.artifact, self.output, action, inherited)

    def native(self):
        p = self.base / 'opaque-native-sample.csv'
        p.write_bytes(b'opaque generated fixture, not measured data')
        return self.adapter.efficacy(p, self.worker / 'validation.csv', 1729) | {'job_sha256': m.digest(self.job)}

    def test_native_reuses_model_on_cpu_and_keeps_negative_score(self):
        result = self.call()
        self.assertEqual(self.adapter.samples, [(8, 101)])
        self.assertEqual(self.adapter.metrics, [1729])
        self.assertFalse(result['new_fit_started'])
        self.assertFalse(result['requires_gpu'])
        self.assertFalse(result['admission_performed_by_core'])
        self.assertEqual(result['artifact_bytes'], sum(row['bytes'] for row in self.rows))
        self.assertEqual(json.loads((self.output / 'native.json').read_bytes())['value'], -2.0)

    def test_common_schedule_requires_native_evidence_and_exact_seed_replay(self):
        inherited = self.native()
        result = self.call('sample', inherited)
        self.assertEqual(self.adapter.samples, [(8,101),(8,211),(8,307),(32,101),(32,211),(32,307),(8,101)])
        self.assertEqual(result['completed_steps'][-1], 'repeat')
        self.assertIsNone(result['mfs_v2'])
        self.assertTrue(json.loads((self.output / 'sample.json').read_bytes())['sample_replay_exact'])

    def test_exposed_cuda_rejects_before_adapter_callbacks(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples or self.adapter.metrics)

    def test_wrong_loky_limit_rejects_before_adapter_callbacks(self):
        os.environ['LOKY_MAX_CPU_COUNT'] = '16'
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples or self.adapter.metrics)

    def test_model_drift_rejects_before_pickle_loader_callback(self):
        (self.artifact / 'model.pt').write_bytes(b'changed opaque artifact')
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)

    def test_projection_or_worker_drift_rejects_before_callbacks(self):
        (self.worker / 'projection.json').write_bytes(b'changed opaque projection')
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)

    def test_adapter_or_native_objective_drift_rejects_before_callbacks(self):
        self.job['native_objective']['name'] = 'mfs_v2'
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples or self.adapter.metrics)

    def test_model_mutation_after_sample_prevents_metric_and_keeps_failure(self):
        self.adapter.mutate = True
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.metrics)
        failure = json.loads((self.output / 'failure.json').read_bytes())
        self.assertEqual(failure['status'], 'failed')
        self.assertFalse(failure['new_fit_started'])

    def test_native_failure_retains_elapsed_cost_and_complete_sample_step(self):
        self.adapter.fail_metric = True
        with self.assertRaises(RuntimeError): self.call()
        failure = json.loads((self.output / 'failure.json').read_bytes())
        self.assertGreaterEqual(failure['elapsed_core_seconds'], 0)
        self.assertEqual(failure['completed_steps'], ['native_sample'])
        self.assertFalse((self.output / 'native.json').exists())

    def test_common_sample_replay_mismatch_fails_on_first_sample(self):
        inherited = self.native()
        inherited['synthetic_sha256'] = 'f'*64
        with self.assertRaises(ValueError): self.call('sample', inherited)
        self.assertTrue((self.output / 'failure.json').exists())
        self.assertFalse((self.output / 'sample.json').exists())
        self.assertEqual(self.adapter.samples, [(8, 101)])

    def test_added_artifact_alias_rejects_before_callbacks(self):
        (self.artifact / 'model-alias').symlink_to(self.artifact / 'model.pt')
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)

    def test_sealed_worker_member_rejects_before_callbacks(self):
        (self.worker / 'test.csv').write_bytes(b'opaque sealed marker, not actual data')
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples or self.adapter.metrics)

    def test_fitted_job_identity_mismatch_rejects_before_callbacks(self):
        self.fit['job_sha256'] = 'd'*64
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples or self.adapter.metrics)

    def test_fitted_round_mismatch_rejects_before_callbacks(self):
        self.fit['round_sha256'] = 'd'*64
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)

    def test_partial_fit_cannot_be_reused(self):
        self.fit['status'] = 'failed'
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)

    def test_unexpected_fit_path_rejects_before_callbacks(self):
        alias = self.base / 'other-artifact'
        alias.mkdir()
        with self.assertRaises(ValueError):
            m.cpu_reuse(self.adapter, self.job, self.fit, alias, self.output, 'native')
        self.assertFalse(self.adapter.samples)

    def test_complete_byte_charge_cannot_omit_projection(self):
        self.fit['artifact_bytes'] -= (self.artifact / 'projection.json').stat().st_size
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)

    def test_inherited_native_job_identity_mismatch_rejects_before_callbacks(self):
        inherited = self.native()
        inherited['job_sha256'] = 'd'*64
        with self.assertRaises(ValueError): self.call('sample', inherited)
        self.assertFalse(self.adapter.samples)

    def test_native_metric_claim_or_lineage_drift_rejects_before_callbacks(self):
        inherited = self.native()
        for key, value in (('validation_sha256','d'*64), ('mfs_v2',0.99), ('direction','minimize')):
            with self.subTest(key=key):
                changed = copy.deepcopy(inherited)
                changed[key] = value
                with self.assertRaises(ValueError): self.call('sample', changed)
        self.assertFalse(self.adapter.samples)

    def test_native_metric_required_before_common_sampling(self):
        with self.assertRaises(ValueError): self.call('sample', None)
        self.assertFalse(self.adapter.samples)

    def test_output_cannot_write_into_predecessor(self):
        self.output = self.root / 'new-output'
        self.output.mkdir()
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(list(self.output.iterdir()) or self.adapter.samples)

    def test_output_cannot_write_into_worker(self):
        self.output = self.worker / 'new-output'
        self.output.mkdir()
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)

    def test_source_guard_cost_included_in_success_elapsed(self):
        # Staged source/worker/model validation occurs between the first and
        # final clock observations, not before the operation's elapsed clock.
        with patch.object(m.time, 'monotonic', side_effect=(10.0, 25.0)):
            result = self.call()
        self.assertEqual(result['elapsed_core_seconds'], 15.0)

    def test_worker_mutation_after_metric_cannot_publish_native_success(self):
        original = self.adapter.efficacy
        def change_worker(*args):
            result = original(*args)
            (self.worker / 'train.csv').write_bytes(b'changed opaque worker')
            return result
        self.adapter.efficacy = change_worker
        with self.assertRaises(ValueError): self.call()
        self.assertTrue((self.output / 'failure.json').exists())
        self.assertFalse((self.output / 'native.json').exists())

    def test_adapter_path_drift_rejects_before_callbacks(self):
        Path(self.adapter.__file__).write_bytes(b'changed opaque source')
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)

    def test_final_flag_rejects_before_callbacks(self):
        self.job['final'] = 1
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.adapter.samples)


if __name__ == '__main__':
    unittest.main()
