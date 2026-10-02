"""Missing/duplicate cells and native selection cannot acquire published scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark.publish_s3_matched import (
    CONFIGS, SEEDS, SIZES, native_winner, schema, summarize, verify_cell, verify_frozen_sources,
)
from research.benchmark.manifest import digest
from research.benchmark.score import sha256


def cells():
    return [dict(dataset='synthetic', method=m, configuration=c, fit_seed=11,
                 sample_seed=s, size_multiplier=z, status='ok', charged_artifact_bytes=100,
                 utility={a: {'retention': v, 'informative': True}
                          for a in ('catboost', 'linear', 'mlp')})
            for m, c in CONFIGS for z in SIZES for s, v in zip(SEEDS, (0.5, 0.7, 0.9))]


def attempts():
    return [dict(job={'trial': t, 'dataset': 'synthetic', 'method': 'TVAE', 'fit_seed': 11,
                      'train_sha256': 'a'*64, 'validation_sha256': 'b'*64,
                      'projection_sha256': 'c'*64, 'config': {'trial': t}},
                 native_kpi={'value': v}, artifact_bytes=100, wall_seconds=10,
                 common_retention=100 if t == 0 else -100)
            for t, v in enumerate((0.1, 0.8, 0.5, 0.2))]


class S3MatchedPanelTests(unittest.TestCase):
    def test_incomplete_and_duplicate_matrices_fail(self):
        rows = cells()
        for broken in (rows[:-1], rows + [rows[0]], rows[1:] + [rows[1]]):
            with self.assertRaisesRegex(ValueError, 'incomplete or duplicated'):
                summarize(broken, ['synthetic'])
        result = summarize(rows, ['synthetic'])
        self.assertEqual(len(result), 16)
        self.assertEqual(result[0]['utility']['catboost']['median_retention'], 0.7)

    def test_missing_or_low_signal_sample_makes_median_null(self):
        rows = cells()
        rows[0]['status'] = 'fit_unavailable'
        rows[6]['utility']['linear']['informative'] = False
        rows[12]['utility']['mlp']['retention'] = None
        rows[18]['utility']['catboost'] = {'status': 'failed', 'error_type': 'ValueError'}
        result = summarize(rows, ['synthetic'])
        for auditor in ('catboost', 'linear', 'mlp'):
            self.assertTrue(any(r['utility'][auditor]['median_retention'] is None for r in result))

    def test_native_winner_ignores_common_utility(self):
        rows = attempts()
        self.assertEqual(native_winner(rows)['job']['trial'], 1)
        rows[2]['native_kpi']['value'] = .8
        rows[2]['artifact_bytes'] = 99
        self.assertEqual(native_winner(rows)['job']['trial'], 2)
        for mutation in ('lineage', 'time', 'duplicate'):
            bad = copy.deepcopy(rows)
            if mutation == 'lineage':
                bad[0]['job']['validation_sha256'] = 'd'*64
            elif mutation == 'time':
                bad[0]['wall_seconds'] = 43200
            else:
                bad[0]['job']['trial'] = 1
            with self.assertRaises(ValueError):
                native_winner(bad)

    def test_missing_receipt_rejected_before_metrics(self):
        target = Path('target/s3-matched-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as root, patch(
                'research.benchmark.publish_s3_matched.read') as read:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                verify_cell(Path(root), {'synthetic': True}, 'a'*64, {})
            read.assert_not_called()

    def test_rehashed_cost_or_metric_receipt_cannot_replace_frozen_receipt(self):
        target = Path('target/s3-matched-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            job = {'synthetic': True}
            path = root / 'cells' / digest(job) / 'receipt.json'
            path.parent.mkdir(parents=True)
            original = {'job': job, 'operations': [{'elapsed_seconds': 1}],
                        'evidence_files': {'metrics.json': 'a'*64}}
            path.write_text(json.dumps(original))
            anchors = {str(path): sha256(path)}
            for changed in (original | {'operations': [{'elapsed_seconds': 124}]},
                            original | {'evidence_files': {'metrics.json': 'b'*64}}):
                path.write_text(json.dumps(changed))
                with patch('research.benchmark.publish_s3_matched.BASE', root), patch(
                        'research.benchmark.publish_s3_matched.read') as read:
                    with self.assertRaisesRegex(ValueError, 'frozen validation receipt changed'):
                        verify_cell(root, job, 'a'*64, anchors)
                    read.assert_not_called()

    def test_frozen_source_binary_runtime_and_partition_drift_rejected(self):
        target = Path('target/s3-matched-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            dope, native, sdv = (root / n for n in ('dope', 'neural-native-v1', 'sdv'))

            def put(path, data=b'fixture'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return sha256(path)

            package = root / 'pilot-24h/dope-target-refinement-v1/package'
            metric = package / 'research/benchmark/pilot_metrics.py'
            metric_hash = put(metric)
            manifest_hash = put(package / 'package-manifest.json', json.dumps(
                {'files': {'research/benchmark/pilot_metrics.py': metric_hash}}).encode())
            worker = root / 'worker'
            worker_files = {name: put(worker / name, b'0,1\n')
                            for name in ('train.csv', 'validation.csv')}
            dependency = native / 'deps/numpy/core.py'
            dependency_hash = put(dependency)
            library = root / 'installed/torch/core.py'
            library_hash = put(library)
            runtime = root / 's3-native-matched-runtime-v1/xbabe2/runtime.lock.json'
            runtime_hash = put(runtime, json.dumps({'packages': {'torch': {
                'root': str(library.parent), 'files': {'core.py': library_hash}}}}).encode())
            binary = root / 'bin/generator'
            adapter = native / 'package/research/benchmark/sdv_adapter.py'
            locks = {
                'dope_fit': {'source_files': {'round.py': put(dope / 'source/round.py')},
                             'package_manifest_sha256': manifest_hash,
                             'gpu_binary_path': str(binary), 'gpu_binary_sha256': put(binary),
                             'host_runtime_locks': {}, 'workers': {'synthetic': {
                                 'path': str(worker), 'files': worker_files}}},
                'dope_validation': {'source_files': {'evaluate.py': put(dope / 'validation-v1/source/evaluate.py')},
                                    'package_manifest_sha256': manifest_hash, 'metric_source_sha256': metric_hash,
                                    'metric_dependency_root': str(native / 'deps'),
                                    'metric_dependency_files': {'numpy/core.py': dependency_hash}},
                'native_fit': {'source_files': {'sdv_adapter.py': put(adapter)},
                               'dependency_files': {'numpy/core.py': dependency_hash}},
                'native_validation': {'source_files': {'evaluate.py': put(sdv / 'source/evaluate.py')},
                                      'metric_package_manifest_sha256': manifest_hash,
                                      'native_source_files': {'sdv_adapter.py': sha256(adapter)},
                                      'dependency_files': {'numpy/core.py': dependency_hash},
                                      'runtime_sha256': runtime_hash, 'jobs': []},
            }
            from contextlib import ExitStack
            with ExitStack() as stack:
                for key, value in {'BASE': root, 'DOPE': dope, 'NATIVE': native, 'SDV': sdv}.items():
                    stack.enter_context(patch('research.benchmark.publish_s3_matched.' + key, value))
                self.assertGreater(verify_frozen_sources(locks, {}), 0)
                for path in (metric, adapter, binary, worker / 'train.csv', worker / 'validation.csv',
                             dope / 'validation-v1/source/evaluate.py', sdv / 'source/evaluate.py',
                             dependency, library):
                    original = path.read_bytes()
                    path.write_bytes(b'changed')
                    try:
                        with self.subTest(path=path.name), self.assertRaisesRegex(
                                ValueError, 'frozen source or input changed'):
                            verify_frozen_sources(locks, {})
                    finally:
                        path.write_bytes(original)

    def test_committed_schema_cannot_promote_or_unseal(self):
        import jsonschema
        document = {'format': 'synthetic', 'official_tests_opened': False,
                    'production_certified': False, 'mfs_v2': None, 'ptf_v1': None}
        contract = schema(document)
        jsonschema.validate(document, contract)
        for key, value in [('official_tests_opened', True), ('production_certified', True),
                           ('mfs_v2', .99), ('ptf_v1', .99)]:
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(document | {key: value}, contract)


if __name__ == '__main__':
    unittest.main()
