"""Missing/duplicate cells and native selection cannot acquire published scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark.publish_s3_confirmation import (
    CONFIGS, PINS, SEEDS, SIZES, native_winner, schema, summarize, verify_cell, verify_frozen_sources,
    verify_native_fit, verify_dope_artifact, completed_controllers, verify_execution, safe,
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


class S3ConfirmationPanelTests(unittest.TestCase):
    def test_manifest_rejects_replaced_schema_tables_and_figures(self):
        from research.benchmark import publish_s3_confirmation_manifest as manifest
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            report = {'official_tests_opened': False, 'production_certified': False,
                      'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None,
                      'paired_superiority': None, 'cells': [], 'dataset_ids': [], 'summary': []}
            expected = {'.json': json.dumps(report).encode(),
                        '.schema.json': json.dumps(schema(report)).encode(),
                        '.csv': b'csv', '.md': b'markdown', '.svg': b'svg', '.pdf': b'pdf'}
            for suffix, data in expected.items():
                (root / (manifest.NAME + suffix)).write_bytes(data)
            with patch.object(manifest, 'RESULTS', root), patch.object(
                    manifest, 'build_report', return_value=report), patch.object(
                    manifest, 'tables', return_value=('csv', 'markdown')), patch.object(
                    manifest, 'render', side_effect=lambda r, kind: kind.encode()):
                for suffix in ('.schema.json', '.csv', '.md', '.svg', '.pdf'):
                    path = root / (manifest.NAME + suffix)
                    path.write_bytes(b'{}' if suffix == '.schema.json' else b'rewritten')
                    try:
                        with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'differs|differ'):
                            manifest.build()
                    finally:
                        path.write_bytes(expected[suffix])

    def test_native_objective_must_bind_frozen_implementation_validation_and_sample(self):
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            artifact = root / 'artifact'
            artifact.mkdir()
            model = artifact / 'model.json'
            model.write_text('{}\n')
            job = {'validation_sha256': 'a'*64}
            lock = {'jobs': [job], 'source_files': {'sdv_adapter.py': 'b'*64}}
            original = {'job': job, 'round_sha256': PINS['native_fit'], 'status': 'ok',
                        'validation_only': True, 'mfs_v2': None, 'ptf_v1': None,
                        'evidence_files': {'artifact/model.json': sha256(model)},
                        'artifact_inventory': [{'path': 'model.json', 'bytes': model.stat().st_size,
                                                'sha256': sha256(model)}],
                        'artifact_bytes': model.stat().st_size,
                        'sample': {'child': {'sha256': 'c'*64}}, 'sampling_repeated_sha256': 'c'*64,
                        'native_kpi': {'partition': 'validation', 'direction': 'maximize',
                                       'objective': 'sdmetrics_mean_regression_r2', 'value': .5,
                                       'components': {'LinearRegression': .5, 'MLPRegressor': .5},
                                       'implementation_sha256': 'b'*64, 'validation_sha256': 'a'*64,
                                       'synthetic_sha256': 'c'*64, 'seed': 1729}}
            path = root / 'receipt.json'
            with patch('research.benchmark.publish_s3_confirmation.BASE', root):
                path.write_text(json.dumps(original))
                verify_native_fit(path, lock, sha256(path))
                for key, value in [('implementation_sha256', 'd'*64),
                                   ('validation_sha256', 'd'*64), ('synthetic_sha256', 'd'*64),
                                   ('seed', 11)]:
                    changed = copy.deepcopy(original)
                    changed['native_kpi'][key] = value
                    path.write_text(json.dumps(changed))
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'native objective changed'):
                        verify_native_fit(path, lock, sha256(path))

    def test_failed_or_over_budget_operation_cannot_supply_successful_metrics(self):
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            job = {'synthetic': True}
            path = root / 'cells' / digest(job) / 'receipt.json'
            path.parent.mkdir(parents=True)
            original = {'job': job, 'round_sha256': 'a'*64, 'status': 'ok',
                        'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                        'new_fits_started': 0, 'evidence_files': {},
                        'operations': [{'tag': t, 'status': 'ok', 'exit_code': 0,
                                        'elapsed_seconds': 1} for t in ('sample', 'repeat', 'metric')]}
            for update in ({'status': 'timeout'}, {'exit_code': 1}, {'elapsed_seconds': 611}):
                row = copy.deepcopy(original)
                row['operations'][0].update(update)
                path.write_text(json.dumps(row))
                with patch('research.benchmark.publish_s3_confirmation.BASE', root), patch(
                        'research.benchmark.publish_s3_confirmation.read', return_value=row) as read:
                    with self.assertRaisesRegex(ValueError, 'failed or over-budget operation'):
                        verify_cell(root, job, 'a'*64, {str(path): sha256(path)})
                    self.assertEqual(read.call_count, 1)

    def test_admission_source_and_owner_inventory_drift_rejected(self):
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            source = root / 'source/coordinator.py'
            source.parent.mkdir()
            source.write_text('fixture coordinator\n')
            scanner = root / 'scanner.py'
            scanner.write_text('fixture scanner\n')
            dependency = root / 'budget.json'
            dependency.write_text('{}\n')
            lock = root / 'execution.lock.json'
            lock.write_text(json.dumps({
                'source_files': {'coordinator.py': sha256(source)},
                'owner_scan_source': str(scanner), 'owner_scan_source_sha256': sha256(scanner),
                'required_files': {str(dependency): sha256(dependency)},
                'recognized_owner_sources': {str(source): {'sha256': sha256(source)}}}))
            with patch('research.benchmark.publish_s3_confirmation.BASE', root):
                refs = {}
                verify_execution(lock, sha256(lock), refs)
                self.assertEqual(set(refs), {str(p) for p in (lock, source, scanner, dependency)})
                for path in (source, scanner, dependency):
                    original = path.read_bytes()
                    path.write_bytes(b'drift')
                    try:
                        with self.subTest(path=path.name), self.assertRaisesRegex(
                                ValueError, 'execution dependency changed'):
                            verify_execution(lock, sha256(lock), {})
                    finally:
                        path.write_bytes(original)

    def test_evidence_directory_alias_rejected(self):
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            retained = root / 'retained'
            retained.mkdir()
            (retained / 'receipt.json').write_text('{}\n')
            alias = root / 'alias'
            alias.symlink_to(retained, target_is_directory=True)
            with patch('research.benchmark.publish_s3_confirmation.BASE', root):
                with self.assertRaisesRegex(ValueError, 'outside allowed scope'):
                    safe(alias / 'receipt.json')

    def test_live_or_failed_controller_cannot_publish(self):
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            dope, native = root / 'dope', root / 'native'
            controller = dope / 'validation-v2'
            controller.mkdir(parents=True)
            path = controller / 'controller-attempt-0001-exit.json'
            with patch('research.benchmark.publish_s3_confirmation.BASE', root), patch(
                    'research.benchmark.publish_s3_confirmation.DOPE', dope), patch(
                    'research.benchmark.publish_s3_confirmation.SDV', native):
                with self.assertRaisesRegex(ValueError, 'controller not closed'):
                    completed_controllers()
                path.write_text(json.dumps({'exit_code': 1,
                                            'round_sha256': PINS['dope_validation']}))
                with self.assertRaisesRegex(ValueError, 'controller failed'):
                    completed_controllers()

    def test_gpu_artifact_charges_projection_and_requires_verified_training(self):
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            child = root / 'attempt-0001.json'
            model = child.with_suffix('.dpk')
            model.write_bytes(b'fixture kernel')
            worker = root / 'worker'
            worker.mkdir()
            projection = worker / 'projection.json'
            projection.write_text('{}\n')
            charged = model.stat().st_size + projection.stat().st_size
            fitted = {'status': 'ok', 'trained_gpu_target_verified': True,
                      'artifact_bytes': charged, 'artifact_sha256': sha256(model)}
            job = {'worker': {'path': str(worker),
                             'files': {'projection.json': sha256(projection)}}}
            with patch('research.benchmark.publish_s3_confirmation.BASE', root):
                for mutation in (None, 'charge', 'gpu'):
                    payload = copy.deepcopy(fitted)
                    if mutation == 'charge':
                        payload['artifact_bytes'] -= projection.stat().st_size
                    if mutation == 'gpu':
                        payload['trained_gpu_target_verified'] = False
                    child.write_text(json.dumps(payload))
                    parent = {'status': 'ok', 'child': {'receipt': str(child),
                                                      'receipt_sha256': sha256(child)}}
                    if mutation is None:
                        actual, trained = verify_dope_artifact(parent, job, {'status': 'ok'}, {})
                        self.assertEqual(actual, charged)
                        self.assertEqual(trained[0], child)
                    else:
                        with self.assertRaises(ValueError):
                            verify_dope_artifact(parent, job, {'status': 'ok'}, {})

    def test_gpu_admission_failure_has_no_artifact_or_training_claim(self):
        failure = {'status': 'dispatch_failed', 'child': None}
        with patch('research.benchmark.publish_s3_confirmation.read') as read:
            charged, trained = verify_dope_artifact(
                failure, {}, {'status': 'fit_unavailable'}, {})
            self.assertIsNone(charged)
            self.assertIsNone(trained)
            read.assert_not_called()
            for changed in (failure | {'child': {'artifact_bytes': 1}},
                            failure | {'status': 'charged_artifact_cap'}):
                with self.assertRaisesRegex(ValueError, 'unavailable DOPE fit accounting'):
                    verify_dope_artifact(changed, {}, {'status': 'fit_unavailable'}, {})

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
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as root, patch(
                'research.benchmark.publish_s3_confirmation.read') as read:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                verify_cell(Path(root), {'synthetic': True}, 'a'*64, {})
            read.assert_not_called()

    def test_native_artifact_root_link_rejected_before_evidence_reads(self):
        target = Path('target/s3-confirmation-tests')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory).resolve()
            retained = root / 'retained'
            retained.mkdir()
            model = retained / 'model.json'
            model.write_text('{"fixture": true}\n')
            artifact = root / 'artifact'
            artifact.symlink_to(retained, target_is_directory=True)
            job = {'synthetic': True}
            receipt = {'job': job, 'round_sha256': PINS['native_fit'], 'status': 'ok',
                       'validation_only': True, 'mfs_v2': None, 'ptf_v1': None,
                       'evidence_files': {'artifact/model.json': sha256(model)},
                       'artifact_inventory': [{'path': 'model.json', 'bytes': model.stat().st_size,
                                               'sha256': sha256(model)}],
                       'artifact_bytes': model.stat().st_size,
                       'native_kpi': {'partition': 'validation', 'direction': 'maximize',
                                      'objective': 'sdmetrics_mean_regression_r2', 'value': .5,
                                      'components': {'LinearRegression': .5, 'MLPRegressor': .5}}}
            path = root / 'receipt.json'
            path.write_text(json.dumps(receipt))
            with patch('research.benchmark.publish_s3_confirmation.BASE', root), patch(
                    'research.benchmark.publish_s3_confirmation.evidence') as evidence:
                with self.assertRaisesRegex(ValueError, 'artifact root must be an unlinked directory'):
                    verify_native_fit(path, {'jobs': [job]}, sha256(path))
                evidence.assert_not_called()

    def test_rehashed_cost_or_metric_receipt_cannot_replace_frozen_receipt(self):
        target = Path('target/s3-confirmation-tests')
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
                with patch('research.benchmark.publish_s3_confirmation.BASE', root), patch(
                        'research.benchmark.publish_s3_confirmation.read') as read:
                    with self.assertRaisesRegex(ValueError, 'frozen validation receipt changed'):
                        verify_cell(root, job, 'a'*64, anchors)
                    read.assert_not_called()

    def test_frozen_source_binary_runtime_and_partition_drift_rejected(self):
        target = Path('target/s3-confirmation-tests')
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
                'dope_validation': {'source_files': {'evaluate.py': put(dope / 'validation-v2/source/evaluate.py')},
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
                    stack.enter_context(patch('research.benchmark.publish_s3_confirmation.' + key, value))
                self.assertGreater(verify_frozen_sources(locks, {}), 0)
                for path in (metric, adapter, binary, worker / 'train.csv', worker / 'validation.csv',
                             dope / 'validation-v2/source/evaluate.py', sdv / 'source/evaluate.py',
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
