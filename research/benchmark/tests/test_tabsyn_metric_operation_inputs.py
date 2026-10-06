"""Generated metadata controls only; no tables/models/auditors/native actions."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import tabsyn_metric_operation_inputs as bridge

APP_PATH = Path(__file__).resolve().parents[1] / 'tabsyn_validation_inputs.py'
AUTHOR = 'cb5ac0f74ec36ee88e7a974a393dfbef50d42da7'
CONFIG = 'a34729383eac6783c7e754a791094054d54f1062655db9a306cd4cca296c5ebc'


class MetadataFixture:
    def __init__(self, retained=False):
        self.buffers, self.values, self.calls = {}, {}, []
        self.inventory = {'artifact_bytes': 21, 'projection_bytes': 4,
                          'files': {'model.bin': {'bytes': 17, 'sha256': '7' * 64},
                                    'projection.json': {'bytes': 4, 'sha256': '8' * 64}}}
        request = {'lineage_sha256': '1' * 64, 'config': {'config_sha256': CONFIG}, 'deadline_monotonic': 600.,
                   'learning_identity': {'source_commit': AUTHOR},
                   'runtime_ref': {'sha256': '9' * 64}}
        if not retained:
            request['mode'] = 'fit'
        request_ref = self.add('/frozen/operation/request.json', request)
        result = {'complete': True, 'seed': 11, 'rows': 80, 'columns': 3,
                  'code_sha256': request_ref['sha256'], 'operation_admission_sha256': request_ref['sha256'],
                  'config': request['config'], 'source_commit': AUTHOR, 'runtime_sha256': '9' * 64,
                  'generated_fixture': False, 'official_tests_opened': False}
        complete = {'status': 'ok', 'method': 'TabSyn', 'lineage_sha256': '1' * 64,
                    'config_sha256': CONFIG, 'official_tests_opened': False,
                    'production_certified': False, 'mfs_v2': None, 'ptf_v1': None}
        if retained:
            complete.update(result=result, inventory=self.inventory)
        else:
            complete['result'] = {'result': result, 'inventory': self.inventory}
        complete_ref = self.add('/frozen/operation/fit.complete.receipt.json', complete)
        popen_ref = self.add('/frozen/operation/actual-scientific-Popen.receipt.json', {
            'request_ref': request_ref, 'scientific_worker_pid': 123, 'uid': 1000,
            'start_ticks': '100', 'whole_deadline_monotonic': 600., 'official_tests_opened': False})
        exit_row = {'scientific_worker_pid': 123, 'start_ticks': '100', 'actual_worker_exit': 0,
                    'foreign_signals': 0}
        if retained:
            exit_row.update(fit_complete_receipt_present=True, owned_termination_reason=None,
                            official_tests_opened=False, production_certified=False)
        else:
            exit_row['success_receipt_ref'] = complete_ref
        exit_ref = self.add('/frozen/operation/actual-owned-exit.receipt.json', exit_row)
        self.worker = {'fit_request_ref': request_ref, 'fit_complete_ref': complete_ref,
                       'fit_exit_ref': exit_ref, 'artifact_inventory': self.inventory,
                       'config_sha256': CONFIG, 'dimensions': 3,
                       'original_TRAIN_input_ref': {'row_count': 80}}
        self.fit = {'lineage_sha256': '1' * 64, 'retained_first': retained,
                    'actual_metadata_refs': [request_ref, complete_ref, popen_ref, exit_ref]}

    def add(self, path, value):
        data = bridge.canonical(value).encode()
        self.buffers[path], self.values[path] = data, copy.deepcopy(value)
        return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def read(self, ref):
        self.calls.append(ref['path'])
        if ref['path'] not in self.buffers:
            raise FileNotFoundError('captured_metadata_unmaterialized')
        return self.buffers[ref['path']]

    def replace(self, key, mutation):
        old = self.worker[key]
        value = copy.deepcopy(self.values[old['path']])
        mutation(value)
        self.worker[key] = self.add(old['path'], value)
        if key == 'fit_complete_ref' and not self.fit['retained_first']:
            exit_row = self.values[self.worker['fit_exit_ref']['path']]
            exit_row['success_receipt_ref'] = self.worker[key]
            self.worker['fit_exit_ref'] = self.add(self.worker['fit_exit_ref']['path'], exit_row)

    def validate(self):
        return bridge.validate_fit_operation(self.worker, self.fit, self.read)

    def snapshot(self, admitted=False):
        fits, workers, cells = [], {}, []
        for i in range(100):
            lineage = f'{i:064x}'
            fits.append({'lineage_sha256': lineage})
            workers[lineage] = {}
            for multiplier in (1, 4):
                for seed in (101, 211, 307):
                    cells.append({'lineage_sha256': lineage, 'seed': seed, 'row_multiplier': multiplier,
                                  'status': 'unstarted', 'eligible_for_evaluator': False})
        if admitted:
            cells[0].update(status='admitted_new_CPU', eligible_for_evaluator=True)
            fits[0].update(actual_metadata_refs=[{
                'path': '/unmaterialized/actual-scientific-Popen.receipt.json', 'sha256': 'a' * 64}])
            workers[f'{0:064x}'] = {'fit_request_ref': {
                'path': '/unmaterialized/request.json', 'sha256': 'a' * 64}}
        phase_ref = self.add('/frozen/fit-phase.json', {
            'fit_phase_closed': True, 'default_fits_all_ok': True, 'source_commit': AUTHOR,
            'config_sha256': CONFIG, 'official_tests_opened': False, 'lineages': fits})
        return self.add('/frozen/snapshot.json', {
            'format': 'dope-TS-admitted-common-diagnostic-inputs-PREPARATION-ONLY-v1',
            'execution_enabled': False, 'scientific_Popen_authorized': False,
            'actual_evaluator_jobs_started': 0, 'claims': {'official_tests_opened': False},
            'evaluator_ref': {'path': '/frozen/evaluator.py', 'sha256': bridge.EVALUATOR_SHA256},
            'fit_phase100_ref': phase_ref, 'workers': workers, 'sample_cells': cells,
            'status_counts': dict(bridge.Counter(c['status'] for c in cells)),
            'admitted_snapshot_inputs': int(admitted)})

    def operation(self, admitted=False):
        ref = self.snapshot(admitted)
        source = APP_PATH.read_bytes()  # Source bytes only, never learned/input bodies.
        return bridge.prepare_operation(ref, {'path': str(APP_PATH), 'sha256': bridge.ADAPTER_SHA256},
                                        f'{0:064x}', 101, 1, metadata_reader=self.read,
                                        source_reader=lambda _: source)


class Controls(unittest.TestCase):
    def test_reader_cannot_replace_digest_before_json_decoding(self):
        data = b'{"frozen":true}'
        ref = {'path': '/frozen/receipt.json', 'bytes': len(data),
               'sha256': hashlib.sha256(data).hexdigest()}
        original = dict(ref)
        def reader(received):
            changed = b'{not-json'
            received.update(bytes=len(changed), sha256=hashlib.sha256(changed).hexdigest())
            return changed
        with patch.object(bridge.json, 'loads') as decoder:
            with self.assertRaisesRegex(ValueError, 'pinned_bytes_changed'):
                bridge.decode(ref, reader)
            decoder.assert_not_called()
        self.assertEqual(ref, original)

    def test_reader_cannot_replace_declared_size(self):
        data = b'captured-source'
        ref = {'path': '/frozen/adapter.py', 'bytes': len(data) + 1,
               'sha256': hashlib.sha256(data).hexdigest()}
        original = dict(ref)
        def reader(received):
            received['bytes'] = len(data)
            return data
        with self.assertRaisesRegex(ValueError, 'pinned_size_changed'):
            bridge.checked(ref, reader)
        self.assertEqual(ref, original)

    def test_both_actual_fit_receipt_variants_join_without_parent_upgrade(self):
        for retained in (False, True):
            f = MetadataFixture(retained)
            request, ref = f.validate()
            self.assertEqual(ref['path'], '/frozen/operation/actual-scientific-Popen.receipt.json')
            self.assertNotIn('parent_exit', request)
            self.assertFalse(any(Path(p).suffix in ('.csv', '.bin') for p in f.calls))

    def test_fit_request_cannot_claim_another_lineage(self):
        f = MetadataFixture(); f.replace('fit_request_ref', lambda row: row.update(lineage_sha256='2' * 64))
        with self.assertRaisesRegex(ValueError, 'fit_request_lineage_changed'):
            f.validate()

    def test_hash_checked_before_any_json_parse(self):
        f = MetadataFixture(); ref = f.worker['fit_request_ref']
        f.buffers[ref['path']] = b'not JSON'
        with patch.object(bridge.json, 'loads') as loads:
            with self.assertRaisesRegex(ValueError, 'pinned_bytes_changed'):
                f.validate()
            loads.assert_not_called()

    def test_completion_is_really_read_instead_of_trusting_aggregate_fit_ok(self):
        f = MetadataFixture(); f.buffers[f.worker['fit_complete_ref']['path']] = b'changed'
        with self.assertRaisesRegex(ValueError, 'pinned_bytes_changed'):
            f.validate()

    def test_scientific_nonzero_exit_rejects_even_with_complete_model(self):
        f = MetadataFixture(); f.replace('fit_exit_ref', lambda r: r.update(actual_worker_exit=2))
        with self.assertRaisesRegex(ValueError, 'fit_scientific_exit_unavailable'):
            f.validate()

    def test_replaced_PID_or_start_tick_does_not_join(self):
        for mutation in (lambda r: r.update(scientific_worker_pid=124), lambda r: r.update(start_ticks='101')):
            f = MetadataFixture(); f.replace('fit_exit_ref', mutation)
            with self.assertRaisesRegex(ValueError, 'fit_process_join_changed'):
                f.validate()

    def test_wrong_success_receipt_does_not_join(self):
        f = MetadataFixture(); f.replace('fit_exit_ref', lambda r: r['success_receipt_ref'].update(sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'fit_success_receipt_join_changed'):
            f.validate()

    def test_inventory_drift_and_float_seed_reject(self):
        for mutation in (lambda r: r['result']['inventory'].update(artifact_bytes=22),
                         lambda r: r['result']['result'].update(seed=11.)):
            f = MetadataFixture(); f.replace('fit_complete_ref', mutation)
            with self.assertRaisesRegex(ValueError, 'fit_completion_binding_changed'):
                f.validate()

    def test_historical_runtime_binding_must_match_exact_request(self):
        f = MetadataFixture(); f.replace('fit_complete_ref', lambda r: r['result']['result'].update(runtime_sha256='a' * 64))
        with self.assertRaisesRegex(ValueError, 'fit_completion_binding_changed'):
            f.validate()

    def test_invalid_integer_cell_identity_fails_before_adapter_or_receipts(self):
        for seed, mult in ((101., 1), (True, 1), (101, 1.)):
            with patch.object(bridge, 'bind_adapter') as load:
                with self.assertRaisesRegex(ValueError, 'invalid_cell_identity'):
                    bridge.prepare_operation({}, {}, '1' * 64, seed, mult,
                                             metadata_reader=lambda _: b'', source_reader=lambda _: b'')
                load.assert_not_called()

    def test_source_drift_rejects_before_executing_frozen_adapter(self):
        ref = {'path': str(APP_PATH), 'sha256': bridge.ADAPTER_SHA256}
        with patch('builtins.compile') as compile_source:
            with self.assertRaisesRegex(ValueError, 'pinned_bytes_changed'):
                bridge.bind_adapter(ref, lambda _: b'changed source')
            compile_source.assert_not_called()

    def test_adapter_loader_never_consults_cached_bytecode_or_other_source(self):
        source = APP_PATH.read_bytes()
        ref = {'path': str(APP_PATH), 'sha256': bridge.ADAPTER_SHA256}
        with patch.object(bridge.importlib.machinery.SourceFileLoader, 'get_data',
                          side_effect=AssertionError('unexpected cache or file read')):
            loaded = bridge.bind_adapter(ref, lambda _: source)
        self.assertEqual(loaded['AUTHOR'], AUTHOR)
        self.assertTrue(callable(loaded['prepare_cell']))

    def test_source_reader_cannot_rebind_the_adapter_digest_before_compilation(self):
        ref = {'path': str(APP_PATH), 'sha256': bridge.ADAPTER_SHA256}
        def changed(reference):
            reference['sha256'] = hashlib.sha256(b'# different source').hexdigest()
            return b'# different source'
        with patch('builtins.compile') as compile_source:
            with self.assertRaisesRegex(ValueError, 'pinned_bytes_changed'):
                bridge.bind_adapter(ref, changed)
            compile_source.assert_not_called()
        self.assertEqual(ref['sha256'], bridge.ADAPTER_SHA256)

    def test_job_digest_is_deterministic_and_type_exact(self):
        a = {'lineage': '1' * 64, 'seed': 101, 'multiplier': 1}
        self.assertEqual(bridge.digest(a), bridge.digest(dict(reversed(list(a.items())))))
        self.assertNotEqual(bridge.digest(a), bridge.digest(dict(a, seed=101.)))

    def test_duplicate_keys_and_nonfinite_metadata_reject(self):
        for data, reason in ((b'{"x":1,"x":2}', 'duplicate_json_key'),
                             (b'{"x":NaN}', 'nonfinite_json_constant'),
                             (b'{"x":1e999}', 'nonfinite_json_constant')):
            ref = {'path': '/frozen/receipt.json', 'sha256': hashlib.sha256(data).hexdigest()}
            with self.assertRaisesRegex(ValueError, reason):
                bridge.decode(ref, lambda _: data)

    def test_unstarted_cell_returns_unavailable_without_receipt_or_body_reads(self):
        f = MetadataFixture(); result = f.operation()
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['historical_sample_status'], 'unstarted')
        self.assertIsNone(result['evaluator_input'])
        self.assertIsNone(result['historical_parent_exit'])
        self.assertFalse(result['execution_enabled'])
        self.assertEqual(f.calls, ['/frozen/snapshot.json', '/frozen/fit-phase.json'])

    def test_admitted_snapshot_with_unmaterialized_receipt_remains_unavailable(self):
        result = MetadataFixture().operation(admitted=True)
        self.assertEqual(result['reason'], 'captured_receipt_or_source_unmaterialized')
        self.assertIsNone(result['evaluator_input'])

    def test_public_status_has_no_path_or_source_value(self):
        result = MetadataFixture().operation()
        public = {k: result[k] for k in ('status', 'reason', 'historical_parent_exit',
                                        'execution_enabled', 'mfs_v2', 'ptf_v1', 'superiority')}
        self.assertNotIn('/frozen', bridge.canonical(public))
        self.assertIsNone(public['superiority'])


if __name__ == '__main__':
    unittest.main()
