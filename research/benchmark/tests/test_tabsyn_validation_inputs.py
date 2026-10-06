"""Metadata controls; numeric tables, models and auditor imports are forbidden."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import tabsyn_validation_inputs as inputs


class Fixture:
    def __init__(self):
        self.buffers = {}
        self.decoded = {}
        self.calls = []
        self.snapshot = {
            'format': 'dope-TS-admitted-common-diagnostic-inputs-PREPARATION-ONLY-v1',
            'execution_enabled': False, 'scientific_Popen_authorized': False,
            'actual_evaluator_jobs_started': 0, 'claims': {'official_tests_opened': False},
            'evaluator_ref': {'sha256': inputs.EVALUATOR}, 'sample_cells': [], 'workers': {},
        }
        fits = []
        for i in range(100):
            lineage = f'{i:064x}'
            base = f'/frozen/{i:016x}'
            def ref(name):
                return {'path': base + '/' + name, 'sha256': '1' * 64}
            inventory = {'files': {'model': {'bytes': 17},
                                  'projection.json': {'bytes': 3, 'sha256': '1' * 64}},
                         'artifact_bytes': 20, 'projection_bytes': 3}
            train = dict(ref('train.csv'), csv_format='headerless_numeric', row_count=10)
            worker = {'dataset': f'{i:016x}', 'dimensions': 3, 'author_target_index': 0,
                      'common_target_index': 2, 'author_joint_to_common_permutation': [1, 2, 0],
                      'common_to_author_joint_permutation': [2, 0, 1],
                      'sample_CSV_to_common_permutation': [0, 1, 2],
                      'original_TRAIN_input_ref': train,
                      'original_derived_validation_input_ref': dict(ref('validation.csv'),
                                             csv_format='headerless_numeric', row_count=4),
                      'projection_ref': ref('projection.json'),
                      'validation_provenance': {'partition': 'official_training_derived_validation',
                                             'official_tests_opened': False, 'projection_sha256': '1' * 64},
                      'config_sha256': inputs.CONFIG, 'fit_request_ref': ref('fit.request.json'),
                      'fit_complete_ref': ref('fit.complete.json'), 'fit_exit_ref': ref('fit.exit.json'),
                      'artifact_path': base + '/artifact', 'artifact_inventory': inventory,
                      'column_kinds_common_order': ['continuous'] * 3, 'kind_worker_key': str(i),
                      'kind_rule_sha256': '2' * 64, 'kind_producer_source_sha256': '3' * 64,
                      'original_lineage_start': 100., 'original_lineage_deadline': 43300.}
            kinds = {'column_kinds': ['continuous'] * 3, 'train_sha256': train['sha256'],
                     'worker_key': str(i), 'rule_sha256': '2' * 64, 'producer_source_sha256': '3' * 64,
                     'validation_or_synthetic_used_for_types': False, 'official_tests_opened': False}
            worker['kind_map_ref'] = self.add(base + '/kinds.json', kinds)
            self.snapshot['workers'][lineage] = worker
            fits.append(dict(worker, status='fit_ok', lineage_sha256=lineage, fit_seed=11,
                             original_lineage_started_unix_seconds=100.,
                             original_lineage_deadline_unix_seconds=43300.,
                             actual_worker_exit=0))
            for mult in (1, 4):
                for seed in (101, 211, 307):
                    self.snapshot['sample_cells'].append({'lineage_sha256': lineage, 'seed': seed,
                         'row_multiplier': mult, 'status': 'unstarted', 'eligible_for_evaluator': False})
        for position, backend in ((0, 'cpu'), (1, 'cuda:0')):
            cell = self.snapshot['sample_cells'][position]
            worker = self.snapshot['workers'][cell['lineage_sha256']]
            op = '/frozen/operation-' + str(position)
            result = {'path': op + '/sample.csv', 'filename': 'sample.csv', 'rows': 10,
                      'purpose': 'matched', 'seed': cell['seed'], 'sha256': '4' * 64, 'bytes': 59}
            request = {'learning_identity': {'fit_seed': 11, 'method': 'TabSyn',
                        'source_commit': inputs.AUTHOR, 'dataset': worker['dataset'], 'projection_sha256': '1' * 64},
                       'lineage_sha256': cell['lineage_sha256'], 'config': {'config_sha256': inputs.CONFIG},
                       'mode': 'sample', 'artifact_code_ref': worker['fit_request_ref'],
                       'artifact_inventory': worker['artifact_inventory'], 'artifact_path': worker['artifact_path'],
                       'official_tests_opened': False, 'production_certified': False, 'generated_fixture': False,
                       'scientific_fit_allocations': 0, 'lineage_started_unix_seconds': 100.,
                       'entry_ref': {'path': op + '/writer.py', 'sha256': inputs.WRITERS[backend]},
                       'source_files': {op + '/writer.py': inputs.WRITERS[backend]},
                       'sample_backend': 'cpu' if backend == 'cpu' else 'cuda', 'deadline_monotonic': 700.,
                       'samples': [{'filename': 'sample.csv', 'purpose': 'matched', 'rows': 10, 'seed': cell['seed']}]}
            worker['fit_request_ref'] = self.add(worker['fit_request_ref']['path'], {
                'learning_identity': request['learning_identity'],
                'config': request['config'],
                'worker': {'cohort': {'train_sha256': '1' * 64, 'validation_sha256': '1' * 64}}})
            fits[0]['fit_request_ref'] = worker['fit_request_ref']
            request['artifact_code_ref'] = worker['fit_request_ref']
            cell['sample_request_ref'] = self.add(op + '/request.json', request)
            cell['sample_Popen_ref'] = self.add(op + '/Popen.json', {
                'request_ref': cell['sample_request_ref'], 'scientific_worker_pid': 120 + position,
                'scientific_fit_allocations': 0,
                'start_ticks': '100', 'whole_deadline_monotonic': 700.})
            cell['sample_exit_ref'] = self.add(op + '/exit.json', {
                'actual_worker_exit': 0, 'foreign_signals': 0, 'owned_signal_calls': 0,
                'owned_termination_reason': None, 'official_tests_opened': False,
                'scientific_worker_pid': 120 + position, 'start_ticks': '100',
                'scientific_fit_allocations': 0,
                'deadline_rechecked': True, 'completed_monotonic': 140., 'deadline_monotonic': 700.})
            cell['sample_complete_ref'] = self.add(op + '/complete.json', {
                'status': 'ok', 'mode': 'sample', 'lineage_sha256': cell['lineage_sha256'],
                'config_sha256': inputs.CONFIG, 'official_tests_opened': False,
                'production_certified': False, 'mfs_v2': None, 'ptf_v1': None, 'result': [result]})
            cell.update(status='admitted_new_CPU' if backend == 'cpu' else 'admitted_retained_GPU',
                        eligible_for_evaluator=True, actual_worker_exit=0, rows=10, purpose='matched',
                        physical_backend=backend, sample_result=result)
        phase = {'fit_phase_closed': True, 'default_fits_all_ok': True, 'source_commit': inputs.AUTHOR,
                 'config_sha256': inputs.CONFIG, 'official_tests_opened': False, 'lineages': fits}
        self.snapshot['fit_phase100_ref'] = self.add('/frozen/phase.json', phase)
        self.snapshot['status_counts'] = {'admitted_new_CPU': 1, 'admitted_retained_GPU': 1, 'unstarted': 598}
        self.snapshot['admitted_snapshot_inputs'] = 2

    def add(self, path, value):
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        self.buffers[path] = data; self.decoded[path] = value
        return {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

    def change_receipt(self, cell, name, fn):
        ref = cell[name]; value = copy.deepcopy(self.decoded[ref['path']]); fn(value)
        cell[name] = self.add(ref['path'], value)

    def read(self, ref):
        self.calls.append(ref['path'])
        return self.buffers[ref['path']]

    def prepare(self):
        return inputs.prepare(self.add('/frozen/snapshot.json', self.snapshot), self.read)


class Controls(unittest.TestCase):
    def test_cpu_and_gpu_inputs_preserve_target_last_and_unknown_parent(self):
        f = Fixture()
        got = f.prepare()
        self.assertEqual(len(got['jobs']), 2)
        self.assertEqual(got['status_counts']['unstarted'], 598)
        for job in got['jobs']:
            self.assertEqual(job['sample_CSV_to_common_permutation'], [0, 1, 2])
            self.assertEqual(job['charged_artifact_bytes'], 20)
            self.assertEqual(job['projection_bytes'], 3)
            self.assertIsNone(job['historical_parent_exit'])
            self.assertFalse(job['execution_enabled'])
        self.assertFalse(any(Path(p).suffix in ('.csv', '.safetensors') for p in f.calls))

    def test_digest_is_checked_before_json_decode(self):
        f = Fixture(); ref = f.add('/frozen/snapshot.json', f.snapshot)
        f.buffers[ref['path']] = b'not JSON'
        with self.assertRaisesRegex(ValueError, 'frozen metadata bytes changed'), patch.object(inputs.json, 'loads') as loads:
            inputs.prepare(ref, f.read)
        loads.assert_not_called()

    def test_retained_gpu_unrecorded_signal_counter_stays_null(self):
        f = Fixture(); cell = f.snapshot['sample_cells'][1]
        f.change_receipt(cell, 'sample_exit_ref', lambda row: row.pop('owned_signal_calls'))
        self.assertIsNone(f.prepare()['jobs'][1]['historical_owned_signal_calls'])

    def test_retained_header_view_requires_original_payload_identity(self):
        f = Fixture(); worker = next(iter(f.snapshot['workers'].values()))
        source = {'path': '/frozen/header-view', 'files': {'train.csv': '5' * 64, 'validation.csv': '6' * 64}}
        proof = {'format': 'dope-retained-first-worker-header-payload-identity-readonly-v1',
                 'first_fit_request_ref': worker['fit_request_ref'], 'rows': []}
        for name, ref in [('train.csv', worker['original_TRAIN_input_ref']),
                          ('validation.csv', worker['original_derived_validation_input_ref'])]:
            proof['rows'].append({'file': name, 'original_payload_byte_identical': True,
                                 'original_ref': ref, 'payload_sha256': ref['sha256'],
                                 'view_ref': {'path': source['path'] + '/' + name, 'sha256': source['files'][name]}})
        worker['header_view_parent_binding_ref'] = f.add('/frozen/header-proof.json', proof)
        args = (worker, {'worker': source}, worker['original_TRAIN_input_ref'],
                worker['original_derived_validation_input_ref'], f.read)
        inputs.real_input_binding(*args)
        proof['rows'][1]['payload_sha256'] = '7' * 64
        worker['header_view_parent_binding_ref'] = f.add('/frozen/header-proof.json', proof)
        with self.assertRaisesRegex(ValueError, 'payload changed'):
            inputs.real_input_binding(*args)

    def test_monitor_failure_is_not_promoted_by_physical_exit_zero(self):
        f = Fixture(); cell = f.snapshot['sample_cells'][0]
        f.change_receipt(cell, 'sample_exit_ref', lambda row: row.update(owned_signal_calls=1))
        with self.assertRaisesRegex(ValueError, 'integer identity'):
            f.prepare()
        cell['status'] = 'unavailable_monitor'
        with self.assertRaisesRegex(ValueError, 'accounting changed'):
            f.prepare()

    def test_float_fit_seed_and_request_rows_do_not_match_integers(self):
        for change in (lambda q: q['learning_identity'].update(fit_seed=11.),
                       lambda q: q['samples'][0].update(rows=10.)):
            f = Fixture(); cell = f.snapshot['sample_cells'][0]
            f.change_receipt(cell, 'sample_request_ref', change)
            f.change_receipt(cell, 'sample_Popen_ref', lambda row: row.update(request_ref=cell['sample_request_ref']))
            with self.assertRaisesRegex(ValueError, 'identity|default fit'):
                f.prepare()

    def test_stale_process_or_deadline_is_rejected(self):
        for change in (lambda row: row.update(start_ticks='101'),
                       lambda row: row.update(completed_monotonic=701.)):
            f = Fixture(); f.change_receipt(f.snapshot['sample_cells'][0], 'sample_exit_ref', change)
            with self.assertRaisesRegex(ValueError, 'process join|completion'):
                f.prepare()

    def test_kind_map_and_csv_order_are_bound_to_real_training(self):
        for field, value in (('sample_CSV_to_common_permutation', [1, 2, 0]),
                             ('column_kinds_common_order', ['categorical'] * 3),
                             ('original_lineage_deadline', 50000.)):
            f = Fixture(); next(iter(f.snapshot['workers'].values()))[field] = value
            with self.assertRaisesRegex(ValueError, 'order|kinds|clock|default fit'):
                f.prepare()

    def test_excluded_and_duplicate_logical_cells_fail_closed(self):
        f = Fixture(); f.snapshot['sample_cells'][2]['eligible_for_evaluator'] = True
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            f.prepare()
        f = Fixture(); f.snapshot['sample_cells'][2] = copy.deepcopy(f.snapshot['sample_cells'][0])
        with self.assertRaisesRegex(ValueError, 'accounting|duplicated'):
            f.prepare()

    def test_local_alias_and_invalid_digest_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3] / 'target') as directory:
            path = Path(directory) / 'receipt.json'; path.write_text('{}')
            alias = Path(directory) / 'alias'; alias.symlink_to(path)
            ref = {'path': str(alias), 'sha256': hashlib.sha256(b'{}').hexdigest()}
            with self.assertRaisesRegex(ValueError, 'alias'):
                inputs.read_local(ref)
        with self.assertRaisesRegex(ValueError, 'reference'):
            inputs.reference({'path': '/frozen/receipt.json', 'sha256': 0})


if __name__ == '__main__':
    unittest.main()
