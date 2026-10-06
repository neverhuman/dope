"""Native audit admission and rights-safe deterministic publication controls."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import jsonschema

from research.benchmark import publish_tabsyn_native_audit as p


def fixture():
    def ref(name): return {'path': '/opaque/' + name, 'sha256': hashlib.sha256(name.encode()).hexdigest()}
    names = ('request', 'Popen', 'exit', 'complete', 'csv', 'calls', 'closure', 'fit')
    refs = {name: ref(name) for name in names}
    sources = {'/opaque/entry.py': 'c' * 64}
    files = {'model': {'bytes': 2, 'sha256': 'd' * 64}, 'projection': {'bytes': 1, 'sha256': 'e' * 64}}
    fit = {'dataset_id': '0' * 16, 'lineage_sha256': 'a' * 64, 'fit_seed': 11,
           'fit_request': refs['fit'], 'artifact_files': files, 'charged_artifact_bytes': 3,
           'model_bytes': 2, 'projection_bytes': 1}
    row = {'dataset': fit['dataset_id'], 'lineage_sha256': fit['lineage_sha256'],
           'config_sha256': p.CONFIG_SHA, 'fit_request_ref': refs['fit'],
           'artifact_inventory': {'files': files, 'artifact_bytes': 3, 'projection_bytes': 1},
           'logical_closure_ref': refs['closure'], 'original_lineage_started_unix_seconds': 1.,
           'original_lineage_deadline_unix_seconds': 43201., 'status': 'native_audit_admitted',
           'scientific_native_Popen': 1, 'native_input_status': 'admitted_retained',
           'request_ref': refs['request'], 'Popen_ref': refs['Popen'], 'actual_exit_ref': refs['exit'],
           'observed_physical_native_complete_ref': refs['complete'], 'physical_csv_readback_ref': refs['csv'],
           'native_auditor_calls_ref': refs['calls'], 'actual_worker_exit': 0,
           'measured_whole_operation_seconds': 2., 'owned_signal_calls': 0, 'owned_termination_reason': None,
           'deadline_rechecked': True, 'native_result': dict(p.INAPPLICABLE), 'actual_XGB_fit_calls': 0,
           'physical_native_attempt': 1}
    request = {'learning_identity': {'dataset': fit['dataset_id'], 'source_commit': p.AUTHOR, 'fit_seed': 11},
               'lineage_sha256': fit['lineage_sha256'], 'artifact_code_ref': refs['fit'], 'mode': 'native',
               'config': {'config_sha256': p.CONFIG_SHA}, 'source_files': sources,
               'entry_ref': {'path': '/opaque/entry.py', 'sha256': 'c' * 64},
               'lineage_started_unix_seconds': 1., 'sample_backend': 'cpu', 'native_backend': 'cpu_hist',
               'official_tests_opened': False, 'production_certified': False, 'generated_fixture': False,
               'scientific_fit_allocations': 0, 'physical_csv_readback_ref': refs['csv'],
               'joint_synthetic_sha256': 'f' * 64, 'original_tensor_joint_sha256': 'b' * 64,
               'synthetic_ref': ref('sample')}
    popen = {'request_ref': refs['request'], 'CUDA_VISIBLE_DEVICES': '', 'scientific_worker_pid': 7,
             'start_ticks': '99', 'cpu_slot': list(range(80, 96)), 'nice': 19}
    exit_row = {'scientific_worker_pid': 7, 'start_ticks': '99', 'actual_worker_exit': 0,
                'foreign_signals': 0, 'official_tests_opened': False, 'whole_operation_seconds': 2.,
                'owned_signal_calls': 0, 'owned_termination_reason': None, 'deadline_rechecked': True,
                'completed_monotonic': 2., 'deadline_monotonic': 600.}
    complete = {'status': 'ok', 'lineage_sha256': fit['lineage_sha256'], 'config_sha256': p.CONFIG_SHA,
                'mode': 'native', 'native_backend': 'cpu_hist', 'result': dict(p.INAPPLICABLE),
                'author_gpu_hist_reproduction': False, 'official_tests_opened': False,
                'production_certified': False, 'mfs_v2': None, 'ptf_v1': None}
    csv = {'lineage_sha256': fit['lineage_sha256'], 'artifact_code_ref': refs['fit'],
           'physical_csv_joint_sha256': 'f' * 64, 'original_tensor_joint_sha256': 'b' * 64,
           'synthetic_ref': ref('sample'), 'dtype': 'float64', 'order': 'C', 'byteorder': 'little',
           'scientific_values_or_headers_logged': False}
    calls = {'request_sha256': refs['request']['sha256'], 'actual_XGB_fit_calls': 0,
             'auditor_configurations': 36, 'postselection_refits': 4, 'native_backend': 'cpu_hist',
             'author_gpu_hist_reproduction': False}
    closure = {'dataset': fit['dataset_id'], 'lineage_sha256': fit['lineage_sha256'],
               'stage': 'native_audit', 'official_tests_opened': False, 'scientific_fit_allocations': 0,
               'original_lineage_start': 1., 'operations': [{'status': 'ok'}]}
    receipts = dict(zip((refs[name]['path'] for name in names[:-1]),
                        (request, popen, exit_row, complete, csv, calls, closure)))
    return row, fit, receipts, sources


class TabSynNativeAudit(unittest.TestCase):
    def cell(self, f):
        row, fit, receipts, sources = f
        return p.native_cell(row, fit, lambda ref: receipts[ref['path']], sources)

    def test_uninformative_author_objective_is_not_native_winner(self):
        cell = self.cell(fixture())
        self.assertIsNone(cell['native_value']); self.assertIsNone(cell['native_winner'])
        self.assertTrue(cell['default_retained_by_native_audit'])
        for change in ('winner', 'kpi', 'auditor_calls', 'cuda', 'source'):
            f = fixture(); row, _, receipts, _ = f
            if change == 'winner': receipts['/opaque/complete']['result']['native_winner'] = 'tuned'
            if change == 'kpi': receipts['/opaque/complete']['result']['native_value'] = .99
            if change == 'auditor_calls': receipts['/opaque/calls']['actual_XGB_fit_calls'] = 1
            if change == 'cuda': receipts['/opaque/Popen']['CUDA_VISIBLE_DEVICES'] = '0'
            if change == 'source': receipts['/opaque/request']['entry_ref']['sha256'] = '0' * 64
            with self.subTest(change=change), self.assertRaises(ValueError): self.cell(f)

    def test_zero_exit_with_missing_monitoring_is_unavailable(self):
        f = fixture(); row, _, receipts, _ = f
        row.update(status='native_audit_unavailable', owned_signal_calls=1,
                   owned_termination_reason='owned_process_inspection_unavailable', deadline_rechecked=False)
        del row['native_result']
        receipts['/opaque/exit'].update(owned_signal_calls=1,
                   owned_termination_reason='owned_process_inspection_unavailable', deadline_rechecked=False)
        cell = self.cell(f)
        self.assertEqual(cell['actual_worker_exit'], 0); self.assertFalse(cell['native_result_admitted'])
        row['status'] = 'native_audit_admitted'; row['native_result'] = dict(p.INAPPLICABLE)
        with self.assertRaises(ValueError): self.cell(f)

    def test_identity_deadline_and_byte_charge_rejections(self):
        for change in ('bool_exit', 'seed_float', 'pid', 'clock', 'csv', 'bytes', 'late'):
            f = fixture(); row, fit, receipts, _ = f
            if change == 'bool_exit': receipts['/opaque/exit']['actual_worker_exit'] = False
            if change == 'seed_float': receipts['/opaque/request']['learning_identity']['fit_seed'] = 11.
            if change == 'pid': receipts['/opaque/exit']['scientific_worker_pid'] = 8
            if change == 'clock': row['original_lineage_deadline_unix_seconds'] += 600
            if change == 'csv': receipts['/opaque/csv']['physical_csv_joint_sha256'] = '0' * 64
            if change == 'bytes': fit['charged_artifact_bytes'] += 1
            if change == 'late': receipts['/opaque/exit']['completed_monotonic'] = 601
            with self.subTest(change=change), self.assertRaises(ValueError): self.cell(f)

    def report(self):
        return json.loads((p.RESULTS / (p.NAME + '.json')).read_bytes())

    def test_complete_ledger_and_prior_failure_costs(self):
        report = self.report(); s = p.summarize(report['cells'])
        self.assertEqual(s, report['summary']); self.assertEqual(s['status_counts'], p.COUNTS)
        self.assertEqual(s['physical_workers_exit_zero'], 91); self.assertEqual(s['prior_failed_native_attempts'], 68)
        self.assertAlmostEqual(s['measured_current_native_whole_seconds'], 1166.2178418077528)
        self.assertAlmostEqual(s['prior_failed_native_whole_seconds'], 271.99935615994036)
        self.assertEqual(s['charged_model_and_projection_bytes'], 8486920261)
        self.assertEqual(report['publisher_sha256'], hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest())
        for cells in (report['cells'][:-1], report['cells'][:-1] + [report['cells'][0]]):
            with self.assertRaises(ValueError): p.summarize(cells)

    def test_tables_schema_exact_regeneration_and_no_source_values(self):
        report = self.report(); path = p.RESULTS / (p.NAME + '.json'); table, markdown = p.tables(report)
        self.assertEqual(table, path.with_suffix('.csv').read_text())
        self.assertEqual(markdown, path.with_suffix('.md').read_text())
        self.assertEqual(p.schema(report), json.loads(path.with_suffix('.schema.json').read_bytes()))
        jsonschema.validate(report, p.schema(report))
        self.assertEqual(len(table.splitlines()), 101)
        for forbidden in ('native_metadata', 'extrema', 'train.csv', 'validation.csv', 'XGB_parameter_sha256'):
            self.assertNotIn('"' + forbidden + '"', path.read_text())

    def test_no_claims_or_monitoring_upgrade_in_publication(self):
        original = self.report(); schema = p.schema(original)
        for key in (*p.NULL_CLAIMS, *p.FALSE_CLAIMS):
            report = copy.deepcopy(original); report[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError): p.tables(report)
            with self.subTest(schema=key), self.assertRaises(jsonschema.ValidationError): jsonschema.validate(report, schema)
        report = copy.deepcopy(original)
        cell = next(x for x in report['cells'] if x['status'] == 'native_audit_unavailable')
        cell['native_result_admitted'] = True
        with self.assertRaises(ValueError): p.tables(report)


if __name__ == '__main__': unittest.main()
