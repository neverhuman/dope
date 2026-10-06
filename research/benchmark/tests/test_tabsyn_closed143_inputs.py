"""Prepared generated metadata controls. No execution/data dependencies."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import tabsyn_closed143_inputs as bridge
from research.benchmark import tabsyn_metric_operation_inputs as inputs


class Fixture:
    def __init__(self):
        self.buffers, self.calls = {}, []

    def add(self, path, value):
        data = inputs.canonical(value).encode()
        self.buffers[path] = data
        return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def local(self, ref):
        self.calls.append(ref['path'])
        if ref['path'] not in self.buffers:
            raise FileNotFoundError('generated_missing_metadata')
        return self.buffers[ref['path']]

    def mapping(self, rows):
        return self.add('/frozen/custody-map.json', {
            'format': 'dope-TS-frozen-closed143-verified-metadata-source-byte-map-v1',
            'resolver_must_rehash_each_read': True, 'sample_or_model_payloads_copied': False,
            'original_TS_parent_kernel_exit': None, 'CSV_rows_decoded': False, 'entries': rows})

    def report(self):
        jobs = []
        for i in range(143):
            jobs.append({'lineage_sha256': f'{i:064x}', 'sample_seed': 101, 'row_multiplier': 1,
                         'column_kinds': ['continuous'] * 3,
                         'column_order': 'projected_inputs_then_target',
                         'sample_CSV_to_common_permutation': [0, 1, 2], 'synthetic_csv_header_rows': 1,
                         'historical_parent_exit': None, 'execution_enabled': False,
                         'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                         'release_safe_l3': None, 'superiority': None,
                         'train_ref': {'csv_format': 'headerless_numeric'},
                         'validation_ref': {'csv_format': 'headerless_numeric'}})
        return {'format': 'dope-tabsyn-shared-validation-input-preparation-v1', 'jobs': jobs,
                'execution_enabled': False, 'official_tests_opened': False, 'historical_parent_exit': None,
                'physical_samples_rehashed': False, 'numeric_rows_decoded': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'superiority': None,
                'sample_cells': 600, 'status_counts': {'admitted_new_CPU': 95, 'admitted_retained_GPU': 48,
                                                     'unavailable_aggregate_admission_gap': 8,
                                                     'unavailable_monitor': 31, 'unstarted': 418},
                'snapshot_ref': {'path': '/frozen/snapshot.json', 'sha256': 'a' * 64}}


class Controls(unittest.TestCase):
    def test_map_hash_is_checked_before_decoding(self):
        f = Fixture(); ref = f.mapping([]); f.buffers[ref['path']] = b'not JSON'
        with patch.object(inputs.json, 'loads') as loads:
            with self.assertRaisesRegex(ValueError, 'pinned_bytes_changed'):
                bridge.make_custody_reader(ref, local_reader=f.local)
            loads.assert_not_called()

    def test_resolver_uses_path_and_digest_and_rehashes_each_read(self):
        f = Fixture(); a = f.add('/copies/a.json', {'receipt': 'a'}); b = f.add('/copies/b.json', {'receipt': 'b'})
        rows = [dict(original_path='/remote/receipt.json', local_path=r['path'], bytes=r['bytes'], sha256=r['sha256'])
                for r in (a, b)]
        read = bridge.make_custody_reader(f.mapping(rows), local_reader=f.local)
        original = {'path': '/remote/receipt.json', 'sha256': b['sha256']}
        self.assertEqual(read(original), f.buffers[b['path']])
        f.buffers[b['path']] = b'changed'
        with self.assertRaisesRegex(ValueError, 'pinned_bytes_changed'):
            read(original)

    def test_digest_and_path_subclasses_are_rejected_before_custody_reads(self):
        class Forged(str):
            def __eq__(self, other): return True
            def __hash__(self): return hash(expected)
        f = Fixture(); ref = f.add('/copies/a.json', {'receipt': 'a'}); expected = ref['sha256']
        row = dict(original_path='/remote/receipt.json', local_path=ref['path'], bytes=ref['bytes'], sha256=expected)
        read = bridge.make_custody_reader(f.mapping([row]), local_reader=f.local)
        before = len(f.calls)
        for path, pin in ((row['original_path'], Forged('0' * 64)), (Forged(row['original_path']), expected)):
            with self.assertRaisesRegex(ValueError, 'invalid_custody_reference'):
                read({'path': path, 'sha256': pin})
        self.assertEqual(len(f.calls), before)

    def test_duplicate_custody_key_and_unknown_reference_reject(self):
        f = Fixture(); ref = f.add('/copies/a.json', {'ok': True})
        row = dict(original_path='/remote/receipt.json', local_path=ref['path'], bytes=ref['bytes'], sha256=ref['sha256'])
        with self.assertRaisesRegex(ValueError, 'duplicate_or_invalid_custody_entry'):
            bridge.make_custody_reader(f.mapping([row, row]), local_reader=f.local)
        read = bridge.make_custody_reader(f.mapping([row]), local_reader=f.local)
        with self.assertRaises(FileNotFoundError):
            read({'path': '/remote/other.json', 'sha256': ref['sha256']})

    def test_numeric_projection_model_bodies_never_reach_local_reader(self):
        f = Fixture(); read = bridge.make_custody_reader(f.mapping([]), local_reader=f.local)
        calls_before = len(f.calls)
        for name in ('train.csv', 'sample.csv', 'model.json', 'projection.json', 'weights.safetensors', 'numeric-state.npz'):
            with self.assertRaisesRegex(ValueError, 'learned_or_numeric_body_forbidden'):
                read({'path': '/remote/' + name, 'sha256': 'a' * 64})
        self.assertEqual(len(f.calls), calls_before)

    def test_original_declared_size_cannot_drift(self):
        f = Fixture(); ref = f.add('/copies/a.json', {'ok': True})
        row = dict(original_path='/remote/receipt.json', local_path=ref['path'], bytes=ref['bytes'], sha256=ref['sha256'])
        read = bridge.make_custody_reader(f.mapping([row]), local_reader=f.local)
        with self.assertRaisesRegex(ValueError, 'original_size_changed'):
            read({'path': row['original_path'], 'sha256': row['sha256'], 'bytes': row['bytes'] + 1})

    def test_exact143_recipe_selection_preserves_column_identity(self):
        report = Fixture().report()
        selected = bridge.select_recipe(report, f'{0:064x}', 101, 1)
        self.assertEqual(selected['sample_CSV_to_common_permutation'], [0, 1, 2])
        self.assertIsNone(selected['historical_parent_exit'])
        self.assertIsNone(bridge.select_recipe(report, f'{999:064x}', 101, 1))

    def test_duplicate_recipe_and_float_identity_refuse(self):
        for mutation in (lambda r: r['jobs'].__setitem__(1, copy.deepcopy(r['jobs'][0])),
                         lambda r: r['jobs'][0].update(sample_seed=101.)):
            report = Fixture().report(); mutation(report)
            with self.assertRaisesRegex(ValueError, 'population_changed|invalid_recipe_identity'):
                bridge.select_recipe(report, f'{0:064x}', 101, 1)

    def test_second_permutation_and_invented_parent_zero_refuse(self):
        for mutation in (lambda r: r['jobs'][0].update(sample_CSV_to_common_permutation=[1, 2, 0]),
                         lambda r: r.update(historical_parent_exit=0)):
            report = Fixture().report(); mutation(report)
            with self.assertRaisesRegex(ValueError, 'order_or_scope_changed|report_scope_changed'):
                bridge.select_recipe(report, f'{0:064x}', 101, 1)

    def test_recipe_must_equal_receipt_reconstructed_plan(self):
        f = Fixture(); report = f.report(); ref = f.add('/frozen/report.json', report)
        prepared = {'status': 'prepared_metadata_only', 'evaluator_input': dict(report['jobs'][0], rows=999),
                    'job_identity': {}}
        with patch.object(inputs, 'prepare_operation', return_value=prepared):
            with self.assertRaisesRegex(ValueError, 'recipe_differs_from_operation_receipt_bindings'):
                bridge.prepare_recipe(ref, {}, f'{0:064x}', 101, 1,
                                      metadata_reader=f.local, source_reader=f.local, report_reader=f.local)

    def test_matching_recipe_produces_no_second_permutation_or_authority(self):
        f = Fixture(); report = f.report(); ref = f.add('/frozen/report.json', report)
        prepared = {'status': 'prepared_metadata_only', 'evaluator_input': copy.deepcopy(report['jobs'][0]),
                    'job_identity': {'lineage': f'{0:064x}'}, 'execution_enabled': False,
                    'historical_parent_exit': None}
        with patch.object(inputs, 'prepare_operation', return_value=prepared):
            result = bridge.prepare_recipe(ref, {}, f'{0:064x}', 101, 1,
                                           metadata_reader=f.local, source_reader=f.local, report_reader=f.local)
        self.assertIsNone(result['execution_boundary']['additional_column_permutation'])
        self.assertEqual(result['execution_boundary']['synthetic_csv_format'], 'headered_numeric')
        self.assertFalse(result['execution_enabled'])
        self.assertIsNone(result['historical_parent_exit'])


if __name__ == '__main__':
    unittest.main()
