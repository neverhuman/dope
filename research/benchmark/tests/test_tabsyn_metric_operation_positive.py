"""Generated metadata only: no numeric, model, runtime, or evaluator execution."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import tabsyn_metric_operation_inputs as bridge
from research.benchmark import tabsyn_validation_inputs as validation
from research.benchmark.tests.test_tabsyn_validation_inputs import Fixture


class CompleteFixture(Fixture):
    """Extend the existing 100-fit/600-cell fixture with complete receipt joins.

    Source pins are synthetic and patched only while these tests call the real
    metadata functions. Learned/input bodies do not exist in this fixture.
    """
    def __init__(self):
        super().__init__()
        self.sources, self.source_calls = {}, []
        self.adapter = self.source('/generated/validation_adapter.py',
            b'from research.benchmark.tabsyn_validation_inputs import '
            b'AUTHOR, CONFIG, ADMITTED, prepare_cell\n')
        self.author = self.source('/generated/author_adapter.py', b'# generated author source\n')
        self.entry = self.source('/generated/fit_entry.py', b'# generated fit entry\n')
        self.evaluator = self.source('/generated/evaluator.py', b'# generated evaluator source\n')
        self.producer = self.source('/generated/kinds.py', b'# generated TRAIN-only kind producer\n')
        self.writers = {backend: self.source(f'/generated/{name}.py', data)
            for backend, name, data in (
                ('cpu', 'cpu_writer', b'# generated CPU writer\n'),
                ('cuda:0', 'gpu_writer', b'# generated GPU writer\n'))}
        self.runtime = self.add('/generated/runtime.json', {'format': 'generated-runtime-metadata-only'})
        self.phase = copy.deepcopy(self.decoded[self.snapshot['fit_phase100_ref']['path']])
        self.worker = self.snapshot['workers'][f'{0:064x}']
        self.fit = self.phase['lineages'][0]
        self.worker['artifact_inventory']['files']['model']['sha256'] = '7' * 64
        rule = {'format': 'generated-TRAIN-only-kind-rule', 'version': 1}
        catalog = {}
        for worker, fit in zip(self.snapshot['workers'].values(), self.phase['lineages']):
            kinds = copy.deepcopy(self.decoded[worker['kind_map_ref']['path']])
            kinds.update(rule=rule, rule_sha256=bridge.digest(rule),
                         producer_source_sha256=self.producer['sha256'],
                         author_semantic_types=None, author_semantic_categorical_tv_available=False)
            worker.update(kind_map_ref=self.add(worker['kind_map_ref']['path'], kinds),
                          kind_rule_sha256=kinds['rule_sha256'],
                          kind_producer_source_sha256=self.producer['sha256'])
            fit.update({key: copy.deepcopy(worker[key]) for key in (
                'kind_map_ref', 'kind_rule_sha256', 'kind_producer_source_sha256')})
            catalog[worker['kind_worker_key']] = {
                'path': f"{worker['dataset']}/kinds.json", 'sha256': worker['kind_map_ref']['sha256']}
        self.snapshot.update(evaluator_ref=self.evaluator,
            deterministic_order_evidence={'adapter_ref': self.author},
            actual100_kind_catalog_ref=self.add('/generated/kind-catalog.json', {
                'format': 'dope100-actual-kind-map-catalog', 'official_tests_opened': False,
                'numerical_source_changes': False, 'kind_maps': catalog}))
        original = self.decoded[self.worker['fit_request_ref']['path']]
        self.fit_request = dict(copy.deepcopy(original), mode='fit',
            lineage_sha256=self.fit['lineage_sha256'], deadline_monotonic=600.,
            adapter_ref=self.author, entry_ref=self.entry, runtime_ref=self.runtime,
            source_files={ref['path']: ref['sha256'] for ref in (self.author, self.entry)})
        for cell in self.snapshot['sample_cells'][:2]:
            writer = self.writers[cell['physical_backend']]
            self.replace_sample(cell, lambda q: q.update(
                entry_ref=writer, adapter_ref=self.author, runtime_ref=self.runtime,
                source_files={ref['path']: ref['sha256'] for ref in (writer, self.author)}))
        self.rebind_fit()

    def source(self, path, data):
        self.sources[path] = data
        return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def read_source(self, ref):
        self.source_calls.append(ref['path'])
        return self.sources[ref['path']]

    def replace_sample(self, cell, mutate):
        self.change_receipt(cell, 'sample_request_ref', mutate)
        self.change_receipt(cell, 'sample_Popen_ref',
            lambda row: row.update(request_ref=cell['sample_request_ref'], uid=1000))

    def rebind_fit(self, mutate=lambda request: None):
        mutate(self.fit_request)
        request_ref = self.add(self.worker['fit_request_ref']['path'], self.fit_request)
        result = {'complete': True, 'seed': 11, 'rows': 10, 'columns': 3,
            'code_sha256': request_ref['sha256'], 'operation_admission_sha256': request_ref['sha256'],
            'config': self.fit_request['config'], 'source_commit': validation.AUTHOR,
            'runtime_sha256': self.fit_request['runtime_ref']['sha256'],
            'generated_fixture': False, 'official_tests_opened': False}
        complete_ref = self.add(self.worker['fit_complete_ref']['path'], {
            'status': 'ok', 'method': 'TabSyn', 'lineage_sha256': self.fit['lineage_sha256'],
            'config_sha256': validation.CONFIG, 'official_tests_opened': False,
            'production_certified': False, 'mfs_v2': None, 'ptf_v1': None,
            'result': {'result': result, 'inventory': self.worker['artifact_inventory']}})
        popen_ref = self.add('/generated/fit/actual-scientific-Popen.receipt.json', {
            'request_ref': request_ref, 'scientific_worker_pid': 123, 'uid': 1000,
            'start_ticks': '100', 'whole_deadline_monotonic': 600., 'official_tests_opened': False})
        exit_ref = self.add(self.worker['fit_exit_ref']['path'], {
            'scientific_worker_pid': 123, 'start_ticks': '100', 'actual_worker_exit': 0,
            'foreign_signals': 0, 'success_receipt_ref': complete_ref})
        self.worker.update(fit_request_ref=request_ref, fit_complete_ref=complete_ref, fit_exit_ref=exit_ref)
        self.fit.update(fit_request_ref=request_ref, fit_complete_ref=complete_ref, fit_exit_ref=exit_ref,
            artifact_inventory=copy.deepcopy(self.worker['artifact_inventory']), retained_first=False,
            actual_metadata_refs=[request_ref, popen_ref, complete_ref, exit_ref])
        for cell in self.snapshot['sample_cells'][:2]:
            self.replace_sample(cell, lambda q: q.update(artifact_code_ref=request_ref))

    def operation(self, position=0):
        self.snapshot['fit_phase100_ref'] = self.add('/frozen/phase.json', self.phase)
        snapshot_ref = self.add('/generated/snapshot.json', self.snapshot)
        cell = self.snapshot['sample_cells'][position]
        with patch.object(bridge, 'ADAPTER_SHA256', self.adapter['sha256']), \
             patch.object(bridge, 'EVALUATOR_SHA256', self.evaluator['sha256']), \
             patch.object(bridge, 'KINDS_REF', self.producer), \
             patch.object(validation, 'WRITERS', {k: v['sha256'] for k, v in self.writers.items()}):
            return bridge.prepare_operation(snapshot_ref, self.adapter, cell['lineage_sha256'],
                cell['seed'], cell['row_multiplier'], metadata_reader=self.read,
                source_reader=self.read_source)


class CompleteOperationControls(unittest.TestCase):
    def test_complete_CPU_and_GPU_operations_join_sources_kinds_without_body_reads(self):
        for position, backend in ((0, 'cpu'), (1, 'cuda:0')):
            with self.subTest(backend=backend):
                f = CompleteFixture(); got = f.operation(position)
                self.assertEqual(got['status'], 'prepared_metadata_only')
                self.assertEqual(got['job_sha256'], bridge.digest(got['job_identity']))
                self.assertEqual(got['required_runtime_ref'], f.runtime)
                self.assertEqual(got['evaluator_input']['physical_sampling_backend'], backend)
                self.assertEqual(got['evaluator_input']['sample_CSV_to_common_permutation'], [0, 1, 2])
                self.assertIsNone(got['additional_column_permutation'])
                self.assertIsNone(got['historical_parent_exit'])
                self.assertIsNone(got['mfs_v2']); self.assertIsNone(got['superiority'])
                self.assertFalse(got['current_body_bytes_verified'])
                self.assertFalse(got['execution_enabled'])
                expected = [f.author, f.entry, f.writers[backend]]
                self.assertEqual(got['scientific_source_refs'], sorted(
                    [{'path': r['path'], 'sha256': r['sha256']} for r in expected], key=lambda r: r['path']))
                self.assertEqual(got['evaluator_input']['kind_map_ref'], f.worker['kind_map_ref'])
                self.assertIn(f.producer['path'], f.source_calls)
                self.assertIn(f.evaluator['path'], f.source_calls)
                self.assertTrue(set(f.calls).isdisjoint(
                    r['path'] for r in got['required_body_refs_before_dependency_import']))
                self.assertNotIn(f.runtime['path'], f.calls)  # Exposed prerequisite, not runtime certification.

    def test_author_source_omission_in_either_operation_is_rejected(self):
        for operation in ('fit', 'sample'):
            with self.subTest(operation=operation):
                f = CompleteFixture()
                if operation == 'fit':
                    f.rebind_fit(lambda q: q['source_files'].pop(f.author['path']))
                else:
                    f.replace_sample(f.snapshot['sample_cells'][0],
                        lambda q: q['source_files'].pop(f.author['path']))
                with self.assertRaisesRegex(ValueError, 'executable_missing_from_source_inventory'):
                    f.operation()

    def test_fitted_entry_omission_is_rejected_even_with_valid_completion_receipts(self):
        f = CompleteFixture()
        f.rebind_fit(lambda q: q['source_files'].pop(f.entry['path']))
        with self.assertRaisesRegex(ValueError, 'executable_missing_from_source_inventory'):
            f.operation()

    def test_fit_request_lineage_cannot_be_rebound_to_another_fit(self):
        f = CompleteFixture()
        f.rebind_fit(lambda q: q.update(lineage_sha256=f'{1:064x}'))
        with self.assertRaisesRegex(ValueError, 'fit_request_lineage_changed'):
            f.operation()

    def test_sample_runtime_reference_drift_is_rejected(self):
        for field, value in (('sha256', '9' * 64), ('path', '/generated/other-runtime.json'), ('bytes', 999)):
            with self.subTest(field=field):
                f = CompleteFixture()
                f.replace_sample(f.snapshot['sample_cells'][0],
                    lambda q: q.update(runtime_ref=dict(q['runtime_ref'], **{field: value})))
                with self.assertRaisesRegex(ValueError, 'sample_runtime_differs_from_fit'):
                    f.operation()

    def test_all_600_cell_eligibilities_are_checked_even_when_selected_cell_is_valid(self):
        for value in (True, 0):
            with self.subTest(value=value):
                f = CompleteFixture(); f.snapshot['sample_cells'][-1]['eligible_for_evaluator'] = value
                with self.assertRaisesRegex(ValueError, 'cell_eligibility_changed'):
                    f.operation()

    def test_admitted_population_count_is_exact_builtin_integer(self):
        for value in (3, 2., True):
            with self.subTest(value=value):
                f = CompleteFixture(); f.snapshot['admitted_snapshot_inputs'] = value
                with self.assertRaisesRegex(ValueError, 'snapshot_admitted_count_changed'):
                    f.operation()

    def test_kind_catalog_join_cannot_substitute_another_map(self):
        f = CompleteFixture(); ref = f.snapshot['actual100_kind_catalog_ref']
        value = copy.deepcopy(f.decoded[ref['path']])
        value['kind_maps'][f.worker['kind_worker_key']]['sha256'] = '8' * 64
        f.snapshot['actual100_kind_catalog_ref'] = f.add(ref['path'], value)
        with self.assertRaisesRegex(ValueError, 'kind_catalog_join_changed'):
            f.operation()

    def test_scientific_source_buffer_drift_is_not_turned_into_unavailable(self):
        f = CompleteFixture(); f.sources[f.entry['path']] += b'# drift\n'
        with self.assertRaisesRegex(ValueError, 'pinned_bytes_changed'):
            f.operation()


if __name__ == '__main__':
    unittest.main()
