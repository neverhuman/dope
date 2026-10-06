"""Private standard-library controls on generated metadata, no study data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import publish_work_order_b_costs as p


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.costs, cls.gaps, cls.proof = [p.bound(p.HERE / 'results' / 'work-order-b-costs-v1' / name, p.INPUT_PINS[name], p.HERE)
            for name in ('measured-costs.json', 'component-gaps.json', 'source-proof.json')]
        repo = p.HERE.parents[1]
        cls.reports = {name: p.bound(repo / 'research/benchmark/results' / name, pin, repo)
                       for name, pin in cls.costs['source_reports_sha256'].items()}

    def temporary(self):
        # Host rules prohibit /tmp build/evidence directories.
        directory = tempfile.TemporaryDirectory(prefix='controls-', dir=p.HERE.parents[1] / 'target' / 'tmp')
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def receipt(self, **changes):
        return dict(identity='a' * 64 + ':fit', method='generated_fixture', phase='fit', seconds=1.25,
                    host='generated_host', status='ok', new_operation_started=True,
                    native_outcome_class='ok', peak_ram_bytes=None, peak_device_used_mib=None, **changes)

    def test_digest_before_decode(self):
        root = self.temporary(); path = root / 'generated.json'; path.write_bytes(b'not JSON')
        with patch.object(p.json, 'loads', side_effect=AssertionError('decode called')) as decode:
            with self.assertRaisesRegex(ValueError, 'digest changed'):
                p.bound(path, '0' * 64, root)
        decode.assert_not_called()

    def test_json_nonfinite_before_numeric_validation(self):
        root = self.temporary(); path = root / 'generated.json'
        for raw in (b'{"cost":NaN}', b'{"cost":Infinity}', b'{"cost":-Infinity}', b'{"cost":1e999}'):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'nonfinite'):
                    p.bound(path, hashlib.sha256(raw).hexdigest(), root)

    def test_json_duplicate_key(self):
        root = self.temporary(); path = root / 'generated.json'; raw=b'{"cost":1,"cost":2}'
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            p.bound(path, hashlib.sha256(raw).hexdigest(), root)

    def test_bool_nan_negative_costs(self):
        for value in (True, False, float('nan'), float('inf'), -1, '1'):
            with self.subTest(value=value):
                costs = copy.deepcopy(self.costs); costs['rows'][0]['operation_wall_seconds'] = value
                with self.assertRaises(ValueError):
                    p.validate_costs(costs)

    def test_duplicate_physical_alias_charged_once(self):
        row = self.receipt(); result = p.aggregate([row, copy.deepcopy(row)])
        self.assertEqual(result[0]['distinct_phase_receipts'], 1)
        self.assertEqual(result[0]['operation_wall_seconds'], 1.25)

    def test_conflicting_alias(self):
        row = self.receipt(); changed = dict(row, seconds=2.5)
        with self.assertRaisesRegex(ValueError, 'conflicting physical alias'):
            p.aggregate([row, changed])

    def test_alias_numeric_type_distinction(self):
        row = dict(self.receipt(), seconds=1); changed = dict(row, seconds=1.0)
        with self.assertRaisesRegex(ValueError, 'conflicting physical alias'):
            p.aggregate([row, changed])

    def test_scheduling_cut_is_not_method_failure(self):
        row = dict(self.receipt(), seconds=0.0, status='deadline_unstarted',
                   new_operation_started=False, native_outcome_class='scheduling_cutoff')
        result = p.aggregate([row])
        self.assertEqual(result[0]['native_outcome_class_counts'], {'scheduling_cutoff':1})
        self.assertEqual(result[0]['new_phase_operations_started'], 0)
        with self.assertRaisesRegex(ValueError, 'scheduling cut relabeled'):
            p.aggregate([dict(row, native_outcome_class='method_failure')])

    def test_nullable_scores_and_hard_gates(self):
        p.validate_gaps(self.gaps)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority', 'hard_gate_passed'):
            with self.subTest(key=key):
                gaps = copy.deepcopy(self.gaps); gaps[key] = 0.0
                with self.assertRaises(ValueError): p.validate_gaps(gaps)
        self.assertEqual(p.schema(self.gaps)['properties']['mfs_v2'], {'type':'null'})
        for index in range(6):
            gaps = copy.deepcopy(self.gaps); gaps['components'][index]['score'] = 0.0
            with self.assertRaises(ValueError): p.validate_gaps(gaps)

    def test_counts_reject_bools(self):
        costs = copy.deepcopy(self.costs); costs['rows'][0]['ram_observed_operations'] = True
        with self.assertRaises(ValueError): p.validate_costs(costs)
        gaps = copy.deepcopy(self.gaps); gaps['components'][1]['known_fit_values'] = False
        with self.assertRaises(ValueError): p.validate_gaps(gaps)

    def test_repeat_byte_identical_projections(self):
        first = p.projections(self.costs, self.gaps, self.proof)
        second = p.projections(copy.deepcopy(self.costs), copy.deepcopy(self.gaps), copy.deepcopy(self.proof))
        self.assertEqual(first, second)
        self.assertLessEqual(len(first['measured-costs.json']), 20000)
        self.assertLessEqual(len(first['source-proof.json']), 20000)
        published = json.loads(first['measured-costs.json'])
        self.assertEqual(published['run_accounting']['density_shared_validation_v1'], 'alias_of_density_phase_rows_do_not_add')

    def test_deterministic_publish_and_preserve_existing(self):
        output = self.temporary()
        with patch.object(p, 'verify_sources', return_value={'generated':1}):
            first = p.publish(p.HERE, p.HERE / 'results' / 'work-order-b-costs-v1', output)
            before = {path.name:path.read_bytes() for path in output.iterdir()}
            second = p.publish(p.HERE, p.HERE / 'results' / 'work-order-b-costs-v1', output)
            self.assertEqual(first, second)
            self.assertEqual(before, {path.name:path.read_bytes() for path in output.iterdir()})
            path = output / 'component-gaps.csv'; path.write_bytes(b'generated drift')
            with self.assertRaisesRegex(ValueError, 'existing publication differs'):
                p.publish(p.HERE, p.HERE / 'results' / 'work-order-b-costs-v1', output)

    def test_run_costs_are_reproduced_from_bound_reports(self):
        p.verify_run_costs(self.costs, self.reports)
        costs = copy.deepcopy(self.costs)
        costs['runs'][0]['operation_wall_seconds'] += 1
        with self.assertRaisesRegex(ValueError, 'source or cost changed'):
            p.verify_run_costs(costs, self.reports)

    def test_run_cost_source_and_scheduler_bindings(self):
        for field, value in [('source', 'generated-report.json'),
                             ('scheduler_wall_seconds', 1),
                             ('host_operation_seconds', {'generated_host': 1})]:
            with self.subTest(field=field):
                costs = copy.deepcopy(self.costs); costs['runs'][0][field] = value
                with self.assertRaisesRegex(ValueError, 'source or cost changed'):
                    p.verify_run_costs(costs, self.reports)

    def test_prior_cost_is_bound_and_not_multiplied(self):
        costs = copy.deepcopy(self.costs)
        costs['historical_prior_costs']['dope_closed_fit_seconds'] *= 2
        with self.assertRaisesRegex(ValueError, 'historical prior cost changed'):
            p.verify_run_costs(costs, self.reports)

    def test_observed_operation_count_cannot_exceed_receipts(self):
        for field in ['new_phase_operations_started', 'ram_observed_operations', 'vram_observed_operations']:
            with self.subTest(field=field):
                costs = copy.deepcopy(self.costs)
                costs['rows'][0][field] = costs['rows'][0]['distinct_phase_receipts'] + 1
                with self.assertRaisesRegex(ValueError, 'exceed receipts'):
                    p.validate_costs(costs)

    def test_run_method_phase_and_host_are_bound(self):
        for index, field, value in [(0, 'method', 'other_method'), (0, 'phase', 'other_phase'),
                                    (1, 'host', 'other_host')]:
            with self.subTest(field=field):
                costs = copy.deepcopy(self.costs); costs['runs'][index][field] = value
                with self.assertRaisesRegex(ValueError, 'source or cost changed'):
                    p.verify_run_costs(costs, self.reports)

    def test_added_run_device_cost_rejects(self):
        costs = copy.deepcopy(self.costs); costs['runs'][1]['peak_device_used_mib'] = -1
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            p.validate_costs(costs)
        costs['runs'][1]['peak_device_used_mib'] = 1
        with self.assertRaisesRegex(ValueError, 'unsupported measured run provenance'):
            p.verify_run_costs(costs, self.reports)

    def test_six_component_identity_and_coverage(self):
        for change in ('omit', 'duplicate'):
            gaps = copy.deepcopy(self.gaps)
            if change == 'omit': gaps['components'].pop()
            else: gaps['components'] = [copy.deepcopy(gaps['components'][-1])] * 6
            with self.assertRaisesRegex(ValueError, 'component coverage'):
                p.validate_gaps(gaps)

    def test_hard_gate_identity_and_coverage(self):
        for change in ('omit', 'duplicate', 'empty'):
            gaps = copy.deepcopy(self.gaps)
            if change == 'omit': gaps['hard_gates'].pop()
            elif change == 'empty': gaps['hard_gates'] = []
            else: gaps['hard_gates'][1] = copy.deepcopy(gaps['hard_gates'][0])
            with self.assertRaisesRegex(ValueError, 'hard gate coverage'):
                p.validate_gaps(gaps)

    def test_new_metric_job_or_bool_count_rejects(self):
        for value in (1, True, False):
            gaps = copy.deepcopy(self.gaps); gaps['new_metric_jobs_started'] = value
            with self.assertRaisesRegex(ValueError, 'metric execution claim'):
                p.validate_gaps(gaps)

    def test_source_proof_retains_null_gates_and_zero_execution(self):
        for key, value in [('mfs_v2', .5), ('ptf_v1', .5), ('release_safe', True),
                           ('superiority', True), ('official_tests_opened', True),
                           ('new_metric_jobs_started', 1), ('new_training_or_samples_started', 1),
                           ('logical_aliases_charged_again', True)]:
            with self.subTest(key=key):
                proof = copy.deepcopy(self.proof); proof[key] = value
                with self.assertRaises(ValueError):
                    p.projections(self.costs, self.gaps, proof)

    def test_source_contract_traversal_rejected_before_read(self):
        root = self.temporary(); repo = root / 'repo'; repo.mkdir()
        outside = root / 'outside.json'; outside.write_bytes(b'{}')
        proof = dict(self.proof, source_code_and_contract=[{
            'path': '../outside.json', 'sha256': hashlib.sha256(b'{}').hexdigest()}])
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('outside read')) as read:
            with self.assertRaisesRegex(ValueError, 'path'):
                p.verify_sources(repo, {'source_reports_sha256': {}}, {}, proof, root)
        read.assert_not_called()

    def test_unstarted_receipt_cannot_acquire_operation_cost(self):
        base = dict(self.receipt(), seconds=0, status='deadline_unstarted',
                    new_operation_started=False, native_outcome_class='scheduling_cutoff')
        for change in [{'new_operation_started': True}, {'seconds': 1}]:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, 'unstarted phase'):
                    p.aggregate([dict(base, **change)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
