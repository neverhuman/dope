"""Opaque cohort, gate ordering and table controls; no live metrics."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research.benchmark import publish_density_matched as m
from research.benchmark.tests.test_density_matched_aggregate import fixture


def locks():
    names = ('train.csv', 'validation.csv', 'projection.json',
             'row-group-assignments.json', 'worker-manifest.json')
    jobs = [dict(dataset=f'opaque-{i:03}', worker=dict(files={n: 'a' * 64 for n in names},
            raw_training_derived_split_hashes=dict(train='b' * 64, validation='c' * 64),
            train_rows=10, projected_features=2)) for i in range(100)]
    return dict(jobs=jobs), dict(jobs=copy.deepcopy(jobs))


class Controls(unittest.TestCase):
    def test_all_views_match_without_rows(self):
        x, y = locks()
        result = m.matched_views(x, y)
        self.assertEqual(result['identical_five_projected_file_hashes'], 100)
        self.assertFalse(result['raw_rows_parsed'])
        self.assertFalse(result['official_tests_opened'])

    def test_missing_and_added_dataset_rejected(self):
        for added in (False, True):
            x, y = locks()
            if added: y['jobs'].append(dict(dataset='extra', worker=copy.deepcopy(y['jobs'][0]['worker'])))
            else: y['jobs'].pop()
            with self.subTest(added=added), self.assertRaises(ValueError): m.matched_views(x, y)

    def test_worker_changed_across_variants_rejected(self):
        x, y = locks()
        row = copy.deepcopy(x['jobs'][0]); row['worker']['train_rows'] = 11
        x['jobs'].append(row)
        with self.assertRaises(ValueError): m.matched_views(x, y)

    def test_file_and_raw_partition_and_dimension_drift_rejected(self):
        for field in ('files', 'raw_training_derived_split_hashes', 'projected_features'):
            x, y = locks()
            w = y['jobs'][0]['worker']
            if field == 'files': w[field]['projection.json'] = 'd' * 64
            elif field == 'raw_training_derived_split_hashes': w[field]['train'] = 'd' * 64
            else: w[field] = 3
            with self.subTest(field=field), self.assertRaises(ValueError): m.matched_views(x, y)

    def test_numeric_metadata_not_matched_by_python_equality(self):
        x, y = locks(); y['jobs'][0]['worker']['train_rows'] = 10.0
        with self.assertRaises(ValueError): m.matched_views(x, y)

    def test_partial_gate_blocks_before_metrics_runtime_and_dope(self):
        with patch.object(m.inputs, 'load_closed_inputs', side_effect=ValueError('incomplete')), \
             patch.object(m.metrics, 'replay_closed_metrics') as metric, \
             patch.object(m.costs, 'replay_closed_costs') as cost, \
             patch.object(m.runtime, 'replay') as runtime, \
             patch.object(m.dope, 'build') as dope:
            with self.assertRaisesRegex(ValueError, 'incomplete'): m.build('a' * 64, 'b' * 64)
            for function in (metric, cost, runtime, dope): function.assert_not_called()

    def test_tables_regenerate_from_complete_opaque_cells(self):
        cells = fixture()
        report = dict(m.common.aggregate(cells), cells=cells, scope='opaque validation')
        table, markdown = m.tables(report)
        self.assertEqual(len(table.splitlines()), 2001)
        self.assertIn('Common retention never selects baseline configurations', markdown)
        self.assertIn('superiority stay null', markdown)
        self.assertEqual(len([line for line in markdown.splitlines() if line.startswith('| ')]), 21)
        report['configuration_panels'][0]['utility']['catboost']['median_of_lineage_sample_medians'] = .99
        with self.assertRaises(ValueError): m.tables(report)

    def test_report_pin_rejected_before_file_read_or_json(self):
        with patch.object(m.Path, 'read_bytes') as read, patch.object(m.json, 'loads') as decode:
            with self.assertRaises(ValueError): m.load_report('opaque.json', 'a' * 64)
            read.assert_not_called()
            decode.assert_not_called()

    def test_report_digest_subclass_rejected_before_file_read(self):
        class Derived(str): pass
        with patch.object(m.Path, 'read_bytes') as read:
            with self.assertRaises(ValueError): m.load_report('opaque.json', Derived(m.REPORT_SHA256))
            read.assert_not_called()

    def test_changed_report_rejected_before_json_decoding(self):
        with patch.object(m.Path, 'is_file', return_value=True), \
             patch.object(m.Path, 'is_symlink', return_value=False), \
             patch.object(m.Path, 'read_bytes', return_value=b'changed opaque report'), \
             patch.object(m.json, 'loads') as decode:
            with self.assertRaises(ValueError): m.load_report('opaque.json', m.REPORT_SHA256)
            decode.assert_not_called()

    def test_publication_parent_link_rejected(self):
        base = Path.cwd() / 'target' / 'density-publication-controls'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as area:
            root = Path(area)
            (root / 'real').mkdir()
            (root / 'real/report.json').write_bytes(b'opaque')
            (root / 'alias').symlink_to(root / 'real', target_is_directory=True)
            with self.assertRaises(ValueError): m.regular_bytes(root / 'alias/report.json')
            self.assertEqual(m.regular_bytes(root / 'real/report.json'), b'opaque')


if __name__ == '__main__': unittest.main(verbosity=2)
