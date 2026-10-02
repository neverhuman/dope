"""Complete cells and method-native selection are required for publication."""

import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from research.benchmark.publish_forestdiffusion_native import winner, summary, csv_bytes, markdown
from research.benchmark.score import sha256
from research.benchmark.manifest import digest


class ForestPublicationTests(unittest.TestCase):
    def attempts(self):
        scores = {name: [.5] * 5 for name in ('linear', 'adaboost', 'random_forest', 'xgboost')}
        kpi = {'name': 'author_mean_ml_r2', 'direction': 'maximize', 'value': .5,
               'shared_kpi_used_for_selection': False, 'auditor_fit_seeds': list(range(5)),
               'auditor_seed_scores': scores}
        return [{'job': {'trial': trial, 'config': {'size': trial + 1}, 'input_files': {'train': 'same'},
                         'objective': 'author_mean_ml_r2'},
                 'status': 'ok', 'native_kpi': copy.deepcopy(kpi), 'artifact_bytes': 100 + trial,
                 'elapsed_seconds': 10, 'common_retention': 1 - trial / 10} for trial in range(8)]

    def test_winner_uses_own_native_kpi_then_bytes_not_common_retention(self):
        rows = self.attempts()
        rows[7]['native_kpi']['value'] = .6
        rows[7]['native_kpi']['auditor_seed_scores'] = {name: [.6] * 5 for name in rows[7]['native_kpi']['auditor_seed_scores']}
        self.assertEqual(winner(rows)['job']['trial'], 7)
        rows[6]['native_kpi'] = copy.deepcopy(rows[7]['native_kpi'])
        self.assertEqual(winner(rows)['job']['trial'], 6)
        rows[6]['status'] = 'native_timeout'
        self.assertEqual(winner(rows)['job']['trial'], 7)
        with self.assertRaises(ValueError):
            winner(rows[:-1])
        rows[7]['elapsed_seconds'] = 43201
        with self.assertRaises(ValueError):
            winner(rows)

    def test_missing_or_duplicate_logical_cells_prevent_summary(self):
        cells = [{'dataset': d, 'configuration': c, 'sample_seed': s, 'size_multiplier': z,
                  'status': 'fit_unavailable', 'metrics': None, 'artifact_bytes': None, 'within_l3_bytes': None}
                 for d in ('Adult', 'California', 'News')
                 for c in ('cpu_author_default', 'gpu_author_default', 'native_selected')
                 for s in (101, 211, 307) for z in (1, 2, 4, 8)]
        self.assertTrue(all(r['median_retention'] is None for r in summary(cells)))
        with self.assertRaises(ValueError):
            summary(cells[:-1])
        with self.assertRaises(ValueError):
            summary(cells + [cells[0]])

    def test_committed_complete_panel_and_regenerated_tables(self):
        path = Path(__file__).parents[1] / 'results/pilot24-forestdiffusion-native.json'
        document = json.loads(path.read_text())
        validator = Draft202012Validator(json.loads(path.with_suffix('.schema.json').read_text()))
        validator.validate(document)
        self.assertEqual(len(document['fit_attempts']), 24)
        self.assertEqual(len(document['cells']), 108)
        self.assertEqual(len(document['physical_sample_receipts']), 12)
        self.assertEqual(len(document['matched_references']['cells']), 120)
        for row in document['physical_sample_receipts']:
            job = row['job']
            self.assertNotIn('physical_job_key', job)
            self.assertEqual(job['physical_job_sha256'], digest([
                job['fit_attempt']['sha256'], job['sample_seed'], job['size_multiplier']]))
            self.assertEqual(Path(row['path']).parent.name, job['physical_job_sha256'])
        self.assertEqual(csv_bytes(document), path.with_suffix('.csv').read_bytes())
        self.assertEqual(markdown(document), path.with_suffix('.md').read_bytes())
        self.assertEqual(summary(document['cells']), document['summaries'])
        for key, bad in (('mfs_v2', .99), ('ptf_v1', .99), ('release_safe', .99), ('official_tests_opened', True)):
            changed = {**document, key: bad}
            self.assertFalse(validator.is_valid(changed), key)
        changed = copy.deepcopy(document)
        changed['cells'].pop()
        self.assertFalse(validator.is_valid(changed))

    def test_publication_manifest_binds_all_outputs_and_sources(self):
        root = Path(__file__).parents[1]
        path = root / 'results/pilot24-forestdiffusion-native.manifest.json'
        document = json.loads(path.read_text())
        Draft202012Validator(json.loads(path.with_suffix('.schema.json').read_text())).validate(document)
        self.assertEqual(len(document['artifacts']), 6)
        for name, item in document['artifacts'].items():
            self.assertEqual(item['sha256'], sha256(path.parent / name))
            self.assertEqual(item['bytes'], (path.parent / name).stat().st_size)
        for name, expected in document['publisher_sources'].items():
            self.assertEqual(expected, sha256(root / name))
        self.assertFalse(document['campaign_complete'])
        self.assertFalse(document['production_certified'])


if __name__ == '__main__':
    unittest.main()
