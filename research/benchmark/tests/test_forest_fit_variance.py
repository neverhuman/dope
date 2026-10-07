"""Receipt joins and independent-fit uncertainty for the complete cohort."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = HERE.parent / 'publish_forest_fit_variance.py'
ROOT = HERE.parent / 'results/forest-first-two-fivefit-v1'
spec = importlib.util.spec_from_file_location('forest_fit_variance_under_test', SOURCE)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class FitVarianceControls(unittest.TestCase):
    def test_unbiased_fit_sd(self):
        result = publisher.fit_statistic([1., 2., 3., 4., 5.])
        self.assertEqual(result['mean'], 3.)
        self.assertAlmostEqual(result['between_fit_sample_sd'], math.sqrt(2.5))
        self.assertEqual(result['measured_fits'], 5)

    def test_missing_fit_is_not_dropped(self):
        result = publisher.fit_statistic([1., 2., None, 4., 5.])
        self.assertIsNone(result['mean'])
        self.assertIsNone(result['between_fit_sample_sd'])
        self.assertEqual(result['measured_fits'], 4)
        self.assertEqual(result['unavailable_fits'], 1)

    def test_uninformative_retention_is_null(self):
        result = publisher.fit_statistic([None] * 5)
        self.assertIsNone(result['mean'])
        self.assertIsNone(result['between_fit_sample_sd'])
        self.assertEqual(result['unavailable_fits'], 5)

    def test_five_fit_replicates_required(self):
        with self.assertRaisesRegex(ValueError, 'five_independent_fits_required'):
            publisher.fit_statistic([1., 2., 3., 4.])

    def test_nonfinite_and_boolean_values_rejected(self):
        for value in (float('nan'), float('inf'), True):
            with self.assertRaisesRegex(ValueError, 'invalid_metric_value'):
                publisher.fit_statistic([1., 2., value, 4., 5.])

    def test_negative_native_r2_is_preserved(self):
        result = publisher.fit_statistic([-1., -2., -3., -4., -5.])
        self.assertEqual(result['mean'], -3.)
        self.assertAlmostEqual(result['between_fit_sample_sd'], math.sqrt(2.5))

    def test_changed_metadata_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'target') as tmp:
            path = Path(tmp) / 'receipt.json'
            path.write_text('{"value": 1}')
            ref = dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_text('{"value": 2}')
            with self.assertRaisesRegex(ValueError, 'frozen_metadata_changed'):
                publisher.verified(ref)

    def test_symlink_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'target') as tmp:
            path = Path(tmp) / 'receipt.json'; path.write_text('{}')
            link = Path(tmp) / 'link.json'; link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'invalid_metadata_path'):
                publisher.verified(dict(path=str(link), bytes=2, sha256=hashlib.sha256(b'{}').hexdigest()))

    def test_complete_sample_and_fit_schedule_required(self):
        panel = json.loads((ROOT / 'panel.json').read_text())
        with self.assertRaisesRegex(ValueError, 'complete_three_samples_required'):
            publisher.summarize(panel['fits'], panel['cells'][:-1])
        with self.assertRaisesRegex(ValueError, 'ten_distinct_fits_required'):
            publisher.summarize(panel['fits'][:-1], panel['cells'])

    def test_sample_variation_is_not_counted_as_fit_variation(self):
        panel = json.loads((ROOT / 'panel.json').read_text())
        per_fit, summary = publisher.summarize(panel['fits'], panel['cells'])
        for row in summary:
            group = [r['metrics']['catboost_retention'] for r in per_fit
                     if (r['dataset'], r['row_multiplier']) == (row['dataset'], row['row_multiplier'])]
            self.assertEqual(len(group), 5)
            if all(v is not None for v in group):
                self.assertAlmostEqual(row['metrics']['catboost_retention']['between_fit_sample_sd'], statistics.stdev(group))
            else:
                self.assertIsNone(row['metrics']['catboost_retention']['mean'])
                self.assertIsNone(row['metrics']['catboost_retention']['between_fit_sample_sd'])

    def test_committed_schema_hashes_receipt_joins_and_exact_regeneration(self):
        import jsonschema
        panel = json.loads((ROOT / 'panel.json').read_text())
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        jsonschema.Draft202012Validator(json.loads((ROOT / 'panel.schema.json').read_text())).validate(panel)
        jsonschema.Draft202012Validator(json.loads((ROOT / 'manifest.schema.json').read_text())).validate(manifest)
        for name, ref in manifest['files'].items():
            body = (ROOT / name).read_bytes()
            self.assertEqual(len(body), ref['bytes'], name)
            self.assertEqual(hashlib.sha256(body).hexdigest(), ref['sha256'], name)
        for name, ref in manifest['producer_refs'].items():
            path = SOURCE if name == SOURCE.name else SOURCE.with_name(name)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), ref['sha256'])
        for gate in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority'):
            self.assertIsNone(panel[gate]); self.assertIsNone(manifest[gate])
        self.assertFalse(panel['official_tests_opened'])
        self.assertFalse(panel['full_population_complete'])
        self.assertFalse(panel['native_selected_population_complete'])
        self.assertFalse(panel['counts_as_dope_win'])
        self.assertEqual(len(panel['fits']), 10); self.assertEqual(len(panel['cells']), 60)
        self.assertEqual(len({c['job_sha256'] for c in panel['cells']}), 60)
        for cell in panel['cells']:
            fit = next(f for f in panel['fits'] if (f['dataset'], f['fit_seed']) == (cell['dataset'], cell['fit_seed']))
            identity = dict(cell['job_identity_projection'])
            identity['worker_key'] = identity.pop('worker_partition_sha256')
            self.assertEqual(cell['job_sha256'], publisher.digest(identity))
            self.assertEqual(cell['job_identity_projection']['fit_receipt_sha256'], fit['fit_ref']['sha256'])
            self.assertEqual(cell['job_identity_projection']['source_fit_job_sha256'], fit['job_sha256'])
            self.assertEqual(cell['charged_artifact_bytes'], fit['artifact_bytes'])
            self.assertEqual(cell['config_sha256'], fit['configuration_sha256'])
            self.assertEqual(cell['input_refs']['train_ref']['sha256'], fit['train_sha256'])
            self.assertEqual(cell['input_refs']['validation_ref']['sha256'], fit['validation_sha256'])
            self.assertFalse(cell['official_tests_opened'])
            for gate in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority'):
                self.assertIsNone(cell[gate])
        proof = json.loads((ROOT / 'original-verification.json').read_text())
        originals = {r['path']: r for r in proof['refs']}
        self.assertTrue(proof['actual_complete']); self.assertEqual(proof['fit_checkpoints'], 10)
        self.assertFalse(proof['pickle_loaded']); self.assertFalse(proof['official_tests_opened'])
        self.assertEqual(proof['input_sha256'], panel['inputs']['original_verification_inputs']['sha256'])
        for fit in panel['fits']:
            self.assertEqual(sum(r['bytes'] for r in fit['artifact_inventory']), fit['artifact_bytes'])
            for artifact in fit['artifact_inventory']:
                original = originals[str(Path(fit['fit_ref']['path']).parent / 'artifact' / artifact['path'])]
                self.assertEqual((original['sha256'], original['bytes']), (artifact['sha256'], artifact['bytes']))
            scores = fit['native_auditor_seed_scores']
            self.assertEqual(set(scores), {'adaboost', 'linear', 'random_forest', 'xgboost'})
            self.assertEqual({len(v) for v in scores.values()}, {5})
            self.assertAlmostEqual(statistics.mean(v for values in scores.values() for v in values), fit['native_value'])
        per_fit, summary = publisher.summarize(panel['fits'], panel['cells'])
        self.assertEqual(per_fit, panel['per_fit']); self.assertEqual(summary, panel['summary'])
        for name, body in publisher.render(panel).items():
            self.assertEqual(body, (ROOT / name).read_bytes(), name)
        for field, key in (('sum_generator_fit_seconds', 'fit_seconds'), ('sum_whole_fit_seconds', 'whole_fit_seconds'), ('sum_native_audit_seconds', 'native_seconds')):
            self.assertEqual(manifest['costs'][field], sum(f[key] for f in panel['fits']))
        self.assertEqual(manifest['costs']['sum_common_metric_seconds'], sum(c['metric_seconds'] for c in panel['cells']))
        self.assertFalse(manifest['costs']['clocks_are_additive'])
        checkpoint_rows = list(csv.DictReader((ROOT / 'checkpoint-kpis.csv').read_text().splitlines()))
        self.assertEqual(len(checkpoint_rows), 10)
        self.assertEqual({int(r['fit_seed']) for r in checkpoint_rows}, {11, 23, 37, 53, 71})


if __name__ == '__main__':
    unittest.main()
