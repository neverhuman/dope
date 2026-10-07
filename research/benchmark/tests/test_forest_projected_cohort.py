"""Verify homogeneous projection, complete fit replicates and frozen publication."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import unittest
from unittest.mock import patch

from research.benchmark import publish_forest_fit_variance as publisher

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'research/benchmark/results/forest-third-five-fivefit-v1'


class ProjectedForestCohort(unittest.TestCase):
    def panel(self):
        return json.loads((ROOT / 'panel.json').read_text())

    def test_invalid_projection_rejected_before_receipt_reads(self):
        class Untrusted(str):
            def __hash__(self):
                raise AssertionError('untrusted hash executed')
        for datasets, selected in ((['a'] * 5, False), (['a', 'b', 'c', 'd', 1], False),
                                   (['a', 'b', 'c', 'd', 'e'], True),
                                   ([Untrusted('a'), 'b', 'c', 'd', 'e'], False)):
            inputs = dict(format='forest-complete-cohort-fivefit-input-v1', lineages=5,
                          cohort_projection=dict(datasets=datasets, shared_metric_selection_used=selected))
            with patch.object(publisher, 'verified', side_effect=AssertionError('receipt read')):
                with self.assertRaisesRegex(ValueError, 'invalid_cohort_projection'):
                    publisher.build(inputs)

    def test_excluded_bad_metric_is_rejected_before_projection(self):
        # Entire original metric batches must be verified even for a lineage
        # outside the reporting projection. No synthetic learner is invoked.
        panel = self.panel()
        inputs = copy.deepcopy(panel['inputs'])
        proof = json.loads((ROOT / 'original-verification.json').read_text())
        cell = copy.deepcopy(panel['cells'][0])
        identity = cell.pop('job_identity_projection')
        identity['worker_key'] = identity.pop('worker_partition_sha256')
        cell['job_identity'] = identity
        cell['dataset'] = '0000000000000000'
        cell['status'] = 'unavailable'
        batch = inputs['metric_batches'][0]
        receipt = dict(path='/excluded-cell.json', bytes=1, sha256='0' * 64)
        manifest = dict(jobs=[dict(job_sha256=cell['job_sha256'], job_identity=identity)])
        values = {
            inputs['original_verification']['path']: proof,
            inputs['original_verification_inputs']['path']: dict(fits=inputs['fits']),
            batch['input_manifest']['path']: manifest,
            batch['receipt_lock']['path']: dict(actual_complete=True, count=1, receipts=[receipt],
                                              official_tests_opened=False, manifest_sha256=publisher.digest(manifest)),
            batch['process_exit']['path']: dict(exit_code=0, manifest_sha256=batch['input_manifest']['sha256']),
            receipt['path']: cell,
        }
        with patch.object(publisher, 'verified', side_effect=lambda ref: values[ref['path']]):
            with self.assertRaisesRegex(ValueError, 'metric_identity_changed'):
                publisher.build(inputs)

    def test_five_fits_three_samples_and_nested_sd(self):
        panel = self.panel()
        self.assertEqual((panel['lineages'], panel['generator_fits'], panel['common_sample_cells']), (5, 25, 150))
        datasets = set(panel['inputs']['cohort_projection']['datasets'])
        self.assertEqual({f['dataset'] for f in panel['fits']}, datasets)
        self.assertEqual({c['dataset'] for c in panel['cells']}, datasets)
        self.assertFalse(panel['inputs']['cohort_projection']['shared_metric_selection_used'])
        per_fit, summary = publisher.summarize(panel['fits'], panel['cells'], 5)
        self.assertEqual((per_fit, summary), (panel['per_fit'], panel['summary']))
        for row in summary:
            group = [p for p in per_fit if (p['dataset'], p['row_multiplier']) == (row['dataset'], row['row_multiplier'])]
            self.assertEqual({p['fit_seed'] for p in group}, {11, 23, 37, 53, 71})
            for metric, value in row['metrics'].items():
                values = [p['metrics'][metric] for p in group]
                if all(v is not None for v in values):
                    self.assertEqual(value['mean'], statistics.mean(values))
                    self.assertEqual(value['between_fit_sample_sd'], statistics.stdev(values))
                else:
                    self.assertIsNone(value['mean'])
                    self.assertIsNone(value['between_fit_sample_sd'])
        for name, raw in publisher.render(panel).items():
            self.assertEqual(raw, (ROOT / name).read_bytes())

    def test_checkpoint_identity_split_bytes_and_native_formula(self):
        panel = self.panel()
        proof = json.loads((ROOT / 'original-verification.json').read_text())
        original = {r['path']: r for r in proof['refs']}
        self.assertEqual(proof['fit_checkpoints'], 25)
        self.assertFalse(proof['pickle_loaded'])
        for fit in panel['fits']:
            self.assertEqual(sum(r['bytes'] for r in fit['artifact_inventory']), fit['artifact_bytes'])
            self.assertGreater(fit['artifact_bytes'], 10240)
            for r in fit['artifact_inventory']:
                prior = original[str(Path(fit['fit_ref']['path']).parent / 'artifact' / r['path'])]
                self.assertEqual((r['bytes'], r['sha256']), (prior['bytes'], prior['sha256']))
            native = fit['native_auditor_seed_scores']
            self.assertEqual(set(native), {'adaboost', 'linear', 'random_forest', 'xgboost'})
            self.assertEqual({len(v) for v in native.values()}, {5})
            self.assertAlmostEqual(statistics.mean(v for scores in native.values() for v in scores), fit['native_value'])
        for cell in panel['cells']:
            identity = dict(cell['job_identity_projection'])
            identity['worker_key'] = identity.pop('worker_partition_sha256')
            self.assertEqual(publisher.digest(identity), cell['job_sha256'])
            fit = next(f for f in panel['fits'] if (f['dataset'], f['fit_seed']) == (cell['dataset'], cell['fit_seed']))
            self.assertEqual(identity['fit_receipt_sha256'], fit['fit_ref']['sha256'])
            self.assertEqual(cell['config_sha256'], fit['configuration_sha256'])
            self.assertEqual(cell['charged_artifact_bytes'], fit['artifact_bytes'])
            self.assertEqual(cell['input_refs']['train_ref']['sha256'], fit['train_sha256'])
            self.assertEqual(cell['input_refs']['validation_ref']['sha256'], fit['validation_sha256'])
        rows = list(csv.DictReader((ROOT / 'checkpoint-kpis.csv').read_text().splitlines()))
        self.assertEqual(len(rows), 25)

    def test_schemas_manifest_hashes_costs_and_null_gates(self):
        import jsonschema
        panel = self.panel()
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        for name, value in (('panel', panel), ('manifest', manifest)):
            jsonschema.Draft202012Validator(json.loads((ROOT / (name + '.schema.json')).read_text())).validate(value)
        for name, ref in manifest['files'].items():
            raw = (ROOT / name).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (ref['bytes'], ref['sha256']), name)
        for ref in (*manifest['producer_refs'].values(), manifest['common_flatten_source_ref']):
            raw = (REPO / ref['path']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (ref['bytes'], ref['sha256']))
        for key, field in (('sum_generator_fit_seconds', 'fit_seconds'), ('sum_whole_fit_seconds', 'whole_fit_seconds'), ('sum_native_audit_seconds', 'native_seconds')):
            self.assertEqual(manifest['costs'][key], sum(f[field] for f in panel['fits']))
        self.assertEqual(manifest['costs']['sum_common_metric_seconds'], sum(c['metric_seconds'] for c in panel['cells']))
        self.assertFalse(manifest['costs']['clocks_are_additive'])
        for value in (panel, manifest, *panel['cells']):
            self.assertFalse(value['official_tests_opened'])
            for key in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority'):
                self.assertIsNone(value[key])
        self.assertFalse(panel['full_population_complete'])
        self.assertFalse(panel['native_selected_population_complete'])
        self.assertFalse(panel['counts_as_dope_win'])

    def test_pdf_has_embedded_truetype_fonts(self):
        raw = (ROOT / 'fit-variance.pdf').read_bytes()
        self.assertNotIn(b'/Subtype /Type3', raw)
        self.assertIn(b'/CIDFontType2', raw)
        self.assertIn(b'/FontFile2', raw)


if __name__ == '__main__':
    unittest.main()
