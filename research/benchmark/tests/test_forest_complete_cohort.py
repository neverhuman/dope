"""Complete-cohort uncertainty, byte accounting and reproducible publication."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import unittest

from research.benchmark import publish_forest_fit_variance as publisher

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'research/benchmark/results/forest-next-six-fivefit-v1'


class CompleteForestCohort(unittest.TestCase):
    def panel(self):
        return json.loads((ROOT / 'panel.json').read_text())

    def test_invalid_lineage_counts_fail_before_receipt_reads(self):
        for count in (True, 0, -1, 101, 6.0, '6'):
            with self.assertRaisesRegex(ValueError, 'invalid_lineage_count'):
                publisher.build(dict(format='forest-complete-cohort-fivefit-input-v1', lineages=count))

    def test_declared_cohort_needs_all_fits_before_receipt_reads(self):
        with self.assertRaisesRegex(ValueError, 'complete_fit_cohort_required'):
            publisher.build(dict(format='forest-complete-cohort-fivefit-input-v1', lineages=6, fits=[{}]*29))

    def test_missing_fit_and_sample_are_not_dropped(self):
        panel = self.panel()
        with self.assertRaisesRegex(ValueError, 'complete_distinct_fits_required'):
            publisher.summarize(panel['fits'][:-1], panel['cells'], 6)
        with self.assertRaisesRegex(ValueError, 'complete_three_samples_required'):
            publisher.summarize(panel['fits'], panel['cells'][:-1], 6)

    def test_split_and_configuration_drift_fail(self):
        for key in ('train_sha256', 'validation_sha256', 'configuration_sha256'):
            panel = self.panel()
            panel['fits'][0][key] = '0'*64
            with self.assertRaisesRegex(ValueError, 'same_split_and_configuration_required'):
                publisher.summarize(panel['fits'], panel['cells'], 6)

    def test_independent_fit_sd_and_exact_tables(self):
        panel = self.panel()
        per_fit, summaries = publisher.summarize(panel['fits'], panel['cells'], 6)
        self.assertEqual(per_fit, panel['per_fit'])
        self.assertEqual(summaries, panel['summary'])
        for summary in summaries:
            selected = [r for r in per_fit if (r['dataset'],r['row_multiplier']) == (summary['dataset'],summary['row_multiplier'])]
            self.assertEqual(len(selected), 5)
            for metric, stat in summary['metrics'].items():
                values = [r['metrics'][metric] for r in selected]
                if all(v is not None for v in values):
                    self.assertEqual(stat['mean'], statistics.mean(values))
                    self.assertEqual(stat['between_fit_sample_sd'], statistics.stdev(values))
                else:
                    self.assertIsNone(stat['mean'])
                    self.assertIsNone(stat['between_fit_sample_sd'])
        self.assertTrue(any(s['native_value']['mean'] < 0 for s in summaries))
        for name, data in publisher.render(panel).items():
            self.assertEqual(data, (ROOT/name).read_bytes())

    def test_schemas_manifest_hashes_and_original_checkpoint_joins(self):
        import jsonschema
        panel = self.panel()
        manifest = json.loads((ROOT/'manifest.json').read_text())
        for name, value in (('panel',panel), ('manifest',manifest)):
            jsonschema.Draft202012Validator(json.loads((ROOT/(name+'.schema.json')).read_text())).validate(value)
        for name, frozen in manifest['files'].items():
            body=(ROOT/name).read_bytes()
            self.assertEqual((len(body),hashlib.sha256(body).hexdigest()),(frozen['bytes'],frozen['sha256']),name)
        for frozen in (*manifest['producer_refs'].values(),manifest['common_flatten_source_ref']):
            body=(REPO/frozen['path']).read_bytes()
            self.assertEqual((len(body),hashlib.sha256(body).hexdigest()),(frozen['bytes'],frozen['sha256']))
        proof=json.loads((ROOT/'original-verification.json').read_text())
        schema=json.loads((ROOT/'panel.schema.json').read_text())
        proof_schema=dict(schema, **{'$ref':'#/$defs/original_verification'})
        for key in ('required','properties','additionalProperties'): proof_schema.pop(key)
        jsonschema.Draft202012Validator(proof_schema).validate(proof)
        originals={r['path']:r for r in proof['refs']}
        self.assertEqual(len(panel['fits']),30)
        self.assertEqual(len(panel['cells']),180)
        self.assertEqual(len({c['job_sha256'] for c in panel['cells']}),180)
        for fit in panel['fits']:
            self.assertEqual(sum(r['bytes'] for r in fit['artifact_inventory']),fit['artifact_bytes'])
            for artifact in fit['artifact_inventory']:
                original=originals[str(Path(fit['fit_ref']['path']).parent/'artifact'/artifact['path'])]
                self.assertEqual((original['sha256'],original['bytes']),(artifact['sha256'],artifact['bytes']))
            scores=fit['native_auditor_seed_scores']
            self.assertEqual(set(scores),{'adaboost','linear','random_forest','xgboost'})
            self.assertEqual({len(v) for v in scores.values()},{5})
            self.assertAlmostEqual(statistics.mean(v for group in scores.values() for v in group),fit['native_value'])
        for cell in panel['cells']:
            fit=next(f for f in panel['fits'] if (f['dataset'],f['fit_seed'])==(cell['dataset'],cell['fit_seed']))
            identity=dict(cell['job_identity_projection'])
            identity['worker_key']=identity.pop('worker_partition_sha256')
            self.assertEqual(publisher.digest(identity),cell['job_sha256'])
            self.assertEqual(identity['fit_receipt_sha256'],fit['fit_ref']['sha256'])
            self.assertEqual(identity['source_fit_job_sha256'],fit['job_sha256'])
            self.assertEqual(cell['config_sha256'],fit['configuration_sha256'])
            self.assertEqual(cell['charged_artifact_bytes'],fit['artifact_bytes'])
            for key, fkey in (('train_ref','train_sha256'),('validation_ref','validation_sha256')):
                self.assertEqual(cell['input_refs'][key]['sha256'],fit[fkey])
        for key,fkey in (('sum_generator_fit_seconds','fit_seconds'),('sum_whole_fit_seconds','whole_fit_seconds'),('sum_native_audit_seconds','native_seconds')):
            self.assertEqual(manifest['costs'][key],sum(f[fkey] for f in panel['fits']))
        self.assertEqual(manifest['costs']['sum_common_metric_seconds'],sum(c['metric_seconds'] for c in panel['cells']))
        self.assertFalse(manifest['costs']['clocks_are_additive'])
        for value in (panel,manifest,*panel['cells']):
            self.assertFalse(value['official_tests_opened'])
            for gate in ('mfs_v2','ptf_v1','release_safe_l3','superiority'):
                self.assertIsNone(value[gate])
        self.assertFalse(panel['full_population_complete'])
        self.assertFalse(panel['native_selected_population_complete'])
        self.assertFalse(panel['counts_as_dope_win'])
        rows=list(csv.DictReader((ROOT/'checkpoint-kpis.csv').read_text().splitlines()))
        self.assertEqual(len(rows),30)
        self.assertEqual({int(r['fit_seed']) for r in rows},{11,23,37,53,71})

    def test_every_original_timeout_has_explicit_unscored_disposition(self):
        import jsonschema
        data=json.loads((ROOT/'timeout-dispositions.json').read_text())
        jsonschema.Draft202012Validator(json.loads((ROOT/'timeout-dispositions.schema.json').read_text())).validate(data)
        self.assertEqual(data['status_counts'],{'ok':120,'timeout':8})
        rows=data['timeout_dispositions']
        self.assertEqual({(r['dataset'],r['fit_seed']) for r in rows},
                         {(d,s) for d in ('297b7688a045c583','48b7a3bf81e5f3e5') for s in (23,37,53,71)})
        self.assertEqual(len({r['job_sha256'] for r in rows}),8)
        for row in rows:
            self.assertEqual(row['worker_exit_code'],124)
            self.assertEqual(row['fit_timeout_seconds'],600)
            self.assertIsNone(row['measured_elapsed_seconds'])
            self.assertFalse(row['accepted_artifact'])
            self.assertFalse(row['retry_started'])
            self.assertIsNone(row['native_validation_kpi'])
            self.assertIsNone(row['common_metrics'])
            self.assertIsNone(row['method_quality_conclusion'])

    def test_new_pdf_embeds_truetype_and_has_no_type3_fonts(self):
        body = (ROOT/'fit-variance.pdf').read_bytes()
        self.assertNotIn(b'/Subtype /Type3', body)
        self.assertIn(b'/CIDFontType2', body)
        self.assertIn(b'/FontFile2', body)


if __name__ == '__main__':
    unittest.main()
