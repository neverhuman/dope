"""Focused source projection controls, without scientific execution."""
import csv
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import publish_validation_figure as pub
SOURCE = Path(__file__).resolve().parents[1] / 'results/checkpoint-figure-input-index/figure-kpis.csv'


class ValidationFigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = SOURCE.read_bytes()
        cls.rows = list(csv.DictReader(io.StringIO(cls.body.decode())))

    def changed(self, mutate):
        rows = [r.copy() for r in self.rows]
        selected = next(r for r in rows if r['method'] == 'ARF' and r['configuration'] == 'native_selected')
        mutate(rows, selected)
        out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
        return out.getvalue().encode()

    def test_actual_aggregate_and_schema(self):
        from jsonschema import Draft202012Validator
        report, table = pub.build(SOURCE)
        self.assertEqual(len(report['rows']), 30)
        self.assertEqual(hashlib.sha256(self.body).hexdigest(), pub.SOURCE_SHA)
        Draft202012Validator(pub.schema(report)).validate(report)
        rows = {r['method']: r for r in report['rows'] if r['auditor'] == 'catboost' and r['size_multiplier'] == 4}
        self.assertEqual(rows['DOPE']['informative_matched_lineages'], 97)
        self.assertAlmostEqual(rows['DOPE']['median_null_normalized_tstr_trtr_retention'], .939876353161328)
        self.assertAlmostEqual(rows['ARF']['median_null_normalized_tstr_trtr_retention'], .8255067785547131)
        self.assertEqual(report['source_group_coverage']['DOPE']['4']['complete_three_sample_groups'], 98)
        self.assertEqual(len(report['source_group_coverage']['DOPE']['4']['incomplete_datasets']), 2)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(table.decode())))), 30)
        for name in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.assertIsNone(report[name])
        for name in ('scientific_execution', 'official_tests_opened', 'production_certified', 'paired_significance_measured',
                     'full_five_fit_seed_coverage', 'native_selection_changed', 'complete_neural_comparison'):
            self.assertIs(report[name], False)

    def test_duplicate_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            pub.summarize(self.changed(lambda rows, selected: rows.append(selected.copy())))

    def test_incomplete_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, 'three-seed'):
            pub.summarize(self.changed(lambda rows, selected: selected.update(status_counts='{"ok":2,"infra":1}')))

    def test_typed_schedule_rejected(self):
        for patch in ({'fit_seed': '11.0'}, {'size_multiplier': '4.0'}, {'sample_seeds': '101,101,307'}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                pub.summarize(self.changed(lambda rows, selected: selected.update(patch)))

    def test_controls_must_be_paired(self):
        with self.assertRaisesRegex(ValueError, 'real-vs-real'):
            pub.summarize(self.changed(lambda rows, selected: selected.update(catboost_trtr_mse='123.0')))

    def test_missing_lineage_rejected(self):
        with self.assertRaisesRegex(ValueError, '100-lineage'):
            pub.summarize(self.changed(lambda rows, selected: rows.remove(selected)))

    def test_nonfinite_missing_negative_loss_rejected(self):
        for value in ('nan', 'inf', '', '-0.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pub.summarize(self.changed(lambda rows, selected: selected.update(catboost_tstr_mse=value)))

    def test_retention_is_unclipped_and_nulls_not_pooled(self):
        report, _ = pub.build(SOURCE)
        mlp = next(r for r in report['rows'] if r['method'] == 'DOPE' and r['auditor'] == 'mlp' and r['size_multiplier'] == 4)
        self.assertGreater(mlp['median_null_normalized_tstr_trtr_retention'], 1)
        self.assertEqual(mlp['informative_matched_lineages'], 75)

    def test_partial_median_rejected(self):
        def mutate(rows, ignored):
            missing = next(r for r in rows if r['method'] == 'DOPE' and r['configuration'] == 'features12_steps2048'
                           and r['source_panel'] == 'density-matched-population-validation.json'
                           and r['complete_three_sample_group'] == 'False')
            missing['catboost_tstr_mse'] = '0.001'
        with self.assertRaisesRegex(ValueError, 'partial median'):
            pub.summarize(self.changed(mutate))

    def test_failed_digest_never_reaches_projection(self):
        with patch.object(pub, 'bound_bytes', side_effect=ValueError('frozen publication input changed')), \
             patch.object(pub, 'summarize') as projection:
            with self.assertRaisesRegex(ValueError, 'frozen publication input changed'):
                pub.build(SOURCE)
            projection.assert_not_called()

    def test_committed_outputs_reproduce(self):
        root = SOURCE.parents[1] / 'matched-validation-descriptive-figure'
        report, table = pub.build(SOURCE)
        self.assertEqual((root / 'publication.json').read_bytes(), pub.encoded(report))
        self.assertEqual((root / 'figure-kpis.csv').read_bytes(), table)
        self.assertEqual((root / 'publication.schema.json').read_bytes(), pub.encoded(pub.schema(report)))
        receipt = json.loads((root / 'figures.receipt.json').read_bytes())
        self.assertEqual((root / 'figures.receipt.schema.json').read_bytes(), pub.encoded(pub.schema(receipt)))
        from jsonschema import Draft202012Validator
        Draft202012Validator(pub.schema(receipt)).validate(receipt)
        self.assertEqual(receipt['source_csv_sha256'], pub.SOURCE_SHA)
        self.assertEqual(receipt['publisher_sha256'], report['publisher_sha256'])
        for name, ref in receipt['outputs'].items():
            body = (root / name).read_bytes()
            self.assertEqual(len(body), ref['bytes'])
            self.assertEqual(hashlib.sha256(body).hexdigest(), ref['sha256'])


if __name__ == '__main__':
    unittest.main()
