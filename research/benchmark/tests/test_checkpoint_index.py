"""Validate metric completeness, immutable inputs, and reproducible evidence exports."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jsonschema

from research.benchmark import publish_checkpoint_index as p


def fixture():
    rows = []
    for seed, loss in zip(p.SEEDS, (1., 7., 3.)):
        utility = {a: dict(tstr_loss=loss, trtr_loss=2., retention=.5, informative=True)
                   for a in ('catboost', 'linear', 'mlp')}
        rows.append(dict(method='opaque-method', dataset='0' * 16, configuration='default',
            fit_seed=11, size_multiplier=4, sample_seed=seed, status='ok',
            charged_artifact_bytes=7, mfs_v2=None, ptf_v1=None, utility=utility))
    return dict(official_tests_opened=False, cells=rows)


class CheckpointIndex(unittest.TestCase):
    def test_csv_export_preserves_quoted_fields_and_missing_values(self):
        original = b'a,b,c\r\n"has,comma",,"has""quote"\r\n'
        output = p.csv_lf(original)
        self.assertNotIn(b'\r', output)
        self.assertEqual(list(csv.reader(io.StringIO(original.decode()))),
                         list(csv.reader(io.StringIO(output.decode()))))

    def test_measurements_keep_original_sample_schedule_and_native_values_separate(self):
        panel = fixture(); panel['cells'][0]['native_value'] = 1e9
        row = p.grouped(panel)[0]
        self.assertEqual(row['catboost_tstr_mse'], 3.)
        self.assertEqual(row['catboost_trtr_mse'], 2.)
        self.assertEqual(row['catboost_retention'], .5)
        self.assertTrue(row['complete_three_sample_group'])

    def test_missing_duplicate_and_float_sample_identity_are_rejected(self):
        for change in ('missing', 'duplicate', 'float', 'charge', 'score', 'sealed', 'nonfinite'):
            panel = fixture()
            if change == 'missing': panel['cells'].pop()
            if change == 'duplicate': panel['cells'][0]['sample_seed'] = 211
            if change == 'float': panel['cells'][0]['fit_seed'] = 11.0
            if change == 'charge': panel['cells'][0]['charged_artifact_bytes'] = 8
            if change == 'score': panel['cells'][0]['mfs_v2'] = .99
            if change == 'sealed': panel['official_tests_opened'] = True
            if change == 'nonfinite': panel['cells'][0]['utility']['catboost']['tstr_loss'] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError): p.grouped(panel)

    def test_cutoff_cannot_produce_partial_median_or_method_failure(self):
        panel = fixture(); panel['cells'][0]['status'] = 'deadline_unstarted'
        row = p.grouped(panel)[0]
        self.assertFalse(row['complete_three_sample_group'])
        self.assertEqual(row['status_counts'], {'deadline_unstarted': 1, 'ok': 2})
        self.assertIsNone(row['catboost_tstr_mse']); self.assertIsNone(row['catboost_retention'])

    def test_uninformative_group_retains_measured_loss_but_no_retention(self):
        panel = fixture(); panel['cells'][0]['utility']['catboost']['informative'] = False
        row = p.grouped(panel)[0]
        self.assertEqual(row['catboost_tstr_mse'], 3.); self.assertIsNone(row['catboost_retention'])

    def test_checkpoint_error_must_join_to_the_actual_publication_cells(self):
        panel = fixture(); row = panel['cells'][0]
        checkpoint = {k: row[k] for k in ('method', 'dataset', 'configuration', 'fit_seed')}
        checkpoint['key_error_value'] = 3.
        index = dict(checkpoint_records=[checkpoint])
        panels = {name: dict(cells=[]) for name in p.PANELS}; panels[p.PANELS[0]] = panel
        p.verify_checkpoint_errors(index, panels)
        checkpoint['key_error_value'] = .99
        with self.assertRaises(ValueError): p.verify_checkpoint_errors(index, panels)
        checkpoint['key_error_value'] = None; panel['cells'][0]['status'] = 'deadline_unstarted'
        p.verify_checkpoint_errors(index, panels)

    def test_checkpoint_alias_losses_cannot_disagree_between_panels(self):
        panel = fixture(); row = panel['cells'][0]
        checkpoint = {k: row[k] for k in ('method', 'dataset', 'configuration', 'fit_seed')}
        checkpoint['key_error_value'] = 3.
        panels = {name: dict(cells=[]) for name in p.PANELS}
        panels[p.PANELS[0]] = panel; panels[p.PANELS[1]] = deepcopy(panel)
        panels[p.PANELS[1]]['cells'][0]['utility']['catboost']['tstr_loss'] = 100.
        with self.assertRaises(ValueError): p.verify_checkpoint_errors(dict(checkpoint_records=[checkpoint]), panels)

    def test_bound_bytes_reject_rewritten_bytes_and_parent_alias(self):
        base = p.HERE.parents[1] / 'target/tmp'; base.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=base) as directory:
            root = Path(directory); original = root / 'input.json'; original.write_bytes(b'{"opaque":true}')
            sha = hashlib.sha256(original.read_bytes()).hexdigest()
            self.assertEqual(p.bound_bytes(original, sha), b'{"opaque":true}')
            original.write_bytes(b'corrupt metadata')
            with self.assertRaises(ValueError): p.bound_bytes(original, sha)
            link = root / 'alias'; link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError): p.bound_bytes(link / 'input.json', hashlib.sha256(original.read_bytes()).hexdigest())

    def report(self):
        return json.loads((p.RESULTS / p.NAME / 'publication.json').read_bytes())

    def test_committed_figure_kpis_regenerate_from_existing_publications(self):
        report = self.report(); rows = []
        for name in p.PANELS:
            ref = report['source_publications'][name]
            panel = p.guard.decode(p.bound_bytes((p.RESULTS / name).absolute(), ref['sha256']))
            rows.extend(dict(source_panel=name, source_sha256=ref['sha256'], **r) for r in p.grouped(panel))
        actual = (p.RESULTS / p.NAME / 'figure-kpis.csv').read_bytes()
        self.assertEqual(actual, p.csv_text(rows, tuple(rows[0])))
        self.assertEqual(len(rows), 3600)
        self.assertEqual(sum(r['complete_three_sample_group'] for r in rows), 3408)

    def test_committed_checkpoint_aliases_and_missing_errors_remain_explicit(self):
        report = self.report(); path = p.RESULTS / p.NAME / 'checkpoints.csv'
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        self.assertEqual(len(rows), report['checkpoint_records'])
        self.assertEqual(sum(bool(r['key_error_value']) for r in rows), 1273)
        self.assertEqual(len({(r['method'], r['fit_receipt_sha256']) for r in rows}), 1894)
        self.assertTrue(all(r['key_error_missing_reason'] for r in rows if not r['key_error_value']))
        self.assertTrue(all('MSE at 4n' in r['key_error_metric'] for r in rows))
        self.assertTrue(all(json.loads(r['recorded_artifacts']) for r in rows))

    def test_committed_schema_hashes_source_and_claims(self):
        report = self.report(); root = p.RESULTS / p.NAME
        self.assertEqual(report['publisher_sha256'], hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest())
        self.assertEqual(report['inputs_lock_sha256'], p.INPUT_SHA)
        spec = json.loads((root / 'publication.schema.json').read_bytes())
        self.assertEqual(spec, p.schema(report)); jsonschema.validate(report, spec)
        jsonschema.validate(json.loads((root / 'inputs.lock.json').read_bytes()),
                            json.loads((root / 'inputs.lock.schema.json').read_bytes()))
        for name, pin in report['artifact_tables'].items():
            body = (root / name).read_bytes()
            self.assertEqual(hashlib.sha256(body).hexdigest(), pin['sha256']); self.assertEqual(len(body), pin['bytes'])
        for name, expected in p.CLAIMS.items():
            self.assertIs(report[name], expected)
            bad = deepcopy(report); bad[name] = True if expected is None or expected is False else False
            with self.subTest(claim=name), self.assertRaises(jsonschema.ValidationError): jsonschema.validate(bad, spec)


if __name__ == '__main__': unittest.main()
