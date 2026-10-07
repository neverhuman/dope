"""Native selection snapshots, immutable evidence, complete groups and scalar replay."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import unittest

from research.benchmark import publish_tabddpm_retained as publisher
from research.benchmark.publish_retained_comparison import summarize

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'research/benchmark/results/tabddpm-twelve-lineage-retained-validation'


class TabDDPMRetained(unittest.TestCase):
    def panel(self):
        return json.loads((ROOT / 'panel.json').read_text())

    def test_native_snapshot_rule_ignores_quality_and_input_order(self):
        models = self.panel()['checkpoints']
        expected = self.panel()['native_reporting_frontier']
        self.assertEqual(publisher.native_versions(models), expected)
        self.assertEqual(publisher.native_versions(list(reversed(models))), expected)
        for m in models:
            m['catboost_retention'] = 999 if m['job_sha256'] not in expected.values() else -999
            m['native_value'] = -12345
        self.assertEqual(publisher.native_versions(models), expected)

    def test_conflicting_same_time_native_snapshot_rejected(self):
        model = next(m for m in self.panel()['checkpoints'] if 'native_selected' in m['logical_roles'])
        other = dict(model, native_selection_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'conflicting_native_snapshot'):
            publisher.native_versions([model, other])

    def test_non_builtin_digest_rejected_before_file_access(self):
        class Digest(str):
            def __eq__(self, other):
                return True
        with self.assertRaisesRegex(ValueError, 'invalid_frozen_digest'):
            publisher.read(dict(path='/does/not/exist', bytes=1, sha256=Digest('0' * 64)))

    def test_native_members_are_bound_to_original_selection(self):
        ref = dict(path='/frozen/native.json', sha256='a' * 64)
        selection = dict(trial_native_receipts=[ref])
        self.assertEqual(publisher.native_member_refs(selection, lambda _: self.fail('unexpected read')),
                         {('/frozen/native.json', 'a' * 64)})
        job = dict(method='TabDDPM', fit_seed=11)
        trial = dict(job=job, job_sha256=publisher.digest(job), official_tests_opened=False,
                     evidence_files={'native.json': 'b' * 64})
        selection = dict(all_five_cells_closed=True, trial_receipts=[dict(path='/frozen/receipt.json', sha256='c' * 64)])
        self.assertEqual(publisher.native_member_refs(selection, lambda _: trial),
                         {('/frozen/native.json', 'b' * 64)})
        trial['job']['fit_seed'] = 11.0
        with self.assertRaisesRegex(ValueError, 'historical_native_trial_changed'):
            publisher.native_member_refs(selection, lambda _: trial)

    def test_metadata_bytes_cannot_be_rewritten(self):
        p = ROOT / 'inputs.lock.json'
        ref = dict(path=str(p), bytes=p.stat().st_size, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'frozen_metadata_changed'):
            publisher.verified_bytes(ref)

    def join_fixture(self):
        panel = self.panel()
        cell = panel['physical_panel']['cells'][0]
        record = next(x for x in json.loads((ROOT / 'original-verification.json').read_text())['records']
                      if x['dataset'] == cell['dataset'] and publisher.digest(x['job']['config']) == cell['config_sha256'])
        original = dict(job_identity=dict(source_fit_job_sha256=record['job_sha256'],
                                         matched_receipt_sha256=record['matched_closure']['sha256']))
        return cell, record, original

    def test_float_and_boolean_seed_aliases_rejected(self):
        cell, record, original = self.join_fixture()
        publisher.join_cell(cell, record, original)
        for key, value in [('fit_seed', 11.0), ('sample_seed', float(cell['sample_seed'])), ('row_multiplier', True)]:
            changed = copy.deepcopy(cell); changed[key] = value
            with self.assertRaisesRegex(ValueError, 'identity'):
                publisher.join_cell(changed, record, original)

    def test_original_split_sample_and_operation_drift_rejected(self):
        cell, record, original = self.join_fixture()
        for key in ('train_ref', 'validation_ref', 'projection_ref', 'synthetic_ref'):
            changed = copy.deepcopy(cell); changed['input_hashes'][key] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'changed'):
                publisher.join_cell(changed, record, original)
        original['job_identity']['matched_receipt_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'metric_original_operation_changed'):
            publisher.join_cell(cell, record, original)

    def test_complete_groups_and_exact_exports(self):
        panel = self.panel(); summaries = []
        for role in ('author_default', 'native_selected'):
            group = [{k: v for k, v in r.items() if k not in ('original_fit_job_sha256', 'metric_receipt')}
                     for r in panel['rows'] if r['selection_binding'] == role]
            datasets = sorted({r['dataset'] for r in group})
            summaries += summarize(group, datasets, 'complete_available_' + role)
            with self.assertRaisesRegex(ValueError, 'incomplete_matched_three_seed_group'):
                summarize(group[:-1], datasets, 'incomplete')
        self.assertEqual(summaries, panel['summary'])
        for name, body in publisher.render(panel).items():
            self.assertEqual(body, (ROOT / name).read_bytes(), name)
        csv_rows = list(csv.DictReader((ROOT / 'checkpoint-kpis.csv').read_text().splitlines()))
        self.assertEqual(len(csv_rows), 21)
        self.assertTrue(all(r['catboost_tstr_mse_n'] and r['catboost_tstr_mse_4n'] for r in csv_rows))

    def test_counts_costs_native_pool_limits_and_null_claims(self):
        panel = self.panel(); cells = panel['physical_panel']['cells']
        self.assertEqual((len(cells), len(panel['rows']), len(panel['checkpoints'])), (126, 132, 21))
        self.assertEqual(len({x['job_sha256'] for x in cells}), 126)
        self.assertEqual({x['fit_seed'] for x in cells}, {11})
        self.assertEqual(panel['costs']['charged_artifact_bytes'], sum(m['artifact_bytes'] for m in panel['checkpoints']))
        self.assertEqual(panel['costs']['sum_common_metric_seconds'], sum(x['metric_seconds'] for x in cells))
        self.assertFalse(panel['costs']['clocks_are_additive'])
        self.assertEqual(panel['costs']['new_generator_fits'], 0)
        self.assertTrue(all(m['artifact_bytes'] > 10240 and m['artifact_cap_exceeded'] for m in panel['checkpoints']))
        self.assertTrue(any(m['native_successful_pool_size'] < 5 for m in panel['checkpoints']))
        for m in panel['checkpoints']:
            self.assertEqual(sum(a['bytes'] for a in m['artifact_inventory']), m['artifact_bytes'])
            self.assertEqual(m['native_full_grid_successful'], m['native_successful_pool_size'] == 5)
        for document in (panel, panel['physical_panel'], *cells):
            self.assertFalse(document['official_tests_opened'])
            for key in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority'):
                self.assertIsNone(document[key])
        for key in ('full_population_complete', 'five_fit_coverage', 'native_search_complete',
                    'original_runtime_closure_recertified', 'selection_use', 'counts_as_dope_win'):
            self.assertFalse(panel[key])

    def test_every_json_schema_and_manifest_pin(self):
        import jsonschema
        for name in ('panel', 'original-verification', 'inputs.lock', 'manifest'):
            jsonschema.Draft202012Validator(json.loads((ROOT / (name + '.schema.json')).read_text())).validate(
                json.loads((ROOT / (name + '.json')).read_text()))
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        for name, ref in manifest['files'].items():
            raw = (ROOT / name).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (ref['bytes'], ref['sha256']), name)
        for ref in manifest['producer_refs'].values():
            raw = (REPO / ref['path']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (ref['bytes'], ref['sha256']))


if __name__ == '__main__':
    unittest.main()
