"""Scientific coverage and timeout accounting from immutable receipt bytes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import jsonschema

from research.benchmark import publish_baseline_completion as publisher


class BaselineCompletion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock, cls.docs = publisher.load_sources()
        cls.panel = publisher.build(cls.lock, cls.docs)
        cls.out = publisher.REPO/publisher.RELATIVE

    def test_committed_outputs_reproduce_without_private_inputs(self):
        self.assertEqual(json.loads((self.out/'panel.json').read_text()), self.panel)
        self.assertEqual((self.out/'completion.md').read_text(), publisher.render(self.panel))
        self.assertEqual((publisher.REPO/'docs/whitepaper/generated/baseline-budget.tex').read_text(),
                         publisher.render_budget_tex(self.panel))
        self.assertEqual(json.loads((self.out/'timeout-dispositions.json').read_text()),
                         self.docs['forest_input']['timeout_dispositions'])

    def test_all_json_validates_against_committed_schemas(self):
        for name in ('inputs.lock', 'rate', 'panel', 'timeout-dispositions', 'manifest'):
            with self.subTest(file=name):
                jsonschema.Draft202012Validator(json.loads((self.out/(name+'.schema.json')).read_text())).validate(
                    json.loads((self.out/(name+'.json')).read_text()))
        manifest = json.loads((self.out/'manifest.json').read_text())
        for ref in [manifest['publisher'], *manifest['files']]:
            path = publisher.REPO/ref['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), ref['sha256'])
            if 'bytes' in ref:
                self.assertEqual(path.stat().st_size,ref['bytes'])

    def test_corrupted_input_is_rejected_before_json_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/publisher.RELATIVE/'inputs.lock.json'
            path.parent.mkdir(parents=True)
            path.write_text('invalid JSON')
            with self.assertRaisesRegex(ValueError, 'frozen metadata drift'):
                publisher.load_sources(root)

    def test_all_original_fit_job_closure_bytes_are_authenticated(self):
        for key in ('job', 'fit', 'close'):
            source = copy.deepcopy(self.docs['forest_input'])
            source['originals'][0][key] += ' '
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'original receipt bytes differ'):
                publisher.recount_forest(source)

    def test_budget_and_timeout_projection_cannot_change(self):
        forest = self.docs['forest_input']
        for field, value in [('accepted_artifact',True), ('fit_timeout_seconds',1200),
                             ('measured_elapsed_seconds',600), ('status','ok'), ('common_metrics',0)]:
            source = copy.deepcopy(forest)
            source['timeout_dispositions'][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'timeout disposition differs'):
                publisher.recount_forest(source)

    def test_missing_or_duplicate_timeout_receipt_is_rejected(self):
        for duplicate in (False, True):
            source = copy.deepcopy(self.docs['forest_input'])
            source['timeout_dispositions'].pop()
            if duplicate:
                source['timeout_dispositions'].append(source['timeout_dispositions'][0])
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, 'timeout roster differs'):
                publisher.recount_forest(source)

    def test_fits_and_metric_cells_have_separate_denominators(self):
        rows = {r['method']:r for r in self.panel['rows']}
        self.assertEqual(rows['ARF']['logical_metrics_done'],6000)
        self.assertEqual(rows['ARF']['physical_metric_receipts']+rows['ARF']['aliases'],6000)
        self.assertEqual(rows['TabSyn']['default_fit11_ok'],100)
        self.assertEqual(rows['TabSyn']['logical_metrics_done'],143)
        self.assertEqual(rows['TabSyn']['completed_n4n_lineages'],8)
        self.assertEqual(rows['TabSyn']['completed_five_fit_lineages'],0)
        self.assertEqual(rows['TabDDPM']['logical_metrics_done'],132)
        self.assertEqual(rows['TabDDPM']['pending_metric_slots'],5868)
        forest = rows['Forest-Flow']
        self.assertEqual(forest['default_replication_dispositions'],343)
        self.assertEqual(forest['default_replication_ok'],268)
        self.assertEqual(forest['fit_timeouts'],75)
        self.assertEqual(forest['logical_metrics_done'],390)
        self.assertEqual(forest['completed_five_fit_lineages'],13)
        self.assertEqual(forest['default_only_metric_target'],3000)

    def test_non_generative_native_audits_are_not_fits_or_winners(self):
        ts = next(r for r in self.panel['rows'] if r['method']=='TabSyn')
        self.assertEqual(sum(ts['native_audit_status_counts'].values()),100)
        self.assertFalse(ts['native_selection_complete'])
        self.assertEqual(ts['prior_unmeasured_book_charge_seconds'],21000)
        self.assertEqual(ts['prior_failed_native_attempts'],68)

    def test_primary_paper_macros_match_the_authenticated_completion_receipt(self):
        path = publisher.REPO/'docs/whitepaper/scripts/paper_emit.py'
        spec = importlib.util.spec_from_file_location('completion_paper_emit',path)
        module = importlib.util.module_from_spec(spec)
        # Its adjacent public_hardware module is already imported by the
        # benchmark's paper tests; direct discovery also needs that directory.
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'path', [str(path.parent), *sys.path]):
            spec.loader.exec_module(module)
            commands = module.baseline_budget_commands(lambda name,value: '\\newcommand{\\'+name+'}{'+value+'}\n')
        expected = publisher.render_budget_tex(self.panel).splitlines(keepends=True)[1:]
        self.assertEqual(commands,expected)
        numbers = (publisher.REPO/'docs/whitepaper/generated/numbers.tex').read_text()
        self.assertTrue(all(line in numbers for line in commands))
        with patch.object(Path, 'read_bytes', return_value=b'not JSON'):
            with self.assertRaisesRegex(ValueError, 'baseline completion receipt drift'):
                module.baseline_budget_commands(lambda name,value: value)

    def test_forecast_is_not_compute_or_quality_measurement(self):
        self.assertAlmostEqual(self.panel['forest_recent_dispositions_per_hour'],6.994747643994126)
        row = next(r for r in self.panel['rows'] if r['method']=='Forest-Flow')
        self.assertEqual(row['fit_disposition_eta_mdt'][:16],'2026-10-08T10:11')
        self.assertEqual(row['metric_allowance_hours'],6)
        for field in ('mfs_v2','ptf_v1','release_safe_l3','superiority'):
            self.assertIsNone(self.panel[field])
        self.assertFalse(self.panel['official_tests_opened'])
        self.assertEqual((self.panel['new_fits'],self.panel['new_evaluations']),(0,0))


if __name__=='__main__':
    unittest.main()
