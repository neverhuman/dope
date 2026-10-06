"""Publication integrity controls on committed cost metadata, never study rows."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

BENCHMARK = Path(__file__).resolve().parents[1]
PUBLIC = BENCHMARK / 'results/work-order-b-compute-costs-v2'


def source_module(path, name):
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


class PublicationIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = source_module(PUBLIC / 'render.py', 'cost_renderer')
        cls.publisher_path = BENCHMARK / 'publish_work_order_b_costs.py'
        renderer = cls.renderer
        cls.base = renderer.primitives(cls.publisher_path)
        cls.report = json.loads(renderer.checked(PUBLIC / 'results.json', renderer.REPORT_SHA),
                                parse_constant=cls.base['reject_constant'],
                                parse_float=cls.base['finite_float'],
                                object_pairs_hook=cls.base['unique_object'])

    def validate(self, report):
        self.renderer.validate(report, self.base)

    def test_primitives_use_single_pinned_owner_without_persistent_import_changes(self):
        original = (sys.path[:], sys.pycache_prefix, sys.dont_write_bytecode)
        primitives = self.renderer.primitives(self.publisher_path)
        current = {n.name for n in ast.parse((PUBLIC / 'render.py').read_bytes()).body
                   if isinstance(n, ast.FunctionDef)}
        for name in primitives:
            self.assertNotIn(name, current)
            self.assertEqual(Path(primitives[name].__globals__['__file__']), self.publisher_path)
        self.assertEqual((sys.path, sys.pycache_prefix, sys.dont_write_bytecode), original)
        self.assertNotIn('publish_work_order_b_costs', sys.modules)

    def test_cached_publisher_rejected_before_initialization(self):
        with patch.dict(sys.modules, {'publish_work_order_b_costs': object()}), \
                patch('builtins.compile', side_effect=AssertionError('initialized')) as compiler:
            with self.assertRaisesRegex(ValueError, 'cache must be absent'):
                self.renderer.primitives(self.publisher_path)
            compiler.assert_not_called()

    def test_unexpected_source_cache_rejected_before_initialization(self):
        cache = self.publisher_path.parent / '.cost-report-source-only-cache'
        exists = Path.exists
        with patch.object(Path, 'exists', lambda p: p == cache or exists(p)), \
                patch('builtins.compile', side_effect=AssertionError('initialized')) as compiler:
            with self.assertRaisesRegex(ValueError, 'cache must be absent'):
                self.renderer.primitives(self.publisher_path)
            compiler.assert_not_called()

    def test_committed_csv_schema_exact_isolated_replay(self):
        expected = {name: (PUBLIC / name).read_bytes()
                    for name in ('costs.csv', 'results.schema.json')}
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(PUBLIC / 'render.py'),
            '--publisher', str(self.publisher_path)], capture_output=True, timeout=30,
            env={'PATH': os.defpath, 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1',
                 'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual({name: (PUBLIC / name).read_bytes() for name in expected}, expected)
        self.assertEqual(self.renderer.project(self.report, self.base), expected)
        self.assertEqual(self.renderer.project(copy.deepcopy(self.report), self.base), expected)

    def test_digest_drift_precedes_source_load_and_json_decode(self):
        argv = ['render.py', '--publisher', str(self.publisher_path),
                '--input', str(PUBLIC / 'results.json'), '--output', str(PUBLIC)]
        with patch.object(sys, 'argv', argv), patch.object(self.renderer, 'PUBLISHER_SHA', '0' * 64), \
                patch('builtins.compile', side_effect=AssertionError('source loaded')) as compile_call, \
                patch.object(self.renderer.json, 'loads', side_effect=AssertionError('decoded')) as decode:
            with self.assertRaisesRegex(ValueError, 'digest changed'): self.renderer.main()
            compile_call.assert_not_called(); decode.assert_not_called()
        with patch.object(sys, 'argv', argv), patch.object(self.renderer, 'REPORT_SHA', '0' * 64), \
                patch.object(self.renderer.json, 'loads', side_effect=AssertionError('decoded')) as decode:
            with self.assertRaisesRegex(ValueError, 'digest changed'): self.renderer.main()
            decode.assert_not_called()

    def test_scheduling_cutoff_keeps_recorded_status_and_nonfailure_class(self):
        report = copy.deepcopy(self.report); self.validate(report)
        ctgan = next(r for r in report['rows'] if r['id'] == 'core_phase_01')
        self.assertEqual(ctgan['statuses']['deadline_unstarted'], 260)
        self.assertEqual(report['native_outcome_classes']['CTGAN/fit']['scheduling_cutoff'], 261)
        recorded = copy.deepcopy(ctgan['statuses']); self.renderer.project(report, self.base)
        self.assertEqual(ctgan['statuses'], recorded)
        report['native_outcome_classes']['CTGAN/fit']['method_failure'] = 261
        with self.assertRaisesRegex(ValueError, 'scheduling class'): self.validate(report)

    def test_conflicting_and_identical_row_aliases_are_rejected(self):
        for seconds_delta in (0, 1):
            with self.subTest(delta=seconds_delta):
                report = copy.deepcopy(self.report); alias = copy.deepcopy(report['rows'][0])
                alias['seconds'] += seconds_delta; report['rows'].append(alias)
                with self.assertRaisesRegex(ValueError, 'physical row alias'): self.validate(report)

    def test_unknown_nulls_are_neither_false_nor_zero(self):
        self.validate(self.report)
        for value in (False, 0):
            with self.subTest(value=value):
                report = copy.deepcopy(self.report); report['resources']['historical_co_tenant'] = value
                with self.assertRaisesRegex(ValueError, 'unknown hardware'): self.validate(report)
                report = copy.deepcopy(self.report); report['partial_scopes']['tabddpm']['scientific_seconds'] = value
                with self.assertRaisesRegex(ValueError, 'TD boundary'): self.validate(report)
        self.assertIsNone(self.report['resources']['historical_co_tenant'])
        self.assertIsNone(self.report['partial_scopes']['tabddpm']['coordinator_numeric_exit'])

    def test_arf_native_and_density_alias_cannot_be_charged_twice(self):
        report = copy.deepcopy(self.report)
        alias = copy.deepcopy(next(r for r in report['rows'] if r['method'] == 'ARF'
                                   and r['phase'] == 'fit_and_native_wrapper'))
        alias['id'] = 'generated_distinct_alias'; report['rows'].append(alias)
        with self.assertRaisesRegex(ValueError, 'ARF native wrapper'): self.validate(report)
        report = copy.deepcopy(self.report)
        next(r for r in report['rows'] if r['id'] == 'density_shared_validation_v1')['accounting'] = 'separate_run_aggregate'
        with self.assertRaisesRegex(ValueError, 'density alias'): self.validate(report)


if __name__ == '__main__':
    unittest.main()
