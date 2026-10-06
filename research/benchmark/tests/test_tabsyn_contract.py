"""Opaque provider controls and disclosed numeric TabSyn contract boundaries."""
import builtins
import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path
import sys
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.benchmark import tabsyn_adapter as adapter
from research.benchmark import tabsyn_generated_contract as generated
from research.benchmark import tabsyn_runtime_guard as guard


ROOT = Path(__file__).resolve().parents[3]
PROPOSAL = ROOT/'research/benchmark/tabsyn-native-grid.proposal.json'


def configuration(fixture=False):
    value = json.loads(PROPOSAL.read_bytes())['configs'][0]
    if fixture:
        value['vae_epochs'] = value['diffusion_epochs'] = 2
        resign(value)
    return value


def resign(value):
    canonical = {k: v for k, v in value.items() if k != 'config_sha256'}
    value['config_sha256'] = hashlib.sha256(json.dumps(canonical,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inventory(root):
    return dict(root=str(root), aliases={},
        directories=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir()],
        files={str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=guard.sha(p))
               for p in root.rglob('*') if p.is_file()})


class Runtime(unittest.TestCase):
    def setUp(self):
        area = ROOT/'target/tabsyn-guard-controls'
        area.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=area)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.old = self.base/'previous-preparation'
        self.old.mkdir()
        prefix = self.old/'python/cpython'
        self.python = prefix/'bin/python3.10'
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b'opaque interpreter')
        library = prefix/'lib/python3.10'
        (library/'lib-dynload').mkdir(parents=True)
        (library/'os.py').write_bytes(b'opaque standard library')
        site = self.old/'venv/lib/python3.10/site-packages'
        (site/'numpy').mkdir(parents=True)
        self.provider = site/'numpy/__init__.py'
        self.provider.write_bytes(b'raise AssertionError("initializer executed")')
        source = self.old/'source'
        source.mkdir()
        (source/'opaque.py').write_bytes(b'opaque author source')
        control = self.old/'control'
        control.mkdir()
        (control/'adapter.py').write_bytes(b'opaque previous controls')
        self.control = self.base/'new-control'
        self.own = self.control/'research/benchmark/tabsyn_runtime_guard.py'
        self.own.parent.mkdir(parents=True)
        self.own.write_bytes(b'opaque new guard')
        self.adapter = self.own.with_name('tabsyn_adapter.py')
        self.adapter.write_bytes(b'opaque new adapter')
        self.cache = self.base/'empty-cache'
        self.cache.mkdir()
        self.lock = self.old/'runtime.lock.json'
        self.lock.write_text(json.dumps(dict(format='dope-tabsyn-owned-runtime', version=1,
            official_tests_opened=False, execution_admitted=False,
            gpu_runtime_closure_certified=False, base=str(self.old),
            LD_LIBRARY_PATH='', python=str(self.python), python_sha256=guard.sha(self.python),
            stdlib=str(library), dynload=str(library/'lib-dynload'), package_site=str(site),
            source=str(source), control=str(control), system_files={},
            roots=[inventory(self.old/name) for name in ['python', 'venv', 'source', 'control']])))
        self.expected = guard.sha(self.lock)
        self.code = self.base/'code.lock.json'
        self.code.write_text(json.dumps(dict(format='dope-tabsyn-adapter-controls', version=1,
            runtime_sha256=self.expected, source_commit=adapter.AUTHOR_COMMIT,
            execution_admitted=False, inventory=inventory(self.control),
            author_files={'opaque.py': guard.sha(source/'opaque.py')})))
        self.code_expected = guard.sha(self.code)
        self.paths = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.paths))
        for context in [patch.dict(os.environ, {'LD_LIBRARY_PATH': '', 'LD_PRELOAD': ''}),
                        patch.object(sys, 'executable', str(self.python)),
                        patch.object(sys, 'pycache_prefix', str(self.cache)),
                        patch.object(sys, 'dont_write_bytecode', True),
                        patch.object(sys, 'flags', SimpleNamespace(isolated=1, no_site=1, optimize=0)),
                        patch.object(guard, '__file__', str(self.own)),
                        patch.object(adapter, '__file__', str(self.adapter))]:
            context.start()
            self.addCleanup(context.stop)

    def verify(self):
        return guard.verify(self.lock, self.expected, self.code, self.code_expected)

    def reject_before_import(self, call=None):
        original, initialized = builtins.__import__, []
        def deny(name, *args, **kwargs):
            if name.split('.')[0] in ['numpy', 'torch', 'sklearn', 'safetensors', 'tabsyn', 'eval']:
                initialized.append(name)
                raise AssertionError('dependency initialized before rejection')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=deny), self.assertRaises(ValueError):
            (call or self.verify)()
        self.assertEqual(initialized, [])

    def test_valid_reuse_does_not_change_frozen_preparation(self):
        before = inventory(self.old/'control')
        self.verify()
        self.assertEqual(inventory(self.old/'control'), before)
        self.assertEqual(sys.path[0], str(self.control))

    def test_initializer_mutation_rejected(self):
        self.provider.write_bytes(b'different opaque provider')
        self.reject_before_import()

    def test_added_bytecode_rejected(self):
        (self.provider.parent/'opaque.pyc').write_bytes(b'opaque bytecode')
        self.reject_before_import()

    def test_bootstrap_never_executes_injected_guard_cache(self):
        self.own.write_bytes(b'loaded_from_verified_source = True\n')
        cached = Path(importlib.util.cache_from_source(str(self.own), optimization=''))
        cached.parent.mkdir(parents=True)
        payload = compile('raise AssertionError("cached initializer executed")', str(self.own), 'exec')
        header = importlib.util.MAGIC_NUMBER + struct.pack('<III', 0,
            int(self.own.stat().st_mtime), self.own.stat().st_size)
        cached.write_bytes(header + marshal.dumps(payload))
        loaded = generated.load_guard(self.own, guard.sha(self.own))
        self.assertTrue(loaded.loaded_from_verified_source)
        self.reject_before_import()

    def test_bootstrap_digest_rejects_before_initializer(self):
        self.own.write_bytes(b'raise AssertionError("source initializer executed")\n')
        with self.assertRaises(ValueError):
            generated.load_guard(self.own, '0'*64)

    def test_loss_extraction_skips_original_module_initializer(self):
        source = self.base/'opaque-loss.py'
        source.write_bytes(b'raise AssertionError("author initializer executed")\n'
                           b'def compute_loss(value):\n    return value + 1\n')
        loss = generated.load_original_loss(source, {})
        self.assertEqual(loss(2), 3)

    def test_empty_directory_mutation_rejected(self):
        (self.provider.parent/'added').mkdir()
        self.reject_before_import()

    def test_added_alias_rejected(self):
        (self.provider.parent/'alias').symlink_to(self.provider)
        self.reject_before_import()

    def test_unbound_interpreter_zip_rejected(self):
        (self.python.parents[1]/'lib/python310.zip').write_bytes(b'unbound archive')
        self.reject_before_import()

    def test_control_drift_rejected(self):
        self.adapter.write_bytes(b'different control provider')
        self.reject_before_import()

    def test_old_control_drift_rejected(self):
        (self.old/'control/adapter.py').write_bytes(b'different previous controls')
        self.reject_before_import()

    def test_guard_origin_rejected(self):
        with patch.object(guard, '__file__', str(self.adapter)):
            self.reject_before_import()

    def test_adapter_origin_rejected(self):
        with patch.object(adapter, '__file__', str(self.own)):
            self.reject_before_import()

    def test_runtime_anchor_rejected(self):
        self.expected = '0'*64
        self.reject_before_import()

    def test_code_anchor_rejected(self):
        self.code_expected = '0'*64
        self.reject_before_import()

    def test_preload_rejected(self):
        with patch.dict(os.environ, {'LD_PRELOAD': 'opaque'}):
            self.reject_before_import()

    def test_nonisolated_interpreter_rejected_before_manifest(self):
        with patch.object(sys, 'flags', SimpleNamespace(isolated=0, no_site=1)), \
                patch.object(guard, 'bound', side_effect=AssertionError('manifest read')):
            self.reject_before_import()

    def test_fit_checks_runtime_before_initializer(self):
        self.provider.write_bytes(b'drift')
        self.reject_before_import(lambda: adapter.fit(None, None, b'{}', self.base/'artifact',
            configuration(True), generated_fixture=True, runtime_path=self.lock,
            runtime_sha256=self.expected, code_path=self.code, code_sha256=self.code_expected))

    def test_sample_checks_runtime_before_initializer(self):
        self.provider.write_bytes(b'drift')
        self.reject_before_import(lambda: adapter.sample(self.base/'artifact', 16, 101, {},
            runtime_path=self.lock, runtime_sha256=self.expected,
            code_path=self.code, code_sha256=self.code_expected))

    def test_native_gpu_evaluator_not_admitted(self):
        self.reject_before_import(lambda: adapter.native_objective(None, None, {},
            runtime_path=self.lock, runtime_sha256=self.expected,
            code_path=self.code, code_sha256=self.code_expected))


class Contract(unittest.TestCase):
    def test_full_author_defaults_and_proposed_beta_have_exact_digests(self):
        proposal = json.loads(PROPOSAL.read_bytes())
        for config in proposal['configs']:
            adapter.validate_config(config)
        self.assertFalse(proposal['execution_admitted'])
        self.assertEqual(proposal['native_objective']['auditor_configurations'], 36)
        self.assertFalse(proposal['native_objective']['auditor_grid_is_generator_search_space'])

    def test_reduced_epochs_only_explicit_generated_fixture(self):
        adapter.validate_config(configuration(True), True)
        with self.assertRaises(ValueError):
            adapter.validate_config(configuration(True))

    def test_numeric_alias_configuration_rejected(self):
        for name, value in [('attention_heads', True), ('vae_epochs', 4000.0), ('weight_decay', False)]:
            config = configuration()
            config[name] = value
            resign(config)
            with self.assertRaises(ValueError):
                adapter.validate_config(config)

    def test_real_fit_is_unadmitted_before_any_provider_read(self):
        with patch.object(adapter, 'verify_runtime', side_effect=AssertionError('runtime read')), \
                self.assertRaises(ValueError):
            adapter.fit(None, None, b'{}', None, configuration(), device='cuda:0',
                runtime_path=None, runtime_sha256=None, code_path=None, code_sha256=None)

    def test_native_partial_failure_and_ties(self):
        common = dict(status='ok', native_value=.5, elapsed_seconds=1.)
        rows = [dict(common, artifact_bytes=20, config_sha256='a'*64),
                dict(common, artifact_bytes=10, config_sha256='b'*64),
                dict(common, artifact_bytes=10, config_sha256='a'*64),
                dict(status='partial_failure', native_value=99, artifact_bytes=1,
                     config_sha256='0'*64, elapsed_seconds=600.)]
        self.assertIs(adapter.select_native(rows), rows[2])
        self.assertIsNone(adapter.select_native(rows[-1:]))

    def test_all_failure_time_is_charged(self):
        with self.assertRaises(ValueError):
            adapter.select_native([dict(status='partial_failure', elapsed_seconds=43201.)])

    def test_nonfinite_cost_and_numeric_value_rejected(self):
        for elapsed in [float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                adapter.select_native([dict(status='partial_failure', elapsed_seconds=elapsed)])
        with self.assertRaises(ValueError):
            adapter.select_native([dict(status='ok', native_value=True, elapsed_seconds=1.,
                artifact_bytes=1, config_sha256='a'*64)])

    def test_trial_budget_rejected(self):
        with self.assertRaises(ValueError):
            adapter.select_native([dict(status='partial_failure', elapsed_seconds=0.)]*9)

    def test_expired_whole_fit_deadline(self):
        with self.assertRaises(TimeoutError):
            adapter.check_deadline(0)

    def test_native_dispatch_keeps_author_arrays_and_uses_r2(self):
        # This is an interface seam with a fake evaluator, not GPU evidence.
        import numpy as np
        synthetic = np.array([[2., .2], [3., .3]])
        validation = np.array([[2., .4], [4., .5]])
        metadata = dict(task_type='regression', target_col_idx=[0])
        seen = []
        def evaluate(train, heldout, info):
            seen.append((train, heldout, info))
            return [dict(name='XGBRegressor', r2=-.25)], [dict(name='XGBRegressor', r2=99)]
        admitted = dict(gpu_runtime_closure_certified=True, execution_admitted=True)
        with patch.object(adapter, 'verify_runtime', return_value=admitted), \
                patch.object(adapter, 'seed_all') as seed, \
                patch.dict(sys.modules, {'eval.mle.mle': SimpleNamespace(_evaluate_regression=evaluate)}):
            result = adapter.native_objective(synthetic, validation, metadata,
                runtime_path=None, runtime_sha256=None, code_path=None, code_sha256=None)
        self.assertIs(seen[0][0], synthetic)
        self.assertIs(seen[0][1], validation)
        self.assertIs(seen[0][2], metadata)
        self.assertEqual(result['native_value'], -.25)
        seed.assert_called_once_with(11)

    def test_uninformative_author_target_does_not_select_native_winner(self):
        import numpy as np
        admitted = dict(gpu_runtime_closure_certified=True, execution_admitted=True)
        evaluator = SimpleNamespace(_evaluate_regression=lambda *args: self.fail('uninformative evaluation'))
        with patch.object(adapter, 'verify_runtime', return_value=admitted), \
                patch.dict(sys.modules, {'eval.mle.mle': evaluator}):
            result = adapter.native_objective(np.ones((18, 2)), np.zeros((3, 2)),
                dict(task_type='regression', target_col_idx=[0]), runtime_path=None,
                runtime_sha256=None, code_path=None, code_sha256=None)
        self.assertIsNone(result['native_value'])
        self.assertEqual(result['status'], 'tuning_inapplicable_uninformative_author_target')


if __name__ == '__main__':
    unittest.main()
