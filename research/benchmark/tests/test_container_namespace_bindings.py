"""Private namespace binding controls use opaque references and no real guards."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType
import unittest
from unittest.mock import patch

from research.benchmark import container_namespace_bindings as binding
from research.benchmark.tests import test_container_builtin_bindings as fixture


class NamespaceBindingControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.BuiltinBindingControls()
        self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.sources = self.f.sources
        self.custody = self.f.custody
        self.helper = self.sources.base / 'trusted-namespace-helper.py'
        self.helper.write_bytes(b'# opaque already trusted namespace helper\n'); self.helper.chmod(0o600)
        self.helper_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(binding, '__file__', str(self.helper)); state.start(); self.addCleanup(state.stop)

    def verify(self):
        return binding.prepare_guard_namespaces(str(self.sources.manifest), self.sources.expected,
            self.f.f.f.f.f.compiler_sha, self.f.f.f.f.helper_sha, self.f.f.f.helper_sha,
            self.f.f.helper_sha, self.f.helper_sha, self.helper_sha,
            self.f.f.stdlib, self.f.builtins, batch_started_at=100.0)

    def rejected(self, action):
        with self.assertRaisesRegex(ValueError, '^compressed guard namespace bindings rejected$'): action()

    def capture_parent(self, mutate):
        original = binding.selector.prepare_builtin_bindings
        def call(*args, **kwargs):
            result = original(*args, **kwargs); self.captured = result; return mutate(result)
        return patch.object(binding.selector, 'prepare_builtin_bindings', call)

    def cleared(self):
        for module in self.captured.import_hook.guard_modules.values():
            self.assertNotIn('__builtins__', module.__dict__)

    def test_positive_private_bindings_hook_proxy_lookup_zero_registry_changes(self):
        self.f.builtins.len = len
        before = dict(sys.modules)
        result = self.verify(); prepared = result.builtin_bindings
        for module in prepared.import_hook.guard_modules.values():
            self.assertIs(module.__dict__['__builtins__'], prepared.builtin_references)
        self.assertEqual(before, sys.modules)
        # This synthetic probe verifies Python's mapping-proxy builtin lookup only.
        # No compiled real or fixture guard initializer is executed.
        name = binding.selector.imports.declarations.binding.PACKAGE + '.container_python_custody'
        namespace = prepared.import_hook.guard_modules[name].__dict__
        exec('probe = len(())\nimport hashlib', namespace)
        self.assertEqual(namespace['probe'], 0)
        self.assertIs(namespace['hashlib'], self.f.f.stdlib['hashlib'])
        self.assertIs(result.receipt()['builtins_bound_in_private_namespaces'], True)
        for key in ('source_initializers_run', 'modules_installed', 'actual_import_resolution_verified',
                    'execution_admitted', 'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(result.receipt()[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.assertIsNone(result.receipt()[key])

    def test_existing_builtin_binding_is_rejected_without_overwrite(self):
        foreign = object()
        def mutate(result):
            first = result.import_hook.plan.initialization_order[0]
            result.import_hook.guard_modules[first].__dict__['__builtins__'] = foreign; return result
        with self.capture_parent(mutate): self.rejected(self.verify)
        first = self.captured.import_hook.plan.initialization_order[0]
        self.assertIs(self.captured.import_hook.guard_modules[first].__dict__['__builtins__'], foreign)

    def test_unknown_namespace_key_is_rejected(self):
        def mutate(result):
            first = result.import_hook.plan.initialization_order[0]
            result.import_hook.guard_modules[first].__dict__['unknown'] = None; return result
        with self.capture_parent(mutate): self.rejected(self.verify)
        self.cleared()

    def test_metadata_exact_type_and_empty_loader_reject(self):
        for key, value in [('__file__', 1.0), ('__loader__', object())]:
            def mutate(result):
                first = result.import_hook.plan.initialization_order[0]
                result.import_hook.guard_modules[first].__dict__[key] = value; return result
            with self.capture_parent(mutate): self.rejected(self.verify)
            self.cleared()

    def test_initial_fifo_and_group_write_reject_before_read(self):
        self.helper.chmod(0o660)
        with patch.object(Path, 'open', side_effect=AssertionError('writable helper opened')): self.rejected(self.verify)
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('FIFO opened')): self.rejected(self.verify)

    def test_forged_digest_rejects_before_parent(self):
        class Forged(str):
            def __eq__(self, other): return True
        self.helper_sha = Forged(self.helper_sha)
        with patch.object(binding.selector, 'prepare_builtin_bindings', side_effect=AssertionError('parent reached')):
            self.rejected(self.verify)

    def test_module_subclass_rejects_before_custom_access(self):
        class Hostile(ModuleType):
            def __getattribute__(self, name): raise AssertionError('subclass accessed')
        def mutate(result):
            first = result.import_hook.plan.initialization_order[0]
            modules = dict(result.import_hook.guard_modules); modules[first] = Hostile(first)
            return replace(result, import_hook=replace(result.import_hook, guard_modules=MappingProxyType(modules)))
        with self.capture_parent(mutate): self.rejected(self.verify)

    def test_changed_order_rejects_before_bindings(self):
        def mutate(result): return replace(result, namespace_bindings=result.namespace_bindings[::-1])
        with self.capture_parent(mutate): self.rejected(self.verify)
        self.cleared()

    def test_late_source_drift_clears_owned_bindings(self):
        def mutate(result):
            path = next(iter(self.sources.source.iterdir())); path.write_bytes(b'changed'); return result
        with self.capture_parent(mutate): self.rejected(self.verify)
        self.cleared()

    def test_late_helper_bytes_clear_owned_bindings(self):
        def mutate(result): self.helper.write_bytes(b'changed'); return result
        with self.capture_parent(mutate): self.rejected(self.verify)
        self.cleared()

    def test_late_helper_alias_rejects_before_open_and_clears(self):
        twin = self.sources.base / 'namespace-twin.py'; twin.write_bytes(self.helper.read_bytes())
        def mutate(result): self.helper.unlink(); self.helper.symlink_to(twin); return result
        opened = Path.open
        def refuse(path, *args, **kwargs):
            if path == self.helper and path.is_symlink(): raise AssertionError('late alias opened')
            return opened(path, *args, **kwargs)
        with self.capture_parent(mutate), patch.object(Path, 'open', refuse): self.rejected(self.verify)
        self.cleared()

    def test_final_metadata_drift_clears_owned_bindings(self):
        read = self.custody.read_regular
        def late(path):
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'):
                first = self.captured.import_hook.plan.initialization_order[0]
                self.captured.import_hook.guard_modules[first].__dict__['__file__'] = 'changed'
            return result
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late):
            self.rejected(self.verify)
        self.cleared()

    def test_final_package_reference_drift_clears_owned_bindings(self):
        read = self.custody.read_regular
        def late(path):
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'):
                first = self.captured.import_hook.plan.initialization_order[0]
                self.captured.import_hook.package.__dict__[first.rsplit('.', 1)[1]] = ModuleType('substitute')
            return result
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late):
            self.rejected(self.verify)
        self.cleared()

    def test_original_deadline_before_first_read(self):
        self.sources.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()

    def test_original_deadline_after_final_manifest_clears_bindings(self):
        read = self.custody.read_regular
        final_helper_read = False
        def late(path):
            nonlocal final_helper_read
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'): final_helper_read = True
            if path == self.sources.manifest and final_helper_read:
                self.sources.clock.return_value = 700.0
            return result
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()
        self.cleared()

    def test_ordinary_io_timeout_is_fixed_and_clears_bindings(self):
        read = self.custody.read_regular
        def late(path):
            if path == self.helper and hasattr(self, 'captured'): raise TimeoutError('private namespace detail')
            return read(path)
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late):
            self.rejected(self.verify)
        self.cleared()


if __name__ == '__main__': unittest.main(verbosity=2)
