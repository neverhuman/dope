"""Registry preparation controls keep all real guard initializers unexecuted."""
import hashlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import container_registry_bindings as binding
from research.benchmark.tests import test_container_namespace_bindings as fixture


class RegistryBindingControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.NamespaceBindingControls()
        self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.sources = self.f.sources
        self.custody = self.f.custody
        self.registry = {'unrelated': object()}
        self.package = binding.namespaces.selector.imports.declarations.binding.PACKAGE
        self.helper = self.sources.base / 'trusted-registry-helper.py'
        self.helper.write_bytes(b'# opaque caller-owned registry helper\n'); self.helper.chmod(0o600)
        self.helper_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(binding, '__file__', str(self.helper)); state.start(); self.addCleanup(state.stop)

    def verify(self):
        f = self.f
        return binding.prepare_registry_bindings(str(self.sources.manifest), self.sources.expected,
            f.f.f.f.f.f.compiler_sha, f.f.f.f.f.helper_sha, f.f.f.f.helper_sha,
            f.f.f.helper_sha, f.f.helper_sha, f.helper_sha, self.helper_sha,
            f.f.f.stdlib, f.f.builtins, self.registry, batch_started_at=100.0)

    def rejected(self):
        with self.assertRaisesRegex(ValueError, '^compressed guard registry preparation rejected$'): self.verify()

    def capture_parent(self, mutate):
        original = binding.namespaces.prepare_guard_namespaces
        def call(*args, **kwargs):
            result = original(*args, **kwargs); self.captured = result; return mutate(result)
        return patch.object(binding.namespaces, 'prepare_guard_namespaces', call)

    def test_positive_sixteen_additions_identity_order_registry_unchanged(self):
        before = dict(self.registry)
        result = self.verify(); hook = result.namespaces.builtin_bindings.import_hook
        self.assertIs(result.registry, self.registry); self.assertEqual(self.registry, before)
        self.assertEqual(result.additions[0], (self.package, hook.package))
        self.assertEqual(tuple(name for name, _ in result.additions[1:]), hook.plan.initialization_order)
        for name, module in result.additions[1:]: self.assertIs(module, hook.guard_modules[name])
        self.assertEqual(result.receipt()['registry_additions_planned'], 16)
        self.assertEqual(result.receipt()['registry_entries_installed'], 0)
        for key in ('source_initializers_run', 'actual_import_resolution_verified',
                    'execution_admitted', 'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(result.receipt()[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'): self.assertIsNone(result.receipt()[key])

    def test_package_and_unknown_descendant_collision_before_parent(self):
        for name in (self.package, self.package + '.unknown', self.package + '.unknown.deep'):
            with self.subTest(name=name):
                self.registry[name] = object()
                with patch.object(binding.namespaces, 'prepare_guard_namespaces', side_effect=AssertionError('parent reached')):
                    self.rejected()
                del self.registry[name]

    def test_nonreserved_prefix_is_allowed(self):
        self.registry[self.package + '_unrelated'] = object(); self.verify()

    def test_exact_registry_type_before_iteration(self):
        class Hostile(dict):
            def __iter__(self): raise AssertionError('subclass iterated')
        self.registry = Hostile()
        with patch.object(binding.namespaces, 'prepare_guard_namespaces', side_effect=AssertionError('parent reached')):
            self.rejected()

    def test_key_subclass_reject_before_comparison(self):
        class Hostile(str):
            def __eq__(self, other): raise AssertionError('custom equality used')
            __hash__ = str.__hash__
            def startswith(self, *args): raise AssertionError('custom prefix used')
        self.registry = {Hostile('foreign'): object()}
        with patch.object(binding.namespaces, 'prepare_guard_namespaces', side_effect=AssertionError('parent reached')):
            self.rejected()

    def test_digest_subclass_reject_before_parent(self):
        self.helper_sha = type('Forged', (str,), {'__eq__': lambda self, other: True})(self.helper_sha)
        with patch.object(binding.namespaces, 'prepare_guard_namespaces', side_effect=AssertionError('parent reached')):
            self.rejected()

    def test_late_collision_preserves_foreign_entry(self):
        foreign = object()
        def mutate(result): self.registry[self.package] = foreign; return result
        with self.capture_parent(mutate): self.rejected()
        self.assertIs(self.registry[self.package], foreign)

    def test_final_read_collision_reject(self):
        read = self.custody.read_regular
        def late(path):
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'): self.registry[self.package + '.late'] = object()
            return result
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late): self.rejected()

    def test_final_metadata_drift_reject(self):
        read = self.custody.read_regular
        def late(path):
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'):
                hook = self.captured.builtin_bindings.import_hook
                hook.guard_modules[hook.plan.initialization_order[0]].__dict__['__file__'] = 'changed'
            return result
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late): self.rejected()

    def test_late_source_drift_reject(self):
        def mutate(result): next(iter(self.sources.source.iterdir())).write_bytes(b'changed'); return result
        with self.capture_parent(mutate): self.rejected()

    def test_initial_writable_and_fifo_reject_before_open(self):
        self.helper.chmod(0o660)
        with patch.object(Path, 'open', side_effect=AssertionError('writable opened')): self.rejected()
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('FIFO opened')): self.rejected()

    def test_late_helper_drift_and_alias_reject(self):
        body = self.helper.read_bytes()
        def mutate(result): self.helper.write_bytes(b'changed'); return result
        with self.capture_parent(mutate): self.rejected()
        self.helper.write_bytes(body)
        twin = self.sources.base / 'registry-twin.py'; twin.write_bytes(body)
        def mutate(result): self.helper.unlink(); self.helper.symlink_to(twin); return result
        with self.capture_parent(mutate): self.rejected()

    def test_deadline_before_first_read(self):
        self.sources.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()

    def test_deadline_after_final_manifest(self):
        read = self.custody.read_regular; final_helper_read = False
        def late(path):
            nonlocal final_helper_read
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'): final_helper_read = True
            if path == self.sources.manifest and final_helper_read: self.sources.clock.return_value = 700.0
            return result
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()

    def test_ordinary_io_timeout_sanitized(self):
        read = self.custody.read_regular
        def late(path):
            if path == self.helper and hasattr(self, 'captured'): raise TimeoutError('private registry detail')
            return read(path)
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late): self.rejected()


if __name__ == '__main__': unittest.main(verbosity=2)
