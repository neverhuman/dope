"""Private registry controls use opaque references without live activation."""
import hashlib
import os
from pathlib import Path
from types import MappingProxyType
import unittest
from unittest.mock import patch

from research.benchmark import container_private_registry as private
from research.benchmark.tests import test_container_registry_bindings as fixture


class PrivateRegistryControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.RegistryBindingControls()
        self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.sources = self.f.sources; self.custody = self.f.custody
        self.helper = self.sources.base / 'trusted-private-registry-helper.py'
        self.helper.write_bytes(b'# opaque preowned private registry helper\n'); self.helper.chmod(0o600)
        self.helper_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(private, '__file__', str(self.helper)); state.start(); self.addCleanup(state.stop)

    def verify(self):
        f = self.f.f
        return private.prepare_private_registry(str(self.sources.manifest), self.sources.expected,
            f.f.f.f.f.f.compiler_sha, f.f.f.f.f.helper_sha, f.f.f.f.helper_sha,
            f.f.f.helper_sha, f.f.helper_sha, f.helper_sha, self.f.helper_sha, self.helper_sha,
            f.f.f.stdlib, f.f.builtins, self.f.registry, batch_started_at=100.0)

    def rejected(self):
        with self.assertRaisesRegex(ValueError, '^compressed private registry preparation rejected$'): self.verify()

    def capture_parent(self, mutate):
        original = private.registry_plan.prepare_registry_bindings
        def call(*args, **kwargs):
            result = original(*args, **kwargs); self.captured = result; return mutate(result)
        return patch.object(private.registry_plan, 'prepare_registry_bindings', call)

    def test_positive_readonly_overlay_original_unchanged_reference_identities(self):
        before = dict(self.f.registry); result = self.verify()
        self.assertEqual(self.f.registry, before)
        self.assertIs(type(result.registry_references), MappingProxyType)
        self.assertEqual(len(result.registry_references), len(before) + 16)
        for name, value in before.items(): self.assertIs(result.registry_references[name], value)
        for name, value in result.bindings.additions: self.assertIs(result.registry_references[name], value)
        with self.assertRaises(TypeError): result.registry_references['new'] = object()
        receipt = result.receipt()
        self.assertIs(receipt['private_registry_prepared'], True)
        self.assertEqual(receipt['private_guard_references_bound'], 16)
        self.assertEqual(receipt['live_registry_entries_installed'], 0)
        for key in ('source_initializers_run', 'actual_import_resolution_verified',
                    'execution_admitted', 'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(receipt[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'): self.assertIsNone(receipt[key])

    def test_early_reserved_prefix_collision_before_parent(self):
        self.f.registry[self.f.package + '.unexpected'] = object()
        with patch.object(private.registry_plan, 'prepare_registry_bindings', side_effect=AssertionError('parent reached')):
            self.rejected()

    def test_exact_dict_and_key_types_before_parent_or_callbacks(self):
        class Hostile(dict):
            def __iter__(self): raise AssertionError('custom iteration')
        class Key(str):
            def __eq__(self, other): raise AssertionError('custom equality')
            __hash__ = str.__hash__
        for registry in (Hostile(), {Key('foreign'): object()}):
            self.f.registry = registry
            with patch.object(private.registry_plan, 'prepare_registry_bindings', side_effect=AssertionError('parent reached')):
                self.rejected()

    def test_digest_subclass_before_parent(self):
        self.helper_sha = type('Forged', (str,), {'__eq__': lambda self, other: True})(self.helper_sha)
        with patch.object(private.registry_plan, 'prepare_registry_bindings', side_effect=AssertionError('parent reached')):
            self.rejected()

    def test_late_original_reference_replacement_reject_without_undoing_foreign_change(self):
        foreign = object()
        def mutate(result): self.f.registry['unrelated'] = foreign; return result
        with self.capture_parent(mutate): self.rejected()
        self.assertIs(self.f.registry['unrelated'], foreign)

    def test_late_original_addition_and_removal_reject(self):
        def mutate(result): self.f.registry['new'] = object(); return result
        with self.capture_parent(mutate): self.rejected()
        self.f.registry.pop('new')
        def mutate(result): self.f.registry.pop('unrelated'); return result
        with self.capture_parent(mutate): self.rejected()

    def test_final_reserved_collision_reject_preserving_foreign_value(self):
        read = self.custody.read_regular; foreign = object()
        def late(path):
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'): self.f.registry[self.f.package] = foreign
            return result
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late): self.rejected()
        self.assertIs(self.f.registry[self.f.package], foreign)

    def test_final_private_metadata_drift_reject(self):
        read = self.custody.read_regular
        def late(path):
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'):
                hook = self.captured.namespaces.builtin_bindings.import_hook
                hook.guard_modules[hook.plan.initialization_order[0]].__dict__['__file__'] = 'changed'
            return result
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late): self.rejected()

    def test_late_source_drift_reject(self):
        def mutate(result): next(iter(self.sources.source.iterdir())).write_bytes(b'changed'); return result
        with self.capture_parent(mutate): self.rejected()

    def test_late_own_helper_drift_reject(self):
        def mutate(result): self.helper.write_bytes(b'changed'); return result
        with self.capture_parent(mutate): self.rejected()

    def test_initial_group_write_and_fifo_reject_before_open(self):
        self.helper.chmod(0o660)
        with patch.object(Path, 'open', side_effect=AssertionError('writable opened')): self.rejected()
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('FIFO opened')): self.rejected()

    def test_late_helper_alias_reject(self):
        twin = self.sources.base / 'private-registry-twin.py'; twin.write_bytes(self.helper.read_bytes())
        def mutate(result): self.helper.unlink(); self.helper.symlink_to(twin); return result
        with self.capture_parent(mutate): self.rejected()

    def test_original_deadline_before_first_read(self):
        self.sources.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()

    def test_original_deadline_after_final_manifest(self):
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
            if path == self.helper and hasattr(self, 'captured'): raise TimeoutError('private context')
            return read(path)
        with self.capture_parent(lambda result: result), patch.object(self.custody, 'read_regular', late): self.rejected()


if __name__ == '__main__': unittest.main(verbosity=2)
