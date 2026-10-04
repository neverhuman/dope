"""Owned initialization plans reject drift without executing staged initializers."""
import hashlib
import os
from types import MappingProxyType
import unittest
from unittest.mock import patch

from research.benchmark import container_guard_initialization as initialization
from research.benchmark.tests import test_container_private_registry as fixture


class GuardInitializationControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.PrivateRegistryControls()
        self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.sources = self.f.sources
        self.helper = self.sources.base / 'trusted-initialization-helper.py'
        self.helper.write_bytes(b'# opaque trusted initialization helper\n'); self.helper.chmod(0o600)
        self.helper_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(initialization, '__file__', str(self.helper))
        state.start(); self.addCleanup(state.stop)

    def verify(self):
        f = self.f.f.f
        return initialization.prepare_guard_initialization(str(self.sources.manifest), self.sources.expected,
            f.f.f.f.f.f.compiler_sha, f.f.f.f.f.helper_sha, f.f.f.f.helper_sha,
            f.f.f.helper_sha, f.f.helper_sha, f.helper_sha, self.f.f.helper_sha,
            self.f.helper_sha, self.helper_sha, f.f.f.stdlib, f.f.builtins,
            self.f.f.registry, batch_started_at=100.0)

    def rejected(self):
        with self.assertRaisesRegex(ValueError, '^compressed guard initialization preparation rejected$'):
            self.verify()

    def parent(self, mutate):
        original = initialization.private.prepare_private_registry
        def call(*args, **kwargs):
            self.captured = original(*args, **kwargs)
            return mutate(self.captured)
        return patch.object(initialization.private, 'prepare_private_registry', call)

    def test_positive_bound_code_namespaces_dependencies_zero_execution(self):
        before = dict(self.f.f.registry)
        result = self.verify()
        self.assertEqual(self.f.f.registry, before)
        self.assertEqual(len(result.steps), 15)
        seen = set()
        hook = result.private_registry.bindings.namespaces.builtin_bindings.import_hook
        descriptors = {item.name: item for item in hook.plan.modules}
        for step in result.steps:
            self.assertIs(step.code, descriptors[step.name].code)
            self.assertIs(step.module, result.private_registry.registry_references[step.name])
            self.assertIs(type(step.namespace), MappingProxyType)
            self.assertEqual(dict(step.namespace), step.module.__dict__)
            self.assertTrue(set(step.dependencies) <= seen); seen.add(step.name)
            self.assertIn('staged initializer executed', step.code.co_consts)
            with self.assertRaises(TypeError): step.namespace['new'] = object()
        receipt = result.receipt()
        self.assertEqual(receipt['initialization_steps_bound'], 15)
        self.assertIs(receipt['live_registry_activation_required'], True)
        for name in ('source_initializers_run', 'execution_admitted', 'full_runtime_closure_certified',
                     'official_tests_opened', 'actual_import_resolution_verified'):
            self.assertIs(receipt[name], False)
        for name in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'): self.assertIsNone(receipt[name])

    def test_digest_subclass_rejected_before_parent(self):
        self.helper_sha = type('Forged', (str,), {'__eq__': lambda a, b: True})(self.helper_sha)
        with patch.object(initialization.private, 'prepare_private_registry', side_effect=AssertionError('parent')):
            self.rejected()

    def test_helper_fifo_before_parent_or_reads(self):
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(initialization.private, 'prepare_private_registry', side_effect=AssertionError('parent')):
            self.rejected()

    def test_expired_before_parent(self):
        self.sources.clock.return_value = 700.0
        with patch.object(initialization.private, 'prepare_private_registry', side_effect=AssertionError('parent')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()

    def test_code_replaced_without_loading_or_running_it(self):
        def mutate(result):
            descriptor = result.bindings.namespaces.builtin_bindings.import_hook.plan.modules[0]
            object.__setattr__(descriptor, 'code', compile('raise AssertionError("replacement")', descriptor.code.co_filename, 'exec'))
            return result
        with self.parent(mutate): self.rejected()

    def test_nested_code_and_sign_bits_compared(self):
        for original, replacement in (('def f():\n return 1\n', 'def f():\n return 2\n'),
                                      ('x = 0.0', 'x = -0.0'), ('x = 1', 'x = True')):
            self.assertNotEqual(initialization.code_signature(compile(original, 'owned', 'exec')),
                                initialization.code_signature(compile(replacement, 'owned', 'exec')))

    def test_executable_metadata_compared(self):
        code = compile('x = 1', 'owned', 'exec')
        for changed in (code.replace(co_filename='changed'), code.replace(co_flags=code.co_flags | 0x1000000)):
            self.assertNotEqual(initialization.code_signature(code), initialization.code_signature(changed))

    def test_unsupported_constant_subclass_never_calls_equality(self):
        class Hostile(str):
            def __eq__(self, other): raise AssertionError('custom equality')
        with self.assertRaises(ValueError): initialization.code_signature(Hostile('value'))

    def test_changed_import_rows_rejected(self):
        def mutate(result):
            plan = result.bindings.namespaces.builtin_bindings.import_hook.plan
            object.__setattr__(plan, 'imports', tuple((name, ()) for name, rows in plan.imports))
            return result
        with self.parent(mutate): self.rejected()

    def test_invalid_dependency_order_rejected(self):
        def mutate(result):
            plan = result.bindings.namespaces.builtin_bindings.import_hook.plan
            object.__setattr__(plan, 'initialization_order', tuple(reversed(plan.initialization_order)))
            return result
        with self.parent(mutate): self.rejected()

    def late(self, mutate):
        read = self.f.custody.read_regular
        def changed(path):
            result = read(path)
            if path == self.helper and hasattr(self, 'captured'): mutate(self.captured)
            return result
        return patch.object(self.f.custody, 'read_regular', changed)

    def test_final_namespace_mutation_rejected(self):
        def mutate(result):
            hook = result.bindings.namespaces.builtin_bindings.import_hook
            hook.guard_modules[hook.plan.initialization_order[0]].__dict__['extra'] = object()
        with self.parent(lambda r: r), self.late(mutate): self.rejected()

    def test_final_private_reference_replacement_rejected(self):
        def mutate(result):
            hook = result.bindings.namespaces.builtin_bindings.import_hook
            hook.package.__dict__[hook.plan.initialization_order[0].rsplit('.', 1)[1]] = object()
        with self.parent(lambda r: r), self.late(mutate): self.rejected()

    def test_final_code_replacement_rejected(self):
        def mutate(result):
            descriptor = result.bindings.namespaces.builtin_bindings.import_hook.plan.modules[0]
            object.__setattr__(descriptor, 'code', compile('x = 1', 'changed', 'exec'))
        with self.parent(lambda r: r), self.late(mutate): self.rejected()

    def test_final_coordinated_descriptor_and_namespace_drift_rejected(self):
        def mutate(result):
            hook = result.bindings.namespaces.builtin_bindings.import_hook
            descriptor = hook.plan.modules[0]
            object.__setattr__(descriptor, 'file', 'changed')
            hook.guard_modules[descriptor.name].__dict__['__file__'] = 'changed'
        with self.parent(lambda r: r), self.late(mutate): self.rejected()

    def test_final_registry_foreign_change_preserved(self):
        foreign = object()
        def mutate(result): self.f.f.registry['unrelated'] = foreign
        with self.parent(lambda r: r), self.late(mutate): self.rejected()
        self.assertIs(self.f.f.registry['unrelated'], foreign)

    def test_final_source_drift_rejected(self):
        def mutate(result):
            p = self.sources.paths['container_startup.py']; p.chmod(0o600); p.write_bytes(b'changed')
        with self.parent(lambda r: r), self.late(mutate): self.rejected()

    def test_final_deadline_remains_typed(self):
        def mutate(result): self.sources.clock.return_value = 700.0
        with self.parent(lambda r: r), self.late(mutate):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()

    def test_ordinary_io_timeout_is_fixed_integrity_error(self):
        read = self.f.custody.read_regular
        def failed(path):
            if path == self.helper: raise TimeoutError('private diagnostic')
            return read(path)
        with patch.object(self.f.custody, 'read_regular', failed): self.rejected()


if __name__ == '__main__': unittest.main(verbosity=2)
