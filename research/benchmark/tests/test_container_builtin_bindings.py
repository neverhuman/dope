"""Opaque builtin selections prepare bindings without executing startup guards."""
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from research.benchmark import container_builtin_bindings as bindings
from research.benchmark.tests import test_container_import_hook as fixture


class BuiltinBindingControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.ImportHookControls()
        self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.sources = self.f.f.sources
        self.custody = bindings.imports.declarations.binding.compiler.sources
        self.helper = self.sources.base / 'trusted-builtin-helper.py'
        self.helper.write_bytes(b'# opaque already trusted builtin helper\n'); self.helper.chmod(0o600)
        self.helper_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(bindings, '__file__', str(self.helper)); state.start(); self.addCleanup(state.stop)
        self.builtins = ModuleType('builtins')
        self.builtins.__dict__.update({name: object() for name in bindings.BUILTIN_NAMES})
        self.builtins.__import__ = object()
        self.prepared = self.verify()

    def verify(self):
        return bindings.prepare_builtin_bindings(str(self.sources.manifest), self.sources.expected,
            self.f.f.f.f.compiler_sha, self.f.f.f.helper_sha, self.f.f.helper_sha,
            self.f.helper_sha, self.helper_sha, self.f.stdlib, self.builtins,
            batch_started_at=100.0)

    def rejected(self, action):
        with self.assertRaisesRegex(ValueError, '^compressed guard builtin bindings rejected$'): action()

    def test_positive_fixed_references_hook_order_zero_installs(self):
        before = dict(sys.modules)
        result = self.verify()
        self.assertEqual(len(result.builtin_references), 33)
        self.assertIs(result.builtin_references['__import__'], result.import_hook)
        self.assertIsNot(result.builtin_references['__import__'], self.builtins.__import__)
        self.assertEqual(tuple(name for name, _ in result.namespace_bindings), result.import_hook.plan.initialization_order)
        for name, refs in result.namespace_bindings:
            self.assertIs(refs, result.builtin_references)
            self.assertNotIn('__builtins__', result.import_hook.guard_modules[name].__dict__)
        for name in ('open', 'exec', 'eval', 'compile', 'getattr', '__loader__'):
            self.assertNotIn(name, result.builtin_references)
        with self.assertRaises(TypeError): result.builtin_references['open'] = object()
        self.assertEqual(before, sys.modules)
        for key in ('builtins_installed', 'source_initializers_run', 'modules_installed',
                    'actual_import_resolution_verified', 'execution_admitted',
                    'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(result.receipt()[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.assertIsNone(result.receipt()[key])

    def test_builtin_reference_selection_copied_before_parent_work(self):
        original = bindings.imports.prepare_import_hook
        expected = self.builtins.len
        def mutate(*args, **kwargs):
            self.builtins.len = object(); return original(*args, **kwargs)
        with patch.object(bindings.imports, 'prepare_import_hook', mutate): result = self.verify()
        self.assertIs(result.builtin_references['len'], expected)

    def test_module_subclass_rejects_before_custom_access(self):
        class Hostile(ModuleType):
            def __getattribute__(self, name): raise AssertionError('subclass accessed')
        self.builtins = Hostile('builtins'); self.rejected(self.verify)

    def test_name_metadata_requires_exact_builtin_string(self):
        class Forged(str):
            def __eq__(self, other): return True
        self.builtins.__name__ = Forged('builtins'); self.rejected(self.verify)

    def test_missing_member_never_calls_dynamic_module_getter(self):
        self.builtins.__dict__.pop('len')
        self.builtins.__getattr__ = lambda name: (_ for _ in ()).throw(AssertionError('dynamic getter'))
        self.rejected(self.verify)

    def test_initial_helper_fifo_rejects_before_reads(self):
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('FIFO opened')): self.rejected(self.verify)

    def test_initial_helper_alias_rejects_before_reads(self):
        twin = self.sources.base / 'builtin-twin.py'; twin.write_bytes(self.helper.read_bytes())
        self.helper.unlink(); self.helper.symlink_to(twin)
        with patch.object(Path, 'open', side_effect=AssertionError('alias opened')): self.rejected(self.verify)

    def test_group_writable_helper_rejects_before_read(self):
        self.helper.chmod(0o660)
        with patch.object(Path, 'open', side_effect=AssertionError('writable helper opened')):
            self.rejected(self.verify)

    def test_forged_helper_digest_rejects_before_parent(self):
        class Forged(str):
            def __eq__(self, other): return True
        self.helper_sha = Forged(self.helper_sha)
        with patch.object(bindings.imports, 'prepare_import_hook', side_effect=AssertionError('parent reached')):
            self.rejected(self.verify)

    def test_late_helper_alias_rejects_before_open(self):
        original = bindings.imports.prepare_import_hook
        twin = self.sources.base / 'builtin-twin.py'; twin.write_bytes(self.helper.read_bytes())
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs); self.helper.unlink(); self.helper.symlink_to(twin); return result
        opened = Path.open
        def refuse(path, *args, **kwargs):
            if path == self.helper and path.is_symlink(): raise AssertionError('late alias opened')
            return opened(path, *args, **kwargs)
        with patch.object(bindings.imports, 'prepare_import_hook', mutate), patch.object(Path, 'open', refuse):
            self.rejected(self.verify)

    def test_late_helper_bytes_reject(self):
        original = bindings.imports.prepare_import_hook
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs); self.helper.write_bytes(b'changed'); return result
        with patch.object(bindings.imports, 'prepare_import_hook', mutate): self.rejected(self.verify)

    def test_late_manifest_bytes_reject(self):
        original = bindings.imports.prepare_import_hook
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs); self.sources.manifest.write_bytes(b'changed'); return result
        with patch.object(bindings.imports, 'prepare_import_hook', mutate): self.rejected(self.verify)

    def test_original_deadline_before_first_read(self):
        self.sources.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()

    def test_original_deadline_after_final_rehash(self):
        original = bindings.imports.prepare_import_hook
        entered = False
        def mark(*args, **kwargs):
            nonlocal entered
            result = original(*args, **kwargs); entered = True; return result
        read = self.custody.read_regular
        def late(path):
            result = read(path)
            if entered and path == self.sources.manifest: self.sources.clock.return_value = 700.0
            return result
        with patch.object(bindings.imports, 'prepare_import_hook', mark), patch.object(self.custody, 'read_regular', late):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'): self.verify()

    def test_ordinary_io_timeout_is_fixed_integrity_error(self):
        read = self.custody.read_regular
        def broken(path):
            if path == self.helper: raise TimeoutError('private builtin detail')
            return read(path)
        with patch.object(self.custody, 'read_regular', broken): self.rejected(self.verify)


if __name__ == '__main__': unittest.main(verbosity=2)
