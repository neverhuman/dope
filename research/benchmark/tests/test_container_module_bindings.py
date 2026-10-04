"""Opaque module descriptors never execute guard initializers or install imports."""
from dataclasses import FrozenInstanceError
import hashlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from research.benchmark import container_module_bindings as binding
from research.benchmark.tests import test_container_owned_code as fixture


class ModuleBindingControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.OwnedCodeControls()
        self.addCleanup(self.f.doCleanups)
        self.f.setUp()
        self.helper = self.f.f.base / 'trusted-binding-helper.py'
        self.helper.write_bytes(b'# opaque already trusted binding identity\n')
        self.helper.chmod(0o600)
        self.helper_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(binding, '__file__', str(self.helper))
        state.start(); self.addCleanup(state.stop)
        self.verify()

    def verify(self):
        return binding.prepare_module_bindings(str(self.f.f.manifest), self.f.f.expected,
            self.f.compiler_sha, self.helper_sha, batch_started_at=100.0)

    def rejected(self):
        with self.assertRaisesRegex(ValueError, '^compressed guard module binding rejected$'):
            self.verify()

    def test_owned_code_paths_names_and_immutable_descriptors(self):
        before = dict(sys.modules)
        result = self.verify()
        self.assertEqual(before, sys.modules)
        self.assertEqual(len(result.modules), 15)
        for module in result.modules:
            name = module.name.removeprefix(binding.PACKAGE + '.') + '.py'
            self.assertEqual(module.package, binding.PACKAGE)
            self.assertEqual(module.file, str(self.f.f.paths[name]))
            self.assertEqual(module.source_sha256, self.f.f.lock['source_files'][name]['sha256'])
            self.assertEqual(module.code.co_filename, '<owned startup guard: ' + name + '>')
            self.assertIn('staged initializer executed', module.code.co_consts)
            with self.assertRaises(FrozenInstanceError): module.file = 'substituted'
        receipt = result.receipt()
        self.assertEqual(receipt['outer_seconds_remaining'], 575.0)
        for key in ('source_initializers_run', 'modules_installed', 'execution_admitted',
                    'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(receipt[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.assertIsNone(receipt[key])

    def test_binding_digest_rejects_before_compiler(self):
        class Forged(str):
            def __eq__(self, other): return True
        self.helper_sha = Forged(self.helper_sha)
        with patch.object(binding.compiler, 'prepare_startup_code',
                          side_effect=AssertionError('compiler reached')): self.rejected()

    def test_binding_fifo_rejects_before_open(self):
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('FIFO opened')): self.rejected()

    def test_source_drift_after_compilation_rejects_binding(self):
        original = binding.compiler.prepare_startup_code
        def mutate(*args, **kwargs):
            compiled = original(*args, **kwargs)
            path = self.f.f.paths['container_startup.py']
            path.chmod(0o600); path.write_bytes(b'raise RuntimeError("substitute")\n')
            return compiled
        with patch.object(binding.compiler, 'prepare_startup_code', mutate): self.rejected()

    def construct_and_mutate(self, action):
        original = binding.ModuleBinding; changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed: changed = True; action()
            return result
        return patch.object(binding, 'ModuleBinding', mutate)

    def test_late_source_alias_rejects_before_open(self):
        path = self.f.f.paths['container_startup.py']
        twin = self.f.f.base / 'source-twin.py'; twin.write_bytes(path.read_bytes())
        def mutate(): path.unlink(); path.symlink_to(twin)
        original = Path.open
        def refuse(pathname, *args, **kwargs):
            if pathname == path and path.is_symlink(): raise AssertionError('source alias opened')
            return original(pathname, *args, **kwargs)
        with self.construct_and_mutate(mutate), patch.object(Path, 'open', refuse): self.rejected()

    def test_late_extra_source_rejects_complete_inventory(self):
        def mutate(): (self.f.f.paths['container_startup.py'].parent / 'extra.py').write_bytes(b'opaque')
        with self.construct_and_mutate(mutate): self.rejected()

    def test_late_shared_write_mode_rejects_even_identical_bytes(self):
        def mutate(): self.f.f.paths['container_startup.py'].chmod(0o660)
        with self.construct_and_mutate(mutate): self.rejected()

    def test_late_binding_helper_alias_rejects_before_open(self):
        twin = self.f.f.base / 'binding-twin.py'; twin.write_bytes(self.helper.read_bytes())
        def mutate(): self.helper.unlink(); self.helper.symlink_to(twin)
        original = Path.open
        def refuse(path, *args, **kwargs):
            if path == self.helper and path.is_symlink(): raise AssertionError('binding alias opened')
            return original(path, *args, **kwargs)
        with self.construct_and_mutate(mutate), patch.object(Path, 'open', refuse): self.rejected()

    def test_late_compiler_helper_alias_rejects_before_open(self):
        twin = self.f.f.base / 'compiler-twin.py'; twin.write_bytes(self.f.helper.read_bytes())
        def mutate(): self.f.helper.unlink(); self.f.helper.symlink_to(twin)
        original = Path.open
        def refuse(path, *args, **kwargs):
            if path == self.f.helper and path.is_symlink(): raise AssertionError('compiler alias opened')
            return original(path, *args, **kwargs)
        with self.construct_and_mutate(mutate), patch.object(Path, 'open', refuse): self.rejected()

    def test_original_deadline_before_first_read(self):
        self.f.f.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_deadline_during_final_source_pass_prevents_success(self):
        def expire(): self.f.f.clock.return_value = 700.0
        with self.construct_and_mutate(expire):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_deadline_after_final_binding_helper_read_prevents_success(self):
        original = binding.compiler.sources.read_regular; reads = 0
        def expire(path):
            nonlocal reads
            result = original(path)
            if path == self.helper:
                reads += 1
                if reads == 2: self.f.f.clock.return_value = 700.0
            return result
        with patch.object(binding.compiler.sources, 'read_regular', expire):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()
        self.assertEqual(reads, 2)

    def test_ordinary_helper_io_timeout_is_integrity_rejection(self):
        original = binding.compiler.sources.read_regular
        def broken(path):
            if path == self.helper: raise TimeoutError('private path detail')
            return original(path)
        with patch.object(binding.compiler.sources, 'read_regular', broken): self.rejected()


if __name__ == '__main__': unittest.main(verbosity=2)
