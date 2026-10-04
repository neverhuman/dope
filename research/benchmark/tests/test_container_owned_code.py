"""Compile opaque sources without importing their deliberately failing initializers."""
import builtins
import hashlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from research.benchmark import container_owned_code as compiler
from research.benchmark.tests import test_container_guard_sources as fixture


class OwnedCodeControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.GuardSourceControls()
        self.addCleanup(self.f.doCleanups)
        self.f.setUp()
        self.helper = self.f.base / 'trusted-compiler.py'
        self.helper.write_bytes(b'# opaque already trusted compiler identity\n')
        self.helper.chmod(0o600)
        self.compiler_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(compiler, '__file__', str(self.helper))
        state.start(); self.addCleanup(state.stop)
        self.verify()

    def verify(self):
        return compiler.prepare_startup_code(str(self.f.manifest), self.f.expected,
                                             self.compiler_sha, batch_started_at=100.0)

    def rejected(self):
        with self.assertRaisesRegex(ValueError, '^compressed guard compilation rejected$'):
            self.verify()

    def test_owned_code_objects_never_run_initializers(self):
        result = self.verify()
        self.assertIs(type(result.modules), tuple)
        self.assertEqual({name for name, _ in result.modules}, compiler.sources.SOURCE_NAMES)
        for name, code in result.modules:
            self.assertIs(type(code), types.CodeType)
            self.assertEqual(code.co_filename, '<owned startup guard: ' + name + '>')
            self.assertIn('staged initializer executed', code.co_consts)
        self.assertEqual(dict(result.source_sha256),
                         {name: row['sha256'] for name, row in self.f.lock['source_files'].items()})
        receipt = result.receipt()
        self.assertEqual(receipt['source_files_compiled'], 15)
        self.assertEqual(receipt['outer_seconds_remaining'], 575.0)
        for key in ('source_initializers_run', 'execution_admitted',
                    'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(receipt[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.assertIsNone(receipt[key])

    def test_compiler_digest_rejects_before_source_verification(self):
        self.compiler_sha = '0' * 64
        with patch.object(compiler.sources, 'verify_startup_sources',
                          side_effect=AssertionError('source check reached')):
            self.rejected()

    def test_compiler_fifo_rejects_before_open(self):
        import os
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('compiler FIFO opened')):
            self.rejected()

    def test_source_drift_rejects_before_compilation(self):
        path = self.f.paths['container_startup.py']
        path.chmod(0o600); path.write_bytes(b'changed source')
        with patch.object(builtins, 'compile', side_effect=AssertionError('compile reached')):
            self.rejected()

    def test_invalid_owned_syntax_is_fixed_error(self):
        path = self.f.paths['container_startup.py']
        body = b'private broken syntax !\n'
        path.chmod(0o600); path.write_bytes(body); path.chmod(0o400)
        self.f.lock['source_files'][path.name].update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        self.f.freeze()
        self.rejected()

    def test_compiler_preserves_asserts_and_isolation_flags(self):
        original = builtins.compile; flags = []
        def inspect(body, filename, mode, **kwargs):
            flags.append((mode, kwargs))
            return original(body, filename, mode, **kwargs)
        with patch.object(builtins, 'compile', inspect): self.verify()
        self.assertEqual(flags, [('exec', {'flags': 0, 'dont_inherit': True, 'optimize': 0})] * 15)

    def test_later_source_path_change_never_substitutes_compiled_bytes(self):
        original = builtins.compile; changed = False
        def mutate(body, filename, mode, **kwargs):
            nonlocal changed
            if filename.endswith('container_startup.py>'):
                path = self.f.paths['container_startup.py']
                path.chmod(0o600); path.write_bytes(b'raise RuntimeError("substituted")')
                changed = True
            return original(body, filename, mode, **kwargs)
        with patch.object(builtins, 'compile', mutate): result = self.verify()
        self.assertTrue(changed)
        code = dict(result.modules)['container_startup.py']
        self.assertIn('staged initializer executed', code.co_consts)
        self.assertNotIn('substituted', code.co_consts)

    def test_deadline_before_initial_reads(self):
        self.f.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_deadline_during_source_verification_remains_typed(self):
        original = compiler.sources.read_regular
        def expire(path):
            result = original(path)
            if path == self.f.manifest: self.f.clock.return_value = 700.0
            return result
        with patch.object(compiler.sources, 'read_regular', expire):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_deadline_after_compile_prevents_success(self):
        original = builtins.compile
        def expire(*args, **kwargs):
            result = original(*args, **kwargs); self.f.clock.return_value = 700.0
            return result
        with patch.object(builtins, 'compile', expire):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_final_compiler_alias_rejects_before_open(self):
        original = builtins.compile; altered = False
        twin = self.f.base / 'compiler-twin.py'; twin.write_bytes(self.helper.read_bytes())
        opened = Path.open
        def mutate(*args, **kwargs):
            nonlocal altered
            result = original(*args, **kwargs)
            if not altered:
                self.helper.unlink(); self.helper.symlink_to(twin); altered = True
            return result
        def reject_alias(path, *args, **kwargs):
            if path == self.helper and path.is_symlink(): raise AssertionError('compiler alias opened')
            return opened(path, *args, **kwargs)
        with patch.object(builtins, 'compile', mutate), patch.object(Path, 'open', reject_alias):
            self.rejected()

    def test_deadline_after_final_manifest_read_prevents_success(self):
        original = compiler.sources.read_regular; reads = 0
        def expire(path):
            nonlocal reads
            result = original(path)
            if path == self.f.manifest:
                reads += 1
                if reads == 3: self.f.clock.return_value = 700.0
            return result
        with patch.object(compiler.sources, 'read_regular', expire):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()
        self.assertEqual(reads, 3)

    def test_ordinary_compiler_io_timeout_is_integrity_error(self):
        original = compiler.sources.read_regular
        def broken(path):
            if path == self.helper: raise TimeoutError('private compiler detail')
            return original(path)
        with patch.object(compiler.sources, 'read_regular', broken): self.rejected()


if __name__ == '__main__': unittest.main(verbosity=2)
