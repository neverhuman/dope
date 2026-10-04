"""Opaque import callbacks resolve supplied references without normal import loading."""
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from research.benchmark import container_import_hook as hook
from research.benchmark.tests import test_container_import_declarations as fixture


class ImportHookControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.ImportDeclarationControls()
        self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.helper = self.f.sources.base / 'trusted-hook-helper.py'
        self.helper.write_bytes(b'# opaque already trusted hook identity\n'); self.helper.chmod(0o600)
        self.helper_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(hook, '__file__', str(self.helper)); state.start(); self.addCleanup(state.stop)
        self.stdlib = {name: ModuleType(name) for name in hook.declarations.STDLIB_NAMES}
        self.stdlib['pathlib'].Path = object()
        self.stdlib['dataclasses'].dataclass = object()
        self.prepared = self.verify()
        self.prefix = hook.declarations.binding.PACKAGE + '.'
        self.context = self.prepared.guard_modules[self.prefix + 'container_python_custody'].__dict__

    def verify(self):
        return hook.prepare_import_hook(str(self.f.sources.manifest), self.f.sources.expected,
            self.f.f.f.compiler_sha, self.f.f.helper_sha, self.f.helper_sha,
            self.helper_sha, self.stdlib, batch_started_at=100.0)

    def rejected(self, action):
        with self.assertRaisesRegex(ValueError, '^compressed guard import hook rejected$'): action()

    def test_positive_owned_namespaces_absolute_relative_members_zero_init(self):
        before = dict(sys.modules)
        self.assertIs(self.prepared('hashlib', self.context, fromlist=None), self.stdlib['hashlib'])
        self.assertIs(self.prepared('pathlib', self.context, fromlist=('Path',)), self.stdlib['pathlib'])
        system = self.prepared.guard_modules[self.prefix + 'container_system_custody'].__dict__
        package = self.prepared('', system, fromlist=('container_python_custody',), level=1)
        self.assertIs(package.container_python_custody, self.prepared.guard_modules[self.prefix + 'container_python_custody'])
        self.assertEqual(before, sys.modules)
        self.assertEqual(len(self.prepared.guard_modules), 15)
        self.assertNotIn('__path__', package.__dict__)
        for module in self.prepared.guard_modules.values():
            self.assertNotIn('__builtins__', module.__dict__)
            self.assertIsNone(module.__cached__)
        for key in ('source_initializers_run', 'modules_installed', 'actual_import_resolution_verified',
                    'execution_admitted', 'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(self.prepared.receipt()[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.assertIsNone(self.prepared.receipt()[key])

    def test_source_reference_dictionary_is_owned_before_parent_work(self):
        original = hook.declarations.prepare_import_declarations
        expected = self.stdlib['hashlib']
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs); self.stdlib['hashlib'] = ModuleType('substitute')
            return result
        with patch.object(hook.declarations, 'prepare_import_declarations', mutate): result = self.verify()
        context = result.guard_modules[self.prefix + 'container_python_custody'].__dict__
        self.assertIs(result('hashlib', context, fromlist=None), expected)

    def test_initial_helper_fifo_rejects_before_reads(self):
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('FIFO opened')):
            self.rejected(self.verify)

    def test_forged_digest_rejects_before_parent_work(self):
        class Forged(str):
            def __eq__(self, other): return True
        self.helper_sha = Forged(self.helper_sha)
        with patch.object(hook.declarations, 'prepare_import_declarations',
                          side_effect=AssertionError('parent reached')): self.rejected(self.verify)

    def test_incomplete_or_misnamed_module_references_reject(self):
        self.stdlib.pop('math'); self.rejected(self.verify)
        self.stdlib['math'] = ModuleType('forged'); self.rejected(self.verify)

    def test_module_subclass_rejects_before_custom_access(self):
        class Hostile(ModuleType):
            def __getattribute__(self, name): raise AssertionError('subclass accessed')
        self.stdlib['math'] = Hostile('math'); self.rejected(self.verify)

    def test_per_guard_undeclared_absolute_or_member_rejects(self):
        self.rejected(lambda: self.prepared('math', self.context))
        self.rejected(lambda: self.prepared('pathlib', self.context, fromlist=('PurePosixPath',)))

    def test_missing_preowned_member_never_runs_module_getattr(self):
        self.stdlib['pathlib'].__dict__.pop('Path')
        self.stdlib['pathlib'].__getattr__ = lambda name: (_ for _ in ()).throw(AssertionError('dynamic member getter'))
        self.rejected(lambda: self.prepared('pathlib', self.context, fromlist=('Path',)))

    def test_context_copy_cannot_impersonate_owned_namespace(self):
        self.rejected(lambda: self.prepared('hashlib', dict(self.context), fromlist=None))

    def test_changed_context_metadata_rejects(self):
        self.context['__file__'] = 'changed'
        self.rejected(lambda: self.prepared('hashlib', self.context, fromlist=None))

    def test_request_exact_builtin_types_and_no_wildcards(self):
        class Forged(str):
            def __eq__(self, other): return True
        for action in (lambda: self.prepared(Forged('hashlib'), self.context),
                       lambda: self.prepared('hashlib', self.context, level=0.0),
                       lambda: self.prepared('hashlib', self.context, level=False),
                       lambda: self.prepared('pathlib', self.context, fromlist=['Path']),
                       lambda: self.prepared('pathlib', self.context, fromlist=('*',)),
                       lambda: self.prepared('pathlib', self.context, fromlist=(Forged('Path'),))):
            self.rejected(action)

    def test_relative_requests_use_only_declared_owned_members(self):
        system = self.prepared.guard_modules[self.prefix + 'container_system_custody'].__dict__
        self.rejected(lambda: self.prepared('', system, fromlist=('container_startup',), level=1))
        self.rejected(lambda: self.prepared('container_python_custody', system,
                                          fromlist=('container_python_custody',), level=1))
        self.prepared.package.container_python_custody = ModuleType('substitute')
        self.rejected(lambda: self.prepared('', system, fromlist=('container_python_custody',), level=1))

    def test_original_timer_applies_to_later_import_callbacks(self):
        self.f.sources.clock.return_value = 700.0
        with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            self.prepared('hashlib', self.context, fromlist=None)

    def test_original_timer_applies_before_preparation_reads(self):
        self.f.sources.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_late_helper_alias_rejects_before_open(self):
        original = hook.declarations.prepare_import_declarations
        twin = self.f.sources.base / 'hook-twin.py'; twin.write_bytes(self.helper.read_bytes())
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs); self.helper.unlink(); self.helper.symlink_to(twin)
            return result
        opened = Path.open
        def refuse(p, *args, **kwargs):
            if p == self.helper and p.is_symlink(): raise AssertionError('hook alias opened')
            return opened(p, *args, **kwargs)
        with patch.object(hook.declarations, 'prepare_import_declarations', mutate), patch.object(Path, 'open', refuse):
            self.rejected(self.verify)

    def test_ordinary_io_timeout_is_sanitized(self):
        original = hook.declarations.binding.compiler.sources.read_regular
        def broken(p):
            if p == self.helper: raise TimeoutError('private hook detail')
            return original(p)
        with patch.object(hook.declarations.binding.compiler.sources, 'read_regular', broken):
            self.rejected(self.verify)


if __name__ == '__main__': unittest.main(verbosity=2)
