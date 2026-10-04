"""Plan opaque import declarations without imports, initializers or namespaces."""
import hashlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from research.benchmark import container_import_declarations as declarations
from research.benchmark.tests import test_container_module_bindings as fixture


class ImportDeclarationControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.ModuleBindingControls()
        self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.sources = self.f.f.f
        self.helper = self.sources.base / 'trusted-import-helper.py'
        self.helper.write_bytes(b'# opaque already trusted import planning identity\n')
        self.helper.chmod(0o600)
        self.helper_sha = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        state = patch.object(declarations, '__file__', str(self.helper))
        state.start(); self.addCleanup(state.stop)
        self.replace('container_python_custody.py', b'import hashlib as digest\nfrom pathlib import Path as BoundPath\n')
        self.replace('container_system_custody.py', b'from . import container_python_custody as python\n')
        self.replace('container_startup.py', b'from . import container_system_custody as system\nfrom dataclasses import dataclass\n')
        self.verify()

    def replace(self, name, body):
        body += b'raise AssertionError("staged initializer executed")\n'
        p = self.sources.paths[name]; p.chmod(0o600); p.write_bytes(body); p.chmod(0o400)
        self.sources.lock['source_files'][name].update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        self.sources.freeze()

    def verify(self):
        return declarations.prepare_import_declarations(str(self.sources.manifest), self.sources.expected,
            self.f.f.compiler_sha, self.f.helper_sha, self.helper_sha, batch_started_at=100.0)

    def rejected(self):
        with self.assertRaisesRegex(ValueError, '^compressed guard import declarations rejected$'):
            self.verify()

    def test_positive_declarations_aliases_order_and_no_initialization(self):
        before = dict(sys.modules); result = self.verify(); self.assertEqual(before, sys.modules)
        prefix = declarations.binding.PACKAGE + '.'
        rows = dict(result.imports)
        self.assertEqual(rows[prefix + 'container_python_custody'],
                         (('absolute', 'hashlib', None, 'digest'), ('absolute', 'pathlib', 'Path', 'BoundPath')))
        self.assertEqual(rows[prefix + 'container_system_custody'],
                         (('relative', prefix + 'container_python_custody', None, 'python'),))
        order = result.initialization_order
        self.assertEqual(set(order), {module.name for module in result.modules})
        self.assertLess(order.index(prefix + 'container_python_custody'), order.index(prefix + 'container_system_custody'))
        self.assertLess(order.index(prefix + 'container_system_custody'), order.index(prefix + 'container_startup'))
        self.assertEqual(result.receipt()['declared_imports'], 5)
        self.assertEqual(result.outer_seconds_remaining, 575.0)
        for key in ('actual_import_resolution_verified', 'source_initializers_run', 'modules_installed',
                    'execution_admitted', 'full_runtime_closure_certified', 'official_tests_opened'):
            self.assertIs(result.receipt()[key], False)
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.assertIsNone(result.receipt()[key])

    def test_exact_builtin_digest_before_binding(self):
        class Forged(str):
            def __eq__(self, other): return True
        self.helper_sha = Forged(self.helper_sha)
        with patch.object(declarations.binding, 'prepare_module_bindings',
                          side_effect=AssertionError('binding reached')): self.rejected()

    def test_helper_fifo_before_open(self):
        self.helper.unlink(); os.mkfifo(self.helper, 0o600)
        with patch.object(Path, 'open', side_effect=AssertionError('FIFO opened')): self.rejected()

    def test_undeclared_absolute_dependency_rejects_without_import(self):
        self.replace('container_startup.py', b'import torch\n'); self.rejected()
        self.assertNotIn('torch', sys.modules)

    def test_unknown_relative_dependency_rejects(self):
        self.replace('container_startup.py', b'from . import unknown_guard\n'); self.rejected()

    def test_parent_relative_level_rejects(self):
        self.replace('container_startup.py', b'from .. import container_system_custody\n'); self.rejected()

    def test_named_relative_module_rejects(self):
        self.replace('container_startup.py', b'from .container_system_custody import require\n'); self.rejected()

    def test_wildcard_standard_import_rejects(self):
        self.replace('container_startup.py', b'from pathlib import *\n'); self.rejected()

    def test_cycle_rejects_before_initialization(self):
        self.replace('container_python_custody.py', b'from . import container_startup\n'); self.rejected()

    def test_all_nested_static_declarations_are_visible(self):
        self.replace('container_startup.py', b'def later():\n    import math as arithmetic\n')
        result = self.verify(); rows = dict(result.imports)
        self.assertEqual(rows[declarations.binding.PACKAGE + '.container_startup'],
                         (('absolute', 'math', None, 'arithmetic'),))

    def after_parse(self, action):
        original = declarations.ast.parse; changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed: changed = True; action()
            return result
        return patch.object(declarations.ast, 'parse', mutate)

    def test_source_drift_after_parse_rejects(self):
        def mutate():
            p = self.sources.paths['container_startup.py']; p.chmod(0o600); p.write_bytes(b'changed')
        with self.after_parse(mutate): self.rejected()

    def test_binding_helper_late_alias_before_read(self):
        twin = self.sources.base / 'binding-twin.py'; twin.write_bytes(self.f.helper.read_bytes())
        def mutate(): self.f.helper.unlink(); self.f.helper.symlink_to(twin)
        original = Path.open
        def refuse(p, *args, **kwargs):
            if p == self.f.helper and p.is_symlink(): raise AssertionError('late binding alias opened')
            return original(p, *args, **kwargs)
        with self.after_parse(mutate), patch.object(Path, 'open', refuse): self.rejected()

    def test_original_deadline_before_first_read(self):
        self.sources.clock.return_value = 700.0
        with patch.object(Path, 'open', side_effect=AssertionError('expired read')):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_deadline_during_parsing_prevents_success(self):
        with self.after_parse(lambda: setattr(self.sources.clock, 'return_value', 700.0)):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_deadline_after_final_manifest_read_prevents_success(self):
        original = declarations.binding.compiler.sources.read_regular; finished = False
        def expire(p):
            nonlocal finished
            result = original(p)
            if p == self.helper and not finished: finished = True
            elif p == self.helper: self.sources.clock.return_value = 699.0
            elif p == self.sources.manifest and self.sources.clock.return_value == 699.0:
                self.sources.clock.return_value = 700.0
            return result
        with patch.object(declarations.binding.compiler.sources, 'read_regular', expire):
            with self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
                self.verify()

    def test_ordinary_io_timeout_is_fixed_integrity_rejection(self):
        original = declarations.binding.compiler.sources.read_regular
        def broken(p):
            if p == self.helper: raise TimeoutError('private import planning detail')
            return original(p)
        with patch.object(declarations.binding.compiler.sources, 'read_regular', broken): self.rejected()


if __name__ == '__main__': unittest.main(verbosity=2)
