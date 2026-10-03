"""Opaque pre-interpreter inventory controls; no candidate code is executed."""
from contextlib import ExitStack
import builtins
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import container_python_custody as custody


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


class ImportInputControls(unittest.TestCase):
    def setUp(self):
        target = Path('target').resolve()
        target.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='container-python-control-', dir=target)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'scratch'
        self.base.mkdir()
        self.python = self.root / 'usr/bin/python3.12'
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b'opaque interpreter, never executable')
        self.python_alias = self.python.parent / 'python3'
        self.python_alias.symlink_to('python3.12')
        self.library = self.root / 'usr/lib/python3.12'
        (self.library / '__pycache__').mkdir(parents=True)
        (self.library / 'lib-dynload').mkdir()
        (self.library / 'opaque.py').write_bytes(b'opaque source')
        (self.library / '__pycache__/opaque.pyc').write_bytes(b'opaque bytecode')
        (self.library / 'lib-dynload/opaque.so').write_bytes(b'opaque extension')
        self.stdlib_alias = self.library / 'opaque-alias.py'
        self.stdlib_alias.symlink_to('opaque.py')
        self.site = self.base / 'site'
        (self.site / 'numpy').mkdir(parents=True)
        self.marker = self.root / 'initializer-executed'
        (self.site / 'numpy/__init__.py').write_text(
            f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('executed')\n")
        self.aux_site = self.base / 'aux-site'
        self.aux_site.mkdir()
        (self.aux_site / 'opaque.py').write_bytes(b'opaque auxiliary source')
        self.cache = self.base / 'empty-cache'
        self.cache.mkdir()
        self.metric = self.base / 'metric.py'
        self.metric.write_bytes(b'opaque metric')
        self.runtime_file = self.base / 'runtime.lock.json'
        self.auxiliary_file = self.base / 'auxiliary.lock.json'
        self.zip = self.library.parent / 'python312.zip'
        self.runtime = {'python': str(self.python), 'python_sha256': sha(self.python),
                        'python_symlink': 'python3.12', 'absent_zip': str(self.zip),
                        'site': str(self.site), 'files': {'numpy/__init__.py': identity(self.site / 'numpy/__init__.py')},
                        'empty_bytecode_cache': str(self.cache), 'metric_source': str(self.metric),
                        'metric_sha256': sha(self.metric)}
        self.auxiliary = {'runtime_path': str(self.runtime_file), 'site': str(self.aux_site),
                          'files': {'opaque.py': identity(self.aux_site / 'opaque.py')},
                          'stdlib_all_files': {str(p): identity(p) for p in self.library.rglob('*') if p.is_file()},
                          'stdlib_aliases': {str(self.stdlib_alias): {'target': 'opaque.py',
                              'resolved': str(self.library / 'opaque.py'), 'sha256': sha(self.stdlib_alias)}}}
        self.freeze()
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in [('BASE', self.base), ('PYTHON', self.python), ('PYTHON_ALIAS', self.python_alias),
                            ('STDLIB', self.library), ('ABSENT_ZIP', self.zip)]:
            stack.enter_context(patch.object(custody, name, value))

    def freeze(self):
        self.runtime_file.write_text(json.dumps(self.runtime, sort_keys=True))
        self.runtime_sha = sha(self.runtime_file)
        self.auxiliary['runtime_sha256'] = self.runtime_sha
        self.auxiliary_file.write_text(json.dumps(self.auxiliary, sort_keys=True))
        self.auxiliary_sha = sha(self.auxiliary_file)

    def verify(self, **overrides):
        args = {'runtime_path': str(self.runtime_file), 'runtime_sha256': self.runtime_sha,
                'auxiliary_path': str(self.auxiliary_file), 'auxiliary_sha256': self.auxiliary_sha}
        args.update(overrides)
        original_import = builtins.__import__
        def checked_import(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'catboost', 'torch'}:
                self.fail('candidate dependency initialized')
            return original_import(name, *args, **kwargs)
        with patch('subprocess.run', side_effect=AssertionError('candidate process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('candidate process started')), \
                patch('builtins.__import__', side_effect=checked_import):
            return custody.verify_import_inputs(**args)

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed Python custody rejected$'):
            self.verify(**overrides)
        self.assertFalse(self.marker.exists())

    def test_complete_inputs_prove_only_file_custody(self):
        result = self.verify()
        self.assertTrue(result['python_import_inputs_verified'])
        self.assertFalse(result['execution_admitted'])
        self.assertFalse(result['full_runtime_closure_certified'])
        self.assertEqual(result['processes_started'], 0)
        self.assertEqual(result['package_files'], 2)
        self.assertEqual(result['stdlib_files'], 4)
        self.assertFalse(result['official_tests_opened'])
        for key in ['mfs_v2', 'ptf_v1', 'release_safe', 'superiority']:
            self.assertIsNone(result[key])
        self.assertFalse(self.marker.exists())

    def test_manifest_drift_rejected_before_parsing(self):
        self.runtime_file.write_bytes(b'changed private manifest content')
        with patch.object(custody.json, 'loads', side_effect=AssertionError('manifest parsed before hash')):
            self.rejected()

    def test_digest_subclasses_and_bad_digests_rejected(self):
        class Permissive(str):
            def __eq__(self, other):
                return True
        for value in [Permissive(self.runtime_sha), self.runtime_sha.upper(), None, 1, 'invalid']:
            with self.subTest(value_type=type(value).__name__):
                self.rejected(runtime_sha256=value)

    def test_interpreter_drift_rejected(self):
        self.python.write_bytes(b'changed interpreter')
        self.rejected()

    def test_interpreter_alias_drift_rejected(self):
        other = self.python.parent / 'other'
        other.write_bytes(self.python.read_bytes())
        self.python_alias.unlink()
        self.python_alias.symlink_to('other')
        self.rejected()

    def test_added_zip_and_broken_zip_alias_rejected(self):
        self.zip.write_bytes(b'uninventoried import root')
        self.rejected()
        self.zip.unlink()
        self.zip.symlink_to('missing')
        self.rejected()

    def test_bytecode_drift_rejected(self):
        (self.library / '__pycache__/opaque.pyc').write_bytes(b'changed executable bytecode')
        self.rejected()

    def test_added_stdlib_bytecode_rejected(self):
        (self.library / '__pycache__/extra.pyc').write_bytes(b'added executable bytecode')
        self.rejected()

    def test_added_directory_alias_in_each_import_root_rejected(self):
        for root in [self.site, self.aux_site, self.library]:
            with self.subTest(root=root.name):
                alias = root / 'added-directory-alias'
                alias.symlink_to(root, target_is_directory=True)
                self.rejected()
                alias.unlink()

    def test_declared_stdlib_alias_must_keep_target_and_bytes(self):
        other = self.library / 'other.py'
        other.write_bytes((self.library / 'opaque.py').read_bytes())
        self.auxiliary['stdlib_all_files'][str(other)] = identity(other)
        self.freeze()
        self.stdlib_alias.unlink()
        self.stdlib_alias.symlink_to('other.py')
        self.rejected()

    def test_package_drift_rejected(self):
        (self.site / 'numpy/__init__.py').write_bytes(b'changed candidate initializer')
        self.rejected()

    def test_cache_and_metric_drift_rejected(self):
        cache_file = self.cache / 'extra.pyc'
        cache_file.write_bytes(b'added cache')
        self.rejected()
        cache_file.unlink()
        self.metric.write_bytes(b'changed metric')
        self.rejected()

    def test_auxiliary_binding_cannot_select_other_runtime(self):
        self.auxiliary['runtime_path'] = str(self.metric)
        self.freeze()
        self.rejected()

    def test_import_root_escape_rejected(self):
        self.runtime['site'] = str(self.root)
        self.freeze()
        self.rejected()

    def test_final_manifest_rehash_rejects_concurrent_change(self):
        original_sha = custody.sha
        def changing_sha(path):
            value = original_sha(path)
            if path == self.metric:
                self.runtime_file.write_bytes(b'changed during inventory check')
            return value
        with patch.object(custody, 'sha', side_effect=changing_sha):
            self.rejected()

    def reject_manifest_swap(self, auxiliary):
        manifest = self.auxiliary if auxiliary else self.runtime
        path = self.auxiliary_file if auxiliary else self.runtime_file
        target = self.aux_site / 'opaque.py' if auxiliary else self.site / 'numpy/__init__.py'
        name = 'opaque.py' if auxiliary else 'numpy/__init__.py'
        frozen_bytes, expected = path.read_bytes(), sha(path)
        target.write_bytes(b'opaque changed initializer, never executed')
        alternate = dict(manifest, files={name: identity(target)})
        alternate_bytes = json.dumps(alternate, sort_keys=True).encode()
        metric_sha = sha(self.metric)
        original_hash = hashlib.sha256
        events = []
        class SwapAfterHash:
            def __init__(self, data=b''):
                self.inner = original_hash(data)
            def update(self, data):
                self.inner.update(data)
            def hexdigest(self):
                result = self.inner.hexdigest()
                if result == expected and not events:
                    path.write_bytes(alternate_bytes)
                    events.append('replaced_after_hash')
                elif result == metric_sha and events:
                    path.write_bytes(frozen_bytes)
                    events.append('restored_before_final_hash')
                return result
        with patch.object(custody.hashlib, 'sha256', side_effect=SwapAfterHash):
            self.rejected()
        self.assertIn('replaced_after_hash', events)

    def test_runtime_manifest_hash_parse_swap_rejected(self):
        self.reject_manifest_swap(auxiliary=False)

    def test_auxiliary_manifest_hash_parse_swap_rejected(self):
        self.reject_manifest_swap(auxiliary=True)


if __name__ == '__main__':
    unittest.main()
