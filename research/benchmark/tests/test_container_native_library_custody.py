"""Original library-directory custody uses opaque files, not runtime loading."""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import container_native_library_custody as library
from research.benchmark.tests import test_container_native_sampler_custody as fixture


class NativeLibraryControls(unittest.TestCase):
    def setUp(self):
        self.native = fixture.NativeSamplerControls()
        self.addCleanup(self.native.doCleanups)
        self.native.setUp()
        self.base = self.native.base
        self.directory = self.base / 'original-library'; self.directory.mkdir()
        (self.directory / 'liba.so').write_bytes(b'opaque original library bytes')
        (self.directory / 'alias.so').symlink_to('./liba.so')
        self.validation_path = self.base / 'validation.json'
        self.runtime_path = self.base / 'runtime.json'
        self.runtime = {'host': 'xbabe2', 'binary': str(self.native.binary),
            'binary_sha256': fixture.sha(self.native.binary), 'torch_library_directory': str(self.directory),
            'root_inventory_files': {str(self.directory): sorted(str(p) for p in self.directory.iterdir())},
            'files': {str(p): {'resolved_path': str(p.resolve()), 'bytes': p.stat().st_size,
                             'sha256': fixture.sha(p)} for p in self.directory.iterdir()}}
        self.validation = {'native_library_directory': str(self.directory),
            'gpu_binary': str(self.native.binary), 'gpu_binary_sha256': fixture.sha(self.native.binary)}
        self.manifest = self.base / 'library.json'
        self.lock = {'format': 'dope-native-library-directory-preparation', 'version': 1,
            'sampler_path': str(self.native.manifest), 'sampler_sha256': self.native.digest,
            'validation_path': str(self.validation_path), 'library_directory': str(self.directory),
            'directory_snapshot': library.loader.directory_identity(str(self.directory)),
            'files': {str(p): {'identity': library.loader.path_identity(str(p)),
                'bytes': p.stat().st_size, 'sha256': fixture.sha(p)} for p in self.directory.iterdir()},
            'loader_helper_sha256': fixture.sha(Path(library.loader.__file__)),
            'sampler_helper_sha256': fixture.sha(Path(library.sampler.__file__)),
            'library_helper_sha256': fixture.sha(Path(library.__file__)), 'candidate_processes_started': 0}
        for key in ('actual_loader_selection_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            self.lock[key] = False
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.lock[key] = None
        self.freeze_parents(); self.freeze(); self.verify()

    def freeze_parents(self):
        self.runtime_path.write_text(json.dumps(self.runtime, sort_keys=True))
        self.validation['host_runtimes'] = {'xbabe2': {'path': str(self.runtime_path),
                                                     'sha256': fixture.sha(self.runtime_path)}}
        self.validation_path.write_text(json.dumps(self.validation, sort_keys=True))
        self.lock['validation_sha256'] = fixture.sha(self.validation_path)

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock, sort_keys=True)); self.digest = fixture.sha(self.manifest)

    def verify(self, **overrides):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'torch', 'scipy', 'sklearn', 'pandas', 'sdv'}:
                self.fail('candidate dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('candidate process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('candidate process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('candidate library loaded')), \
                patch.object(library.sampler.bootstrap.time, 'monotonic', return_value=125.0):
            return library.verify_native_library_directory(**dict({'library_path': str(self.manifest),
                'library_sha256': self.digest, 'batch_started_at': 100.0}, **overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed native library custody rejected$'):
            self.verify(**overrides)

    def test_owned_original_directory_has_no_runtime_or_execution_claim(self):
        result = self.verify()
        self.assertEqual(result['selected_directory_files_verified'], 2)
        self.assertIs(result['literal_alias_snapshots_verified'], True)
        self.assertEqual(result['outer_seconds_remaining'], 575.0)
        for key in ('actual_loader_selection_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            self.assertIs(result[key], False)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_outer_and_helper_digests_reject_before_directory_walk(self):
        class Digest(str): pass
        with patch.object(library.loader, 'directory_identity', side_effect=AssertionError('unowned walk')):
            self.rejected(library_sha256='0' * 64); self.rejected(library_sha256=Digest(self.digest))
            for key in ('loader_helper_sha256', 'sampler_helper_sha256', 'library_helper_sha256'):
                old = self.lock[key]; self.lock[key] = '0' * 64; self.freeze(); self.rejected(); self.lock[key] = old

    def test_historical_host_and_binary_binding(self):
        old = copy.deepcopy(self.runtime)
        for key, value in [('host', 'xbabe3'), ('binary', str(self.base / 'other.bin')),
                           ('binary_sha256', '0' * 64), ('torch_library_directory', str(self.base))]:
            self.runtime = copy.deepcopy(old); self.runtime[key] = value
            self.freeze_parents(); self.freeze(); self.rejected()

    def test_directory_original_validation_binding(self):
        self.validation['native_library_directory'] = str(self.base)
        self.freeze_parents(); self.freeze(); self.rejected()

    def test_inventory_membership_and_duplicates(self):
        paths = self.runtime['root_inventory_files'][str(self.directory)]
        for value in (paths[:-1], paths + paths[:1], ['../opaque'], {}):
            self.runtime['root_inventory_files'][str(self.directory)] = value
            self.freeze_parents(); self.freeze(); self.rejected()

    def test_added_and_missing_directory_nodes(self):
        for kind in ('file', 'directory', 'alias'):
            path = self.directory / 'added'
            if kind == 'file': path.write_bytes(b'opaque added')
            elif kind == 'directory': path.mkdir()
            else: path.symlink_to('liba.so')
            self.rejected()
            if kind == 'directory': path.rmdir()
            else: path.unlink()
            self.verify()
        (self.directory / 'alias.so').unlink(); self.rejected()

    def test_literal_alias_target_change_same_contents_rejects(self):
        alias = self.directory / 'alias.so'; alias.unlink(); alias.symlink_to('liba.so')
        self.assertEqual(fixture.sha(alias), self.lock['files'][str(alias)]['sha256'])
        self.rejected()

    def test_file_size_digest_and_original_reference_changes(self):
        path = str(self.directory / 'liba.so'); old = copy.deepcopy(self.lock['files'][path])
        for key, value in [('bytes', True), ('bytes', float(old['bytes'])), ('sha256', '0' * 64)]:
            self.lock['files'][path] = copy.deepcopy(old); self.lock['files'][path][key] = value
            self.freeze(); self.rejected()
        self.lock['files'][path] = old; self.freeze()
        self.runtime['files'][path]['resolved_path'] = str(self.base / 'other.so')
        self.freeze_parents(); self.freeze(); self.rejected()

    def test_nonregular_file_rejected_before_hash(self):
        path = self.directory / 'liba.so'; path.unlink(); os.mkfifo(path)
        original = library.sampler.system.python.sha
        def guarded(p):
            if p == path: self.fail('nonregular library hashed')
            return original(p)
        with patch.object(library.sampler.system.python, 'sha', side_effect=guarded): self.rejected()

    def test_final_directory_and_file_changes_reject(self):
        original = library.sampler.system.python.sha
        path = self.directory / 'liba.so'; blob = path.read_bytes(); calls = []
        def changed(p):
            result = original(p)
            if p == path:
                calls.append(True)
                if len(calls) == 2: (self.directory / 'late.so').write_bytes(b'late child')
            return result
        with patch.object(library.sampler.system.python, 'sha', side_effect=changed): self.rejected()
        (self.directory / 'late.so').unlink(); self.verify()
        calls.clear()
        def content_changed(p):
            result = original(p)
            if p == path:
                calls.append(True)
                if len(calls) == 1: path.write_bytes(b'late file')
            return result
        with patch.object(library.sampler.system.python, 'sha', side_effect=content_changed): self.rejected()
        path.write_bytes(blob)

    def test_parent_io_timeout_is_fixed_rejection(self):
        original = Path.read_bytes
        def timeout(path):
            if path == self.runtime_path: raise TimeoutError('opaque private native library')
            return original(path)
        with patch.object(Path, 'read_bytes', timeout): self.rejected()

    def test_final_parent_nonregular_rejected_before_hash(self):
        original = library.loader.directory_identity; walks = []
        def change(path):
            result = original(path); walks.append(True)
            if len(walks) == 2:
                self.runtime_path.unlink(); os.mkfifo(self.runtime_path)
            return result
        original_sha = library.sampler.system.python.sha
        def guarded(path):
            if path == self.runtime_path: self.fail('late nonregular parent hashed')
            return original_sha(path)
        with patch.object(library.loader, 'directory_identity', side_effect=change), \
                patch.object(library.sampler.system.python, 'sha', side_effect=guarded): self.rejected()

    def test_final_helper_alias_same_bytes_rejects(self):
        helper = self.base / 'library-helper.py'; helper.write_bytes(Path(library.__file__).read_bytes())
        other = self.base / 'other-helper.py'; other.write_bytes(helper.read_bytes())
        self.lock['library_helper_sha256'] = fixture.sha(helper); self.freeze()
        with patch.object(library, '__file__', str(helper)):
            self.verify()
            original = library.loader.directory_identity; walks = []
            def alias(path):
                result = original(path); walks.append(True)
                if len(walks) == 2: helper.unlink(); helper.symlink_to(other)
                return result
            with patch.object(library.loader, 'directory_identity', side_effect=alias): self.rejected()

    def test_original_and_final_timer(self):
        with patch.object(library.sampler.bootstrap.time, 'monotonic', return_value=700.0), \
                self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            library.verify_native_library_directory(str(self.manifest), self.digest, batch_started_at=100.0)
        with patch.object(library.sampler.bootstrap.time, 'monotonic', side_effect=[125.0] * 5 + [700.0]), \
                self.assertRaisesRegex(TimeoutError, '^compressed bootstrap batch deadline exhausted$'):
            library.verify_native_library_directory(str(self.manifest), self.digest, batch_started_at=100.0)

    def test_exact_false_flags_counters_and_scores(self):
        old = copy.deepcopy(self.lock)
        for key, value in [('version', 1.0), ('candidate_processes_started', False),
                           ('execution_admitted', 0), ('actual_loader_selection_verified', True), ('mfs_v2', .99)]:
            self.lock = copy.deepcopy(old); self.lock[key] = value; self.freeze(); self.rejected()


if __name__ == '__main__': unittest.main()
