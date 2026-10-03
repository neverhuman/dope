"""Opaque loader snapshot controls; no real loader or candidate is invoked."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import container_loader_custody as loader
from research.benchmark.tests.test_container_elf_inputs import opaque_object


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LoaderInputControls(unittest.TestCase):
    def setUp(self):
        target = Path('target').resolve(); target.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='loader-control-', dir=target)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.patch = patch.object(loader.system.python, 'BASE', self.base)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.library = self.base / 'lib'; self.library.mkdir()
        self.provider = self.library / 'provider.so'; self.provider.write_bytes(opaque_object())
        self.alias = self.library / 'alias.so'; self.alias.symlink_to('provider.so')
        self.directory_alias = self.base / 'lib-alias'; self.directory_alias.symlink_to('lib')
        self.tool = self.base / 'metadata-tool'; self.tool.write_bytes(b'opaque, never executable')
        self.cache = self.base / 'cache'; self.cache.write_bytes(b'opaque cache bytes')
        self.config = self.base / 'configuration'; self.config.write_bytes(b'opaque configuration')
        self.absent = self.base / 'missing-preload'
        self.marker = self.base / 'candidate-initialized'
        self.declarations = self.base / 'declarations.json'
        self.declarations.write_text(json.dumps({'helper_sha256': sha(Path(loader.system.elf.__file__))}))
        self.unresolved = self.base / 'unresolved.json'
        self.unresolved.write_text(json.dumps({'declarations_path': str(self.declarations),
                                              'declarations_sha256': sha(self.declarations)}))
        for name in ('stdout', 'stderr'):
            (self.base / ('cache-print.' + name)).write_bytes(b'opaque transcript')
        self.cache_receipt = self.base / 'cache-receipt.json'
        self.cache_receipt.write_text(json.dumps({'metadata_tool_argv': [str(self.tool), '-p', '-C', str(self.cache)],
            'tool_sha256': sha(self.tool), 'cache_sha256': sha(self.cache),
            **{name + '_sha256': sha(self.base / ('cache-print.' + name)) for name in ('stdout', 'stderr')}}))
        self.lock = {'format': 'dope-shared-validation-loader-input-preparation', 'version': 1,
                     'files': {}, 'directories': {}, 'absent': {}, 'gaps': []}
        for key, path in [('declarations', self.declarations), ('unresolved', self.unresolved),
                          ('cache_receipt', self.cache_receipt)]:
            self.lock.update({key + '_path': str(path), key + '_sha256': sha(path)})
        for key in ('official_tests_opened', 'actual_loader_selection_verified',
                    'complete_provider_search_verified', 'cache_default_preload_hwcaps_dlopen_complete',
                    'cross_host_verified', 'full_runtime_closure_certified', 'execution_admitted'):
            self.lock[key] = False
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.lock[key] = None
        for key in ('candidate_libraries_loaded', 'generator_fits', 'auditor_fits', 'samples_generated'):
            self.lock[key] = 0
        for path, role in [(self.tool, 'metadata-tool'), (self.cache, 'cache'),
                           (self.config, 'configuration'), (self.directory_alias / self.alias.name, 'candidate-provider')]:
            row = loader.path_identity(str(path))
            row.update(bytes=path.stat().st_size, sha256=sha(path), role=role)
            if role == 'candidate-provider':
                row['declarations'] = loader.system.elf.inspect_elf_inputs(path.read_bytes(), sha(path))
            self.lock['files'][str(path)] = row
        self.lock['directories'][str(self.directory_alias)] = loader.directory_identity(str(self.directory_alias))
        self.lock['absent'][str(self.absent)] = loader.path_identity(str(self.absent))
        self.manifest = self.base / 'loader.json'; self.freeze()

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock, sort_keys=True))
        self.digest = sha(self.manifest)

    def verify(self, **overrides):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'catboost', 'torch'}:
                self.marker.write_text('unexpected dependency initializer')
                self.fail('candidate dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('library loaded')):
            result = loader.verify_loader_inputs(**dict({'loader_path': str(self.manifest),
                                                        'loader_sha256': self.digest}, **overrides))
        self.assertFalse(self.marker.exists())
        return result

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed loader input custody rejected$'):
            self.verify(**overrides)
        self.assertFalse(self.marker.exists())

    def test_snapshot_success_preserves_unadmitted_null_claims(self):
        result = self.verify()
        self.assertEqual(result['selected_files_verified'], 4)
        self.assertEqual(result['directory_snapshots_verified'], 1)
        self.assertEqual(result['absence_records_verified'], 1)
        self.assertIs(result['selected_loader_inputs_verified'], True)
        for key in ('actual_loader_selection_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            self.assertIs(result[key], False)
        self.assertEqual(result['candidate_processes_started'], 0)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_manifest_drift_precedes_json_parse(self):
        self.manifest.write_bytes(b'changed private input')
        with patch.object(loader.system.json, 'loads', side_effect=AssertionError('unbound JSON parsed')):
            self.rejected()

    def test_external_digest_subclass_rejected_before_parse(self):
        class Permissive(str):
            def __eq__(self, other):
                return True
        with patch.object(loader.system.json, 'loads', side_effect=AssertionError('invalid digest parsed')):
            self.rejected(loader_sha256=Permissive(self.digest))

    def test_manifest_alias_rejected(self):
        alias = self.base / 'manifest-alias'; alias.symlink_to(self.manifest.name)
        self.rejected(loader_path=str(alias))

    def test_every_parent_digest_and_transcript_rechecked(self):
        for path in (self.declarations, self.unresolved, self.cache_receipt,
                     self.base / 'cache-print.stdout', self.base / 'cache-print.stderr'):
            with self.subTest(path=path.name):
                old = path.read_bytes(); path.write_bytes(b'changed private parent')
                self.rejected(); path.write_bytes(old)

    def test_parent_binding_mismatch_rejected(self):
        self.unresolved.write_text(json.dumps({'declarations_path': str(self.config),
                                              'declarations_sha256': sha(self.declarations)}))
        self.lock['unresolved_sha256'] = sha(self.unresolved); self.freeze(); self.rejected()

    def test_cache_and_tool_binding_mismatch_rejected(self):
        original = self.cache_receipt.read_bytes()
        for key in ('cache_sha256', 'tool_sha256'):
            with self.subTest(key=key):
                row = json.loads(original); row[key] = '0' * 64
                self.cache_receipt.write_text(json.dumps(row))
                self.lock['cache_receipt_sha256'] = sha(self.cache_receipt); self.freeze(); self.rejected()

    def test_helper_digest_precedes_elf_parser(self):
        self.declarations.write_text(json.dumps({'helper_sha256': '0' * 64}))
        self.lock['declarations_sha256'] = sha(self.declarations)
        self.unresolved.write_text(json.dumps({'declarations_path': str(self.declarations),
                                              'declarations_sha256': sha(self.declarations)}))
        self.lock['unresolved_sha256'] = sha(self.unresolved); self.freeze()
        with patch.object(loader.system.elf, 'inspect_elf_inputs', side_effect=AssertionError('unbound parser')):
            self.rejected()

    def test_cache_configuration_and_provider_bytes_drift(self):
        for path in (self.cache, self.config, self.provider):
            with self.subTest(path=path.name):
                old = path.read_bytes(); path.write_bytes(b'changed candidate file')
                self.rejected(); path.write_bytes(old)

    def test_file_mode_and_alias_retarget_rejected(self):
        old = self.provider.stat().st_mode & 0o777
        self.provider.chmod(old ^ 0o100); self.rejected(); self.provider.chmod(old)
        other = self.library / 'other.so'; other.write_bytes(self.provider.read_bytes())
        self.alias.unlink(); self.alias.symlink_to(other.name); self.rejected()

    def test_added_directory_file_and_alias_rejected(self):
        for kind in ('file', 'directory', 'alias'):
            with self.subTest(kind=kind):
                added = self.library / 'added'
                if kind == 'file':
                    added.write_bytes(b'opaque uninventoried bytes')
                elif kind == 'directory':
                    added.mkdir()
                else:
                    added.symlink_to('provider.so')
                self.rejected()
                if kind == 'directory':
                    added.rmdir()
                else:
                    added.unlink()

    def test_ancestor_alias_drift_with_same_bytes_rejected(self):
        self.directory_alias.unlink(); self.directory_alias.symlink_to(str(self.library))
        self.rejected()

    def test_frozen_absence_rejects_added_file_and_broken_alias(self):
        self.absent.write_bytes(b'added preload'); self.rejected(); self.absent.unlink()
        self.absent.symlink_to('still-missing'); self.rejected()

    def test_special_file_rejected_before_read(self):
        self.config.unlink(); os.mkfifo(self.config)
        original = Path.read_bytes
        def checked(path):
            if path == self.config:
                self.fail('special input opened')
            return original(path)
        with patch.object(Path, 'read_bytes', checked):
            self.rejected()

    def test_numeric_identity_and_claim_bypasses_rejected(self):
        row = self.lock['files'][str(self.cache)]; row['bytes'] = float(row['bytes'])
        self.freeze(); self.rejected(); row['bytes'] = int(row['bytes'])
        for key, value in [('version', True), ('candidate_libraries_loaded', False),
                           ('execution_admitted', 0), ('mfs_v2', .99), ('gaps', [{}])]:
            with self.subTest(key=key):
                old = self.lock[key]; self.lock[key] = value
                self.freeze(); self.rejected(); self.lock[key] = old

    def test_wrong_owned_provider_declarations_rejected(self):
        row = self.lock['files'][str(self.directory_alias / self.alias.name)]
        row['declarations']['declarations']['needed'] = ['invented.so']
        self.freeze(); self.rejected()

    def test_symlink_target_dot_parts_follow_kernel_component_order(self):
        nested = self.library / 'nested'; nested.mkdir()
        hop = self.base / 'hop'; hop.symlink_to('lib/nested')
        alias = self.base / 'dot-alias'; alias.symlink_to('hop/../provider.so')
        identity = loader.path_identity(str(alias))
        self.assertEqual(identity['resolved'], str(self.provider))
        self.assertEqual([row['path'] for row in identity['aliases']], [str(alias), str(hop)])

    def test_alias_cycle_rejected_without_open(self):
        self.alias.unlink(); self.alias.symlink_to(self.alias.name)
        self.rejected()

    def test_final_recheck_detects_file_parent_and_directory_changes(self):
        original = loader.system.elf.inspect_elf_inputs
        for path in (self.tool, self.manifest, self.unresolved):
            with self.subTest(path=path.name):
                old = path.read_bytes()
                def changed(blob, expected):
                    result = original(blob, expected); path.write_bytes(b'changed after first hash')
                    return result
                with patch.object(loader.system.elf, 'inspect_elf_inputs', side_effect=changed):
                    self.rejected()
                path.write_bytes(old)
        snapshot = loader.directory_identity
        added = self.library / 'late-alias'
        def changed_directory(path):
            result = snapshot(path)
            if not added.is_symlink():
                added.symlink_to('provider.so')
            return result
        with patch.object(loader, 'directory_identity', side_effect=changed_directory):
            self.rejected()


if __name__ == '__main__':
    unittest.main()
