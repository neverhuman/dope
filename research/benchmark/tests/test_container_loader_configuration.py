"""Owned opaque configuration declarations; no file, library or process lookup."""
import hashlib
import builtins
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import container_loader_configuration as configuration
from research.benchmark.tests import test_container_loader_custody as fixture


class ConfigurationControls(unittest.TestCase):
    def inspect(self, blob):
        with patch('subprocess.run', side_effect=AssertionError('process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('library loaded')), \
                patch('builtins.open', side_effect=AssertionError('configuration path opened')):
            return configuration.inspect_configuration(blob, hashlib.sha256(blob).hexdigest())

    def rejected(self, blob):
        with self.assertRaisesRegex(ValueError, '^compressed loader configuration rejected$'):
            self.inspect(blob)

    def test_order_comments_duplicate_and_include_declarations(self):
        blob = b'# opaque\n /opaque/first # comment\n\ninclude\t/opaque/config/*.conf\r\n/opaque/first\n'
        result = self.inspect(blob)
        self.assertEqual(result['records'], [{'line': 2, 'kind': 'directory', 'value': '/opaque/first'},
            {'line': 4, 'kind': 'include', 'value': '/opaque/config/*.conf'},
            {'line': 5, 'kind': 'directory', 'value': '/opaque/first'}])
        for key in ('include_expansion_verified', 'actual_loader_selection_verified',
                    'complete_provider_search_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            self.assertIs(result[key], False)
        self.assertEqual(result['candidate_processes_started'], 0)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_external_digest_before_decode(self):
        with self.assertRaisesRegex(ValueError, '^compressed loader configuration rejected$'):
            configuration.inspect_configuration(b'\xff', '0' * 64)

    def test_builtin_bytes_and_digest_identities(self):
        class Permissive(str):
            def __eq__(self, other):
                return True
        blob = b'/opaque/dir';digest = hashlib.sha256(blob).hexdigest()
        for data, value in [(bytearray(blob), digest), (blob, Permissive(digest)),
                            (blob, digest.upper()), (blob, None)]:
            with self.subTest(kind=type(data).__name__, digest=type(value).__name__), self.assertRaises(ValueError):
                configuration.inspect_configuration(data, value)

    def test_unknown_directives_and_multiple_arguments_rejected(self):
        for blob in (b'hwcap 1 opaque', b'include', b'include /opaque/a /opaque/b',
                     b'/opaque/a /opaque/b', b'include-missing /opaque/*.conf'):
            self.rejected(blob)

    def test_only_canonical_absolute_directory_and_fixed_parent_include_paths(self):
        for value in ('relative', '//opaque/lib', '/opaque/../lib', '/opaque/./lib', '/opaque/lib/',
                      '/opaque/$LIB', '/opaque/a\\b', '/opaque/*', '/opaque/[ab]',
                      'include /opaque/*/lib.conf', 'include /opaque/?/lib.conf',
                      'include /opaque/[ab]/lib.conf', 'include /opaque/{a,b}.conf',
                      'include /opaque/[ab].conf', 'include /opaque/?.conf'):
            with self.subTest(value=value):
                self.rejected(value.encode())

    def test_utf8_controls_nul_and_non_newline_separators_rejected(self):
        for blob in (b'\xff', b'/opaque/dir\0', '/opaque/a\u2028/opaque/b'.encode(),
                     b'/opaque/a\v/opaque/b', b'/opaque/a\f/opaque/b'):
            self.rejected(blob)

    def test_record_and_input_byte_limits(self):
        self.rejected(b'#' * 1048577)
        self.rejected(b'/opaque/dir\n' * 4097)
        self.assertEqual(len(self.inspect(b'/opaque/dir\n' * 4096)['records']), 4096)

    def test_empty_and_comments_only_supported_without_claiming_expansion(self):
        self.assertEqual(self.inspect(b'')['records'], [])
        self.assertEqual(self.inspect(b'# opaque\n \t\r\n')['records'], [])


class ConfigurationSnapshotControls(unittest.TestCase):
    freeze = fixture.LoaderInputControls.freeze

    def setUp(self):
        fixture.LoaderInputControls.setUp(self)
        self.config_directory = self.base / 'config'; self.config_directory.mkdir()
        self.first = self.config_directory / 'a.conf'
        self.second = self.config_directory / 'b.conf'
        self.search_directory = self.base / 'new-search'; self.search_directory.mkdir()
        self.search_absent = self.base / 'missing-search'
        self.config.write_text('include ' + str(self.config_directory / '*.conf') + '\n')
        self.first.write_text(str(self.search_directory) + '\n')
        self.second.write_text(str(self.search_absent) + '\n')
        for path in (self.config, self.first, self.second):
            self.capture_config(path)
        self.lock['directories'][str(self.config_directory)] = configuration.loader.directory_identity(str(self.config_directory))
        self.freeze()
        self.search = {'format': 'dope-shared-validation-loader-search-input-preparation', 'version': 1,
            'loader_path': str(self.manifest), 'loader_sha256': self.digest,
            'configuration_helper_sha256': fixture.sha(Path(configuration.__file__)),
            'configuration_root_path': str(self.config), 'configuration_files': [],
            'configuration_directory_declarations': [str(self.search_directory), str(self.search_absent)],
            'additional_directories': {str(self.search_directory): configuration.loader.directory_identity(str(self.search_directory))},
            'additional_absent': {str(self.search_absent): configuration.loader.path_identity(str(self.search_absent))},
            'missing_configuration_directory_records': 2, 'candidate_processes_started': 0}
        for key in ('actual_loader_selection_verified', 'complete_provider_search_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            self.search[key] = False
        for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            self.search[key] = None
        self.search_manifest = self.base / 'search.json'; self.capture_records(); self.freeze_search()
        self.verify()  # Every rejection control starts from an independently accepted fixture.

    def capture_config(self, path):
        row = configuration.loader.path_identity(str(path))
        row.update(role='configuration', bytes=path.stat().st_size, sha256=fixture.sha(path))
        self.lock['files'][str(path)] = row

    def capture_records(self):
        self.search['configuration_files'] = [
            {'path': path, 'sha256': row['sha256'],
             'declarations': configuration.inspect_configuration(Path(row['resolved']).read_bytes(), row['sha256'])}
            for path, row in json.loads(self.manifest.read_bytes())['files'].items()
            if row['role'] == 'configuration']

    def freeze_search(self):
        self.search_manifest.write_text(json.dumps(self.search, sort_keys=True))
        self.search_digest = fixture.sha(self.search_manifest)

    def verify(self, **overrides):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'torch', 'catboost'}:
                self.fail('candidate dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('library loaded')):
            return configuration.verify_configuration_inputs(**dict({'search_path': str(self.search_manifest),
                'search_sha256': self.search_digest}, **overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed loader configuration rejected$'):
            self.verify(**overrides)

    def test_bound_selected_configs_and_missing_directory_records(self):
        result = self.verify()
        self.assertEqual(result['configuration_files_verified'], 3)
        self.assertEqual(result['configured_directory_declarations_verified'], 2)
        self.assertEqual(result['additional_directory_snapshots_verified'], 1)
        self.assertEqual(result['additional_absence_snapshots_verified'], 1)
        self.assertIs(result['include_file_membership_verified'], True)
        for key in ('actual_loader_selection_verified', 'complete_provider_search_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            self.assertIs(result[key], False)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_manifest_and_builtin_digest_before_parse(self):
        class Permissive(str):
            def __eq__(self, other):
                return True
        with patch.object(configuration.loader.system.json, 'loads', side_effect=AssertionError('unbound JSON parsed')):
            self.rejected(search_sha256=Permissive(self.search_digest))
            self.search_manifest.write_bytes(b'changed private proposal'); self.rejected()

    def test_parent_or_helper_drift_precedes_configuration_decode(self):
        with patch.object(configuration, 'inspect_configuration', side_effect=AssertionError('unbound configuration decoded')):
            self.search['configuration_helper_sha256'] = '0' * 64; self.freeze_search(); self.rejected()
            self.search['configuration_helper_sha256'] = fixture.sha(Path(configuration.__file__))
            self.freeze_search(); self.manifest.write_bytes(b'changed selected parent'); self.rejected()

    def test_owned_configuration_and_record_changes_rejected(self):
        self.first.write_bytes(b'changed body'); self.rejected()
        self.first.write_text(str(self.search_directory) + '\n')
        self.search['configuration_files'][0]['declarations']['records'][0]['line'] = 1.0
        self.freeze_search(); self.rejected()

    def test_missing_extra_and_changed_directory_records(self):
        old = copy.deepcopy(self.search)
        for group in ('additional_directories', 'additional_absent'):
            self.search = copy.deepcopy(old); self.search[group] = {}; self.freeze_search(); self.rejected()
        self.search = copy.deepcopy(old)
        self.search['additional_absent'][str(self.absent)] = configuration.loader.path_identity(str(self.absent))
        self.freeze_search(); self.rejected()
        self.search = copy.deepcopy(old); self.freeze_search()
        (self.search_directory / 'new-alias').symlink_to('still-missing'); self.rejected()

    def test_added_absence_path_rejects(self):
        self.search_absent.write_bytes(b'added input'); self.rejected()

    def test_include_matching_file_must_have_selected_body(self):
        added = self.config_directory / 'added.conf'; added.write_bytes(b'/opaque/unbound\n')
        self.lock['directories'][str(self.config_directory)] = configuration.loader.directory_identity(str(self.config_directory))
        self.freeze(); self.search['loader_sha256'] = self.digest; self.freeze_search(); self.rejected()

    def test_selected_body_outside_include_membership_rejects(self):
        added = self.config_directory / 'unreferenced.txt'; added.write_bytes(b'# opaque\n')
        self.capture_config(added)
        self.lock['directories'][str(self.config_directory)] = configuration.loader.directory_identity(str(self.config_directory))
        self.freeze(); self.search['loader_sha256'] = self.digest
        self.capture_records(); self.freeze_search(); self.rejected()

    def test_wrong_root_and_numeric_claims_reject(self):
        for key, value in [('configuration_root_path', str(self.first)), ('version', True),
                           ('candidate_processes_started', False), ('execution_admitted', 0),
                           ('mfs_v2', .99), ('missing_configuration_directory_records', 2.0)]:
            with self.subTest(key=key):
                old = self.search[key]; self.search[key] = value
                self.freeze_search(); self.rejected(); self.search[key] = old

    def test_final_recheck_catches_configuration_snapshot_and_parent_mutations(self):
        original = configuration.inspect_configuration
        for path in (self.first, self.manifest, self.search_manifest):
            with self.subTest(path=path.name):
                old = path.read_bytes()
                def changed(blob, expected):
                    result = original(blob, expected); path.write_bytes(b'changed after owned parse'); return result
                with patch.object(configuration, 'inspect_configuration', side_effect=changed):
                    self.rejected()
                path.write_bytes(old)
        snapshot = configuration.loader.directory_identity
        late = self.search_directory / 'late'
        def changed_directory(path):
            result = snapshot(path)
            if path == str(self.search_directory) and not late.exists():
                late.write_bytes(b'added after snapshot')
            return result
        with patch.object(configuration.loader, 'directory_identity', side_effect=changed_directory):
            self.rejected()


if __name__ == '__main__':
    unittest.main()
