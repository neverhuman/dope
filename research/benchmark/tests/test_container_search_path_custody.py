"""Opaque declared path snapshots; no real loader, dependency or process starts."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import container_search_path_custody as custody
from research.benchmark.tests import test_container_loader_catalog as fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProjectionControls(unittest.TestCase):
    def test_literal_origin_tokens_and_absolute_components(self):
        for value, expected in [('$ORIGIN/opaque', '/opaque/declared/opaque'),
                                ('${ORIGIN}/../shared', '/opaque/shared'),
                                ('${ORIGIN}suffix', '/opaque/declaredsuffix'),
                                ('/opaque/absolute/./lib', '/opaque/absolute/lib'),
                                ('$ORIGIN', '/opaque/declared')]:
            with self.subTest(value=value):
                self.assertEqual(custody.project_component('/opaque/declared/object.so', value), expected)

    def test_empty_relative_and_unsupported_dynamic_components_reject(self):
        for value in ('', 'relative', '$LIB/lib', '$PLATFORM', '$ORIGIN_extra',
                      '${ORIGIN', '//opaque/lib', '/opaque/\0lib'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                custody.project_component('/opaque/object.so', value)

    def test_exact_builtin_identity_and_canonical_origin(self):
        class Alias(str):
            pass
        for origin, value in [(Alias('/opaque/object.so'), '$ORIGIN'),
                              ('/opaque/object.so', Alias('$ORIGIN')), ('relative', '/opaque'),
                              ('//opaque/object.so', '/opaque'), ('/opaque/./object.so', '/opaque')]:
            with self.subTest(origin_type=type(origin).__name__), self.assertRaises(ValueError):
                custody.project_component(origin, value)

    def test_projection_opens_no_path_and_does_not_resolve_origin_alias(self):
        with patch('pathlib.Path.resolve', side_effect=AssertionError('origin resolved')), \
                patch('builtins.open', side_effect=AssertionError('path opened')), \
                patch('subprocess.Popen', side_effect=AssertionError('process started')):
            self.assertEqual(custody.project_component('/opaque/declared-alias/object.so', '$ORIGIN/lib'),
                             '/opaque/declared-alias/lib')


class SearchSnapshotControls(unittest.TestCase):
    def setUp(self):
        self.f = fixture.CatalogControls('runTest'); self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        f.config.write_bytes(b'# opaque bounded config\n')
        f.lock['files'][str(f.config)].update(bytes=f.config.stat().st_size, sha256=sha(f.config))
        for path in (f.base / 'opaque', f.directory_alias / 'opaque'):
            path.mkdir()
            f.lock['directories'][str(path)] = custody.catalog.loader.directory_identity(str(path))
        f.lock['directories'][str(f.directory_alias)] = custody.catalog.loader.directory_identity(str(f.directory_alias))
        f.lock['absent'][str(f.bundle.parent / 'opaque')] = custody.catalog.loader.path_identity(str(f.bundle.parent / 'opaque'))
        f.freeze()
        self.search = {'format': 'dope-shared-validation-loader-search-input-preparation', 'version': 1,
            'loader_path': str(f.manifest), 'loader_sha256': f.digest,
            'configuration_helper_sha256': sha(Path(custody.configuration.__file__)),
            'configuration_root_path': str(f.config),
            'configuration_files': [{'path': str(f.config), 'sha256': sha(f.config),
                'declarations': custody.configuration.inspect_configuration(f.config.read_bytes(), sha(f.config))}],
            'configuration_directory_declarations': [], 'additional_directories': {},
            'additional_absent': {}, 'missing_configuration_directory_records': 0,
            'candidate_processes_started': 0}
        self.search_path = f.base / 'search.json'
        self.lock = {'format': 'dope-declared-search-path-projection-preparation', 'version': 1,
            'search_path': str(self.search_path), 'loader_path': str(f.manifest), 'loader_sha256': f.digest,
            'declarations_path': str(f.declarations), 'declarations_sha256': sha(f.declarations),
            'projection_helper_sha256': sha(Path(custody.__file__)), 'candidate_processes_started': 0,
            'unsupported_components': 0, 'unbound_snapshots': 0, 'records': []}
        for row in f.objects:
            self.lock['records'].append(self.record(row['path'], row['expected_sha256'], 'root',
                'absent' if row['path'] == str(f.bundle) else 'directory'))
        for path, row in json.loads(f.manifest.read_bytes())['files'].items():
            if row['role'] == 'candidate-provider':
                self.lock['records'].append(self.record(path, row['sha256'], 'selected-provider',
                    'absent' if path == str(f.bundle) else 'directory'))
        for lock in (self.search, self.lock):
            for key in ('actual_loader_selection_verified', 'complete_provider_search_verified',
                        'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
                lock[key] = False
            for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
                lock[key] = None
        self.lock['actual_loader_origin_verified'] = False
        self.manifest = f.base / 'projection.json'; self.freeze_search(); self.freeze()
        self.verify()  # Negative controls start from a validated complete fake parent chain.

    def record(self, path, digest, group, kind):
        return {'origin': {'path': path, 'sha256': digest, 'group': group}, 'kind': 'runpath',
                'ordinal': 0, 'component_ordinal': 0, 'declared_value': '$ORIGIN/opaque',
                'supported_projection': True, 'declared_path_projection': str(Path(path).parent / 'opaque'),
                'snapshot_kind': kind}

    def freeze_search(self):
        self.search_path.write_text(json.dumps(self.search, sort_keys=True))
        self.lock['search_sha256'] = sha(self.search_path)

    def freeze(self):
        self.manifest.write_text(json.dumps(self.lock, sort_keys=True)); self.digest = sha(self.manifest)

    def verify(self, **overrides):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'torch', 'sklearn', 'catboost'}:
                self.fail('dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('library loaded')):
            return custody.verify_declared_search_paths(**dict({'projection_path': str(self.manifest),
                'projection_sha256': self.digest}, **overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed declared search path rejected$'):
            self.verify(**overrides)

    def test_root_and_provider_snapshots_preserve_absent_execution_claims(self):
        result = self.verify()
        self.assertEqual(result['declared_components_verified'], 4)
        self.assertEqual(result['root_components_verified'], 2)
        self.assertEqual(result['selected_provider_components_verified'], 2)
        self.assertIs(result['declared_path_snapshots_verified'], True)
        for key in ('actual_loader_origin_verified', 'actual_loader_selection_verified',
                    'complete_provider_search_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            self.assertIs(result[key], False)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_external_builtin_digest_before_json(self):
        class Alias(str):
            pass
        with patch.object(custody.catalog.loader.system.json, 'loads', side_effect=AssertionError('JSON decoded')):
            for digest in (Alias(self.digest), self.digest.upper(), '0' * 64):
                self.rejected(projection_sha256=digest)

    def test_missing_extra_and_reordered_records_reject(self):
        old = copy.deepcopy(self.lock['records'])
        for records in (old[:-1], old + old[:1], list(reversed(old))):
            self.lock['records'] = records; self.freeze(); self.rejected()

    def test_numeric_and_component_identity_distinctions_reject(self):
        original = copy.deepcopy(self.lock)
        for key, value in [('ordinal', 0.0), ('component_ordinal', False), ('supported_projection', 1),
                           ('declared_value', '$ORIGIN/other'), ('snapshot_kind', 'directory'),
                           ('declared_path_projection', '/opaque/invented')]:
            self.lock = copy.deepcopy(original)
            index = 1 if key == 'snapshot_kind' else 0
            self.lock['records'][index][key] = value; self.freeze(); self.rejected()

    def test_manifest_claim_and_count_identity_reject(self):
        original = copy.deepcopy(self.lock)
        for key, value in [('version', True), ('unsupported_components', False), ('unbound_snapshots', 0.0),
                           ('candidate_processes_started', False), ('actual_loader_origin_verified', 0),
                           ('execution_admitted', True), ('mfs_v2', .99)]:
            self.lock = copy.deepcopy(original); self.lock[key] = value; self.freeze(); self.rejected()

    def test_parent_and_helper_identity_reject(self):
        for key in ('projection_helper_sha256', 'search_sha256', 'loader_sha256', 'declarations_sha256'):
            original = self.lock[key]; self.lock[key] = '0' * 64; self.freeze(); self.rejected()
            self.lock[key] = original

    def test_root_bytes_and_owned_provider_bytes_reject(self):
        for path in (self.f.root, self.f.provider):
            original = path.read_bytes(); path.write_bytes(b'changed opaque input')
            self.rejected(); path.write_bytes(original)

    def test_directory_child_drift_and_filled_absence_reject(self):
        child = self.f.base / 'opaque' / 'new-child'; child.write_bytes(b'opaque')
        self.rejected(); child.unlink()
        missing = self.f.bundle.parent / 'opaque'; missing.mkdir(); self.rejected()

    def test_omitted_directory_even_if_parent_hashes_refreshed_rejects(self):
        del self.f.lock['directories'][str(self.f.base / 'opaque')]
        self.f.freeze(); self.search['loader_sha256'] = self.f.digest
        self.lock['loader_sha256'] = self.f.digest; self.freeze_search(); self.freeze(); self.rejected()

    def test_late_manifest_and_snapshot_changes_reject(self):
        original = custody.project_component
        for path in (self.manifest, self.search_path, self.f.manifest, self.f.root):
            blob = path.read_bytes(); changed = []
            def mutate(*args):
                result = original(*args)
                if not changed:
                    changed.append(True); path.write_bytes(b'changed after owned projection')
                return result
            with self.subTest(path=path.name), patch.object(custody, 'project_component', side_effect=mutate):
                self.rejected()
            path.write_bytes(blob)

    def test_parent_guards_run_before_and_after_projection(self):
        with patch.object(custody.configuration, 'verify_configuration_inputs',
                          wraps=custody.configuration.verify_configuration_inputs) as config, \
                patch.object(custody.catalog, 'verify_candidate_catalog',
                             wraps=custody.catalog.verify_candidate_catalog) as catalog:
            self.verify(); self.assertEqual(config.call_count, 2); self.assertEqual(catalog.call_count, 2)


if __name__ == '__main__':
    unittest.main()
