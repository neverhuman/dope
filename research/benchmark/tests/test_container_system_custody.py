"""Opaque ELF-file custody controls; no library or candidate starts."""
import builtins
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research.benchmark import container_system_custody as system
from research.benchmark.tests.test_container_elf_inputs import opaque_object
from research.benchmark.tests import test_container_python_custody as opaque_inputs

identity, sha = opaque_inputs.identity, opaque_inputs.sha


class SystemInputControls(unittest.TestCase):
    def freeze(self):
        opaque_inputs.ImportInputControls.freeze(self)

    def setUp(self):
        opaque_inputs.ImportInputControls.setUp(self)
        self.python.write_bytes(opaque_object())
        self.extension = self.site / 'numpy/opaque.so'
        self.extension.write_bytes(opaque_object())
        stdlib = self.library / 'lib-dynload/opaque.so'
        stdlib.write_bytes(opaque_object())
        self.elf_alias = self.library / 'opaque-alias.so'
        self.elf_alias.symlink_to('lib-dynload/opaque.so')
        self.runtime['python_sha256'] = sha(self.python)
        self.runtime['files']['numpy/opaque.so'] = identity(self.extension)
        self.auxiliary['stdlib_all_files'].update({str(p): identity(p) for p in [stdlib, self.elf_alias]})
        self.auxiliary['stdlib_aliases'][str(self.elf_alias)] = {
            'target': 'lib-dynload/opaque.so', 'resolved': str(stdlib), 'sha256': sha(stdlib)}
        self.freeze()
        self.declaration_file = self.base / 'declarations.lock.json'
        self.lock = {'format': 'dope-shared-validation-system-input-preparation', 'version': 1,
                     'runtime_path': str(self.runtime_file), 'runtime_sha256': self.runtime_sha,
                     'auxiliary_path': str(self.auxiliary_file), 'auxiliary_sha256': self.auxiliary_sha,
                     'helper_sha256': sha(Path(system.elf.__file__)), 'objects': []}
        for key in ['official_tests_opened', 'loader_resolution_verified',
                    'system_library_file_inventory_complete', 'full_runtime_closure_certified',
                    'cross_host_runtime_identity_verified', 'sdv_predecessor_closed',
                    'density_priority_satisfied', 'capacity_admitted', 'execution_admitted',
                    'selection_performed']:
            self.lock[key] = False
        for key in ['mfs_v2', 'ptf_v1', 'release_safe', 'superiority']:
            self.lock[key] = None
        for key in ['candidate_libraries_loaded', 'generator_fits', 'auditor_fits', 'samples_generated', 'gaps']:
            self.lock[key] = 0
        for path, digest, group, alias in system.selected_inputs(self.runtime, self.auxiliary):
            self.lock['objects'].append({'path': str(path), 'group': group, 'expected_sha256': digest,
                'declared_alias': alias, 'bytes': path.stat().st_size, 'file_hash_verified': True,
                'status': 'declarations_recorded',
                'declarations': system.elf.inspect_elf_inputs(path.read_bytes(), digest)})
        self.lock['declaration_objects'] = self.lock['declarations_recorded'] = len(self.lock['objects'])
        self.freeze_lock()

    def freeze_lock(self):
        self.declaration_file.write_text(json.dumps(self.lock, sort_keys=True))
        self.declaration_sha = sha(self.declaration_file)

    def verify(self, **overrides):
        args = {'declarations_path': str(self.declaration_file), 'declarations_sha256': self.declaration_sha,
                'runtime_path': str(self.runtime_file), 'runtime_sha256': self.runtime_sha,
                'auxiliary_path': str(self.auxiliary_file), 'auxiliary_sha256': self.auxiliary_sha}
        args.update(overrides)
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'catboost', 'torch'}:
                self.fail('candidate dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('candidate process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('candidate process started')):
            result = system.verify_declared_elf_inputs(**args)
        self.assertFalse(self.marker.exists())
        return result

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed ELF file custody rejected$'):
            self.verify(**overrides)
        self.assertFalse(self.marker.exists())

    def test_complete_files_and_aliases_do_not_admit_execution(self):
        result = self.verify()
        self.assertEqual(result['declared_elf_files_verified'], 4)
        self.assertTrue(result['python_import_inputs_verified'])
        for key in ['loader_resolution_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened']:
            self.assertIs(result[key], False)
        self.assertEqual(result['candidate_processes_started'], 0)
        self.assertTrue(all(result[k] is None for k in ['mfs_v2', 'ptf_v1', 'release_safe', 'superiority']))

    def test_frozen_manifest_drift_rejects_before_parsing(self):
        self.declaration_file.write_bytes(b'private altered manifest')
        with patch.object(system.json, 'loads', side_effect=AssertionError('unbound manifest parsed')):
            self.rejected()

    def test_external_parent_drift_rejected(self):
        for key in ['runtime_sha256', 'auxiliary_sha256']:
            with self.subTest(key=key):
                self.rejected(**{key: '0' * 64})

    def test_parent_binding_cannot_select_another_runtime(self):
        self.lock['runtime_path'] = str(self.metric)
        self.freeze_lock(); self.rejected()

    def test_helper_drift_precedes_parser_use(self):
        self.lock['helper_sha256'] = '0' * 64
        self.freeze_lock()
        with patch.object(system.elf, 'inspect_elf_inputs', side_effect=AssertionError('unbound parser used')):
            self.rejected()

    def test_omitted_duplicate_or_added_members_rejected(self):
        original = list(self.lock['objects'])
        for rows in [original[:-1], original[:-1] + [original[0]], original + [original[0]]]:
            with self.subTest(rows=len(rows)):
                self.lock['objects'] = rows
                self.freeze_lock(); self.rejected()

    def test_changed_declarations_or_numeric_identity_rejected(self):
        row = self.lock['objects'][0]
        row['declarations']['declarations']['needed'] = ['invented.so']
        self.freeze_lock(); self.rejected()
        row['declarations'] = system.elf.inspect_elf_inputs(self.python.read_bytes(), sha(self.python))
        row['bytes'] = float(row['bytes'])
        self.freeze_lock(); self.rejected()

    def test_import_extension_and_added_file_drift_rejected(self):
        self.extension.write_bytes(b'changed candidate extension')
        self.rejected()
        self.extension.write_bytes(opaque_object())
        (self.site / 'numpy/extra.so').write_bytes(opaque_object())
        self.rejected()

    def test_alias_target_drift_rejected(self):
        self.elf_alias.unlink(); self.elf_alias.symlink_to('lib-dynload/missing.so')
        self.rejected()

    def test_directory_alias_rejected(self):
        (self.site / 'added-alias').symlink_to(self.site, target_is_directory=True)
        self.rejected()

    def test_claim_and_boolean_number_bypasses_rejected(self):
        for key, value in [('execution_admitted', True), ('official_tests_opened', 0),
                           ('mfs_v2', 0.99), ('version', True), ('gaps', False),
                           ('declaration_objects', 4.0)]:
            with self.subTest(key=key):
                original = self.lock[key]; self.lock[key] = value
                self.freeze_lock(); self.rejected(); self.lock[key] = original

    def test_digest_subclass_rejected(self):
        class Permissive(str):
            def __eq__(self, other):
                return True
        self.rejected(declarations_sha256=Permissive(self.declaration_sha))

    def test_final_rehash_rejects_changes_after_parsing(self):
        original = system.elf.inspect_elf_inputs
        def changed(blob, expected):
            result = original(blob, expected)
            self.declaration_file.write_bytes(b'changed after owned metadata parse')
            return result
        with patch.object(system.elf, 'inspect_elf_inputs', side_effect=changed):
            self.rejected()


if __name__ == '__main__':
    unittest.main()
