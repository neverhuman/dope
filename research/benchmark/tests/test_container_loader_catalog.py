"""Opaque catalog replay controls; no real loader, dependency or process starts."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

from research.benchmark import container_loader_catalog as catalog
from research.benchmark.tests import test_container_loader_custody as fixture
from research.benchmark.tests.test_container_elf_inputs import opaque_object


def opaque_provider(marker=0):
    blob = opaque_object()
    struct.pack_into('<I', blob, 64 + 56, 4)  # No interpreter declaration.
    struct.pack_into('<qQ', blob, 512 + 4 * 16, 0, 0)  # No audit declaration.
    blob[1000] = marker
    return bytes(blob)


class CatalogControls(unittest.TestCase):
    freeze = fixture.LoaderInputControls.freeze

    def setUp(self):
        fixture.LoaderInputControls.setUp(self)
        self.provider.write_bytes(opaque_provider())
        self.root = self.base / 'root.so'; self.root.write_bytes(opaque_provider(1))
        bundle = self.base / 'bundle'; bundle.mkdir()
        self.bundle = bundle / 'libopaque.so'; self.bundle.write_bytes(opaque_provider(2))
        self.objects = []
        for path, group in [(self.root, 'interpreter'), (self.bundle, 'package-extension')]:
            self.objects.append({'path': str(path), 'group': group, 'expected_sha256': fixture.sha(path),
                'declarations': catalog.loader.system.elf.inspect_elf_inputs(path.read_bytes(), fixture.sha(path))})
        for path in (self.directory_alias / self.alias.name, self.bundle):
            row = catalog.loader.path_identity(str(path))
            row.update(bytes=path.stat().st_size, sha256=fixture.sha(path), role='candidate-provider',
                       declarations=catalog.loader.system.elf.inspect_elf_inputs(path.read_bytes(), fixture.sha(path)))
            self.lock['files'][str(path)] = row
        self.stdout = self.base / 'cache-print.stdout'
        self.stdout.write_text('opaque header\n\tlibopaque.so (libc6,x86-64) => '
                               + str(self.directory_alias / self.alias.name) + '\n'
                               + '\tlibopaque.so (libc6,AArch64) => /opaque/wrong-architecture\n')
        self.cache_row = json.loads(self.cache_receipt.read_bytes())
        self.cache_row['stdout_sha256'] = fixture.sha(self.stdout)
        self.cache_receipt.write_text(json.dumps(self.cache_row))
        self.lock['cache_receipt_sha256'] = fixture.sha(self.cache_receipt)
        self.lock['candidate_enumeration_policy'] = catalog.POLICY
        self.freeze_parents()
        # Written explicitly, rather than by the function under test.
        paths = [str(self.bundle), str(self.directory_alias / self.alias.name)]
        self.expected = [dict(edge, candidate_paths=paths) for edge in self.roots if edge['kind'] == 'needed']
        self.expected.append({'origin': {'path': paths[1], 'sha256': fixture.sha(self.provider)},
                              'kind': 'needed', 'ordinal': 0, 'declared_value': 'libopaque.so',
                              'candidate_paths': paths})
        self.lock['candidate_edges'] = copy.deepcopy(self.expected); self.freeze()

    def freeze_parents(self):
        self.roots = []
        for row in self.objects:
            origin = {'path': row['path'], 'sha256': row['expected_sha256']}
            inspected = row['declarations']
            if inspected['interpreter'] is not None:
                self.roots.append(dict(origin=origin, kind='interpreter', ordinal=0,
                                       declared_value=inspected['interpreter']))
            for kind in ('needed', 'rpath', 'runpath'):
                for i, name in enumerate(inspected['declarations'][kind]):
                    self.roots.append(dict(origin=origin, kind=kind, ordinal=i, declared_value=name))
        self.declarations.write_text(json.dumps({'helper_sha256': fixture.sha(Path(catalog.loader.system.elf.__file__)),
                                                'objects': self.objects}))
        self.unresolved.write_text(json.dumps({'declarations_path': str(self.declarations),
            'declarations_sha256': fixture.sha(self.declarations), 'declared_edges': self.roots}))
        self.lock['declarations_sha256'] = fixture.sha(self.declarations)
        self.lock['unresolved_sha256'] = fixture.sha(self.unresolved)

    def verify(self, **overrides):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'catboost', 'torch'}:
                self.fail('candidate dependency initialized')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), \
                patch('subprocess.run', side_effect=AssertionError('process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('process started')), \
                patch('ctypes.CDLL', side_effect=AssertionError('library loaded')):
            return catalog.verify_candidate_catalog(**dict({'loader_path': str(self.manifest),
                                                            'loader_sha256': self.digest}, **overrides))

    def rejected(self, **overrides):
        with self.assertRaisesRegex(ValueError, '^compressed loader candidate catalog rejected$'):
            self.verify(**overrides)

    def test_explicit_order_recursive_cycle_and_competing_identities(self):
        result = self.verify()
        self.assertEqual(result['ordered_candidate_edges_verified'], 3)
        self.assertEqual(result['competing_provider_identity_edges'], 3)
        self.assertIs(result['ordered_candidate_catalog_verified'], True)
        for key in ('actual_loader_selection_verified', 'complete_provider_search_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            self.assertIs(result[key], False)
        self.assertEqual(result['candidate_processes_started'], 0)
        self.assertTrue(all(result[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))

    def test_changed_missing_added_and_reordered_edges(self):
        for edges in (self.expected[:-1], self.expected + self.expected[:1], list(reversed(self.expected))):
            self.lock['candidate_edges'] = copy.deepcopy(edges); self.freeze(); self.rejected()

    def test_identical_bytes_at_distinct_paths_remain_competing(self):
        self.provider.write_bytes(self.bundle.read_bytes())
        path = str(self.directory_alias / self.alias.name)
        self.lock['files'][path].update(sha256=fixture.sha(self.provider),
            declarations=catalog.loader.system.elf.inspect_elf_inputs(self.provider.read_bytes(), fixture.sha(self.provider)))
        self.lock['candidate_edges'][-1]['origin']['sha256'] = fixture.sha(self.provider)
        self.freeze()
        self.assertEqual(self.verify()['competing_provider_identity_edges'], 3)

    def test_numeric_and_origin_identity_mismatches(self):
        for field, value in [('ordinal', 0.0), ('ordinal', False), ('kind', 'runpath'),
                             ('declared_value', 'invented.so'), ('origin', {'path': 'wrong', 'sha256': '0' * 64})]:
            with self.subTest(field=field):
                self.lock['candidate_edges'] = copy.deepcopy(self.expected)
                self.lock['candidate_edges'][0][field] = value
                self.freeze(); self.rejected()

    def test_candidate_omission_order_duplicate_and_unbound_path(self):
        for paths in ([str(self.bundle)], list(reversed(self.expected[0]['candidate_paths'])),
                      self.expected[0]['candidate_paths'] * 2, ['/opaque/unbound']):
            self.lock['candidate_edges'] = copy.deepcopy(self.expected)
            self.lock['candidate_edges'][0]['candidate_paths'] = paths
            self.freeze(); self.rejected()

    def test_root_catalog_omission_even_if_outer_digest_refreshed(self):
        row = json.loads(self.unresolved.read_bytes()); row['declared_edges'].pop()
        self.unresolved.write_text(json.dumps(row)); self.lock['unresolved_sha256'] = fixture.sha(self.unresolved)
        self.freeze(); self.rejected()

    def test_root_owned_bytes_and_declaration_replay(self):
        self.root.write_bytes(b'changed opaque root'); self.rejected()
        self.root.write_bytes(opaque_provider(1))
        self.objects[0]['declarations']['declarations']['needed'] = []
        self.freeze_parents(); self.freeze(); self.rejected()

    def test_transcript_digest_before_decoding_and_replay(self):
        self.stdout.write_bytes(b'\xffprivate unbound transcript')
        with patch.object(catalog, 'replay_edges', side_effect=AssertionError('unbound transcript parsed')):
            self.rejected()

    def test_external_builtin_digest_precedes_json(self):
        class Permissive(str):
            def __eq__(self, other):
                return True
        with patch.object(catalog.loader.system.json, 'loads', side_effect=AssertionError('unbound JSON parsed')):
            self.rejected(loader_sha256=Permissive(self.digest))

    def test_rebound_cache_transcript_changes_candidate_order(self):
        original = self.stdout.read_text()
        self.stdout.write_text(original.replace('x86-64', 'AArch64'))
        self.cache_row['stdout_sha256'] = fixture.sha(self.stdout)
        self.cache_receipt.write_text(json.dumps(self.cache_row))
        self.lock['cache_receipt_sha256'] = fixture.sha(self.cache_receipt)
        self.freeze(); self.rejected()

    def test_missing_recursive_provider_binding(self):
        del self.lock['files'][str(self.directory_alias / self.alias.name)]
        self.freeze(); self.rejected()

    def test_direct_absolute_dependencies_and_interpreter_seed_order(self):
        provider = str(self.directory_alias / self.alias.name)
        origin = {'path': str(self.root), 'sha256': fixture.sha(self.root)}
        roots = [dict(origin=origin, kind='interpreter', ordinal=0, declared_value=provider),
                 dict(origin=origin, kind='needed', ordinal=0, declared_value=provider)]
        edges = catalog.replay_edges(self.objects, roots, self.lock['files'], self.stdout.read_bytes())
        self.assertEqual(edges[0], dict(roots[0], candidate_paths=[provider]))
        self.assertEqual(edges[1]['origin']['path'], provider)
        self.assertEqual(edges[2], dict(roots[1], candidate_paths=[provider]))
        self.assertTrue(all(path in self.lock['files'] for edge in edges for path in edge['candidate_paths']))

    def test_relative_dependency_path_and_dynamic_tokens_rejected(self):
        origin = {'path': str(self.root), 'sha256': fixture.sha(self.root)}
        for name in ('relative/provider.so', '$ORIGIN/provider.so', '/opaque/$LIB/provider.so'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                catalog.replay_edges(self.objects, [dict(origin=origin, kind='needed', ordinal=0,
                    declared_value=name)], self.lock['files'], self.stdout.read_bytes())

    def test_duplicate_cache_path_keeps_first_occurrence(self):
        raw = self.stdout.read_bytes()
        line = raw.splitlines()[1] + b'\n'
        self.assertEqual(catalog.replay_edges(self.objects, self.roots, self.lock['files'], raw + line),
                         self.expected)

    def test_additional_root_loader_declarations_rejected(self):
        self.root.write_bytes(bytes(opaque_object()))
        self.objects[0].update(expected_sha256=fixture.sha(self.root),
            declarations=catalog.loader.system.elf.inspect_elf_inputs(self.root.read_bytes(), fixture.sha(self.root)))
        self.freeze_parents(); self.freeze(); self.rejected()

    def test_policy_and_gate_claims_rejected(self):
        for key, value in [('candidate_enumeration_policy', 'actual glibc resolution'),
                           ('execution_admitted', True), ('mfs_v2', .99)]:
            old = self.lock[key]; self.lock[key] = value
            self.freeze(); self.rejected(); self.lock[key] = old

    def test_bounded_recursive_enumeration(self):
        with patch.object(catalog, 'MAX_EDGES', 2):
            self.rejected()

    def test_final_rehash_rejects_root_parent_and_provider_mutations(self):
        original = catalog.replay_edges
        for path in (self.root, self.provider, self.unresolved, self.stdout):
            with self.subTest(path=path.name):
                old = path.read_bytes()
                def changed(*args):
                    result = original(*args); path.write_bytes(b'changed after replay'); return result
                with patch.object(catalog, 'replay_edges', side_effect=changed):
                    self.rejected()
                path.write_bytes(old)


if __name__ == '__main__':
    unittest.main()
