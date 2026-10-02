"""Frozen evidence and author-native contract checks without optional libraries."""

import copy
import json
import importlib.util
import marshal
from pathlib import Path
import py_compile
import tempfile
import types
import unittest
from unittest.mock import patch

from research.benchmark import publish_tabddpm_contract as publisher
from research.benchmark import tabddpm_bytecode
from research.benchmark.score import sha256


class TabDDPMContractTests(unittest.TestCase):
    def setUp(self):
        target = Path('target/tabddpm-contract-tests')
        target.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=target)
        self.root = Path(self.directory.name).resolve()

    def tearDown(self):
        self.directory.cleanup()

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def custody(self, paths):
        return self.put('custody.json', {
            'official_tests_opened': False, 'new_fits_started': 0,
            'files': [{'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in paths]})

    def test_changed_receipt_and_sources_fail_before_consumption(self):
        for name in ('receipt.json', 'source/adapter.py', 'bin/generator.so',
                     'worker/train.csv', 'worker/validation.csv', 'artifact/model.json'):
            with self.subTest(name=name):
                path = self.put(name, {'fixture': 1})
                lock = self.custody([path])
                expected = sha256(lock)
                path.write_text('{"fixture": 2}')
                reads = []
                original = publisher.read
                def read(item):
                    reads.append(item)
                    return original(item)
                with patch.object(publisher, 'read', read):
                    with self.assertRaisesRegex(ValueError, 'original TabDDPM evidence changed'):
                        publisher.verify_custody(lock, expected, self.root)
                self.assertEqual(reads, [lock])

    def test_rewritten_custody_cannot_reanchor_evidence(self):
        path = self.put('receipt.json', {'fixture': 1})
        lock = self.custody([path])
        expected = sha256(lock)
        path.write_text('{"fixture": 2}')
        self.custody([path])
        with patch.object(publisher, 'read') as read:
            with self.assertRaisesRegex(ValueError, 'custody lock changed'):
                publisher.verify_custody(lock, expected, self.root)
            read.assert_not_called()

    def test_artifact_root_link_rejected_without_hashing_artifact(self):
        retained = self.put('retained/model.json', {'fixture': 1})
        (self.root / 'artifact').symlink_to(retained.parent, target_is_directory=True)
        link = self.root / 'artifact/model.json'
        lock = self.custody([link])
        expected = sha256(lock)
        hashes = []
        def hash_file(path):
            hashes.append(path)
            return sha256(path)
        with patch.object(publisher, 'sha256', hash_file):
            with self.assertRaisesRegex(ValueError, 'evidence contains link'):
                publisher.verify_custody(lock, expected, self.root)
        self.assertEqual(hashes, [lock])

    def test_sealed_partition_rejected_without_hashing_rows(self):
        path = self.put('evaluator/test.csv', {'fixture': 1})
        lock = self.custody([path])
        expected = sha256(lock)
        hashes = []
        def hash_file(path):
            hashes.append(path)
            return sha256(path)
        with patch.object(publisher, 'sha256', hash_file):
            with self.assertRaisesRegex(ValueError, 'outside sealed scope'):
                publisher.verify_custody(lock, expected, self.root)
        self.assertEqual(hashes, [lock])

    def test_added_author_or_runtime_source_rejected(self):
        environment, author = self.root / 'environment', self.root / 'author'
        library = self.put('environment/library.py', {'fixture': 1})
        source = self.put('author/source.py', {'fixture': 1})
        files = {str(library): sha256(library), str(source): sha256(source)}
        publisher.verify_source_inventory(files, environment, author)
        for parent in (environment, author):
            extra = parent / 'added.py'
            extra.touch()
            try:
                with self.assertRaisesRegex(ValueError, 'file inventory changed'):
                    publisher.verify_source_inventory(files, environment, author)
            finally:
                extra.unlink()

    def test_directory_and_file_aliases_rejected_in_both_inventories(self):
        environment, author = self.root / 'environment', self.root / 'author'
        library = self.put('environment/library.py', {'fixture': 1})
        source = self.put('author/source.py', {'fixture': 1})
        external = self.put('outside/package/__init__.py', {'fixture': 1})
        files = {str(library): sha256(library), str(source): sha256(source)}
        for root in (environment, author):
            for target in (external, external.parent):
                alias = root / 'added_package'
                alias.symlink_to(target, target_is_directory=target.is_dir())
                try:
                    with self.subTest(root=root, target=target), self.assertRaisesRegex(
                            ValueError, 'inventory contains unpinned link'):
                        publisher.verify_source_inventory(files, environment, author)
                finally:
                    alias.unlink()

    def test_exact_pinned_abi_alias_cannot_admit_another_link(self):
        environment, author = self.root / 'environment', self.root / 'author'
        library = self.put('environment/lib/library.py', {'fixture': 1})
        source = self.put('author/source.py', {'fixture': 1})
        alias = environment / 'lib64'
        alias.symlink_to('lib', target_is_directory=True)
        files = {str(library): sha256(library), str(source): sha256(source)}
        aliases = {str(alias): {'symlink_target': 'lib', 'resolved_path': str(library.parent)}}
        publisher.verify_source_inventory(files, environment, author, aliases)
        alias.unlink()
        alias.symlink_to(author, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'inventory contains unpinned link'):
            publisher.verify_source_inventory(files, environment, author, aliases)
        alias.unlink()
        with self.assertRaisesRegex(ValueError, 'pinned alias inventory changed'):
            publisher.verify_source_inventory(files, environment, author, aliases)

    def cache(self, source):
        path = Path(importlib.util.cache_from_source(str(source)))
        py_compile.compile(str(source), cfile=str(path), doraise=True)
        row = {'path': str(path), 'sha256': sha256(path), 'source_path': str(source),
               'source_sha256': sha256(source)}
        return path, row

    def test_pinned_cache_equivalence_and_hash_checked_without_execution(self):
        marker = self.root / 'initializer-executed'
        source = self.root / 'library.py'
        source.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
        cache, row = self.cache(source)
        self.assertTrue(tabddpm_bytecode.verify([row])[0]['code_equivalent'])
        self.assertFalse(marker.exists())
        cache.write_bytes(cache.read_bytes() + b'changed')
        with patch.object(tabddpm_bytecode.marshal, 'loads') as decode:
            with self.assertRaisesRegex(ValueError, 'cache or pinned source changed'):
                tabddpm_bytecode.verify([row])
            decode.assert_not_called()
        self.assertFalse(marker.exists())

    def test_header_valid_malicious_cache_rejected_even_with_frozen_cache_hash(self):
        marker = self.root / 'initializer-executed'
        source = self.root / 'library.py'
        source.write_text('answer = 42\n')
        cache, row = self.cache(source)
        malicious = compile(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n',
                            str(source), 'exec')
        cache.write_bytes(cache.read_bytes()[:16] + marshal.dumps(malicious))
        row['sha256'] = sha256(cache)
        with self.assertRaisesRegex(ValueError, 'cache differs from pinned source'):
            tabddpm_bytecode.verify([row])
        self.assertFalse(marker.exists())

    def test_added_executable_cache_rejected_in_both_inventories(self):
        environment, author = self.root / 'environment', self.root / 'author'
        library = self.put('environment/library.py', {'fixture': 1})
        source = self.put('author/source.py', {'fixture': 1})
        files = {str(library): sha256(library), str(source): sha256(source)}
        for root in (environment, author):
            extra = root / 'added.pyc'
            extra.touch()
            try:
                with self.subTest(root=root), self.assertRaisesRegex(ValueError, 'file inventory changed'):
                    publisher.verify_source_inventory(files, environment, author)
            finally:
                extra.unlink()

    def test_equal_numeric_values_with_different_constant_types_are_rejected(self):
        source = self.root / 'library.py'
        source.write_text('answer = 1\n')
        cache, row = self.cache(source)
        original = compile(source.read_bytes(), str(source), 'exec')
        changed = original.replace(co_consts=tuple(True if type(x) is int and x == 1 else x
                                                   for x in original.co_consts))
        cache.write_bytes(cache.read_bytes()[:16] + marshal.dumps(changed))
        row['sha256'] = sha256(cache)
        with self.assertRaisesRegex(ValueError, 'cache differs from pinned source'):
            tabddpm_bytecode.verify([row])

    def test_changed_cache_blocks_build_before_metric_reads_or_verifier_launch(self):
        environment = self.root / 'environment'
        environment.mkdir()
        source = environment / 'library.py'
        source.write_text('answer = 42\n')
        cache, row = self.cache(source)
        row['bytes'] = cache.stat().st_size
        library = self.root / 'interpreter/lib'
        bootstrap = self.put('interpreter/lib/bootstrap.py', {'fixture': 1})
        lock = self.put('executable-custody.json', {
            'original_custody': {'path': str(publisher.CUSTODY), 'sha256': publisher.CUSTODY_SHA256},
            'historical_fit_executable_closure_verified': False, 'aliases': [], 'caches': [row],
            'interpreter_library': {'root': str(library), 'aliases': [], 'files': [
                {'path': str(bootstrap), 'bytes': bootstrap.stat().st_size, 'sha256': sha256(bootstrap)}]}})
        expected = sha256(lock)
        cache.write_bytes(cache.read_bytes() + b'changed')
        reads = []
        original_read = publisher.read
        def read(path):
            reads.append(path)
            return original_read(path)
        with patch.object(publisher, 'ENV', environment), \
                patch.object(publisher, 'INTERPRETER_LIB', library), \
                patch.object(publisher, 'EXECUTABLE_CUSTODY', lock), \
                patch.object(publisher, 'EXECUTABLE_CUSTODY_SHA256', expected), \
                patch.object(publisher, 'verify_custody', return_value={str(source): sha256(source)}), \
                patch.object(publisher, 'read', read), \
                patch.object(publisher.subprocess, 'run') as launch:
            with self.assertRaisesRegex(ValueError, 'executable cache or source changed'):
                publisher.build()
            launch.assert_not_called()
        self.assertEqual(reads, [lock])

    def test_signed_zero_and_nested_numeric_bits_cannot_match_pinned_source(self):
        def change(code):
            def constant(value):
                if isinstance(value, types.CodeType):
                    return change(value)
                if type(value) is float and value == 0.0:
                    return -0.0
                if type(value) is complex and value == 0j:
                    return complex(-0.0, -0.0)
                if isinstance(value, tuple):
                    return tuple(constant(x) for x in value)
                if isinstance(value, frozenset):
                    return frozenset(constant(x) for x in value)
                return value
            return code.replace(co_consts=tuple(constant(x) for x in code.co_consts))
        for code in ('answer = 0.0\n', 'answer = 0j\n', 'answer = (0.0, 0j)\n',
                     'def outer():\n    def inner():\n        return 0.0\n    return inner\n'):
            with self.subTest(code=code):
                source = self.root / 'library.py'
                source.write_text(code)
                cache, row = self.cache(source)
                compiled = compile(source.read_bytes(), str(source), 'exec')
                cache.write_bytes(cache.read_bytes()[:16] + marshal.dumps(change(compiled)))
                row['sha256'] = sha256(cache)
                with self.assertRaisesRegex(ValueError, 'cache differs from pinned source'):
                    tabddpm_bytecode.verify([row])

    def test_interpreter_source_cache_archive_and_alias_rejected_before_launch(self):
        library = self.root / 'interpreter/lib'
        bootstrap = self.put('interpreter/lib/bootstrap.py', {'fixture': 1})
        cache = self.put('interpreter/lib/__pycache__/bootstrap.cpython-39.pyc', {'fixture': 1})
        files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in (bootstrap, cache)]
        lock = self.put('executable-custody.json', {
            'original_custody': {'path': str(publisher.CUSTODY), 'sha256': publisher.CUSTODY_SHA256},
            'historical_fit_executable_closure_verified': False, 'aliases': [], 'caches': [],
            'interpreter_library': {'root': str(library), 'aliases': [], 'files': files}})
        expected = sha256(lock)
        for kind in ('source', 'cache', 'archive', 'extension', 'alias'):
            if kind in ('source', 'cache'):
                changed = bootstrap if kind == 'source' else cache
                original = changed.read_bytes()
                changed.write_bytes(original + b'changed')
            elif kind == 'alias':
                changed = library / 'added_package'
                changed.symlink_to(self.root, target_is_directory=True)
            else:
                changed = library / ('python39.zip' if kind == 'archive' else 'added.so')
                changed.touch()
            try:
                with self.subTest(kind=kind), patch.object(publisher, 'INTERPRETER_LIB', library), \
                        patch.object(publisher, 'EXECUTABLE_CUSTODY', lock), \
                        patch.object(publisher, 'EXECUTABLE_CUSTODY_SHA256', expected), \
                        patch.object(publisher.subprocess, 'run') as launch:
                    with self.assertRaisesRegex(ValueError, 'library|unpinned link'):
                        publisher.verify_executable_closure({})
                    launch.assert_not_called()
            finally:
                if kind in ('source', 'cache'):
                    changed.write_bytes(original)
                else:
                    changed.unlink()

    def test_verifier_bootstrap_uses_empty_cache_prefix_and_disables_cache_writes(self):
        environment, author, library = (self.root / name for name in ('environment', 'author', 'interpreter/lib'))
        dependency = self.put('environment/library.py', {'fixture': 1})
        source = self.put('author/source.py', {'fixture': 1})
        bootstrap = self.put('interpreter/lib/bootstrap.py', {'fixture': 1})
        lock = self.put('executable-custody.json', {
            'original_custody': {'path': str(publisher.CUSTODY), 'sha256': publisher.CUSTODY_SHA256},
            'historical_fit_executable_closure_verified': False, 'aliases': [], 'caches': [],
            'interpreter_library': {'root': str(library), 'aliases': [], 'files': [
                {'path': str(bootstrap), 'bytes': bootstrap.stat().st_size, 'sha256': sha256(bootstrap)}]}})
        expected = sha256(lock)
        observed = []
        def launch(argv, **kwargs):
            self.assertEqual(argv[1:5], ['-I', '-S', '-B', '-X'])
            self.assertTrue(argv[5].startswith('pycache_prefix='))
            prefix = Path(argv[5].split('=', 1)[1])
            self.assertTrue(prefix.is_dir())
            self.assertEqual(list(prefix.iterdir()), [])
            self.assertEqual(json.loads(kwargs['input']), [])
            observed.append(prefix)
            return types.SimpleNamespace(returncode=0, stdout='{"status":"ok","caches":[]}')
        with patch.object(publisher, 'ENV', environment), patch.object(publisher, 'AUTHOR', author), \
                patch.object(publisher, 'INTERPRETER_LIB', library), \
                patch.object(publisher, 'EXECUTABLE_CUSTODY', lock), \
                patch.object(publisher, 'EXECUTABLE_CUSTODY_SHA256', expected), \
                patch.object(publisher.subprocess, 'run', launch):
            _, claim = publisher.verify_executable_closure({str(p): sha256(p) for p in (dependency, source)})
        self.assertFalse(observed[0].exists())
        self.assertTrue(claim['current_declared_inventories_verified'])
        self.assertFalse(claim['current_executable_closure_verified'])
        self.assertFalse(claim['historical_fit_executable_closure_verified'])

    def test_native_five_seed_objective_rejects_partial_duplicate_or_shared_kpi(self):
        native = {'components': [{'sample_seed': s, 'r2': .5} for s in range(5)],
                  'name': 'author_five_synthetic_seed_validation_catboost_r2_mean',
                  'direction': 'maximize', 'partition': 'official_training_derived_validation',
                  'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None, 'value': .5}
        self.assertEqual(publisher.native_value(native), .5)
        for change in ('partial', 'duplicate', 'nonfinite', 'boolean', 'shared_kpi', 'wrong_mean'):
            row = copy.deepcopy(native)
            if change == 'partial':
                row['components'].pop()
            elif change == 'duplicate':
                row['components'][0]['sample_seed'] = 1
            elif change == 'nonfinite':
                row['components'][0]['r2'] = float('nan')
            elif change == 'boolean':
                row['components'][1]['sample_seed'] = True
            elif change == 'shared_kpi':
                row['name'] = 'mfs_v2'
            else:
                row['value'] = .6
            with self.subTest(change=change), self.assertRaises(ValueError):
                publisher.native_value(row)

    def test_alias_costs_count_once_and_failed_fit_costs_remain(self):
        sample = self.put('current/sample.operation.json', {'status': 'ok', 'elapsed_seconds': 2})
        inherited = self.put('inherited/sample.operation.json', {'status': 'ok', 'elapsed_seconds': 2})
        failed = self.put('failed/fit.operation.json', {'status': 'failed', 'elapsed_seconds': 7})
        aliases = [{'alias_path': str(inherited), 'physical_operation_path': str(sample)}]
        result = publisher.physical_operations([sample, inherited, failed], aliases)
        self.assertEqual(len(result), 2)
        self.assertEqual(sum(r['elapsed_seconds'] for r in result.values()), 9)
        inherited.write_text('{"status": "ok", "elapsed_seconds": 3}')
        with self.assertRaisesRegex(ValueError, 'inherited operation differs'):
            publisher.physical_operations([sample, inherited, failed], aliases)

    def test_contract_schema_cannot_promote_or_unseal(self):
        import jsonschema
        row = {'format': 'fixture', 'official_tests_opened': False, 'native_tuning_complete': False,
               'production_certified': False, 'counts_as_dope_win': False,
               'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None}
        contract = publisher.schema(row)
        jsonschema.validate(row, contract)
        for key in ('official_tests_opened', 'native_tuning_complete', 'production_certified',
                    'counts_as_dope_win', 'mfs_v2', 'ptf_v1', 'release_safe_l3'):
            with self.subTest(key=key), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(row | {key: True if row[key] is False else .99}, contract)


if __name__ == '__main__':
    unittest.main()
