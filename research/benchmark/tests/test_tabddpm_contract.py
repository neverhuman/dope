"""Frozen evidence and author-native contract checks without optional libraries."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import publish_tabddpm_contract as publisher
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
