"""Hermetic publication rejection and deterministic output controls."""
import json
from pathlib import Path
import tempfile
import unittest

from research.benchmark import publish_dope_refinement_progress as publish


class RefinementProgress(unittest.TestCase):
    def setUp(self):
        Path('target').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir='target')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.file = self.root / 'receipt.json'
        self.file.write_text('{"status":"ok"}')
        self.expected = publish.sha(self.file)

    def test_bound_receipt(self):
        self.assertEqual(publish.bound(self.file, self.expected), {'status': 'ok'})

    def test_rewritten_receipt_rejected(self):
        self.file.write_text('{"status":"changed"}')
        with self.assertRaises(ValueError): publish.bound(self.file, self.expected)

    def test_digest_subclass_rejected(self):
        class Digest(str):
            def __eq__(self, value): return True
        with self.assertRaises(ValueError): publish.bound(self.file, Digest(self.expected))

    def test_linked_receipt_rejected(self):
        link = self.root / 'linked.json'; link.symlink_to(self.file)
        with self.assertRaises(ValueError): publish.bound(link, self.expected)

    def test_linked_parent_rejected(self):
        link = self.root / 'alias'; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): publish.bound(link / self.file.name, self.expected)

    def source(self):
        source = self.root / 'source'; source.mkdir()
        entry = source / 'entry.py'; entry.write_text('# opaque original source\n')
        return source, {'source_files': {str(entry): publish.sha(entry)}}

    def test_source_inventory(self):
        _, lock = self.source(); publish.verify_sources(lock)

    def test_added_bytecode_rejected(self):
        source, lock = self.source(); (source / 'added.pyc').write_bytes(b'opaque bytecode')
        with self.assertRaises(ValueError): publish.verify_sources(lock)

    def test_directory_alias_rejected(self):
        source, lock = self.source(); (source / 'alias').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): publish.verify_sources(lock)

    def test_changed_source_rejected(self):
        source, lock = self.source(); (source / 'entry.py').write_text('# changed\n')
        with self.assertRaises(ValueError): publish.verify_sources(lock)

    def test_canonical_integer_identity(self):
        self.assertNotEqual(publish.identity({'seed': 11}), publish.identity({'seed': 11.0}))

    def test_tables_deterministic_and_no_gated_score(self):
        cell = dict(method='DOPE', configuration='frozen', size_multiplier=4, charged_artifact_bytes=100,
                    utility={a: {'retention': .75} for a in ('catboost', 'linear', 'mlp')})
        report = dict(scope='Closed progress batch; full matrix pending.', cells=[cell] * 3, matched_reference_cells=[])
        self.assertEqual(publish.tables(report), publish.tables(json.loads(json.dumps(report))))
        csv, md = publish.tables(report)
        self.assertIn('DOPE,frozen,4,3,100,0.75,0.75,0.75', csv)
        self.assertIn('MFS-v2/PTF-v1/release/superiority stay null', md)


if __name__ == '__main__': unittest.main()
