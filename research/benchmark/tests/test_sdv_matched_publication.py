"""Hermetic verified-byte and complete paired-panel publication controls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import publish_sdv_matched as p


class VerifiedBytes(unittest.TestCase):
    def setUp(self):
        # Fixture output stays in the lane target; no rows or weights are used.
        root = Path('target/sdv-publication-controls'); root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.json'
        self.data = b'{"status":"frozen"}'
        self.path.write_bytes(self.data)
        self.sha = hashlib.sha256(self.data).hexdigest()

    def test_committed_reference_hash_before_parse(self):
        with patch.object(p.json, 'loads', side_effect=AssertionError('parser must not execute')):
            with self.assertRaises(ValueError): p.committed(self.path, '0' * 64)

    def test_exact_verified_buffer_parsed(self):
        with patch.object(Path, 'read_bytes', return_value=b'{"status":"changed"}'):
            with self.assertRaises(ValueError): p.committed(self.path, self.sha)

    def test_duplicate_json_keys_rejected(self):
        data = b'{"status":"frozen","status":"changed"}'
        self.path.write_bytes(data)
        with self.assertRaises(ValueError): p.committed(self.path, hashlib.sha256(data).hexdigest())

    def test_nonfinite_json_rejected(self):
        for data in (b'{"value":NaN}', b'{"value":1e999}'):
            with self.subTest(data=data):
                self.path.write_bytes(data)
                with self.assertRaises(ValueError): p.committed(self.path, hashlib.sha256(data).hexdigest())

    def test_digest_subclass_rejected(self):
        class Digest(str): pass
        with self.assertRaises(ValueError): p.committed(self.path, Digest(self.sha))

    def test_linked_reference_rejected(self):
        link = self.path.with_name('alias.json'); link.symlink_to(self.path.name)
        with self.assertRaises(ValueError): p.committed(link, self.sha)

    def test_scratch_anchor_rejected_before_metric_reads(self):
        with patch.object(p.inputs, 'bound_json', side_effect=ValueError('changed anchor')) as read:
            with patch.object(p.inputs, 'verify_refs', side_effect=AssertionError('must not execute')):
                with self.assertRaises(ValueError): p.build()
        self.assertEqual(read.call_args.args, (p.ROOT / 'receipt-lock-v1.json', p.RECEIPTS))

    def test_verified_reference_accepts_frozen_metadata(self):
        self.assertEqual(p.committed(self.path, self.sha), {'status': 'frozen'})


if __name__ == '__main__': unittest.main()
