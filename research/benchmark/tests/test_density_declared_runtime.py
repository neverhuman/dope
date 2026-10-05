"""Small opaque file-inventory controls, no real runtime inventory scanned."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import density_declared_runtime_replay as m


class Controls(unittest.TestCase):
    def setUp(self):
        area_root = Path.cwd() / 'target' / 'density-runtime-controls'
        area_root.mkdir(parents=True, exist_ok=True)
        self.area = tempfile.TemporaryDirectory(dir=area_root)
        self.addCleanup(self.area.cleanup)
        self.root = Path(self.area.name).resolve()
        p = patch.object(m.inputs, 'BASE', self.root); p.start(); self.addCleanup(p.stop)
        self.files = {}
        for name in ('module.py', '__pycache__/module.pyc'):
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'opaque fixture')
            self.files[str(path)] = dict(bytes=14, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_regular_sources_and_bytecode_hashed(self):
        self.assertEqual(set(m.files_under(self.root, self.files)), set(self.files))

    def test_added_regular_source_rejected(self):
        (self.root / 'added.py').write_bytes(b'opaque')
        with self.assertRaises(ValueError): m.files_under(self.root, self.files)

    def test_changed_bytecode_rejected(self):
        (self.root / '__pycache__/module.pyc').write_bytes(b'changed')
        with self.assertRaises(ValueError): m.files_under(self.root, self.files)

    def test_directory_alias_rejected(self):
        (self.root / 'alias').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): m.files_under(self.root, self.files)

    def test_special_entry_rejected(self):
        os.mkfifo(self.root / 'special')
        with self.assertRaises(ValueError): m.files_under(self.root, self.files)

    def test_bytes_boolean_rejected(self):
        self.files[str(self.root / 'module.py')]['bytes'] = True
        with self.assertRaises(ValueError): m.files_under(self.root, self.files)

    def test_digest_subclass_rejected(self):
        class Derived(str): pass
        row = self.files[str(self.root / 'module.py')]
        row['sha256'] = Derived(row['sha256'])
        with self.assertRaises(ValueError): m.files_under(self.root, self.files)

    def test_system_file_wrong_digest_rejected_without_execution(self):
        path = self.root / 'module.py'
        with self.assertRaises(ValueError): m.rehash_system_file(path, '0' * 64, 14)
        m.rehash_system_file(path, self.files[str(path)]['sha256'], 14)


if __name__ == '__main__': unittest.main(verbosity=2)
