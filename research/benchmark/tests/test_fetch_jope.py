import os
import tempfile
import unittest
from pathlib import Path

from research.benchmark.fetch_jope import used_bytes


ROOT = Path(__file__).resolve().parents[3]


class ScratchAccountingTests(unittest.TestCase):
    def test_broken_and_external_symlinks_charge_only_link_bytes(self):
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            base = Path(directory)
            scratch = base / "scratch"
            scratch.mkdir()
            (scratch / "artifact.bin").write_bytes(b"12345")
            outside = base / "outside.bin"
            outside.write_bytes(b"x" * 1000)
            external = scratch / "external.bin"
            broken = scratch / "broken.bin"
            linked_dir = scratch / "linked-dir"
            external.symlink_to(outside)
            broken.symlink_to(base / "removed.bin")
            linked_dir.symlink_to(base, target_is_directory=True)
            expected = 5 + sum(len(os.readlink(path)) for path in
                               (external, broken, linked_dir))
            self.assertEqual(used_bytes(scratch), expected)


if __name__ == "__main__":
    unittest.main()
