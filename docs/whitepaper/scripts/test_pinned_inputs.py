"""Registered-input reads refuse every unregistered, redirected, or changed body before it is decoded."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from pinned_inputs import PinRefused, check_name, present, read_pinned

BODY = b'{"secret_value": 31337}\n'


def pin(body: bytes) -> tuple[int, str]:
    return len(body), hashlib.sha256(body).hexdigest()


class PinnedInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "results"
        self.root.mkdir()
        (self.root / "ledger.json").write_bytes(BODY)
        self.registry = {"ledger.json": pin(BODY)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_registered_file_returns_exactly_its_declared_bytes(self):
        self.assertEqual(read_pinned(self.root, "ledger.json", self.registry), BODY)
        (self.root / "sub").mkdir()
        (self.root / "sub" / "nested.json").write_bytes(BODY)
        self.assertEqual(read_pinned(self.root, "sub/nested.json", {"sub/nested.json": pin(BODY)}), BODY)

    def test_unsafe_or_unregistered_names_refuse_before_any_filesystem_call(self):
        for name in ("", "/etc/passwd", "../x.json", "sub/../ledger.json", "a\\b.json", ".", "other.json"):
            with self.subTest(name), self.assertRaises(PinRefused):
                read_pinned(self.root, name, self.registry)
        with self.assertRaises(PinRefused):
            check_name("../results-other/ledger.json")
        with self.assertRaises(PinRefused):
            present(self.root, "../ledger.json")

    def test_a_results_other_sibling_is_never_inside_results(self):
        sibling = self.base / "results-other"
        sibling.mkdir()
        (sibling / "ledger.json").write_bytes(BODY)
        with self.assertRaisesRegex(PinRefused, "traverses"):
            read_pinned(self.root, "../results-other/ledger.json", {"../results-other/ledger.json": pin(BODY)})

    def test_symlinked_leaves_ancestors_and_roots_refuse_even_with_matching_bytes(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "ledger.json").write_bytes(BODY)
        (self.root / "leaf.json").symlink_to(outside / "ledger.json")
        (self.root / "redirect").symlink_to(outside, target_is_directory=True)
        cases = {"leaf.json": self.root, "redirect/ledger.json": self.root, "ledger.json": self.base / "alias"}
        (self.base / "alias").symlink_to(self.root, target_is_directory=True)
        for name, root in cases.items():
            with self.subTest(name), self.assertRaisesRegex(PinRefused, "symlink"):
                read_pinned(root, name, {name: pin(BODY)})

    def test_a_changed_length_or_digest_refuses_and_names_no_content(self):
        changed = BODY.replace(b"31337", b"31338")
        (self.root / "ledger.json").write_bytes(changed)
        with self.assertRaisesRegex(PinRefused, "digest differs") as caught:
            read_pinned(self.root, "ledger.json", self.registry)
        self.assertNotIn("3133", str(caught.exception))
        (self.root / "ledger.json").write_bytes(BODY + b" ")
        with self.assertRaisesRegex(PinRefused, "length differs"):
            read_pinned(self.root, "ledger.json", self.registry)

    def test_directories_missing_files_and_malformed_entries(self):
        (self.root / "folder.json").mkdir()
        with self.assertRaisesRegex(PinRefused, "regular file"):
            read_pinned(self.root, "folder.json", {"folder.json": pin(BODY)})
        with self.assertRaises(FileNotFoundError):
            read_pinned(self.root, "gone.json", {"gone.json": pin(BODY)})
        self.assertFalse(present(self.root, "gone.json"))
        for entry in ((0, "0" * 64), (len(BODY), "short"), ("24", "0" * 64)):
            with self.subTest(entry), self.assertRaisesRegex(PinRefused, "malformed"):
                read_pinned(self.root, "ledger.json", {"ledger.json": entry})

    def test_the_body_is_read_once_through_the_descriptor(self):
        calls = []
        real = os.read

        def counting(descriptor, size):
            calls.append(size)
            return real(descriptor, size)

        os.read = counting
        try:
            self.assertEqual(read_pinned(self.root, "ledger.json", self.registry), BODY)
        finally:
            os.read = real
        self.assertEqual(sum(1 for size in calls if size), 2)
        self.assertLessEqual(max(calls), len(BODY) + 1)


if __name__ == "__main__":
    unittest.main()
