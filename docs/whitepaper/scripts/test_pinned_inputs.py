"""Registered-input reads refuse every unregistered, redirected, or changed body before it is decoded."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import pinned_inputs
from pinned_inputs import PinRefused, Registry, check_name, present, read_pinned

BODY = b'{"secret_value": 31337}\n'


def pin(body: bytes) -> tuple[int, str]:
    return len(body), hashlib.sha256(body).hexdigest()


class PinnedInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(os.path.realpath(self.tmp.name))
        self.root = self.base / "results"
        self.root.mkdir()
        (self.root / "ledger.json").write_bytes(BODY)
        self.registry = Registry(self.root, {"ledger.json": pin(BODY)})

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_registered_file_returns_exactly_its_declared_bytes(self):
        self.assertEqual(read_pinned(self.registry, "ledger.json"), BODY)
        (self.root / "sub").mkdir()
        (self.root / "sub" / "nested.json").write_bytes(BODY)
        self.assertEqual(read_pinned(Registry(self.root, {"sub/nested.json": pin(BODY)}), "sub/nested.json"), BODY)

    def test_unsafe_or_unregistered_names_refuse_before_any_filesystem_call(self):
        for name in ("", "/etc/passwd", "../x.json", "sub/../ledger.json", "a\\b.json", ".", "other.json"):
            with self.subTest(name), self.assertRaises(PinRefused):
                read_pinned(self.registry, name)
        with self.assertRaises(PinRefused):
            check_name("../results-other/ledger.json")
        with self.assertRaises(PinRefused):
            present(self.registry, "../ledger.json")

    def test_a_results_other_sibling_is_never_inside_results(self):
        sibling = self.base / "results-other"
        sibling.mkdir()
        (sibling / "ledger.json").write_bytes(BODY)
        name = "../results-other/ledger.json"
        with self.assertRaisesRegex(PinRefused, "traverses"):
            read_pinned(Registry(self.root, {name: pin(BODY)}), name)

    def test_symlinked_leaves_ancestors_and_roots_refuse_even_with_matching_bytes(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "ledger.json").write_bytes(BODY)
        (self.root / "leaf.json").symlink_to(outside / "ledger.json")
        (self.root / "redirect").symlink_to(outside, target_is_directory=True)
        (self.base / "alias").symlink_to(self.root, target_is_directory=True)
        cases = {"leaf.json": self.root, "redirect/ledger.json": self.root, "ledger.json": self.base / "alias"}
        for name, root in cases.items():
            with self.subTest(name), self.assertRaisesRegex(PinRefused, "symlink"):
                read_pinned(Registry(root, {name: pin(BODY)}), name)

    def test_an_ancestor_swapped_mid_walk_cannot_redirect_the_read(self):
        (self.root / "sub").mkdir()
        (self.root / "sub" / "ledger.json").write_bytes(BODY)
        decoy = self.base / "decoy"
        decoy.mkdir()
        (decoy / "ledger.json").write_bytes(BODY.replace(b"31337", b"99999"))
        real_open, real_lstat, swapped = os.open, os.lstat, []

        def swap_after(call):
            def wrapped(path, *args, **kwargs):
                result = call(path, *args, **kwargs)
                if not swapped and os.fspath(path) in ("sub", str(self.root / "sub")):
                    # The reader has just checked or opened "sub"; replace it with a link to a decoy.
                    swapped.append(True)
                    os.rename(self.root / "sub", self.root / "sub-held")
                    (self.root / "sub").symlink_to(decoy, target_is_directory=True)
                return result
            return wrapped

        pinned_inputs.os.open, pinned_inputs.os.lstat = swap_after(real_open), swap_after(real_lstat)
        try:
            body = read_pinned(Registry(self.root, {"sub/ledger.json": pin(BODY)}), "sub/ledger.json")
        finally:
            pinned_inputs.os.open, pinned_inputs.os.lstat = real_open, real_lstat
        self.assertEqual(swapped, [True])
        self.assertEqual(body, BODY)
        self.assertTrue((self.root / "sub").is_symlink())

    def test_a_changed_length_or_digest_refuses_and_names_no_content(self):
        changed = BODY.replace(b"31337", b"31338")
        (self.root / "ledger.json").write_bytes(changed)
        with self.assertRaisesRegex(PinRefused, "digest differs") as caught:
            read_pinned(self.registry, "ledger.json")
        self.assertNotIn("3133", str(caught.exception))
        (self.root / "ledger.json").write_bytes(BODY + b" ")
        with self.assertRaisesRegex(PinRefused, "length differs"):
            read_pinned(self.registry, "ledger.json")

    def test_directories_fifos_missing_files_and_malformed_entries(self):
        (self.root / "folder.json").mkdir()
        os.mkfifo(self.root / "pipe.json")
        for name in ("folder.json", "pipe.json"):
            with self.subTest(name), self.assertRaisesRegex(PinRefused, "regular file"):
                read_pinned(Registry(self.root, {name: pin(BODY)}), name)
        gone = Registry(self.root, {"gone.json": pin(BODY)})
        with self.assertRaises(FileNotFoundError):
            read_pinned(gone, "gone.json")
        self.assertFalse(present(gone, "gone.json"))
        for entry in ((0, "0" * 64), (len(BODY), "short"), ("24", "0" * 64)):
            with self.subTest(entry), self.assertRaisesRegex(PinRefused, "malformed"):
                read_pinned(Registry(self.root, {"ledger.json": entry}), "ledger.json")

    def test_every_component_is_opened_beneath_a_held_descriptor_without_following_links(self):
        calls = []
        real_open = os.open

        def recording_open(path, flags, *args, dir_fd=None, **kwargs):
            calls.append((path, flags, dir_fd))
            return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

        pinned_inputs.os.open = recording_open
        try:
            self.assertEqual(read_pinned(self.registry, "ledger.json"), BODY)
        finally:
            pinned_inputs.os.open = real_open
        self.assertEqual(calls[0][0], "/")
        components = [*self.root.parts[1:], "ledger.json"]
        self.assertEqual([path for path, _flags, _fd in calls[1:]], components)
        self.assertTrue(all(fd is not None and flags & os.O_NOFOLLOW for _path, flags, fd in calls[1:]))


if __name__ == "__main__":
    unittest.main()
