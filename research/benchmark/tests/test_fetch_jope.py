import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.benchmark import fetch_jope, prepare_s3
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

    def test_directory_scan_error_fails_closed(self):
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            scratch = Path(directory)
            with patch("research.benchmark.fetch_jope.os.scandir",
                       side_effect=PermissionError("scan denied")):
                with self.assertRaises(PermissionError):
                    used_bytes(scratch)

    def test_missing_root_fails_closed(self):
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            with self.assertRaises(NotADirectoryError):
                used_bytes(Path(directory) / "missing")


class ExplicitBucketTests(unittest.TestCase):
    def test_missing_or_invalid_bucket_refuses_before_mkdir_or_aws_even_for_cached_objects(self):
        invalid = (None, "", "s3://private-source", "../private", "Uppercase",
                   "ab", "a" * 64, "a..b", "a.-b", "a-.b", "192.168.1.1",
                   "xn--reserved", "private--x-s3", "private\nsource", 17)
        root = Path("/mnt/dope-fetch-fixture")
        entries = (
            lambda bucket: fetch_jope.fetch_object("fixture", root / "data.csv", None, bucket=bucket),
            lambda bucket: fetch_jope.fetch("dataset", root, None, bucket=bucket),
            lambda bucket: prepare_s3.run(root, None, bucket=bucket),
            lambda bucket: prepare_s3.prepare_entry({}, root, None, bucket=bucket),
        )
        with patch.object(Path, "mkdir") as mkdir, \
                patch.object(Path, "exists", return_value=True), \
                patch.object(fetch_jope.subprocess, "run") as aws:
            for bucket in invalid:
                for entry in entries:
                    with self.subTest(bucket=bucket, entry=entries.index(entry)):
                        with self.assertRaises(ValueError) as error:
                            entry(bucket)
                        self.assertEqual(str(error.exception), "explicit supported S3 bucket name required")
                        mkdir.assert_not_called()
                        aws.assert_not_called()

    def test_explicit_bucket_preserves_optional_profile_and_bounded_aws_argv(self):
        destination = Path("/mnt/dope-fetch-fixture/data.csv")
        with patch.object(Path, "mkdir") as mkdir, \
                patch.object(Path, "exists", return_value=False), \
                patch.object(fetch_jope.subprocess, "run") as aws:
            for profile in (None, "user-supplied"):
                fetch_jope.fetch_object("fixture", destination, profile, bucket="fixture.bucket-name")
                command = ["aws"] + (["--profile", profile] if profile else [])
                aws.assert_called_with(command + ["s3", "cp", "s3://fixture.bucket-name/fixture",
                                                   str(destination), "--only-show-errors"],
                                       check=True, capture_output=True, text=True, timeout=300)
            self.assertEqual(mkdir.call_count, 2)

    @staticmethod
    def fixture():
        payload = b"synthetic fetch fixture"
        metadata = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        row = {"dataset_hash": "fixture", "origin": "real", "task": "regression",
               "provenance": {"license": {"status": "recorded", "spdx": "MIT",
                                           "evidence_url": "https://example.org/license"}},
               "files": {"train": metadata, "test": metadata, "meta": metadata}}
        return row, {"dataset_hash": "fixture", "files": row["files"]}

    @contextlib.contextmanager
    def catalog_fixture(self, row, manifest, *, size=0, catalog_digest=None, blob_digest=None,
                        blob_size=None):
        expected = row["files"]["train"]["sha256"]
        def digest(path):
            if path.name == "catalog.jsonl":
                return catalog_digest if catalog_digest is not None else fetch_jope.CATALOG_SHA256
            if path.name == "manifest.json":
                return "0" * 64
            return blob_digest if blob_digest is not None else expected
        with patch.object(fetch_jope, "fetch_object") as fetch_object, \
                patch.object(fetch_jope, "sha256", side_effect=digest), \
                patch.object(fetch_jope, "used_bytes", return_value=size), \
                patch.object(Path, "open", return_value=io.StringIO(json.dumps(row) + "\n")), \
                patch.object(Path, "read_text", return_value=json.dumps(manifest)), \
                patch.object(Path, "stat", return_value=SimpleNamespace(
                    st_size=row["files"]["train"]["bytes"] if blob_size is None else blob_size)):
            yield fetch_object

    def test_fetch_passes_explicit_bucket_only_to_selected_authenticated_objects(self):
        row, manifest = self.fixture()
        with self.catalog_fixture(row, manifest) as objects:
            receipt = fetch_jope.fetch("fixture", Path("/mnt/dope-fetch-fixture"), None,
                                       bucket="explicit-fixture-bucket")
        keys = [call.args[0] for call in objects.call_args_list]
        self.assertEqual(keys, ["catalog/v1/catalog.jsonl", "datasets/fixture/manifest.json"]
                         + [f"blobs/sha256/{row['files'][kind]['sha256']}" for kind in row["files"]])
        self.assertTrue(all(call.kwargs == {"bucket": "explicit-fixture-bucket"}
                            for call in objects.call_args_list))
        self.assertEqual(receipt["catalog_sha256"], fetch_jope.CATALOG_SHA256)
        self.assertEqual(set(receipt["files"]), {"train", "test", "meta"})

    def test_explicit_source_does_not_bypass_hash_rights_or_scratch_checks(self):
        for refusal in ("catalog", "rights", "budget", "manifest", "blob", "blob_size"):
            row, manifest = self.fixture()
            options = {}
            if refusal == "catalog":
                options["catalog_digest"] = "f" * 64
            elif refusal == "rights":
                row["provenance"]["license"]["status"] = "unknown"
            elif refusal == "budget":
                options["size"] = fetch_jope.LIMIT
            elif refusal == "manifest":
                manifest["dataset_hash"] = "different"
            elif refusal == "blob":
                options["blob_digest"] = "f" * 64
            else:
                options["blob_size"] = row["files"]["train"]["bytes"] + 1
            with self.subTest(refusal=refusal), self.catalog_fixture(row, manifest, **options), \
                    self.assertRaises(ValueError):
                fetch_jope.fetch("fixture", Path("/mnt/dope-fetch-fixture"), None,
                                 bucket="explicit-fixture-bucket")

    def test_prepare_wrapper_propagates_bucket_without_actual_preparation(self):
        root = prepare_s3.ROOT / "explicit-fetch-fixture"
        row = {"dataset_hash": "fixture"}
        excluded = {"id": "fixture", "status": "excluded", "reason": "synthetic fixture"}
        with patch.object(Path, "exists", return_value=False), \
                patch.object(fetch_jope, "fetch_object") as objects, \
                patch.object(prepare_s3, "selected_catalog", return_value=([row], 0)), \
                patch.object(prepare_s3, "prepare_entry", return_value=excluded) as entry, \
                patch.object(prepare_s3, "write_once"), \
                patch.object(prepare_s3, "sha256", return_value="0" * 64):
            summary = prepare_s3.run(root, "user-supplied", bucket="explicit-fixture-bucket")
        objects.assert_called_once_with("catalog/v1/catalog.jsonl", root / "catalog/v1/catalog.jsonl",
                                        "user-supplied", bucket="explicit-fixture-bucket")
        entry.assert_called_once_with(row, root, "user-supplied", bucket="explicit-fixture-bucket")
        self.assertEqual(summary["excluded_after_checks"], 1)

    def test_prepare_entry_passes_bucket_to_fetch_before_preparing_data(self):
        root = prepare_s3.ROOT / "explicit-fetch-fixture"
        with patch.object(prepare_s3, "fetch", side_effect=ValueError("synthetic source boundary")) as fetch, \
                self.assertRaisesRegex(ValueError, "synthetic source boundary"):
            prepare_s3.prepare_entry({"dataset_hash": "fixture"}, root, None,
                                      bucket="explicit-fixture-bucket")
        fetch.assert_called_once_with("fixture", root, None, bucket="explicit-fixture-bucket")

    def test_both_clis_forward_explicit_bucket_and_optional_profile(self):
        for module, entry, arguments in (
                (fetch_jope, "fetch", ["fetch", "fixture", "/mnt/dope-fetch-fixture"]),
                (prepare_s3, "run", ["prepare", "--scratch", "/mnt/dope-fetch-fixture"])):
            result = {"eligible_catalog_entries": 0, "unknown_rights_excluded": 0,
                      "prepared": 0, "excluded_after_checks": 0}
            with self.subTest(module=module.__name__), \
                    patch.object(sys, "argv", arguments + ["--bucket", "explicit-fixture-bucket",
                                                           "--profile", "user-supplied"]), \
                    patch.object(module, entry, return_value=result) as operation, \
                    contextlib.redirect_stdout(io.StringIO()):
                module.main()
            self.assertEqual(operation.call_args.kwargs, {"bucket": "explicit-fixture-bucket"})
            self.assertEqual(operation.call_args.args[-1], "user-supplied")

    def test_both_clis_require_bucket_before_fetch_or_preparation(self):
        for module, arguments in ((fetch_jope, ["fetch", "fixture", "/mnt/dope-fetch-fixture"]),
                                  (prepare_s3, ["prepare"])):
            with self.subTest(module=module.__name__), patch.object(sys, "argv", arguments), \
                    patch.object(Path, "mkdir") as mkdir, \
                    patch.object(fetch_jope.subprocess, "run") as aws, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                module.main()
            self.assertEqual(error.exception.code, 2)
            mkdir.assert_not_called()
            aws.assert_not_called()


if __name__ == "__main__":
    unittest.main()
