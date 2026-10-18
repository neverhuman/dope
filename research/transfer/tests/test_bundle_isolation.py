"""Linux access-denial and deleted-source inventory replay, never a Python sampler."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from research.transfer.evidence import canonical_json

ROOT = Path(__file__).resolve().parents[3]
WORKER = r"""
import hashlib, json, os, socket, sys
from research.transfer.bundle import BundleError
from research.transfer.bundle_isolation import verify_source_free_bundle
root, digest, outside, fd = sys.argv[1:]
try:
    verified = verify_source_free_bundle(root, digest)
    denied = []
    for path in (outside, root + '/../private-source.csv', '/proc/self/environ'):
        try:
            with open(path, 'rb') as stream:
                stream.read(1)
        except OSError:
            denied.append(True)
        else:
            denied.append(False)
    try:
        socket.socket()
    except OSError:
        denied.append(True)
    else:
        denied.append(False)
    try:
        os.read(int(fd), 1)
    except OSError:
        denied.append(True)
    else:
        denied.append(False)
    denied.append('DOPE_PRIVATE_SOURCE' not in os.environ)
    for path in (outside, root + '/..'):
        try:
            os.open(path, os.O_PATH)
        except OSError:
            denied.append(True)
        else:
            denied.append(False)
    assert all(denied)
    print(json.dumps({'payload_sha256': hashlib.sha256(verified.retained_payload).hexdigest(),
                     'accounting': dict(verified.accounting), 'denied': denied}, sort_keys=True))
except BundleError as error:
    print(json.dumps({'rejected': error.code}, sort_keys=True), flush=True)
"""


class BundleIsolationTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "target/b01-evidence/python"
        base.mkdir(parents=True, exist_ok=True)
        # Preserve fixture files as access-proof evidence under the lane target.
        self.root = Path(tempfile.mkdtemp(prefix="case-", dir=base))
        self.bundle = self.root / "recipient"
        self.bundle.mkdir()
        self.outside = self.root / "private-source.csv"
        self.outside.write_bytes(b"private-header,private-value,999999\n")
        payload = bytearray(56)
        payload[:4] = b"DPK3"
        payload[4:7] = bytes([3, 2, 1])
        payload[8] = 8
        payload[12] = 1
        self.payload = bytes(payload)
        (self.bundle / "payload.dpk").write_bytes(self.payload)
        self.value = {
            "amortization_count": 1, "bundle_id": "closure",
            "capabilities": ["byte-inventory", "native-dpk-sample"],
            "components": [{"bytes": len(payload), "consumed_by_sampler": True,
                            "depends_on": [], "executable": False, "handler": "none",
                            "id": "payload", "kind": "dpk3", "path": "payload.dpk",
                            "required": True, "role": "sampling_payload",
                            "sha256": hashlib.sha256(payload).hexdigest()}],
            "format": "dope.generator-bundle/v1", "sampling_handler": "native-inline-dpk",
            "transport_framing": "included", "version": 1,
        }

    def manifest(self):
        raw = canonical_json(self.value)
        (self.bundle / "manifest.json").write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def run_worker(self, digest=None, root=None):
        digest = digest or self.manifest()
        with self.outside.open("rb") as source:
            return subprocess.run(
                [sys.executable, "-B", "-c", WORKER, str(root or self.bundle), digest,
                 str(self.outside), str(source.fileno())],
                cwd=self.bundle, env={"PYTHONPATH": str(ROOT), "DOPE_PRIVATE_SOURCE": str(self.outside)},
                pass_fds=(source.fileno(),), capture_output=True, text=True, timeout=20, check=False,
            )

    def assert_rejected(self, result, code):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"rejected": code})
        self.assertEqual(result.stderr, "")

    def test_clean_directory_reload_after_deleted_source_is_identical(self):
        first = self.run_worker()
        self.assertEqual(first.returncode, 0, first.stderr)
        original = json.loads(first.stdout)
        source = self.root / "deleted-fit-partition"
        source.mkdir()
        rows = source / "rows.csv"
        rows.write_bytes(b"private-header,private-value\n")
        rows.unlink()
        source.rmdir()
        self.assertFalse(source.exists())
        replay = self.run_worker()
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout), original)
        self.assertEqual(original["payload_sha256"], hashlib.sha256(self.payload).hexdigest())
        charged = sum(p.stat().st_size for p in self.bundle.iterdir())
        self.assertEqual(original["accounting"]["charged_sampling_bytes"], charged)
        self.assertEqual(original["denied"], [True] * 8)
        self.assertNotIn("private-", first.stdout)
        (self.root / "inventory-access-proof.json").write_text(first.stdout)

    def test_external_projection_source_and_environment_paths_are_rejected(self):
        for path in ("../projection.json", str(self.outside), "$DOPE_PRIVATE_SOURCE",
                     "${DOPE_PRIVATE_SOURCE}/projection.json", "a/../../rows.csv"):
            with self.subTest(path=path):
                self.value["components"][0]["path"] = path
                self.assert_rejected(self.run_worker(), "unsafe_path")

    def test_undeclared_projection_priors_checkpoint_and_source_are_rejected(self):
        digest = self.manifest()
        for name in ("projection.json", "private-priors.bin", "checkpoint.bin", "source.csv"):
            with self.subTest(name=name):
                sidecar = self.bundle / name
                sidecar.write_bytes(b"private-value")
                self.assert_rejected(self.run_worker(digest), "undeclared_component")
                sidecar.unlink()

    def test_missing_dependency_is_rejected_before_inventory(self):
        self.value["components"][0]["depends_on"] = ["projection"]
        self.assert_rejected(self.run_worker(), "unbound_dependency")

    def test_optional_uncharged_source_rows_are_refused_before_component_reads(self):
        optional = dict(self.value["components"][0], id="optional", role="materialized_rows",
                        path="private-rows.bin", required=False, consumed_by_sampler=False,
                        kind="opaque")
        self.value["components"].insert(0, optional)
        (self.bundle / "private-rows.bin").symlink_to(self.outside)
        self.assert_rejected(self.run_worker(), "uncharged_dependency")

    def test_payload_symlink_and_symlinked_ancestor_are_rejected(self):
        digest = self.manifest()
        payload = self.bundle / "payload.dpk"
        payload.unlink()
        payload.symlink_to(self.outside)
        self.assert_rejected(self.run_worker(digest), "symlink_component")
        alias = self.root / "alias"
        alias.symlink_to(self.bundle, target_is_directory=True)
        self.assert_rejected(self.run_worker(digest, alias), "bundle_isolation_refused")
        self.assert_rejected(self.run_worker(digest, alias / "nested"), "bundle_isolation_refused")
        self.assert_rejected(self.run_worker(digest, self.bundle / ".." / "recipient"),
                             "bundle_directory_unsafe")

    def test_manifest_symlink_and_unbound_private_fields_are_rejected(self):
        digest = self.manifest()
        manifest = self.bundle / "manifest.json"
        manifest.unlink()
        manifest.symlink_to(self.outside)
        self.assert_rejected(self.run_worker(digest), "symlink_component")
        manifest.unlink()
        self.value["source_path"] = str(self.outside)
        self.assert_rejected(self.run_worker(), "unknown_field")

    def test_symlinked_component_directory_is_rejected(self):
        payload = self.bundle / "payload.dpk"
        payload.unlink()
        outside = self.root / "weights"
        outside.mkdir()
        (outside / "payload.dpk").write_bytes(self.payload)
        (self.bundle / "weights").symlink_to(outside, target_is_directory=True)
        self.value["components"][0]["path"] = "weights/payload.dpk"
        self.assert_rejected(self.run_worker(), "symlink_component")


if __name__ == "__main__":
    unittest.main()
