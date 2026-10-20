"""Mandatory real control-worker boundary fixtures; no scientific qualification."""
import copy
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

if len(sys.argv) > 1 and sys.argv[1] == "--denial-worker":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.transfer.tdl import control_boundary
from research.transfer.tdl.control_boundary import (
    ABI, SCOPE, SOURCES, BoundaryError, canonical, digest, load_boundary,
    regular_bytes, sample_control, validate_output,
)
from research.transfer.tdl.generators import GaussianCopulaGenerator, CompressedRowsGenerator
from research.transfer.tdl.literal_worker import _control_document, _control_sample, _literal_payload
from research.transfer.tdl.charge import canonical_container
from research.transfer.tdl.runtime import python_runtime_paths, native_runtime_paths

ROOT = Path(__file__).resolve().parents[3]
LIBRARY = ROOT / "target/debug/libdope_kernel.so"


def denial_worker():
    library, pin, source, descriptor = sys.argv[2:]
    runtime = load_boundary(library, pin, ABI)
    assert runtime.confine_current_bundle_directory() == 0
    try:
        os.open(source, os.O_RDONLY)
        raise AssertionError("source_open_allowed")
    except OSError as error:
        assert error.errno in (errno.EACCES, errno.EPERM)
    # Bundle read is allowed only before the second seal.
    fd = os.open("control.tdl", os.O_RDONLY)
    raw = os.read(fd, 1 << 20)
    os.close(fd)
    assert raw
    assert runtime.seal_sampling_process() == 0
    for path in (source, "control.tdl"):
        try:
            os.open(path, os.O_RDONLY)
            raise AssertionError("read_allowed")
        except OSError as error:
            assert error.errno == errno.EPERM
    try:
        os.read(int(descriptor), 1)
        raise AssertionError("inherited_fd_open")
    except OSError as error:
        assert error.errno == errno.EBADF
    try:
        socket.socket()
        raise AssertionError("network_allowed")
    except OSError as error:
        assert error.errno == errno.EPERM
    try:
        os.execv("/bin/true", ["true"])
        raise AssertionError("exec_allowed")
    except OSError as error:
        assert error.errno == errno.EPERM
    os.write(1, b"source=EPERM bundle=EPERM fd=EBADF network=EPERM exec=EPERM\n")


class ControlBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Always reconcile this checkout source with Cargo, even if a DSO exists.
        subprocess.run(["cargo", "build", "--locked", "--lib"], cwd=ROOT, check=True)
        if not LIBRARY.is_file():
            raise AssertionError("own_checkout_cdylib_missing")
        source_dir = ROOT / "research/transfer/tdl"
        paths = python_runtime_paths() | native_runtime_paths([LIBRARY, sys.executable])
        paths |= {source_dir / name for name in SOURCES}
        pins = {str(path.resolve()): digest(regular_bytes(path.resolve())) for path in paths}
        cls.runtime = {"scope": SCOPE, "abi": ABI, "library": str(LIBRARY.resolve()),
                       "library_sha256": digest(regular_bytes(LIBRARY)),
                       "python": str(Path(sys.executable).resolve()), "source_dir": str(source_dir),
                       "source_pins": {name: digest(regular_bytes(source_dir / name)) for name in SOURCES},
                       "closure_pins": pins}

    def setUp(self):
        base = ROOT / "target/control-boundary-tests"
        base.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "private_source.csv"
        self.source.write_text("secret_x,secret_y\n0,10\n1,20\n2,30\n3,40\n")
        schema = self.root / "schema.json"
        schema.write_bytes(canonical({"format": "dope-table-schema", "version": 1, "columns": [
            {"id": "c0001", "source_name": "secret_x", "role": "feature", "dtype": "continuous"},
            {"id": "c0002", "source_name": "secret_y", "role": "target", "dtype": "continuous"}]}))
        self.artifacts = {}
        for method, cls in (("gaussian_copula", GaussianCopulaGenerator),
                            ("compressed_real_rows", CompressedRowsGenerator)):
            model = cls(byte_budget=32768)
            model.fit(self.source, schema, 11)
            artifact = self.root / (method + ".tdl")
            model.export(artifact)
            self.artifacts[method] = artifact
        self.serial = 0

    def sample(self, artifact, method, rows=16, seed=(1 << 64) - 2, runtime=None):
        self.serial += 1
        return sample_control(artifact, digest(regular_bytes(artifact)), method, rows, seed,
                              trusted_runtime=self.runtime if runtime is None else runtime,
                              workdir=self.root / ("run-" + str(self.serial)), wall_seconds=60)

    def test_both_controls_exact_repeated_seed_and_source_absent(self):
        self.source.unlink()
        for method, artifact in self.artifacts.items():
            document = _control_document(_literal_payload(artifact.read_bytes()), method)
            for rows in (16, 64):
                expected = b"".join(struct.pack("<d", v) for row in
                                    _control_sample(document, rows, (1 << 64) - 2) for v in row)
                first, receipt = self.sample(artifact, method, rows)
                second, _ = self.sample(artifact, method, rows)
                self.assertEqual(first, expected)
                self.assertEqual(first, second)
                self.assertEqual(receipt["sample_sha256"], digest(expected))
                self.assertFalse(receipt["pipeline_complete"])
                public = json.dumps(receipt)
                self.assertNotIn(str(self.root), public)
                self.assertNotIn("secret_x", public)
                self.assertNotIn("secret_y", public)

    def test_actual_denials_in_dedicated_fresh_process(self):
        artifact = self.artifacts["compressed_real_rows"]
        bundle = self.root / "denial-bundle"
        bundle.mkdir()
        (bundle / "control.tdl").write_bytes(artifact.read_bytes())
        fd = os.open(self.source, os.O_RDONLY)
        try:
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                     "--denial-worker", str(LIBRARY), self.runtime["library_sha256"],
                                     str(self.source), str(fd)], cwd=bundle, pass_fds=(fd,),
                                    stdin=subprocess.DEVNULL, capture_output=True, check=False)
        finally:
            os.close(fd)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"source=EPERM bundle=EPERM fd=EBADF network=EPERM exec=EPERM\n")

    def test_pin_abi_library_and_artifact_corruption_refused(self):
        artifact = self.artifacts["compressed_real_rows"]
        for key, value in (("library_sha256", "0" * 64), ("abi", 2), ("abi", True), ("scope", "artifact_says_trusted"),
                           ("library", str(self.root / "missing.so"))):
            runtime = copy.deepcopy(self.runtime)
            runtime[key] = value
            with self.assertRaises(BoundaryError):
                self.sample(artifact, "compressed_real_rows", runtime=runtime)
        corrupt = self.root / "corrupt.so"
        corrupt.write_bytes(b"not_a_library")
        with self.assertRaises(OSError):
            load_boundary(corrupt, digest(corrupt.read_bytes()), ABI)
        artifact.write_bytes(b"private_invalid_artifact")
        with self.assertRaises(BoundaryError):
            self.sample(artifact, "compressed_real_rows")
        with self.assertRaises(BoundaryError):
            sample_control(artifact, "0" * 64, "compressed_real_rows", 16, 1,
                           trusted_runtime=self.runtime, workdir=self.root / "bad-pin", wall_seconds=60)

    def test_symlink_fifo_bomb_and_missing_artifact_refused(self):
        artifact = self.artifacts["compressed_real_rows"]
        alias, fifo = self.root / "alias", self.root / "fifo"
        alias.symlink_to(artifact)
        os.mkfifo(fifo)
        for path in (alias, fifo, self.root / "missing"):
            with self.assertRaises((BoundaryError, OSError)):
                regular_bytes(path)
        bomb = self.root / "bomb.tdl"
        payload = gzip.compress(b"x" * ((16 << 20) + 1))
        bomb.write_bytes(canonical_container([{"name": "literal_payload", "hex": payload.hex()}]))
        with self.assertRaises(BoundaryError):
            self.sample(bomb, "compressed_real_rows")

    def test_missing_bundle_is_refused_by_actual_worker(self):
        original = control_boundary.bounded_child

        def remove_bundle_then_execute(argv, cwd, wall_seconds, output_limit):
            (Path(cwd) / "control.tdl").unlink()
            return original(argv, cwd, wall_seconds, output_limit)

        with mock.patch.object(control_boundary, "bounded_child", side_effect=remove_bundle_then_execute):
            with self.assertRaises(BoundaryError):
                self.sample(self.artifacts["compressed_real_rows"], "compressed_real_rows")

    def test_partial_output_and_failed_exit_are_rejected(self):
        artifact = self.artifacts["compressed_real_rows"]
        raw, receipt = self.sample(artifact, "compressed_real_rows")
        request = json.loads((self.root / "run-1/runtime/request.json").read_bytes())
        for code, private, public in ((0, raw[:-8], canonical(receipt)), (1, raw, canonical(receipt)),
                                      (0, raw, canonical(receipt)[:-1])):
            with self.assertRaises((BoundaryError, ValueError)):
                validate_output(code, private, public, request)


    def test_nonfinite_and_forged_claim_receipts_are_rejected(self):
        raw, receipt = self.sample(self.artifacts["compressed_real_rows"], "compressed_real_rows")
        request = json.loads((self.root / "run-1/runtime/request.json").read_bytes())
        for value in (float("nan"), float("inf"), -float("inf")):
            invalid = struct.pack("<d", value) + raw[8:]
            forged = dict(receipt, sample_sha256=digest(invalid))
            with self.assertRaises(BoundaryError):
                validate_output(0, invalid, canonical(forged), request)
        for key in ("scientific_admission", "citable", "abi"):
            forged = dict(receipt, **{key: True})
            with self.assertRaises(BoundaryError):
                validate_output(0, raw, canonical(forged), request)

    def test_startup_import_failure_is_redacted(self):
        original = control_boundary.bounded_child
        observed = []

        def corrupt_import_then_execute(argv, cwd, wall_seconds, output_limit):
            helper = Path(argv[-1]).parent / "literal_worker.py"
            helper.chmod(0o600)
            helper.write_text("raise RuntimeError('PRIVATE_STARTUP_VALUE')\n")
            result = original(argv, cwd, wall_seconds, output_limit)
            observed.append(result)
            return result

        with mock.patch.object(control_boundary, "bounded_child", side_effect=corrupt_import_then_execute):
            with self.assertRaises(BoundaryError):
                self.sample(self.artifacts["compressed_real_rows"], "compressed_real_rows")
        self.assertEqual(len(observed), 1)
        self.assertNotEqual(observed[0][0], 0)
        self.assertEqual(observed[0][1], b"")
        self.assertEqual(observed[0][2], b"control_boundary_refused\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--denial-worker":
        denial_worker()
    else:
        unittest.main()
