"""FOLLOW-L1a: real native/threaded reads, closed inventory, Rust parity input."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from research.transfer.tdl import charge
from research.transfer.tdl.access import AccessTrace
from research.transfer.tdl.common import MetricError
from research.transfer.tdl.runtime import closure_inventory, native_runtime_paths, python_runtime_paths
from research.transfer.tdl.syscall_trace import _calls, verify_sampler_access_trace

ROOT = Path(__file__).resolve().parents[3]


def workspace():
    (ROOT / "target").mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="trace-test-")


def inventory(root, runtime):
    sections = [{"name": "weights", "hex": "010203"}]
    artifact, prior = root / "artifact", root / "prior"
    artifact.write_bytes(charge.canonical_container(sections))
    prior.write_bytes(b"fixed-prior")
    closure, paths = closure_inventory(runtime)
    closure.append({"name": "prior", "hex": prior.read_bytes().hex(),
                    "scope": "shared_prior", "dataset_specific": False})
    paths["prior"] = prior
    manifest = {"sections": sections, "closure_files": closure, "amortization_count": 3}
    return artifact, prior, manifest, paths


def parity_evidence():
    """Rust integration test consumes this real trace, without local paths."""
    cat = str(Path(shutil.which("cat")).resolve())
    with workspace() as directory:
        artifact, prior, manifest, paths = inventory(Path(directory), native_runtime_paths([cat]))
        trace = AccessTrace.from_manifest(artifact, manifest, paths)
        stdout, receipt = trace.run([cat, str(artifact), str(prior)])
        assert stdout == artifact.read_bytes() + prior.read_bytes()
        declared = receipt["declared_payload"] + receipt["declared_closure"]
        observed = [event["inventory_id"] for event in receipt["events"] if event["operation"] == "read"]
        cases = [observed, [], ["hidden"], ["../private"], ["PAYLOAD"], ["payload", "prior"]]
        admission = []
        for reads in cases:
            try:
                verify_sampler_access_trace(declared, [], reads)
                reason = None
            except MetricError as error:
                reason = str(error)
            admission.append({"observed": reads, "reason": reason})
        return {"manifest": manifest, "receipt": receipt, "admission": admission}


class AccessTraceTests(unittest.TestCase):
    def test_native_worker_bytes_and_closure_match_charges(self):
        evidence = parity_evidence()
        receipt = evidence["receipt"]
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["trace_kind"], "linux_strace")
        self.assertEqual(receipt["byte_accounting"], charge.evaluate(evidence["manifest"]))
        reads = {event["inventory_id"] for event in receipt["events"]}
        self.assertTrue({"payload", "prior"} <= reads)
        self.assertTrue(reads <= set(receipt["declared_payload"] + receipt["declared_closure"]))
        self.assertEqual([case["reason"] for case in evidence["admission"]],
                         [None, "empty_sampler_trace", *["undeclared_sampler_file_access"] * 3, None])

    def test_audit_diagnostic_is_incomplete_and_observes_other_python_threads(self):
        with workspace() as directory:
            hidden = Path(directory) / "private-header"
            hidden.write_text("secret-extremum")
            trace = AccessTrace({})
            reasons = []
            def read():
                try:
                    hidden.read_bytes()
                except MetricError as error:
                    reasons.append(str(error))
            self.assertEqual(trace.receipt()["status"], "incomplete")
            with trace.activate():
                worker = threading.Thread(target=read)
                worker.start()
                worker.join()
            self.assertEqual(reasons, ["sampler_uncharged_read"])
            self.assertEqual(trace.receipt()["status"], "failed")
            public = json.dumps(trace.receipt())
            for value in (directory, "private-header", "secret-extremum"):
                self.assertNotIn(value, public)

    def test_native_undeclared_reads_and_symlinks_never_release_stdout(self):
        cat = str(Path(shutil.which("cat")).resolve())
        with workspace() as directory:
            root = Path(directory)
            artifact, prior, manifest, paths = inventory(root, native_runtime_paths([cat]))
            hidden, link = root / "private-header", root / "link"
            hidden.write_text("secret-extremum")
            link.symlink_to(hidden)
            for path in (hidden, link):
                trace = AccessTrace.from_manifest(artifact, manifest, paths)
                with self.assertRaisesRegex(MetricError, "undeclared_sampler_file_access"):
                    trace.run([cat, str(artifact), str(path)])
                receipt = trace.receipt()
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["trace_kind"], "linux_strace")
                public = json.dumps(receipt)
                for value in (directory, "private-header", "secret-extremum"):
                    self.assertNotIn(value, public)

    def test_threaded_c_fopen_is_traced_and_undeclared_native_open_is_rejected(self):
        runtime = python_runtime_paths()
        worker = ("import ctypes,threading,sys;lib=ctypes.CDLL(None);"
                  "lib.fopen.argtypes=[ctypes.c_char_p,ctypes.c_char_p];lib.fopen.restype=ctypes.c_void_p;"
                  "lib.fclose.argtypes=[ctypes.c_void_p];"
                  "t=threading.Thread(target=lambda:lib.fclose(lib.fopen(sys.argv[1].encode(),b'rb')));"
                  "t.start();t.join()")
        with workspace() as directory:
            artifact, prior, manifest, paths = inventory(Path(directory), runtime)
            trace = AccessTrace.from_manifest(artifact, manifest, paths)
            _, receipt = trace.run([sys.executable, "-S", "-B", "-c", worker, str(prior)])
            self.assertIn({"operation": "read", "inventory_id": "prior"}, receipt["events"])
            hidden = Path(directory) / "hidden"
            hidden.write_bytes(b"secret")
            with self.assertRaisesRegex(MetricError, "undeclared_sampler_file_access"):
                trace.run([sys.executable, "-S", "-B", "-c", worker, str(hidden)])
            self.assertEqual(trace.receipt()["status"], "failed")

    def test_changed_or_missing_inventory_and_unavailable_tracer_fail_closed(self):
        cat = str(Path(shutil.which("cat")).resolve())
        with workspace() as directory:
            artifact, prior, manifest, paths = inventory(Path(directory), native_runtime_paths([cat]))
            with self.assertRaisesRegex(MetricError, "closure_inventory_mismatch"):
                AccessTrace.from_manifest(artifact, manifest, {})
            trace = AccessTrace.from_manifest(artifact, manifest, paths)
            with patch("research.transfer.tdl.syscall_trace.shutil.which", return_value=None):
                with self.assertRaisesRegex(MetricError, "sampler_trace_unavailable"):
                    trace.run([cat, str(artifact)])
                self.assertEqual(trace.receipt()["status"], "failed")
            prior.write_bytes(b"replaced")
            with self.assertRaisesRegex(MetricError, "sampler_inventory_changed"):
                trace.run([cat, str(prior)])
            with self.assertRaisesRegex(MetricError, "charge_inventory_mismatch"):
                AccessTrace.from_manifest(artifact, manifest, paths)
            with self.assertRaisesRegex(MetricError, "missing_charge_manifest"):
                AccessTrace({artifact: "payload"}).run([cat, str(artifact)])

    def test_interrupted_or_unparseable_trace_is_rejected(self):
        for log, reason in (("1 openat( <unfinished ...>\n", "incomplete_sampler_trace"),
                            ("unknown format\n", "malformed_sampler_trace"),
                            ("1 <... openat resumed>) = 3\n", "malformed_sampler_trace")):
            with self.subTest(log=log), self.assertRaisesRegex(MetricError, reason):
                list(_calls(log))


if __name__ == "__main__":
    unittest.main()
