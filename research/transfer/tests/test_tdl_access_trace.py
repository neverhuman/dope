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


class BoundedControlTests(unittest.TestCase):
    def source(self, root, size=128):
        from research.transfer.tdl.generators import CompressedRowsGenerator, GaussianCopulaGenerator
        import csv
        schema = {"format": "dope-table-schema", "version": 1, "columns": [
            {"id": "c0001", "source_name": "SECRET_x", "role": "feature", "dtype": "continuous"},
            {"id": "c0002", "source_name": "SECRET_fixed", "role": "feature", "dtype": "continuous"},
            {"id": "c0003", "source_name": "SECRET_y", "role": "target", "dtype": "continuous"}]}
        rows = [[(i - size / 2) / 16, 7., (i - size / 2) / 8] for i in range(size)]
        source, metadata = root / "SECRET_source.csv", root / "SECRET_schema.json"
        with source.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow([column["source_name"] for column in schema["columns"]])
            writer.writerows(rows)
        metadata.write_text(json.dumps(schema))
        return source, metadata, rows, (GaussianCopulaGenerator, CompressedRowsGenerator)

    def test_real_controls_source_deleted_artifact_replay_and_complete_byte_charge(self):
        from research.transfer.tdl.literal_worker import _control_document, _literal_payload
        with workspace() as directory:
            root = Path(directory)
            source, schema, training, classes = self.source(root)
            for cls in classes:
                port = cls(byte_budget=2048)
                port.fit(source, schema, (1 << 64) - 1)
                artifact = root / cls.__name__
                receipt = port.export(artifact)
                repeated = cls(byte_budget=2048)
                repeated.fit(source, schema, (1 << 64) - 1)
                duplicate = root / (cls.__name__ + "-copy")
                repeated.export(duplicate)
                self.assertEqual(artifact.read_bytes(), duplicate.read_bytes())
                self.assertLessEqual(artifact.stat().st_size, 2048)
                self.assertEqual(receipt["payload_bytes"], artifact.stat().st_size)
                content = _literal_payload(artifact.read_bytes())
                doc = _control_document(content, port.method)
                self.assertEqual(doc["fit_rows"], len(training))
                with self.assertRaises(MetricError):
                    cls.reload(artifact, byte_budget=artifact.stat().st_size - 1)
            source.unlink()
            schema.unlink()
            for cls in classes:
                artifact = root / cls.__name__
                port = cls.reload(artifact, byte_budget=2048)
                sampled, receipt = port.sample(300, (1 << 64) - 2)
                same, _ = cls.reload(artifact, byte_budget=2048).sample(300, (1 << 64) - 2)
                self.assertEqual(sampled, same)
                self.assertFalse(receipt["citable"])
                self.assertEqual(receipt["access_trace"]["status"], "passed")
                self.assertEqual(receipt["access_trace"]["trace_kind"], "linux_strace")
                self.assertGreater(receipt["sampling_closure_bytes"], receipt["payload_bytes"])
                self.assertEqual({row[1] for row in sampled}, {7.})
                self.assertNotIn("SECRET", json.dumps(receipt))
                if port.method == "compressed_real_rows":
                    self.assertTrue(set(map(tuple, sampled)) <= set(map(tuple, training)))
                else:
                    x, y = zip(*[(row[0], row[2]) for row in sampled])
                    mx, my = sum(x) / len(x), sum(y) / len(y)
                    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
                    scale = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** .5
                    self.assertGreater(cov / scale, .98)
                    self.assertGreaterEqual(min(x), min(row[0] for row in training))
                    self.assertLessEqual(max(x), max(row[0] for row in training))

    def test_target_first_input_keeps_feature_target_identity_across_controls(self):
        import csv
        from research.transfer.tdl.literal_worker import _control_document, _control_sample
        with workspace() as directory:
            root = Path(directory)
            source, schema, training, classes = self.source(root)
            doc = json.loads(schema.read_text())
            doc["columns"] = [doc["columns"][i] for i in (2, 1, 0)]
            schema.write_text(json.dumps(doc))
            with source.open("w") as stream:
                writer = csv.writer(stream)
                writer.writerow([column["source_name"] for column in doc["columns"]])
                writer.writerows([[row[i] for i in (2, 1, 0)] for row in training])
            for cls in classes:
                port = cls(byte_budget=2048)
                port.fit(source, schema, 42)
                fitted = _control_document(port._payload, port.method)
                self.assertEqual(fitted["column_order"], "features_then_target")
                sampled = _control_sample(fitted, 64, 17)
                self.assertTrue(all(row[0] == 7. and abs(row[2] - 2 * row[1]) < .05 for row in sampled))

    def test_permuted_csv_header_preserves_schema_column_identity(self):
        import csv
        from research.transfer.tdl.literal_worker import _control_document, _control_sample
        with workspace() as directory:
            root = Path(directory)
            source, schema, training, classes = self.source(root)
            with source.open("w") as stream:
                writer = csv.writer(stream)
                writer.writerow(["SECRET_y", "SECRET_x", "SECRET_fixed"])
                writer.writerows([[row[2], row[0], row[1]] for row in training])
            for cls in classes:
                port = cls(byte_budget=2048)
                port.fit(source, schema, 42)
                fitted = _control_document(port._payload, port.method)
                sampled = _control_sample(fitted, 64, 17)
                self.assertTrue(all(row[1] == 7. and abs(row[2] - 2 * row[0]) < .05 for row in sampled))
            source.write_text("SECRET_y,SECRET_y,SECRET_fixed\n1,2,3\n")
            for cls in classes:
                with self.assertRaises(MetricError):
                    cls(byte_budget=2048).fit(source, schema, 42)

    def test_caps_schema_types_and_fit_failure_do_not_leave_stale_artifact(self):
        import copy
        with workspace() as directory:
            root = Path(directory)
            source, schema, _, classes = self.source(root)
            valid = json.loads(schema.read_text())
            for cls in classes:
                port = cls(byte_budget=2048)
                port.fit(source, schema, 0)
                for change in ({"dtype": "nominal"}, {"role": "excluded_key"}, {"dtype": "count"}):
                    altered = copy.deepcopy(valid)
                    altered["columns"][0].update(change)
                    schema.write_text(json.dumps(altered))
                    with self.assertRaisesRegex(MetricError, "unsupported_control_schema"):
                        port.fit(source, schema, 0)
                    with self.assertRaisesRegex(MetricError, "generator_not_fitted"):
                        port.export(root / "must-not-exist")
                schema.write_text(json.dumps(valid))
                with self.assertRaisesRegex(MetricError, "control_artifact_exceeds_cap"):
                    cls(byte_budget=1).fit(source, schema, 0)
                for seed in (True, -1, 1 << 64):
                    with self.assertRaises(MetricError):
                        port.fit(source, schema, seed)
                schema.write_text('{"format":"invalid","format":"dope-table-schema"}')
                with self.assertRaisesRegex(MetricError, "invalid_control_artifact"):
                    port.fit(source, schema, 0)
                schema.write_text(json.dumps(valid))

    def test_bounded_read_rejects_symlinks_fifos_gzip_bombs_and_changed_artifacts(self):
        import gzip
        import os
        from research.transfer.tdl.literal_worker import MAX_CONTROL_JSON, _control_document, _literal_payload
        with workspace() as directory:
            root = Path(directory)
            source, schema, _, classes = self.source(root)
            for cls in classes:
                port = cls(byte_budget=2048)
                port.fit(source, schema, 1)
                artifact = root / cls.__name__
                port.export(artifact)
                loaded = cls.reload(artifact, byte_budget=2048)
                with self.assertRaisesRegex(MetricError, "generator_artifact_exists"):
                    port.export(artifact)
                link = root / (cls.__name__ + "-link")
                link.symlink_to(artifact)
                with self.assertRaises(MetricError):
                    cls.reload(link, byte_budget=2048)
                fifo = root / (cls.__name__ + "-fifo")
                os.mkfifo(fifo)
                with self.assertRaises(MetricError):
                    cls.reload(fifo, byte_budget=2048)
                content = _literal_payload(artifact.read_bytes())
                doc = _control_document(content, loaded.method)
                doc["fit_seed"] = 2
                replacement = charge.canonical_container([{"name": "literal_payload", "hex":
                    gzip.compress(json.dumps(doc).encode(), mtime=0).hex()}])
                artifact.write_bytes(replacement)
                with self.assertRaisesRegex(MetricError, "generator_artifact_changed"):
                    loaded.sample(1, 1)
                with self.assertRaises(MetricError):
                    _control_document(gzip.compress(b"a" * (MAX_CONTROL_JSON + 1)), loaded.method)
                with self.assertRaises(MetricError):
                    _control_document(content + content, loaded.method)

    def test_copula_ties_and_constant_columns_remain_finite_and_seeded(self):
        from research.transfer.tdl.literal_worker import _copula_state, _control_sample
        rows = [[float(i % 3), 9.] for i in range(30)]
        factor, ordered = _copula_state(rows, 0)
        document = {"method": "gaussian_copula", "columns": 2, "factor": factor,
                    "quantile_count": 3, "quantiles": [[column[0], column[15], column[-1]] for column in ordered]}
        a, b = _control_sample(document, 30, 7), _control_sample(document, 30, 8)
        self.assertNotEqual(a, b)
        self.assertEqual(a, _control_sample(document, 30, 7))
        self.assertTrue(all(0 <= row[0] <= 2 and row[1] == 9 for row in a))


if __name__ == "__main__":
    unittest.main()
