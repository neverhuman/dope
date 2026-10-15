"""One native artifact's pinned byte accounting, and the README and supplement sentences bound to it."""
import copy
import json
import unittest

import native_byte_conformance as N


def measured_records():
    """A synthetic receipt and evidence pair in the measured shape; the values only exercise the checks."""
    inspected = {"byte_profile_id": "native-raw-complete-v1", "complete_sampling_bytes": 100, "framing_bytes": 10,
                 "transform_schema_bytes": 60, "native_payload_bytes": 30, "transport_archive_bytes": 100,
                 "generic_runtime_bytes": None, "cold_sampling_closure_bytes": None}
    replay = dict(inspected, generic_runtime_bytes=5000, cold_sampling_closure_bytes=5100)
    receipt = {
        "artifact_sha256": "ab" * 32,
        "inspection": {"accounting": inspected, "profile": "raw-numeric-symbolic-v1",
                       "format": "dope-raw-sampling-artifact", "version": 1},
        "arms": [{"sampling": {"complete_sampling_bytes": 100},
                  "source_free_replay": {"inspection": {"accounting": copy.deepcopy(replay)}}} for _ in (1, 4)],
        "source_rows": 50, "fit_rows": 40, "query_rows": 10, "fit_groups": 8, "query_groups": 2,
        "seed": 7, "split_sha256": "cd" * 32,
    }
    diagnostic = {"sha256": "ef" * 32, "stdout_sha256": "12" * 32}
    evidence = {
        "artifact_sha256": "ab" * 32, "byte_profile_id": "native-raw-complete-v1", "profile": "raw-numeric-symbolic-v1",
        "artifact_file_bytes": 100, "source_free_replays": [{"multiplier": 1}, {"multiplier": 4}],
        "runtime_scope": "synthetic", "measured_binary_sha256": "34" * 32, "measured_binary_file_bytes": 4000,
        "measured_source": "5" * 40, "measured_tree": "6" * 40, "actual_dependency_versions": {"numpy": "0"},
        "saved_research_receipt_sha256": "78" * 32, "saved_report_html_sha256": "9a" * 32,
        "compile_inspect_cli_diagnostics": {"native-inspection.json": diagnostic, "native-fit.json": diagnostic},
        "source": {"attribution": "synthetic", "doi": "10.0/synthetic", "license": "CC-BY-4.0", "sha256": "bc" * 32,
                   "bytes": 1},
    }
    return receipt, evidence


class NativeByteConformanceTests(unittest.TestCase):
    def test_the_tracked_pin_matches_the_readme_and_supplement(self):
        pin = N.check()
        self.assertLess(N.PIN.stat().st_size, 16 * 1024)
        accounting = pin["accounting"]
        self.assertEqual(pin["artifact"]["file_bytes"], accounting["complete_sampling_bytes"])
        self.assertEqual(accounting["cold_sampling_closure_bytes"],
                         accounting["generic_runtime_bytes"] + accounting["complete_sampling_bytes"])
        self.assertEqual(pin["kind"], "engineering_conformance_example")

    def test_the_projection_cross_checks_every_measured_record(self):
        receipt, evidence = measured_records()
        pin = N.project(receipt, evidence)
        N.check_pin(pin)
        self.assertEqual(pin["accounting"], {"complete_sampling_bytes": 100, "transport_archive_bytes": 100,
                                             "framing_bytes": 10, "transform_schema_bytes": 60,
                                             "native_payload_bytes": 30, "generic_runtime_bytes": 5000,
                                             "cold_sampling_closure_bytes": 5100})
        self.assertEqual(pin["agreement"]["source_free_replays"], [1, 4])

    def test_disagreeing_records_are_refused(self):
        def replay(receipt, key, value):
            receipt["arms"][1]["source_free_replay"]["inspection"]["accounting"][key] = value

        cases = {
            "parts do not sum": lambda r, e: r["inspection"]["accounting"].__setitem__("framing_bytes", 11),
            "transport is another object": lambda r, e: r["inspection"]["accounting"].__setitem__("transport_archive_bytes", 120),
            "a replay inspected another total": lambda r, e: replay(r, "complete_sampling_bytes", 101),
            "a replay has another profile": lambda r, e: replay(r, "byte_profile_id", "native-residual-v1"),
            "closure is not runtime plus artifact": lambda r, e: replay(r, "cold_sampling_closure_bytes", 5200),
            "replays measured different runtimes": lambda r, e: replay(r, "generic_runtime_bytes", 5001),
            "a sampling arm reports another total": lambda r, e: r["arms"][0]["sampling"].__setitem__("complete_sampling_bytes", 99),
            "artifact hash differs": lambda r, e: e.__setitem__("artifact_sha256", "ac" * 32),
            "file size differs": lambda r, e: e.__setitem__("artifact_file_bytes", 101),
            "profile differs": lambda r, e: e.__setitem__("profile", "raw-numeric-residual-v1"),
        }
        for name, change in cases.items():
            receipt, evidence = measured_records()
            change(receipt, evidence)
            with self.subTest(name), self.assertRaises(N.ConformanceError):
                N.project(receipt, evidence)

    def test_published_sentences_that_drift_from_the_pin_are_refused(self):
        pin = json.loads(N.PIN.read_text())
        readme = (N.REPO / "README.md").read_text()
        supplement = (N.PAPER / "supplement.tex").read_text()
        generated = N.GENERATED.read_text()
        self.assertEqual(N.text_failures(pin, readme, supplement, generated), [])
        self.assertEqual(generated, N.supplement_paragraph(pin))
        swapped = readme.replace("56 bytes of framing, 1,080 bytes of fitted schema",
                                 "1,080 bytes of framing, 56 bytes of fitted schema")
        self.assertNotEqual(swapped, readme)
        drifted = {
            "README swaps framing and schema": (swapped, supplement, generated),
            "README total": (readme.replace("all report 1,558 bytes", "all report 1,559 bytes"), supplement, generated),
            "README source commit": (readme.replace("commit `59437b1`", "commit `0000000`"), supplement, generated),
            "generated profile": (readme, supplement, generated.replace(r"\texttt{native-raw-complete-v1}",
                                                                        r"\texttt{native-raw-v1}")),
            "generated runtime": (readme, supplement, generated.replace("72{,}049{,}152 bytes", "72{,}049{,}000 bytes")),
            "supplement drops the input": (readme, supplement.replace(r"\input{generated/native-byte-conformance.tex}", ""),
                                           generated),
        }
        for name, texts in drifted.items():
            with self.subTest(name):
                self.assertTrue(N.text_failures(pin, *texts))
        changed = copy.deepcopy(pin)
        changed["accounting"]["native_payload_bytes"] = 423
        self.assertNotEqual(N.supplement_paragraph(changed), generated)
        with self.assertRaises(N.ConformanceError):
            N.text_failures(pin, readme.replace("### One measured native artifact", "### Another heading"), supplement,
                            generated)

    def test_the_pin_refuses_private_paths_and_inconsistent_totals(self):
        pin = json.loads(N.PIN.read_text())
        for change in (lambda p: p["runtime"].__setitem__("scope", "/home/user/target/bin"),
                       lambda p: p["accounting"].__setitem__("native_payload_bytes", 423),
                       lambda p: p["accounting"].__setitem__("cold_sampling_closure_bytes", 1),
                       lambda p: p.__setitem__("format", "other")):
            changed = copy.deepcopy(pin)
            change(changed)
            with self.subTest(json.dumps(changed)[:40]), self.assertRaises(N.ConformanceError):
                N.check_pin(changed)

    def test_the_pin_has_a_closed_shape_valid_digests_and_bounded_parts(self):
        """C2 P17-F2: malformed digests, extra fields, and balanced negative parts are refused."""
        pin = json.loads(N.PIN.read_text())
        sentinel = "zq-raw-source-row-7781"

        def accounting(framing, schema):
            def change(p):
                p["accounting"]["framing_bytes"], p["accounting"]["transform_schema_bytes"] = framing, schema
            return change

        cases = {
            "nonhex artifact digest": lambda p: p["artifact"].__setitem__("sha256", p["artifact"]["sha256"][:12] + "z" * 52),
            "short evidence digest": lambda p: p["evidence_sha256"].__setitem__("research_receipt", "ab" * 10),
            "malformed source commit": lambda p: p["measured_run"].__setitem__("source_commit", "59437b1"),
            "extra top-level field": lambda p: p.__setitem__("raw_source_rows", [sentinel]),
            "extra source field": lambda p: p["source_data"].__setitem__("header", sentinel),
            "missing field": lambda p: p["runtime"].pop("binary_file_bytes"),
            "balanced negative part": accounting(-1, 1137),
            "boolean count": lambda p: p["source_data"].__setitem__("rows", True),
            "float total": lambda p: p["accounting"].__setitem__("generic_runtime_bytes", 72049152.0),
            "agreement not true": lambda p: p["agreement"].__setitem__("file_size_equals_complete", 1),
            "bad dependency version": lambda p: p["measured_run"]["dependency_versions"].__setitem__("numpy", sentinel),
        }
        for name, change in cases.items():
            changed = copy.deepcopy(pin)
            change(changed)
            with self.subTest(name), self.assertRaises(N.ConformanceError) as raised:
                N.check_pin(changed)
            self.assertNotIn(sentinel, str(raised.exception))
        raw = N.PIN.read_bytes()
        self.assertEqual(N.load_pin(raw), pin)
        with self.assertRaises(N.ConformanceError):
            N.load_pin(raw.rstrip(b"\n") + b" " * N.MAX_PIN_BYTES)
        with self.assertRaises(N.ConformanceError):
            N.load_pin(raw.replace(b'"version": 1\n', b'"version": 1, "version": 1\n'))


if __name__ == "__main__":
    unittest.main()
