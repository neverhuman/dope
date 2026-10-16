"""K0 replay custody checks on isolated fixtures. No benchmark data is read and no auditor runs."""

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes import rescore_adapters as A
from research.benchmark.review_fixes import rescore_ledger as R

CONTRACT = {"predeclare_sha256": "p" * 64, "evaluator_sha256": "e" * 64, "auditor_seed": 1729,
            "dependencies": {"numpy": "1.26.4", "scipy": "1.16.3", "scikit-learn": "1.7.2", "catboost": "1.2.10"}}
LEDGER = {"name": "fixture", "file": "fixture/ledger.json", "path": "/nonexistent", "sha256": "l" * 64}
DATASET = "0123456789abcdef"
CELL_KEYS = {"format", "status", "reason", "method", "configuration", "dataset", "fit_seed", "sample_seed", "size",
             "synthetic_sha256", "ledger", "ledger_sha256", "replay_equal", "replay_max_abs_difference",
             "predeclare_sha256", "official_tests_opened"}


class StubPilot:
    def __init__(self, retention=0.5):
        self.calls = []
        self.retention = retention

    def measure(self, train, validation, synthetic, task, seed=1729):
        self.calls.append({"synthetic": Path(synthetic), "bytes": Path(synthetic).read_bytes(), "seed": seed})
        utility = {name: {"retention": self.retention} for name in A.AUDITORS}
        return {"implementation_sha256": CONTRACT["evaluator_sha256"], "dependencies": CONTRACT["dependencies"],
                "utility": utility, "metric_seconds": 0.01}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_job(csv=None, digest=None, **overrides):
    job = A.job("fixture", "GaussianCopula", "default", DATASET, 11, 101, 1, csv=csv, sha=digest,
                stored={name: 0.5 for name in A.AUDITORS})
    job.update(overrides)
    return job


class ScoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.pilot = StubPilot()
        R._STATE.clear()
        R._STATE.update({"pilot": self.pilot, "auth": {DATASET: {
            "task": "binary", "train_sha256": "t" * 64, "validation_sha256": "v" * 64}}})

    def tearDown(self):
        R._STATE.clear()
        self.tmp.cleanup()

    def score(self, job):
        task = {"jobs": [job], "ledger": LEDGER, "contract": CONTRACT, "out": str(self.out),
                "tmp": str(self.root / "scratch"), "workers": str(self.root / "workers")}
        result = R.score_group(task)
        return result, json.loads(R.cell_path(self.out, job).read_text())

    def write(self, name: str, data: bytes) -> Path:
        path = self.root / "samples" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path


class HeaderTests(ScoreCase):
    def test_header_is_stripped_into_a_removed_temporary_copy(self):
        data = b"f0,target\r\n1.5,0\r\n2.5,1\r\n"
        path = self.write("sample-101-1n.csv", data)
        _, cell = self.score(fixture_job(path, sha(data), has_header=True))
        self.assertEqual(cell["status"], "ok")
        self.assertEqual(self.pilot.calls[0]["bytes"], b"1.5,0\r\n2.5,1\r\n")
        self.assertNotEqual(self.pilot.calls[0]["synthetic"], path)
        self.assertFalse(self.pilot.calls[0]["synthetic"].exists())
        self.assertEqual(cell["synthetic_sha256"], sha(data))

    def test_header_declaration_must_match_the_file(self):
        meta = {}
        numeric = self.write("numeric.csv", b"1,0\n")
        header = self.write("header.csv", b"a,b\n1,0\n")
        self.assertEqual(R.headerless(numeric, False, self.root, meta), numeric)
        with self.assertRaises(A.Refused) as caught:
            R.headerless(numeric, True, self.root, meta)
        self.assertEqual(caught.exception.code, "header_absent")
        with self.assertRaises(A.Refused) as caught:
            R.headerless(header, False, self.root, meta)
        self.assertEqual(caught.exception.code, "unexpected_header")


class CustodyTests(ScoreCase):
    def test_hash_mismatch_is_refused_before_scoring(self):
        path = self.write("n1-seed101.csv", b"1,0\n")
        _, cell = self.score(fixture_job(path, "0" * 64))
        self.assertEqual(cell["status"], "sample_hash_mismatch")
        self.assertTrue(cell["reason"].startswith("observed_sha256:"))
        self.assertEqual(cell["synthetic_sha256"], "0" * 64)
        self.assertNotIn("report", cell)
        self.assertEqual(self.pilot.calls, [])

    def test_test_csv_and_evaluator_paths_are_refused_by_name(self):
        for path, code in ((self.root / "worker" / "test.csv", "refused_test_path"),
                           (self.root / "evaluator" / "n1-seed101.csv", "refused_evaluator_path")):
            _, cell = self.score(fixture_job(path, "0" * 64))
            self.assertEqual((cell["status"], cell["reason"]), ("failed", code))
        self.assertEqual(self.pilot.calls, [])

    def test_symlinked_file_or_directory_is_refused(self):
        data = b"1,0\n"
        real = self.write("real.csv", data)
        link = self.root / "samples" / "link.csv"
        link.symlink_to(real)
        linked_dir = self.root / "linked"
        linked_dir.symlink_to(real.parent, target_is_directory=True)
        for path in (link, linked_dir / "real.csv"):
            _, cell = self.score(fixture_job(path, sha(data)))
            self.assertEqual((cell["status"], cell["reason"]), ("failed", "refused_symlink"))
        self.assertEqual(self.pilot.calls, [])

    def test_absent_samples_are_recorded_never_imputed(self):
        _, cell = self.score(fixture_job(self.root / "gone.csv", "a" * 64))
        self.assertEqual((cell["status"], cell["reason"]), ("sample_absent", "csv_missing"))
        self.assertEqual((cell["replay_equal"], cell["replay_max_abs_difference"]), (None, None))
        gap = fixture_job(blocked=["sample_absent", "ledger_fit_unavailable"], sample_seed=211)
        _, cell = self.score(gap)
        self.assertEqual((cell["status"], cell["reason"], cell["synthetic_sha256"]),
                         ("sample_absent", "ledger_fit_unavailable", None))
        self.assertNotIn("report", cell)
        self.assertEqual(self.pilot.calls, [])

    def test_worker_inputs_must_match_the_ledger(self):
        data = b"1,0\n"
        path = self.write("n1-seed101.csv", data)
        _, cell = self.score(fixture_job(path, sha(data), expected_inputs={"train": "x" * 64}))
        self.assertEqual((cell["status"], cell["reason"]), ("failed", "worker_input_mismatch"))

    def test_evaluator_digest_change_fails_the_cell(self):
        data = b"1,0\n"
        path = self.write("n1-seed101.csv", data)
        self.pilot.measure = lambda *args, **kwargs: {"implementation_sha256": "x", "utility": {}}
        _, cell = self.score(fixture_job(path, sha(data)))
        self.assertEqual((cell["status"], cell["reason"]), ("failed", "evaluator_digest_mismatch"))


class ErrorTests(ScoreCase):
    def test_bad_csv_records_only_the_exception_type(self):
        path = self.write("sample.csv", b"1,0\n")
        with patch.object(self.pilot, "measure", side_effect=ValueError("source value forbidden")):
            _, cell = self.score(fixture_job(path, sha(path.read_bytes())))
        self.assertEqual((cell["status"], cell["reason"]), ("failed", "ValueError"))
        self.assertNotIn("source value forbidden", json.dumps(cell))

    def test_catboost_failure_keeps_a_typed_unavailable_cell(self):
        from catboost import CatBoostError
        path = self.write("sample.csv", b"1,0\n")
        with patch.object(self.pilot, "measure", side_effect=CatBoostError("source value forbidden")):
            _, cell = self.score(fixture_job(path, sha(path.read_bytes())))
        self.assertEqual((cell["status"], cell["reason"]), ("failed", "CatBoostError"))
        self.assertNotIn("source value forbidden", json.dumps(cell))

    def test_unexpected_failure_stops_and_removes_the_headerless_copy(self):
        path = self.write("sample.csv", b"f0,target\n1,0\n")
        with patch.object(self.pilot, "measure", side_effect=AssertionError("internal invariant")):
            with self.assertRaises(AssertionError):
                self.score(fixture_job(path, sha(path.read_bytes()), has_header=True))
        self.assertFalse(any((self.root / "scratch").glob("*.headerless.csv")))
        self.assertFalse(self.out.exists())


class ReplayTests(ScoreCase):
    def test_comparison_is_exact_and_reports_the_largest_gap(self):
        fresh = {name: {"retention": 0.5} for name in A.AUDITORS}
        self.assertEqual(R.replay({name: 0.5 for name in A.AUDITORS}, fresh), (True, 0.0))
        equal, worst = R.replay({"catboost": 0.5 + 1e-12, "linear": 0.5, "mlp": 0.5}, fresh)
        self.assertFalse(equal)
        self.assertAlmostEqual(worst, 1e-12, delta=1e-15)
        self.assertEqual(R.replay(None, fresh), (None, None))
        equal, worst = R.replay({"catboost": None, "linear": 0.5, "mlp": 0.5}, fresh)
        self.assertFalse(equal)
        self.assertFalse(math.isinf(worst))

    def test_ok_cell_has_the_exact_schema_and_replay_fields(self):
        data = b"1,0\n"
        path = self.write("n1-seed101.csv", data)
        result, cell = self.score(fixture_job(path, sha(data)))
        self.assertEqual(set(cell), CELL_KEYS | {"report"})
        self.assertEqual((cell["status"], cell["replay_equal"], cell["replay_max_abs_difference"]), ("ok", True, 0.0))
        self.assertEqual(cell["report"]["implementation_sha256"], CONTRACT["evaluator_sha256"])
        self.assertIs(cell["official_tests_opened"], False)
        self.assertEqual(self.pilot.calls[0]["seed"], 1729)
        self.assertTrue(result[0]["physical"])

    def test_shared_csv_is_scored_once_for_every_logical_cell(self):
        data = b"1,0\n"
        path = self.write("n1-seed101.csv", data)
        jobs = [fixture_job(path, sha(data)), fixture_job(path, sha(data), configuration="native_selected"),
                fixture_job(blocked=["sample_absent", "ledger_x"]), fixture_job(blocked=["sample_absent", "ledger_x"])]
        groups = R.groups_of(jobs)
        self.assertEqual([len(group) for group in groups], [2, 1, 1])
        task = {"jobs": groups[0], "ledger": LEDGER, "contract": CONTRACT, "out": str(self.out),
                "tmp": str(self.root / "scratch"), "workers": str(self.root / "workers")}
        self.assertEqual([item["status"] for item in R.score_group(task)], ["ok", "ok"])
        self.assertEqual(len(self.pilot.calls), 1)


class LayoutTests(unittest.TestCase):
    def test_cell_path_and_split_seed(self):
        out = Path("/out")
        forest = A.job("forest", A.FOREST, "default", DATASET, 11, 307, 4)
        self.assertEqual(R.cell_path(out, forest), out / "ForestDiffusion_Forest-Flow/default" / DATASET
                         / "fit-11/sample-307-4n.cell.json")
        boot = A.job("controls", "real_bootstrap_4n", "control", DATASET, None, 101, 4)
        self.assertEqual(R.cell_path(out, boot).parent.name, "fit-none")
        split = A.job("controls", "predictor_only", "control_split2027", DATASET, 11, 101, 1, split_seed=2027)
        record = R.cell_record(split, "failed", "x", LEDGER, CONTRACT)
        self.assertEqual(set(record), CELL_KEYS | {"split_seed"})
        self.assertNotIn("split_seed", R.cell_record(boot, "failed", "x", LEDGER, CONTRACT))

    def test_locations_mark_other_hosts_private_disks(self):
        self.assertEqual(A.location("/home/ubuntu/dope-scratch-x1/a/b.csv"), "xbabe1")
        self.assertTrue(A.remote_elsewhere("/home/ubuntu/dope-scratch-x1/a/b.csv", "xbabe2"))
        self.assertFalse(A.remote_elsewhere("/home/ubuntu/dope-scratch-x1/a/b.csv", "xbabe1"))
        self.assertFalse(A.remote_elsewhere("/mnt/fast-scratch/dope-benchmark/a.csv", "xbabe3"))

    def test_plan_round_trip_is_bound_to_the_ledger_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans" / "fixture.jsonl"
            jobs = [fixture_job("/x/n1-seed101.csv", "a" * 64)]
            R.write_plan(path, LEDGER, jobs, "xbabe2")
            self.assertEqual(R.load_plan(path, LEDGER), jobs)
            with self.assertRaises(SystemExit):
                R.load_plan(path, {**LEDGER, "sha256": "m" * 64})
            lines = path.read_text().splitlines()
            header = {**json.loads(lines[0]), "adapters_sha256": "s" * 64}
            path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
            with self.assertRaises(SystemExit):
                R.load_plan(path, LEDGER)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_controls_identities(self):
        rows = [{"kind": "predictor_only", "dataset": DATASET, "fit_seed": 23, "sample_seed": 101, "size": 4},
                {"kind": "split", "dataset": DATASET, "fit_seed": 11, "split_seed": 2027, "sample_seed": 211, "size": 1},
                {"kind": "real_bootstrap_4n", "dataset": DATASET, "fit_seed": None, "sample_seed": 307, "size": 4}]
        ledger = self.root / "cells.jsonl"
        ledger.write_text("".join(json.dumps({**row, "status": "ok", "synthetic_sha256": "a" * 64}) + "\n"
                                  for row in rows))
        jobs = A.controls(ledger, {"controls": self.root / "controls"})
        spec = [(job["method"], job["configuration"], job["fit_seed"], job["split_seed"], Path(job["csv_path"]).name)
                for job in jobs]
        self.assertEqual(spec, [
            ("predictor_only", "control", 23, None, "predictor-fit23-sample101-size4.synthetic.csv"),
            ("predictor_only", "control_split2027", 11, 2027, "split2027-sample211-size1.synthetic.csv"),
            ("real_bootstrap_4n", "control", None, None, "real-bootstrap-sample307.synthetic.csv")])
        self.assertTrue(jobs[1]["train_path"].endswith("split2027-sample211-size1.train.csv"))

    def test_density_skips_dope_and_records_ledger_gaps(self):
        base = {"dataset": DATASET, "fit_seed": 11, "sample_seed": 101, "size_multiplier": 1, "configuration": "default"}
        evidence = {"metric_path": "/m/attempt/n1-seed101.metric.json", "sample_sha256": "b" * 64,
                    "sample_seed": 101, "size_multiplier": 1}
        cells = [{**base, "method": "DOPE", "status": "ok"},
                 {**base, "method": "GaussianCopula", "status": "ok", "sample_evidence": evidence,
                  "utility": {"catboost": {"retention": 0.25}}},
                 {**base, "method": "Chow-Liu", "status": "fit_unavailable", "sample_evidence": None}]
        ledger = self.root / "density.json"
        ledger.write_text(json.dumps({"cells": cells}))
        jobs = A.density(ledger, {})
        self.assertEqual([job["method"] for job in jobs], ["GaussianCopula", "Chow-Liu"])
        self.assertEqual(jobs[0]["csv_path"], "/m/attempt/n1-seed101.csv")
        self.assertEqual(jobs[0]["stored_retention"], {"catboost": 0.25, "linear": None, "mlp": None})
        self.assertEqual(jobs[1]["blocked"], ["sample_absent", "ledger_fit_unavailable"])

    def test_arf_lost_receipt_falls_back_to_its_sibling_samples(self):
        sample_dir = self.root / "samples" / "job"
        receipt = self.root / "receipts" / "r1.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({"input_refs": {"synthetic_ref": {
            "path": str(sample_dir / "n1-seed101.csv"), "sha256": "a" * 64, "csv_format": "headerless_numeric"}}}))
        common = {"method": "ARF", "dataset": DATASET, "fit_seed": 23, "row_multiplier": 1,
                  "selection_binding": "author_default"}
        rows = [{**common, "sample_seed": 101, "input_hashes": {"synthetic": "a" * 64, "train": "t", "validation": "v"},
                 "metric_receipt_ref": {"path": str(receipt), "sha256": sha(receipt.read_bytes())}},
                {**common, "sample_seed": 211, "input_hashes": {"synthetic": "c" * 64, "train": "t", "validation": "v"},
                 "metric_receipt_ref": {"path": str(self.root / "receipts" / "lost.json"), "sha256": "0" * 64}}]
        ledger = self.root / "panel.json"
        ledger.write_text(json.dumps({"rows": rows}))
        jobs = A.arf(ledger, {"host": "xbabe2"})
        self.assertEqual([job["route"] for job in jobs], ["receipt", "sibling"])
        self.assertEqual(jobs[1]["csv_path"], str(sample_dir / "n1-seed211.csv"))
        self.assertEqual(jobs[1]["csv_sha256"], "c" * 64)
        rows[1]["metric_receipt_ref"]["path"] = str(self.root / "remote" / "lost.json")
        ledger.write_text(json.dumps({"rows": rows}))
        with patch.object(A, "LOCATIONS", ((str(self.root) + "/", "xbabe2"),)):
            with self.assertRaises(A.RemoteOnly):
                A.arf(ledger, {"host": "xbabe3"})


if __name__ == "__main__":
    unittest.main()
