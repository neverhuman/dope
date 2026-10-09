"""TabSyn evaluator replay on toy partitions and a toy evaluator; no real model or CSV."""

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.benchmark.review_fixes import bind_tabsyn_cells as bind
from research.benchmark.review_fixes import tabsyn_evaluator_replay as replay
from research.benchmark.review_fixes.test_bind_tabsyn_cells import FOREIGN_PILOT, tabsyn_cell

ROOT = Path(__file__).resolve().parents[3]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class EvaluatorReplayTests(unittest.TestCase):
    def setUp(self):
        target = ROOT / "target/evaluator-replay-tests"
        target.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        dataset = "ab" * 8
        workers = root / "workers" / dataset
        workers.mkdir(parents=True)
        partitions = {"train": b"0,0\n1,1\n2,2\n3,3\n", "validation": b"0,1\n",
                      "projection": json.dumps({"task": "regression"}).encode()}
        for name, raw in partitions.items():
            (workers / (f"{name}.json" if name == "projection" else f"{name}.csv")).write_bytes(raw)
        self.samples = root / "samples"
        self.sample = self.samples / dataset / "sample-101-1n.csv"
        self.sample.parent.mkdir(parents=True)
        self.sample.write_bytes(b"f0,target\n0,0\n1,1\n2,2\n3,3\n")
        self.cell = tabsyn_cell(dataset=dataset, synthetic_sha256=sha(self.sample.read_bytes()))
        ledger = root / "ledger.jsonl"
        ledger.write_text(bind.canonical_line(self.cell) + "\n")
        source = root / "partitions.jsonl"
        source.write_text(json.dumps({"dataset": dataset, "input_hashes": {
            name: sha(raw) for name, raw in partitions.items()}}) + "\n")
        self.out = root / "evaluator-replay.jsonl"
        self.work = root / "work"
        self.report = {"implementation_sha256": bind.PILOT_SHA256, "dependencies": dict(bind.DEPENDENCIES),
                       "null_loss": self.cell["null_loss"], "utility": self.cell["utility"],
                       "rows": self.cell["rows"]}
        self.pilot = types.SimpleNamespace(measure=lambda *_args: json.loads(json.dumps(self.report)))
        for name, value in (("WORKERS", root / "workers"), ("LEDGER", ledger), ("LEDGER_SHA256", sha(ledger.read_bytes())),
                            ("PARTITIONS", source), ("PARTITIONS_SHA256", sha(source.read_bytes()))):
            p = patch.object(replay, name, value)
            p.start()
            self.addCleanup(p.stop)
        for p in (patch.object(replay, "load_pilot", return_value=self.pilot),
                  patch.object(replay.platform, "node", return_value="toy-host")):
            p.start()
            self.addCleanup(p.stop)

    def run_replay(self):
        argv = ["tabsyn_evaluator_replay", "--samples", str(self.samples), "--work", str(self.work),
                "--out", str(self.out), "--host", "toy-host"]
        with patch.object(replay.sys, "argv", argv):
            replay.main()

    def test_receipt_is_written_only_for_a_byte_identical_cell(self):
        self.run_replay()
        receipts = bind.decode_jsonl(self.out.read_bytes(), "receipts")
        bind.assert_evaluated([self.cell], receipts)
        meta = json.loads(self.out.with_suffix(".meta.json").read_text())
        self.assertEqual((meta["matched"], meta["mismatched"]), (1, 0))
        self.assertFalse((self.work / "partitions").exists() and any((self.work / "partitions").iterdir()))

    def test_any_difference_writes_no_receipt_and_reports_its_size(self):
        self.report["utility"] = json.loads(json.dumps(self.report["utility"]))
        self.report["utility"]["mlp"]["retention"] = 0.5 + 2 ** -40
        with self.assertRaisesRegex(SystemExit, "1 of 1 re-scored cells differ"):
            self.run_replay()
        self.assertFalse(self.out.exists())
        mismatch = json.loads((self.work / "evaluator-replay-mismatch.json").read_text())
        self.assertGreater(mismatch["mismatched"][0]["max_abs_difference"], 0)

    def test_substituted_sample_is_refused_before_decode(self):
        self.sample.write_bytes(b"f0,target\n9,9\n")
        with patch.object(np, "loadtxt", side_effect=AssertionError("decode before auth")) as decode:
            with self.assertRaisesRegex(ValueError, "hash differs"):
                self.run_replay()
        decode.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_foreign_evaluator_or_runtime_is_refused(self):
        for key, value, reason in (("implementation_sha256", FOREIGN_PILOT, "different implementation"),
                                   ("dependencies", {**bind.DEPENDENCIES, "catboost": "1.2.5"}, "runtime differs")):
            with self.subTest(key=key):
                saved = self.report[key]
                self.report[key] = value
                with self.assertRaisesRegex(ValueError, reason):
                    self.run_replay()
                self.report[key] = saved
        self.assertFalse(self.out.exists())

    def test_redirected_partition_is_refused(self):
        workers = replay.WORKERS / ("ab" * 8)
        moved = workers.parent.parent / "outside"
        workers.rename(moved)
        workers.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "redirected"):
            self.run_replay()


if __name__ == "__main__":
    unittest.main()
