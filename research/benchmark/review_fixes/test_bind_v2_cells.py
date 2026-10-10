"""Custody checks for the v2 scalar binder on fixture cells. No benchmark data is read."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.benchmark.review_fixes import bind_v2_cells as B

CONTRACT = B._contract()
DATASET = "0123456789abcdef"


def raw_cell(**overrides) -> dict:
    report = {
        "implementation_sha256": CONTRACT["implementation_sha256"], "dependencies": dict(CONTRACT["dependencies"]),
        "null_loss": 2.0, "marginal_ks_mean": 0.1, "pair_correlation_fidelity": 0.9, "c2st_auc": 0.6,
        "utility": {auditor: {"trtr_loss": 1.0, "tstr_loss": 1.2, "informative": True, "retention": 0.8}
                    for auditor in B.AUDITORS},
    }
    cell = {"official_tests_opened": False, "status": "ok", "method": "DOPE", "arm": "headline",
            "dataset": DATASET, "fit_seed": 11, "sample_seed": 101, "size": 4, "report": report,
            "synthetic_sha256": "a" * 64, "binary_sha256": CONTRACT["headline_binary"]}
    cell.update(overrides)
    return cell


class PublicCellTests(unittest.TestCase):
    def test_a_contract_cell_binds_without_paths_or_constant_digests(self):
        cell = B.public_cell(raw_cell(), CONTRACT, "block_a")
        self.assertEqual(cell["configuration"], "headline")
        self.assertEqual(cell["utility"]["catboost"]["retention"], 0.8)
        self.assertNotIn("implementation_sha256", cell)
        self.assertNotIn("binary_sha256", cell)
        self.assertIs(cell["official_tests_opened"], False)

    def test_off_contract_cells_are_refused(self):
        report = raw_cell()["report"]
        cases = {
            "official test flag": raw_cell(official_tests_opened=None),
            "only ok cells": raw_cell(status="failed"),
            "unknown method": raw_cell(method="Mystery"),
            "16-hex id": raw_cell(dataset="../escape"),
            "evaluator digest": raw_cell(report={**report, "implementation_sha256": "0" * 64}),
            "library versions": raw_cell(report={**report, "dependencies": {"numpy": "2.0.0"}}),
            "declared sample grid": raw_cell(sample_seed=999),
            "fit seed": raw_cell(fit_seed="11"),
            "synthetic digest": raw_cell(synthetic_sha256="short"),
            "headline-binary": raw_cell(binary_sha256="f" * 64),
            "diagnostic-binary": raw_cell(arm="headline_bnew", binary_sha256=CONTRACT["headline_binary"]),
            "not finite": raw_cell(report={**report, "null_loss": float("nan")}),
        }
        for message, cell in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                B.public_cell(cell, CONTRACT, "block_a")

    def test_a_failed_auditor_is_kept_as_failed(self):
        report = raw_cell()["report"]
        report["utility"]["mlp"] = {"status": "failed", "reason": "degenerate"}
        cell = B.public_cell(raw_cell(report=report), CONTRACT, "k0")
        self.assertEqual(cell["utility"]["mlp"], {"status": "failed"})


class CollectTests(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / "target" / "bind-v2-tests"
        target.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def write(self, relative: str, document: dict) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
        return path

    def test_override_names_the_configuration_and_duplicates_are_refused(self):
        self.write("a/fit-11/sample-101-4n.metric.json", raw_cell(arm="ignored", binary_sha256=None))
        cells = B.collect([(self.root / "a", "replay", "fourseed_bnew")], CONTRACT)
        self.assertEqual([cell["configuration"] for cell in cells], ["fourseed_bnew"])
        self.write("b/fit-11/sample-101-4n.metric.json", raw_cell(arm="ignored", binary_sha256=None))
        with self.assertRaisesRegex(ValueError, "duplicate cell identity"):
            B.collect([(self.root / "a", "replay", "fourseed_bnew"), (self.root / "b", "again", "fourseed_bnew")],
                      CONTRACT)

    def test_symlinks_unfinished_writes_and_skipped_status(self):
        path = self.write("c/fit-11/sample-101-4n.metric.json", raw_cell())
        self.write("c/fit-11/sample-211-4n.metric.json", raw_cell(status="sample_absent"))
        self.assertEqual(len(B.collect([(self.root / "c", "block_a", None)], CONTRACT)), 1)
        (path.parent / "sample-307-4n.metric.json.tmp").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unfinished write"):
            B.collect([(self.root / "c", "block_a", None)], CONTRACT)
        link = self.root / "d" / "fit-11" / "sample-101-4n.metric.json"
        link.parent.mkdir(parents=True)
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "refusing"):
            B.collect([(self.root / "d", "block_a", None)], CONTRACT)

    def test_fit_bytes_are_model_plus_projection_and_must_match_the_file(self):
        fit = {"official_tests_opened": False, "status": "ok", "arm": "headline", "dataset": DATASET,
               "fit_seed": 11, "artifact_bytes": 5, "projection_bytes": 7, "binary_sha256": "b" * 64}
        self.write("e/fit-11/fit.json", fit)
        (self.root / "e" / "fit-11" / "model.dpk").write_bytes(b"12345")
        record = B.collect_fits([(self.root / "e", "block_a", None)])[0]
        self.assertEqual(record["charged_bytes"], 12)
        (self.root / "e" / "fit-11" / "model.dpk").write_bytes(b"123")
        with self.assertRaisesRegex(ValueError, "model bytes differ"):
            B.collect_fits([(self.root / "e", "block_a", None)])
        self.write("e/fit-11/fit.json", {**fit, "official_tests_opened": True})
        with self.assertRaisesRegex(ValueError, "official test flag"):
            B.collect_fits([(self.root / "e", "block_a", None)])

    def test_ledger_meta_records_the_digest_and_contract(self):
        cells = [B.public_cell(raw_cell(), CONTRACT, "block_a")]
        out = self.root / "ledger" / "scalar-cells.jsonl"
        meta = B.write_ledger(cells, out, CONTRACT)
        self.assertEqual(meta["sha256"], hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual(meta["evaluator_sha256"], CONTRACT["implementation_sha256"])
        self.assertEqual(json.loads(out.with_suffix(".meta.json").read_text())["cells"], 1)


if __name__ == "__main__":
    unittest.main()
