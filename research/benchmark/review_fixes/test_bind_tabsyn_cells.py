"""TabSyn scalar binder on public toy fixtures; no CSV or partition is read."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes import bind_tabsyn_cells as tabsyn

ROOT = Path(__file__).resolve().parents[3]
FOREIGN_PILOT = "6cac487cac89900f871e45e07c19d02e65f8da857385c5f1204716580546e72a"


def tabsyn_cell(**overrides):
    cell = {
        "dataset": "ab" * 8, "formal_dp": False, "null_loss": 0.2, "official_tests_opened": False,
        "rows": {"synthetic": 4, "train": 4, "validation": 1}, "sample_seed": 101, "size": 1,
        "status": "ok", "synthetic_sha256": "cd" * 32,
        "utility": {auditor: {"informative": True, "low_signal_noninferior": None, "retention": 0.5,
                              "trtr_loss": 0.1, "tstr_loss": 0.15} for auditor in tabsyn.AUDITORS},
    }
    cell.update(overrides)
    return cell


def receipt_for(cell, **overrides):
    receipt = {
        "cell_sha256": tabsyn.cell_sha256(cell), "dataset": cell["dataset"],
        "dependencies": dict(tabsyn.DEPENDENCIES), "formal_dp": False,
        "implementation_sha256": tabsyn.PILOT_SHA256, "official_tests_opened": False,
        "sample_seed": cell["sample_seed"], "size": cell["size"],
        "synthetic_sha256": cell["synthetic_sha256"], "train_sha256": "ef" * 32,
        "validation_sha256": "01" * 32,
    }
    receipt.update(overrides)
    return receipt


def gate(**overrides):
    document = {"dataset": "02a45777900441ce", "expected_retention": 0.75, "match": True,
                "official_tests_opened": False, "row_multiplier": 4, "sample_seed": 101,
                "scored_retention": 0.75}
    document.update(overrides)
    return document


EIGHT = {"cells": [{"dataset": "02a45777900441ce", "sample_seed": 101, "row_multiplier": 4,
                    "metrics": {"utility": {"auditors": {"catboost": {"retention": 0.75}}}}}]}


class TabSynBinderTests(unittest.TestCase):
    def test_declared_evaluator_matches_the_v2_contract(self):
        declaration = json.loads((ROOT / "research/benchmark/review_fixes/predeclare_v2.json").read_text())
        self.assertEqual(tabsyn.PILOT_SHA256, declaration["evaluator"]["utility_sha256"])
        self.assertEqual(tabsyn.DEPENDENCIES, declaration["evaluator"]["dependencies"])

    def test_open_flags_and_foreign_keys_are_refused(self):
        tabsyn.assert_public_cell(tabsyn_cell())
        for flag in (True, 1, 0, None, "false"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "flag"):
                tabsyn.assert_public_cell(tabsyn_cell(official_tests_opened=flag))
        with self.assertRaisesRegex(ValueError, "public scalar set"):
            tabsyn.assert_public_cell(tabsyn_cell(csv="/scratch/sample.csv"))
        with self.assertRaisesRegex(ValueError, "sample seed"):
            tabsyn.assert_public_cell(tabsyn_cell(sample_seed=11))

    def test_every_cell_needs_a_receipt_from_the_pinned_evaluator(self):
        cell = tabsyn_cell()
        tabsyn.assert_evaluated([cell], [receipt_for(cell)])
        refusals = {
            "does not carry the pinned pilot evaluator": [],
            "does not carry the pinned pilot evaluator implementation": [receipt_for(cell, implementation_sha256=FOREIGN_PILOT)],
            "dependencies differ": [receipt_for(cell, dependencies={**tabsyn.DEPENDENCIES, "numpy": "2.0.0"})],
            "official test flag": [receipt_for(cell, official_tests_opened=None)],
            "duplicated": [receipt_for(cell), receipt_for(cell)],
        }
        for reason, receipts in refusals.items():
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                tabsyn.assert_evaluated([cell], receipts)

    def test_receipt_binds_the_exact_cell_bytes(self):
        cell = tabsyn_cell()
        receipt = receipt_for(cell)
        changed = copy.deepcopy(cell)
        changed["utility"]["catboost"]["retention"] = 0.5000001
        with self.assertRaisesRegex(ValueError, "does not bind"):
            tabsyn.assert_evaluated([changed], [receipt])
        other = receipt_for(tabsyn_cell(sample_seed=211))
        with self.assertRaisesRegex(ValueError, "outside the ledger"):
            tabsyn.assert_evaluated([cell], [receipt, other])

    def test_parser_gate_must_reproduce_the_published_retention(self):
        tabsyn.assert_parser_gate(gate(), EIGHT)
        for document, reason in (
            (gate(match=False), "did not match"),
            (gate(official_tests_opened=True), "did not match"),
            (gate(sample_seed=211), "identity"),
            (gate(scored_retention=0.7), "published retention"),
            (gate(expected_retention=0.7, scored_retention=0.7), "published retention"),
            ({**gate(), "path": "/scratch/parser.csv"}, "public set"),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                tabsyn.assert_parser_gate(document, EIGHT)
        with self.assertRaisesRegex(ValueError, "published retention"):
            tabsyn.assert_parser_gate(gate(), {"cells": EIGHT["cells"] * 2})

    def test_replay_refuses_before_reduction(self):
        cell = tabsyn_cell()
        with patch("docs.whitepaper.scripts.review_tabsyn.payload_from_cells") as reduce:
            with self.assertRaisesRegex(ValueError, "pinned pilot evaluator"):
                tabsyn.replay([cell], [], gate(), EIGHT)
            reduce.assert_not_called()
        with patch("docs.whitepaper.scripts.review_tabsyn.payload_from_cells", return_value={"a": 1}) as reduce:
            self.assertEqual(tabsyn.replay([cell], [receipt_for(cell)], gate(), EIGHT), '{\n  "a": 1\n}\n')
            reduce.assert_called_once_with([cell], gate())

    def test_committed_ledger_is_fully_bound_to_evaluator_receipts(self):
        cells = tabsyn.decode_jsonl(tabsyn.LEDGER.read_bytes(), "ledger")
        receipts = tabsyn.decode_jsonl(tabsyn.RECEIPTS.read_bytes(), "receipts")
        tabsyn.assert_unique_cells(cells)
        tabsyn.assert_evaluated(cells, receipts)
        tabsyn.assert_parser_gate(json.loads(tabsyn.GATE.read_text()), json.loads(tabsyn.EIGHT_PANEL.read_text()))
        self.assertEqual(len(cells), 594)


if __name__ == "__main__":
    unittest.main()
