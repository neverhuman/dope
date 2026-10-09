#!/usr/bin/env python3
"""Bind the committed TabSyn scalar cells to the pinned pilot evaluator.

The committed ledger keeps the public utility scalars of every sampled TabSyn
identity. Its producer dropped the evaluator digest, so each cell is admitted
only through an evaluator receipt: an independent re-score with the
authenticated research/benchmark/pilot_metrics.py whose public cell is
byte-identical to the ledger line. A cell without such a receipt, or with a
receipt for another implementation or dependency set, is refused. The parser
gate must match the committed eight-lineage panel. Nothing here opens a CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "research" / "benchmark" / "results"
TABSYN = RESULTS / "review-fixes-tabsyn-v1"
LEDGER = TABSYN / "scalar-cells.jsonl"
RECEIPTS = TABSYN / "evaluator-replay.jsonl"
GATE = TABSYN / "parser-check.json"
PANEL = TABSYN / "panel.json"
EIGHT_PANEL = RESULTS / "tabsyn-eight-lineage-validation" / "panel.json"

# predeclare_v2.json evaluator.utility_sha256 and evaluator.dependencies.
PILOT_SHA256 = "6c03892685df856f7a491ca79acb1c28fe8270ead8bbfea1ba50228cb6f0e7e3"
DEPENDENCIES = {"catboost": "1.2.10", "numpy": "1.26.4", "scikit-learn": "1.7.2", "scipy": "1.16.3"}
AUDITORS = ("catboost", "linear", "mlp")
SIZES = (1, 4)
SAMPLE_SEEDS = (101, 211, 307)
CELL_KEYS = {
    "dataset",
    "formal_dp",
    "null_loss",
    "official_tests_opened",
    "rows",
    "sample_seed",
    "size",
    "status",
    "synthetic_sha256",
    "utility",
}
ROW_KEYS = {"synthetic", "train", "validation"}
UTILITY_KEYS = {"informative", "low_signal_noninferior", "retention", "trtr_loss", "tstr_loss"}
RECEIPT_KEYS = {
    "cell_sha256",
    "dataset",
    "dependencies",
    "formal_dp",
    "implementation_sha256",
    "official_tests_opened",
    "sample_seed",
    "size",
    "synthetic_sha256",
    "train_sha256",
    "validation_sha256",
}
GATE_KEYS = {
    "dataset",
    "expected_retention",
    "match",
    "official_tests_opened",
    "row_multiplier",
    "sample_seed",
    "scored_retention",
}
GATE_IDENTITY = {"dataset": "02a45777900441ce", "sample_seed": 101, "row_multiplier": 4}


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def assert_public_cell(cell: dict) -> None:
    if not isinstance(cell, dict) or set(cell) != CELL_KEYS:
        raise ValueError("TabSyn cell keys are not the public scalar set")
    if cell["official_tests_opened"] is not False or cell["formal_dp"] is not False:
        raise ValueError("TabSyn cell official test or formal privacy flag is not closed")
    if cell["status"] != "ok":
        raise ValueError("TabSyn cell status is not ok")
    if not _hex(cell["dataset"], 16):
        raise ValueError("TabSyn cell dataset is not a lineage id")
    if cell["size"] not in SIZES or not _int(cell["size"]):
        raise ValueError("TabSyn cell size is outside the predeclaration")
    if cell["sample_seed"] not in SAMPLE_SEEDS or not _int(cell["sample_seed"]):
        raise ValueError("TabSyn cell sample seed is outside the predeclaration")
    if not _hex(cell["synthetic_sha256"], 64):
        raise ValueError("TabSyn cell synthetic hash is not 64 hex characters")
    if not _finite(cell["null_loss"]):
        raise ValueError("TabSyn cell null loss is not finite")
    rows = cell["rows"]
    if not isinstance(rows, dict) or set(rows) != ROW_KEYS:
        raise ValueError("TabSyn cell row counts are not the public set")
    if any(not _int(rows[key]) or rows[key] < 1 for key in ROW_KEYS):
        raise ValueError("TabSyn cell row count is not a positive int")
    utility = cell["utility"]
    if not isinstance(utility, dict) or set(utility) != set(AUDITORS):
        raise ValueError("TabSyn cell utility auditors are not the public set")
    for auditor in AUDITORS:
        block = utility[auditor]
        if not isinstance(block, dict) or set(block) != UTILITY_KEYS:
            raise ValueError("TabSyn cell utility keys are not the public set")
        if not isinstance(block["informative"], bool):
            raise ValueError("TabSyn cell utility flags are not the public set")
        if not isinstance(block["low_signal_noninferior"], (bool, type(None))):
            raise ValueError("TabSyn cell utility flags are not the public set")
        if block["retention"] is not None and not _finite(block["retention"]):
            raise ValueError("TabSyn cell retention is not finite")
        if not _finite(block["trtr_loss"]) or not _finite(block["tstr_loss"]):
            raise ValueError("TabSyn cell utility loss is not finite")


def identity_of(cell: dict) -> tuple:
    return (cell["dataset"], cell["size"], cell["sample_seed"])


def canonical_line(cell: dict) -> str:
    return json.dumps(cell, sort_keys=True, separators=(",", ":"))


def cell_sha256(cell: dict) -> str:
    return hashlib.sha256(canonical_line(cell).encode()).hexdigest()


def canonical_jsonl(records: list[dict], key) -> str:
    return "".join(canonical_line(record) + "\n" for record in sorted(records, key=key))


def decode_jsonl(raw: bytes, label: str) -> list:
    records = []
    for line in raw.decode().splitlines():
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line is not json") from exc
    if not records:
        raise ValueError(f"{label} is empty")
    return records


def assert_unique_cells(cells: list) -> None:
    seen = set()
    for cell in cells:
        assert_public_cell(cell)
        identity = identity_of(cell)
        if identity in seen:
            raise ValueError("TabSyn ledger contains a duplicate identity")
        seen.add(identity)


def receipt_identity(receipt: dict) -> tuple:
    return (receipt["dataset"], receipt["size"], receipt["sample_seed"])


def assert_receipt(receipt: dict) -> None:
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise ValueError("TabSyn evaluator receipt keys are not the public set")
    if receipt["official_tests_opened"] is not False or receipt["formal_dp"] is not False:
        raise ValueError("TabSyn evaluator receipt official test flag is not closed")
    if receipt["implementation_sha256"] != PILOT_SHA256:
        raise ValueError("TabSyn cell does not carry the pinned pilot evaluator implementation")
    if receipt["dependencies"] != DEPENDENCIES:
        raise ValueError("TabSyn cell evaluator dependencies differ from the declaration")
    for key in ("cell_sha256", "synthetic_sha256", "train_sha256", "validation_sha256"):
        if not _hex(receipt[key], 64):
            raise ValueError("TabSyn evaluator receipt digest is not 64 hex characters")


def assert_evaluated(cells: list, receipts: list) -> None:
    """Every cell must carry one receipt from the pinned evaluator for its exact bytes."""
    bound = {}
    for receipt in receipts:
        assert_receipt(receipt)
        identity = receipt_identity(receipt)
        if identity in bound:
            raise ValueError("TabSyn evaluator receipt identity is duplicated")
        bound[identity] = receipt
    for cell in cells:
        receipt = bound.pop(identity_of(cell), None)
        if receipt is None:
            raise ValueError("TabSyn cell does not carry the pinned pilot evaluator implementation")
        if receipt["synthetic_sha256"] != cell["synthetic_sha256"] or receipt["cell_sha256"] != cell_sha256(cell):
            raise ValueError("TabSyn evaluator receipt does not bind this cell's bytes")
    if bound:
        raise ValueError("TabSyn evaluator receipts name a cell outside the ledger")


def published_gate_retention(eight_panel: dict):
    found = [
        cell for cell in eight_panel.get("cells") or []
        if all(cell.get(key) == value for key, value in GATE_IDENTITY.items())
    ]
    if len(found) != 1:
        return None
    return found[0]["metrics"]["utility"]["auditors"]["catboost"]["retention"]


def assert_parser_gate(gate: dict, eight_panel: dict) -> None:
    """The predeclared parser check: published sample, published CatBoost retention."""
    if not isinstance(gate, dict) or set(gate) != GATE_KEYS:
        raise ValueError("TabSyn parser gate keys are not the public set")
    if gate["official_tests_opened"] is not False or gate["match"] is not True:
        raise ValueError("TabSyn parser gate did not match with closed tests")
    if any(gate[key] != value for key, value in GATE_IDENTITY.items()):
        raise ValueError("TabSyn parser gate identity differs from the predeclaration")
    expected = published_gate_retention(eight_panel)
    if not _finite(expected) or gate["expected_retention"] != expected or gate["scored_retention"] != expected:
        raise ValueError("TabSyn parser gate does not reproduce the published retention")


def replay(cells: list, receipts: list, gate: dict, eight_panel: dict) -> str:
    """Decoded public inputs in, canonical panel text out. Raises on any refusal."""
    from research.benchmark.review_fixes.bind_control_cells import canonical_panel
    from docs.whitepaper.scripts.review_tabsyn import payload_from_cells

    assert_unique_cells(cells)
    assert_evaluated(cells, receipts)
    assert_parser_gate(gate, eight_panel)
    return canonical_panel(payload_from_cells(cells, gate))


def replay_files() -> str:
    return replay(
        decode_jsonl(LEDGER.read_bytes(), "TabSyn ledger"),
        decode_jsonl(RECEIPTS.read_bytes(), "TabSyn evaluator receipts"),
        json.loads(GATE.read_bytes()),
        json.loads(EIGHT_PANEL.read_bytes()),
    )


def write_meta(panel_path: Path = PANEL) -> dict:
    if replay_files().encode() != panel_path.read_bytes():
        raise ValueError("recomputed TabSyn panel does not match the committed aggregate")
    ledger, receipts = LEDGER.read_bytes(), RECEIPTS.read_bytes()
    meta = {
        "cell_count": len(decode_jsonl(ledger, "TabSyn ledger")),
        "evaluator_receipts_sha256": hashlib.sha256(receipts).hexdigest(),
        "format": "dope-review-fix-tabsyn-scalar-cells-v1",
        "formal_dp": False,
        "implementation_sha256": PILOT_SHA256,
        "official_tests_opened": False,
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "parser_gate_sha256": hashlib.sha256(GATE.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(ledger).hexdigest(),
        "source": "public utility scalars only; sample CSVs and partitions are not in this ledger",
    }
    LEDGER.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="replay the committed ledger to the panel")
    parser.add_argument("--write-meta", action="store_true")
    args = parser.parse_args()
    try:
        if args.write_meta:
            meta = write_meta()
            print(f"TabSyn scalar ledger {meta['sha256']} bound to {meta['cell_count']} evaluator receipts")
            return
        if replay_files().encode() != PANEL.read_bytes():
            raise ValueError("recomputed TabSyn panel does not match the committed aggregate")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.check:
        print("TabSyn scalar cells match the committed panel")


if __name__ == "__main__":
    main()
