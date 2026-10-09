#!/usr/bin/env python3
"""Bind public scalar control cells to the committed aggregate.

The scratch tree keeps one JSON document per cell beside the synthetic CSV
tables. This tool admits the JSON documents only. It refuses an official-test
flag, a duplicate identity, or a key outside the scalar set, recomputes the
panel in memory, and writes a ledger only when that panel matches the
committed aggregate. It does not write paper tex and it does not open test.csv.
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

CELL_KEYS = {
    "dataset",
    "fit_seed",
    "kind",
    "metrics",
    "official_tests_opened",
    "reason",
    "sample_seed",
    "size",
    "split_seed",
    "status",
    "synthetic_sha256",
}
METRIC_KEYS = {
    "c2st_auc",
    "marginal_ks_mean",
    "null_loss",
    "pair_correlation_fidelity",
    "rows",
    "utility",
}
ROW_KEYS = {"synthetic", "train", "validation"}
UTILITY_KEYS = {
    "informative",
    "low_signal_noninferior",
    "retention",
    "trtr_loss",
    "tstr_loss",
}
AUDITORS = ("catboost", "linear", "mlp")
PANEL = REPO / "research" / "benchmark" / "results" / "review-fixes-controls-v1" / "panel.json"
FORMAT = "dope-review-fix-control-scalar-cells-v1"


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def assert_public_scalar(cell: dict) -> None:
    if set(cell) != CELL_KEYS:
        raise ValueError("control cell keys are not the public scalar set")
    if cell["official_tests_opened"] is not False:
        raise ValueError("official test flag is not closed")
    if cell["status"] != "ok" or cell["reason"] is not None:
        raise ValueError("control cell status is not ok")
    if not isinstance(cell["dataset"], str) or not cell["dataset"]:
        raise ValueError("control cell dataset is not a string")
    if not _hex(cell["synthetic_sha256"], 64):
        raise ValueError("control cell synthetic hash is not 64 hex characters")
    if cell["kind"] not in {"predictor_only", "real_bootstrap_4n", "split"}:
        raise ValueError("control cell kind is outside the predeclaration")
    if not isinstance(cell["sample_seed"], int) or isinstance(cell["sample_seed"], bool):
        raise ValueError("control cell sample seed is not an int")
    if cell["size"] not in (1, 4):
        raise ValueError("control cell size is outside the predeclaration")
    fit_seed = cell["fit_seed"]
    if fit_seed is not None and (not isinstance(fit_seed, int) or isinstance(fit_seed, bool)):
        raise ValueError("control cell fit seed is not an int")
    split_seed = cell["split_seed"]
    if split_seed is not None and (not isinstance(split_seed, int) or isinstance(split_seed, bool)):
        raise ValueError("control cell split seed is not an int")
    metrics = cell["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS:
        raise ValueError("control cell metrics are not the public scalar set")
    for key in ("c2st_auc", "marginal_ks_mean", "null_loss", "pair_correlation_fidelity"):
        if not _finite(metrics[key]):
            raise ValueError("control cell metric is not finite")
    rows = metrics["rows"]
    if not isinstance(rows, dict) or set(rows) != ROW_KEYS:
        raise ValueError("control cell row counts are not the public set")
    for key in ROW_KEYS:
        if not isinstance(rows[key], int) or isinstance(rows[key], bool) or rows[key] < 0:
            raise ValueError("control cell row count is not a non-negative int")
    utility = metrics["utility"]
    if not isinstance(utility, dict) or set(utility) != set(AUDITORS):
        raise ValueError("control cell utility auditors are not the public set")
    for auditor in AUDITORS:
        block = utility[auditor]
        if not isinstance(block, dict) or set(block) != UTILITY_KEYS:
            raise ValueError("control cell utility keys are not the public set")
        low_signal = block["low_signal_noninferior"]
        if not isinstance(block["informative"], bool) or not isinstance(low_signal, (bool, type(None))):
            raise ValueError("control cell utility flags are not the public set")
        retention = block["retention"]
        if retention is not None and not _finite(retention):
            raise ValueError("control cell retention is not finite")
        for key in ("trtr_loss", "tstr_loss"):
            if not _finite(block[key]):
                raise ValueError("control cell utility value is not finite")


def identity_of(cell: dict) -> tuple:
    return (
        cell["dataset"],
        cell["kind"],
        cell["fit_seed"],
        cell["split_seed"],
        cell["sample_seed"],
        cell["size"],
    )


def load_cell_dir(root: Path) -> list[dict]:
    if not root.is_dir():
        raise ValueError("control cell directory is missing")
    found = []
    seen = set()
    paths = sorted(path for path in root.glob("*/*") if path.is_file())
    for path in paths:
        if path.suffix != ".json":
            continue
        if path.name.endswith(".tmp"):
            raise ValueError("control cell directory contains a temporary file")
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("control cell is not json") from exc
        if not isinstance(document, dict):
            raise ValueError("control cell is not an object")
        assert_public_scalar(document)
        identity = identity_of(document)
        if identity in seen:
            raise ValueError("control cell identity is duplicated")
        seen.add(identity)
        found.append(document)
    if not found:
        raise ValueError("control cell directory has no json documents")
    return found


def canonical_jsonl(cells: list[dict]) -> str:
    ordered = sorted(cells, key=identity_of)
    return "".join(json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n" for cell in ordered)


def load_jsonl(path: Path) -> list[dict]:
    seen = set()
    found = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("control ledger line is not json") from exc
        if not isinstance(document, dict):
            raise ValueError("control ledger line is not an object")
        assert_public_scalar(document)
        identity = identity_of(document)
        if identity in seen:
            raise ValueError("control cell identity is duplicated")
        seen.add(identity)
        found.append(document)
    if not found:
        raise ValueError("control ledger is empty")
    return found


def canonical_panel(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def differing_paths(left, right, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(differing_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{prefix} length"]
        paths = []
        for index, (item, other) in enumerate(zip(left, right)):
            paths.extend(differing_paths(item, other, f"{prefix}[{index}]"))
        return paths
    if left != right:
        return [prefix or "<root>"]
    return []


def assert_matches_panel(cells: list[dict], panel_path: Path) -> dict:
    from docs.whitepaper.scripts.review_controls import payload_from_cells

    recomputed = payload_from_cells(cells)
    committed = json.loads(panel_path.read_text())
    if canonical_panel(recomputed) != canonical_panel(committed):
        paths = differing_paths(recomputed, committed)[:12]
        joined = ", ".join(paths) if paths else "canonical bytes"
        raise ValueError(f"recomputed control panel does not match the committed aggregate: {joined}")
    return recomputed


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_ledger(cells: list[dict], ledger_path: Path, panel_path: Path) -> str:
    text = canonical_jsonl(cells)
    digest = sha256_text(text)
    meta = {
        "cell_count": len(cells),
        "format": FORMAT,
        "official_tests_opened": False,
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "predeclaration_sha256": json.loads(panel_path.read_text())["predeclaration_sha256"],
        "sha256": digest,
        "source": "scalar json documents only; synthetic csv tables are not in this ledger",
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(ledger_path)
    meta_path = ledger_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--write-ledger", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.cells is not None:
        cells = load_cell_dir(args.cells)
    elif args.ledger is not None:
        cells = load_jsonl(args.ledger)
    else:
        raise SystemExit("a cell directory or a ledger is required")
    try:
        assert_matches_panel(cells, args.panel)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.write_ledger is not None:
        digest = write_ledger(cells, args.write_ledger, args.panel)
        print(f"control scalar ledger {digest}")
    if args.check:
        print("control scalar cells match the committed panel")


if __name__ == "__main__":
    main()
