#!/usr/bin/env python3
"""Bind the scored holdout-privacy cells to the committed privacy aggregate.

privacy_fidelity.py writes one JSON document per method, configuration,
lineage, size and sample seed, with the absolute scratch path of the scored
sample CSV. This tool admits those documents only. It refuses a symlink, a
.tmp staging or non-JSON file, an open test or formal-privacy flag, a method or
configuration outside the published set, a key outside the public schema, a
misfiled or duplicated identity, and an unknown unavailable reason. It drops
the scratch path, recomputes the panel in memory, and writes the ledger only
when that panel is byte-identical to the committed aggregate. It reads no CSV.
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

RAW = Path("/mnt/fast-scratch/dope-benchmark/review-fixes-lane-b/privacy")
PRIVACY = REPO / "research" / "benchmark" / "results" / "review-fixes-privacy-v1"
LEDGER = PRIVACY / "scalar-cells.jsonl"
PANEL = PRIVACY / "panel.json"
FORMAT = "dope-review-fix-privacy-scalar-cells-v1"
# The scored source snapshot of the lane-B run (git 999a0f8, privacy_fidelity.py
# 80adb6aa, expanded_validation_metrics.py a952062c) is recorded, not trusted.
PRODUCER = {
    "path": "research/benchmark/review_fixes/privacy_fidelity.py",
    "git_sha": "999a0f897519fe70367da73e9126eaba444cc42b",
    "sha256": "80adb6aa3ee793d882ce3b2f5b50c94fd2ebc9964c3df67e412a594926dabfa7",
    "expanded_implementation_sha256": "a952062c0f83805f6442a440a5ae15293a843349d423eaa106a9c56a20d14c0f",
}
PUBLISHED = {
    ("DOPE", "features12_steps2048"),
    ("GaussianCopula", "native_selected"),
    ("Chow-Liu", "native_selected"),
    ("independent_marginals", "native_selected"),
    ("CTGAN", "native_selected"),
    ("TVAE", "native_selected"),
    ("ARF", "author_default"),
    ("ARF", "native_selected"),
    ("ForestDiffusion/Forest-Flow", "native_selected"),
}
SIZES = (1, 4)
SAMPLE_SEEDS = (101, 211, 307)
FLAGS = ("official_tests_opened", "formal_dp", "hipaa_deidentification")
IDENTITY_KEYS = {"configuration", "dataset", "method", "sample_seed", "size"}
UNAVAILABLE_KEYS = IDENTITY_KEYS | set(FLAGS) | {"reason", "status"}
OK_KEYS = UNAVAILABLE_KEYS | {
    "c2st_catboost_auc",
    "c2st_rows_per_class",
    "c2st_status",
    "dcr_fit_median",
    "dcr_validation_median",
    "distance_mia_auc",
    "distance_mia_status",
    "nndr_fit_median",
    "rows",
    "synthetic_sha256",
}
DISTANCES = ("dcr_fit_median", "dcr_validation_median", "nndr_fit_median", "distance_mia_auc")
ROW_KEYS = {"fit", "synthetic", "validation"}
REASONS = {"sample_csv_absent", "sample_csv_unresolved"}
STRIPPED = ("csv",)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _assert_identity(cell: dict) -> None:
    if (cell["method"], cell["configuration"]) not in PUBLISHED:
        raise ValueError("privacy cell method or configuration is outside the published set")
    if not _hex(cell["dataset"], 16):
        raise ValueError("privacy cell dataset is not a lineage id")
    if not _int(cell["size"]) or cell["size"] not in SIZES:
        raise ValueError("privacy cell size is outside the predeclaration")
    if not _int(cell["sample_seed"]) or cell["sample_seed"] not in SAMPLE_SEEDS:
        raise ValueError("privacy cell sample seed is outside the predeclaration")


def assert_public_cell(cell: dict) -> None:
    """The ledger schema: identity, closed flags, status, scalars, sample digest."""
    if not isinstance(cell, dict) or "status" not in cell:
        raise ValueError("privacy cell keys are not the public scalar set")
    expected = OK_KEYS if cell["status"] == "ok" else UNAVAILABLE_KEYS
    if set(cell) != expected:
        raise ValueError("privacy cell keys are not the public scalar set")
    if any(cell[flag] is not False for flag in FLAGS):
        raise ValueError("privacy cell official test or formal privacy flag is not closed")
    _assert_identity(cell)
    if cell["status"] != "ok":
        if cell["status"] != "unavailable" or cell["reason"] not in REASONS:
            raise ValueError("privacy cell unavailable reason is outside the producer set")
        return
    if cell["reason"] is not None:
        raise ValueError("privacy cell status is ok with a reason")
    if not _hex(cell["synthetic_sha256"], 64):
        raise ValueError("privacy cell synthetic hash is not 64 hex characters")
    rows = cell["rows"]
    if not isinstance(rows, dict) or set(rows) != ROW_KEYS or any(not _int(rows[k]) or rows[k] < 1 for k in ROW_KEYS):
        raise ValueError("privacy cell row counts are not positive ints")
    if cell["distance_mia_status"] != "ok" or any(not _finite(cell[key]) for key in DISTANCES):
        raise ValueError("privacy cell distance metric is not finite")
    if cell["c2st_status"] == "ok":
        if not _finite(cell["c2st_catboost_auc"]) or not _int(cell["c2st_rows_per_class"]):
            raise ValueError("privacy cell detector value is not finite")
    elif cell["c2st_status"] != "unavailable" or cell["c2st_catboost_auc"] is not None or cell["c2st_rows_per_class"] is not None:
        raise ValueError("privacy cell detector status is outside the producer set")


def identity_of(cell: dict) -> tuple:
    return (cell["method"], cell["configuration"], cell["dataset"], cell["size"], cell["sample_seed"])


def expected_path(cell: dict) -> Path:
    return Path(*cell["method"].split("/"), cell["configuration"], cell["dataset"],
                f"size{cell['size']}-seed{cell['sample_seed']}.json")


def public_cell(document: dict) -> dict:
    """Drop the absolute scratch path; every remaining key must be public."""
    if not isinstance(document, dict):
        raise ValueError("privacy cell is not an object")
    csv = document.get("csv")
    if "csv" not in document or not (csv is None or (isinstance(csv, str) and csv.startswith("/"))):
        raise ValueError("privacy cell scratch path field is not the producer form")
    cell = {key: value for key, value in document.items() if key not in STRIPPED}
    assert_public_cell(cell)
    return cell


def assert_unique_cells(cells: list) -> None:
    seen = set()
    for cell in cells:
        assert_public_cell(cell)
        identity = identity_of(cell)
        if identity in seen:
            raise ValueError("privacy ledger contains a duplicate identity")
        seen.add(identity)


def load_cell_dir(root: Path) -> list[dict]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("privacy cell directory is missing or redirected")
    found, seen = [], set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("privacy cell directory contains a symlink")
        if path.is_dir():
            continue
        if path.name.endswith(".tmp"):
            raise ValueError("privacy cell directory contains a .tmp staging file")
        if path.suffix != ".json" or not path.is_file():
            raise ValueError("privacy cell directory contains a non-json file")
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("privacy cell is not json") from exc
        cell = public_cell(document)
        if path.relative_to(root) != expected_path(cell):
            raise ValueError("privacy cell is filed under another identity")
        identity = identity_of(cell)
        if identity in seen:
            raise ValueError("privacy cell identity is duplicated")
        seen.add(identity)
        found.append(cell)
    if not found:
        raise ValueError("privacy cell directory has no json documents")
    return found


def canonical_jsonl(cells: list[dict]) -> str:
    ordered = sorted(cells, key=identity_of)
    return "".join(json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n" for cell in ordered)


def decode_jsonl(raw: bytes) -> list[dict]:
    cells = []
    for line in raw.decode().splitlines():
        if not line:
            continue
        try:
            cells.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError("privacy ledger line is not json") from exc
    if not cells:
        raise ValueError("privacy ledger is empty")
    return cells


def replay(cells: list[dict]) -> str:
    """Public cells in, canonical panel text out."""
    from research.benchmark.review_fixes.bind_control_cells import canonical_panel
    from docs.whitepaper.scripts.review_privacy import payload_from_cells

    assert_unique_cells(cells)
    return canonical_panel(payload_from_cells(cells))


def reference_rows(cells: list[dict]) -> dict:
    """Per-lineage holdout (validation) and fit (training) reference sizes."""
    per = {}
    for cell in cells:
        if cell["status"] == "ok":
            pair = (cell["rows"]["validation"], cell["rows"]["fit"])
            if per.setdefault(cell["dataset"], pair) != pair:
                raise ValueError("privacy lineage has two reference sizes")
    if not per:
        raise ValueError("privacy ledger has no scored lineage")

    def median(values):
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    return {
        "lineages": len(per),
        "holdout_median": median([pair[0] for pair in per.values()]),
        "fit_median": median([pair[1] for pair in per.values()]),
        "fit_to_holdout_median": median([pair[1] / pair[0] for pair in per.values()]),
    }


def write_ledger(cells: list[dict], ledger_path: Path, panel_path: Path) -> str:
    text = canonical_jsonl(cells)
    if replay(decode_jsonl(text.encode())).encode() != panel_path.read_bytes():
        raise ValueError("recomputed privacy panel does not match the committed aggregate")
    digest = hashlib.sha256(text.encode()).hexdigest()
    panel = json.loads(panel_path.read_text())
    meta = {
        "cell_count": len(cells),
        "format": FORMAT,
        "formal_dp": False,
        "hipaa_deidentification": False,
        "official_tests_opened": False,
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "predeclaration_sha256": panel["predeclaration_sha256"],
        "producer": PRODUCER,
        "reference_rows": reference_rows(cells),
        "sha256": digest,
        "source": "scalar json documents only; the absolute scratch csv path is dropped",
        "stripped_fields": list(STRIPPED),
    }
    staging = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    staging.write_text(text)
    staging.replace(ledger_path)
    ledger_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, help="raw privacy cell root")
    parser.add_argument("--ledger", type=Path, help="replay an existing ledger")
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--write-ledger", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.cells is not None:
            cells = load_cell_dir(args.cells)
        elif args.ledger is not None:
            cells = decode_jsonl(args.ledger.read_bytes())
        else:
            raise SystemExit("a cell directory or a ledger is required")
        if args.write_ledger is not None:
            digest = write_ledger(cells, args.write_ledger, args.panel)
            print(f"privacy scalar ledger {digest} cells {len(cells)}")
            return
        if replay(cells).encode() != args.panel.read_bytes():
            raise ValueError("recomputed privacy panel does not match the committed aggregate")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.check:
        print("privacy scalar cells match the committed panel")


if __name__ == "__main__":
    main()
