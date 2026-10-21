#!/usr/bin/env python3
"""Re-score every committed TabSyn scalar cell with the pinned pilot evaluator.

tabsyn_sample.py --score kept the utility block of pilot_metrics.measure but
dropped the implementation digest and dependency versions it reports. This tool
re-scores each ledger identity from its stored synthetic CSV with the
authenticated pilot_metrics.py bytes, through authenticated copies of the
published train and validation partitions, and writes one receipt per cell
only when every re-scored public cell is byte-identical to its ledger line.
Samples are local copies that must hash to the ledger's synthetic_sha256 before
they are decoded. No official test file is opened and no paper output is
written. A mismatch is reported with its largest absolute difference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.benchmark.review_fixes.bind_tabsyn_cells import (  # noqa: E402
    DEPENDENCIES,
    LEDGER,
    PILOT_SHA256,
    RECEIPTS,
    assert_unique_cells,
    canonical_jsonl,
    canonical_line,
    cell_sha256,
    decode_jsonl,
    identity_of,
    receipt_identity,
)

LEDGER_SHA256 = "b9fd4600ec54fc89b938eb627f31d6112176fc4137e040c3686c15efd17d2d62"
PILOT = REPO / "research" / "benchmark" / "pilot_metrics.py"
PARTITIONS = REPO / "research/benchmark/results/arf-tabsyn-followon-validation-v1/scalar-cells.jsonl"
PARTITIONS_SHA256 = "a9c9c6f63ed4439eca38402109beebdc2a5fec6c3be4a1037ec8754b4486578c"
WORKERS = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker")
SAMPLE_SOURCE = "xbabe1:/home/ubuntu/dope-scratch-x1/dope-rf-tabsyn-sample-v1"
FORMAT = "dope-review-fix-tabsyn-evaluator-replay-v1"


def authenticated(path: Path, expected: str, root: Path | None = None) -> bytes:
    if path.name == "test.csv" or path.is_symlink():
        raise ValueError("official test or redirected input refused")
    if root is not None and path.resolve() != root.resolve() / path.relative_to(root):
        raise ValueError("redirected input refused")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("evaluator replay input hash differs")
    return raw


def partition_pins() -> dict:
    """Published train, validation and projection digests for every lineage."""
    pins = {}
    for row in decode_jsonl(authenticated(PARTITIONS, PARTITIONS_SHA256), "partition source"):
        hashes = row["input_hashes"]
        triple = {key: hashes[key] for key in ("train", "validation", "projection")}
        if pins.setdefault(row["dataset"], triple) != triple:
            raise ValueError("partition source has two digests for one lineage")
    return pins


def load_pilot():
    """Execute only the authenticated evaluator bytes; never reread them for code."""
    captured = authenticated(PILOT, PILOT_SHA256)
    loader = SourceFileLoader("pinned_pilot_metrics", str(PILOT))
    loader.get_code = lambda _name: loader.source_to_code(captured, str(PILOT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def stage_lineage(dataset: str, pins: dict, work: Path) -> dict:
    """Copy authenticated partition bytes so the evaluator never reads shared scratch."""
    pin = pins.get(dataset)
    if pin is None:
        raise ValueError("lineage is outside the published partition source")
    source = WORKERS / dataset
    projection = json.loads(authenticated(source / "projection.json", pin["projection"], WORKERS))
    staged = work / "partitions" / dataset
    staged.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("train", "validation"):
        raw = authenticated(source / f"{name}.csv", pin[name], WORKERS)
        target = staged / f"{name}.csv"
        target.write_bytes(raw)
        paths[name] = target
    return {"task": projection["task"], "train": paths["train"], "validation": paths["validation"],
            "train_sha256": pin["train"], "validation_sha256": pin["validation"], "root": staged}


def rescore(cell: dict, staged: dict, samples: Path, work: Path, pilot) -> tuple[dict, dict]:
    import numpy as np

    dataset, size, seed = identity_of(cell)
    name = f"sample-{seed}-{size}n.csv"
    raw = authenticated(samples / dataset / name, cell["synthetic_sha256"], samples)
    # Same decode and headerless rewrite as tabsyn_sample.score_samples.
    table = np.loadtxt(io.BytesIO(raw), delimiter=",", skiprows=1, ndmin=2)
    headerless = work / "headerless" / dataset / f"sample-{seed}-{size}n.headerless.csv"
    headerless.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(headerless, table, delimiter=",")
    report = pilot.measure(staged["train"], staged["validation"], headerless, staged["task"])
    if report.get("implementation_sha256") != PILOT_SHA256:
        raise ValueError("evaluator reported a different implementation digest")
    if report.get("dependencies") != DEPENDENCIES:
        raise ValueError("evaluator runtime differs from the declared dependencies")
    public = {
        "dataset": dataset, "sample_seed": seed, "size": size, "status": "ok",
        "synthetic_sha256": hashlib.sha256(raw).hexdigest(),
        "null_loss": report["null_loss"], "utility": report["utility"], "rows": report["rows"],
        "official_tests_opened": False, "formal_dp": False,
    }
    receipt = {
        "dataset": dataset, "size": size, "sample_seed": seed,
        "synthetic_sha256": public["synthetic_sha256"],
        "cell_sha256": cell_sha256(public),
        "implementation_sha256": report["implementation_sha256"],
        "dependencies": report["dependencies"],
        "train_sha256": staged["train_sha256"],
        "validation_sha256": staged["validation_sha256"],
        "official_tests_opened": False, "formal_dp": False,
    }
    return public, receipt


def max_difference(left, right) -> float | None:
    """Largest absolute numeric difference; None when the structures differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return None
        values = [max_difference(left[key], right[key]) for key in left]
        return None if any(value is None for value in values) else max(values, default=0.0)
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        return 0.0 if left == right else None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True, help="authenticated local sample copies")
    parser.add_argument("--work", type=Path, required=True, help="scratch for staged partitions")
    parser.add_argument("--out", type=Path, default=RECEIPTS)
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    if platform.node() != args.host:
        raise SystemExit(f"refusing to run on {platform.node()}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    cells = decode_jsonl(authenticated(LEDGER, LEDGER_SHA256), "TabSyn ledger")
    assert_unique_cells(cells)
    pins = partition_pins()
    pilot = load_pilot()
    receipts, mismatches = [], []
    staged, current = None, None
    for index, cell in enumerate(cells, 1):
        if cell["dataset"] != current:
            if staged is not None:
                shutil.rmtree(staged["root"])
            current = cell["dataset"]
            staged = stage_lineage(current, pins, args.work)
        public, receipt = rescore(cell, staged, args.samples, args.work, pilot)
        if canonical_line(public) == canonical_line(cell):
            receipts.append(receipt)
        else:
            mismatches.append({"identity": list(identity_of(cell)),
                               "max_abs_difference": max_difference(public, cell)})
        if index % 30 == 0:
            print(f"re-scored {index} of {len(cells)}; mismatched {len(mismatches)}", flush=True)
    if staged is not None:
        shutil.rmtree(staged["root"])
    if mismatches:
        report = args.work / "evaluator-replay-mismatch.json"
        report.write_text(json.dumps({"mismatched": mismatches, "cells": len(cells)}, indent=2) + "\n")
        raise SystemExit(f"{len(mismatches)} of {len(cells)} re-scored cells differ; see {report}")
    text = canonical_jsonl(receipts, receipt_identity)
    args.out.write_text(text)
    meta = {
        "format": FORMAT,
        "cell_count": len(receipts),
        "matched": len(receipts),
        "mismatched": 0,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "ledger_sha256": LEDGER_SHA256,
        "implementation_sha256": PILOT_SHA256,
        "dependencies": DEPENDENCIES,
        "partition_source": {"path": PARTITIONS.relative_to(REPO).as_posix(), "sha256": PARTITIONS_SHA256},
        "sample_source": SAMPLE_SOURCE + "; each CSV authenticated by the ledger synthetic_sha256 before decode",
        "host": args.host,
        "threads": "OMP, OpenBLAS and MKL at 1; CatBoost thread_count 4 from the evaluator",
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule": "a receipt exists only for a re-scored public cell byte-identical to its ledger line",
        "official_tests_opened": False,
        "formal_dp": False,
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"TabSyn evaluator receipts {len(receipts)} {meta['sha256']}")


if __name__ == "__main__":
    main()
