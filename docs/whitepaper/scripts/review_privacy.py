#!/usr/bin/env python3
"""Aggregate holdout privacy and GBDT C2ST cell JSON into one table.

The cell files are the output of research/benchmark/review_fixes/privacy_fidelity.py.
Logistic C2ST stored on the population ledgers is not relabeled here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from research.benchmark.review_fixes.receipt_panel import (  # noqa: E402
    GENERATED,
    RESULTS,
    fmt,
    names_of,
    tex_name,
    write_tex,
    _load,
    _sha,
    _table,
    ci,
)
from research.benchmark.review_fixes.stats import cluster_of, median_ci  # noqa: E402

METRICS = (
    "dcr_validation_median",
    "dcr_fit_median",
    "nndr_fit_median",
    "distance_mia_auc",
    "c2st_catboost_auc",
)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _cells(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.glob("*/*/*/*.json")):
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict):
            found.append(document)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cells = _cells(args.cells)
    if not cells:
        print("privacy cells not scored yet")
        return
    record, _sha_record = _load("s3-lineage-record.json")
    names = names_of(record)
    grouped = defaultdict(dict)
    rows_per_class = []
    for cell in cells:
        if cell.get("status") != "ok" or cell.get("sample_seed") not in (101, 211, 307):
            continue
        key = (cell.get("method"), cell.get("configuration"), cell.get("dataset"), cell.get("size"))
        seed = cell["sample_seed"]
        if seed in grouped[key]:
            raise ValueError(f"duplicate privacy cell {key} {seed}")
        grouped[key][seed] = cell
        if _finite(cell.get("c2st_rows_per_class")):
            rows_per_class.append(int(cell["c2st_rows_per_class"]))
    summaries = []
    for (method, configuration, size) in sorted({(key[0], key[1], key[3]) for key in grouped}):
        for metric in METRICS:
            values = {}
            for (method_i, configuration_i, dataset, size_i), seeds in grouped.items():
                if (method_i, configuration_i, size_i) != (method, configuration, size):
                    continue
                if set(seeds) != {101, 211, 307} or not isinstance(dataset, str):
                    continue
                triple = [seeds[seed].get(metric) for seed in (101, 211, 307)]
                if all(_finite(value) for value in triple):
                    values[dataset] = float(sorted(float(value) for value in triple)[1])
            summary = median_ci(
                [values[key] for key in sorted(values)],
                f"privacy|{method}|{configuration}|{size}|{metric}",
                [cluster_of(names.get(key, key), key) for key in sorted(values)],
            )
            summary.update({
                "method": method,
                "configuration": configuration,
                "size": size,
                "metric": metric,
            })
            summaries.append(summary)
    payload = {
        "format": "dope-review-fix-privacy",
        "version": 1,
        "cells": len(cells),
        "predeclaration_sha256": _sha(REPO / "research" / "benchmark" / "review_fixes" / "predeclare.json"),
        "rows_per_class": {
            "n": len(rows_per_class),
            "min": min(rows_per_class) if rows_per_class else None,
            "max": max(rows_per_class) if rows_per_class else None,
        },
        "summaries": summaries,
        "note": "distance_mia_auc is an empirical attack AUC. It is not formal differential privacy and not HIPAA de-identification. c2st_catboost_auc is the GBDT detector. Stored logistic C2ST is a different column.",
        "claims": {"formal_dp": False, "hipaa_deidentification": False, "mfs_v2": None, "ptf_v1": None,
                   "release_safe_l3": None, "superiority": None},
    }
    out = RESULTS / "review-fixes-privacy-v1"
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out / "panel.json").write_text(text)
    (GENERATED / "review-privacy.json").write_text(text)
    lines = []
    for row in summaries:
        lines.append(
            f"{tex_name(row['method'])} & {tex_name(row['configuration'])} & {row['size']}$n$ & "
            f"{tex_name(row['metric'])} & {row['n']} & {fmt(row['median'])} & {ci(row)} \\\\"
        )
    write_tex("review-privacy.tex", _table(
        "Method & Configuration & Size & Metric & $n$ lineages & Median & Hierarchical CI",
        lines,
    ))
    holdout = [row for row in summaries if row["metric"] == "dcr_validation_median" and row["method"] == "DOPE"]
    for row in holdout:
        print(
            f"DOPE holdout DCR size {row['size']}n median {fmt(row['median'])} "
            f"{ci(row)} on {row['n']} lineages. Empirical only, not formal DP."
        )
    if args.check and not (GENERATED / "review-privacy.tex").is_file():
        raise SystemExit("review-privacy.tex was not written")


if __name__ == "__main__":
    main()
