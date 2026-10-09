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
DENSITY_MATCH = (
    ("DOPE", "features12_steps2048"),
    ("GaussianCopula", "native_selected"),
    ("Chow-Liu", "native_selected"),
    ("independent_marginals", "native_selected"),
)


def _matched_density(grouped, names) -> list[dict]:
    rows = []
    for size in (1, 4):
        for metric in METRICS:
            per_method = {}
            for method, configuration in DENSITY_MATCH:
                values = {}
                for (method_i, configuration_i, dataset, size_i), seeds in grouped.items():
                    if (method_i, configuration_i, size_i) != (method, configuration, size):
                        continue
                    if set(seeds) != {101, 211, 307} or not isinstance(dataset, str):
                        continue
                    triple = [seeds[seed].get(metric) for seed in (101, 211, 307)]
                    if all(_finite(value) for value in triple):
                        values[dataset] = float(sorted(float(value) for value in triple)[1])
                per_method[(method, configuration)] = values
            present = [set(values) for values in per_method.values()]
            intersection = set.intersection(*present) if present else set()
            for method, configuration in DENSITY_MATCH:
                chosen = {
                    dataset: per_method[(method, configuration)][dataset]
                    for dataset in sorted(intersection)
                }
                summary = median_ci(
                    [chosen[key] for key in sorted(chosen)],
                    f"privacy-matched|{method}|{configuration}|{size}|{metric}",
                    [cluster_of(names.get(key, key), key) for key in sorted(chosen)],
                )
                summary.update({
                    "method": method,
                    "configuration": configuration,
                    "size": size,
                    "metric": metric,
                    "coverage": "matched density intersection",
                })
                rows.append(summary)
    return rows


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _cells(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".tmp") or not path.is_file():
            continue
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed privacy cell {path}") from error
        if not isinstance(document, dict):
            raise ValueError(f"malformed privacy cell {path}")
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
    unavailable = defaultdict(int)
    size_mismatch = 0
    declared_rows = defaultdict(set)
    for cell in cells:
        if cell.get("status") != "ok":
            unavailable[(cell.get("method"), cell.get("reason"))] += 1
            continue
        if cell.get("sample_seed") not in (101, 211, 307):
            continue
        rows = cell.get("rows") or {}
        fit_rows = rows.get("fit")
        synthetic_rows = rows.get("synthetic")
        size = cell.get("size")
        declared = (
            isinstance(fit_rows, int) and isinstance(synthetic_rows, int)
            and size in (1, 4) and synthetic_rows == fit_rows * size
        )
        if not declared:
            size_mismatch += 1
            continue
        key = (cell.get("method"), cell.get("configuration"), cell.get("dataset"), size)
        seed = cell["sample_seed"]
        if seed in grouped[key]:
            raise ValueError(f"duplicate privacy cell {key} {seed}")
        grouped[key][seed] = cell
        if _finite(cell.get("c2st_rows_per_class")):
            rows_per_class.append(int(cell["c2st_rows_per_class"]))
            declared_rows[(cell.get("dataset"), size, seed)].add(int(cell["c2st_rows_per_class"]))
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
                "coverage": "unmatched; declared synthetic row count equals fit rows times size",
            })
            summaries.append(summary)
    matched = _matched_density(grouped, names)
    payload = {
        "format": "dope-review-fix-privacy",
        "version": 1,
        "cells": len(cells),
        "predeclaration_sha256": _sha(REPO / "research" / "benchmark" / "review_fixes" / "predeclare.json"),
        "rows_per_class": {
            "n": len(rows_per_class),
            "min": min(rows_per_class) if rows_per_class else None,
            "max": max(rows_per_class) if rows_per_class else None,
            "dataset_size_seed_disagreements": sum(1 for values in declared_rows.values() if len(values) > 1),
        },
        "size_mismatch_cells": size_mismatch,
        "unavailable": [
            {"method": method, "reason": reason, "n": count}
            for (method, reason), count in sorted(unavailable.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
        ],
        "summaries": summaries,
        "matched_density": matched,
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
    for row in matched:
        lines.append(
            f"{tex_name(row['method'])} & matched density & {row['size']}$n$ & "
            f"{tex_name(row['metric'])} & {row['n']} & {fmt(row['median'])} & {ci(row)} \\\\"
        )
    write_tex("review-privacy.tex", (
        "% Empirical attack metrics only. Not formal differential privacy and not HIPAA de-identification. "
        "c2st\\_catboost\\_auc is the GBDT detector. Stored logistic C2ST is a different table. "
        "Unmatched rows keep each method's own lineages whose synthetic row count equals fit rows times size. "
        "Matched density rows use the intersection of DOPE, GaussianCopula, Chow-Liu, and independent marginals. "
        "Unavailable cells are not wins. "
        + " ".join(
            f"{tex_name(row['method'])} {row['reason']} {row['n']}."
            for row in payload["unavailable"]
        )
        + "\n"
    ) + _table(
        "Method & Configuration & Size & Metric & $n$ lineages & Median & Hierarchical CI",
        lines,
        "lllllrr",
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
