#!/usr/bin/env python3
"""Aggregate holdout privacy and GBDT C2ST cell JSON into one table.

The cell files are the output of research/benchmark/review_fixes/privacy_fidelity.py.
payload_from_cells reduces the bound scalar ledger to the committed panel, and the
table takes its DCR reference sizes from that ledger. Logistic C2ST stored on the
population ledgers is not relabeled here.
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
    ci,
)
from research.benchmark.review_fixes.bind_privacy_cells import decode_jsonl, reference_rows  # noqa: E402
from research.benchmark.review_fixes.review_tex import assert_distinct, longtable  # noqa: E402
from research.benchmark.review_fixes.stats import cluster_of, median_ci  # noqa: E402

from research.benchmark.expanded_validation_metrics import MAX_ROWS  # noqa: E402

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


def _c2st_rows(cell: dict) -> int | None:
    if not _finite(cell.get("c2st_catboost_auc")):
        return None
    rows = cell.get("rows") or {}
    validation = rows.get("validation")
    synthetic = rows.get("synthetic")
    count = cell.get("c2st_rows_per_class")
    if (
        type(validation) is not int or validation < 1
        or type(synthetic) is not int or synthetic < 1
        or type(count) is not int
        or count != min(validation, synthetic, MAX_ROWS)
    ):
        raise ValueError("privacy C2ST class-n disagrees with declared rows")
    return count


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


METRIC_LABELS = {
    "dcr_validation_median": "holdout DCR",
    "dcr_fit_median": "fit DCR",
    "nndr_fit_median": "fit NNDR",
    "distance_mia_auc": "distance MIA AUC",
    "c2st_catboost_auc": "CatBoost C2ST AUC",
}


def _count(value) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{float(value):,.1f}"


def ledger_references(panel_path: Path, payload: dict) -> dict:
    """Reference sizes come from the bound scalar ledger beside the panel."""
    ledger = panel_path.with_name("scalar-cells.jsonl")
    if ledger.is_symlink() or not ledger.is_file():
        raise SystemExit("privacy table needs the bound scalar ledger beside its panel")
    cells = decode_jsonl(ledger.read_bytes())
    if len(cells) != payload["cells"]:
        raise SystemExit("privacy scalar ledger does not match the panel cell count")
    return reference_rows(cells)


def render_privacy(payload: dict, references: dict) -> str:
    lines, keys = [], []
    for cohort, rows in (("own lineages", payload["summaries"]), ("matched density", payload["matched_density"])):
        for row in rows:
            keys.append((row["method"], row["configuration"], cohort, row["size"], row["metric"]))
            lines.append(
                f"{tex_name(row['method'])} & {tex_name(row['configuration'])} & {cohort} & {row['size']}$n$ & "
                f"{METRIC_LABELS[row['metric']]} & {row['n']} & {fmt(row['median'])} & {ci(row)} \\\\"
            )
    assert_distinct(keys, "privacy table")
    unavailable = "; ".join(
        f"{tex_name(row['method'])} {row['n']} ({tex_name(row['reason'])})" for row in payload["unavailable"]
    ) or "none"
    caption = (
        "\\textbf{Holdout privacy and CatBoost detection, fit seed 11.} Rows cover the published synthetic samples. "
        "Each value is the median over lineages of each lineage's median over sample seeds 101, 211 and 307; "
        "$n$ counts lineages with three finite seeds. Own-lineage rows keep every scored lineage whose "
        "synthetic row count equals fit rows times size; matched-density rows use the lineages shared by "
        "DOPE, GaussianCopula, Chow-Liu and independent marginals. Interval: 95\\% family-cluster bootstrap "
        "percentile interval of the median, 10{,}000 draws. Holdout DCR uses each lineage's validation rows "
        f"as the reference set (median {_count(references['holdout_median'])} rows); fit DCR and fit NNDR use "
        f"its training rows (median {_count(references['fit_median'])} rows, a median "
        f"{references['fit_to_holdout_median']:.1f} times as many), so the two distances use reference sets "
        "of different sizes. Distance MIA AUC is an empirical membership attack, and CatBoost C2ST uses "
        "min(validation, synthetic, 800) rows per class. Empirical attack scores are not formal differential "
        "privacy and not HIPAA de-identification. Unavailable cells stay out of every median and are not "
        f"wins: {unavailable}. No Holm family."
    )
    note = (
        "% c2st\\_catboost\\_auc is the GBDT detector. Stored logistic C2ST is a different table. "
        f"Reference sizes cover {references['lineages']} scored lineages.\n"
    )
    return longtable(
        caption,
        "Method & Configuration & Cohort & Size & Metric & $n$ & Median & 95\\% CI",
        lines,
        "lllllrrr",
        note,
        label="tab:review-privacy",
    )


def payload_from_cells(cells: list[dict]) -> dict:
    """Reduce scalar privacy cells to the committed aggregate. Writes nothing."""
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
        count = _c2st_rows(cell)
        if count is not None:
            rows_per_class.append(count)
            declared_rows[(cell.get("dataset"), size, seed)].add(count)
    if any(len(values) > 1 for values in declared_rows.values()):
        raise ValueError("privacy C2ST class-n differs across methods for a dataset/size/seed")
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
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--from-panel", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.from_panel or (args.check and args.cells is None):
        panel_path = args.from_panel or (RESULTS / "review-fixes-privacy-v1" / "panel.json")
        payload = json.loads(panel_path.read_text())
        rendered = render_privacy(payload, ledger_references(panel_path, payload))
        tex_path = GENERATED / "review-privacy.tex"
        json_path = GENERATED / "review-privacy.json"
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.check:
            if not tex_path.is_file() or tex_path.read_text() != rendered:
                raise SystemExit("review-privacy.tex does not match the panel")
            if not json_path.is_file() or json_path.read_text() != text:
                raise SystemExit("review-privacy.json does not match the panel")
            print("privacy check ok")
            return
        if text != panel_path.read_text():
            raise SystemExit("privacy panel is not canonical json")
        (GENERATED / "review-privacy.json").write_text(text)
        write_tex("review-privacy.tex", rendered)
        return
    if args.cells is None:
        print("privacy cells not scored yet")
        return
    cells = _cells(args.cells)
    if not cells:
        print("privacy cells not scored yet")
        return
    payload = payload_from_cells(cells)
    out = RESULTS / "review-fixes-privacy-v1"
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out / "panel.json").write_text(text)
    (GENERATED / "review-privacy.json").write_text(text)
    write_tex("review-privacy.tex", render_privacy(payload, reference_rows(cells)))
    holdout = [row for row in payload["summaries"] if row["metric"] == "dcr_validation_median" and row["method"] == "DOPE"]
    for row in holdout:
        print(
            f"DOPE holdout DCR size {row['size']}n median {fmt(row['median'])} "
            f"{ci(row)} on {row['n']} lineages. Empirical only, not formal DP."
        )
    if args.check and not (GENERATED / "review-privacy.tex").is_file():
        raise SystemExit("review-privacy.tex was not written")


if __name__ == "__main__":
    main()
