#!/usr/bin/env python3
"""Paired DOPE-versus-control tables from scored control cells.

Reads cell JSON written by research/benchmark/review_fixes/controls.py.
Split-seed cells stay in their own table. Fit seeds 23, 37, 53, and 71 are
all reported. No cell is dropped because of its retention.
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
    primary_map,
    reduce_cells,
    tex_name,
    write_tex,
    _load,
    _sha,
    _table,
    ci,
)
from research.benchmark.review_fixes.stats import (  # noqa: E402
    cluster_of,
    holm,
    median_ci,
    tost_mean,
    wilcoxon_p,
    win_tie_loss,
)

AUDITORS = ("catboost", "linear", "mlp")
SAMPLE_SEEDS = (101, 211, 307)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _cells(root: Path) -> list[dict]:
    found = []
    if not root.is_dir():
        return found
    for path in sorted(root.glob("*/*.json")):
        if path.name.endswith(".tmp"):
            continue
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict):
            found.append(document)
    return found


def _median_map(cells: list[dict], kind: str, fit_seed, size: int) -> dict:
    grouped = defaultdict(dict)
    for cell in cells:
        if cell.get("kind") != kind or cell.get("status") != "ok":
            continue
        if fit_seed is not None and cell.get("fit_seed") != fit_seed:
            continue
        if cell.get("split_seed") not in (None,):
            continue
        if cell.get("size") != size or cell.get("sample_seed") not in SAMPLE_SEEDS:
            continue
        slot = grouped[(cell.get("dataset"), size)]
        seed = cell["sample_seed"]
        if seed in slot:
            raise ValueError(f"duplicate control cell {kind} {cell.get('dataset')} {size} {seed}")
        slot[seed] = cell
    found = {auditor: {} for auditor in AUDITORS}
    for (dataset, _size), seeds in grouped.items():
        if set(seeds) != set(SAMPLE_SEEDS) or not isinstance(dataset, str):
            continue
        for auditor in AUDITORS:
            values = []
            keep = True
            for seed in SAMPLE_SEEDS:
                block = ((seeds[seed].get("metrics") or {}).get("utility") or {}).get(auditor) or {}
                retention = block.get("retention")
                if block.get("informative") is not True or not _finite(retention):
                    keep = False
                    break
                values.append(float(retention))
            if keep:
                found[auditor][dataset] = float(sorted(values)[1])
    return found


def _paired(left: dict, right: dict, names: dict, label: str) -> dict:
    keys = sorted(set(left) & set(right))
    diff = [left[key] - right[key] for key in keys]
    clusters = [cluster_of(names.get(key, key), key) for key in keys]
    summary = median_ci(diff, label, clusters)
    summary.update(win_tie_loss(diff))
    summary["wilcoxon_p"] = wilcoxon_p(diff)
    summary["tost"] = tost_mean(diff)
    summary["mean"] = summary["tost"]["mean"]
    summary["n_intersection"] = len(keys)
    return summary


def _split_rows(cells, names) -> list[dict]:
    rows = []
    for split_seed in (2027, 2999, 4099, 8191):
        for size in (1, 4):
            grouped = defaultdict(dict)
            for cell in cells:
                if cell.get("kind") != "split" or cell.get("status") != "ok":
                    continue
                if cell.get("split_seed") != split_seed or cell.get("fit_seed") != 11:
                    continue
                if cell.get("size") != size or cell.get("sample_seed") not in SAMPLE_SEEDS:
                    continue
                grouped[cell.get("dataset")][cell["sample_seed"]] = cell
            values = {}
            for dataset, seeds in grouped.items():
                if set(seeds) != set(SAMPLE_SEEDS) or not isinstance(dataset, str):
                    continue
                triple = []
                keep = True
                for seed in SAMPLE_SEEDS:
                    block = ((seeds[seed].get("metrics") or {}).get("utility") or {}).get("catboost") or {}
                    retention = block.get("retention")
                    if block.get("informative") is not True or not _finite(retention):
                        keep = False
                        break
                    triple.append(float(retention))
                if keep:
                    values[dataset] = float(sorted(triple)[1])
            summary = median_ci(
                [values[key] for key in sorted(values)],
                f"split|{split_seed}|{size}|catboost",
                [cluster_of(names.get(key, key), key) for key in sorted(values)],
            )
            summary.update({"split_seed": split_seed, "size": size, "auditor": "catboost"})
            rows.append(summary)
    return rows


def render_controls(payload: dict) -> str:
    lines = []
    for row in payload["paired"]:
        flag = "yes" if row["tost"]["equivalent"] else "no"
        lines.append(
            f"{row['size']}$n$ & {tex_name(row['auditor'])} & {tex_name(row['method'])} & "
            f"{row['n']} & {fmt(row['median'])} & {ci(row)} & {fmt(row['mean'])} & {flag} & "
            f"{row['wins']}/{row['ties']}/{row['losses']} & {fmt(row['holm_p'], 3)} \\\\"
        )
    for row in payload["predictor_fit_seeds"]:
        if row["auditor"] != "catboost":
            continue
        lines.append(
            f"{row['size']}$n$ & catboost & predictor fit {row['fit_seed']} & "
            f"{row['n']} & {fmt(row['median'])} & {ci(row)} & --- & --- & --- & --- \\\\"
        )
    for row in payload["split_seeds"]:
        lines.append(
            f"{row['size']}$n$ & catboost & split {row['split_seed']} & "
            f"{row['n']} & {fmt(row['median'])} & {ci(row)} & --- & --- & --- & --- \\\\"
        )
    return (
        "% Size n Holm family has 3 tests (predictor-only times three auditors). "
        "real\\_bootstrap\\_4n has no size-n arm, so it is not a member of that family. "
        "Size 4n Holm family has 6 tests. "
        "Predictor fit seeds 23, 37, 53, and 71 are all reported and are not in the Holm family. "
        "Split-seed rows are a separate sensitivity and are not pooled with fit seed 11. "
        "Wilcoxon and TOST treat lineages as iid. The interval is the family-cluster bootstrap.\n"
        + _table(
            "Size & Auditor & Control & $n$ & Median diff. or level & Hierarchical CI & Mean & TOST & W/T/L & Holm $p$",
            lines,
            "lllrrrrllr",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--from-panel", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.from_panel or (args.check and args.cells is None):
        panel_path = args.from_panel or (RESULTS / "review-fixes-controls-v1" / "panel.json")
        payload = json.loads(panel_path.read_text())
        rendered = render_controls(payload)
        tex_path = GENERATED / "review-controls.tex"
        if args.check:
            if not tex_path.is_file() or tex_path.read_text() != rendered:
                raise SystemExit("review-controls.tex does not match the panel")
            print("controls check ok")
            return
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if text != panel_path.read_text():
            raise SystemExit("controls panel is not canonical json")
        (GENERATED / "review-controls.json").write_text(text)
        write_tex("review-controls.tex", rendered)
        return
    if args.cells is None:
        print("controls not scored yet")
        return
    cells = _cells(args.cells)
    if not cells:
        print("controls not scored yet")
        return
    record, _record_sha = _load("s3-lineage-record.json")
    density, _density_sha = _load("density-matched-population-validation.json")
    names = names_of(record)
    dope = primary_map(reduce_cells(density["cells"], "DOPE", "features12_steps2048"))
    del density
    controls = {
        ("real_bootstrap_4n", "resample", 4): _median_map(cells, "real_bootstrap_4n", None, 4),
        ("predictor_only", "fit_seed_11", 1): _median_map(cells, "predictor_only", 11, 1),
        ("predictor_only", "fit_seed_11", 4): _median_map(cells, "predictor_only", 11, 4),
    }
    seed_rows = []
    for fit_seed in (11, 23, 37, 53, 71):
        for size in (1, 4):
            mapped = _median_map(cells, "predictor_only", fit_seed, size)
            for auditor in AUDITORS:
                summary = median_ci(
                    [mapped[auditor][key] for key in sorted(mapped[auditor])],
                    f"predictor-seed|{fit_seed}|{size}|{auditor}",
                    [cluster_of(names.get(key, key), key) for key in sorted(mapped[auditor])],
                )
                summary.update({"fit_seed": fit_seed, "size": size, "auditor": auditor, "kind": "predictor_only"})
                seed_rows.append(summary)
    paired_rows = []
    for size in (1, 4):
        drafted = []
        family = []
        specs = []
        if size == 4:
            specs.append(("real_bootstrap_4n", "resample", controls[("real_bootstrap_4n", "resample", 4)]))
        specs.append(("predictor_only", "fit_seed_11", controls[("predictor_only", "fit_seed_11", size)]))
        for auditor in AUDITORS:
            for method, configuration, mapped in specs:
                item = _paired(
                    dope[size][auditor],
                    mapped[auditor],
                    names,
                    f"paired|controls|{size}|{auditor}|{method}|{configuration}",
                )
                item.update({
                    "size": size, "auditor": auditor, "method": method, "configuration": configuration,
                })
                drafted.append(item)
                family.append(item["wilcoxon_p"])
        adjusted = holm(family)
        family_n = len(family)
        if size == 1:
            holm_note = (
                "size-n family is the three predictor-only auditor tests; "
                "real_bootstrap_4n has no size-n arm in the predeclaration"
            )
        else:
            holm_note = (
                "size-4n family is six tests: three auditors times "
                "real_bootstrap_4n and predictor_only"
            )
        for item, p_value in zip(drafted, adjusted):
            item["holm_p"] = p_value
            item["holm_family_n"] = family_n
            item["holm_note"] = holm_note
            item["tests_treat_lineages_as_iid"] = True
            paired_rows.append(item)
    payload = {
        "format": "dope-review-fix-controls",
        "version": 1,
        "cells": len(cells),
        "predeclaration_sha256": _sha(REPO / "research" / "benchmark" / "review_fixes" / "predeclare.json"),
        "paired": paired_rows,
        "predictor_fit_seeds": seed_rows,
        "split_seeds": _split_rows(cells, names),
        "claims": {"formal_dp": False, "mfs_v2": None, "ptf_v1": None, "release_safe_l3": None, "superiority": None},
    }
    out = RESULTS / "review-fixes-controls-v1"
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out / "panel.json").write_text(text)
    (GENERATED / "review-controls.json").write_text(text)
    write_tex("review-controls.tex", render_controls(payload))
    catboost = [row for row in paired_rows if row["auditor"] == "catboost"]
    for row in catboost:
        print(
            f"DOPE minus {row['method']} {row['configuration']} CatBoost size {row['size']}n: "
            f"median {fmt(row['median'])} {ci(row)} on {row['n']} lineages; "
            f"TOST {'equivalent' if row['tost']['equivalent'] else 'not equivalent'} at ±0.02."
        )
    if args.check and not (GENERATED / "review-controls.tex").is_file():
        raise SystemExit("review-controls.tex was not written")


if __name__ == "__main__":
    main()
