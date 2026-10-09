#!/usr/bin/env python3
"""Hierarchical bootstrap for DOPE fit seeds 23, 37, 53, and 71.

A finished sample cell is one metric JSON. A fit seed contributes only when
all three sample seeds have a finite retention. Missing seeds are not imputed.
This table is not the published fit-seed-11 result. Official test files are
not opened.
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
    AUDITORS,
    GENERATED,
    RESULTS,
    ci,
    fmt,
    tex_name,
    write_tex,
)
from research.benchmark.review_fixes.stats import (  # noqa: E402
    SIMULATED,
    cluster_of,
    family_of,
    median_ci,
)

FIT_SEEDS = (23, 37, 53, 71)
SAMPLE_SEEDS = (101, 211, 307)
SIZES = (1, 4)
PANEL = RESULTS / "review-fixes-fit-seeds-v1" / "panel.json"
JSON_OUT = GENERATED / "review-fit-seeds.json"
TEX_OUT = GENERATED / "review-fit-seeds.tex"


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_metrics(root: Path) -> list[dict]:
    found = []
    seen = set()
    for path in sorted(root.rglob("sample-*.metric.json")):
        if path.name == "test.csv" or "test.csv" in path.parts:
            raise SystemExit("refusing a path named test.csv")
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise SystemExit(f"malformed fit-seed metric {path.name}") from error
        if row.get("official_tests_opened") is True:
            raise SystemExit("fit-seed metric opened an official test")
        key = (row.get("dataset"), row.get("fit_seed"), row.get("sample_seed"), row.get("size"))
        if key in seen:
            raise SystemExit("duplicate fit-seed metric")
        seen.add(key)
        if row.get("fit_seed") not in FIT_SEEDS or row.get("sample_seed") not in SAMPLE_SEEDS:
            raise SystemExit("fit-seed metric is outside the declared seeds")
        if row.get("size") not in SIZES or row.get("method") != "DOPE":
            raise SystemExit("fit-seed metric is outside the declared grid")
        if row.get("configuration") != "features12_steps2048":
            raise SystemExit("fit-seed metric is not features12_steps2048")
        found.append(row)
    if not found:
        raise SystemExit("no fit-seed metric cells")
    return found


def _auditor_row(row: dict, auditor: str) -> dict:
    direct = row.get("utility")
    if isinstance(direct, dict) and auditor in direct:
        return direct.get(auditor) or {}
    return ((row.get("metrics") or {}).get("utility") or {}).get(auditor) or {}


def _retention(row: dict, auditor: str):
    utility = _auditor_row(row, auditor)
    if utility.get("informative") is True and _finite(utility.get("retention")):
        return float(utility["retention"])
    return None


def triples(rows: list[dict]) -> dict:
    """Median of three sample seeds. An incomplete triple is counted and unused."""
    buckets = defaultdict(dict)
    names = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        names[row["dataset"]] = row.get("display_name") or row["dataset"]
        for auditor in AUDITORS:
            key = (row["dataset"], row["fit_seed"], row["size"], auditor)
            buckets[key][row["sample_seed"]] = _retention(row, auditor)
    complete = []
    incomplete = 0
    for key, samples in sorted(buckets.items()):
        dataset, fit_seed, size, auditor = key
        if set(samples) != set(SAMPLE_SEEDS) or any(samples[seed] is None for seed in SAMPLE_SEEDS):
            incomplete += 1
            continue
        ordered = [samples[seed] for seed in SAMPLE_SEEDS]
        complete.append({
            "dataset": dataset,
            "display_name": names[dataset],
            "fit_seed": fit_seed,
            "size": size,
            "auditor": auditor,
            "retention": float(sorted(ordered)[1]),
            "cluster": cluster_of(names[dataset], dataset),
            "family": family_of(names[dataset]),
        })
    return {"complete": complete, "incomplete_triples": incomplete, "names": names}


def _summary(items: list[dict], label: str) -> dict:
    if not items:
        return {"n": 0, "median": None, "lo": None, "hi": None}
    return median_ci(
        [item["retention"] for item in items],
        label,
        [item["cluster"] for item in items],
    )


def reduce_panel(panel: dict) -> dict:
    reduced = triples(panel["cells"])
    rows = reduced["complete"]
    blocks = []
    for size in SIZES:
        for auditor in AUDITORS:
            chosen = [item for item in rows if item["size"] == size and item["auditor"] == auditor]
            blocks.append({
                "kind": "hierarchical",
                "size": size,
                "auditor": auditor,
                "fit_seed": None,
                "family": None,
                **_summary(chosen, f"fit-seeds|{auditor}|size{size}|hier"),
            })
            kept = [item for item in chosen if item["family"] not in SIMULATED]
            blocks.append({
                "kind": "excluding_simulated",
                "size": size,
                "auditor": auditor,
                "fit_seed": None,
                "family": None,
                **_summary(kept, f"fit-seeds|{auditor}|size{size}|excluding-simulated"),
            })
            for fit_seed in FIT_SEEDS:
                seeded = [item for item in chosen if item["fit_seed"] == fit_seed]
                blocks.append({
                    "kind": "fit_seed",
                    "size": size,
                    "auditor": auditor,
                    "fit_seed": fit_seed,
                    "family": None,
                    **_summary(seeded, f"fit-seeds|{auditor}|size{size}|seed{fit_seed}"),
                })
            families = sorted({item["family"] for item in chosen})
            for family in families:
                grouped = [item for item in chosen if item["family"] == family]
                blocks.append({
                    "kind": "family",
                    "size": size,
                    "auditor": auditor,
                    "fit_seed": None,
                    "family": family,
                    **_summary(grouped, f"fit-seeds|{auditor}|size{size}|family:{family}"),
                })
    return {
        "format": "dope-review-fix-fit-seeds-summary-v1",
        "fit_seeds": list(FIT_SEEDS),
        "sample_seeds": list(SAMPLE_SEEDS),
        "sizes": list(SIZES),
        "profile": "features12_steps2048",
        "pooled_with_fit_seed_11": False,
        "missing_seeds_imputed": False,
        "official_tests_opened": False,
        "scored_cells": len(panel["cells"]),
        "incomplete_triples": reduced["incomplete_triples"],
        "blocks": blocks,
    }


def _compact(row: dict) -> dict:
    utility = {}
    for auditor in AUDITORS:
        source = ((row.get("metrics") or {}).get("utility") or {}).get(auditor) or {}
        utility[auditor] = {
            "informative": source.get("informative"),
            "retention": source.get("retention") if _finite(source.get("retention")) else None,
        }
    return {
        "dataset": row.get("dataset"),
        "display_name": row.get("display_name"),
        "method": "DOPE",
        "configuration": "features12_steps2048",
        "fit_seed": row.get("fit_seed"),
        "sample_seed": row.get("sample_seed"),
        "size": row.get("size"),
        "status": row.get("status"),
        "task": row.get("task"),
        "model_sha256": row.get("model_sha256"),
        "synthetic_sha256": row.get("synthetic_sha256"),
        "source_git_sha": row.get("source_git_sha"),
        "runner_sha256": row.get("runner_sha256"),
        "utility": utility,
        "official_tests_opened": False,
    }


def build_panel(rows: list[dict]) -> dict:
    return {
        "format": "dope-review-fix-fit-seeds-v1",
        "fit_seeds": list(FIT_SEEDS),
        "sample_seeds": list(SAMPLE_SEEDS),
        "sizes": list(SIZES),
        "profile": "features12_steps2048",
        "pooled_with_fit_seed_11": False,
        "missing_seeds_imputed": False,
        "official_tests_opened": False,
        "cells": [_compact(row) for row in rows],
    }


def render(summary: dict) -> str:
    lines = [
        "% Fit seeds 23, 37, 53, and 71. Not the published fit-seed-11 table.",
        "% A fit seed is included only when sample seeds 101, 211, and 307 all have a finite retention.",
        "\\begin{tabular}{lllrlrr}",
        "\\toprule",
        "Kind & Size & Auditor & Seed or family & $n$ & Median & Hierarchical CI \\\\",
        "\\midrule",
    ]
    for block in summary["blocks"]:
        if block["kind"] == "family" and block["auditor"] != "catboost":
            continue
        label = block["family"] or ("" if block["fit_seed"] is None else str(block["fit_seed"]))
        lines.append(
            f"{tex_name(block['kind'])} & {block['size']}$n$ & {tex_name(block['auditor'])} & "
            f"{tex_name(label)} & {block['n']} & {fmt(block['median'])} & {ci(block)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    return "\n".join(lines)


def emit(panel: dict) -> dict:
    summary = reduce_panel(panel)
    body = render(summary)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    PANEL.write_text(json.dumps(panel, sort_keys=True) + "\n")
    JSON_OUT.write_text(json.dumps(summary, sort_keys=True) + "\n")
    write_tex(TEX_OUT.name, body)
    return summary


def _check(panel: dict) -> None:
    summary = reduce_panel(panel)
    if JSON_OUT.read_text() != json.dumps(summary, sort_keys=True) + "\n":
        raise SystemExit("fit-seed json drift")
    if TEX_OUT.read_text() != render(summary):
        raise SystemExit("fit-seed tex drift")
    print("fit-seeds check ok", flush=True)


def _self_check() -> None:
    def cell(dataset, name, fit_seed, sample_seed, size, retention):
        return {
            "dataset": dataset,
            "display_name": name,
            "method": "DOPE",
            "configuration": "features12_steps2048",
            "fit_seed": fit_seed,
            "sample_seed": sample_seed,
            "size": size,
            "status": "ok",
            "official_tests_opened": False,
            "metrics": {"utility": {
                auditor: {"informative": retention is not None, "retention": retention}
                for auditor in AUDITORS
            }},
        }
    rows = []
    for sample_seed, retention in ((101, 0.1), (211, 0.3), (307, 0.5)):
        rows.append(cell("aa00000000000001", "plain", 23, sample_seed, 1, retention))
    reduced = triples(rows)
    if reduced["incomplete_triples"] != 0 or len(reduced["complete"]) != len(AUDITORS):
        raise SystemExit("complete triple was dropped")
    if abs(reduced["complete"][0]["retention"] - 0.3) > 1e-12:
        raise SystemExit("sample-seed median is not the middle value")
    partial = triples(rows[:2])
    if partial["complete"] or partial["incomplete_triples"] != len(AUDITORS):
        raise SystemExit("partial triple was imputed")
    summary = reduce_panel(build_panel(rows))
    hier = next(
        block for block in summary["blocks"]
        if block["kind"] == "hierarchical" and block["auditor"] == "catboost" and block["size"] == 1
    )
    if hier["n"] != 1 or abs(hier["median"] - 0.3) > 1e-12:
        raise SystemExit("compact panel median drifted")
    print("fit-seeds self-check ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-metrics", type=Path)
    parser.add_argument("--from-panel", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.from_metrics:
        panel = build_panel(load_metrics(args.from_metrics))
    elif args.from_panel:
        panel = json.loads(args.from_panel.read_text())
    else:
        raise SystemExit("pass --from-metrics or --from-panel")
    if panel.get("official_tests_opened") is True:
        raise SystemExit("fit-seed panel opened an official test")
    if args.check:
        _check(panel)
        return
    summary = emit(panel)
    catboost = next(
        block for block in summary["blocks"]
        if block["kind"] == "hierarchical" and block["auditor"] == "catboost" and block["size"] == 1
    )
    print(
        f"fit-seeds hierarchical catboost size 1 n={catboost['n']} "
        f"incomplete_triples={summary['incomplete_triples']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
