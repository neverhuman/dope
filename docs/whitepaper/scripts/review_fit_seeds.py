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
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

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
    DRAWS,
    SIMULATED,
    cluster_of,
    family_of,
    median_ci,
    rng_for,
)

ESTIMAND = "median of lineage medians of completed fit-seed retentions"

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
        if row.get("size") not in SIZES:
            raise SystemExit("fit-seed metric is outside the declared grid")
        if row.get("status") == "ok" and (
            row.get("method") != "DOPE" or row.get("configuration") != "features12_steps2048"
        ):
            raise SystemExit("fit-seed metric is outside the declared grid")
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


def _on_grid(row: dict) -> bool:
    return (
        row.get("fit_seed") in FIT_SEEDS
        and row.get("sample_seed") in SAMPLE_SEEDS
        and row.get("size") in SIZES
        and isinstance(row.get("dataset"), str)
        and bool(row["dataset"])
    )


def triples(rows: list[dict], planned_datasets: list[str] | None = None) -> dict:
    """Median of three sample seeds. An incomplete triple is counted and unused."""
    buckets = defaultdict(dict)
    names = {}
    failed_cells = 0
    for row in rows:
        if not _on_grid(row):
            raise SystemExit("fit-seed metric is outside the declared grid")
        if row.get("status") != "ok":
            failed_cells += 1
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
    if planned_datasets is None:
        datasets = sorted({row["dataset"] for row in rows})
    else:
        if not isinstance(planned_datasets, list) or not all(isinstance(item, str) and item for item in planned_datasets):
            raise SystemExit("fit-seed planned grid is not a dataset list")
        datasets = planned_datasets
    present = {(row["dataset"], row["fit_seed"], row["size"]) for row in rows}
    missing = sum(
        1
        for dataset in datasets
        for fit_seed in FIT_SEEDS
        for size in SIZES
        if (dataset, fit_seed, size) not in present
    )
    return {
        "complete": complete,
        "incomplete_triples": incomplete,
        "missing_triples": missing,
        "failed_cells": failed_cells,
        "names": names,
    }


def _empty_summary() -> dict:
    return {
        "n": 0,
        "n_fits": 0,
        "median": None,
        "lo": None,
        "hi": None,
        "iid_lo": None,
        "iid_hi": None,
        "estimand": ESTIMAND,
    }


def _nested_summary(items: list[dict], label: str) -> dict:
    """Family, then lineage, then fit. Each lineage has equal weight."""
    if not items:
        return _empty_summary()
    lineages: dict[str, dict] = {}
    for item in items:
        slot = lineages.setdefault(item["dataset"], {"cluster": item["cluster"], "values": []})
        slot["values"].append(float(item["retention"]))
    n = len(lineages)
    n_fits = sum(len(slot["values"]) for slot in lineages.values())
    lineage_points = [float(np.median(slot["values"])) for slot in lineages.values()]
    point = float(np.median(lineage_points))
    if n == 1:
        return {
            "n": 1,
            "n_fits": n_fits,
            "median": point,
            "lo": point,
            "hi": point,
            "iid_lo": point,
            "iid_hi": point,
            "estimand": ESTIMAND,
        }
    clusters: dict[str, list[str]] = {}
    for dataset, slot in lineages.items():
        clusters.setdefault(slot["cluster"], []).append(dataset)
    one_each = all(len(members) == 1 for members in clusters.values()) and all(
        len(slot["values"]) == 1 for slot in lineages.values()
    )
    if one_each:
        base = median_ci(
            [item["retention"] for item in items],
            label,
            [item["cluster"] for item in items],
        )
        base["n_fits"] = n_fits
        base["estimand"] = ESTIMAND
        return base
    rng = rng_for(label + "|nested")
    cluster_ids = list(clusters)
    grouped = [
        [np.asarray(lineages[dataset]["values"], dtype=float) for dataset in clusters[cluster]]
        for cluster in cluster_ids
    ]
    n_clusters = len(grouped)
    samples = np.empty(DRAWS, dtype=float)
    for draw in range(DRAWS):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        lineage_draws = []
        for index in chosen:
            members = grouped[int(index)]
            picked = rng.integers(0, len(members), size=len(members))
            for member_index in picked:
                values = members[int(member_index)]
                take = rng.integers(0, len(values), size=len(values))
                lineage_draws.append(float(np.median(values[take])))
        samples[draw] = float(np.median(lineage_draws))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    iid = median_ci(lineage_points, label + "|lineage-iid")
    return {
        "n": n,
        "n_fits": n_fits,
        "median": point,
        "lo": float(lo),
        "hi": float(hi),
        "iid_lo": iid["lo"],
        "iid_hi": iid["hi"],
        "estimand": ESTIMAND,
    }


def _summary(items: list[dict], label: str) -> dict:
    return _nested_summary(items, label)


def _require_cells(panel: dict) -> None:
    cells = panel.get("cells") if isinstance(panel, dict) else None
    if not isinstance(cells, list):
        raise SystemExit("fit-seed panel cells are not a cell list")
    planned = panel.get("planned_datasets")
    if planned is not None and (
        not isinstance(planned, list) or not all(isinstance(item, str) and item for item in planned)
    ):
        raise SystemExit("fit-seed planned grid is not a dataset list")
    for row in cells:
        if not isinstance(row, dict) or not _on_grid(row):
            raise SystemExit("fit-seed panel cell is outside the declared grid")
        if row.get("official_tests_opened") is True:
            raise SystemExit("fit-seed panel opened an official test")
        if row.get("status") == "ok" and (
            row.get("method") != "DOPE" or row.get("configuration") != "features12_steps2048"
        ):
            raise SystemExit("fit-seed panel cell is outside the declared grid")


def reduce_panel(panel: dict) -> dict:
    reduced = triples(panel["cells"], planned_datasets=panel.get("planned_datasets"))
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
        "estimand": ESTIMAND,
        "scored_cells": len(panel["cells"]),
        "incomplete_triples": reduced["incomplete_triples"],
        "missing_triples": reduced["missing_triples"],
        "failed_cells": reduced["failed_cells"],
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
    if hier["n"] != 1 or hier.get("n_fits") != 1 or abs(hier["median"] - 0.3) > 1e-12:
        raise SystemExit("compact panel median drifted")
    weighted = []
    for sample_seed in SAMPLE_SEEDS:
        weighted.append(cell("aa000000000000aa", "feynman_aa", 23, sample_seed, 1, 0.1))
    for fit_seed in (23, 37):
        for sample_seed in SAMPLE_SEEDS:
            weighted.append(cell("bb000000000000bb", "feynman_bb", fit_seed, sample_seed, 1, 0.9))
    weighted_summary = reduce_panel(build_panel(weighted))
    weighted_hier = next(
        block for block in weighted_summary["blocks"]
        if block["kind"] == "hierarchical" and block["auditor"] == "catboost" and block["size"] == 1
    )
    if weighted_hier["n"] != 2 or weighted_hier["n_fits"] != 3 or abs(weighted_hier["median"] - 0.5) > 1e-12:
        raise SystemExit("lineage weight drifted")
    single = []
    for fit_seed, retention in ((23, 0.2), (37, 0.8)):
        for sample_seed in SAMPLE_SEEDS:
            single.append(cell("cc000000000000cc", "plain_cc", fit_seed, sample_seed, 1, retention))
    single_summary = reduce_panel(build_panel(single))
    single_hier = next(
        block for block in single_summary["blocks"]
        if block["kind"] == "hierarchical" and block["auditor"] == "catboost" and block["size"] == 1
    )
    if single_hier["n"] != 1 or single_hier["n_fits"] != 2 or abs(single_hier["median"] - 0.5) > 1e-12:
        raise SystemExit("single-lineage fit count drifted")
    planned = triples(rows, planned_datasets=[rows[0]["dataset"], "dd000000000000dd"])
    if planned["missing_triples"] != len(FIT_SEEDS) * len(SIZES) * 2 - 1 or planned["failed_cells"] != 0:
        raise SystemExit("missing planned triple was omitted")
    failed = cell("aa00000000000001", "plain", 37, 101, 1, None)
    failed["status"] = "failed"
    failed.pop("method", None)
    failed.pop("configuration", None)
    counted = triples(rows + [failed])
    if counted["failed_cells"] != 1 or any(item["fit_seed"] == 37 for item in counted["complete"]):
        raise SystemExit("failed cell was imputed")
    try:
        _require_cells({"cells": 3})
    except SystemExit as error:
        if "cell list" not in str(error):
            raise
    else:
        raise SystemExit("aggregate input was accepted")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sample-failed.metric.json").write_text(json.dumps({
            "status": "failed",
            "dataset": "aa00000000000001",
            "fit_seed": 23,
            "sample_seed": 101,
            "size": 1,
            "official_tests_opened": False,
        }))
        loaded = load_metrics(root)
        if len(loaded) != 1 or loaded[0].get("method") is not None:
            raise SystemExit("failure record aborted ingestion")
        (root / "sample-unprofiled.metric.json").write_text(json.dumps({
            "status": "ok",
            "dataset": "bb00000000000002",
            "fit_seed": 37,
            "sample_seed": 211,
            "size": 4,
            "official_tests_opened": False,
        }))
        try:
            load_metrics(root)
        except SystemExit as error:
            if "declared grid" not in str(error):
                raise
        else:
            raise SystemExit("ok record without a profile was accepted")
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
    _require_cells(panel)
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
