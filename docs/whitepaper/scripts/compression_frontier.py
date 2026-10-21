#!/usr/bin/env python3
"""The compression frontier: worst-auditor retention against charged bytes.

This is the front-page figure of the paper and the README. It reads committed,
SHA-256-pinned ledgers only, fits nothing, and opens no official test file.

For every generator on the v2 validation panel it computes, at synthetic size 4n:

* worst-auditor retention: for each fit, the sample-seed median retention of each
  task-trained auditor (CatBoost, linear, MLP); the fit value is the smallest of the
  three (the T_robust rule of the TDL contract); the lineage value is the median of
  its fit values; the level is the median over lineages with a nested
  cluster -> lineage -> fit bootstrap (research/benchmark/review_fixes/stats_v2.py);
* charged bytes: the median and interquartile range already published in
  docs/whitepaper/generated/v2-headline.json (model plus projection sidecar);
* retained lift per KiB: worst-auditor retention divided by median KiB, a
  diagnostic ratio that is only read beside the frontier, never alone.

Renderers (figure, README block, TeX) live in compression_frontier_render.py.

Only lineages on which all three auditors are informative enter, so every method
shares one denominator. Every number carries the label selection-exposed
validation, because the panel informed DOPE's profile choice.

Usage:
    python3 -B docs/whitepaper/scripts/compression_frontier.py          # write
    python3 -B docs/whitepaper/scripts/compression_frontier.py --check  # verify text outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1760000000")

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.benchmark.review_fixes.stats import cluster_of  # noqa: E402
from research.benchmark.review_fixes.stats_v2 import nested_level, nested_paired  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compression_frontier_render import (  # noqa: E402
    EVIDENCE, FIGURE_PDF, FIGURE_PNG, FULL_PANEL_MIN, render, text_outputs)

RESULTS = REPO / "research" / "benchmark" / "results"

INPUTS = {
    "cells": (RESULTS / "review-fixes-v2" / "scalar-cells.jsonl",
              "ea4168621dfaadfb5735d5df74d384e57f46f8e16b8053402b663a6960a94745"),
    "fits": (RESULTS / "review-fixes-v2" / "fits.jsonl",
             "723a83226ffb2bd510a23a06cb6363d5c4fd25b230dc4fea0d42c2a08d634687"),
    "headline": (REPO / "docs" / "whitepaper" / "generated" / "v2-headline.json",
                 "44cc100e1ea1eecf9ea3f2e39f83ddcf63b1edb8d23dbb002c4973c57f3c4833"),
    "lineages": (RESULTS / "s3-lineage-record.json",
                 "bbd0852c49f595315104261efa4eaa6f50db257e323b8c002aac6737a2dbe7d7"),
    "foundation": (REPO / "paper" / "generated" / "hero-frontier.csv",
                   "ec8932c113cfd58f91b8e8e166ff57794206470f4ea933e31f27a14d0169485e"),
}

AUDITORS = ("catboost", "linear", "mlp")
SIZE = 4

# (method, configuration, display, role). Order is the legend order.
ARMS = [
    ("DOPE", "headline", "DOPE (ours)", "generator"),
    ("TabSyn", "scaled_200_vae_1000_diffusion", "TabSyn", "generator"),
    ("ForestDiffusion/Forest-Flow", "native_selected", "Forest-Flow", "generator"),
    ("ARF", "author_default", "ARF", "generator"),
    ("TVAE", "native_selected", "TVAE", "generator"),
    ("GaussianCopula", "native_selected", "Gaussian copula", "generator"),
    ("Chow-Liu", "native_selected", "Chow-Liu tree", "generator"),
    ("CTGAN", "native_selected", "CTGAN", "generator"),
    ("independent_marginals", "native_selected", "Independent marginals", "generator"),
    ("DOPE", "product_default", "DOPE, untuned default", "diagnostic"),
    ("predictor_only", "control", "Predictor-only control", "control"),
    ("real_bootstrap_4n", "control", "Real rows resampled 4n", "control"),
]



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated(key: str) -> Path:
    path, expected = INPUTS[key]
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{path.relative_to(REPO)}: sha256 {actual} != pinned {expected}")
    return path


# ----------------------------------------------------------------------------- statistics

def lineage_names() -> dict[str, dict]:
    record = json.loads(authenticated("lineages").read_text())
    return {row["dataset"]: row for row in record["rows"]}


def worst_auditor_items(names: dict[str, dict]) -> tuple[dict, dict, set]:
    """Fit-level worst-auditor values and per-auditor values for every arm."""
    per_sample: dict = defaultdict(lambda: defaultdict(list))
    informative: dict = defaultdict(set)
    wanted = {(method, config) for method, config, _, _ in ARMS}
    with authenticated("cells").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["size"] != SIZE or (row["method"], row["configuration"]) not in wanted:
                continue
            for auditor in AUDITORS:
                cell = (row.get("utility") or {}).get(auditor)
                if cell and cell.get("informative") and cell.get("retention") is not None:
                    value = float(cell["retention"])
                    if math.isfinite(value):
                        key = (row["method"], row["configuration"], row["dataset"], row["fit_seed"])
                        per_sample[key][auditor].append(value)
                        informative[row["dataset"]].add(auditor)
    complete = {dataset for dataset, auditors in informative.items() if len(auditors) == len(AUDITORS)}
    worst: dict = defaultdict(list)
    each: dict = defaultdict(lambda: defaultdict(list))
    for (method, config, dataset, fit_seed), auditors in per_sample.items():
        if dataset not in complete:
            continue
        cluster = cluster_of(names[dataset]["display_name"], dataset)
        medians = {a: statistics.median(v) for a, v in auditors.items()}
        for auditor, value in medians.items():
            each[(method, config)][auditor].append(
                {"dataset": dataset, "fit_seed": fit_seed, "cluster": cluster, "value": value})
        if len(medians) == len(AUDITORS):
            worst[(method, config)].append(
                {"dataset": dataset, "fit_seed": fit_seed, "cluster": cluster, "value": min(medians.values())})
    return worst, each, complete


def headline_bytes() -> dict:
    payload = json.loads(authenticated("headline").read_text())
    out = {}
    for arm in payload["arms"]:
        stats = arm.get("bytes") or {}
        if stats.get("median") is not None:
            out[(arm["method"], arm["configuration"])] = {
                "median": float(stats["median"]), "q1": float(stats["q1"]), "q3": float(stats["q3"]),
                "n": int(stats["n"])}
    return out


def raw_table_bytes(names: dict[str, dict], complete: set) -> dict:
    """Size of each fit view stored as float32 (fit rows x (features + target) x 4 bytes)."""
    sizes = sorted(4 * row["fit_rows"] * (row["features"] + 1)
                   for dataset, row in names.items() if dataset in complete)
    q1, median, q3 = (float(x) for x in _quantiles(sizes))
    return {"median": median, "q1": q1, "q3": q3, "n": len(sizes)}


def dope_bits_per_cell(names: dict[str, dict], complete: set) -> dict:
    per_lineage = defaultdict(list)
    with authenticated("fits").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["method"] == "DOPE" and row["configuration"] == "headline" and row["status"] == "ok":
                per_lineage[row["dataset"]].append((row["charged_bytes"], row["artifact_bytes"],
                                                    row["projection_bytes"]))
    bits, ratios, sidecar = [], [], []
    for dataset, values in per_lineage.items():
        if dataset not in complete:
            continue
        cells = names[dataset]["fit_rows"] * (names[dataset]["features"] + 1)
        charged = statistics.median(v[0] for v in values)
        bits.append(8.0 * charged / cells)
        ratios.append(4.0 * cells / charged)
        sidecar.append(statistics.median(v[2] / v[0] for v in values))
    return {"bits_per_cell_median": statistics.median(bits),
            "float32_ratio_median": statistics.median(ratios),
            "sidecar_share_median": statistics.median(sidecar), "n": len(bits)}


def _quantiles(values):
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], statistics.median(values), q[2]


def foundation_summary() -> list[dict]:
    """The pinned foundation-model hero rows (historical; auditors unqualified)."""
    import csv
    with authenticated("foundation").open() as stream:
        rows = list(csv.DictReader(stream))
    return [{"auditor": row["auditor_display"], "method": row["method_display"],
             "retention": float(row["retention"]), "lo": float(row["interval_lo"]),
             "hi": float(row["interval_hi"]), "bytes": int(row["artifact_bytes"]),
             "lineages": int(row["observed_lineages"]), "evidence_status": row["evidence_status"]}
            for row in rows]


def compute() -> dict:
    names = lineage_names()
    worst, each, complete = worst_auditor_items(names)
    bytes_by_arm = headline_bytes()
    rows = []
    for method, config, display, role in ARMS:
        items = worst.get((method, config), [])
        label = f"compression-frontier|{method}|{config}|worst|{SIZE}"
        level = nested_level(items, label)
        auditors = {}
        for auditor in AUDITORS:
            sub = nested_level(each[(method, config)][auditor], f"{label}|{auditor}", draws=2000)
            auditors[auditor] = {k: sub[k] for k in ("n", "median", "lo", "hi")}
        size = bytes_by_arm.get((method, config))
        lift = None
        if size and level["median"] is not None and level["median"] > 0:
            lift = level["median"] / (size["median"] / 1024.0)
        rows.append({
            "method": method, "configuration": config, "display": display, "role": role,
            "lineages": level["n"], "worst_median": level["median"], "worst_lo": level["lo"],
            "worst_hi": level["hi"], "auditors": auditors, "bytes": size,
            "lift_per_kib": lift, "evidence_status": EVIDENCE,
        })
    generators = [r for r in rows if r["role"] == "generator" and r["bytes"] and r["lineages"] >= FULL_PANEL_MIN]
    frontier = []
    for row in sorted(generators, key=lambda r: r["bytes"]["median"]):
        if not frontier or row["worst_median"] > frontier[-1]["worst_median"]:
            frontier.append(row)
    ranked = sorted((r for r in rows if r["role"] == "generator" and r["lift_per_kib"]),
                    key=lambda r: -r["lift_per_kib"])
    for position, row in enumerate(ranked, start=1):
        row["lift_rank"] = position
    dope = next(r for r in rows if r["method"] == "DOPE" and r["configuration"] == "headline")
    tabsyn = next(r for r in rows if r["method"] == "TabSyn")
    paired = nested_paired(worst[("DOPE", "headline")], worst[("TabSyn", tabsyn["configuration"])],
                           "compression-frontier|DOPE-TabSyn|worst|4", min_fits_left=3, min_fits_right=3)
    paired = {k: paired[k] for k in ("n", "hl", "hl_lo", "hl_hi", "wins", "ties", "losses") if k in paired}
    runner_up = ranked[1]
    return {
        "format": "dope-compression-frontier-v1",
        "evidence_status": EVIDENCE,
        "official_tests_opened": False,
        "size": f"{SIZE}n",
        "auditors": list(AUDITORS),
        "aggregation": "fit = min over auditors of the sample-seed median; lineage = median of fits; "
                       "level = median of lineages; 95% nested cluster->lineage->fit bootstrap, 10,000 draws",
        "lineages_all_auditors_informative": len(complete),
        "inputs": {key: {"path": str(path.relative_to(REPO)), "sha256": digest}
                   for key, (path, digest) in INPUTS.items()},
        "rows": rows,
        "frontier": [r["display"] for r in frontier],
        "lift_ranking": [r["display"] for r in ranked],
        "lift_ranking_scope": "Descriptive across method-specific available validation cohorts; not matched-cohort superiority or TDL qualification",
        "dope_vs_tabsyn_worst": paired,
        "byte_ratio_tabsyn_over_dope": tabsyn["bytes"]["median"] / dope["bytes"]["median"],
        "lift_ratio_dope_over_runner_up": dope["lift_per_kib"] / runner_up["lift_per_kib"],
        "runner_up": runner_up["display"],
        "foundation": foundation_summary(),
        "raw_table_float32_bytes": raw_table_bytes(names, complete),
        "dope_density": dope_bits_per_cell(names, complete),
    }


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail when a text output would change")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    data = compute()
    outputs = text_outputs(data)
    if args.check:
        stale = [name for name, text in outputs.items()
                 if not (REPO / name).exists() or (REPO / name).read_text() != text]
        stale += [name for name in (FIGURE_PDF, FIGURE_PNG) if not (REPO / name).exists()]
        for name in stale:
            print(f"stale or missing: {name}", file=sys.stderr)
        if not stale:
            print("compression frontier outputs are current")
        return 1 if stale else 0
    for name, text in outputs.items():
        (REPO / name).write_text(text)
    if not args.no_figure:
        render(data, {"pdf": REPO / FIGURE_PDF, "png": REPO / FIGURE_PNG})
    print(json.dumps({"frontier": data["frontier"], "lift_ranking": data["lift_ranking"],
                      "byte_ratio": round(data["byte_ratio_tabsyn_over_dope"]),
                      "lift_ratio": round(data["lift_ratio_dope_over_runner_up"], 2),
                      "pair": data["dope_vs_tabsyn_worst"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
