"""Text outputs of the equal-byte study: paper macros and tables, the README block.

Inputs are the committed study summary (research/benchmark/equal_bytes/results)
and, for the generator markers in Figure 1, the frozen v2 ledger
(research/benchmark/results/review-fixes-v2/scalar-cells.jsonl plus
docs/whitepaper/generated/v2-headline.json for median charged bytes).

Figures and the command line are in equal_bytes_render.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from equal_bytes_common import (ADULT, AUDITORS, BLUE, GENERATED, GRAY, GRID, HEADLINE, INK, LEDGER, MUTED,  # noqa: F401
                                ORANGE, README_BEGIN, README_END, REPO, RESULTS, TAU, coverage, fmt, human_bytes,
                                latex_table)
from equal_bytes_sections import (adult_macros, adult_readme_rows, adult_table, copula_macros,  # noqa: F401
                                  gap_cells, panel_macros, sensitivity_macros)

GENERATOR_ARMS = [
    ("DOPE", "headline", "DOPE, published GPU run"),
    ("TabSyn", "scaled_200_vae_1000_diffusion", "TabSyn"),
    ("ARF", "author_default", "ARF"),
    ("TVAE", "native_selected", "TVAE"),
    ("CTGAN", "native_selected", "CTGAN"),
    ("ForestDiffusion/Forest-Flow", "native_selected", "Forest-Flow"),
    ("GaussianCopula", "native_selected", "Gaussian copula"),
    ("Chow-Liu", "native_selected", "Chow-Liu tree"),
    ("independent_marginals", "native_selected", "Independent marginals"),
]


def per_lineage_bytes(method: str, configuration: str) -> dict[str, float]:
    """Charged bytes per lineage, read as docs/whitepaper/scripts/v2_bytes.py reads them."""
    results = REPO / "research/benchmark/results"
    grouped: dict[str, list[float]] = {}
    if method == "DOPE":
        for line in (results / "review-fixes-v2/fits.jsonl").read_text().splitlines():
            fit = json.loads(line)
            if fit["configuration"] == configuration and fit["status"] == "ok" and isinstance(fit["charged_bytes"], int):
                grouped.setdefault(fit["dataset"], []).append(fit["charged_bytes"])
    elif method == "TabSyn":
        for cell in json.loads((results / "tabsyn-default100-scaled-cuda-fits.json").read_text())["cells"]:
            files = cell.get("artifact_files") or {}
            if files:
                grouped.setdefault(cell["dataset_id"], []).append(
                    sum(m["bytes"] for n, m in files.items() if not n.startswith("diffusion-epoch-")))
        for line in (results / "review-fixes-tabsyn-seeds-v2/fits.jsonl").read_text().splitlines():
            fit = json.loads(line)
            if fit["status"] == "ok" and isinstance(fit["charged_bytes"], int):
                grouped.setdefault(fit["dataset"], []).append(fit["charged_bytes"])
    elif method == "ARF":
        field = {"author_default": "default_charged_bytes", "native_selected": "selected_charged_bytes"}[configuration]
        for cell in json.loads((results / "arf-s3-population-native.json").read_text())["cells"]:
            if isinstance(cell.get(field), int):
                grouped.setdefault(cell["dataset"], []).append(cell[field])
    else:
        name = {"CTGAN": "sdv-matched-population-validation.json", "TVAE": "sdv-matched-population-validation.json",
                "ForestDiffusion/Forest-Flow": "s3-matched-forest-confirmation-validation.json"}.get(
                    method, "density-matched-population-validation.json")
        for cell in json.loads((results / name).read_text())["cells"]:
            if cell["method"] == method and cell["configuration"] == configuration \
                    and isinstance(cell.get("charged_artifact_bytes"), int):
                grouped.setdefault(cell["dataset"], []).append(cell["charged_artifact_bytes"])
    return {key: float(np.median(values)) for key, values in grouped.items()}


def published_dope_per_auditor() -> dict[str, dict[str, float]]:
    """Published GPU headline run: clipped retention per auditor at 4n, by PMLB name."""
    names = {e["dataset_id"]: e["pmlb_id"] for e in json.loads(
        (REPO / "ops/ci/paper-inputs/pmlb-sources.json").read_text())["entries"]}
    fits: dict = {}
    with LEDGER.open() as handle:
        for line in handle:
            if '"method": "DOPE"' not in line and '"method":"DOPE"' not in line:
                continue
            cell = json.loads(line)
            if cell["configuration"] != "headline" or cell["size"] != 4:
                continue
            fit = fits.setdefault(names[cell["dataset"]], {}).setdefault(cell["fit_seed"], {})
            for auditor in AUDITORS:
                utility = cell["utility"][auditor]
                fit.setdefault(auditor, []).append(utility["retention"] if utility["informative"] else None)
    out = {}
    for name, by_fit in fits.items():
        out[name] = {}
        for auditor in AUDITORS:
            values = [float(np.median(v[auditor])) for v in by_fit.values()
                      if all(x is not None for x in v[auditor])]
            if values:
                out[name][auditor] = min(max(float(np.median(values)), 0.0), 1.0)
    return out


def parity() -> dict:
    """Native CPU training (this release) against the published GPU run, lineage by lineage."""
    published = published_dope_per_auditor()
    differences: dict[str, list[float]] = {a: [] for a in AUDITORS}
    for path in sorted((RESULTS / "lineages").glob("*.json")) + sorted((RESULTS / "panel").glob("*.json")):
        record = json.loads(path.read_text())
        gain = {a: record["null"] - record["real"][a] for a in AUDITORS}
        fits = [row for row in record.get("dope", []) if row.get("rung") == "L4" and "loss" in row]
        for auditor in AUDITORS:
            if record["dataset"] not in published or auditor not in published[record["dataset"]] \
                    or gain[auditor] < 0.01 * abs(record["null"]) or not fits:
                continue
            mine = float(np.median([min(max((record["null"] - f["loss"][auditor]) / gain[auditor], 0.0), 1.0)
                                    for f in fits]))
            differences[auditor].append(mine - published[record["dataset"]][auditor])
    return {a: {"median": float(np.median(v)), "n": len(v), "p10": float(np.quantile(v, 0.1)),
                "p90": float(np.quantile(v, 0.9)), "mean_abs": float(np.mean(np.abs(v)))}
            for a, v in differences.items() if v}


def generator_points() -> list[dict]:
    """Worst-learner retention at 4n per lineage and generator on the historical panel."""
    wanted = {(m, c) for m, c, _ in GENERATOR_ARMS}
    cells: dict = {}
    with LEDGER.open() as handle:
        for line in handle:
            cell = json.loads(line)
            key = (cell["method"], cell["configuration"])
            if key not in wanted or cell["size"] != 4:
                continue
            fit = cells.setdefault(key, {}).setdefault(cell["dataset"], {}).setdefault(cell.get("fit_seed"), {})
            for auditor in AUDITORS:
                utility = cell["utility"][auditor]
                fit.setdefault(auditor, []).append(utility["retention"] if utility["informative"] else None)
    arms = {(a["method"], a["configuration"]): a for a in json.loads(HEADLINE.read_text())["arms"]}
    points = []
    for method, configuration, label in GENERATOR_ARMS:
        robust = {}
        for dataset, fits in cells.get((method, configuration), {}).items():
            per_auditor = {}
            for auditor in AUDITORS:
                values = [float(np.median(v[auditor])) for v in fits.values()
                          if auditor in v and all(x is not None for x in v[auditor])]
                if values:
                    per_auditor[auditor] = min(max(float(np.median(values)), 0.0), 1.0)
            if per_auditor:
                robust[dataset] = min(per_auditor.values())
        sizes = per_lineage_bytes(method, configuration)
        points.append({"method": method, "configuration": configuration, "label": label,
                       "bytes": arms[(method, configuration)]["bytes"]["median"],
                       "robust_median": float(np.median(list(robust.values()))), "lineages": len(robust),
                       "per_lineage": {k: {"robust": v, "bytes": sizes.get(k)} for k, v in sorted(robust.items())}})
    return points


def scaling_rows() -> list[dict]:
    """Bytes of the headline-shape program and of the whole table, by fit rows."""
    rows = []
    for path in sorted((RESULTS / "scaling").glob("*.json")):
        record = json.loads(path.read_text())
        for entry in record["sizes"]:
            rows.append({"dataset": record["dataset"], "n": entry["n"], "dope_bytes": entry["dope"]["bytes"],
                         "full_table_bytes": entry["full_table_bytes"]})
    return rows


def codec_factor(summary: dict) -> float:
    """Median over tables of historical over packed program bytes, scaling charge excluded."""
    return float(np.median([(x["dope"]["L4"]["historical_codec_bytes"] - x["projection_bytes"])
                            / (x["dope"]["L4"]["bytes"] - x["projection_bytes"])
                            for x in summary["lineages"] if "headline" in x]))


def macros(summary: dict, generators: list[dict], scaling: list[dict]) -> str:
    a = summary["aggregate"]
    largest = [r for r in scaling if r["n"] == max(x["n"] for x in scaling if x["dataset"] == r["dataset"])]
    tabsyn = next(g for g in generators if g["method"] == "TabSyn")
    published = next(g for g in generators if g["method"] == "DOPE")
    values = {
        "EbLineages": str(a["lineages"]),
        "EbLargeLineages": str(a["large_lineages"]),
        "EbBytes": f"{a['headline_bytes_median']:,.0f}",
        "EbHistoricalBytes": f"{a['historical_style_bytes_median']:,.0f}",
        "EbByteFactor": fmt(a["byte_reduction_factor_median"], 1),
        "EbCodecFactor": fmt(codec_factor(summary), 2),
        "EbProjectionJson": f"{np.median([x['projection_json_bytes'] for x in summary['lineages']]):,.0f}",
        "EbProjection": f"{np.median([x['projection_bytes'] for x in summary['lineages']]):,.0f}",
        "EbPublishedTables": str(published["lineages"]),
        "EbRobust": fmt(a["headline_robust_median"][0]),
        "EbRobustLo": fmt(a["headline_robust_median"][1]),
        "EbRobustHi": fmt(a["headline_robust_median"][2]),
        **gap_cells("Eb", a),
        "EbRhoSize": fmt(a["gap_vs_table_size_spearman"][0]),
        "EbRhoSizeN": str(a["gap_vs_table_size_spearman"][2]),
        "EbRhoSizeOutgrown": fmt(a["gap_vs_table_size_spearman_outgrown"][0]),
        "EbRhoSizeOutgrownN": str(a["gap_vs_table_size_spearman_outgrown"][2]),
        **{f"EbLarge{name.capitalize()}{field}": value
           for name, cells in a["large_per_learner_gap"].items()
           for field, value in zip(("Gap", "Wins", "Losses"), (fmt(cells[0]), str(cells[1]), str(cells[2])))},
        "EbRatioLarge": fmt(a["data_equivalence_ratio_large_median"], 1),
        "EbBeyondFull": str(a["beyond_full_table"]),
        "EbDefaultRobust": fmt(a["cpu_default_robust_median"]),
        "EbTrainedGain": fmt(a["trained_minus_default_median"][0], 3),
        "EbTrainedGainLo": fmt(a["trained_minus_default_median"][1], 3),
        "EbTrainedGainHi": fmt(a["trained_minus_default_median"][2], 3),
        "EbTdlEightyDope": str(a["tdl_wins"]["T80"]["dope_smaller"]),
        "EbTdlEightyData": str(a["tdl_wins"]["T80"]["data_smaller"]),
        "EbTdlNinetyDope": str(a["tdl_wins"]["T90"]["dope_smaller"]),
        "EbTdlNinetyData": str(a["tdl_wins"]["T90"]["data_smaller"]),
        "EbTabSynFactor": f"{tabsyn['bytes'] / published['bytes']:,.0f}",
        "EbTabSynRobust": fmt(tabsyn["robust_median"]),
        "EbPublishedRobust": fmt(published["robust_median"]),
        "EbPublishedBytes": f"{published['bytes']:,.0f}",
        "EbRhoRows": fmt(a["ratio_vs_rows_spearman"][0]),
        "EbRhoRowsN": str(a["ratio_vs_rows_spearman"][2]),
        "EbRhoProxy": fmt(a["ratio_vs_proxy_large_spearman"][0]),
        "EbRhoProxyN": str(a["ratio_vs_proxy_large_spearman"][2]),
        "EbScaleDatasets": str(len({r["dataset"] for r in scaling})),
        "EbScaleMaxN": f"{max(r['n'] for r in scaling):,}",
        "EbScaleDopeMin": f"{min(r['dope_bytes'] for r in scaling):,}",
        "EbScaleDopeMax": f"{max(r['dope_bytes'] for r in scaling):,}",
        "EbScaleFullMax": f"{max(r['full_table_bytes'] for r in largest):,}",
        "EbScaleCells": str(len(scaling)),
    }
    values.update(adult_macros())
    values.update(sensitivity_macros(summary))
    values.update(copula_macros())
    values.update(CONSTANTS)
    for auditor, result in parity().items():
        name = auditor.capitalize()
        values[f"EbParity{name}"] = fmt(result["median"], 3)
        values[f"EbParity{name}N"] = str(result["n"])
        values[f"EbParity{name}Lo"], values[f"EbParity{name}Hi"] = fmt(result["p10"]), fmt(result["p90"])
        values[f"EbParity{name}Abs"] = fmt(result["mean_abs"])
    values.update(panel_macros())
    lines = ["% Generated by docs/whitepaper/scripts/equal_bytes_render.py; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(values.items())]
    return "\n".join(lines) + "\n"


LEARNER_LABELS = (("linear", "least squares"), ("mlp", "MLP"), ("catboost", "CatBoost"))


def readme_block(summary: dict, generators: list[dict], scaling: list[dict]) -> str:
    a, sensitivity = summary["aggregate"], summary["sensitivity"]
    no_mlp = sensitivity["without_mlp"]
    tabsyn = next(g for g in generators if g["method"] == "TabSyn")
    published = next(g for g in generators if g["method"] == "DOPE")
    dope_bytes = [r["dope_bytes"] for r in scaling]
    match = parity()
    panel = panel_macros()
    largest = max(r["full_table_bytes"] for r in scaling)
    interval = lambda block, key="equal_byte": (f"{fmt(block[key + '_gap_median'][0])} "
                                                f"[{fmt(block[key + '_gap_median'][1])}, {fmt(block[key + '_gap_median'][2])}]")
    t80, t90 = a["tdl_wins"]["T80"], a["tdl_wins"]["T90"]
    per_learner = ", ".join(f"{fmt(a['large_per_learner_gap'][key][0])} ({label})" for key, label in LEARNER_LABELS)
    lines = [
        README_BEGIN,
        "<!-- Generated by docs/whitepaper/scripts/equal_bytes_render.py; edit the generator, not this block. -->",
        "Evidence status: development validation on public tables (selection-exposed; not a sealed TDL result).",
        "The equal-byte cohort preserves the historical name-based mask. It includes simulated pollen and is not a measured-only real-world cohort; source origin remains mixed or unverified.",
        "",
        f"- **The smallest generator at its transfer level.** On the historical {published['lineages']}-table panel "
        f"the published DOPE run keeps {fmt(published['robust_median'])} of the real gain for the worst of three "
        f"learners at {published['bytes']:,.0f} B. TabSyn keeps {fmt(tabsyn['robust_median'])} with "
        f"{tabsyn['bytes'] / published['bytes']:,.0f}x the bytes, and no generator's frontier is above DOPE's at any budget.",
        f"- **More than the real rows that fit in its bytes, once a table outgrows the program.** On the "
        f"{a['large_lineages']} tables in the historical mixed/unknown-origin cohort that reach the study's cap of 2,000 fit rows, a program keeps "
        f"{interval(a, 'large_equal_byte')} more of the real gain than the most real rows that fit in exactly its bytes "
        f"({a['large_equal_byte_wins']} wins, {a['large_equal_byte_losses']} losses, "
        f"{a['large_lineages'] - a['large_equal_byte_wins'] - a['large_equal_byte_losses']} within 0.01), "
        f"and real rows need a median "
        f"{fmt(a['data_equivalence_ratio_large_median'], 1)}x its bytes to match it. The advantage grows with how much "
        f"larger the whole table is than the program (Spearman {fmt(a['gap_vs_table_size_spearman'][0])} over "
        f"{a['lineages']} tables); tables of a few hundred rows compress to about a program's size and are cheapest "
        f"shipped whole, so over all {a['lineages']} tables the gap is {interval(a)} ({a['equal_byte_wins']} wins, "
        f"{a['equal_byte_losses']} losses).",
        f"- **Checked against the obvious objections.** The real-row arm gets the best of three lossless codecs, an "
        f"exact byte match, and an MLP also trained on the rows resampled to the synthetic size (early-stopped on "
        f"held-out rows). The {fmt(a['large_equal_byte_gap_median'][0])} is for the worst of three learners; per learner "
        f"on the large tables it is {per_learner}. "
        f"Leaving the MLP out, the large-table gap is "
        f"{interval(no_mlp, 'large_equal_byte')} ({no_mlp['large_equal_byte_wins']} wins, "
        f"{no_mlp['large_equal_byte_losses']} {'loss' if no_mlp['large_equal_byte_losses'] == 1 else 'losses'})."
        + (f" On all {panel['EbPanelLineages']} panel lineages, simulated families included, it is "
           f"{panel['EbPanelLargeGap']} on the {panel['EbPanelLarge']} large tables ({panel['EbPanelLargeWins']} wins, "
           f"{panel['EbPanelLargeLosses']} losses)." if panel else ""),
        f"- **Where it stops.** A program this small is a lossy channel: its retention plateaus, so to clear the 0.8 "
        f"bar real rows need fewer bytes on {t80['data_smaller']} of {a['lineages']} tables (DOPE on {t80['dope_smaller']}), "
        f"and at 0.9 on {t90['data_smaller']} (DOPE on {t90['dope_smaller']}). Raising that ceiling at fixed bytes is the "
        f"open problem.",
        f"- **Honest bytes, no GPU.** Packed values make the same programs {fmt(codec_factor(summary), 2)}x smaller with "
        f"bit-identical samples for the headline shape; with the column scaling charged as two 4-byte floats per column the "
        f"median charge is {a['headline_bytes_median']:,.0f} B ({a['historical_style_bytes_median']:,.0f} B before). "
        f"From 250 to {max(x['n'] for x in scaling):,} rows a program stays between {min(dope_bytes):,} and "
        f"{max(dope_bytes):,} B while the whole table grows to {largest:,} B. The network trains in Rust; its CatBoost "
        f"retention shows no systematic shift from the published GPU run (median difference "
        f"{fmt(match['catboost']['median'], 3)} over {match['catboost']['n']} tables, tenth to ninetieth percentile "
        f"{fmt(match['catboost']['p10'])} to {fmt(match['catboost']['p90'])}; different splits, not a paired replay).",
        "",
        "| UCI Adult, scored on the official test file | Bytes | Logistic | MLP | CatBoost |",
        "|---|---:|---:|---:|---:|",
        *adult_readme_rows(),
        "",
        "Cells are retention: the share of the real rows' log-loss gain over the class prior (below zero is worse "
        "than the prior).",
        README_END,
    ]
    return "\n".join(lines)


CONSTANTS = {
    # Program shape of the headline recipe (rust/compiler/target_fitting/native_training.rs).
    "cScreen": "12", "cHidden": "16", "cParams": "237", "cKnots": "17", "cBits": "8", "cSteps": "2{,}048",
    "cParents": "16", "cNoiseInputs": "4", "cRate": "2\\times 10^{-3}", "cDecay": "0.01", "cClip": "5",
    "cSkew": "2", "cSmoothing": "10", "cSynthetic": "4n", "cFitCap": "2{,}000", "cTestShare": "30\\%",
    "cTau": "0.8", "cTauStrict": "0.9", "cRowCap": "4{,}096",
    # Small integers for mathematics (the public lint refuses bare digits in prose).
    "One": "1", "Two": "2", "Four": "4", "Six": "6", "Ten": "10", "Eight": "8",
}


def generators_table(generators: list[dict]) -> str:
    rows = []
    for g in sorted(generators, key=lambda g: g["bytes"]):
        qualified = [v for v in g["per_lineage"].values() if v["robust"] >= TAU]
        name = "\\textbf{DOPE} (published run)" if g["method"] == "DOPE" else g["label"]
        rows.append([name, human_bytes(g["bytes"]), fmt(g["robust_median"]),
                     f"{len(qualified)}/{g['lineages']}", f"{g['bytes'] / generators[0]['bytes']:,.0f}"
                     if g["method"] != "DOPE" else "1"])
    return latex_table(["Generator", "Median bytes", "Worst-learner retention", "Tables at $T\\geq\\cTau$",
                        "Bytes / DOPE"], rows, "lrrrr")


def ladder_table(summary: dict) -> str:
    names = {"L1": "6 inputs, width 8, 9 knots, $q6$", "L2": "12 inputs, 9 knots, $q6$",
             "L3": "12 inputs, 17 knots, $q6$", "L4": "12 inputs, 17 knots, $q8$ (headline shape)",
             "L5": "24 inputs, 33 knots, $q8$"}
    rows = []
    lineages = [x for x in summary["lineages"] if x.get("dope")]
    for rung, label in names.items():
        points = [x["dope"][rung] for x in lineages if rung in x["dope"]]
        rows.append([rung, label, f"{np.median([p['bytes'] for p in points]):,.0f}",
                     fmt(float(np.median([p["robust"] for p in points if p["robust"] is not None]))),
                     str(sum(1 for p in points if p["robust"] is not None and p["robust"] >= TAU))])
    defaults = [x["cpu_default"] for x in lineages if x.get("cpu_default")]
    rows.append(["--", "CPU default: untrained head, historical codec",
                 f"{np.median([p['bytes'] for p in defaults]):,.0f}",
                 fmt(float(np.median([p["robust"] for p in defaults if p["robust"] is not None]))),
                 str(sum(1 for p in defaults if p["robust"] is not None and p["robust"] >= TAU))])
    historical = [x["dope"]["L4"]["historical_codec_bytes"] - x["projection_bytes"] + x["projection_json_bytes"]
                  for x in lineages]
    head = [x["dope"]["L4"] for x in lineages]
    rows.append(["L4", "same programs, historical codec + JSON scaling", f"{np.median(historical):,.0f}",
                 fmt(float(np.median([p["robust"] for p in head]))),
                 str(sum(1 for p in head if p["robust"] >= TAU))])
    return latex_table(["Rung", "Program", "Bytes", "Worst-learner $T$", "Tables at $T\\geq\\cTau$"],
                       rows, "llrrr")


def lineage_table(summary: dict) -> str:
    rows = []
    for x in sorted(summary["lineages"], key=lambda x: x["dataset"]):
        head = x.get("headline") or {}
        cells = [x["dataset"].replace("_deprecated_", "").replace("_", "\\_"), f"{x['n_fit']:,}", str(x["d"])]
        cells.append(f"{head['bytes']:,.0f}" if head else "--")
        cells.append(fmt(head["robust"]) if head else "--")
        cells.append(fmt(head["data_at_equal_bytes"]) if head and head.get("data_at_equal_bytes") is not None else "--")
        ratio = head.get("data_equivalence_ratio") if head else None
        cells.append(fmt(ratio, 1) if ratio else ("$>$ table" if head else "--"))
        for key in ("T80", "T90"):
            tdl = x["tdl"][key]
            cells.append(" / ".join(f"{v:,}" if v else "--" for v in (tdl["dope"], tdl["data"])))
        rows.append(cells)
    return latex_table(["Table", "$n$", "$p$", "Bytes", "DOPE $T$", "Rows $T$", "Ratio",
                        "TDL80 DOPE / rows", "TDL90 DOPE / rows"], rows, "lrrrrrrrr")


def text_outputs() -> dict[str, str]:
    """Every text file this renderer owns, keyed by repository path."""
    summary = json.loads((RESULTS / "summary.json").read_text())
    generators = generator_points()
    scaling = scaling_rows()
    return {
        "research/benchmark/equal_bytes/results/generators.json": json.dumps(generators, indent=1) + "\n",
        "paper/generated/equal-bytes-macros.tex": macros(summary, generators, scaling),
        "paper/generated/readme-equal-bytes.md": readme_block(summary, generators, scaling) + "\n",
        "paper/generated/equal-bytes-generators.tex": generators_table(generators),
        "paper/generated/equal-bytes-ladder.tex": ladder_table(summary),
        "paper/generated/equal-bytes-adult.tex": adult_table(),
        "paper/generated/equal-bytes-lineages.tex": lineage_table(summary),
    }


def write_text(path: Path, text: str, check: bool, failures: list[str]) -> None:
    if check:
        if not path.is_file() or path.read_text() != text:
            failures.append(f"{path.relative_to(REPO)} differs from its generator")
    else:
        path.write_text(text)
