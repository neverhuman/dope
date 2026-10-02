"""Recreate the bounded GPU research figure from committed validation results."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from statistics import median

from research.benchmark.manifest import digest
from research.benchmark.score import sha256

from research.benchmark.publish_target_research import (
    AUDITORS,
    DATASETS,
    FIT_SEEDS,
    PROFILES,
    SAMPLE_SEEDS,
    render as check_tables,
)


def render(report, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    check_tables(report)
    matplotlib.rcParams["svg.hashsalt"] = "dope-bounded-target-research-v1"
    figure, axes = plt.subplots(3, 3, figsize=(18, 11.5), sharex=True)
    colors = ("#32688e", "#1792a4", "#7253a3", "#c64e6c")
    for i, dataset in enumerate(DATASETS):
        for j, auditor in enumerate(AUDITORS):
            axis = axes[i, j]
            for k, profile in enumerate(PROFILES):
                rows = [
                    next(
                        r
                        for r in report["summaries"]
                        if (
                            r["dataset"],
                            r["auditor"],
                            r["profile"],
                            r["size_multiplier"],
                        )
                        == (dataset, auditor, profile, size)
                    )
                    for size in (1, 4)
                ]
                x = [k * 0.04 - 0.06, 1 + k * 0.04 - 0.06]
                centers = [r["median_of_fit_medians"] for r in rows]
                label = (
                    profile.replace("features", "").replace("_steps", " features / ")
                    + " steps"
                )
                for location, row in zip(x, rows):
                    values = [
                        r["median_retention"]
                        for r in row["fit_medians"]
                        if r["median_retention"] is not None
                    ]
                    axis.scatter(
                        [location] * len(values),
                        values,
                        s=14,
                        color=colors[k],
                        alpha=0.35,
                    )
                    if values:
                        axis.vlines(
                            location,
                            min(values),
                            max(values),
                            color=colors[k],
                            linewidth=1,
                        )
                if all(v is not None for v in centers):
                    axis.plot(
                        x,
                        centers,
                        color=colors[k],
                        marker="D",
                        markersize=4,
                        linewidth=1.7,
                        label=f"DOPE {label}",
                    )
                else:
                    low_signal = all(
                        r["status_counts"] == {"ok": 6} and r["complete_fit_seeds"] == 0
                        for r in rows
                    )
                    axis.plot(
                        [],
                        [],
                        color=colors[k],
                        label=f"DOPE {label}: {'noninformative' if low_signal else 'incomplete'}",
                    )
            reference = [
                r
                for r in report["matched_baseline_summaries"]
                if (r["dataset"], r["auditor"]) == (dataset, auditor)
            ]
            groups = sorted({(r["method"], r["configuration"]) for r in reference})
            for k, (method, config) in enumerate(groups):
                rows = [
                    next(
                        r
                        for r in reference
                        if (r["method"], r["configuration"], r["size_multiplier"])
                        == (method, config, size)
                    )
                    for size in (1, 4)
                ]
                centers = [r["median_of_fit_medians"] for r in rows]
                label = f'{method} {config.replace("_and_", " + ").replace("_", " ")}'
                color = (
                    "#777777"
                    if method == "DOPE"
                    else ("#c08722" if method == "CTGAN" else "#248355")
                )
                if all(v is None for v in centers):
                    axis.plot(
                        [],
                        [],
                        color=color,
                        marker="x",
                        linestyle="none",
                        label=f"{label}: {rows[0]['complete_fit_seeds']}/2 fits",
                    )
                    continue
                for size, location in zip((1, 4), (0, 1)):
                    for seed in FIT_SEEDS:
                        values = [
                            r["utility"][auditor]["retention"]
                            for r in report["matched_baseline_cells"]
                            if (
                                r["dataset"],
                                r["method"],
                                r["configuration"],
                                r["size_multiplier"],
                                r["fit_seed"],
                            )
                            == (dataset, method, config, size, seed)
                            and r["status"] == "ok"
                            and r["utility"][auditor]["informative"] is True
                            and r["utility"][auditor]["retention"] is not None
                        ]
                        if len(values) == len(SAMPLE_SEEDS):
                            axis.scatter(
                                location, median(values), color=color, s=12, alpha=0.3
                            )
                if all(v is not None for v in centers):
                    axis.plot(
                        [0, 1],
                        centers,
                        color=color,
                        linewidth=1.4,
                        linestyle="--" if config == "default" else ":",
                        marker="o",
                        markersize=3,
                        label=label,
                    )
                else:
                    axis.plot([], [], color=color, label=f"{label}: incomplete")
            axis.set_title(
                f'{dataset} · {auditor.title() if auditor != "linear" else "Linear/logistic"}'
            )
            axis.set_xticks([0, 1], ["n", "4n"])
            axis.set_xlim(-0.17, 1.17)
            axis.grid(axis="y", color="#d7dce1", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(fontsize=6.8, loc="best")
            if j == 0:
                axis.set_ylabel("Null-normalized TSTR/TRTR retention")
            if not axis.has_data():
                axis.set_yticks([])
                axis.text(
                    0.5,
                    0.45,
                    "No informative retention estimates\n"
                    "Low-signal flags and losses are in the JSON",
                    ha="center",
                    transform=axis.transAxes,
                    fontsize=9,
                    color="#4a5661",
                )
    figure.suptitle(
        "DOPE GPU target research: four frozen profiles and matched native/default references",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.011,
        "Training-derived validation only; two fit seeds, three sample seeds; points = fit medians; lines = median of both fits.\n"
        "News uses lossless packed artifacts and has no matched neural reference here. Official tests sealed; gated scores and release safety null.",
        ha="center",
        fontsize=8,
        color="#4a5661",
    )
    figure.tight_layout(rect=[0, 0.055, 1, 0.965])
    svg, pdf = BytesIO(), BytesIO()
    figure.savefig(
        svg, format="svg", metadata={"Date": None, "Creator": "DOPE benchmark"}
    )
    figure.savefig(
        pdf,
        format="pdf",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "DOPE benchmark"},
    )
    plt.close(figure)
    destination.with_suffix(".svg").write_text(
        "".join(line.rstrip() + "\n" for line in svg.getvalue().decode().splitlines())
    )
    destination.with_suffix(".pdf").write_bytes(pdf.getvalue())


def main():
    path = Path(__file__).with_name("results") / "pilot24-target-gpu-research.json"
    report = json.loads(path.read_text())
    render(report, path)
    references = [r["receipt"] for r in report["raw_cells"] + report["packed_cells"]]
    manifest = {
        "format": "dope-target-gpu-research-publication-manifest",
        "version": 1,
        "scope": "training_derived_validation_only",
        "artifacts": {
            path.with_suffix(suffix).name: {
                "sha256": sha256(path.with_suffix(suffix)),
                "bytes": path.with_suffix(suffix).stat().st_size,
            }
            for suffix in (".json", ".schema.json", ".csv", ".md", ".svg", ".pdf")
        },
        "sources": {
            name: sha256(path.parent.parent / name)
            for name in (
                "publish_target_research.py",
                "target_research_receipts.py",
                "publish_target_research_figure.py",
            )
        },
        "input_locks": report["locks"],
        "new_gpu_fits": len(report["fit_attempts"]),
        "new_validation_cells": len(references),
        "validation_receipt_index_sha256": digest(references),
        "official_tests_opened": False,
        "production_certified": False,
        "mfs_v2": None,
        "ptf_v1": None,
    }
    from jsonschema import Draft202012Validator

    schema_path = path.with_name("pilot24-target-gpu-research.manifest.schema.json")
    Draft202012Validator(json.loads(schema_path.read_text())).validate(manifest)
    path.with_name("pilot24-target-gpu-research.manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    print("bounded GPU validation figures regenerated")


if __name__ == "__main__":
    main()
