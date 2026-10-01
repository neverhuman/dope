"""Publish verified conditional GPU research, including the failed original fits."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .score import sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
RESULTS = Path(__file__).with_name("results")


def build() -> dict:
    rounds, cells = [], []
    for revision, directory, stages in (
        ("original", "conditional-gpu-v1", ("discovery",)),
        ("readout_refit", "conditional-readout-v1", ("discovery", "confirmation")),
    ):
        root = SCRATCH / directory
        source = root / "source/files.json"
        for name, expected in json.loads(source.read_text()).items():
            if sha256(source.parent / name) != expected:
                raise ValueError("archived conditional source changed")
        for stage in stages:
            lock_path = root / f"{stage}.lock.json"
            lock = json.loads(lock_path.read_text())
            if lock.get("source_manifest_sha256", sha256(source)) != sha256(source):
                raise ValueError("conditional source manifest changed")
            rounds.append({"revision": revision, "stage": stage, "path": str(lock_path),
                           "sha256": sha256(lock_path), "binary_sha256": lock["gpu_binary_sha256"],
                           "source_manifest_path": str(source), "source_manifest_sha256": sha256(source),
                           "fit_compute_ceiling_seconds": lock["gpu_fit_compute_ceiling_seconds"]})
            for index, job in enumerate(lock["gpu_jobs"]):
                path = root / stage / f"cell-{index:02d}.json"
                cell = json.loads(path.read_text())
                fit_path = Path(cell["fit"]["receipt"])
                fit = json.loads(fit_path.read_text())
                if (cell["job"] != job or cell["round_sha256"] != sha256(lock_path)
                        or sha256(fit_path) != cell["fit"]["receipt_sha256"]
                        or fit["identity"]["binary_sha256"] != lock["gpu_binary_sha256"]
                        or sha256(fit_path.with_suffix(".log")) != fit["log_sha256"]):
                    raise ValueError("conditional identity or fit evidence changed")
                metric = cell["common_validation"]
                if metric is not None:
                    metric_path = fit_path.parent / "common-validation.json"
                    projection = SCRATCH / "s3-v1/prepared/worker" / job["dataset"] / "projection.json"
                    artifact = fit_path.with_suffix(".dpk")
                    if (sha256(metric_path) != cell["metric_sha256"]
                            or json.loads(metric_path.read_text()) != metric
                            or sha256(fit_path.parent / "sample.csv") != cell["sample_sha256"]
                            or sha256(fit_path.parent / "repeat.csv") != cell["sample_sha256"]
                            or sha256(artifact) != fit["artifact_sha256"]
                            or artifact.stat().st_size + projection.stat().st_size != fit["artifact_bytes"]
                            or sha256(projection) != fit["identity"]["projection_sha256"]):
                        raise ValueError("conditional artifact, sample or metric changed")
                cells.append({"revision": revision, "stage": stage, "dataset": job["dataset"],
                              "candidate": job["candidate"], "fit_seed": job["seed"], "sample_seed": 101,
                              "status": fit["status"], "host": fit["host"],
                              "artifact_bytes": fit["artifact_bytes"], "artifact_sha256": fit["artifact_sha256"],
                              "fit_seconds": fit["elapsed_seconds"], "peak_gpu_used_mib": fit["peak_gpu_used_mib"],
                              "energy_joules_estimate": fit["energy_joules_estimate"],
                              "cell_path": str(path), "cell_sha256": sha256(path),
                              "attempt_path": str(fit_path), "attempt_sha256": sha256(fit_path),
                              "metric_sha256": cell.get("metric_sha256"), "common_validation": metric})
    baseline_path = RESULTS / "gpu-validation-research.json"
    baseline = json.loads(baseline_path.read_text())
    refs = [r for r in baseline["cells"] if r["phase"] in ("discovery", "confirmation")
            and r["family"] == "symbolic_default"]
    if len(cells) != 36 or len(refs) != 12:
        raise ValueError("conditional matrix is incomplete")
    summary = []
    for revision, stage, candidate in sorted({(c["revision"], c["stage"], c["candidate"]) for c in cells}):
        group = [c for c in cells if (c["revision"], c["stage"], c["candidate"]) == (revision, stage, candidate)]
        values = [c["common_validation"]["utility"]["catboost"].get("retention") for c in group
                  if c["common_validation"] is not None]
        values = [v for v in values if v is not None]
        summary.append({"revision": revision, "stage": stage, "candidate": candidate,
                        "fit_cells": len(group), "measured_retention_cells": len(values),
                        "median_catboost_retention": statistics.median(values) if values else None,
                        "artifacts_within_l3": sum(c["artifact_bytes"] is not None and c["artifact_bytes"] <= 10240 for c in group),
                        "copy_screen_passes": sum(c["common_validation"] is not None and all(v == 0 for v in c["common_validation"]["copy_counts"].values()) for c in group)})
    return {"format": "dope-conditional-gpu-validation", "version": 1,
            "scope": "validation_only_one_fit_seed_one_sample_seed_n_rows",
            "rounds": rounds, "cells": cells, "summary": summary,
            "symbolic_reference_cells": refs, "symbolic_results_sha256": sha256(baseline_path),
            "total_fit_seconds": sum(c["fit_seconds"] for c in cells),
            "ptf_v1": None, "mfs_v2": None, "production_certified": False,
            "official_tests_opened": False, "production_family_selected": False}


def render(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "dope-conditional-validation-v1"
    rows = []
    for c in report["cells"]:
        m = c["common_validation"]
        rows.append({k: c[k] for k in ("revision", "stage", "dataset", "candidate", "status", "artifact_bytes", "fit_seconds")}
                    | {"catboost_retention": m["utility"]["catboost"].get("retention") if m else None,
                       "exact_copies": m["copy_counts"]["exact"] if m else None,
                       "near_copies": m["copy_counts"]["near"] if m else None})
    with path.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"compact_neural_residual_symbolic": "#207a62", "symbolic_autoregressive_residual": "#305faf"}
    for ax, stage in zip(axes, ("discovery", "confirmation")):
        datasets = sorted({r["dataset"] for r in rows if r["stage"] == stage})
        for candidate, color in colors.items():
            for revision, marker, offset in (("original", "x", -0.12), ("readout_refit", "o", 0.12)):
                points = [r for r in rows if r["stage"] == stage and r["candidate"] == candidate
                          and r["revision"] == revision and r["catboost_retention"] is not None]
                if points:
                    ax.scatter([datasets.index(r["dataset"]) + offset for r in points],
                               [r["catboost_retention"] for r in points], c=color, marker=marker,
                               label=("q8" if candidate == "compact_neural_residual_symbolic" else "q10") + " / " + revision)
        refs = [r for r in report["symbolic_reference_cells"] if r["phase"] == stage and r["catboost_retention"] is not None]
        ax.scatter([datasets.index(r["dataset"]) for r in refs], [r["catboost_retention"] for r in refs],
                   c="#666666", marker="_", s=90, label="Symbolic default")
        ax.axhline(1, c="#bbbbbb", linestyle="--", linewidth=0.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(len(datasets)), [d[:6] for d in datasets], rotation=35, ha="right")
        ax.set_title(stage.capitalize())
        ax.set_ylabel("Validation CatBoost retention (symlog)")
        ax.set_xlabel("Dataset hash prefix")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("DOPE conditional GPU readout repair — validation research; no release claim", fontsize=11)
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    import jsonschema
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    path = RESULTS / "conditional-gpu-validation.json"
    report = json.loads(path.read_text()) if args.render_existing else build()
    jsonschema.validate(report, json.loads(path.with_suffix(".schema.json").read_text()))
    if not args.render_existing:
        path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    render(report, path)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
