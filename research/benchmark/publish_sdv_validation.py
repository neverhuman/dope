"""Native selection and shared validation outcomes, without official test access."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from . import pilot_metrics
from .manifest import digest
from .score import sha256
from .sdv_round import ROOT, WORKERS, verify_attempt, write_once


RESULTS = Path(__file__).with_name("results")


def winner(trials: list[dict]) -> dict | None:
    successful = [trial for trial in trials if trial["status"] == "ok"]
    return min(successful, key=lambda trial: (-trial["native_kpi"]["value"],
               trial["artifact_bytes"], digest(trial["job"]["config"]))) if successful else None


def final_selection(lock_path: Path, trials: list[dict], paths: dict, best: dict) -> None:
    """Bind the measured native choice to the shared runner's selection contract."""
    from .runner import resolve_configuration
    job = best["job"]
    entry = json.loads(Path(__file__).with_name("methods.lock.json").read_text())["methods"][job["method"]]
    root = ROOT / "runner-selections" / f"{job['dataset']}-{job['method']}"
    rows = []
    for trial in trials:
        trial_job = trial["job"]
        row = {"config": trial_job["config"], "status": trial["status"] if trial["status"] in
               ("ok", "timeout") else "failed", "wall_seconds": trial["wall_seconds"],
               "attempt_sha256": sha256(paths[digest(trial_job)])}
        if trial["status"] == "ok":
            model_hash = next(item["sha256"] for item in trial["artifact_inventory"] if item["path"] == "model.pt")
            metric = {**trial["native_kpi"], "method": trial_job["method"], "artifact_sha256": model_hash,
                      "source_attempt_sha256": row["attempt_sha256"]}
            metric_path = root / f"trial-{trial_job['trial']:02d}.metric.json"
            if metric_path.exists():
                if json.loads(metric_path.read_text()) != metric:
                    raise ValueError("canonical native metric changed")
            else:
                write_once(metric_path, metric)
            row.update({"native_kpi": metric["value"], "artifact_bytes": trial["artifact_bytes"],
                        "artifact_sha256": model_hash, "metric_receipt_path": str(metric_path),
                        "metric_receipt_sha256": sha256(metric_path)})
        rows.append(row)
    selected = {"format": "dope-benchmark-validation-selection", "version": 1,
                "dataset": job["dataset"], "method": job["method"], "partition": "validation",
                "round_sha256": sha256(lock_path), "objective": entry["native_objective"],
                "selected_trial_index": trials.index(best), "selected_config": job["config"],
                "trials": rows, "total_wall_seconds": sum(t["wall_seconds"] for t in trials),
                "test_opened": False}
    output = root / "selection.json"
    if output.exists():
        if json.loads(output.read_text()) != selected:
            raise ValueError("canonical native selection changed")
    else:
        write_once(output, selected)
    resolve_configuration({"method": job["method"], "configuration": {
        "kind": "tuned", "values": job["config"], "selection_path": str(output),
        "selection_sha256": sha256(output)}}, entry, job["dataset"], job["train_rows"], ROOT)


def shared_metrics(receipt_path: Path, receipt: dict) -> dict:
    job = receipt["job"]
    output = ROOT / "shared-metrics" / f"{digest(job)}.json"
    source_hash = sha256(Path(pilot_metrics.__file__))
    identity = {"attempt_sha256": sha256(receipt_path), "metric_source_sha256": source_hash,
                "train_sha256": job["train_sha256"], "validation_sha256": job["validation_sha256"],
                "sample_sha256": receipt["sample"]["child"]["sha256"]}
    worker = WORKERS / job["dataset"]
    for part in ("train", "validation"):
        if sha256(worker / f"{part}.csv") != job[f"{part}_sha256"]:
            raise ValueError("shared metric worker changed")
    if sha256(receipt_path.parent / "sample.csv") != identity["sample_sha256"]:
        raise ValueError("shared metric sample changed")
    if output.exists():
        report = json.loads(output.read_text())
        if report["identity"] != identity:
            raise ValueError("existing shared metric identity changed")
    else:
        report = {"identity": identity, "metrics": pilot_metrics.measure(
            worker / "train.csv", worker / "validation.csv", receipt_path.parent / "sample.csv", "regression")}
        write_once(output, report)
    m = report["metrics"]
    return {"metric_receipt_path": str(output), "metric_receipt_sha256": sha256(output),
            "retention": {name: values.get("retention") for name, values in m["utility"].items()},
            "marginal_ks_mean": m["marginal_ks_mean"], "c2st_auc": m["c2st_auc"],
            "copy_counts": m["copy_counts"], "real_vs_real_control_counts": m["real_vs_real_control_counts"]}


def build(evaluate: bool) -> dict:
    lock_path = ROOT / "round.lock.json"
    lock = json.loads(lock_path.read_text())
    cells = []
    for dataset, method in sorted({(job["dataset"], job["method"]) for job in lock["jobs"]}):
        jobs = [j for j in lock["jobs"] if (j["dataset"], j["method"]) == (dataset, method)]
        trials = []
        paths = {}
        for job in jobs:
            found = sorted((ROOT / "jobs" / digest(job)).glob("attempt-*/receipt.json"))
            if not found:
                continue
            p = found[-1]
            receipt = json.loads(p.read_text())
            if receipt["job"] != job or receipt["round_sha256"] != sha256(lock_path):
                raise ValueError("trial identity changed")
            verify_attempt(p.parent, receipt)
            trials.append(receipt)
            paths[digest(job)] = p
        if len(trials) != len(jobs):
            continue
        best = winner(trials)
        if best:
            final_selection(lock_path, trials, paths, best)
        selection = {"format": "dope-sdv-validation-selection", "version": 1,
                     "round_sha256": sha256(lock_path), "dataset": dataset, "method": method,
                     "objective": lock["methods"][method]["native_objective"],
                     "selected_job": digest(best["job"]) if best else None,
                     "attempts": [{"path": str(paths[digest(t["job"])]),
                                   "sha256": sha256(paths[digest(t["job"])])} for t in trials]}
        selected_path = ROOT / "selections" / f"{dataset}-{method}.json"
        if selected_path.exists():
            if json.loads(selected_path.read_text()) != selection:
                raise ValueError("native selection changed")
        else:
            write_once(selected_path, selection)
        rows = []
        for trial in trials:
            job = trial["job"]
            key = digest(job)
            row = {"trial": job["trial"], "configuration": job["config"],
                   "status": trial["status"], "host": trial["host"],
                   "native_validation_kpi": trial.get("native_kpi"),
                   "artifact_bytes": trial.get("artifact_bytes"),
                   "fit_seconds": trial.get("fit", {}).get("wall_seconds"),
                   "wall_seconds": trial["wall_seconds"],
                   "peak_gpu_used_mib": trial.get("fit", {}).get("peak_device_used_mib"),
                   "energy_joules_estimate": trial.get("fit", {}).get("energy_joules_estimate"),
                   "attempt_path": str(paths[key]), "attempt_sha256": sha256(paths[key]),
                   "default": job["trial"] == 0, "native_selected": best is trial,
                   "common_validation": None}
            if trial["status"] == "ok" and (row["default"] or row["native_selected"]) and evaluate:
                row["common_validation"] = shared_metrics(paths[key], trial)
            rows.append(row)
        cells.append({"dataset": dataset, "method": method, "stage": jobs[0]["stage"],
                      "selection_receipt_sha256": sha256(selected_path), "trials": rows})
    dope_path = RESULTS / "gpu-validation-research.json"
    dope = json.loads(dope_path.read_text())
    datasets = {j["dataset"] for j in lock["jobs"]}
    baseline = [row for row in dope["cells"] if row["dataset"] in datasets
                and row["phase"] in ("discovery", "confirmation")
                and (row["family"] == "symbolic_default" or
                     (row["family"] == "micro_tvae_4_16" and row["target_weight"] == 2.0
                      and row["structural_penalty"] == 0.0))]
    if len(baseline) != 24:
        raise ValueError("DOPE research reference is incomplete")
    dope_root = ROOT.parent / "gpu-research-v1"
    for number in (0, 2):
        prior_lock = dope_root / f"round{number}.lock.json"
        if (sha256(prior_lock) != dope["round_lock_sha256"][str(number)]
                or json.loads(prior_lock.read_text())["cohort_lock_sha256"] != lock["cohort_sha256"]):
            raise ValueError("DOPE reference uses a different cohort")
    for row in baseline:
        if sha256(Path(row["attempt_receipt_path"])) != row["attempt_receipt_sha256"]:
            raise ValueError("DOPE reference attempt changed")
    refined_path = RESULTS / "conditional-gpu-validation.json"
    refined = json.loads(refined_path.read_text())
    for row in refined["rounds"]:
        p = Path(row["path"])
        if sha256(p) != row["sha256"] or json.loads(p.read_text())["cohort_sha256"] != lock["cohort_sha256"]:
            raise ValueError("DOPE refinement reference cohort changed")
    refined_cells = [c for c in refined["cells"] if c["revision"] == "readout_refit"]
    if len(refined_cells) != 24 or {c["dataset"] for c in refined_cells} != datasets:
        raise ValueError("DOPE refinement reference incomplete")
    return {"format": "dope-sdv-native-validation-comparison", "version": 1,
            "scope": "validation_only_one_fit_seed_one_sample_seed_n_rows",
            "round_lock_path": str(lock_path), "round_lock_sha256": sha256(lock_path),
            "dope_results_sha256": sha256(dope_path), "methods": lock["methods"],
            "dope_refinement_results_sha256": sha256(refined_path),
            "dope_refinement_cells": refined_cells,
            "cells": cells, "dope_reference_cells": baseline,
            "planned_fit_attempts": len(lock["jobs"]),
            "complete": len(cells) == 24 and evaluate,
            "selection_optimism": "Selected configurations are measured on the validation partition used for native selection; no test generalization claim.",
            "ptf_v1": None, "mfs_v2": None, "production_certified": False,
            "official_tests_opened": False}


def render(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams["svg.hashsalt"] = "dope-sdv-validation-v1"
    rows = []
    for ref in report["dope_reference_cells"]:
        rows.append({"dataset": ref["dataset"], "method": "DOPE symbolic" if ref["family"] == "symbolic_default"
                     else "DOPE GPU micro-TVAE", "kind": "default", "status": ref["status"],
                     "native_kpi": None, "bytes": ref["artifact_bytes"],
                     "retention": ref["catboost_retention"], "stage": ref["phase"],
                     "exact_copies": ref["exact_copies"], "near_copies": ref["near_copies"]})
    for ref in report["dope_refinement_cells"]:
        metric = ref["common_validation"]
        rows.append({"dataset": ref["dataset"], "method": "DOPE GPU conditional " +
                     ("q8" if ref["candidate"] == "compact_neural_residual_symbolic" else "q10"),
                     "kind": "default", "status": ref["status"], "native_kpi": None,
                     "bytes": ref["artifact_bytes"], "retention": metric["utility"]["catboost"].get("retention")
                     if metric else None, "stage": ref["stage"],
                     "exact_copies": metric["copy_counts"]["exact"] if metric else None,
                     "near_copies": metric["copy_counts"]["near"] if metric else None})
    for cell in report["cells"]:
        for kind in ("default", "native_selected"):
            for trial in cell["trials"]:
                if trial[kind]:
                    metric = trial["common_validation"]
                    rows.append({"dataset": cell["dataset"], "method": cell["method"], "kind": kind,
                                 "status": trial["status"], "bytes": trial["artifact_bytes"],
                                 "native_kpi": trial["native_validation_kpi"]["value"] if trial["native_validation_kpi"] else None,
                                 "retention": metric["retention"]["catboost"] if metric else None,
                                 "stage": cell["stage"],
                                 "exact_copies": metric["copy_counts"]["exact"] if metric else None,
                                 "near_copies": metric["copy_counts"]["near"] if metric else None})
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"DOPE symbolic": "#285f86", "DOPE GPU micro-TVAE": "#50a7c2", "CTGAN": "#c06b23", "TVAE": "#774c9b",
              "DOPE GPU conditional q8": "#207a62", "DOPE GPU conditional q10": "#68852b"}
    for ax, stage in zip(axes, ("discovery", "confirmation")):
        for row in rows:
            if row["stage"] == stage and row["retention"] is not None and row["bytes"]:
                ax.scatter(row["bytes"], row["retention"], c=colors[row["method"]],
                           marker="x" if row["kind"] == "native_selected" else "o", alpha=0.7, s=38)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1)
        ax.axvline(10240, linestyle="--", color="#444444", linewidth=1)
        ax.set_xlabel("Charged artifact bytes (log scale); dashed line = L3 cap")
        ax.set_title(stage.capitalize())
        ax.set_ylabel("Validation CatBoost retention (symlog)")
        ax.spines[["top", "right"]].set_visible(False)
    legend = [Line2D([0], [0], marker="o", linestyle="", color=c, label=n) for n, c in colors.items()]
    legend.append(Line2D([0], [0], marker="x", linestyle="", color="black", label="Native selected"))
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Shared S3 validation data — one fit/sample seed; not a test or release result", fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    summary = {}
    for method, kind in sorted({(row["method"], row["kind"]) for row in rows}):
        group = [r for r in rows if (r["method"], r["kind"]) == (method, kind)]
        values = [r["retention"] for r in group if r["retention"] is not None]
        summary[method + " / " + kind] = {"cells": len(group), "measured": len(values),
            "median_catboost_retention": statistics.median(values) if values else None,
            "artifacts_within_l3": sum(r["bytes"] is not None and r["bytes"] <= 10240 for r in group)}
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    if args.render_existing:
        import jsonschema
        path = RESULTS / "sdv-native-validation.json"
        report = json.loads(path.read_text())
        jsonschema.validate(report, json.loads((RESULTS / "sdv-native-validation.schema.json").read_text()))
        render(report, path)
        return
    report = build(args.evaluate)
    if args.publish:
        if not report["complete"]:
            raise ValueError("comparison matrix is incomplete")
        import jsonschema
        schema = json.loads((RESULTS / "sdv-native-validation.schema.json").read_text())
        jsonschema.validate(report, schema)
        path = RESULTS / "sdv-native-validation.json"
        path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
        render(report, path)
    else:
        print(json.dumps({"completed_cells": len(report["cells"]), "complete": report["complete"]}))


if __name__ == "__main__":
    main()
