"""Complete frozen TabPC pilot: author NLL selection and common validation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from . import publish_news_q10_packed, publish_synthpop_matched
from .manifest import digest
from .publish_native_neural_fivefit import evidence, sealed
from .score import artifact_inventory, sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
FIT = ROOT / "tabpc-native-v1"
EVAL = ROOT / "tabpc-native-validation-v1"
RESULT = HERE / "results/pilot24-tabpc-native.json"
DATASETS = ("Adult", "California", "News")
SEEDS = (101, 211, 307)
SIZES = (1, 2, 4, 8)
FIT_PIN = "718c97ae3c3294d35971b611e0a763760cc8ff748bfad7e2c0fa8046607543f0"
EVAL_PIN = "091da8cda85d0de8b22e480dfb1def149d903a297abd758b6d755c788d9b1a3c"


def read(path):
    return json.loads(Path(path).read_text())


def native_winner(attempts):
    """Common utility never enters the native likelihood selection."""
    if len(attempts) != 8 or {r["job"]["trial"] for r in attempts} != set(range(8)):
        raise ValueError("native trial matrix incomplete")
    good = [r for r in attempts if r["status"] == "ok"]
    for row in good:
        kpi = row["native_kpi"]
        if (
            kpi["direction"] != "minimize"
            or kpi["objective"] != "author_transformed_validation_mean_nll"
            or kpi["partition"] != "validation"
            or kpi["preprocessing_fit_partition"] != "train_only"
            or kpi["validation_sha256"] != row["job"]["validation_sha256"]
            or not math.isfinite(kpi["value"])
        ):
            raise ValueError("native likelihood evidence changed")
    return min(
        good,
        key=lambda r: (
            r["native_kpi"]["value"],
            r["artifact_bytes"],
            digest(r["job"]["config"]),
        ),
        default=None,
    )


def summaries(cells):
    expected = {
        (d, c, s, z)
        for d in DATASETS
        for c in ("default", "tuned")
        for s in SEEDS
        for z in (1, 4)
    }
    keys = [
        (r["dataset"], r["configuration"], r["sample_seed"], r["size_multiplier"])
        for r in cells
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("TabPC common metric matrix differs")
    result = []
    for dataset in DATASETS:
        for configuration in ("default", "tuned"):
            for size in (1, 4):
                rows = [
                    r
                    for r in cells
                    if (r["dataset"], r["configuration"], r["size_multiplier"])
                    == (dataset, configuration, size)
                ]
                values = [
                    r["metrics"]["utility"]["catboost"]["retention"]
                    for r in rows
                    if r["status"] == "ok"
                    and r["metrics"]["utility"]["catboost"]["informative"] is True
                    and r["metrics"]["utility"]["catboost"]["retention"] is not None
                ]
                if any(not math.isfinite(v) for v in values):
                    raise ValueError("nonfinite common retention")
                complete = len(values) == len(SEEDS)
                result.append(
                    {
                        "dataset": dataset,
                        "configuration": configuration,
                        "size_multiplier": size,
                        "complete_informative_samples": len(values),
                        "median_catboost_retention": (
                            median(values) if complete else None
                        ),
                        "min_catboost_retention": min(values) if complete else None,
                        "max_catboost_retention": max(values) if complete else None,
                        "artifact_bytes": rows[0]["artifact_bytes"],
                        "cell_status_counts": dict(Counter(r["status"] for r in rows)),
                    }
                )
    return result


def sample_counts(rows):
    keys = [
        (
            r["job"]["dataset"],
            tuple(r["job"]["config_tracks"]),
            r["job"]["sample_seed"],
            r["job"]["size_multiplier"],
        )
        for r in rows
    ]
    expected = {
        (d, (c,), s, z)
        for d in ("California", "News")
        for c in ("default", "tuned")
        for s in SEEDS
        for z in SIZES
    }
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("TabPC wider sample matrix incomplete")
    return dict(Counter(r["status"] for r in rows))


def build():
    fit_path, eval_path = FIT / "round.lock.json", EVAL / "round.lock.json"
    fit_lock, eval_lock = read(fit_path), read(eval_path)
    sealed(fit_lock)
    sealed(eval_lock)
    if (
        sha256(fit_path) != FIT_PIN
        or sha256(eval_path) != EVAL_PIN
        or eval_lock["parent_round_sha256"] != FIT_PIN
        or len(fit_lock["jobs"]) != 24
        or len(eval_lock["jobs"]) != 48
        or eval_lock["new_fits_started"] != 0
        or eval_lock["new_tuning_trials"] != 0
        or fit_lock["per_cell_trial_cap"] != 8
        or fit_lock["per_cell_wall_cap_seconds"] != 43200
        or fit_lock["per_fit_seconds"] != 600
        or fit_lock["gpu_vram_mib_cap"] != 16384
    ):
        raise ValueError("TabPC frozen pilot identity changed")
    for root, lock in ((FIT, fit_lock), (EVAL, eval_lock)):
        for name, h in lock["source_manifest"].items():
            if sha256(root / name) != h:
                raise ValueError("TabPC frozen source changed")
        if sha256(Path(lock["runtime_lock"]["path"])) != lock["runtime_lock"]["sha256"]:
            raise ValueError("TabPC runtime identity changed")
    attempts, by_job, worker_hashes = [], {}, {}
    for job in fit_lock["jobs"]:
        path = FIT / "jobs" / digest(job) / "attempt-0001/receipt.json"
        receipt = read(path)
        sealed(receipt)
        evidence(path.parent, receipt)
        if receipt["job"] != job or receipt["round_lock_sha256"] != FIT_PIN:
            raise ValueError("TabPC native attempt lineage changed")
        host = read(path.parent / "host.json")
        if (
            host["active_gpu_processes"]
            or host["gpus"][0]["memory_free_mib"] < fit_lock["minimum_free_vram_mib"]
            or host["memory"]["MemAvailable"] < fit_lock["minimum_available_ram_bytes"]
            or not set(receipt["cpu_slot"]).issubset(host["allowed_cpus"])
            or receipt["peak_gpu_used_mib"] > 16384
        ):
            raise ValueError("TabPC native admission or memory cap changed")
        worker = Path(fit_lock["worker_root"]) / job["dataset"]
        if (worker / "test.csv").exists() or "evaluator" in worker.resolve().parts:
            raise ValueError("TabPC worker includes official tests")
        hashes = {
            n: sha256(worker / n)
            for n in ("train.csv", "validation.csv", "projection.json")
        }
        if (
            any(
                hashes[f"{n}.csv"] != job[f"{n}_sha256"]
                for n in ("train", "validation")
            )
            or hashes["projection.json"] != job["projection_sha256"]
        ):
            raise ValueError("TabPC training-derived worker changed")
        worker_hashes[job["dataset"]] = hashes
        child = receipt.get("child", {})
        if (
            receipt["status"] == "failed"
            and "torch.OutOfMemoryError" not in (path.parent / "stderr").read_text()
        ):
            raise ValueError("TabPC native failure classification changed")
        if receipt["status"] == "ok":
            inventory, charged = artifact_inventory(
                path.parent / "artifact",
                [r["path"] for r in child["artifact_inventory"]],
            )
            if (
                inventory != child["artifact_inventory"]
                or charged != child["artifact_bytes"]
                or receipt["elapsed_seconds"] > 600
            ):
                raise ValueError(
                    "TabPC artifact charge or successful fit timeout changed"
                )
        row = {
            "job": job,
            "status": receipt["status"],
            "receipt_path": str(path),
            "receipt_sha256": sha256(path),
            "elapsed_seconds": receipt["elapsed_seconds"],
            "native_kpi": child.get("native_kpi"),
            "artifact_bytes": child.get("artifact_bytes"),
            "peak_gpu_used_mib": receipt["peak_gpu_used_mib"],
            "energy_joules_estimate": receipt["energy_joules_estimate"],
            "host": receipt["host"],
            "evidence_files": receipt["evidence_files"],
            "source_unavailable": False,
            "contributes_dope_win": False,
        }
        attempts.append(row)
        by_job[digest(job)] = row
    selections = []
    for reference in eval_lock["native_selection_receipts"]:
        path = Path(reference["path"])
        selected = read(path)
        sealed(selected)
        rows = [r for r in attempts if r["job"]["dataset"] == reference["dataset"]]
        winner = native_winner(rows)
        if (
            sha256(path) != reference["sha256"]
            or selected["round_lock_sha256"] != FIT_PIN
            or selected["objective"] != fit_lock["native_objective"]
            or selected["shared_kpi_used_for_selection"] is not False
            or selected["selected_trial"]
            != (winner["job"]["trial"] if winner else None)
            or sum(r["elapsed_seconds"] for r in rows) > 43200
        ):
            raise ValueError("TabPC native selection changed")
        for prior_attempt in selected["attempts"]:
            current = by_job[digest(prior_attempt["job"])]
            if (
                any(
                    prior_attempt[key] != current[key]
                    for key in ("status", "native_kpi", "artifact_bytes")
                )
                or prior_attempt["sha256"] != current["receipt_sha256"]
            ):
                raise ValueError("TabPC selection attempt changed")
        selections.append(
            {
                "dataset": reference["dataset"],
                "selected_trial": selected["selected_trial"],
                "default_native_kpi": next(r for r in rows if r["job"]["trial"] == 0)[
                    "native_kpi"
                ],
                "selected_native_kpi": winner["native_kpi"] if winner else None,
                "receipt_path": str(path),
                "receipt_sha256": sha256(path),
                "status": selected["status"],
            }
        )
    bindings = {}
    for reference in eval_lock["bindings"]:
        path = Path(reference["path"])
        binding = read(path)
        original = by_job[digest(binding["fit_job"])]
        current, charged = artifact_inventory(
            Path(binding["artifact_path"]),
            [r["path"] for r in binding["current_inventory"]],
        )
        prior = {r["path"]: r for r in binding["prior_inventory"]}
        metadata = read(Path(binding["artifact_path"]) / "adapter.json")
        if (
            sha256(path) != reference["sha256"]
            or binding["fit_receipt_sha256"] != original["receipt_sha256"]
            or binding["learned_parameter_bytes_changed"] is not False
            or binding["metadata_only_rebinding"] is not True
            or current != binding["current_inventory"]
            or charged != binding["artifact_bytes"]
            or charged != original["artifact_bytes"]
            or metadata["runtime_lock_sha256"] != binding["current_runtime_lock_sha256"]
            or binding["current_runtime_lock_sha256"]
            != eval_lock["runtime_lock"]["sha256"]
            or binding["prior_runtime_lock_sha256"]
            != fit_lock["runtime_lock"]["sha256"]
            or any(
                row != prior[row["path"]]
                for row in current
                if row["path"] != "adapter.json"
            )
        ):
            raise ValueError("TabPC learned bytes or metadata binding changed")
        bindings[str(path)] = binding
    sample_rows, cells = [], []
    for job in eval_lock["jobs"]:
        path = EVAL / "cells" / digest(job) / "attempt-0001.json"
        receipt = read(path)
        sealed(receipt)
        evidence(path.parent / path.stem, receipt)
        binding = bindings[job["binding_path"]]
        if (
            receipt["job"] != job
            or receipt["round_sha256"] != EVAL_PIN
            or sha256(Path(job["binding_path"])) != job["binding_sha256"]
            or receipt["fit_receipt_sha256"] != binding["fit_receipt_sha256"]
            or receipt["artifact_bytes"] != binding["artifact_bytes"]
            or job["fit_seed"] != 23
            or job["train_sha256"] != worker_hashes[job["dataset"]]["train.csv"]
            or job["validation_sha256"]
            != worker_hashes[job["dataset"]]["validation.csv"]
            or job["projection_sha256"]
            != worker_hashes[job["dataset"]]["projection.json"]
        ):
            raise ValueError("TabPC sample lineage changed")
        metrics = None
        if receipt["status"] == "ok":
            if (
                sha256(path.parent / path.stem / "sample.csv")
                != receipt["sample_sha256"]
                or receipt["repeat_sha256"] != receipt["sample_sha256"]
            ):
                raise ValueError("TabPC sample replay changed")
            if job["size_multiplier"] in (1, 4):
                metric_path = path.parent / path.stem / "metrics.json"
                metrics = read(metric_path)
                if (
                    sha256(metric_path) != receipt["metric_sha256"]
                    or receipt["metric_replay_status"] != "exact"
                    or metrics["implementation_sha256"]
                    != eval_lock["metric_source_sha256"]
                    or metrics["rows"]["synthetic"] != job["row_count"]
                    or metrics["gate_profile_complete"] is not False
                    or metrics["mfs_v2"] is not None
                ):
                    raise ValueError("TabPC common metric evidence changed")
        sample_rows.append(
            {
                "job": job,
                "status": receipt["status"],
                "artifact_bytes": receipt["artifact_bytes"],
                "receipt_path": str(path),
                "receipt_sha256": sha256(path),
                "fit_receipt_sha256": receipt["fit_receipt_sha256"],
                "sample_sha256": receipt["sample_sha256"],
                "metric_sha256": receipt["metric_sha256"],
                "elapsed_seconds": receipt["elapsed_seconds"],
            }
        )
        if job["size_multiplier"] in (1, 4):
            for track in job["config_tracks"]:
                cells.append(
                    {
                        "dataset": job["dataset"],
                        "configuration": track,
                        "fit_seed": 23,
                        "sample_seed": job["sample_seed"],
                        "size_multiplier": job["size_multiplier"],
                        "status": receipt["status"],
                        "artifact_bytes": receipt["artifact_bytes"],
                        "metrics": metrics,
                        "sample_receipt_sha256": sha256(path),
                        "metric_receipt_sha256": receipt["metric_sha256"],
                        "source_unavailable": False,
                        "contributes_dope_win": False,
                    }
                )
    sample_counts(sample_rows)
    for failed in eval_lock["failed_tracks"]:
        if (
            failed["source_unavailable"] is not False
            or failed["contributes_dope_win"] is not False
        ):
            raise ValueError("TabPC native failures were counted as wins")
        if (
            failed["fit_receipt_path"] is not None
            and sha256(Path(failed["fit_receipt_path"])) != failed["fit_receipt_sha256"]
        ):
            raise ValueError("TabPC failed fit evidence changed")
        for seed in SEEDS:
            for size in (1, 4):
                cells.append(
                    {
                        "dataset": failed["dataset"],
                        "configuration": failed["configuration"],
                        "fit_seed": 23,
                        "sample_seed": seed,
                        "size_multiplier": size,
                        "status": "fit_unavailable",
                        "artifact_bytes": None,
                        "metrics": None,
                        "sample_receipt_sha256": None,
                        "metric_receipt_sha256": None,
                        "source_unavailable": False,
                        "contributes_dope_win": False,
                    }
                )
    packed_path = HERE / "results/pilot24-news-q10-packed.json"
    if publish_news_q10_packed.build() != read(packed_path):
        raise ValueError("DOPE packed validation proof changed")
    dope = publish_synthpop_matched.dope_reference()
    dope_rows = [
        {"dataset": d, "sample_seed": s, "size_multiplier": z, **value}
        for (d, s, z), value in sorted(dope.items())
    ]
    return {
        "format": "dope-tabpc-complete-native-pilot-validation",
        "version": 1,
        "scope": "training_derived_validation_only_single_fit_seed",
        "official_tests_opened": False,
        "mfs_v2": None,
        "ptf_v1": None,
        "release_safe": None,
        "production_certified": False,
        "source_sha256": sha256(Path(__file__)),
        "helper_source_sha256": {
            name: sha256(HERE / name)
            for name in (
                "manifest.py",
                "score.py",
                "publish_native_neural_fivefit.py",
                "publish_news_q10_packed.py",
                "publish_synthpop_matched.py",
            )
        },
        "native_round_sha256": FIT_PIN,
        "validation_round_sha256": EVAL_PIN,
        "native_objective": fit_lock["native_objective"],
        "native_attempts": attempts,
        "native_status_counts": dict(Counter(r["status"] for r in attempts)),
        "native_selections": selections,
        "sample_receipts": sample_rows,
        "sample_status_counts": sample_counts(sample_rows),
        "failed_tracks": eval_lock["failed_tracks"],
        "bindings": eval_lock["bindings"],
        "source_manifests": {
            "native": fit_lock["source_manifest"],
            "validation": eval_lock["source_manifest"],
        },
        "runtime_locks": {
            "training": fit_lock["runtime_lock"],
            "sampling": eval_lock["runtime_lock"],
        },
        "worker_hashes": worker_hashes,
        "common_cells": cells,
        "summaries": summaries(cells),
        "dope_references": dope_rows,
        "input_report_sha256": {
            name: sha256(HERE / "results" / name)
            for name in ("pilot24-native-neural.json", "pilot24-news-q10-packed.json")
        },
        "cost": {
            "native_attempt_seconds": sum(r["elapsed_seconds"] for r in attempts),
            "common_sample_and_metric_seconds": sum(
                r["elapsed_seconds"] for r in sample_rows
            ),
            "new_validation_fits": 0,
            "new_validation_tuning_trials": 0,
        },
        "notes": [
            "Author TabPC and Cirkit sources are Apache-2.0 and GPL-3.0-or-later respectively, as pinned in TABPC.md; this is a research comparator, not a product dependency.",
            "Each dataset has eight native validation-NLL trials; no common retention or DOPE KPI selected TabPC configurations.",
            "Adult default fails and all eight native trials fail; its two unavailable tracks confer no DOPE win.",
            "All 48 planned common sample jobs and 24 measured n/4n jobs are complete. The three News tuned 8n timeouts have no inferred utility.",
            "These are one-fit-seed descriptive validation outcomes, not the final five-fit campaign or a paired public-core superiority test.",
            "The copied V5 runtime rebinds metadata only; original source, receipts and learned bytes are preserved. No new fits occurred during common validation.",
            "DOPE references reuse prior q8 and losslessly packed q10 samples on the same normalized inputs; inverse/projection-only utility, full privacy attacks and release gates remain unmeasured.",
            "MFS-v2/PTF-v1 and release-safe status remain null; no formal privacy, certification or production-coverage claim follows.",
        ],
    }


def render(report):
    sealed(report)
    if report["sample_status_counts"] != sample_counts(report["sample_receipts"]):
        raise ValueError("TabPC sample counts differ from receipts")
    if report["summaries"] != summaries(report["common_cells"]):
        raise ValueError("TabPC summary differs from complete cells")
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "dataset",
            "configuration",
            "fit_seed",
            "sample_seed",
            "size_multiplier",
            "status",
            "artifact_bytes",
            "catboost_retention",
            "linear_retention",
            "mlp_retention",
            "sample_receipt_sha256",
            "metric_receipt_sha256",
        )
    )
    for row in report["common_cells"]:
        values = [
            row["metrics"]["utility"][a]["retention"] if row["metrics"] else None
            for a in ("catboost", "linear", "mlp")
        ]
        writer.writerow(
            [
                row[k]
                for k in (
                    "dataset",
                    "configuration",
                    "fit_seed",
                    "sample_seed",
                    "size_multiplier",
                    "status",
                    "artifact_bytes",
                )
            ]
            + values
            + [row["sample_receipt_sha256"], row["metric_receipt_sha256"]]
        )
    fmt = lambda value: "unavailable" if value is None else f"{value:.4f}"
    lines = [
        "# TabPC author-native pilot and matched DOPE validation",
        "",
        "Complete frozen pilot scope; fit seed23, sample seeds101/211/307. Eight TabPC trials per dataset selected only by author transformed validation mean NLL (minimize). Native NLL values are not ranked against other methods. Official tests remain sealed; MFS-v2/PTF-v1 and release-safe status remain null.",
        "",
        "## Common CatBoost retention",
        "",
        "| Dataset | Method | Configuration | n | 4n | Charged bytes |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        dope = [r for r in report["dope_references"] if r["dataset"] == dataset]
        values = {
            z: median(r["retention"] for r in dope if r["size_multiplier"] == z)
            for z in (1, 4)
        }
        lines.append(
            f'| {dataset} | DOPE | {dope[0]["configuration"]} | {fmt(values[1])} | {fmt(values[4])} | {dope[0]["artifact_bytes"]} |'
        )
        for config in ("default", "tuned"):
            rows = {
                r["size_multiplier"]: r
                for r in report["summaries"]
                if r["dataset"] == dataset and r["configuration"] == config
            }
            charged = rows[1]["artifact_bytes"]
            lines.append(
                f'| {dataset} | TabPC | {config} | {fmt(rows[1]["median_catboost_retention"])} | {fmt(rows[4]["median_catboost_retention"])} | {charged if charged is not None else "unavailable"} |'
            )
    lines += [
        "",
        "Group medians require all three informative sample-seed results. The JSON retains every sample, all three auditor losses/retentions, marginal/joint/C2ST screens, exact/near-copy counts and real-vs-real controls. The CSV exports n/4n cell status, bytes and three auditor retentions. Artifact byte eligibility is separate from release safety; all successful TabPC artifacts exceed 10,240 bytes.",
        "",
        "## Native validation NLL",
        "",
        "| Dataset | Default NLL | Selected NLL | Selected trial |",
        "|---|---:|---:|---:|",
    ]
    for row in report["native_selections"]:
        lines.append(
            f'| {row["dataset"]} | {fmt(row["default_native_kpi"]["value"] if row["default_native_kpi"] else None)} | {fmt(row["selected_native_kpi"]["value"] if row["selected_native_kpi"] else None)} | {row["selected_trial"] if row["selected_trial"] is not None else "unavailable"} |'
        )
    lines += [
        "",
        "## Complete failures, sources and costs",
        "",
        f'Native fit statuses: `{json.dumps(report["native_status_counts"], sort_keys=True)}` across24 attempts. Sample statuses: `{json.dumps(report["sample_status_counts"], sort_keys=True)}` across48 jobs. All24 planned n/4n metric cells succeeded; the three News tuned8n sample timeouts remain explicit. Adult default and tuned are unavailable after eight native GPU allocation failures. Executable failures confer no DOPE win.',
        "",
        f'Summed native attempt time: {report["cost"]["native_attempt_seconds"]:.2f}s. Common sampling/evaluation/replay elapsed time: {report["cost"]["common_sample_and_metric_seconds"]:.2f}s. No new common-validation fits or tuning trials. Other DOPE and comparator R&D is reported separately, not claimed equal.',
        "",
        "Source/runtime commits, license evidence, compatibility changes and charged fit/sample contracts are in [TABPC.md](../TABPC.md). Original training artifacts and receipts are preserved; V5 changes only integrity/error handling and metadata binding. Bulk data, weights and detailed logs remain on scratch.",
        "",
        "This is a complete single-fit-seed pilot panel; final five-fit campaign locks, public-core paired analysis, projection-only utility, full privacy attacks and production coverage remain incomplete. Official tests remain sealed and every gated score is null.",
        "",
    ]
    return stream.getvalue(), "\n".join(lines)


def main():
    from jsonschema import Draft202012Validator

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    schema = read(RESULT.with_suffix(".schema.json"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    if (
        report["source_sha256"] != sha256(Path(__file__))
        or report["production_certified"] is not False
    ):
        raise ValueError("TabPC publication source or claim changed")
    table, markdown = render(report)
    if not args.from_json:
        RESULT.write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    RESULT.with_suffix(".csv").write_text(table)
    RESULT.with_suffix(".md").write_text(markdown)
    print(
        json.dumps(
            {
                "native_attempts": len(report["native_attempts"]),
                "samples": len(report["sample_receipts"]),
                "common_cells": len(report["common_cells"]),
            }
        )
    )


if __name__ == "__main__":
    main()
