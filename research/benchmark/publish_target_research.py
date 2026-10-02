"""Publish complete bounded GPU research and matched native references."""

from __future__ import annotations
import argparse, csv, io, json, math
from collections import Counter
from pathlib import Path
from statistics import median
from research.benchmark.score import sha256
from research.benchmark.publish_native_neural_fivefit import sealed
from research.benchmark.target_research_receipts import (
    ROOT,
    DATASETS,
    PROFILES,
    FIT_SEEDS,
    SAMPLE_SEEDS,
    SIZES,
    AUDITORS,
    PINS,
    read,
    require,
    reference,
    keys,
    fit_attempts,
    validation_cells,
)

HERE = Path(__file__).resolve().parent


def summarize(cells):
    expected = {
        (d, p, f, s, z)
        for d in DATASETS
        for p in PROFILES
        for f in FIT_SEEDS
        for s in SAMPLE_SEEDS
        for z in SIZES
    }
    actual = [keys(r) for r in cells]
    require(
        len(actual) == len(set(actual)) and set(actual) == expected,
        "eligible research matrix incomplete or duplicated",
    )
    result = []
    for dataset in DATASETS:
        for profile in PROFILES:
            for size in SIZES:
                for auditor in AUDITORS:
                    rows = [
                        r
                        for r in cells
                        if (r["dataset"], r["profile"], r["size_multiplier"])
                        == (dataset, profile, size)
                    ]
                    fits = []
                    for seed in FIT_SEEDS:
                        values = [
                            r["metrics"]["utility"][auditor]["retention"]
                            for r in rows
                            if r["fit_seed"] == seed
                            and r["status"] == "ok"
                            and r["metrics"]["utility"][auditor]["informative"] is True
                            and r["metrics"]["utility"][auditor]["retention"]
                            is not None
                        ]
                        require(
                            all(math.isfinite(v) for v in values),
                            "nonfinite research retention",
                        )
                        fits.append(
                            {
                                "fit_seed": seed,
                                "informative_samples": len(values),
                                "median_retention": (
                                    median(values) if len(values) == 3 else None
                                ),
                            }
                        )
                    values = [
                        r["median_retention"]
                        for r in fits
                        if r["median_retention"] is not None
                    ]
                    result.append(
                        {
                            "dataset": dataset,
                            "profile": profile,
                            "size_multiplier": size,
                            "auditor": auditor,
                            "fit_medians": fits,
                            "complete_fit_seeds": len(values),
                            "median_of_fit_medians": (
                                median(values) if len(values) == 2 else None
                            ),
                            "status_counts": dict(Counter(r["status"] for r in rows)),
                        }
                    )
    return result


def baseline_summary(cells):
    groups = {
        "Adult": (
            ("CTGAN", "default"),
            ("CTGAN", "tuned"),
            ("TVAE", "default_and_tuned"),
            ("DOPE", "q8_selected"),
        ),
        "California": (
            ("CTGAN", "default_and_tuned"),
            ("TVAE", "default"),
            ("TVAE", "tuned"),
            ("DOPE", "q8_selected"),
        ),
    }
    expected = {
        (d, m, c, f, s, z)
        for d, values in groups.items()
        for m, c in values
        for f in FIT_SEEDS
        for s in SAMPLE_SEEDS
        for z in SIZES
    }
    actual = [
        (
            r["dataset"],
            r["method"],
            r["configuration"],
            r["fit_seed"],
            r["sample_seed"],
            r["size_multiplier"],
        )
        for r in cells
    ]
    require(
        len(actual) == len(set(actual)) and set(actual) == expected,
        "matched baseline matrix incomplete",
    )
    rows = []
    for dataset, configs in groups.items():
        for method, config in configs:
            for size in SIZES:
                for auditor in AUDITORS:
                    group = [
                        r
                        for r in cells
                        if (
                            r["dataset"],
                            r["method"],
                            r["configuration"],
                            r["size_multiplier"],
                        )
                        == (dataset, method, config, size)
                    ]
                    values = []
                    for seed in FIT_SEEDS:
                        retained = [
                            r["utility"][auditor]["retention"]
                            for r in group
                            if r["fit_seed"] == seed
                            and r["status"] == "ok"
                            and r["utility"][auditor]["informative"] is True
                            and r["utility"][auditor]["retention"] is not None
                        ]
                        require(
                            all(math.isfinite(v) for v in retained),
                            "nonfinite matched retention",
                        )
                        if len(retained) == 3:
                            values.append(median(retained))
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "configuration": config,
                            "size_multiplier": size,
                            "auditor": auditor,
                            "complete_fit_seeds": len(values),
                            "median_of_fit_medians": (
                                median(values) if len(values) == 2 else None
                            ),
                            "status_counts": dict(Counter(r["status"] for r in group)),
                        }
                    )
    return rows


def build():
    paths = {
        "fit": ROOT / "round.lock.json",
        "raw": ROOT / "validation-v2/round.lock.json",
        "pack": ROOT / "news-packed-v1/round.lock.json",
        "packed": ROOT / "news-packed-validation-v1/round.lock.json",
        "reference": ROOT / "publication-inputs-v1.json",
    }
    locks = {k: read(p) for k, p in paths.items()}
    for key, path in paths.items():
        require(sha256(path) == PINS[key], "frozen research lock changed")
        sealed(locks[key])
    fit = locks["fit"]
    require(
        fit["gpu_fit_compute_ceiling_seconds"] == 14400
        and fit["fit_timeout_seconds"] == 600
        and fit["gpu_vram_mib_cap"] == 16384
        and fit["artifact_cap_bytes"] == 10240
        and fit["research_compute_separate_from_final_cell_parity"]
        and fit["artifact_contract_changes"] is False,
        "research budget changed",
    )
    require(
        sha256(ROOT / "source/round.py") == fit["source_sha256"]
        and sha256(Path(fit["gpu_binary_path"])) == fit["gpu_binary_sha256"]
        and sha256(ROOT / "package/package-manifest.json")
        == fit["package_manifest_sha256"],
        "fit source changed",
    )
    for name, h in read(ROOT / "package/package-manifest.json")["files"].items():
        require(sha256(ROOT / "package" / name) == h, "frozen package changed")
    for key, folder, filename in [
        ("raw", "validation-v2", "evaluate.py"),
        ("pack", "news-packed-v1", "pack.py"),
        ("packed", "news-packed-validation-v1", "evaluate.py"),
    ]:
        require(
            sha256(ROOT / folder / "source" / filename) == locks[key]["source_sha256"],
            "frozen evaluation source changed",
        )
    require(
        locks["raw"]["fit_round_sha256"] == PINS["fit"]
        and locks["packed"]["fit_round_sha256"] == PINS["fit"]
        and locks["pack"]["parent_round_sha256"] == PINS["fit"]
        and locks["packed"]["packing_round_sha256"] == PINS["pack"]
        and locks["packed"]["raw_validation_round_sha256"] == PINS["raw"],
        "research lineage changed",
    )
    require(
        locks["raw"]["metric_source_sha256"]
        == locks["packed"]["metric_source_sha256"]
        == sha256(ROOT / "package/research/benchmark/pilot_metrics.py"),
        "metric source changed",
    )
    workers = {}
    for dataset, worker in fit["workers"].items():
        root = Path(worker["path"])
        require(
            not (root / "test.csv").exists()
            and "evaluator" not in root.resolve().parts,
            "worker includes official tests",
        )
        for name, h in worker["files"].items():
            require(sha256(root / name) == h, "frozen training worker changed")
        workers[dataset] = worker["files"]
    for runtime in fit["host_runtime_locks"].values():
        require(
            sha256(Path(runtime["path"])) == runtime["sha256"],
            "host runtime lock changed",
        )
        for package in read(runtime["path"])["packages"].values():
            for name, h in package["files"].items():
                require(
                    sha256(Path(package["root"]) / name) == h, "host runtime changed"
                )
    attempts, index = fit_attempts(fit)
    raw = validation_cells("validation-v2", locks["raw"], index)
    packed = validation_cells("news-packed-validation-v1", locks["packed"], index)
    eligible = [r for r in raw if r["dataset"] != "News"] + packed
    summary = summarize(eligible)
    ref = locks["reference"]
    baseline_path = Path(ref["matched_baseline_report_path"])
    require(
        sha256(baseline_path) == ref["matched_baseline_report_sha256"],
        "matched reference changed",
    )
    baseline = read(baseline_path)
    sealed(baseline)
    for d in ("Adult", "California"):
        require(
            all(workers[d][n] == h for n, h in baseline["worker_hashes"][d].items()),
            "matched baseline inputs differ",
        )
    baselines = []
    for cell in baseline["cells"]:
        if cell["fit_seed"] not in FIT_SEEDS:
            continue
        if cell["sample_receipt_path"] is not None:
            receipt_path = Path(cell["sample_receipt_path"])
            require(
                sha256(receipt_path) == cell["sample_receipt_sha256"],
                "matched baseline receipt changed",
            )
        else:
            require(
                cell["status"] == "fit_unavailable", "matched sample receipt missing"
            )
            fit_row = next(
                r
                for r in baseline["fit_attempts"]
                if r["fit_receipt_sha256"] == cell["fit_receipt_sha256"]
            )
            require(
                sha256(Path(fit_row["fit_receipt_path"])) == cell["fit_receipt_sha256"]
                and fit_row["status"] != "ok",
                "matched failed fit changed",
            )
        if cell["status"] == "ok":
            require(
                cell["metric_replay_status"]
                in ("exact", "prior_immutable_receipt_verified"),
                "matched baseline metric replay changed",
            )
        baselines.append(cell)
    base_summary = baseline_summary(baselines)
    packing_completion = ROOT / "news-packed-v1/completion.json"
    completion = read(packing_completion)
    sealed(completion)
    require(
        completion["round_sha256"] == PINS["pack"]
        and completion["artifacts"] == 8
        and completion["exact_parity_samples"] == 24
        and locks["packed"]["packing_completion_sha256"] == sha256(packing_completion),
        "packing completion changed",
    )
    fixture_path = Path(fit["gpu_fixture_receipt"]["path"])
    fixture = read(fixture_path)
    require(
        sha256(fixture_path) == fit["gpu_fixture_receipt"]["sha256"]
        and fixture["status"] == "ok"
        and fixture["generated_gpu_fit_calls"] == 2,
        "GPU generated preflight changed",
    )
    preflights = []
    for item in fit["preflight_failed_attempts"]:
        p = Path(item["path"])
        require(sha256(p) == item["sha256"], "failed GPU preflight changed")
        preflights.append(reference(p))
    unlaunched = ROOT / "validation-v1/unlaunched-source-repair.json"
    require(
        read(unlaunched)["new_fits_started"] == 0,
        "unlaunched validation repair changed",
    )
    return {
        "format": "dope-pilot24-bounded-target-gpu-validation",
        "version": 1,
        "scope": "training_derived_validation_only",
        "source_sha256": sha256(Path(__file__)),
        "receipt_checker_source_sha256": sha256(HERE / "target_research_receipts.py"),
        "locks": {k: reference(v) for k, v in paths.items()},
        "metric_source_sha256": locks["raw"]["metric_source_sha256"],
        "source_patch_sha256": fit["source_patch_sha256"],
        "gpu_binary_sha256": fit["gpu_binary_sha256"],
        "worker_hashes": workers,
        "profiles": list(PROFILES),
        "fit_seeds": list(FIT_SEEDS),
        "sample_seeds": list(SAMPLE_SEEDS),
        "size_multipliers": list(SIZES),
        "fit_attempts": attempts,
        "raw_cells": raw,
        "packed_cells": packed,
        "eligible_cells": eligible,
        "summaries": summary,
        "matched_baseline_report": reference(baseline_path),
        "matched_baseline_cells": baselines,
        "matched_native_selections": baseline["native_selections"],
        "matched_baseline_summaries": base_summary,
        "status_counts": {
            "raw_fits": dict(Counter(r["raw_status"] for r in attempts)),
            "raw_validation": dict(Counter(r["status"] for r in raw)),
            "packed_validation": dict(Counter(r["status"] for r in packed)),
            "matched_baselines": dict(Counter(r["status"] for r in baselines)),
        },
        "cost": {
            "new_gpu_fit_seconds": sum(r["elapsed_gpu_fit_seconds"] for r in attempts),
            "dispatch_process_seconds_including_preparation": sum(
                r["dispatch_elapsed_seconds"] for r in attempts
            ),
            "new_gpu_fit_ceiling_seconds": 14400,
            "maximum_gpu_used_mib": max(r["peak_gpu_used_mib"] for r in attempts),
            "device_energy_estimate_joules_including_baseline": sum(
                r["energy_joules_estimate"] or 0 for r in attempts
            ),
            "sampling_metric_operation_seconds": sum(
                op["elapsed_seconds"]
                for cell in raw + packed
                for op in cell["operations"]
            ),
            "lossless_packing_wall_seconds": completion["wall_seconds"],
            "research_cost_separate_from_final_per_cell_parity": True,
            "complete_total_historical_rd_cost": False,
            "aggregate_resource_admission_complete": False,
        },
        "preflight": {
            "failed_attempts": preflights,
            "passed_generated_gpu_fixture": reference(fixture_path),
            "unlaunched_validation_repair": reference(unlaunched),
            "packing_completion": reference(packing_completion),
        },
        "notes": [
            "Four bounded research profiles, two fit seeds and n/4n validation; no final global family or production configuration selected.",
            "All eight original News raw projection byte failures remain visible. Packed artifacts preserve exact learned bytes and reconstruct the full projection; byte eligibility does not establish release safety.",
            "Adult/California references use the same workers, fit seeds11/23 and three sample seeds. CTGAN/TVAE author-default/native-selected configurations are already frozen; no shared KPI enters comparator selection. News has no matched two-fit neural reference in this panel.",
            "The reference includes complete five-fit results; this panel extracts the two matching fit seeds and does not replace their five-fit estimates. Failed executable cells contribute no DOPE win.",
            "GPU host runtime pins cover Torch/NumPy code and shared objects, not a complete final environment closure. Metric sources/versions and dependency files are frozen separately.",
            "New GPU research cells have exact sample and metric replays. Prior CTGAN/TVAE fit-23 metrics retain their prior_immutable_receipt_verified label; no new exact replay is claimed for those reference metrics. Copy/near screens and real-vs-real controls are retained. Complete privacy attacks, projection-only utility cost and public-core/product coverage remain missing.",
            "Earlier architecture research and baseline native tuning costs remain reported in their separate immutable panels; total historical R&D cost is incomplete. No equal-total-R&D-spend or paired superiority claim.",
        ],
        "citation_keys": ["xu2019modeling"],
        "official_tests_opened": False,
        "mfs_v2": None,
        "ptf_v1": None,
        "production_certified": False,
    }


def number(value):
    return "null" if value is None else f"{value:.4f}"


def render(report):
    sealed(report)
    require(report["production_certified"] is False, "research certification changed")
    require(
        len(report["raw_cells"]) == 144
        and len(report["packed_cells"]) == 48
        and report["eligible_cells"]
        == [r for r in report["raw_cells"] if r["dataset"] != "News"]
        + report["packed_cells"],
        "research publication matrix changed",
    )
    require(
        report["summaries"] == summarize(report["eligible_cells"])
        and report["matched_baseline_summaries"]
        == baseline_summary(report["matched_baseline_cells"]),
        "research summaries changed",
    )
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "dataset",
            "method",
            "configuration",
            "auditor",
            "size_multiplier",
            "complete_fit_seeds",
            "median_of_fit_medians",
        ]
    )
    for row in report["summaries"]:
        writer.writerow(
            [
                row["dataset"],
                "DOPE",
                row["profile"],
                row["auditor"],
                row["size_multiplier"],
                row["complete_fit_seeds"],
                number(row["median_of_fit_medians"]),
            ]
        )
    for row in report["matched_baseline_summaries"]:
        writer.writerow(
            [
                row["dataset"],
                row["method"],
                row["configuration"],
                row["auditor"],
                row["size_multiplier"],
                row["complete_fit_seeds"],
                number(row["median_of_fit_medians"]),
            ]
        )
    lines = [
        "# Bounded DOPE GPU target research",
        "",
        "Training-derived validation only. Four fixed profiles, fit seeds11/23, sample seeds101/211/307, three shared auditors at n/4n. Official tests remain sealed; MFS-v2/PTF-v1 and release safety are null.",
        "",
        "## DOPE profiles",
        "",
        "| Dataset | Profile | CatBoost n | CatBoost 4n | Charged bytes |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for profile in PROFILES:
            values = {
                r["size_multiplier"]: r["median_of_fit_medians"]
                for r in report["summaries"]
                if (r["dataset"], r["profile"], r["auditor"])
                == (dataset, profile, "catboost")
            }
            charges = [
                r["artifact_bytes"]
                for r in report["fit_attempts"]
                if (r["dataset"], r["profile"]) == (dataset, profile)
            ]
            lines.append(
                f"| {dataset} | {profile} | {number(values[1])} | {number(values[4])} | {min(charges)}–{max(charges)} |"
            )
    lines += [
        "",
        "## Matched author-native/default references",
        "",
        "These references use the identical Adult/California workers and matching fit seeds11/23. Their configs retain the author-native selections from the complete five-fit panel; the shared KPI did not tune them. This extraction does not replace the original five-fit analysis.",
        "",
        "| Dataset | Method | Configuration | CatBoost n | CatBoost 4n |",
        "|---|---|---|---:|---:|",
    ]
    groups = sorted(
        {
            (r["dataset"], r["method"], r["configuration"])
            for r in report["matched_baseline_summaries"]
        }
    )
    for dataset, method, config in groups:
        values = {
            r["size_multiplier"]: r["median_of_fit_medians"]
            for r in report["matched_baseline_summaries"]
            if (r["dataset"], r["method"], r["configuration"], r["auditor"])
            == (dataset, method, config, "catboost")
        }
        lines.append(
            f"| {dataset} | {method} | {config} | {number(values[1])} | {number(values[4])} |"
        )
    cost = report["cost"]
    lines += [
        "",
        "## Accounting",
        "",
        f'All24GPUfits accounted: {json.dumps(report["status_counts"]["raw_fits"],sort_keys=True)}. GPU fit elapsed {cost["new_gpu_fit_seconds"]:.3f}s; dispatch process time including preparation {cost["dispatch_process_seconds_including_preparation"]:.3f}s. Peak device use {cost["maximum_gpu_used_mib"]}MiB. Architecture research cost is separate from final per-cell parity; historical R&D cost is incomplete.',
        "",
        f'Raw validation identities: {json.dumps(report["status_counts"]["raw_validation"],sort_keys=True)}. Separate packed News identities: {json.dumps(report["status_counts"]["packed_validation"],sort_keys=True)}. All eight original raw byte failures are retained; lossless packing changes no learned bytes or samples.',
        "",
        "Generated fixture launch failure and unlaunched validation-source repair are preserved. All cells retain immutable receipt/source hashes, costs, three-auditor losses, copy/near and real-vs-real controls. The GPU research used per-job free-RAM admission; complete aggregate reservation enforcement was introduced later and is not established for this round.",
        "",
    ]
    lines += [note + "\n" for note in report["notes"]]
    lines += [
        "Native baseline author reference: `xu2019modeling` in the corrected local bibliography.",
        "",
    ]
    return stream.getvalue(), "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(args.result) if args.from_json else build()
    sealed(report)
    require(
        report["source_sha256"] == sha256(Path(__file__))
        and report["receipt_checker_source_sha256"]
        == sha256(HERE / "target_research_receipts.py"),
        "publication source changed",
    )
    from jsonschema import Draft202012Validator

    schema = read(args.result.with_suffix(".schema.json"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    csv_text, markdown = render(report)
    if not args.from_json:
        args.result.write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    args.result.with_suffix(".csv").write_text(csv_text)
    args.result.with_suffix(".md").write_text(markdown)
    print(
        json.dumps(
            {
                "raw_cells": len(report["raw_cells"]),
                "packed_cells": len(report["packed_cells"]),
                "status_counts": report["status_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
