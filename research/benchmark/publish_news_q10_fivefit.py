"""Publish the five-fit-seed News q10 validation replication and costing."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import median

from . import publish_news_q10_packed
from .manifest import digest
from .score import sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
SNAP = ROOT.parent / "source-snapshots"
FIRST_ROUND = ROOT / "dope-news-q10-v1"
PACK11 = ROOT / "dope-news-q10-seed11-packed-v1"
PACK23 = ROOT / "dope-news-q10-packed-v2"
NEW_SAMPLES = ROOT / "dope-news-q10-multiseed-samples-v1"
NEW_METRICS = ROOT / "dope-news-q10-multiseed-validation-v1"
FIVE = ROOT / "dope-news-q10-fivefit-validation-v1"
RESULT = HERE / "results/pilot24-news-q10-fivefit.json"
PRIOR = HERE / "results/pilot24-news-q10-packed.json"
CAP = 10_240
FITS = (11, 23, 37, 53, 71)
SEEDS = (101, 211, 307)
SIZES = (1, 4)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def metric_payload(metric: dict) -> dict:
    return {key: value for key, value in metric.items()
            if key != "metric_seconds"}


def validated_inputs() -> tuple[dict, dict, dict]:
    prior = read(PRIOR)
    five_lock_path = FIVE / "round.lock.json"
    five_lock = read(five_lock_path)
    five_manifest_path = FIVE / "manifest.json"
    five_manifest = read(five_manifest_path)
    new_lock_path = NEW_METRICS / "round.lock.json"
    new_lock = read(new_lock_path)
    new_manifest_path = NEW_METRICS / "manifest.json"
    new_manifest = read(new_manifest_path)
    if (prior != publish_news_q10_packed.build()
            or prior["official_tests_opened"] is not False
            or prior["mfs_v2"] is not None or prior["ptf_v1"] is not None
            or five_lock["format"]
            != "dope-news-q10-fivefit-validation-replay-round"
            or five_lock["source_sha256"] != sha256(
                SNAP / "dope_news_q10_fivefit_reconcile_v1.py")
            or five_lock["fit_seeds"] != list(FITS)
            or five_lock["sample_seeds"] != list(SEEDS)
            or five_lock["size_multipliers"] != list(SIZES)
            or five_manifest["round_lock_sha256"] != sha256(five_lock_path)
            or five_manifest["exact_old_cells"] != 12
            or five_manifest["exact_new_cells"] != 18
            or five_manifest["total_exact_cells"] != 30
            or five_manifest["new_validation_manifest_sha256"]
            != sha256(new_manifest_path)
            or new_lock["format"]
            != "dope-news-q10-multiseed-validation-round"
            or new_lock["source_sha256"] != sha256(
                SNAP / "dope_news_q10_multiseed_evaluate_v1.py")
            or new_manifest["round_lock_sha256"] != sha256(new_lock_path)
            or new_manifest["exact_cells"] != 18
            or new_manifest["official_tests_opened"] is not False
            or five_manifest["official_tests_opened"] is not False
            or five_manifest["mfs_v2"] is not None
            or five_manifest["ptf_v1"] is not None):
        raise ValueError("News q10 five-fit validation lineage changed")
    return five_lock, five_manifest, new_lock


def packed_sample_checks() -> tuple[dict, dict]:
    old_lock = read(FIRST_ROUND / "sample.lock.json")
    pack11_lock_path = PACK11 / "round.lock.json"
    pack11_lock = read(pack11_lock_path)
    pack23_lock_path = PACK23 / "round.lock.json"
    pack23_lock = read(pack23_lock_path)
    pack23_receipt = read(PACK23 / "receipt.json")
    new_lock_path = NEW_SAMPLES / "round.lock.json"
    new_lock = read(new_lock_path)
    if (len(old_lock["jobs"]) != 24
            or len(pack11_lock["jobs"]) != 12
            or pack11_lock["parent_sample_round_sha256"]
            != sha256(FIRST_ROUND / "sample.lock.json")
            or pack11_lock["source_sha256"] != sha256(
                SNAP / "dope_news_q10_seed11_pack_v1.py")
            or len(pack23_lock["sample_cells"]) != 12
            or pack23_receipt["round_lock_sha256"] != sha256(pack23_lock_path)
            or pack23_receipt["sample_checks_exact"] != 12
            or len(new_lock["jobs"]) != 36
            or new_lock["source_sha256"] != sha256(
                SNAP / "dope_news_q10_multiseed_sample_v1.py")):
        raise ValueError("News q10 packed sample matrix changed")
    packed = {}
    for job in pack11_lock["jobs"]:
        path = PACK11 / "jobs" / digest(job) / "attempt-0001/receipt.json"
        row = read(path)
        key = (11, job["sample_seed"], job["size_multiplier"])
        if (row["job"] != job
                or row["round_lock_sha256"] != sha256(pack11_lock_path)
                or row["status"] != "exact"
                or row["sample_sha256"] != job["original_sample_sha256"]
                or row["artifact_bytes"] != job["charged_artifact_bytes"]
                or row["artifact_bytes"] > CAP or key in packed):
            raise ValueError("News seed-11 packed sample check changed")
        packed[key] = {"check_sha256": sha256(path),
                       "sample_sha256": row["sample_sha256"],
                       "artifact_bytes": row["artifact_bytes"]}
    for cell in pack23_lock["sample_cells"]:
        job = cell["job"]
        path = PACK23 / "sample-checks" / f"{digest(cell)}.json"
        row = read(path)
        key = (23, job["sample_seed"], job["size_multiplier"])
        if (row["cell"] != cell
                or row["round_sha256"] != sha256(pack23_lock_path)
                or row["status"] != "exact"
                or row["sample_sha256"] != cell["sample_sha256"]
                or pack23_receipt["artifact_bytes"] > CAP or key in packed):
            raise ValueError("News seed-23 packed sample check changed")
        packed[key] = {"check_sha256": sha256(path),
                       "sample_sha256": row["sample_sha256"],
                       "artifact_bytes": pack23_receipt["artifact_bytes"]}
    for job in new_lock["jobs"]:
        path = NEW_SAMPLES / "jobs" / digest(job) / "attempt-0001/receipt.json"
        row = read(path)
        key = (job["fit_seed"], job["sample_seed"], job["size_multiplier"])
        if (row["job"] != job
                or row["round_lock_sha256"] != sha256(new_lock_path)
                or row["status"] != "ok"
                or row["sample_sha256"] != row["repeat_sha256"]
                or row["artifact_bytes"] != job["charged_artifact_bytes"]
                or row["artifact_bytes"] > CAP or key in packed):
            raise ValueError("News replicated packed sample changed")
        packed[key] = {"check_sha256": sha256(path),
                       "sample_sha256": row["sample_sha256"],
                       "artifact_bytes": row["artifact_bytes"]}
    if (len(packed) != 60
            or set(packed) != {(fit, seed, size) for fit in FITS
                               for seed in SEEDS for size in (1, 2, 4, 8)}):
        raise ValueError("News five-fit packed sample grid incomplete")
    return packed, {"old_sample_round": sha256(FIRST_ROUND / "sample.lock.json"),
                    "seed11_packed_round": sha256(pack11_lock_path),
                    "seed23_packed_round": sha256(pack23_lock_path),
                    "new_sample_round": sha256(new_lock_path),
                    "seed11_packing_source": sha256(
                        SNAP / "dope_news_q10_seed11_pack_v1.py"),
                    "new_sampling_source": sha256(
                        SNAP / "dope_news_q10_multiseed_sample_v1.py")}


def build() -> dict:
    five_lock, five_manifest, new_lock = validated_inputs()
    packed, sample_locks = packed_sample_checks()
    old_checks = {row["cell_digest"]: row for row in five_manifest["cells"]}
    new_manifest = read(NEW_METRICS / "manifest.json")
    new_checks = {row["cell_digest"]: row for row in new_manifest["cells"]}
    if len(old_checks) != 12 or len(new_checks) != 18:
        raise ValueError("News five-fit exact metric check matrix incomplete")
    cells = []
    fits = {}
    for cell in five_lock["old_cells"]:
        job = cell["job"]
        fit_seed, seed, size = (job["fit_seed"], job["sample_seed"],
                                job["size_multiplier"])
        path = Path(cell["original_receipt_path"])
        receipt = read(path)
        replay_path = FIVE / "replay" / f"{digest(cell)}.json"
        replay = read(replay_path)
        check = old_checks[digest(cell)]
        identity = packed[(fit_seed, seed, size)]
        if (sha256(path) != cell["original_receipt_sha256"]
                or receipt["job"] != job
                or receipt["sample_sha256"] != identity["sample_sha256"]
                or sha256(Path(cell["packed_check_path"]))
                != cell["packed_check_sha256"]
                or identity["check_sha256"] != cell["packed_check_sha256"]
                or identity["artifact_bytes"] != cell["packed_artifact_bytes"]
                or check["replay_receipt_sha256"] != sha256(replay_path)
                or replay["cell"] != cell or replay["status"] != "exact"
                or replay["round_lock_sha256"] != sha256(FIVE / "round.lock.json")
                or replay["original_receipt_sha256"] != sha256(path)
                or replay["metric_payload"] != metric_payload(receipt["metric"])):
            raise ValueError("News prior-round exact metric replay changed")
        cells.append((job, receipt["metric"], sha256(path),
                      sha256(replay_path), identity))
        fit_path = Path(job["fit_receipt_path"])
        if sha256(fit_path) != job["fit_receipt_sha256"]:
            raise ValueError("News prior-round GPU fit receipt changed")
        fits[fit_seed] = (fit_path, read(fit_path))
    for cell in new_lock["cells"]:
        job = cell["job"]
        fit_seed, seed, size = (job["fit_seed"], job["sample_seed"],
                                job["size_multiplier"])
        path = NEW_METRICS / "metrics" / f"{digest(cell)}.json"
        metric = read(path)
        replay_path = NEW_METRICS / "replay" / f"{digest(cell)}.json"
        replay = read(replay_path)
        check = new_checks[digest(cell)]
        identity = packed[(fit_seed, seed, size)]
        if (metric["cell"] != cell
                or metric["round_lock_sha256"]
                != sha256(NEW_METRICS / "round.lock.json")
                or metric["metric_source_sha256"]
                != new_lock["metric_source_sha256"]
                or cell["sample_sha256"] != identity["sample_sha256"]
                or cell["sample_receipt_sha256"] != identity["check_sha256"]
                or check["metric_receipt_sha256"] != sha256(path)
                or check["replay_receipt_sha256"] != sha256(replay_path)
                or replay["cell"] != cell or replay["status"] != "exact"
                or replay["metric_receipt_sha256"] != sha256(path)
                or replay["metric_payload"] != metric_payload(metric["metrics"])):
            raise ValueError("News new exact metric replay changed")
        cells.append((job, metric["metrics"], sha256(path),
                      sha256(replay_path), identity))
        fit_path = Path(job["fit_receipt_path"])
        if sha256(fit_path) != job["fit_receipt_sha256"]:
            raise ValueError("News replicated GPU fit receipt changed")
        fits[fit_seed] = (fit_path, read(fit_path))
    if len(cells) != 30 or set(fits) != set(FITS):
        raise ValueError("News five-fit metric cells incomplete")
    fit_cost = []
    common_identity = None
    for seed in FITS:
        path, fit = fits[seed]
        packed_bytes = packed[(seed, 101, 1)]["artifact_bytes"]
        identity = fit["identity"]
        comparable = {key: identity[key] for key in
                      ("binary_sha256", "candidate", "dataset",
                       "fit_deadline_seconds", "probe_source_sha256",
                       "projection_sha256", "split", "structural_penalty",
                       "target_weight", "tier", "train_sha256",
                       "validation_sha256")}
        if common_identity is None:
            common_identity = comparable
        if (fit["status"] != "ok"
                or identity["seed"] != seed
                or identity["fit_deadline_seconds"] != 600
                or identity["candidate"]
                != "symbolic_autoregressive_residual"
                or identity["target_weight"] != 2.0
                or identity["structural_penalty"] != 0.0
                or identity["tier"] != "l3"
                or comparable != common_identity
                or fit["artifact_sha256"] is None
                or fit["artifact_bytes"] <= CAP
                or fit["elapsed_seconds"] > 600
                or packed_bytes > CAP):
            raise ValueError("News GPU fit cost or artifact changed")
        fit_cost.append({"fit_seed": seed,
                         "fit_receipt_sha256": sha256(path),
                         "raw_artifact_bytes": fit["artifact_bytes"],
                         "packed_artifact_bytes": packed_bytes,
                         "gpu_host": fit["host"],
                         "gpu_elapsed_seconds": fit["elapsed_seconds"],
                         "peak_gpu_used_mib": fit["peak_gpu_used_mib"],
                         "energy_joules_estimate":
                             fit["energy_joules_estimate"]})
    public_cells = []
    for job, metric, metric_hash, replay_hash, identity in cells:
        auditor = metric["utility"]
        if (metric["implementation_sha256"]
            != five_lock["metric_source_sha256"]
                or metric["mfs_v2"] is not None
                or metric["gate_profile_complete"] is not False
                or auditor["catboost"]["retention"] is None):
            raise ValueError("News five-fit metric value changed")
        public_cells.append({"fit_seed": job["fit_seed"],
                             "sample_seed": job["sample_seed"],
                             "size_multiplier": job["size_multiplier"],
                             "packed_artifact_bytes":
                                 identity["artifact_bytes"],
                             "within_l3_bytes": True,
                             "catboost_retention":
                                 auditor["catboost"]["retention"],
                             "linear_retention": auditor["linear"]["retention"],
                             "mlp_retention": auditor["mlp"]["retention"],
                             "mlp_informative": auditor["mlp"]["informative"],
                             "mlp_low_signal_noninferior":
                                 auditor["mlp"]["low_signal_noninferior"],
                             "marginal_ks_mean": metric["marginal_ks_mean"],
                             "pair_correlation_fidelity":
                                 metric["pair_correlation_fidelity"],
                             "c2st_auc": metric["c2st_auc"],
                             "exact_copies": metric["copy_counts"]["exact"],
                             "near_copies": metric["copy_counts"]["near"],
                             "real_vs_real_control_counts":
                                 metric["real_vs_real_control_counts"],
                             "release_safe": None,
                             "packed_sample_check_sha256":
                                 identity["check_sha256"],
                             "metric_receipt_sha256": metric_hash,
                             "metric_replay_sha256": replay_hash})
    expected = {(fit, seed, size) for fit in FITS for seed in SEEDS
                for size in SIZES}
    if {(c["fit_seed"], c["sample_seed"], c["size_multiplier"])
        for c in public_cells} != expected:
        raise ValueError("News five-fit public metric matrix incomplete")
    public_cells.sort(key=lambda c: (c["fit_seed"], c["size_multiplier"],
                                     c["sample_seed"]))
    fit_summary = []
    for fit in FITS:
        for size in SIZES:
            rows = [c for c in public_cells if c["fit_seed"] == fit
                    and c["size_multiplier"] == size]
            if len(rows) != 3:
                raise ValueError("News fit summary incomplete")
            fit_summary.append({"fit_seed": fit, "size_multiplier": size,
                                "sample_seeds": 3,
                                "median_catboost_retention": median(
                                    c["catboost_retention"] for c in rows),
                                "min_catboost_retention": min(
                                    c["catboost_retention"] for c in rows),
                                "max_catboost_retention": max(
                                    c["catboost_retention"] for c in rows),
                                "packed_artifact_bytes": rows[0][
                                    "packed_artifact_bytes"],
                                "cells_with_exact_copies": sum(
                                    c["exact_copies"] > 0 for c in rows),
                                "cells_with_near_copies": sum(
                                    c["near_copies"] > 0 for c in rows)})
    summary = []
    for size in SIZES:
        rows = [r for r in fit_summary if r["size_multiplier"] == size]
        summary.append({"size_multiplier": size, "fit_seeds": 5,
                        "sample_seeds_per_fit": 3,
                        "median_of_fit_medians_catboost_retention": median(
                            r["median_catboost_retention"] for r in rows),
                        "min_fit_median_catboost_retention": min(
                            r["median_catboost_retention"] for r in rows),
                        "max_fit_median_catboost_retention": max(
                            r["median_catboost_retention"] for r in rows),
                        "packed_artifacts_within_l3": sum(
                            r["packed_artifact_bytes"] <= CAP for r in rows)})
    return {"format": "dope-pilot24-news-q10-fivefit-validation",
            "version": 1, "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "l3_byte_cap": CAP,
            "configuration": common_identity,
            "locks": {**sample_locks,
                      "old_metric_replay_round": sha256(FIVE / "round.lock.json"),
                      "fivefit_replay_source": sha256(
                          SNAP / "dope_news_q10_fivefit_reconcile_v1.py"),
                      "old_new_exact_manifest": sha256(FIVE / "manifest.json"),
                      "new_validation_round": sha256(NEW_METRICS / "round.lock.json"),
                      "new_metric_source": sha256(
                          SNAP / "dope_news_q10_multiseed_evaluate_v1.py"),
                      "new_validation_manifest": sha256(
                          NEW_METRICS / "manifest.json"),
                      "prior_packed_seed23_report": sha256(PRIOR),
                      "first_packing_failure": sha256(
                          ROOT / "dope-news-q10-packed-v1/failure-receipt.json")},
            "packed_sample_checks_exact": len(packed),
            "common_metric_replays_exact": len(public_cells),
            "fit_cost": fit_cost, "cells": public_cells,
            "fit_summary": fit_summary, "summary": summary,
            "notes": ["Five GPU fit seeds and three sample seeds at n and 4n give 30 validation-only common metric cells. All five packed artifacts are under 10,240 bytes and 60 n/2n/4n/8n sample identities replayed exactly.",
                      "The q10 architecture and configuration came from earlier validation-only research; its discovery compute is reported separately. This round replicates one fixed configuration across fit seeds.",
                      "One high-retention fit seed is not a production pass. The report gives the median across five fit-seed medians and the full seed spread; MLP retention is null when its real auditor is noninformative.",
                      "The earlier seed-23 CTGAN/TVAE native-tuned comparison used one fit seed. Its paired result is separate from this five-fit DOPE stability panel.",
                      "Lossless projection packing changes artifact charge without changing generated sample bytes. The first failed seed-23 packing receipt remains pinned.",
                      "Official tests remain sealed; privacy attacks and all release gates are incomplete, so MFS-v2, PTF-v1, production certification, and paper superiority claims are unavailable."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def render(report: dict) -> tuple[str, str]:
    table = io.StringIO()
    fields = ("fit_seed", "size_multiplier", "sample_seeds",
              "median_catboost_retention", "min_catboost_retention",
              "max_catboost_retention", "packed_artifact_bytes",
              "cells_with_exact_copies", "cells_with_near_copies")
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["fit_summary"])
    lines = ["# News q10 DOPE: five GPU fit seeds on pilot validation", "",
             "The q10 generator was trained independently at fit seeds "
             "11, 23, 37, 53, and 71. Lossless packed artifacts each fit "
             "within 10,240 bytes. All 60 n/2n/4n/8n sample identities "
             "matched their deterministic reference hashes, and all 30 "
             "n/4n validation metric payloads replayed exactly.", "",
             "| Fit seed | n median CatBoost retention | 4n median | Packed bytes |",
             "|---:|---:|---:|---:|"]
    for fit in FITS:
        one = next(x for x in report["fit_summary"] if x["fit_seed"] == fit
                   and x["size_multiplier"] == 1)
        four = next(x for x in report["fit_summary"] if x["fit_seed"] == fit
                    and x["size_multiplier"] == 4)
        lines.append(f"| {fit} | {one['median_catboost_retention']:.4f} | "
                     f"{four['median_catboost_retention']:.4f} | "
                     f"{one['packed_artifact_bytes']:,} |")
    one, four = report["summary"]
    lines += ["", "Median across the five fit-seed medians: "
              f"{one['median_of_fit_medians_catboost_retention']:.4f} at n and "
              f"{four['median_of_fit_medians_catboost_retention']:.4f} at 4n. "
              "This is training-derived validation evidence. The one-fit-seed "
              "CTGAN/TVAE comparison is a separate paired panel. Official "
              "tests remain sealed; MFS-v2, PTF-v1 and production "
              "certification are null.", ""]
    return table.getvalue(), "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-news-q10-fivefit-validation"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("News q10 five-fit report changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    csv_text, markdown = render(report)
    RESULT.with_suffix(".csv").write_text(csv_text)
    RESULT.with_suffix(".md").write_text(markdown)
    print(json.dumps({"fit_seeds": len(report["fit_cost"]),
                      "metric_cells": len(report["cells"]),
                      "sample_checks": report["packed_sample_checks_exact"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
