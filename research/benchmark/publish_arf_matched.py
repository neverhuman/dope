"""Publish ARF and DOPE paired pilot validation with explicit unavailable cells."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import median

from . import publish_arf_combined, publish_synthpop_matched
from .manifest import digest
from .score import sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
RUN = ROOT / "arf-final-validation-v1"
SAMPLES = ROOT / "arf-final-samples-v1"
RESULT = HERE / "results/pilot24-arf-matched.json"
NATIVE = HERE / "results/arf-combined-native-validation.json"
DATASETS = ("Adult", "California", "News")
SEEDS = (101, 211, 307)
SIZES = (1, 4)
CAP = 10_240


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def native_cells() -> dict[str, dict]:
    report = read(NATIVE)
    if (report != publish_arf_combined.build()
            or report["aggregate"]["trials"] != 24
            or report["aggregate"]["timeout_trials"] != 8
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None
            or report["ptf_v1"] is not None):
        raise ValueError("ARF eight-trial native report changed")
    cells = {row["dataset"]: row for row in report["cells"]}
    if set(cells) != set(DATASETS):
        raise ValueError("ARF native datasets incomplete")
    return cells


def build() -> dict:
    lock_path = RUN / "round.lock.json"
    manifest_path = RUN / "manifest.json"
    sample_lock_path = SAMPLES / "round.lock.json"
    lock, manifest, sample_lock = (read(path) for path in
                                    (lock_path, manifest_path,
                                     sample_lock_path))
    if (lock["format"] != "dope-arf-pilot-fixed-seed-validation-round"
            or lock["sample_round_sha256"] != sha256(sample_lock_path)
            or len(sample_lock["jobs"]) != 48
            or len(lock["cells"]) != 23
            or len(lock["failed_sample_cells"]) != 1
            or len(lock["unavailable_default_cells"]) != 2
            or manifest["exact_cells"] != 23
            or manifest["round_lock_sha256"] != sha256(lock_path)
            or manifest["failed_sample_cells"]
            != lock["failed_sample_cells"]
            or manifest["unavailable_default_cells"]
            != lock["unavailable_default_cells"]
            or manifest["official_tests_opened"] is not False
            or lock["official_tests_opened"] is not False
            or lock["mfs_v2"] is not None or lock["ptf_v1"] is not None):
        raise ValueError("ARF matched validation matrix changed")
    replayed = {row["cell_digest"]: row for row in manifest["cells"]}
    if len(replayed) != 23:
        raise ValueError("ARF exact replay matrix incomplete")
    native = native_cells()
    dope = publish_synthpop_matched.dope_reference()
    cells = []
    for cell in lock["cells"]:
        job = cell["job"]
        dataset, kind, seed, size = (job["dataset"], job["kind"],
                                     job["sample_seed"],
                                     job["size_multiplier"])
        fit_path = Path(job["fit_receipt_path"])
        fit = read(fit_path)
        code = digest(cell)
        metric_path = RUN / "metrics" / f"{code}.json"
        replay_path = RUN / "replay" / f"{code}.json"
        metric, replay = read(metric_path), read(replay_path)
        check = replayed[code]
        choice = native[dataset]
        if (dataset not in DATASETS or kind not in ("default", "tuned")
                or seed not in SEEDS or size not in SIZES
                or job["fit_seed"] != 23
                or sha256(fit_path) != job["fit_receipt_sha256"]
                or fit["status"] != "ok"
                or fit["job"]["dataset"] != dataset
                or fit["job"]["kind"] != kind
                or fit["job"]["fit_seed"] != 23
                or fit["job"]["selection_receipt_sha256"]
                != choice["selection_receipt_sha256"]
                or (kind == "tuned" and
                    (fit["job"]["native_round"], fit["job"]["native_trial"])
                    != (choice["selected_round"], choice["selected_trial"]))
                or (kind == "default" and
                    (fit["job"]["native_round"], fit["job"]["native_trial"])
                    != ("original", 0))
                or fit["artifact_bytes"] != job["artifact_bytes"]
                or fit["artifact_bytes"] <= CAP
                or sha256(Path(cell["sample_receipt_path"]))
                != cell["sample_receipt_sha256"]
                or sha256(Path(cell["sample_receipt_path"]).parent
                          / "sample.csv") != cell["sample_sha256"]
                or metric["cell"] != cell
                or metric["round_lock_sha256"] != sha256(lock_path)
                or metric["metric_source_sha256"]
                != lock["metric_source_sha256"]
                or metric["metrics"]["implementation_sha256"]
                != lock["metric_source_sha256"]
                or metric["official_tests_opened"] is not False
                or metric["mfs_v2"] is not None
                or metric["ptf_v1"] is not None
                or check["metric_receipt_sha256"] != sha256(metric_path)
                or check["replay_receipt_sha256"] != sha256(replay_path)
                or replay["cell"] != cell
                or replay["status"] != "exact"
                or replay["round_lock_sha256"] != sha256(lock_path)
                or replay["metric_receipt_sha256"] != sha256(metric_path)
                or replay["official_tests_opened"] is not False
                or replay["mfs_v2"] is not None
                or replay["ptf_v1"] is not None
                or replay["metric_payload"] != {name: value
                    for name, value in metric["metrics"].items()
                    if name != "metric_seconds"}):
            raise ValueError("ARF fit, metric, or replay changed")
        outcome = metric["metrics"]
        retention = outcome["utility"]["catboost"]["retention"]
        if (retention is None or outcome["mfs_v2"] is not None
                or outcome["gate_profile_complete"] is not False):
            raise ValueError("ARF common validation outcome missing")
        trial = (choice["trials"][0] if kind == "default" else next(
            row for row in choice["trials"]
            if row["round"] == choice["selected_round"]
            and row["trial"] == choice["selected_trial"]))
        if trial["status"] != "ok":
            raise ValueError("ARF available fit lacks native KPI")
        reference = dope[(dataset, seed, size)]
        copy = outcome["copy_counts"]
        cells.append({"dataset": dataset, "configuration": kind,
                      "fit_seed": 23, "sample_seed": seed,
                      "size_multiplier": size,
                      "native_objective": "heldout_forde_mean_log_density",
                      "native_validation_kpi": trial["native_kpi"],
                      "arf_artifact_bytes": fit["artifact_bytes"],
                      "arf_within_l3_bytes": False,
                      "arf_retention": retention,
                      "arf_exact_copies": copy["exact"],
                      "arf_near_copies": copy["near"],
                      "arf_copy_gate_pass":
                          copy["exact"] == 0 and copy["near"] == 0,
                      "arf_release_safe": False,
                      "dope_configuration": reference["configuration"],
                      "dope_artifact_bytes": reference["artifact_bytes"],
                      "dope_retention": reference["retention"],
                      "dope_release_safe": None,
                      "difference_dope_minus_arf":
                          reference["retention"] - retention,
                      "fit_receipt_sha256": sha256(fit_path),
                      "sample_receipt_sha256": cell[
                          "sample_receipt_sha256"],
                      "metric_receipt_sha256": sha256(metric_path),
                      "replay_receipt_sha256": sha256(replay_path),
                      "dope_metric_receipt_sha256":
                          reference["metric_receipt_sha256"]})
    expected = {(dataset, kind, seed, size)
                for dataset, kind in (("Adult", "tuned"),
                                      ("California", "default"),
                                      ("California", "tuned"),
                                      ("News", "tuned"))
                for seed in SEEDS for size in SIZES}
    failed = []
    for row in lock["failed_sample_cells"]:
        job = row["job"]
        key = (job["dataset"], job["kind"], job["sample_seed"],
               job["size_multiplier"])
        if (key not in expected or row["status"] != "timeout"
                or sha256(Path(row["sample_receipt_path"]))
                != row["sample_receipt_sha256"]):
            raise ValueError("ARF timeout receipt changed")
        reference = dope[(job["dataset"], job["sample_seed"],
                          job["size_multiplier"])]
        failed.append({"dataset": job["dataset"],
                       "configuration": job["kind"],
                       "fit_seed": 23,
                       "sample_seed": job["sample_seed"],
                       "size_multiplier": job["size_multiplier"],
                       "status": row["status"],
                       "sample_receipt_sha256":
                           row["sample_receipt_sha256"],
                       "arf_retention": None,
                       "dope_retention": reference["retention"],
                       "contributes_dope_win": False})
    if ({(row["dataset"], row["configuration"], row["sample_seed"],
          row["size_multiplier"]) for row in cells + failed} != expected
            or len(cells) != 23 or len(failed) != 1):
        raise ValueError("ARF complete 24-cell n/4n grid changed")
    unavailable = []
    for row in lock["unavailable_default_cells"]:
        choice = native[row["dataset"]]
        if (row["dataset"] not in ("Adult", "News")
                or row["status"] != "timeout"
                or choice["default_status"] != "timeout"
                or row["selection_receipt_sha256"]
                != choice["selection_receipt_sha256"]):
            raise ValueError("ARF original default availability changed")
        unavailable.append({"dataset": row["dataset"],
                            "configuration": "default",
                            "status": "native_fit_timeout",
                            "selection_receipt_sha256":
                                row["selection_receipt_sha256"],
                            "contributes_dope_win": False})
    if len(unavailable) != 2:
        raise ValueError("ARF unavailable defaults incomplete")
    cells.sort(key=lambda row: (row["dataset"], row["configuration"],
                                row["size_multiplier"], row["sample_seed"]))
    summary = []
    for dataset, kind in (("Adult", "tuned"), ("California", "default"),
                          ("California", "tuned"), ("News", "tuned")):
        for size in SIZES:
            rows = [row for row in cells if row["dataset"] == dataset
                    and row["configuration"] == kind
                    and row["size_multiplier"] == size]
            missing = [row for row in failed if row["dataset"] == dataset
                       and row["configuration"] == kind
                       and row["size_multiplier"] == size]
            if len(rows) + len(missing) != 3 or not rows:
                raise ValueError("ARF summary missing a cell")
            summary.append({"dataset": dataset, "configuration": kind,
                            "size_multiplier": size,
                            "paired_sample_seeds": len(rows),
                            "failed_sample_seeds": len(missing),
                            "native_validation_kpi": rows[0][
                                "native_validation_kpi"],
                            "arf_artifact_bytes": rows[0]["arf_artifact_bytes"],
                            "dope_artifact_bytes": rows[0]["dope_artifact_bytes"],
                            "median_arf_retention": median(
                                row["arf_retention"] for row in rows),
                            "median_dope_retention": median(
                                row["dope_retention"] for row in rows),
                            "median_paired_difference": median(
                                row["difference_dope_minus_arf"]
                                for row in rows),
                            "arf_cells_failing_copy_gate": sum(
                                not row["arf_copy_gate_pass"] for row in rows)})
    return {"format": "dope-pilot24-arf-matched-validation",
            "version": 1, "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "l3_byte_cap": CAP,
            "locks": {"native_report": sha256(NATIVE),
                      "final_sample_round": sha256(sample_lock_path),
                      "final_validation_round": sha256(lock_path),
                      "exact_replay_manifest": sha256(manifest_path),
                      "dope_neural_report": sha256(
                          publish_synthpop_matched.NEURAL),
                      "dope_packed_news_report": sha256(
                          publish_synthpop_matched.PACKED)},
            "native_selections": {dataset: {"selection_receipt_sha256":
                                  choice["selection_receipt_sha256"],
                                  "selected_round": choice["selected_round"],
                                  "selected_trial": choice["selected_trial"],
                                  "selected_native_kpi": choice[
                                      "selected_native_kpi"],
                                  "default_status": choice["default_status"],
                                  "tuning_attempts": choice["tuning_attempts"],
                                  "tuning_wall_seconds": choice[
                                      "tuning_wall_seconds"]}
                                  for dataset, choice in native.items()},
            "cells": cells, "failed_sample_cells": failed,
            "unavailable_default_cells": unavailable,
            "summary": summary,
            "notes": ["ARF used author-affiliated arfpy source; tuning maximized a frozen held-out FORDE mean log density implemented as a labeled study evaluator of the author's fitted factors. KPI values compare ARF configurations within one dataset only.",
                      "Original default fits timed out on Adult and News under the unchanged cap and are explicit unavailable cells; they contribute no DOPE win.",
                      "One Adult tuned 4n sample timed out and has no ARF common metric or comparison win.",
                      "All available ARF fitted artifacts exceed the 10,240-byte L3 cap; copy counts are reported per sample. No release-safe ARF or DOPE conclusion follows.",
                      "Official tests remain sealed; MFS-v2, PTF-v1, production certification, and paper superiority claims are unavailable."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def render(report: dict) -> tuple[str, str]:
    table = io.StringIO()
    fields = ("dataset", "configuration", "size_multiplier",
              "paired_sample_seeds", "failed_sample_seeds",
              "native_validation_kpi", "arf_artifact_bytes",
              "dope_artifact_bytes", "median_arf_retention",
              "median_dope_retention", "median_paired_difference",
              "arf_cells_failing_copy_gate")
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["summary"])
    lines = ["# DOPE versus ARF: paired pilot validation", "",
             "ARF tuning used its frozen held-out FORDE mean log density "
             "within each dataset. The shared outcome is CatBoost TSTR/TRTR "
             "retention at fit seed 23 and paired sample seeds on "
             "training-derived validation rows.", "",
             "| Dataset | ARF config | Size | Paired seeds | Native density | "
             "ARF retention | DOPE retention | Paired DOPE − ARF | "
             "ARF bytes | Copy-gate failures |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        lines.append(f"| {row['dataset']} | {row['configuration']} | "
                     f"{row['size_multiplier']}n | "
                     f"{row['paired_sample_seeds']}/3 | "
                     f"{row['native_validation_kpi']:.6f} | "
                     f"{row['median_arf_retention']:.4f} | "
                     f"{row['median_dope_retention']:.4f} | "
                     f"{row['median_paired_difference']:+.4f} | "
                     f"{row['arf_artifact_bytes']:,} | "
                     f"{row['arf_cells_failing_copy_gate']}/"
                     f"{row['paired_sample_seeds']} |")
    lines += ["", "Adult and News original-default fits timed out and are "
              "explicitly unavailable. One Adult tuned 4n sample timed out; "
              "its ARF metric is absent and it contributes no DOPE win. "
              "All ARF fitted artifacts exceed L3. JSON retains 23 measured "
              "cells, the failed sample, unavailable defaults, row-copy "
              "counts, native selections, and immutable receipt hashes. "
              "Official tests remain sealed; MFS-v2, PTF-v1, and production "
              "certification are null. This pilot is not a paper superiority "
              "result.", ""]
    return table.getvalue(), "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-arf-matched-validation"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("ARF matched report changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    csv_text, markdown = render(report)
    RESULT.with_suffix(".csv").write_text(csv_text)
    RESULT.with_suffix(".md").write_text(markdown)
    print(json.dumps({"cells": len(report["cells"]),
                      "failed_sample_cells": len(report["failed_sample_cells"]),
                      "unavailable_default_cells":
                          len(report["unavailable_default_cells"]),
                      "summary_rows": len(report["summary"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()
