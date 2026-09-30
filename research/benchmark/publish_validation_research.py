"""Regenerate rights-safe exploratory tables and figure from scratch receipts."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

from .score import sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
ROOT = SCRATCH / "gpu-research-v1"
RESULTS = Path(__file__).with_name("results")


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _metric(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("mfs_v2") is not None or report.get("gate_profile_complete") is not False:
        raise ValueError("validation pilot metric cannot be a gate pass")
    return {"metric_sha256": sha256(path),
            "catboost_retention": _number(report["utility"]["catboost"].get("retention")),
            "linear_retention": _number(report["utility"]["linear"].get("retention")),
            "mlp_retention": _number(report["utility"]["mlp"].get("retention")),
            "c2st_auc": _number(report["c2st_auc"]),
            "exact_copies": report["copy_counts"]["exact"],
            "near_copies": report["copy_counts"]["near"]}


def _gpu_cells(round_number: int) -> list[dict]:
    queue = ROOT / f"round{round_number}-queue"
    cells = []
    for path in sorted(queue.glob("*/attempt-*.json")):
        if not re.fullmatch(r"attempt-\d{4}\.json", path.name):
            continue
        receipt = json.loads(path.read_text())
        child = receipt.get("child")
        fit = json.loads(Path(child["receipt"]).read_text()) if child else {}
        metric_path = Path(child["receipt"]).parent / "validation-metrics.json" if child else None
        job = receipt["job"]
        cell = {"phase": {0: "discovery", 1: "discovery_refinement", 2: "confirmation"}[round_number],
                "dataset": job["dataset"], "family": job["candidate"],
                "target_weight": job["target_weight"],
                "structural_penalty": job["structural_penalty"],
                "status": receipt["status"], "host": receipt["host"],
                "attempt_receipt_path": str(path), "attempt_receipt_sha256": sha256(path),
                "artifact_bytes": fit.get("artifact_bytes"),
                "encoded_candidate_bytes": fit.get("encoded_candidate_bytes"),
                "failure_code": fit.get("failure_code"),
                "fit_seconds": _number(fit.get("elapsed_seconds")),
                "peak_gpu_used_mib": fit.get("peak_gpu_used_mib"),
                "energy_joules_estimate": _number(fit.get("energy_joules_estimate"))}
        if metric_path and metric_path.exists():
            cell.update(_metric(metric_path))
        else:
            cell.update({"metric_sha256": None, "catboost_retention": None,
                         "linear_retention": None, "mlp_retention": None,
                         "c2st_auc": None, "exact_copies": None, "near_copies": None})
        cells.append(cell)
    return cells


def _symbolic_cells(round_number: int) -> list[dict]:
    base = ROOT / f"round{round_number}-symbolic"
    cells = []
    for path in sorted((base / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text())
        dataset = receipt.get("cell", {}).get("dataset")
        if dataset is None:
            continue
        child = receipt["child"][0] if receipt.get("child") else None
        fit_path = base / "results" / child["fit_key"] / "fit-receipt.json" if child else None
        fit = json.loads(fit_path.read_text()) if fit_path and fit_path.exists() else {}
        metric_path = base / "metrics" / f"{dataset}.json"
        cell = {"phase": "discovery" if round_number == 0 else "confirmation",
                "dataset": dataset, "family": "symbolic_default",
                "target_weight": None, "structural_penalty": None,
                "status": receipt["status"], "host": fit.get("host"),
                "attempt_receipt_path": str(path), "attempt_receipt_sha256": sha256(path),
                "artifact_bytes": fit.get("artifact_bytes"),
                "encoded_candidate_bytes": None, "failure_code": None,
                "fit_seconds": _number(fit.get("fit_seconds")),
                "peak_gpu_used_mib": None, "energy_joules_estimate": None}
        cell.update(_metric(metric_path) if metric_path.exists() else {
            "metric_sha256": None, "catboost_retention": None,
            "linear_retention": None, "mlp_retention": None,
            "c2st_auc": None, "exact_copies": None, "near_copies": None})
        cells.append(cell)
    return cells


def _survey_cells() -> list[dict]:
    base = ROOT / "round3-symbolic"
    cells = []
    fits = {row["fit_identity"]["dataset"]: row
            for path in (base / "results").glob("*/fit-receipt.json")
            if (row := json.loads(path.read_text()))}
    for path in sorted((base / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text())
        dataset = receipt["cell"]["dataset"]
        metric_path = base / "metrics" / f"{dataset}.json"
        fit = fits.get(dataset, {})
        cell = {"phase": "all_validation_lineages", "dataset": dataset,
                "family": "symbolic_default", "target_weight": None,
                "structural_penalty": None, "status": receipt["status"],
                "host": fit.get("host"), "attempt_receipt_path": str(path),
                "attempt_receipt_sha256": sha256(path),
                "artifact_bytes": fit.get("artifact_bytes"),
                "encoded_candidate_bytes": None, "failure_code": None,
                "fit_seconds": _number(fit.get("fit_seconds")),
                "peak_gpu_used_mib": None, "energy_joules_estimate": None}
        cell.update(_metric(metric_path) if metric_path.exists() else {
            "metric_sha256": None, "catboost_retention": None,
            "linear_retention": None, "mlp_retention": None,
            "c2st_auc": None, "exact_copies": None, "near_copies": None})
        cells.append(cell)
    return cells


def build() -> dict:
    cells = (_gpu_cells(0) + _gpu_cells(1) + _gpu_cells(2)
             + _symbolic_cells(0) + _symbolic_cells(2) + _survey_cells())
    if len(cells) != 172:
        raise ValueError("research matrix is incomplete")
    confirmation = [cell for cell in cells if cell["phase"] == "confirmation"]
    paired = []
    for symbolic in (cell for cell in confirmation if cell["family"] == "symbolic_default"):
        neural = [cell for cell in confirmation if cell["dataset"] == symbolic["dataset"]
                  and cell["family"] == "micro_tvae_4_16"]
        default = next(cell for cell in neural if cell["target_weight"] == 2.0
                       and cell["structural_penalty"] == 0.0)
        paired.append({"dataset": symbolic["dataset"],
                       "symbolic_catboost_retention": symbolic["catboost_retention"],
                       "micro_default_catboost_retention": default["catboost_retention"],
                       "symbolic_minus_micro_default":
                           symbolic["catboost_retention"] - default["catboost_retention"]})
    survey = [cell for cell in cells if cell["phase"] == "all_validation_lineages"]
    measured = [cell["catboost_retention"] for cell in survey
                if cell["catboost_retention"] is not None]
    return {"format": "dope-benchmark-validation-research", "version": 1,
            "scope": "validation_only_single_fit_seed_single_sample_seed",
            "s3_data_lock_sha256": sha256(RESULTS / "s3-data.lock.json"),
            "round_lock_sha256": {str(number): sha256(ROOT / f"round{number}.lock.json")
                                    for number in range(4)},
            "cells": sorted(cells, key=lambda cell: (cell["phase"], cell["dataset"],
                                                      cell["family"], cell["target_weight"] or 0,
                                                      cell["structural_penalty"] or 0)),
            "confirmation_pairs": sorted(paired, key=lambda row: row["dataset"]),
            "aggregate": {"cells": len(cells),
                          "gpu_fits": 60,
                          "gpu_l3_artifact_fits": sum(cell["status"] == "ok" for cell in cells
                                                      if cell["family"] != "symbolic_default"),
                          "s3_candidate_lineages": 100,
                          "survey_completed": sum(cell["status"] == "ok" for cell in survey),
                          "survey_catboost_retention_measured": len(measured),
                          "survey_catboost_retention_median": statistics.median(measured)
                          if measured else None,
                          "confirmation_symbolic_above_micro_default":
                              sum(row["symbolic_minus_micro_default"] > 0 for row in paired)},
            "ptf_v1": None, "mfs_v2": None,
            "production_certified": False, "public_or_s3_test_opened": False}


def _figure(pairs: list[dict]) -> tuple[bytes, bytes]:
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "dope-validation-v1"
    figure, axis = plt.subplots(figsize=(8, 3.8))
    labels = [row["dataset"] for row in pairs]
    values = [row["symbolic_minus_micro_default"] for row in pairs]
    axis.barh(labels, values, color="#2c6e91")
    axis.invert_yaxis()
    axis.set_xlabel("Symbolic minus micro-TVAE CatBoost retention")
    axis.set_title("Validation-only confirmation, one fit and sample seed")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    svg, pdf = BytesIO(), BytesIO()
    figure.savefig(svg, format="svg", metadata={"Date": None, "Creator": "DOPE benchmark"})
    figure.savefig(pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None,
                                                "Creator": "DOPE benchmark"})
    plt.close(figure)
    svg_bytes = svg.getvalue()
    svg_bytes = b"\n".join(line.rstrip(b" \t") for line in svg_bytes.split(b"\n"))
    return svg_bytes, pdf.getvalue()


def main() -> None:
    result = build()
    RESULTS.mkdir(exist_ok=True)
    svg, pdf = _figure(result["confirmation_pairs"])
    outputs = {RESULTS / "gpu-validation-research.json":
               (json.dumps(result, sort_keys=True, indent=2) + "\n").encode(),
               RESULTS / "gpu-validation-retention.svg": svg,
               RESULTS / "gpu-validation-retention.pdf": pdf}
    table = RESULTS / "gpu-validation-research.csv"
    from io import StringIO
    stream = StringIO()
    fields = ["phase", "dataset", "family", "target_weight", "structural_penalty",
              "status", "artifact_bytes", "encoded_candidate_bytes", "failure_code",
              "catboost_retention", "linear_retention", "mlp_retention", "metric_sha256",
              "attempt_receipt_sha256"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(result["cells"])
    outputs[table] = stream.getvalue().encode()
    for path, encoded in outputs.items():
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError("committed exploratory result differs from scratch receipts")
        else:
            path.write_bytes(encoded)
    print(json.dumps({path.name: sha256(path) for path in outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
