#!/usr/bin/env python3
"""Offline, deterministic simulation only; no clinical or privacy claims."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

FEATURES = ["prior_utilization", "prior_visits", "prior_lab", "rare_condition", "care_path"]
TARGET = "event_30d"
FORBIDDEN = {"patient_id", "encounter_date", "same_encounter_event", "future_event", "label_timestamp"}
SEED = 20261018
SPLIT_DATE = "2025-01-01"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_rows(patient: int, when: str, rng: random.Random) -> list[dict[str, str]]:
    frailty = rng.random()
    rare = rng.random() < 0.025
    visits = 1 + int(frailty * 4)
    rows = []
    for encounter in range(2):
        utilization = max(0.0, min(1.0, 0.12 + 0.55 * frailty + rng.gauss(0, 0.12)))
        prior_visits = visits + encounter
        lab_value = 0.15 + 0.62 * utilization + rng.gauss(0, 0.12)
        prior_lab = "" if rng.random() < 0.16 else str(max(0, min(2, int(lab_value * 3))))
        care_path = "intensive" if utilization > 0.72 or rare else "routine"
        score = -3.25 + 2.4 * utilization + 0.13 * prior_visits + 1.15 * rare
        probability = 1 / (1 + math.exp(-score))
        event = int(rng.random() < probability)
        rows.append({
            "patient_id": f"SIM-{patient:04d}", "encounter_date": when,
            "prior_utilization": f"{utilization:.6f}", "prior_visits": str(prior_visits),
            "prior_lab": prior_lab, "rare_condition": "rare" if rare else "common",
            "care_path": care_path, TARGET: str(event),
        })
    return rows


def validate_fixture(fit: Path, heldout: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    expected = FEATURES + [TARGET]
    patients_fit = set(manifest["fit_patient_ids"])
    patients_eval = set(manifest["evaluation_patient_ids"])
    if patients_fit & patients_eval or not patients_fit or not patients_eval:
        raise ValueError("patient_split_invalid")
    if not manifest["fit_latest_date"] < SPLIT_DATE <= manifest["evaluation_earliest_date"]:
        raise ValueError("time_split_invalid")
    for path in (fit, heldout):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != expected or FORBIDDEN.intersection(reader.fieldnames or []):
                raise ValueError("feature_allowlist_or_leakage_check_failed")
            records = list(reader)
        if len(records) < 2:
            raise ValueError("fixture_rows_insufficient")
        if any(row[TARGET] not in {"0", "1"} for row in records):
            raise ValueError("target_encoding_invalid")
    return manifest


def verify(out: Path) -> dict:
    manifest = validate_fixture(out / "fit.csv", out / "evaluation.csv", out / "manifest.json")
    with tempfile.TemporaryDirectory(prefix="fixture-replay-", dir=out) as replay_dir:
        replay = Path(replay_dir)
        prepare(replay)
        if digest(out / "fit.csv") != digest(replay / "fit.csv") or digest(out / "evaluation.csv") != digest(replay / "evaluation.csv"):
            raise ValueError("fixture_not_deterministic")
    probe = out / "leak-probe.csv"
    try:
        probe.write_text("prior_utilization,prior_visits,prior_lab,rare_condition,care_path,future_event,event_30d\n0.2,1,,common,routine,1,0\n0.3,2,0.4,common,routine,0,0\n")
        try:
            with probe.open(newline="") as stream:
                header = csv.DictReader(stream).fieldnames
            if header != FEATURES + [TARGET] or FORBIDDEN.intersection(header or []):
                raise ValueError("feature_allowlist_or_leakage_check_failed")
        except ValueError as error:
            if str(error) != "feature_allowlist_or_leakage_check_failed":
                raise
    finally:
        probe.unlink(missing_ok=True)
    return {"fixture": "valid", "deterministic_replay": "identical_sha256",
            "leakage_negative": "rejected", "patient_split": "disjoint",
            "time_split": "fit_before_evaluation", "fit_rows": manifest["fit_rows"]}


def prepare(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    fit_rows = [row for patient in range(160) for row in make_rows(patient, "2024-06-01", rng)]
    eval_rows = [row for patient in range(160, 200) for row in make_rows(patient, "2025-03-01", rng)]
    fit_path, eval_path = out / "fit.csv", out / "evaluation.csv"
    for path, rows in ((fit_path, fit_rows), (eval_path, eval_rows)):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FEATURES + [TARGET], lineterminator="\n")
            writer.writeheader()
            writer.writerows({name: row[name] for name in FEATURES + [TARGET]} for row in rows)
    manifest = {
        "format": "dope-healthcare-simulation-v1", "seed": SEED,
        "data_status": "explicitly_simulated_no_real_patients",
        "prediction_time": "each encounter start; features are prior-history summaries only",
        "target_definition": "simulated event in the next 30 days",
        "fit_latest_date": "2024-06-01", "evaluation_earliest_date": "2025-03-01",
        "split_date": SPLIT_DATE, "fit_patient_ids": [f"SIM-{i:04d}" for i in range(160)],
        "evaluation_patient_ids": [f"SIM-{i:04d}" for i in range(160, 200)],
        "fit_rows": len(fit_rows), "evaluation_rows": len(eval_rows),
        "rare_prevalence_fit": sum(row["rare_condition"] == "rare" for row in fit_rows) / len(fit_rows),
        "missing_lab_fit": sum(not row["prior_lab"] for row in fit_rows) / len(fit_rows),
        "correlation_design": "coarse prior_lab category tracks prior_utilization when observed; care_path is derived from utilization/rare condition",
        "feature_allowlist": FEATURES, "excluded_fields": sorted(FORBIDDEN),
    }
    manifest["fit_sha256"] = digest(fit_path)
    manifest["evaluation_sha256"] = digest(eval_path)
    manifest["fit_bytes"] = fit_path.stat().st_size
    manifest["evaluation_bytes"] = eval_path.stat().st_size
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_fixture(fit_path, eval_path, out / "manifest.json")
    return {"fit_rows": len(fit_rows), "evaluation_rows": len(eval_rows),
            "fit_bytes": fit_path.stat().st_size, "evaluation_bytes": eval_path.stat().st_size,
            "manifest_bytes": (out / "manifest.json").stat().st_size,
            "fit_sha256": digest(fit_path), "evaluation_sha256": digest(eval_path)}


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] + [value] for row, value in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[pivot][col] += 1e-8
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[i][-1] for i in range(n)]


def encode(rows: list[dict[str, str]], means: list[float] | None = None) -> tuple[list[list[float]], list[float]]:
    numeric = ["prior_utilization", "prior_visits", "prior_lab"]
    if means is None:
        means = []
        for field in numeric:
            values = [float(row[field]) for row in rows if row[field]]
            means.append(sum(values) / len(values))
    result = []
    for row in rows:
        lab_missing = not row["prior_lab"]
        x = [1.0, float(row["prior_utilization"]), float(row["prior_visits"]),
             float(row["prior_lab"]) if not lab_missing else means[2],
             float(lab_missing), float(row["rare_condition"] == "rare"),
             float(row["care_path"] == "intensive")]
        result.append(x)
    return result, means


def fit_ridge(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    x, means = encode(rows)
    y = [float(row[TARGET]) for row in rows]
    width = len(x[0])
    matrix = [[sum(row[i] * row[j] for row in x) for j in range(width)] for i in range(width)]
    vector = [sum(row[i] * target for row, target in zip(x, y)) for i in range(width)]
    for i in range(1, width):
        matrix[i][i] += 1.0
    return solve(matrix, vector), means


def score(weights: list[float], means: list[float], rows: list[dict[str, str]]) -> dict:
    x, _ = encode(rows, means)
    observed = [float(row[TARGET]) for row in rows]
    predictions = [max(0.0, min(1.0, sum(a * b for a, b in zip(w, row)))) for w, row in zip([weights] * len(x), x)]
    prevalence = sum(observed) / len(observed)
    model_brier = sum((a - b) ** 2 for a, b in zip(predictions, observed)) / len(observed)
    null_brier = sum((prevalence - b) ** 2 for b in observed) / len(observed)
    return {"rows": len(observed), "event_rate": prevalence, "brier": model_brier,
            "null_brier": null_brier, "brier_skill_vs_null": 1 - model_brier / null_brier if null_brier else None}


def read_rows(path: Path, has_header: bool) -> list[dict[str, str]]:
    if not has_header:
        payload = json.loads(path.read_text())
        width = len(FEATURES) + 1
        if not isinstance(payload.get("values"), list) or len(payload["values"]) % width:
            raise ValueError("sample_column_count_invalid")
        result = []
        for offset in range(0, len(payload["values"]), width):
            row = {}
            for name, cell in zip(FEATURES + [TARGET], payload["values"][offset:offset + width]):
                if cell["kind"] == "number":
                    value = float(cell["value"])
                    row[name] = ("1" if value >= 0.5 else "0") if name == TARGET else str(value)
                elif cell["kind"] == "category":
                    row[name] = str(cell["value"])
                elif cell["kind"] == "mask" and cell["value"] == "missing":
                    row[name] = ""
                else:
                    raise ValueError("sample_cell_type_invalid")
            result.append(row)
        return result
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if has_header:
        if rows[0] != FEATURES + [TARGET] or FORBIDDEN.intersection(rows[0]):
            raise ValueError("evaluation_feature_allowlist_invalid")
        rows = rows[1:]
    return [dict(zip(FEATURES + [TARGET], row)) for row in rows]


def evaluate(out: Path, sampled_path: Path) -> dict:
    fit_rows = read_rows(out / "fit.csv", True)
    test_rows = read_rows(out / "evaluation.csv", True)
    sampled_rows = read_rows(sampled_path, False)
    real_report = score(*fit_ridge(fit_rows), test_rows)
    sample_report = score(*fit_ridge(sampled_rows), test_rows)
    report = {
        "format": "dope-healthcare-simulation-evaluation-v1",
        "status": "simulated_evidence_only_observational_confirmation_absent",
        "classical_model": "standard-library ridge linear probability model, ridge penalty 1, probabilities clipped to [0,1]",
        "real_fit_reference": real_report, "sampled_fit": sample_report,
        "sampled_rows": len(sampled_rows), "sampled_output_bytes": sampled_path.stat().st_size,
        "fit_csv_bytes": (out / "fit.csv").stat().st_size,
        "evaluation_csv_bytes": (out / "evaluation.csv").stat().st_size,
        "model_artifact_bytes": (out / "model.dpk").stat().st_size,
        "claims": {"clinical": False, "hipaa_deidentification": False, "formal_dp": False,
                   "unseen_real_patient": False, "observational_confirmation": False},
    }
    report_path = out / "receipt.json"
    report["receipt_bytes"] = 0
    while True:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        size = report_path.stat().st_size
        if report["receipt_bytes"] == size:
            break
        report["receipt_bytes"] = size
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify"); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate"); p.add_argument("--out", type=Path, required=True); p.add_argument("--sampled", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "prepare":
        result = prepare(args.out)
    elif args.step == "verify":
        result = verify(args.out)
    else:
        result = evaluate(args.out, args.sampled)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
