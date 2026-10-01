"""Pre-registered paired public-core L3 comparison from frozen gate reports."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .admission import assess


def interval(values: list[float], comparisons: int, repetitions: int = 10000) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("at least two paired datasets required")
    rng = random.Random(1729)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                   for _ in range(repetitions))
    tail = 0.05 / (2 * comparisons)
    return means[int(tail * repetitions)], means[min(repetitions - 1, int((1 - tail) * repetitions))]


def sign_pvalue(values: list[int]) -> float | None:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    discordant = positive + negative
    if discordant == 0:
        return None
    return sum(math.comb(discordant, k) for k in range(positive, discordant + 1)) / 2**discordant


def compare(matrix: list[dict], reports: list[dict], method_lock: dict) -> dict:
    """Each matrix row names an applicable public-core dataset and method."""
    selected = {(row["dataset"], row["method"]) for row in matrix
                if row["panel"] == "public_core" and row["track"] == "common_numeric"
                and row["tier"] == "l3" and row["applicable"]}
    by_key = {}
    for report in reports:
        key = (report["dataset"], report["method"])
        if key in by_key:
            raise ValueError("duplicate dataset-method report")
        by_key[key] = report
    comparators = sorted({method for _, method in selected if method != "dope"
                          and method_lock["methods"][method]["group"] == "compact"
                          and method_lock["methods"][method]["status"] == "locked"})
    output = []
    for method in comparators:
        datasets = sorted(dataset for dataset, name in selected if name == method
                          and (dataset, "dope") in selected)
        pass_differences = []
        score_differences = []
        baseline_completed = 0
        failures = {"dope_missing_or_timeout": 0, "baseline_missing_or_timeout": 0}
        for dataset in datasets:
            dope = by_key.get((dataset, "dope"))
            baseline = by_key.get((dataset, method))
            if dope is None or dope.get("status") in ("missing", "timeout"):
                failures["dope_missing_or_timeout"] += 1
            if baseline is None or baseline.get("status") in ("missing", "timeout"):
                failures["baseline_missing_or_timeout"] += 1
            if baseline and baseline.get("status") == "ok":
                baseline_completed += 1
            dope_pass = bool(dope and dope.get("status") == "ok" and dope.get("eligible") is True)
            base_pass = bool(baseline and baseline.get("status") == "ok" and baseline.get("eligible") is True)
            pass_differences.append(int(dope_pass) - int(base_pass))
            if dope_pass and base_pass:
                if not isinstance(dope.get("score"), (float, int)) or not isinstance(baseline.get("score"), (float, int)):
                    raise ValueError("paired passer lacks MFS-v2 score")
                score_differences.append(dope["score"] - baseline["score"])
        pass_ci = interval(pass_differences, len(comparators)) if len(datasets) >= 2 else None
        score_ci = interval(score_differences, len(comparators)) if len(score_differences) >= 10 else None
        output.append({"comparator": method, "paired_datasets": len(datasets),
                       "gate_pass_difference": sum(pass_differences) / len(datasets) if datasets else None,
                       "gate_pass_familywise_95_ci": pass_ci,
                       "gate_pass_one_sided_sign_p": sign_pvalue(pass_differences),
                       "gate_pass_superiority": False,
                       "baseline_completed_datasets": baseline_completed,
                       "paired_passing_datasets": len(score_differences),
                       "mfs_v2_difference_among_paired_passers":
                           sum(score_differences) / len(score_differences) if score_differences else None,
                       "mfs_v2_familywise_95_ci": score_ci,
                       "mfs_v2_conclusion": "superior" if score_ci and score_ci[0] > 0 else "inconclusive",
                       "failures": failures})
    ordered = sorted((row for row in output if row["gate_pass_one_sided_sign_p"] is not None),
                     key=lambda row: row["gate_pass_one_sided_sign_p"])
    running = 0.0
    for index, row in enumerate(ordered):
        running = max(running, min(1.0, row["gate_pass_one_sided_sign_p"] * (len(output) - index)))
        row["gate_pass_holm_adjusted_p"] = running
    for row in output:
        row.setdefault("gate_pass_holm_adjusted_p", None)
        adjusted_p = row["gate_pass_holm_adjusted_p"]
        pass_ci = row["gate_pass_familywise_95_ci"]
        row["gate_pass_superiority"] = bool(
            row["baseline_completed_datasets"] and pass_ci and pass_ci[0] > 0
            and adjusted_p is not None and adjusted_p < 0.05)
    return {"format": "dope-benchmark-primary-analysis", "version": 1,
            "comparison_set": "paired_applicable_public_core_common_numeric_l3",
            "interval_method": "dataset_cluster_bootstrap_bonferroni_familywise_95",
            "test_method": "paired_one_sided_sign_test_holm_adjusted",
            "comparators": output,
            "overall_superiority": bool(output) and all(row["gate_pass_superiority"] for row in output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("reports", type=Path)
    parser.add_argument("methods", type=Path)
    args = parser.parse_args()
    admission = assess(Path(__file__).resolve().parents[2], args.methods.parent)
    if not admission["admitted"]:
        parser.error("final analysis admission failed: " + ", ".join(admission["blockers"]))
    matrix = json.loads(args.matrix.read_text())["cells"]
    reports = [json.loads(line) for line in args.reports.open()]
    print(json.dumps(compare(matrix, reports, json.loads(args.methods.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
