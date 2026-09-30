"""Reconcile the bounded pilot's immutable job and runner receipts.

The admission queue was frozen before supplementary compatibility probes.  A
repair receipt supersedes only the matching infrastructure failure, and both
attempts remain in the failure ledger.  This script never opens test tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .pilot_queue import DATASETS, FIT_SEEDS, ROSTER


ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-4h")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile_jobs(root: Path) -> tuple[list[dict], list[dict]]:
    """Return one final status per planned cell and every failed attempt."""
    expected = {f"{dataset}-{method}-{seed}" for dataset in DATASETS
                for method in ROSTER for seed in FIT_SEEDS}
    originals = {path.stem: _read(path) for path in (root / "receipts").glob("*.json")}
    if set(originals) != expected:
        raise ValueError(f"queue receipt set differs from frozen matrix: {sorted(expected ^ set(originals))}")
    repairs = {path.stem.removesuffix("-attempt2"): _read(path)
               for path in (root / "repair-receipts").glob("*.json")}
    if not set(repairs).issubset(expected):
        raise ValueError("repair outside frozen matrix")
    ledger = []
    final = []
    for name in sorted(expected):
        original = originals[name]
        if original["status"] != "ok":
            ledger.append({"cell": name, "attempt": 1, "status": original["status"],
                           "reason": original.get("reason"), "error_type": original.get("error_type"),
                           "log_sha256": original.get("log_sha256")})
        current = original
        if name in repairs:
            if original["status"] != "failed":
                raise ValueError(f"repair does not supersede a failure: {name}")
            original_log = root / "logs" / f"{name}.log"
            if not original_log.exists() or "adapter source digest changed" not in original_log.read_text():
                raise ValueError(f"repair is not the registered package-path failure: {name}")
            current = repairs[name]
            if current["status"] != "ok":
                ledger.append({"cell": name, "attempt": 2, "status": current["status"],
                               "reason": current.get("reason"),
                               "error_type": current.get("error_type"),
                               "log_sha256": current.get("log_sha256")})
        final.append({"cell": name, "dataset": original["dataset"],
                      "method": original["method"], "fit_seed": original["fit_seed"],
                      "host": current.get("host"), "status": current["status"],
                      "attempts": 2 if name in repairs else 1,
                      "elapsed_seconds": current["elapsed_seconds"]})
    return final, ledger


def summarize_runner(root: Path) -> dict:
    fits = defaultdict(list)
    samples = defaultdict(list)
    for path in (root / "results").glob("*/fit-receipt.json"):
        receipt = _read(path)
        key = (receipt["fit_identity"]["method"], receipt["fit_identity"]["dataset"])
        fits[key].append(receipt)
        if receipt["status"] == "ok":
            for sample_path in path.parent.glob("*.receipt.json"):
                samples[key].append(_read(sample_path))
    result = {}
    for key in sorted(fits):
        method, dataset = key
        fit_receipts = fits[key]
        sample_receipts = samples[key]
        result[f"{dataset}/{method}"] = {
            "fit_status_counts": dict(Counter(r["status"] for r in fit_receipts)),
            "fit_seconds": [round(r["fit_seconds"], 3) for r in fit_receipts],
            "artifact_bytes": [r["artifact_bytes"] for r in fit_receipts if r["status"] == "ok"],
            "expected_sample_cells": 8 * sum(r["status"] == "ok" for r in fit_receipts),
            "observed_sample_receipts": len(sample_receipts),
            "sample_status_counts": dict(Counter(r["status"] for r in sample_receipts)),
            "sample_seconds_total": round(sum(r["sample_seconds"] for r in sample_receipts), 3),
            "sample_determinism_failures": sum(r.get("artifact_sampling_verified") is False
                                               for r in sample_receipts),
        }
    return result


def build_report(root: Path) -> dict:
    jobs, ledger = reconcile_jobs(root)
    by_method = {method: dict(Counter(j["status"] for j in jobs if j["method"] == method))
                 for method in ROSTER}
    evidence = [root / "matrix.json", root / "queue-summary.json",
                root / "host-inventory.json"]
    evidence += sorted((root / "receipts").glob("*.json"))
    evidence += sorted((root / "repair-receipts").glob("*.json"))
    evidence += sorted((root / "results").glob("*/fit-receipt.json"))
    evidence += sorted((root / "results").glob("*/*.receipt.json"))
    evidence += sorted((root / "logs").glob("*.log"))
    evidence += sorted((root / "repair-logs").glob("*.log"))
    supplemental = {}
    for directory in ("aim-probe", "tabpc-probe", "aim-runner-results",
                      "aim-runner-hardcap-results", "aim-runner-hardcap-retry-results"):
        paths = sorted((root / directory).glob("**/receipt.json"))
        paths += sorted((root / directory).glob("**/fit-receipt.json"))
        paths += sorted((root / directory).glob("**/*.receipt.json"))
        if paths:
            supplemental[directory] = [
                {"path": str(path.relative_to(root)), "sha256": _hash(path),
                 "status": _read(path).get("status", "probe"),
                 "fit_seconds": _read(path).get("fit_seconds"),
                 "artifact_bytes": _read(path).get("artifact_bytes")}
                for path in paths]
    evidence += sorted((root / "validation-metrics").glob("*.json"))
    evidence += sorted(root.glob("aim-*.log"))
    evidence += sorted(root.glob("tabpc-*.log"))
    evidence += sorted((root / "compact-supplement").glob("receipts/*.json"))
    evidence += sorted((root / "compact-supplement").glob("results/*/fit-receipt.json"))
    evidence += sorted((root / "compact-supplement").glob("results/*/*.receipt.json"))
    evidence += sorted((root / "compact-supplement").glob("logs/*.log"))
    compact_supplement = {}
    for path in (root / "compact-supplement" / "receipts").glob("*.json"):
        receipt = _read(path)
        compact_supplement[path.stem] = {"status": receipt["status"],
                                         "elapsed_seconds": receipt["elapsed_seconds"]}
    return {
        "format": "dope-four-hour-pilot-reconciled-report", "version": 1,
        "scope": "validation_only_no_public_test", "fit_cells": len(jobs),
        "timing_note": "Original queue elapsed_seconds can include host-slot wait; use runner fit_seconds and sample_seconds for compute costs. Repair queue elapsed_seconds starts after slot admission.",
        "registered_matrix_bounds": {
            "pilot_fit_cells": 3 * 7 * 2,
            "pilot_potential_sample_cells": 3 * 7 * 2 * 2 * 4,
            "public_core_unique_names_before_task_filter": 20,
            "protected_default_methods_including_dope": 11,
            "protected_default_fit_cells_upper_bound": 20 * 11 * 5,
            "protected_default_sample_cells_upper_bound": 20 * 11 * 5 * 3 * 4,
            "fourteen_day_required_default_fit_cells_per_hour": round(20 * 11 * 5 / (14 * 24), 3),
            "fourteen_day_required_default_sample_cells_per_hour": round(20 * 11 * 5 * 3 * 4 / (14 * 24), 3),
            "excludes": ["tuning", "validation", "privacy_attacks", "author_faithful_track",
                         "neural_llm_dp_panels", "extension", "robustness"],
        },
        "final_status_counts": dict(Counter(j["status"] for j in jobs)),
        "by_method": by_method, "jobs": jobs, "failure_ledger": ledger,
        "runner_costs": summarize_runner(root),
        "supplemental_receipts": supplemental,
        "compact_supplement": compact_supplement,
        "evidence_sha256": {str(path.relative_to(root)): _hash(path)
                            for path in evidence if path.exists()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
