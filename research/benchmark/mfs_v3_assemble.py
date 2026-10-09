"""Assemble an MFS-v3 density receipt from measured sidecars.

Missing auditors, failed gates, and undefined retentions stay null.
The scalar comes only from ``evaluate_v3``.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from research.benchmark.mfs_v3_panel import BYTE_CAP, SAMPLE_SEEDS, retention
from research.benchmark.representation import (
    NORMALIZER,
    TABULAR_AUDITORS,
    TABULAR_PROTOCOL,
    representation_closeness,
)
from research.benchmark.mfs_v3_score import PREREQUISITES, evaluate_v3

AUDITORS = ("kumo_tabular_l", "mitra_v2", "tabicl2")
HEADLINE = "kumo_tabular_l"


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _min(values: list[float]) -> float | None:
    if len(values) != len(SAMPLE_SEEDS) or any(not _finite(value) or not 0 <= value <= 1 for value in values):
        return None
    return float(min(values))


def _max(values: list[float]) -> float | None:
    if len(values) != len(SAMPLE_SEEDS) or any(not _finite(value) or not 0 <= value <= 1 for value in values):
        return None
    return float(max(values))


def seed_retention(row: dict) -> float | None:
    if not all(_finite(row.get(key)) and row[key] >= 0 for key in ("null_loss", "real_loss", "synthetic_loss")):
        return None
    return retention(float(row["null_loss"]), float(row["real_loss"]), float(row["synthetic_loss"]))


def lineage_retention(rows: list[dict]) -> float | None:
    if len(rows) != len(SAMPLE_SEEDS):
        return None
    values = [seed_retention(row) for row in rows]
    if any(value is None for value in values):
        return None
    return float(np.median(values))


def distribution_fidelity(marginal: float, sliced: float, mmd: float) -> float | None:
    if any(not _finite(value) or not 0 <= value <= 1 for value in (marginal, sliced, mmd)):
        return None
    return 0.50 * marginal + 0.25 * sliced + 0.25 * mmd


def compactness(artifact_bytes: float) -> float | None:
    if not _finite(artifact_bytes):
        return None
    return float(min(1.0, max(0.0, 1.0 - artifact_bytes / BYTE_CAP)))


def _gap(row: dict | None) -> float | None:
    if row is None or not all(_finite(row.get(key)) and 0 <= row[key] <= 1 for key in ("distance", "d_null")):
        return None
    return float(row["distance"]) - float(row["d_null"])


def closeness_observation(seeds: list[dict]) -> dict | None:
    """Worst seed whose embedding gates pass, or none if any seed fails."""
    observations = []
    for seed in seeds:
        auditors = seed.get("auditors") or {}
        headline = auditors.get(HEADLINE)
        gap_mitra = _gap(auditors.get("mitra_v2"))
        gap_tabicl = _gap(auditors.get("tabicl2"))
        if (
            headline is None
            or gap_mitra is None
            or gap_tabicl is None
            or not all(_finite(headline.get(key)) for key in ("distance", "d_null", "d_match"))
        ):
            return None
        closeness = representation_closeness(
            float(headline["distance"]),
            float(headline["d_null"]),
            float(headline["d_match"]),
            gap_mitra,
            gap_tabicl,
        )
        if closeness is None:
            return None
        observations.append((closeness, headline, gap_mitra, gap_tabicl))
    if len(observations) != len(SAMPLE_SEEDS):
        return None
    _closeness, headline, gap_mitra, gap_tabicl = min(observations, key=lambda item: item[0])
    return {
        "distance": float(headline["distance"]),
        "d_null": float(headline["d_null"]),
        "d_match": float(headline["d_match"]),
        "gap_mitra": gap_mitra,
        "gap_tabicl": gap_tabicl,
    }


def utility_component(retentions: list[float], bound) -> float | None:
    if len(retentions) != len(SAMPLE_SEEDS) or any(not _finite(value) for value in retentions):
        return None
    value = bound(retentions)
    if not _finite(value):
        return None
    return float(min(1.0, max(0.0, value)))


def score_lineage(seeds: list[dict], bound) -> dict:
    by_auditor = {
        name: lineage_retention([seed.get("auditors", {}).get(name) or {} for seed in seeds])
        for name in AUDITORS
    }
    holdouts = [seed.get("holdout") or {} for seed in seeds]
    marginal = _min([row.get("marginal_fidelity") for row in holdouts])
    sliced = _min([row.get("sliced_wasserstein_fidelity") for row in holdouts])
    mmd = _min([row.get("mmd_fidelity") for row in holdouts])
    dependence = _min([row.get("dependence_fidelity") for row in holdouts])
    coverage = _min([row.get("coverage_realism") for row in holdouts])
    driver = _min([row.get("driver_fidelity") for row in holdouts])
    distribution = None
    if marginal is not None and sliced is not None and mmd is not None:
        distribution = distribution_fidelity(marginal, sliced, mmd)
    bytes_known = [type(seed.get("artifact_bytes")) is int and 0 < seed["artifact_bytes"] <= BYTE_CAP
                   for seed in seeds]
    artifact_bytes = max(seed["artifact_bytes"] for seed in seeds) if all(bytes_known) and seeds else None
    headline_rows = [seed.get("auditors", {}).get(HEADLINE) or {} for seed in seeds]
    headline_retentions = [seed_retention(row) for row in headline_rows]
    # The gate needs finite losses from every auditor. A noninformative retention
    # is still a completed measurement when the three losses are finite.
    losses_ready = all(
        all(
            _finite((seed.get("auditors", {}).get(name) or {}).get(key))
            and (seed["auditors"][name][key] >= 0)
            for key in ("null_loss", "real_loss", "synthetic_loss")
        )
        for seed in seeds
        for name in AUDITORS
    )
    evidence = {
        **{key: len(seeds) == len(SAMPLE_SEEDS) and all(seed.get(key) is True for seed in seeds)
           for key in PREREQUISITES},
        "normalizer": NORMALIZER,
        "utility_protocol": TABULAR_PROTOCOL,
        "utility_auditors": list(TABULAR_AUDITORS) if losses_ready else [],
        "encoder": HEADLINE,
        "exact_row_matches": 0 if seeds and all(type(seed.get("exact_row_matches")) is int
                                               and seed["exact_row_matches"] == 0 for seed in seeds) else None,
        "near_copy_ok": all(seed.get("near_copy_ok") is True for seed in seeds),
        "cleartext_absent": all(seed.get("cleartext_absent") is True for seed in seeds),
        "membership_auc": _max([row.get("membership_auc") for row in holdouts]),
        "attribute_inference_advantage": _max([row.get("attribute_inference_advantage") for row in holdouts]),
        "artifact_bytes": artifact_bytes,
        "mfs_components": {
            "utility_transfer": utility_component(
                [value for value in headline_retentions if value is not None],
                bound,
            )
            if all(value is not None for value in headline_retentions)
            else None,
            "driver_fidelity": driver,
            "distribution_fidelity": distribution,
            "structure_fidelity": dependence,
            "coverage_realism": coverage,
            "compactness": compactness(artifact_bytes) if artifact_bytes is not None else None,
        },
    }
    observation = closeness_observation(seeds)
    if observation is not None:
        evidence.update(observation)
    report = evaluate_v3(evidence)
    return {
        "counts_as_dope_win": False,
        "failed_gates": report["failed_gates"],
        "official_tests_opened": False,
        "retention": by_auditor,
        "score": report["score"],
        "superiority": None,
    }


def compact_seed(seed: dict) -> dict:
    """Losses, distances, and gates. No vectors and no source strings."""
    auditors = {}
    for name, row in (seed.get("auditors") or {}).items():
        auditors[name] = {
            key: row.get(key)
            for key in ("null_loss", "real_loss", "synthetic_loss", "distance", "d_null", "d_match")
        }
    holdout = seed.get("holdout") or {}
    return {
        **{key: seed.get(key) for key in PREREQUISITES},
        "artifact_bytes": seed.get("artifact_bytes"),
        "auditors": auditors,
        "cleartext_absent": seed.get("cleartext_absent"),
        "exact_row_matches": seed.get("exact_row_matches"),
        "holdout": {
            key: holdout.get(key)
            for key in (
                "attribute_inference_advantage",
                "coverage_realism",
                "dependence_fidelity",
                "driver_fidelity",
                "marginal_fidelity",
                "membership_auc",
                "mmd_fidelity",
                "sliced_wasserstein_fidelity",
            )
        },
        "near_copy_ok": seed.get("near_copy_ok"),
        "sample_seed": seed.get("sample_seed"),
    }


def load_seeds(cohort_cell_dir: Path) -> list[dict]:
    seeds = []
    for sample_seed in SAMPLE_SEEDS:
        base = cohort_cell_dir / f"seed{sample_seed}"
        prepared = json.loads(base.with_suffix(".json").read_text())
        auditors = {}
        for name in AUDITORS:
            path = base.with_name(base.name + f".{name}.json")
            if path.is_file():
                auditors[name] = json.loads(path.read_text())
        holdout_path = base.with_name(base.name + ".holdout.json")
        seeds.append(
            {
                **{key: prepared.get(key) for key in PREREQUISITES},
                "artifact_bytes": prepared.get("artifact_bytes"),
                "auditors": auditors,
                "cleartext_absent": prepared.get("cleartext_absent"),
                "exact_row_matches": prepared.get("exact_row_matches"),
                "holdout": json.loads(holdout_path.read_text()) if holdout_path.is_file() else {},
                "near_copy_ok": prepared.get("near_copy_ok"),
                "sample_seed": sample_seed,
            }
        )
    return seeds


def panel_receipt(lineages: list[dict]) -> dict:
    return {
        "counts_as_dope_win": False,
        "format": "dope-mfs-v3-density-panel",
        "lineages": lineages,
        "official_tests_opened": False,
        "superiority": None,
        "version": 1,
    }


METHODS = ("DOPE", "GaussianCopula", "Chow-Liu", "independent_marginals")
COMPARATORS = ("GaussianCopula", "Chow-Liu", "independent_marginals")


def binary_lower_bound(binary: Path):
    """One-sided 95% bound from the production helper. A non-finite result stays null."""

    def bound(values: list[float]) -> float | None:
        completed = subprocess.run(
            [str(binary), "lower", *(f"{value:.17g}" for value in values)],
            check=True,
            capture_output=True,
            text=True,
        )
        text = completed.stdout.strip()
        if text == "null":
            return None
        value = float(text)
        if not math.isfinite(value):
            return None
        return value

    return bound


def retention_maps(lineages: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    maps = {name: {method: {} for method in METHODS} for name in AUDITORS}
    for row in lineages:
        method = row.get("method")
        dataset = row.get("dataset")
        if method not in METHODS or not isinstance(dataset, str):
            continue
        for name, value in (row.get("retention") or {}).items():
            if name in maps and _finite(value):
                maps[name][method][dataset] = float(value)
    return maps


def _summaries(maps: dict[str, dict[str, dict[str, float]]]) -> dict:
    import sys

    scripts = Path(__file__).resolve().parents[2] / "docs" / "whitepaper" / "scripts"
    sys.path.insert(0, str(scripts))
    from compute_panel import paired_test, summarize

    block = {}
    for auditor, methods in maps.items():
        pairs = {}
        dope = methods["DOPE"]
        for comparator in COMPARATORS:
            pair = paired_test(dope, methods[comparator], None, len(COMPARATORS))
            pairs[comparator] = {
                key: pair[key]
                for key in (
                    "n",
                    "median_difference",
                    "lo",
                    "hi",
                    "dope_median",
                    "dope_lo",
                    "dope_hi",
                    "other_median",
                    "other_lo",
                    "other_hi",
                    "wins",
                    "ties",
                    "losses",
                )
            }
        block[auditor] = {
            "methods": {method: summarize(values.values()) for method, values in methods.items()},
            "pairs": pairs,
        }
    return block


def collect_panel(cohort_path: Path, bound) -> dict:
    document = json.loads(cohort_path.read_text())
    if document.get("counts_as_dope_win") is not False or document.get("official_tests_opened") is not False:
        raise ValueError("cohort claims are not allowed")
    grouped: dict[tuple[str, str], Path] = {}
    for cell in document["cells"]:
        mapped = Path(cell["mapped"])
        grouped.setdefault((cell["method"], cell["dataset"]), mapped.parent)
    lineages = []
    failures: Counter[str] = Counter()
    scored = 0
    for method, dataset in sorted(grouped):
        seeds = load_seeds(grouped[(method, dataset)])
        for seed in seeds:
            present = set((seed.get("auditors") or {}))
            if present != set(AUDITORS):
                raise ValueError("a lineage is missing a tabular auditor")
        row = score_lineage(seeds, bound)
        row["dataset"] = dataset
        row["method"] = method
        row["seeds"] = [compact_seed(seed) for seed in seeds]
        if row["score"] is None:
            for gate in row["failed_gates"]:
                failures[f"{method}:{gate}"] += 1
        else:
            scored += 1
        lineages.append(row)
    receipt = panel_receipt(lineages)
    receipt["retention_summary"] = _summaries(retention_maps(lineages))
    receipt["scored_lineages"] = scored
    receipt["failed_gate_counts"] = dict(sorted(failures.items()))
    return receipt


def main() -> None:
    import os

    cohort = Path(os.environ["MFS_V3_COHORT"])
    binary = Path(os.environ["MFS_V3_BIN"])
    out = Path(os.environ["MFS_V3_RECEIPT"])
    receipt = collect_panel(cohort, binary_lower_bound(binary))
    if receipt.get("counts_as_dope_win") is not False or receipt.get("official_tests_opened") is not False:
        raise ValueError("cohort claims are not allowed")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    print(
        f"lineages {len(receipt['lineages'])} scored {receipt['scored_lineages']} -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
