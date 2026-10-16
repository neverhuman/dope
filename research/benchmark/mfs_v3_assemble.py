"""Assemble an MFS-v3 density receipt from measured sidecars.

Missing auditors, failed gates, and undefined retentions stay null.
The scalar comes only from ``evaluate_v3``.
Boolean gates pass only on the complete expected seed set with literal True
measurements. A literal False establishes failure even with other evidence
missing; otherwise missing, malformed, empty or incomplete evidence stays null.
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
from research.benchmark.mfs_v3_score import (
    PREREQUISITES,
    V4_PREREQUISITES,
    V4_TABULAR_AUDITORS,
    evaluate_v3,
    evaluate_v4,
)

AUDITORS = ("kumo_tabular_l", "mitra_v2", "tabicl2")
HEADLINE = "kumo_tabular_l"


def _boolean_gate_evidence_for(seeds: list[dict], prerequisites: tuple[str, ...], fmt: str) -> dict:
    complete = (
        len(seeds) == len(SAMPLE_SEEDS)
        and all(type(seed.get("sample_seed")) is int for seed in seeds)
        and {seed["sample_seed"] for seed in seeds} == set(SAMPLE_SEEDS)
    )
    values = {}
    for key in (*prerequisites, "near_copy_ok", "cleartext_absent"):
        measured = [seed.get(key) for seed in seeds]
        values[key] = (
            False if any(value is False for value in measured)
            else True if complete and all(value is True for value in measured)
            else None
        )
    def gate_name(key):
        return "near_copy" if key == "near_copy_ok" else key
    return {
        "format": fmt,
        "version": 1,
        "complete_seed_set": complete,
        "values": values,
        "unavailable_gates": [gate_name(key) for key, value in values.items() if value is None],
        "measured_failed_gates": [gate_name(key) for key, value in values.items() if value is False],
    }


def boolean_gate_evidence(seeds: list[dict]) -> dict:
    """Versioned tri-state diagnostics; failed_gates still blocks unknown evidence."""
    return _boolean_gate_evidence_for(seeds, PREREQUISITES, "dope-mfs-v3-boolean-gate-evidence-v1")


def boolean_gate_evidence_v4(seeds: list[dict]) -> dict:
    """Versioned tri-state diagnostics for MFS-v4 prerequisites and release gates."""
    return _boolean_gate_evidence_for(seeds, V4_PREREQUISITES, "dope-mfs-v4-boolean-gate-evidence-v1")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sample_mean_and_sd(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2 or any(not _finite(value) for value in values):
        return None
    m = float(len(values))
    total = sum(float(value) for value in values)
    if not math.isfinite(total):
        return None
    mean = total / m
    sq_sum = 0.0
    for value in values:
        diff = float(value) - mean
        sq = diff * diff
        if not math.isfinite(sq):
            return None
        sq_sum += sq
        if not math.isfinite(sq_sum):
            return None
    variance = sq_sum / (m - 1.0)
    if not math.isfinite(variance) or variance < 0.0:
        return None
    return float(mean), float(math.sqrt(variance))


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
    mean_val = sum(float(value) for value in retentions) / len(retentions)
    value = bound(retentions)
    if not _finite(value) or float(value) > mean_val + 1e-12:
        return None
    return float(min(1.0, max(0.0, value)))


def score_lineage(seeds: list[dict], bound) -> dict:
    gates = boolean_gate_evidence(seeds)
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
    artifact_bytes = max(seed["artifact_bytes"] for seed in seeds) if gates["complete_seed_set"] and all(bytes_known) else None
    headline_rows = [seed.get("auditors", {}).get(HEADLINE) or {} for seed in seeds]
    headline_retentions = [seed_retention(row) for row in headline_rows]
    # The gate needs finite losses from every auditor. A noninformative retention
    # is still a completed measurement when the three losses are finite.
    losses_ready = gates["complete_seed_set"] and all(
        all(
            _finite((seed.get("auditors", {}).get(name) or {}).get(key))
            and (seed["auditors"][name][key] >= 0)
            for key in ("null_loss", "real_loss", "synthetic_loss")
        )
        for seed in seeds
        for name in AUDITORS
    )
    evidence = {
        **gates["values"],
        "normalizer": NORMALIZER,
        "utility_protocol": TABULAR_PROTOCOL,
        "utility_auditors": list(TABULAR_AUDITORS) if losses_ready else [],
        "encoder": HEADLINE,
        "exact_row_matches": 0 if gates["complete_seed_set"] and all(type(seed.get("exact_row_matches")) is int
                                               and seed["exact_row_matches"] == 0 for seed in seeds) else None,
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
        "gate_evidence": gates,
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


def _single_seed_v4_evidence(seed: dict) -> dict:
    holdout = seed.get("holdout") if isinstance(seed.get("holdout"), dict) else {}
    auditors = seed.get("auditors") if isinstance(seed.get("auditors"), dict) else {}
    headline = auditors.get(HEADLINE) if isinstance(auditors.get(HEADLINE), dict) else {}

    def pick(key, default=None):
        if key in seed:
            return seed.get(key)
        if key in holdout:
            return holdout.get(key)
        return default

    ret = seed.get("retention") if "retention" in seed else seed_retention(headline)
    distance = seed.get("distance") if "distance" in seed else headline.get("distance")
    d_null = seed.get("d_null") if "d_null" in seed else headline.get("d_null")
    d_match = seed.get("d_match") if "d_match" in seed else headline.get("d_match")
    gap_mitra = seed.get("gap_mitra") if "gap_mitra" in seed else _gap(auditors.get("mitra_v2"))
    gap_tabicl = seed.get("gap_tabicl") if "gap_tabicl" in seed else _gap(auditors.get("tabicl2"))

    if "utility_auditors" in seed:
        utility_auditors = seed.get("utility_auditors")
    elif auditors and all(
        all(
            _finite((auditors.get(name) or {}).get(key)) and auditors[name][key] >= 0
            for key in ("null_loss", "real_loss", "synthetic_loss")
        )
        for name in V4_TABULAR_AUDITORS
    ):
        utility_auditors = list(V4_TABULAR_AUDITORS)
    else:
        utility_auditors = []

    if isinstance(seed.get("mfs_components"), dict):
        components = dict(seed["mfs_components"])
    else:
        marginal = holdout.get("marginal_fidelity")
        sliced = holdout.get("sliced_wasserstein_fidelity")
        mmd = holdout.get("mmd_fidelity")
        dist = (
            distribution_fidelity(marginal, sliced, mmd)
            if marginal is not None and sliced is not None and mmd is not None
            else None
        )
        components = {
            "distribution_fidelity": dist,
            "structure_fidelity": holdout.get("dependence_fidelity"),
            "coverage_realism": holdout.get("coverage_realism"),
        }

    legacy_cohort_seed = bool(auditors)
    return {
        **{key: seed.get(key) for key in V4_PREREQUISITES},
        "encoder": (
            seed.get("encoder")
            if "encoder" in seed
            else (HEADLINE if legacy_cohort_seed else None)
        ),
        "distance": distance,
        "d_null": d_null,
        "d_match": d_match,
        "gap_mitra": gap_mitra,
        "gap_tabicl": gap_tabicl,
        "exact_row_matches": seed.get("exact_row_matches"),
        "near_copy_ok": seed.get("near_copy_ok"),
        "cleartext_absent": seed.get("cleartext_absent"),
        "membership_auc": pick("membership_auc"),
        "attribute_inference_advantage": pick("attribute_inference_advantage"),
        "artifact_bytes": seed.get("artifact_bytes"),
        "encoded_numbers": pick("encoded_numbers"),
        "real_rows": pick("real_rows"),
        "real_columns": pick("real_columns"),
        "normalizer": (
            seed.get("normalizer")
            if "normalizer" in seed
            else (NORMALIZER if legacy_cohort_seed else None)
        ),
        "utility_protocol": (
            seed.get("utility_protocol")
            if "utility_protocol" in seed
            else (TABULAR_PROTOCOL if legacy_cohort_seed else None)
        ),
        "utility_auditors": utility_auditors,
        "retention": ret,
        "predictor_only_retention": pick("predictor_only_retention"),
        "independent_marginals_retention": pick("independent_marginals_retention"),
        "real_bootstrap_4n_retention": pick("real_bootstrap_4n_retention"),
        "real_importances": pick("real_importances"),
        "synthetic_importances": pick("synthetic_importances"),
        "real_signs": pick("real_signs"),
        "synthetic_signs": pick("synthetic_signs"),
        "mfs_components": components,
    }


def driver_observation_v4(seeds: list[dict]) -> dict | None:
    """Worst seed whose holdout driver attribution gates pass, or None if any seed fails."""
    if len(seeds) != len(SAMPLE_SEEDS):
        return None
    observations = []
    for idx, seed in enumerate(seeds):
        ev = _single_seed_v4_evidence(seed)
        rep = evaluate_v4(ev)
        drv_fid = rep.get("driver_fidelity")
        drv_attr = rep.get("driver_attribution")
        if drv_fid is None or drv_attr is None:
            return None
        sample_seed = seed.get("sample_seed")
        sort_seed = sample_seed if type(sample_seed) is int else idx
        observations.append((float(drv_fid), float(drv_attr["top_k_overlap"]), sort_seed, ev))
    _fid, _overlap, _seed, worst_ev = min(observations, key=lambda item: (item[0], item[1], item[2]))
    return {
        "real_importances": list(worst_ev["real_importances"]),
        "synthetic_importances": list(worst_ev["synthetic_importances"]),
        "real_signs": list(worst_ev["real_signs"]),
        "synthetic_signs": list(worst_ev["synthetic_signs"]),
    }


def score_lineage_v4(seeds: list[dict], bound) -> dict:
    """Score a 3-seed MFS-v4 lineage fail-closed with Rust parity and seed variance."""
    gates = boolean_gate_evidence_v4(seeds)
    complete = gates["complete_seed_set"]
    ordered_seeds = sorted(seeds, key=lambda s: s["sample_seed"]) if complete else list(seeds)
    seed_evidences = [_single_seed_v4_evidence(s) for s in ordered_seeds]
    seed_reports = [evaluate_v4(ev) for ev in seed_evidences]

    def _imp_len(val) -> int:
        return len(val) if isinstance(val, (list, tuple)) else -1

    first = seed_evidences[0] if seed_evidences else {}
    schema_consistent = (
        complete
        and type(first.get("real_rows")) is int
        and type(first.get("real_columns")) is int
        and isinstance(first.get("real_importances"), (list, tuple))
        and isinstance(first.get("synthetic_importances"), (list, tuple))
        and all(
            type(ev.get("real_rows")) is int
            and type(ev.get("real_columns")) is int
            and (
                ev.get("real_rows"),
                ev.get("real_columns"),
                _imp_len(ev.get("real_importances")),
                _imp_len(ev.get("synthetic_importances")),
                ev.get("normalizer"),
                ev.get("utility_protocol"),
                ev.get("utility_auditors"),
                ev.get("encoder"),
            )
            == (
                first.get("real_rows"),
                first.get("real_columns"),
                _imp_len(first.get("real_importances")),
                _imp_len(first.get("synthetic_importances")),
                first.get("normalizer"),
                first.get("utility_protocol"),
                first.get("utility_auditors"),
                first.get("encoder"),
            )
            for ev in seed_evidences
        )
    )
    all_seeds_eligible = (
        complete
        and schema_consistent
        and all(r["eligible"] and r["score"] is not None for r in seed_reports)
    )

    retentions = [ev.get("retention") for ev in seed_evidences]
    valid_lcb = (
        utility_component([float(r) for r in retentions], bound)
        if complete and all(_finite(r) for r in retentions)
        else None
    )

    worst_drv = (
        driver_observation_v4(ordered_seeds)
        if complete and schema_consistent
        else None
    )
    repr_candidates = [
        (
            float(r["representation_closeness"]),
            ordered_seeds[i]["sample_seed"] if complete else i,
            seed_evidences[i],
        )
        for i, r in enumerate(seed_reports)
        if r.get("representation_closeness") is not None
    ]
    worst_repr = (
        min(repr_candidates, key=lambda item: (item[0], item[1]))[2]
        if complete and len(repr_candidates) == len(SAMPLE_SEEDS)
        else {}
    )

    def _finite_max(key: str) -> float | None:
        vals = [ev.get(key) for ev in seed_evidences]
        if complete and all(_finite(v) for v in vals):
            return float(max(float(v) for v in vals))
        return None

    bootstrap_vals = [ev.get("real_bootstrap_4n_retention") for ev in seed_evidences]
    if complete and all(v is not None for v in bootstrap_vals):
        bootstrap_4n = (
            float(max(float(v) for v in bootstrap_vals))
            if all(_finite(v) for v in bootstrap_vals)
            else float("nan")
        )
    elif complete and any(v is not None and not _finite(v) for v in bootstrap_vals):
        bootstrap_4n = float("nan")
    else:
        bootstrap_4n = None

    bytes_ok = complete and all(
        type(ev.get("artifact_bytes")) is int and 0 < ev["artifact_bytes"] <= BYTE_CAP
        for ev in seed_evidences
    )
    nums_ok = complete and all(
        type(ev.get("encoded_numbers")) is int and ev["encoded_numbers"] >= 1
        for ev in seed_evidences
    )
    agg_evidence = {
        **gates["values"],
        "encoder": (
            first.get("encoder")
            if complete and all(ev.get("encoder") == first.get("encoder") for ev in seed_evidences)
            else None
        ),
        "distance": worst_repr.get("distance"),
        "d_null": worst_repr.get("d_null"),
        "d_match": worst_repr.get("d_match"),
        "gap_mitra": worst_repr.get("gap_mitra"),
        "gap_tabicl": worst_repr.get("gap_tabicl"),
        "exact_row_matches": (
            0
            if complete
            and all(
                type(ev.get("exact_row_matches")) is int and ev["exact_row_matches"] == 0
                for ev in seed_evidences
            )
            else None
        ),
        "membership_auc": _max([ev.get("membership_auc") for ev in seed_evidences]),
        "attribute_inference_advantage": _max(
            [ev.get("attribute_inference_advantage") for ev in seed_evidences]
        ),
        "artifact_bytes": (
            max(ev["artifact_bytes"] for ev in seed_evidences) if bytes_ok else None
        ),
        "encoded_numbers": (
            max(ev["encoded_numbers"] for ev in seed_evidences) if nums_ok else None
        ),
        "real_rows": first.get("real_rows") if schema_consistent else None,
        "real_columns": first.get("real_columns") if schema_consistent else None,
        "normalizer": (
            first.get("normalizer")
            if complete
            and all(ev.get("normalizer") == first.get("normalizer") for ev in seed_evidences)
            else None
        ),
        "utility_protocol": (
            first.get("utility_protocol")
            if complete
            and all(
                ev.get("utility_protocol") == first.get("utility_protocol")
                for ev in seed_evidences
            )
            else None
        ),
        "utility_auditors": (
            list(first.get("utility_auditors") or [])
            if complete
            and all(
                ev.get("utility_auditors") == first.get("utility_auditors")
                for ev in seed_evidences
            )
            else []
        ),
        "retention": valid_lcb,
        "predictor_only_retention": _finite_max("predictor_only_retention"),
        "independent_marginals_retention": _finite_max("independent_marginals_retention"),
        "real_bootstrap_4n_retention": bootstrap_4n,
        "real_importances": worst_drv["real_importances"] if worst_drv is not None else None,
        "synthetic_importances": (
            worst_drv["synthetic_importances"] if worst_drv is not None else None
        ),
        "real_signs": worst_drv["real_signs"] if worst_drv is not None else None,
        "synthetic_signs": worst_drv["synthetic_signs"] if worst_drv is not None else None,
        "mfs_components": {
            "distribution_fidelity": _min(
                [(ev.get("mfs_components") or {}).get("distribution_fidelity") for ev in seed_evidences]
            ),
            "structure_fidelity": _min(
                [(ev.get("mfs_components") or {}).get("structure_fidelity") for ev in seed_evidences]
            ),
            "coverage_realism": _min(
                [(ev.get("mfs_components") or {}).get("coverage_realism") for ev in seed_evidences]
            ),
        },
    }

    report = evaluate_v4(agg_evidence)
    for seed_rep in seed_reports:
        for gate, passed in seed_rep["gates"].items():
            if not passed:
                report["gates"][gate] = False
    if not report["gates"].get("positive_predictor_lift") or not report["gates"].get(
        "positive_dependence_lift"
    ):
        report["utility_transfer"] = None
        report["knowledge_compression"] = None
        for term in ("utility_transfer", "knowledge_compression"):
            if term not in report["unavailable_terms"]:
                report["unavailable_terms"].append(term)
    report["gates"]["complete_seed_set"] = complete
    report["gates"]["lineage_schema_consistent"] = schema_consistent
    report["gates"]["all_seeds_eligible"] = all_seeds_eligible
    report["gates"]["utility_lcb_valid"] = valid_lcb is not None
    report["failed_gates"] = [key for key, passed in report["gates"].items() if not passed]
    if not (complete and schema_consistent and all_seeds_eligible and valid_lcb is not None):
        report["eligible"] = False
        report["score"] = None

    seed_variance = None
    if all_seeds_eligible and valid_lcb is not None:
        ret_vals = [float(r) for r in retentions]
        drv_vals = [float(r["driver_fidelity"]) for r in seed_reports]
        score_vals = [float(r["score"]) for r in seed_reports]
        ret_stats = _sample_mean_and_sd(ret_vals)
        drv_stats = _sample_mean_and_sd(drv_vals)
        score_stats = _sample_mean_and_sd(score_vals)
        if ret_stats is not None and drv_stats is not None and score_stats is not None:
            seed_variance = {
                "sample_seeds": list(SAMPLE_SEEDS),
                "retention_mean": ret_stats[0],
                "retention_sample_sd": ret_stats[1],
                "retention_min": min(ret_vals),
                "retention_max": max(ret_vals),
                "retention_lcb": valid_lcb,
                "driver_fidelity_mean": drv_stats[0],
                "driver_fidelity_sample_sd": drv_stats[1],
                "driver_fidelity_min": min(drv_vals),
                "driver_fidelity_max": max(drv_vals),
                "seed_score_mean": score_stats[0],
                "seed_score_sample_sd": score_stats[1],
                "seed_score_min": min(score_vals),
                "seed_score_max": max(score_vals),
            }

    return {
        **report,
        "counts_as_dope_win": False,
        "gate_evidence": gates,
        "official_tests_opened": False,
        "seed_reports": seed_reports,
        "seed_variance": seed_variance,
        "superiority": None,
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
