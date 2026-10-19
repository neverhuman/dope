#!/usr/bin/env python3
"""Authenticated historical foundation-model figure and table data (Q01.FIG-0).

Traces and recomputes the 97-lineage MFS-v3 foundation-model block from
committed SHA-pinned receipts without hard-coding runtime display constants.
Separates the historical 94-lineage feature-target correlation proxy
(`rust/certification/privacy_attacks.rs::driver_fidelity`) from cross-model
feature-driver preservation (`D80`), which remains `unmeasured`, and verifies
the exact reducer provenance of the `0.975` (median of per-lineage seed medians)
versus `0.923` (`0.923106`, median of per-lineage seed minima under the legacy
`_min` reducer; pooled-seed median is `0.973214`) values.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _dir in (ROOT, ROOT / "docs/whitepaper/scripts"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import compute_panel  # noqa: E402

MFS_RECEIPT_REL = "research/benchmark/results/mfs-v3-density-panel.json"
V2_HEADLINE_REL = "docs/whitepaper/generated/v2-headline.json"
V2_PANEL_REL = "research/benchmark/results/review-fixes-v2/panel.json"

PINNED_INPUT_SHA256: dict[str, str] = {
    MFS_RECEIPT_REL: "d2cbbb5c08d045c161e3809e150efeacafadcfbee5e2accde6ed39cc46fe9c25",
    V2_HEADLINE_REL: "44cc100e1ea1eecf9ea3f2e39f83ddcf63b1edb8d23dbb002c4973c57f3c4833",
    V2_PANEL_REL: "049e81158c6d2265f52a474bdd4532a09be3c3d734f034a36ecc7de2bbb610d5",
}

FOUNDATION_METHODS: tuple[str, ...] = (
    "DOPE",
    "GaussianCopula",
    "Chow-Liu",
    "independent_marginals",
)
FOUNDATION_COMPARATORS: tuple[str, ...] = (
    "GaussianCopula",
    "Chow-Liu",
    "independent_marginals",
)
FOUNDATION_AUDITORS: tuple[tuple[str, str], ...] = (
    ("kumo_tabular_l", "Kumo Large"),
    ("tabicl2", "TabICLv2"),
    ("mitra_v2", "Mitra-v2"),
)
METHOD_DISPLAY: dict[str, str] = {
    "DOPE": "DOPE (ours)",
    "GaussianCopula": "Gaussian copula",
    "Chow-Liu": "Chow-Liu tree",
    "independent_marginals": "Independent marginals",
}
HERO_SHORT_TITLE: str = "More predictive gain. About one-third the state."
EXPECTED_SAMPLE_SEEDS: tuple[int, ...] = (101, 211, 307)
EXPECTED_FIT_SEED: int = 11
EXPECTED_SYNTHETIC_CONTEXT: str = "4n"
EXPECTED_LINEAGE_COUNT: int = 97
DRIVER_PROXY_PRODUCER: str = "rust/certification/privacy_attacks.rs::driver_fidelity"
DRIVER_PROXY_KIND: str = "absolute_target_correlation"


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def authenticate_source(rel_path: str, root: Path = ROOT) -> tuple[dict, str]:
    """Authenticate a pinned upstream JSON input before parsing."""
    if rel_path not in PINNED_INPUT_SHA256:
        raise ValueError(f"unregistered foundation source path: {rel_path}")
    path = root / rel_path
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"symlinked or out-of-tree foundation input: {rel_path}")
    if not path.is_file():
        raise ValueError(f"missing foundation input: {rel_path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINNED_INPUT_SHA256[rel_path]:
        raise ValueError(f"foundation input sha256 mismatch for {rel_path}")
    return json.loads(raw), digest


def _quantiles(sorted_vals: list[float]) -> dict[str, float | int]:
    if not sorted_vals:
        raise ValueError("cannot summarize empty value list")
    n = len(sorted_vals)
    med = float(statistics.median(sorted_vals))
    q1 = float(sorted_vals[max(0, int(round(0.25 * (n - 1))))])
    q3 = float(sorted_vals[min(n - 1, int(round(0.75 * (n - 1))))])
    return {"n": n, "median": med, "q1": q1, "q3": q3}


def validate_and_extract_mfs_receipt(receipt: dict) -> dict:
    """Validate cohort membership, sample seeds, charged bytes, and retention cells."""
    if receipt.get("format") != "dope-mfs-v3-density-panel":
        raise ValueError("unexpected foundation receipt format")
    if receipt.get("counts_as_dope_win") is not False or receipt.get("official_tests_opened") is not False:
        raise ValueError("foundation receipt must not claim a win or opened official tests")
    if receipt.get("superiority") is not None:
        raise ValueError("foundation receipt superiority must be null")

    lineages = receipt.get("lineages")
    if not isinstance(lineages, list) or not lineages:
        raise ValueError("foundation receipt lineages must be a non-empty list")

    auditor_keys = {aud_id for aud_id, _ in FOUNDATION_AUDITORS}
    seen_identities: set[tuple[str, str]] = set()
    cohorts_by_method: dict[str, set[str]] = {m: set() for m in FOUNDATION_METHODS}
    retention_maps: dict[str, dict[str, dict[str, float]]] = {
        aud_id: {m: {} for m in FOUNDATION_METHODS} for aud_id in auditor_keys
    }
    bytes_by_method: dict[str, dict[str, float]] = {m: {} for m in FOUNDATION_METHODS}
    proxy_seed_medians: dict[str, dict[str, float]] = {m: {} for m in FOUNDATION_METHODS}
    proxy_seed_mins: dict[str, dict[str, float]] = {m: {} for m in FOUNDATION_METHODS}
    proxy_pooled_seeds: dict[str, list[float]] = {m: [] for m in FOUNDATION_METHODS}
    proxy_missing_by_method: dict[str, set[str]] = {m: set() for m in FOUNDATION_METHODS}

    for row in lineages:
        method = row.get("method")
        dataset = row.get("dataset")
        if method not in FOUNDATION_METHODS or not isinstance(dataset, str) or not dataset:
            raise ValueError("invalid method or dataset identity in foundation receipt")
        ident = (method, dataset)
        if ident in seen_identities:
            raise ValueError(f"duplicate method-lineage cell: {ident}")
        seen_identities.add(ident)
        cohorts_by_method[method].add(dataset)

        if row.get("counts_as_dope_win") is not False or row.get("official_tests_opened") is not False:
            raise ValueError("lineage row claims a win or opened official tests")
        if row.get("superiority") is not None or row.get("score") is not None:
            raise ValueError("lineage row claims non-null superiority or MFS-v3 release score")

        ret = row.get("retention") or {}
        if set(ret) != auditor_keys:
            raise ValueError(f"lineage {ident} auditor keys {set(ret)} != {auditor_keys}")
        for aud_id, val in ret.items():
            if not _finite_number(val):
                raise ValueError(f"non-finite retention for {ident} under {aud_id}")
            retention_maps[aud_id][method][dataset] = float(val)

        seeds = row.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != len(EXPECTED_SAMPLE_SEEDS):
            raise ValueError(f"lineage {ident} does not have {len(EXPECTED_SAMPLE_SEEDS)} sample seeds")
        observed_seeds = tuple(sorted(int(s.get("sample_seed", -1)) for s in seeds))
        if observed_seeds != EXPECTED_SAMPLE_SEEDS:
            raise ValueError(f"lineage {ident} sample seeds {observed_seeds} != {EXPECTED_SAMPLE_SEEDS}")

        seed_bytes: list[int] = []
        seed_proxies: list[float] = []
        for s in seeds:
            b_val = s.get("artifact_bytes")
            if type(b_val) is not int or b_val <= 0:
                raise ValueError(f"lineage {ident} has invalid seed artifact_bytes: {b_val!r}")
            seed_bytes.append(b_val)
            holdout = s.get("holdout") or {}
            d_val = holdout.get("driver_fidelity")
            if d_val is not None:
                if not _finite_number(d_val) or not (0.0 <= float(d_val) <= 1.0):
                    raise ValueError(f"lineage {ident} has invalid driver_fidelity proxy: {d_val!r}")
                seed_proxies.append(float(d_val))
        if len(set(seed_bytes)) != 1:
            raise ValueError(f"lineage {ident} has inconsistent artifact_bytes across sample seeds: {seed_bytes}")
        bytes_by_method[method][dataset] = float(seed_bytes[0])
        if seed_proxies:
            if len(seed_proxies) != len(EXPECTED_SAMPLE_SEEDS):
                raise ValueError(f"lineage {ident} has partial seed driver_fidelity proxy values")
            proxy_seed_medians[method][dataset] = float(statistics.median(seed_proxies))
            proxy_seed_mins[method][dataset] = float(min(seed_proxies))
            proxy_pooled_seeds[method].extend(seed_proxies)
        else:
            proxy_missing_by_method[method].add(dataset)

    expected_cohort = cohorts_by_method["DOPE"]
    if len(expected_cohort) != EXPECTED_LINEAGE_COUNT:
        raise ValueError(f"DOPE lineage count {len(expected_cohort)} != {EXPECTED_LINEAGE_COUNT}")
    for method, cohort in cohorts_by_method.items():
        if cohort != expected_cohort:
            raise ValueError(f"method {method} cohort does not match DOPE cohort")
    missing_ref = proxy_missing_by_method["DOPE"]
    for method, missing_set in proxy_missing_by_method.items():
        if missing_set != missing_ref:
            raise ValueError(f"method {method} missing driver proxy set disagrees with DOPE")

    return {
        "cohort_datasets": tuple(sorted(expected_cohort)),
        "retention_maps": retention_maps,
        "bytes_by_method": bytes_by_method,
        "proxy_seed_medians": proxy_seed_medians,
        "proxy_seed_mins": proxy_seed_mins,
        "proxy_pooled_seeds": proxy_pooled_seeds,
        "proxy_missing_datasets": tuple(sorted(missing_ref)),
    }


_FOUNDATION_SUMMARY_CACHE: dict[tuple[str, str, str, int], dict] = {}


def _bootstrap_interval(
    values, *, level_pct: int = 95, salt: int = 0, estimand: str = "lineage_median_null_normalized_retention"
) -> dict:
    """Lineage-percentile bootstrap interval with explicit interval provenance metadata."""
    if type(level_pct) is not int or not (50 <= level_pct <= 99):
        raise ValueError(f"unsupported interval level_pct: {level_pct!r}")
    tail = round((100 - level_pct) / 200.0, 6)
    quantiles = [tail, round(1.0 - tail, 6)]
    meta = {
        "level_pct": level_pct,
        "quantiles": quantiles,
        "method": "nonparametric_lineage_percentile_bootstrap",
        "estimand": estimand,
        "resampling_unit": "lineage",
        "draws": int(compute_panel.DRAWS),
        "seed": int(compute_panel.SEED),
        "cohort_id": "mfs-v3-foundation-97",
    }
    array = np.asarray(list(values), dtype=float)
    array = np.sort(array[np.isfinite(array)])
    n = int(array.size)
    if n == 0:
        return {"n": 0, "median": None, "lo": None, "hi": None, "interval_metadata": meta}
    median = float(np.median(array))
    if n == 1:
        return {"n": 1, "median": median, "lo": median, "hi": median, "interval_metadata": meta}
    local = compute_panel._generator(array, salt)
    index = local.integers(0, n, size=(compute_panel.DRAWS, n))
    samples = np.median(array[index], axis=1)
    lo, hi = np.quantile(samples, quantiles)
    return {"n": n, "median": median, "lo": float(lo), "hi": float(hi), "interval_metadata": meta}


def load_foundation_summary(root: Path = ROOT, *, interval_level_pct: int = 95) -> dict:
    """Authenticate receipts and recompute the canonical historical foundation block."""
    receipt, mfs_sha = authenticate_source(MFS_RECEIPT_REL, root=root)
    v2_headline, v2_headline_sha = authenticate_source(V2_HEADLINE_REL, root=root)
    _, v2_panel_sha = authenticate_source(V2_PANEL_REL, root=root)

    cache_key = (mfs_sha, v2_headline_sha, v2_panel_sha, int(interval_level_pct))
    if cache_key in _FOUNDATION_SUMMARY_CACHE:
        return copy.deepcopy(_FOUNDATION_SUMMARY_CACHE[cache_key])

    extracted = validate_and_extract_mfs_receipt(receipt)
    stored_summary = receipt.get("retention_summary") or {}
    top_meta = _bootstrap_interval([0.5, 0.5], level_pct=interval_level_pct)["interval_metadata"]

    auditors_out: dict[str, dict] = {}
    for aud_id, aud_label in FOUNDATION_AUDITORS:
        maps = extracted["retention_maps"][aud_id]
        methods_summary: dict[str, dict] = {}
        for method in FOUNDATION_METHODS:
            summary = _bootstrap_interval(maps[method].values(), level_pct=interval_level_pct, salt=0)
            stored_m = ((stored_summary.get(aud_id) or {}).get("methods") or {}).get(method) or {}
            if stored_m.get("median") is not None and abs(float(stored_m["median"]) - summary["median"]) > 1e-9:
                raise ValueError(f"recomputed {aud_id} {method} median disagrees with stored receipt")
            if interval_level_pct == 95 and stored_m.get("lo") is not None:
                if abs(float(stored_m["lo"]) - summary["lo"]) > 1e-9 or abs(float(stored_m["hi"]) - summary["hi"]) > 1e-9:
                    raise ValueError(f"recomputed {aud_id} {method} 95% endpoints disagree with stored receipt")
            methods_summary[method] = summary

        pairs_summary: dict[str, dict] = {}
        for comp in FOUNDATION_COMPARATORS:
            pair = compute_panel.paired_test(maps["DOPE"], maps[comp], None, len(FOUNDATION_COMPARATORS))
            keys = sorted(set(maps["DOPE"]) & set(maps[comp]))
            left_b = _bootstrap_interval([maps["DOPE"][k] for k in keys], level_pct=interval_level_pct, salt=0)
            right_b = _bootstrap_interval([maps[comp][k] for k in keys], level_pct=interval_level_pct, salt=0)
            diff_b = _bootstrap_interval(
                [maps["DOPE"][k] - maps[comp][k] for k in keys],
                level_pct=interval_level_pct,
                salt=1,
                estimand="paired_lineage_retention_difference_median",
            )
            stored_p = ((stored_summary.get(aud_id) or {}).get("pairs") or {}).get(comp) or {}
            if (
                stored_p.get("median_difference") is not None
                and abs(float(stored_p["median_difference"]) - diff_b["median"]) > 1e-9
            ):
                raise ValueError(f"recomputed {aud_id} DOPE-{comp} paired difference disagrees with receipt")
            pairs_summary[comp] = {
                "n": pair["n"],
                "dope_median": left_b["median"],
                "dope_lo": left_b["lo"],
                "dope_hi": left_b["hi"],
                "other_median": right_b["median"],
                "other_lo": right_b["lo"],
                "other_hi": right_b["hi"],
                "median_difference": diff_b["median"],
                "lo": diff_b["lo"],
                "hi": diff_b["hi"],
                "interval_metadata": diff_b["interval_metadata"],
                "wins": pair["wins"],
                "ties": pair["ties"],
                "losses": pair["losses"],
                "wilcoxon_p": pair["wilcoxon_p"],
            }
        compute_panel.attach_holm(pairs_summary, "wilcoxon_p", "holm_p")
        auditors_out[aud_id] = {
            "id": aud_id,
            "display": aud_label,
            "methods": methods_summary,
            "pairs": pairs_summary,
        }

    matched_bytes: dict[str, dict] = {}
    for method in FOUNDATION_METHODS:
        vals = sorted(extracted["bytes_by_method"][method].values())
        matched_bytes[method] = _quantiles(vals)

    code_map = {"DOPE": "Dope", "GaussianCopula": "Gauss", "Chow-Liu": "Chow", "independent_marginals": "Ind"}
    v2_arms = {arm["code"]: arm for arm in v2_headline.get("arms", ())}
    classical_bytes: dict[str, dict] = {}
    for method, code in code_map.items():
        b_info = (v2_arms.get(code) or {}).get("bytes") or {}
        classical_bytes[method] = {
            "n": b_info.get("n"),
            "median": b_info.get("median"),
            "q1": b_info.get("q1"),
            "q3": b_info.get("q3"),
            "cohort_id": "legacy-v2-catboost-99",
        }

    correlation_proxy: dict[str, dict] = {}
    for method in FOUNDATION_METHODS:
        med_vals = sorted(extracted["proxy_seed_medians"][method].values())
        min_vals = sorted(extracted["proxy_seed_mins"][method].values())
        pooled_vals = sorted(extracted["proxy_pooled_seeds"][method])
        q = _quantiles(med_vals)
        correlation_proxy[method] = {
            **q,
            "median_of_seed_mins": float(statistics.median(min_vals)),
            "pooled_seed_median": float(statistics.median(pooled_vals)),
            "missing_lineages": len(extracted["proxy_missing_datasets"]),
            "producer": DRIVER_PROXY_PRODUCER,
            "proxy_kind": DRIVER_PROXY_KIND,
            "is_cross_model_driver_agreement": False,
            "cross_model_driver_preservation": "unmeasured",
        }

    strongest_by_auditor = {
        aud_id: max(FOUNDATION_COMPARATORS, key=lambda c: auditors_out[aud_id]["methods"][c]["median"])
        for aud_id, _ in FOUNDATION_AUDITORS
    }
    if set(strongest_by_auditor.values()) != {"GaussianCopula"}:
        raise ValueError(f"unexpected strongest foundation comparator: {strongest_by_auditor}")

    gc_proxy = correlation_proxy["GaussianCopula"]
    result = {
        "format": "dope-historical-foundation-v1",
        "hero_title": HERO_SHORT_TITLE,
        "evidence_status": "selection_exposed_validation",
        "qualification_state": "specified_unqualified",
        "pretraining_overlap_state": "unknown",
        "cross_model_driver_preservation": "unmeasured",
        "interval_metadata": top_meta,
        "cohort": {
            "id": "mfs-v3-foundation-97",
            "n_lineages": EXPECTED_LINEAGE_COUNT,
            "fit_seed": EXPECTED_FIT_SEED,
            "sample_seeds": list(EXPECTED_SAMPLE_SEEDS),
            "synthetic_context": EXPECTED_SYNTHETIC_CONTEXT,
            "lineage_composition": "historical mixed/simulated PMLB lineages (not 97 independent real-world datasets)",
        },
        "sources": {
            MFS_RECEIPT_REL: mfs_sha,
            V2_HEADLINE_REL: v2_headline_sha,
            V2_PANEL_REL: v2_panel_sha,
        },
        "strongest_foundation_comparator": {
            "method": "GaussianCopula",
            "display": METHOD_DISPLAY["GaussianCopula"],
            "scope_note": (
                "Gaussian copula is the strongest measured comparator on this 97-lineage "
                "historical foundation block, not universally strongest (TabSyn leads the "
                "99-lineage classical CatBoost validation panel at 42,261,045.5 B)."
            ),
        },
        "auditors": auditors_out,
        "matched_charged_bytes": matched_bytes,
        "classical_v2_charged_bytes": classical_bytes,
        "target_correlation_proxy": correlation_proxy,
        "discrepancy_audit": {
            "gaussian_copula_median_of_seed_medians": gc_proxy["median"],
            "gaussian_copula_median_of_seed_mins": gc_proxy["median_of_seed_mins"],
            "gaussian_copula_pooled_seed_median": gc_proxy["pooled_seed_median"],
            "resolution": (
                "0.975 is the 94-lineage median of per-lineage medians across three seeds; "
                "0.923 (0.923106) is the 94-lineage median of per-lineage minimum seed values "
                "(matching historical assembler::_min, not a pooled-seed median, which is 0.973214). "
                "All are absolute feature-target association proxies from "
                "rust/certification/privacy_attacks.rs::driver_fidelity "
                "(absolute_target_correlation); cross-model Kumo/TabICL/Mitra feature-driver "
                "preservation remains unmeasured."
            ),
        },
    }
    _FOUNDATION_SUMMARY_CACHE[cache_key] = copy.deepcopy(result)
    return result


def hero_six_bar_rows(summary: dict | None = None) -> list[dict]:
    """Return the six canonical horizontal bar rows (DOPE vs Gaussian copula across 3 TFMs)."""
    summary = load_foundation_summary() if summary is None else summary
    rows: list[dict] = []
    for aud_id, aud_label in FOUNDATION_AUDITORS:
        aud_block = summary["auditors"][aud_id]
        pair = aud_block["pairs"]["GaussianCopula"]
        for method in ("DOPE", "GaussianCopula"):
            m_stats = aud_block["methods"][method]
            b_stats = summary["matched_charged_bytes"][method]
            rows.append({
                "auditor_id": aud_id,
                "auditor_display": aud_label,
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "n_lineages": int(m_stats["n"]),
                "retention": float(m_stats["median"]),
                "interval_lo": float(m_stats["lo"]),
                "interval_hi": float(m_stats["hi"]),
                "retention_pct": 100.0 * float(m_stats["median"]),
                "interval_lo_pct": 100.0 * float(m_stats["lo"]),
                "interval_hi_pct": 100.0 * float(m_stats["hi"]),
                "interval_metadata": copy.deepcopy(m_stats["interval_metadata"]),
                "matched_artifact_bytes": int(round(float(b_stats["median"]))),
                "paired_median_difference": float(pair["median_difference"]),
                "paired_lo": float(pair["lo"]),
                "paired_hi": float(pair["hi"]),
                "evidence_status": summary["evidence_status"],
                "cross_model_driver_preservation": summary["cross_model_driver_preservation"],
            })
    return rows


def foundation_table_rows(summary: dict | None = None) -> list[dict]:
    """Return the four-method canonical supporting table rows for the 97-lineage foundation block."""
    summary = load_foundation_summary() if summary is None else summary
    rows: list[dict] = []
    for method in FOUNDATION_METHODS:
        b_stats = summary["matched_charged_bytes"][method]
        proxy = summary["target_correlation_proxy"][method]
        aud_cells = {
            aud_id: summary["auditors"][aud_id]["methods"][method]
            for aud_id, _ in FOUNDATION_AUDITORS
        }
        rows.append({
            "method": method,
            "display": METHOD_DISPLAY[method],
            "n_lineages": int(b_stats["n"]),
            "kumo_tabular_l": aud_cells["kumo_tabular_l"],
            "tabicl2": aud_cells["tabicl2"],
            "mitra_v2": aud_cells["mitra_v2"],
            "matched_artifact_bytes": int(round(float(b_stats["median"]))),
            "matched_bytes_q1": float(b_stats["q1"]),
            "matched_bytes_q3": float(b_stats["q3"]),
            "target_correlation_proxy": proxy,
            "cross_model_driver_preservation": "unmeasured",
            "evidence_status": summary["evidence_status"],
        })
    return rows
