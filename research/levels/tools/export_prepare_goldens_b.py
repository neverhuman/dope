"""Export lane-B intermediate and end-to-end conditioning goldens for `dope_core::prepare`.

Covers both conditioning recipes defined in `research/levels`:
1. Full-report v1 (`mode == "full"`): `compute_building_block_landmarks`,
   `compute_inter_landmark_summary_stats`, `build_v3_conditioning_bundle`,
   `col_landmarks17x64`, and `effective_conditioning_digest` across full,
   partial, and empty native reports.
2. `observed-only-v1` (`mode` in `{"none", "fast", "progressive"}`):
   `observed_report_descriptors`, `compute_observed_inter_landmark_stats`,
   `build_observed_conditioning_bundle`, `col_landmarks17x64`, and
   `effective_conditioning_digest` across `none` (zero bundle),
   `partial_macro`, `partial_macro_larger`, and `full_as_observed`.

Usage (from the repository root):
    python3 -B -m research.levels.tools.export_prepare_goldens_b \
        --out engine/dope-core/tests/fixtures/prepare-v1/b/goldens.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.levels.building_block_landmarks import (
    build_v3_conditioning_bundle,
    compute_building_block_landmarks,
    compute_inter_landmark_summary_stats,
    compute_observed_inter_landmark_stats,
    zero_building_block_landmarks,
)
from research.levels.descriptors import (
    FINE_ADDED_INDICES,
    LARGER_ADDED_INDICES,
    MACRO_INDICES,
    compute_level_contrasts,
    compute_table_descriptors,
    extract_report_descriptors,
    level_landmark_mask,
    squash,
)
from research.levels.optional_landmarks import (
    CONDITIONING_DOMAIN,
    FULL_REPORT_RECIPE,
    OBSERVED_RECIPE,
    effective_conditioning_digest,
    report_task,
    squash_f32,
    table_classes,
    table_target_stats,
    unsquash,
)

SCHEMA = "dope.prepare-goldens-b/v1"
MAX_COLS = 64


def _f(x: Any) -> List[Any]:
    """Exact JSON floats (float32 values are widened exactly to float64)."""
    arr = np.asarray(x)
    return np.asarray(arr, dtype=np.float64).tolist()


def observed_target_stats(y: np.ndarray, task: str) -> List[Optional[float]]:
    if task == "regression":
        return [None, None, None]
    return table_target_stats(y, task)


def observed_report_descriptors(
    values23: Sequence[float],
    mask23: Sequence[bool],
    *,
    rows: int,
    features: int,
    classes: int,
    task: str,
    target_stats: Sequence[Optional[float]],
    level: str = "high",
) -> Dict[str, Any]:
    lvl_mask = level_landmark_mask(level)
    present = np.asarray(mask23, dtype=bool) & lvl_mask
    s = np.where(present, np.asarray(values23, dtype=np.float32).astype(np.float64), 0.0)
    raw = np.array([unsquash(v) if p else 0.0 for v, p in zip(s, present)], dtype=np.float64)

    midranks = np.zeros(23, dtype=np.float64)
    valid = np.where(present)[0]
    if valid.size > 1:
        for i in valid:
            less = float(np.sum(raw[valid] < raw[i]))
            eq = float(np.sum(raw[valid] == raw[i]))
            midranks[i] = (less + 0.5 * (eq - 1.0)) / float(valid.size - 1)
    elif valid.size == 1:
        midranks[valid[0]] = 0.5
    contrasts = compute_level_contrasts(raw, present, level=level)

    ts = list(target_stats) + [None, None]
    t_stats = squash(np.array([
        float(ts[0]) if ts[0] is not None and math.isfinite(float(ts[0])) else 0.0,
        float(ts[1]) if ts[1] is not None and math.isfinite(float(ts[1])) else 0.0,
    ], dtype=np.float64))
    rows_c, feats_c = max(int(rows), 1), max(int(features), 1)
    is_clf = 1.0 if int(classes) >= 2 or report_task(task) == "classification" else 0.0
    shape_vec = np.array([
        is_clf,
        math.log10(rows_c) / 6.0,
        math.log2(feats_c) / 12.0,
        math.log10(feats_c / float(rows_c)) / 4.0,
    ], dtype=np.float64)

    def _gmax(idxs: Sequence[int], gate: bool) -> float:
        sel = [s[i] for i in idxs if present[i]]
        return float(max(sel)) if gate and sel else 0.0

    def _gmean(idxs: Sequence[int], gate: bool) -> float:
        sel = [s[i] for i in idxs if present[i]]
        return float(np.mean(sel)) if gate and sel else 0.0

    larger_gate, fine_gate = level in ("high", "ultra"), level == "ultra"
    group = np.array([
        _gmax(MACRO_INDICES, True),
        _gmax(LARGER_ADDED_INDICES, larger_gate),
        _gmax(FINE_ADDED_INDICES, fine_gate),
        _gmean(MACRO_INDICES, True),
        _gmean(LARGER_ADDED_INDICES, larger_gate),
        _gmean(FINE_ADDED_INDICES, fine_gate),
        0.0,
        0.0,
        0.0,
    ], dtype=np.float64)
    spec_summary = np.zeros(16, dtype=np.float64)
    vector = np.concatenate([
        s, present.astype(np.float64), midranks, contrasts, t_stats, shape_vec, spec_summary, group,
    ])
    return {
        "level": level,
        "gains23": s,
        "ses23": np.zeros(23),
        "lambdas23": np.zeros(23),
        "mask23": present,
        "midranks23": midranks,
        "contrasts16": contrasts,
        "target_stats2": t_stats,
        "shape4": shape_vec,
        "spectrum_summary16": spec_summary,
        "group_summary9": group,
        "vector116": vector.astype(np.float32),
        "column_spectrum": np.zeros((feats_c, 9), dtype=np.float32),
        "column_prior_importance": np.full(feats_c, 1.0 / float(feats_c), dtype=np.float32),
    }


def build_observed_conditioning_bundle(
    values23: Sequence[float],
    mask23: Sequence[bool],
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    *,
    task: str,
    level: str = "high",
    target_stats_override: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    X = np.asarray(X_train_scaled, dtype=np.float64)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    y = np.asarray(y_train_scaled, dtype=np.float64).reshape(-1)
    n, p = int(X.shape[0]), int(X.shape[1])
    t_stats = (
        list(target_stats_override)
        if target_stats_override is not None
        else observed_target_stats(y, task)
    )
    rep_desc = observed_report_descriptors(
        values23,
        mask23,
        rows=n,
        features=p,
        classes=table_classes(y, task),
        task=task,
        target_stats=t_stats,
        level=level,
    )
    tab_desc = compute_table_descriptors(X, y, level=level)
    bb_desc = zero_building_block_landmarks(p)
    inter_vec48 = compute_observed_inter_landmark_stats(rep_desc, level=level)
    vec140 = np.concatenate([rep_desc["vector116"], tab_desc["table_vec24"]]).astype(np.float32)
    vec256 = np.concatenate([vec140, inter_vec48, bb_desc["building_block_vec68"]]).astype(np.float32)
    col_desc8 = np.asarray(tab_desc["column_desc"], dtype=np.float32)
    pc = col_desc8.shape[0]
    col_landmarks24 = np.column_stack([
        np.zeros((pc, 9), dtype=np.float32),
        col_desc8,
        np.zeros((pc, 7), dtype=np.float32),
    ]).astype(np.float32)
    emp_base = np.asarray(tab_desc["empirical_driver_weights"], dtype=np.float64)[:pc]
    rep_prior = np.asarray(rep_desc["column_prior_importance"], dtype=np.float64)
    take_p = min(pc, rep_prior.shape[0])
    rep_p = np.zeros(pc, dtype=np.float64)
    rep_p[:take_p] = rep_prior[:take_p]
    v3_drivers = 0.35 * emp_base + 0.35 * rep_p + 0.30 * np.full(pc, 1.0 / max(pc, 1), dtype=np.float64)
    v3_drivers = v3_drivers / max(float(np.sum(v3_drivers)), 1e-9)
    return {
        "vec140": np.nan_to_num(vec140, nan=0.0, posinf=0.0, neginf=0.0),
        "vec256": np.nan_to_num(vec256, nan=0.0, posinf=0.0, neginf=0.0),
        "inter_landmark_vec48": inter_vec48,
        "building_block_vec68": bb_desc["building_block_vec68"],
        "col_landmarks24": np.nan_to_num(col_landmarks24, nan=0.0, posinf=0.0, neginf=0.0),
        "v3_driver_weights": v3_drivers.astype(np.float32),
        "rep_desc": rep_desc,
        "tab_desc": tab_desc,
        "bb_desc": bb_desc,
    }


def pad_col_landmarks(col24: np.ndarray) -> np.ndarray:
    cols = np.zeros((MAX_COLS, 17), dtype=np.float32)
    take = min(MAX_COLS, col24.shape[0])
    cols[:take] = np.asarray(col24, dtype=np.float32)[:take, :17]
    return cols


def report_complete_landmarks(report: Dict[str, Any], *, include_uninformative: bool = False) -> Tuple[List[float], List[bool]]:
    values = [0.0] * 23
    mask = [False] * 23
    states = ("complete", "uninformative") if include_uninformative else ("complete",)
    for c in report.get("cells") or []:
        idx = int(c["id"]) - 1
        if 0 <= idx < 23 and c.get("state") in states:
            g = c.get("gain")
            if isinstance(g, (int, float)) and not isinstance(g, bool):
                s = squash_f32(float(g))
                if s is not None and bool(np.isfinite(s)):
                    values[idx] = float(s) + 0.0
                    mask[idx] = True
    return values, mask


def _report(rng: np.random.Generator, p: int, n: int, task: str, kind: str) -> Dict[str, Any]:
    cells = []
    for idx in range(23):
        if kind == "empty":
            break
        if kind == "partial" and rng.random() < 0.5:
            continue
        state = "complete" if rng.random() > 0.1 else "uninformative"
        gain = float(rng.normal(0.15, 0.25))
        cells.append({
            "id": idx + 1,
            "state": state,
            "gain": gain,
            "se": float(abs(rng.normal(0.03, 0.02))),
            "stable": bool(rng.random() > 0.3),
            "lambda_indices": [int(v) for v in rng.integers(0, 9, 5)],
        })
    spectrum = []
    if kind != "empty":
        for j in range(p):
            vals = [float(v) if rng.random() > 0.1 else None for v in rng.normal(0.1, 0.3, 9)]
            spectrum.append({"slot": j, "values": vals})
    return {
        "task": task,
        "classes": 3 if task == "classification" else 0,
        "rows": n,
        "features": p,
        "cells": cells,
        "target_stats": [float(rng.normal()), float(abs(rng.normal())), None],
        "spectrum": spectrum,
    }


def _table(rng: np.random.Generator, n: int, p: int, flavour: str):
    X = rng.normal(size=(n, p))
    if p >= 3:
        X[:, 1] = rng.integers(0, 3, n)            # discrete column with ties
        X[:, 2] = np.exp(rng.normal(size=n))        # skewed column
    if p >= 5:
        X[:, 4] = 1.25                              # constant column
    if flavour == "classification":
        y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.float64)
        y += (X[:, -1] > 1.0)
    elif flavour == "binary":
        y = (X[:, 0] > 0.3).astype(np.float64)
    elif flavour == "constant_y":
        y = np.full(n, 0.7)
    elif flavour == "ties_y":
        y = np.round(X[:, 0] * 2.0) / 2.0
    else:
        y = 1.4 * np.arcsinh(np.exp(X[:, 0])) + 0.8 * np.sin(1.5 * X[:, -1]) + 0.1 * rng.normal(size=n)
    return X, y


def _flavour_task(flavour: str) -> str:
    if flavour == "classification":
        return "multiclass"
    if flavour == "binary":
        return "binary"
    return "regression"


def _observed_bundles() -> List[Tuple[str, str, List[float], List[bool]]]:
    """Deterministic bundles for `none`, `partial_macro`, `partial_macro_larger`, `full_as_observed`."""
    raw_all = [0.05 * (i + 1) if i % 4 != 3 else -0.02 * (i + 1) for i in range(23)]
    sq_all = [float(squash_f32(g)) + 0.0 for g in raw_all]

    def _subset(slots: Sequence[int]) -> Tuple[List[float], List[bool]]:
        sset = set(slots)
        vals = [sq_all[i] if i in sset else 0.0 for i in range(23)]
        mask = [i in sset for i in range(23)]
        return vals, mask

    z_vals, z_mask = _subset(())
    m_vals, m_mask = _subset(MACRO_INDICES)
    ml_vals, ml_mask = _subset(tuple(MACRO_INDICES) + tuple(LARGER_ADDED_INDICES))
    f_vals, f_mask = _subset(range(23))
    return [
        ("none", "none", z_vals, z_mask),
        ("partial_macro", "progressive", m_vals, m_mask),
        ("partial_macro_larger", "progressive", ml_vals, ml_mask),
        ("full_as_observed", "progressive", f_vals, f_mask),
    ]


CASES = [
    # name, n, p, flavour, level, report-kind
    ("reg_high", 180, 10, "regression", "high", "full"),
    ("reg_ultra", 180, 10, "regression", "ultra", "full"),
    ("reg_medium", 150, 7, "regression", "medium", "partial"),
    ("reg_small", 120, 6, "regression", "small", "full"),
    ("clf3_high", 210, 8, "classification", "high", "full"),
    ("clf2_high_partial", 97, 5, "binary", "high", "partial"),
    ("ties_high", 300, 4, "ties_y", "high", "full"),
    ("empty_report_high", 160, 9, "regression", "high", "empty"),
    ("p1_high", 64, 1, "regression", "high", "full"),
    ("p2_high", 40, 2, "binary", "high", "partial"),
    ("const_y_high", 50, 6, "constant_y", "high", "full"),
    ("tiny_n_high", 5, 3, "regression", "high", "full"),
    ("wide_high", 30, 70, "regression", "high", "full"),
    ("big_n_high", 1100, 3, "regression", "high", "partial"),
]


def build() -> Dict[str, Any]:
    rng = np.random.default_rng(20261027)
    obs_specs = _observed_bundles()
    out_cases = []
    for name, n, p, flavour, level, rkind in CASES:
        X, y = _table(rng, n, p, flavour)
        task = _flavour_task(flavour)
        report = _report(
            rng, p, n, "classification" if flavour in ("classification", "binary") else "regression", rkind
        )
        bb = compute_building_block_landmarks(X, y, level=level)
        rep = extract_report_descriptors(report, level=level)
        inter = compute_inter_landmark_summary_stats(rep, {}, bb, level=level)
        bundle = build_v3_conditioning_bundle(report, X, y, level=level)
        tab = bundle["tab_desc"]
        assert np.array_equal(inter, bundle["inter_landmark_vec48"])
        cols17x64 = pad_col_landmarks(bundle["col_landmarks24"])
        legacy_digest = effective_conditioning_digest(
            bundle["vec256"], cols17x64, recipe=FULL_REPORT_RECIPE, level=level
        )
        comp_vals, comp_mask = report_complete_landmarks(report, include_uninformative=False)
        obs_rep_vals, obs_rep_mask = report_complete_landmarks(report, include_uninformative=True)
        raw_ts = table_target_stats(y, task)
        raw_ts2 = [
            float(raw_ts[0]) if raw_ts[0] is not None and math.isfinite(float(raw_ts[0])) else 0.0,
            float(raw_ts[1]) if raw_ts[1] is not None and math.isfinite(float(raw_ts[1])) else 0.0,
        ]

        obs_entries = []
        for variant, mode, b_vals, b_mask in obs_specs:
            obs = build_observed_conditioning_bundle(b_vals, b_mask, X, y, task=task, level=level)
            obs_cols17x64 = pad_col_landmarks(obs["col_landmarks24"])
            obs_digest = effective_conditioning_digest(
                obs["vec256"], obs_cols17x64, recipe=OBSERVED_RECIPE, level=level
            )
            obs_raw_ts = build_observed_conditioning_bundle(
                b_vals, b_mask, X, y, task=task, level=level,
                target_stats_override=[raw_ts2[0], raw_ts2[1], None],
            )
            obs_raw_ts_digest = effective_conditioning_digest(
                obs_raw_ts["vec256"], obs_cols17x64, recipe=OBSERVED_RECIPE, level=level
            )
            obs_entries.append({
                "variant": variant,
                "mode": mode,
                "recipe": OBSERVED_RECIPE,
                "level": level,
                "task": task,
                "bundle_values23": _f(b_vals),
                "bundle_mask23": [bool(v) for v in b_mask],
                "rep_vector116": _f(obs["rep_desc"]["vector116"]),
                "inter48": _f(obs["inter_landmark_vec48"]),
                "vec256": _f(obs["vec256"]),
                "vec256_with_raw_target_stats": _f(obs_raw_ts["vec256"]),
                "col_landmarks24": _f(obs["col_landmarks24"].reshape(-1)),
                "col_landmarks17x64": _f(obs_cols17x64.reshape(-1)),
                "v3_driver_weights": _f(obs["v3_driver_weights"]),
                "conditioning_digest": obs_digest,
                "conditioning_digest_with_raw_target_stats": obs_raw_ts_digest,
            })

        out_cases.append({
            "name": name,
            "level": level,
            "task": task,
            "recipe": FULL_REPORT_RECIPE,
            "n": n,
            "p": p,
            "x": _f(X.reshape(-1)),
            "y": _f(y),
            "raw_target_stats2": _f(raw_ts2),
            "bb": {k: _f(np.asarray(v).reshape(-1)) for k, v in bb.items()},
            "rep": {
                "gains23": _f(rep["gains23"]),
                "mask23": [bool(v) for v in rep["mask23"]],
                "ses23": _f(rep["ses23"]),
                "lambdas23": _f(rep["lambdas23"]),
                "column_spectrum": _f(np.asarray(rep["column_spectrum"]).reshape(-1)),
                "column_prior_importance": _f(rep["column_prior_importance"]),
                "vector116": _f(rep["vector116"]),
                "complete_values23": _f(comp_vals),
                "complete_mask23": [bool(v) for v in comp_mask],
                "observed_values23": _f(obs_rep_vals),
                "observed_mask23": [bool(v) for v in obs_rep_mask],
            },
            "tab": {
                "table_vec24": _f(tab["table_vec24"]),
                "column_desc": _f(np.asarray(tab["column_desc"]).reshape(-1)),
                "empirical_driver_weights": _f(tab["empirical_driver_weights"]),
            },
            "inter48": _f(inter),
            "vec256": _f(bundle["vec256"]),
            "col_landmarks24": _f(bundle["col_landmarks24"].reshape(-1)),
            "col_landmarks17x64": _f(cols17x64.reshape(-1)),
            "v3_driver_weights": _f(bundle["v3_driver_weights"]),
            "conditioning_digest": legacy_digest,
            "observed": obs_entries,
        })
    return {"schema": SCHEMA, "numpy": np.__version__, "cases": out_cases}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = build()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {len(data['cases'])} cases to {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
