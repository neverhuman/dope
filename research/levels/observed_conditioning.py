"""Observed-only conditioning, stratified subsampling, and progressive landmark driver."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .descriptors import (
    FINE_ADDED_INDICES as FINE, LANDMARK_NAMES, LARGER_ADDED_INDICES as LARGER,
    MACRO_INDICES as MACRO, compute_level_contrasts, compute_table_descriptors,
    level_landmark_mask, squash,
)
from .optional_landmarks import (
    DEFAULT_PROGRESSIVE_SUBSAMPLE, LANDMARK_MODES, PANEL, PROGRESSIVE_GROUPS,
    LandmarkBundle, report_task, squash_f32, table_classes, table_target_stats, unsquash,
)

LEVEL_RANK = {"low": 0, "small": 0, "medium": 1, "high": 2, "ultra": 3}
MAX_COLS = 64
OBSERVED_RECIPE = "observed-only-v1"
FULL_REPORT_RECIPE = "full-report-v1"
CONDITIONING_DOMAIN = b"dope.conditioning/v1\0"
MLC_V3_TRAINING_WORK_BUDGET = 2_000_000_000_000
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _squash(x: Any) -> Any:
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * (np.abs(arr) / (1.0 + np.abs(arr)))


def compute_observed_inter_landmark_stats(rep_desc: Dict[str, Any], *, level: str = "ultra") -> np.ndarray:
    """Observed-only 48 inter-landmark stats (optional none/partial path); full-report builder untouched."""
    lvl_r = LEVEL_RANK.get(level, 3)
    g, m = np.asarray(rep_desc["gains23"], dtype=np.float64), np.asarray(rep_desc["mask23"], dtype=bool)
    mp = lambda idxs: max([float(g[i]) for i in idxs if m[i]], default=None)  # noqa: E731
    clip1 = lambda v: float(np.clip(v, -1.0, 1.0))  # noqa: E731
    active = g[m]
    s_max = max(float(np.max(active)) if active.size > 0 else 0.0, 0.02)
    fam_idx = [[0, 1, 2, 4], [3, 5, 9], [6, 7, 15, 16, 17] if lvl_r >= 1 else [], [8, 12] if lvl_r >= 2 else [], [10, 11, 18, 19, 20] if lvl_r >= 1 else []]
    fam_vec, fam_has = np.zeros(20, dtype=np.float64), [False] * 5
    for f_idx, idxs in enumerate(fam_idx):
        pos = np.array([g[i] for i in idxs if m[i] and math.isfinite(g[i]) and abs(g[i]) > 1e-9], dtype=np.float64)
        if pos.size > 0:
            fam_has[f_idx] = True
            f_max, f_mean, f_std = float(np.max(pos)), float(np.mean(pos)), float(np.std(pos))
            fam_vec[4 * f_idx : 4 * f_idx + 4] = [clip1(f_max), clip1(f_mean), float(np.clip(f_std / s_max, 0.0, 1.0)), clip1(f_max / s_max)]
    r = np.zeros(12, dtype=np.float64)
    lin = max(float(g[0]), 0.0) if m[0] else None
    b1, tail = mp([0, 14]), mp([1, 2])
    best1 = max(b1, 0.01) if b1 is not None else None
    if lin is not None and tail is not None:
        r[0] = clip1((tail - lin) / s_max)
    if lvl_r >= 2:
        sharp, smooth = mp([7, 15]), mp([4])
        if sharp is not None and smooth is not None:
            r[1] = clip1((sharp - smooth) / s_max)
        if m[15] and m[16]:
            r[2] = clip1((g[16] - g[15]) / s_max)
        if m[11] and best1 is not None:
            r[4] = clip1((g[11] - best1) / s_max)
        if m[8] and fam_has[0]:
            r[7] = clip1((g[8] - fam_vec[0]) / s_max)
    if lvl_r >= 3 and m[16] and m[17]:
        r[3] = clip1((g[17] - g[16]) / s_max)
    if m[10] and best1 is not None:
        r[5] = clip1((g[10] - best1) / s_max)
    if m[10] and lin is not None:
        r[6] = float(_squash(float(g[10]) / (lin + 0.02)))
    if lvl_r >= 3:
        if m[20] and m[17]:
            r[8] = math.tanh(math.asinh(max(float(g[20]), 0.0) / (max(float(g[17]), 0.0) + 0.02)) - math.asinh(1.0))
        if m[18] and m[19]:
            r[9] = clip1((g[18] - g[19]) / s_max)
        if m[0] and m[21] and m[17] and m[22]:
            d_lin, d_gbdt = float(g[0] - g[21]), float(g[17] - g[22])
            r[10] = clip1((d_gbdt - d_lin) / (max(abs(d_gbdt), abs(d_lin)) + 0.02))
    elif m[0] and m[21]:
        r[10] = clip1(float(g[0] - g[21]) / s_max)
    if active.size > 1:
        logits = 8.0 * active / s_max
        logits = logits - np.max(logits)
        probs = np.exp(logits) / max(float(np.sum(np.exp(logits))), 1e-12)
        r[11] = float(np.clip(-float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) / max(math.log(probs.size), 1e-9), 0.0, 1.0))
    return np.nan_to_num(np.concatenate([fam_vec, r, np.zeros(8), np.zeros(8)]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def bundle_to_report(
    bundle: LandmarkBundle, *, rows: int, features: int, task: str, classes: int,
    base_report: Optional[Dict[str, Any]] = None,
    target_stats: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    """Native-shaped report for the optional conditioning path."""
    bundle.validate()
    rows, features = int(rows), int(features)
    if base_report is not None:
        base_obs = LandmarkBundle.from_report(base_report, bundle.subsample_n, include_uninformative=True)
        base_comp = LandmarkBundle.from_report(base_report, bundle.subsample_n, complete_only=True)
        if (bundle.is_full() and (base_obs.same_landmarks(bundle) or base_comp.same_landmarks(bundle))
                and base_report.get("rows") == rows and base_report.get("features") == features):
            return base_report
    base_cells: Dict[int, Dict[str, Any]] = {}
    if base_report is not None:
        for cell in base_report.get("cells") or []:
            try:
                base_cells[int(cell.get("id"))] = cell
            except (TypeError, ValueError):
                continue
    cells = []
    for i, (v, m) in enumerate(zip(bundle.values, bundle.mask)):
        if not m:
            continue
        native = base_cells.get(i + 1)
        if native is not None and native.get("state") in ("complete", "uninformative"):
            s = squash_f32(native.get("gain")) if isinstance(native.get("gain"), (int, float)) else None
            if s is not None and float(s) == float(v):
                cells.append(dict(native))
                continue
        cells.append({"id": i + 1, "name": LANDMARK_NAMES[i], "state": "complete", "gain": unsquash(v),
                      "se": None, "stable": False, "lambda_indices": [], "train_rows": [], "query_rows": 0})
    t_stats = (
        list(base_report["target_stats"]) if (base_report is not None and base_report.get("target_stats") is not None)
        else list(target_stats) if target_stats is not None else [None, None, None]
    )
    report: Dict[str, Any] = {
        "panel": PANEL, "kind": "real", "privacy": "private", "task": report_task(task),
        "classes": int(classes), "rows": rows, "features": features, "cells": cells,
        "target_stats": t_stats,
        "spectrum": list(base_report.get("spectrum") or []) if base_report is not None else [],
    }
    if base_report is not None and "options" in base_report:
        report["options"] = base_report["options"]
    return report


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
        return z ^ (z >> np.uint64(31))


def _row_keys(n_rows: int, seed: int, row_ids: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    s = _splitmix64(np.asarray([int(seed) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64))[0]
    if row_ids is None:
        rids = np.arange(n_rows, dtype=np.uint64)
    else:
        if len(row_ids) != n_rows:
            raise ValueError("subsample_row_ids_shape")
        parsed: List[int] = []
        for r in row_ids:
            if isinstance(r, bool) or not isinstance(r, (int, np.integer)) or int(r) < 0:
                raise ValueError("subsample_row_ids_invalid")
            parsed.append(int(r))
        if len(set(parsed)) != n_rows:
            raise ValueError("subsample_row_ids_duplicate")
        rids = np.asarray([r & 0xFFFFFFFFFFFFFFFF for r in parsed], dtype=np.uint64)
    return _splitmix64(rids ^ s), rids


def _largest_remainder(total: int, weights: Sequence[int]) -> List[int]:
    w_sum = int(sum(weights))
    if w_sum <= 0:
        return [0] * len(weights)
    base = [total * int(w) // w_sum for w in weights]
    rem = [total * int(w) % w_sum for w in weights]
    for i in sorted(range(len(weights)), key=lambda i: (-rem[i], i))[: total - sum(base)]:
        base[i] += 1
    return base


def stratified_subsample(
    y: Sequence[float], task: str, n: int, seed: int = 0, row_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Deterministic stratified row subsample; returns sorted int64 indices."""
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    rows = int(arr.size)
    if type(n) is not int or n < 1:
        raise ValueError("subsample_n_invalid")
    if not np.all(np.isfinite(arr)):
        raise ValueError("subsample_target_nonfinite")
    keys, rids = _row_keys(rows, seed, row_ids=row_ids)
    if n >= rows:
        return np.arange(rows, dtype=np.int64)
    if task == "regression":
        order = np.argsort(arr, kind="stable") if row_ids is None else np.lexsort((np.arange(rows), rids, arr))
        strata = [s.astype(np.int64) for s in np.array_split(order, min(10, n))]
        quotas = _largest_remainder(n, [len(s) for s in strata])
    elif task in ("classification", "binary", "multiclass"):
        labels = np.unique(arr)
        strata = [np.flatnonzero(arr == lab).astype(np.int64) for lab in labels]
        counts = [len(s) for s in strata]
        if min(counts) < 2 or n < 2 * len(strata):
            raise ValueError("subsample_class_too_small")
        quotas = [2 + e for e in _largest_remainder(n - 2 * len(strata), [c - 2 for c in counts])]
    else:
        raise ValueError(f"unknown_task_{task}")
    picked = [idx[np.lexsort((idx, rids[idx], keys[idx]))[:q]] for idx, q in zip(strata, quotas) if q > 0]
    return np.sort(np.concatenate(picked)).astype(np.int64)


def merge_reports(base: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Union native group reports: cells by id (incoming wins when it measured the cell)."""
    if base is None:
        return json.loads(json.dumps(incoming))
    out = dict(base)
    cells = {int(c["id"]): c for c in base.get("cells") or []}
    for c in incoming.get("cells") or []:
        cid = int(c["id"])
        if c.get("state") in ("complete", "uninformative") or cid not in cells:
            cells[cid] = c
    out["cells"] = [cells[k] for k in sorted(cells)]
    if incoming.get("spectrum"):
        out["spectrum"] = incoming["spectrum"]
    if out.get("target_stats") is None and incoming.get("target_stats") is not None:
        out["target_stats"] = incoming["target_stats"]
    return out


def progressive_steps(
    compute_group: Callable[[Tuple[int, ...]], Dict[str, Any]], *,
    subsample_n: Optional[int] = DEFAULT_PROGRESSIVE_SUBSAMPLE,
    source_digest: Optional[str] = None, provenance: Optional[str] = None,
    groups: Sequence[Tuple[str, Tuple[int, ...]]] = PROGRESSIVE_GROUPS,
    start: Optional[LandmarkBundle] = None,
) -> Iterator[Tuple[int, str, LandmarkBundle, Optional[Dict[str, Any]]]]:
    """Yield ``(seq, group_name, bundle, merged_native_report)``: start (zero), then +each group."""
    if provenance is None:
        provenance = "native-v1" if subsample_n is None else f"subsample-{subsample_n}-stratified"
    digest = source_digest or hashlib.sha256(f"dope.progressive/v1\0{subsample_n}".encode()).hexdigest()
    bundle = LandmarkBundle.zero() if start is None else start
    merged: Optional[Dict[str, Any]] = None
    yield 0, "start" if start is not None and not start.is_zero() else "zero", bundle, None
    for seq, (name, slots) in enumerate(groups, start=1):
        if all(bundle.mask[s] for s in slots):
            continue
        report = compute_group(tuple(slots))
        allowed = {s + 1 for s in slots}
        scoped = dict(report, cells=[c for c in report.get("cells") or [] if int(c.get("id", 0)) in allowed])
        part = LandmarkBundle.from_report(scoped, subsample_n, source_digest=digest, provenance=provenance)
        bundle = bundle.merge(part)
        merged = merge_reports(merged, scoped)
        yield seq, name, bundle, merged


def progressive_bundles(compute_group: Callable[[Tuple[int, ...]], Dict[str, Any]], **kwargs: Any) -> Iterator[LandmarkBundle]:
    for _, _, bundle, _ in progressive_steps(compute_group, **kwargs):
        yield bundle


def mlc_v3_training_options(seed: int = 0) -> Dict[str, Any]:
    """Canonical native landmark options for DOPE-MLC-v3 training and product encoding (work_budget = 2e12)."""
    from .landmark_features import OPTIONS
    opts = dict(OPTIONS)
    opts["work_budget"] = MLC_V3_TRAINING_WORK_BUDGET
    opts["seed"] = int(seed)
    return opts


def native_group_runner(
    engine: str | Path, csv_path: str | Path, *, target: int, task: str,
    subsample_n: Optional[int] = DEFAULT_PROGRESSIVE_SUBSAMPLE, seed: int = 0,
    options: Optional[Dict[str, Any]] = None, threads: int = 4, wall_seconds: int = 300,
) -> Callable[[Tuple[int, ...]], Dict[str, Any]]:
    opts = dict(mlc_v3_training_options(seed=seed) if options is None else options)
    names = {tuple(sorted(slots)): name for name, slots in PROGRESSIVE_GROUPS}

    def run(slots: Tuple[int, ...]) -> Dict[str, Any]:
        ids = names.get(tuple(sorted(slots))) or ",".join(str(s + 1) for s in sorted(slots))
        cmd = [str(engine), "--threads", str(threads), "landmarks", "real", str(csv_path),
               "--target", str(target), "--task", report_task(task), "--json"]
        for k, v in opts.items():
            cmd += ["--" + k.replace("_", "-"), str(v)]
        cmd += ["--ids", ids]
        if subsample_n is not None:
            cmd += ["--subsample", str(subsample_n), "--subsample-seed", str(seed)]
        result = subprocess.run(cmd, capture_output=True, timeout=wall_seconds, check=False)
        if result.returncode != 0:
            raise RuntimeError("native_group_runner_failed")
        return json.loads(result.stdout)

    return run


def table_csv_bytes(X: np.ndarray, y: np.ndarray) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    X = np.asarray(X, dtype=np.float64)
    writer.writerow([f"c{i}" for i in range(X.shape[1])] + ["target"])
    for row, t in zip(X, np.asarray(y, dtype=np.float64).reshape(-1)):
        writer.writerow([repr(float(v)) for v in row] + [repr(float(t))])
    return stream.getvalue().encode()


def observed_target_stats(
    y: np.ndarray, task: str, *, y_raw: Optional[np.ndarray] = None,
    target_stats: Optional[Sequence[Optional[float]]] = None,
) -> List[Optional[float]]:
    if target_stats is not None:
        return (list(target_stats) + [None, None, None])[:3]
    if y_raw is not None:
        return table_target_stats(y_raw, task)
    if task == "regression":
        return [None, None, None]
    return table_target_stats(y, task)


def observed_report_descriptors(
    bundle: LandmarkBundle, *, rows: int, features: int, classes: int, task: str,
    target_stats: Sequence[Optional[float]], level: str = "high",
) -> Dict[str, Any]:
    lvl_mask = level_landmark_mask(level)
    present = np.asarray(bundle.mask, dtype=bool) & lvl_mask
    s = np.where(present, np.asarray(bundle.values, dtype=np.float32).astype(np.float64), 0.0)
    raw = np.array([unsquash(v) if p else 0.0 for v, p in zip(s, present)], dtype=np.float64)
    midranks, valid = np.zeros(23, dtype=np.float64), np.where(present)[0]
    if valid.size > 1:
        for i in valid:
            less, eq = float(np.sum(raw[valid] < raw[i])), float(np.sum(raw[valid] == raw[i]))
            midranks[i] = (less + 0.5 * (eq - 1.0)) / float(valid.size - 1)
    elif valid.size == 1:
        midranks[valid[0]] = 0.5
    contrasts = compute_level_contrasts(raw, present, level=level)
    ts = list(target_stats) + [None, None]
    t_stats = squash(np.array([float(ts[0]) if ts[0] is not None and math.isfinite(float(ts[0])) else 0.0, float(ts[1]) if ts[1] is not None and math.isfinite(float(ts[1])) else 0.0], dtype=np.float64))
    rows_c, feats_c = max(int(rows), 1), max(int(features), 1)
    is_clf = 1.0 if int(classes) >= 2 or report_task(task) == "classification" else 0.0
    shape_vec = np.array([is_clf, math.log10(rows_c) / 6.0, math.log2(feats_c) / 12.0, math.log10(feats_c / float(rows_c)) / 4.0], dtype=np.float64)
    _gmax = lambda idxs, gate: float(max([s[i] for i in idxs if present[i]], default=0.0)) if gate else 0.0  # noqa: E731
    _gmean = lambda idxs, gate: float(np.mean([s[i] for i in idxs if present[i]])) if gate and any(present[i] for i in idxs) else 0.0  # noqa: E731
    larger_gate, fine_gate = level in ("high", "ultra"), level == "ultra"
    group = np.array([_gmax(MACRO, True), _gmax(LARGER, larger_gate), _gmax(FINE, fine_gate), _gmean(MACRO, True), _gmean(LARGER, larger_gate), _gmean(FINE, fine_gate), 0.0, 0.0, 0.0], dtype=np.float64)
    spec_summary = np.zeros(16, dtype=np.float64)
    vector = np.concatenate([s, present.astype(np.float64), midranks, contrasts, t_stats, shape_vec, spec_summary, group])
    return {
        "level": level, "gains23": s, "ses23": np.zeros(23), "lambdas23": np.zeros(23), "mask23": present,
        "midranks23": midranks, "contrasts16": contrasts, "target_stats2": t_stats, "shape4": shape_vec,
        "spectrum_summary16": spec_summary, "group_summary9": group,
        "vector116": vector.astype(np.float32),
        "column_spectrum": np.zeros((feats_c, 9), dtype=np.float32),
        "column_prior_importance": np.full(feats_c, 1.0 / float(feats_c), dtype=np.float32),
    }


def build_observed_conditioning_bundle(
    bundle: LandmarkBundle, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, *,
    task: str, level: str = "high", y_train_raw: Optional[np.ndarray] = None,
    target_stats: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    from .building_block_landmarks import zero_building_block_landmarks
    X = np.asarray(X_train_scaled, dtype=np.float64)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    y = np.asarray(y_train_scaled, dtype=np.float64).reshape(-1)
    n, p = int(X.shape[0]), int(X.shape[1])
    rep_desc = observed_report_descriptors(
        bundle, rows=n, features=p, classes=table_classes(y, task), task=task,
        target_stats=observed_target_stats(y, task, y_raw=y_train_raw, target_stats=target_stats), level=level,
    )
    tab_desc = compute_table_descriptors(X, y, level=level)
    bb_desc = zero_building_block_landmarks(p)
    inter_vec48 = compute_observed_inter_landmark_stats(rep_desc, level=level)
    vec140 = np.concatenate([rep_desc["vector116"], tab_desc["table_vec24"]]).astype(np.float32)
    vec256 = np.concatenate([vec140, inter_vec48, bb_desc["building_block_vec68"]]).astype(np.float32)
    col_desc8 = np.asarray(tab_desc["column_desc"], dtype=np.float32)
    pc = col_desc8.shape[0]
    col_landmarks24 = np.column_stack([np.zeros((pc, 9), dtype=np.float32), col_desc8,
                                       np.zeros((pc, 7), dtype=np.float32)]).astype(np.float32)
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
        "rep_desc": rep_desc, "tab_desc": tab_desc, "bb_desc": bb_desc,
    }


def effective_conditioning_digest(vec256: np.ndarray, col_landmarks: np.ndarray, *, recipe: str, level: str) -> str:
    vec = np.asarray(vec256, dtype="<f4").reshape(256)
    cols = np.asarray(col_landmarks, dtype="<f4").reshape(64, 17)
    h = hashlib.sha256(CONDITIONING_DOMAIN)
    h.update(recipe.encode() + b"\0" + level.encode() + b"\0")
    h.update(vec.tobytes())
    h.update(cols.tobytes())
    return h.hexdigest()


def build_conditioning(
    X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, *, task: str,
    bundle: Optional[LandmarkBundle] = None, mode: str = "full",
    native_report: Optional[Dict[str, Any]] = None,
    level: str = "high", subsample_n: Optional[int] = None,
    y_train_raw: Optional[np.ndarray] = None,
    target_stats: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    """Select and run the V3 conditioning recipe for an optional-landmark mode."""
    from .building_block_landmarks import build_v3_conditioning_bundle
    if mode not in LANDMARK_MODES:
        raise ValueError(f"unknown_landmark_mode_{mode}")
    if bundle is not None:
        if not isinstance(bundle, LandmarkBundle):
            raise TypeError("landmark_bundle_required")
        bundle = bundle.sanitize()
    X_arr = np.asarray(X_train_scaled, dtype=np.float64)
    X_arr = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
    y_arr = np.asarray(y_train_scaled, dtype=np.float64).reshape(-1)
    table_rows, table_features = int(X_arr.shape[0]), int(X_arr.shape[1])
    if mode == "full":
        if native_report is None:
            raise ValueError("landmark_full_requires_native_report")
        if bundle is not None:
            obs_b = LandmarkBundle.from_report(native_report, bundle.subsample_n, include_uninformative=True)
            comp_b = LandmarkBundle.from_report(native_report, bundle.subsample_n, complete_only=True)
            if not (bundle.same_landmarks(obs_b) or bundle.same_landmarks(comp_b)):
                raise ValueError("landmark_bundle_report_mismatch")
        recipe = FULL_REPORT_RECIPE
        eff_sub = subsample_n if subsample_n is not None else (bundle.subsample_n if bundle is not None else None)
        rep_rows = native_report.get("rows")
        is_sampled = (eff_sub is not None and eff_sub < table_rows) or (isinstance(rep_rows, int) and rep_rows < table_rows)
        overrides: Optional[Dict[str, Any]] = None
        if is_sampled or target_stats is not None or y_train_raw is not None:
            overrides = {
                "rows": table_rows, "features": table_features,
                "classes": table_classes(y_arr, task), "task": report_task(task),
                "target_stats": observed_target_stats(y_arr, task, y_raw=y_train_raw, target_stats=target_stats),
            }
        cond = build_v3_conditioning_bundle(
            native_report, X_train_scaled, y_train_scaled, level=level, report_overrides=overrides,
        )
    else:
        if mode in ("none", "fast") and bundle is not None and any(bundle.mask):
            raise ValueError("landmark_mode_none_rejects_bundle")
        recipe = OBSERVED_RECIPE
        cond = build_observed_conditioning_bundle(
            LandmarkBundle.zero() if (bundle is None or mode in ("none", "fast")) else bundle,
            X_train_scaled, y_train_scaled, task=task, level=level,
            y_train_raw=y_train_raw, target_stats=target_stats,
        )
    vec256 = np.asarray(cond["vec256"], dtype=np.float32).reshape(256)
    col24 = np.asarray(cond["col_landmarks24"], dtype=np.float32)
    cols = np.zeros((MAX_COLS, 17), dtype=np.float32)
    take = min(MAX_COLS, col24.shape[0])
    cols[:take] = col24[:take, :17]
    return {
        "recipe": recipe, "level": level, "cond": cond, "vec256": vec256, "col_landmarks": cols,
        "digest": effective_conditioning_digest(vec256, cols, recipe=recipe, level=level),
    }


def build_conditioning_inputs(
    X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, *, task: str,
    bundle: Optional[LandmarkBundle], mode: str, native_report: Optional[Dict[str, Any]] = None,
    level: str = "high", subsample_n: Optional[int] = None,
    y_train_raw: Optional[np.ndarray] = None,
    target_stats: Optional[Sequence[Optional[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    out = build_conditioning(
        X_train_scaled, y_train_scaled, task=task, bundle=bundle, mode=mode,
        native_report=native_report, level=level, subsample_n=subsample_n,
        y_train_raw=y_train_raw, target_stats=target_stats,
    )
    return out["vec256"], out["col_landmarks"]


def encode_with_landmarks(
    model: Any, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, *,
    landmarks: Optional[LandmarkBundle] = None, landmark_mode: str = "full",
    native_report: Optional[Dict[str, Any]] = None, task: str = "regression", level: str = "ultra",
    **kwargs: Any,
) -> Dict[str, Any]:
    if landmarks is None:
        landmarks = (LandmarkBundle.from_report(native_report) if landmark_mode == "full" and native_report is not None
                     else LandmarkBundle.zero())
    return model.encode_table(
        X_train_scaled, y_train_scaled, native_report, task=task, level=level,
        landmarks=landmarks, landmark_mode=landmark_mode, **kwargs,
    )
