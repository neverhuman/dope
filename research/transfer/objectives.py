"""Objective labels, MFS-v4 transfer primitives, and standard [0, 1] preprocessing.

A stored-target fit is not retraining utility. MFS-v4 combines control-contrastive
downstream lift, holdout feature-driver attribution agreement, and MDL knowledge
compression under hard privacy and byte gates.
"""

from __future__ import annotations

import hashlib
import hmac
import math

from .losses import finite_number

CATEGORICAL_BUCKETS = 65536


def label_objective(kind: str, value: float) -> dict:
    if kind == "stored_target_fit":
        return {"label": "stored_target_fit", "value": float(value), "retraining_utility": None}
    if kind == "measured_retraining":
        return {"label": "measured_retraining", "value": float(value), "retraining_utility": float(value)}
    raise ValueError("stored-target fit is not retraining utility")


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, finite_number(value)))


def _require_positive_int(value: int, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f"invalid {name}")
    return value


def control_contrastive_lift(
    retention: float,
    predictor_only_retention: float,
    independent_marginals_retention: float,
    real_bootstrap_4n_retention: float | None = None,
) -> dict:
    """Contrast downstream retention against predictor-only and marginal controls."""
    r = finite_number(retention)
    pred = finite_number(predictor_only_retention)
    indep = finite_number(independent_marginals_retention)
    boot = None if real_bootstrap_4n_retention is None else finite_number(real_bootstrap_4n_retention)
    lift_pred = finite_number(r - pred)
    lift_indep = finite_number(r - indep)
    lift_boot = None if boot is None else finite_number(r - boot)
    base = _clamp01(r)
    pred_bonus = _clamp01(max(0.0, lift_pred) / 0.20)
    dep_bonus = _clamp01(max(0.0, lift_indep) / 0.25)
    score = _clamp01(0.65 * base + 0.20 * pred_bonus + 0.15 * dep_bonus)
    return {
        "retention": r,
        "predictor_only_retention": pred,
        "independent_marginals_retention": indep,
        "real_bootstrap_4n_retention": boot,
        "lift_over_predictor_only": lift_pred,
        "lift_over_independent_marginals": lift_indep,
        "lift_vs_real_bootstrap_4n": lift_boot,
        "positive_predictor_lift": lift_pred > 0.0,
        "positive_dependence_lift": lift_indep > 0.0,
        "contrastive_utility_score": score,
    }


def driver_attribution_agreement(
    real_importances,
    synthetic_importances,
    real_signs=None,
    synthetic_signs=None,
) -> dict:
    """Score holdout feature-driver fidelity via NDCG@k, Top-k Jaccard, L1 share, and sign agreement."""
    real_imp = [finite_number(v) for v in real_importances]
    syn_imp = [finite_number(v) for v in synthetic_importances]
    features = len(real_imp)
    if features == 0 or features != len(syn_imp) or any(v < 0.0 for v in (*real_imp, *syn_imp)):
        raise ValueError("invalid driver importance vector")
    if real_signs is None or synthetic_signs is None:
        raise ValueError("driver sign vectors are required")
    real_s = list(real_signs)
    syn_s = list(synthetic_signs)
    if not real_s or len(real_s) != len(syn_s):
        raise ValueError("driver sign vector length mismatch")
    if any(type(s) is not int or isinstance(s, bool) or s not in (-1, 0, 1) for s in (*real_s, *syn_s)):
        raise ValueError("invalid driver sign value")
    active = [(r, s) for r, s in zip(real_s, syn_s) if r != 0]
    sign_agreement = 1.0 if not active else sum(1 for r, s in active if r == s) / float(len(active))
    k = max(1, min(features, min(20, max(5, math.ceil(features * 0.10)))))
    real_total = finite_number(math.fsum(real_imp))
    syn_total = finite_number(math.fsum(syn_imp))
    if real_total <= 1e-12:
        ndcg, jaccard, overlap, l1_agree = (1.0, 1.0, 1.0, 1.0) if syn_total <= 1e-12 else (0.0, 0.0, 0.0, 0.0)
    elif syn_total <= 1e-12:
        ndcg, jaccard, overlap, l1_agree = (0.0, 0.0, 0.0, 0.0)
    else:
        real_order = sorted(range(features), key=lambda idx: (-real_imp[idx], idx))
        syn_order = sorted(range(features), key=lambda idx: (-syn_imp[idx], idx))
        real_top = set(real_order[:k])
        syn_top = set(syn_order[:k])
        inter = len(real_top & syn_top)
        union = max(1, len(real_top | syn_top))
        overlap = inter / float(k)
        jaccard = inter / float(union)
        idcg = math.fsum(real_imp[idx] / math.log2(rank + 2.0) for rank, idx in enumerate(real_order[:k]))
        dcg = math.fsum(real_imp[idx] / math.log2(rank + 2.0) for rank, idx in enumerate(syn_order[:k]))
        ndcg = 1.0 if idcg <= 1e-12 else _clamp01(dcg / idcg)
        tv = 0.5 * math.fsum(abs(real_imp[i] / real_total - syn_imp[i] / syn_total) for i in range(features))
        l1_agree = _clamp01(1.0 - tv)
    composite = _clamp01(0.35 * ndcg + 0.25 * jaccard + 0.25 * l1_agree + 0.15 * sign_agreement)
    return {
        "k": k,
        "ndcg_at_k": ndcg,
        "top_k_jaccard": jaccard,
        "top_k_overlap": overlap,
        "l1_attribution_agreement": l1_agree,
        "sign_agreement": sign_agreement,
        "composite_driver_fidelity": composite,
    }


def knowledge_compression_metrics(
    artifact_bytes: int,
    encoded_numbers: int,
    real_rows: int,
    real_columns: int,
    byte_limit: int,
    retained_utility: float,
) -> dict:
    """Compute MDL-inspired knowledge compression and utility-per-kilobyte metrics."""
    artifact_bytes = _require_positive_int(artifact_bytes, "artifact_bytes")
    encoded_numbers = _require_positive_int(encoded_numbers, "encoded_numbers")
    real_rows = _require_positive_int(real_rows, "real_rows")
    real_columns = _require_positive_int(real_columns, "real_columns")
    byte_limit = _require_positive_int(byte_limit, "byte_limit")
    if byte_limit < 2:
        raise ValueError("invalid byte_limit")
    utility = finite_number(retained_utility)
    cells = float(real_rows * real_columns)
    bpc = finite_number(8.0 * float(artifact_bytes) / cells)
    npc = finite_number(float(encoded_numbers) / cells)
    bpc_score = _clamp01(1.0 / (1.0 + bpc))
    sparsity_score = _clamp01(1.0 / (1.0 + float(encoded_numbers) / (float(real_columns) * 8.0)))
    log_byte_score = (
        _clamp01(1.0 - math.log(float(artifact_bytes)) / math.log(float(byte_limit)))
        if artifact_bytes <= byte_limit
        else 0.0
    )
    compression_score = _clamp01(0.40 * bpc_score + 0.35 * sparsity_score + 0.25 * log_byte_score)
    pos_utility = max(0.0, utility)
    return {
        "artifact_bytes": artifact_bytes,
        "encoded_numbers": encoded_numbers,
        "real_rows": real_rows,
        "real_columns": real_columns,
        "byte_limit": byte_limit,
        "bits_per_original_cell": bpc,
        "numbers_per_original_cell": npc,
        "bpc_score": bpc_score,
        "parameter_sparsity_score": sparsity_score,
        "log_byte_score": log_byte_score,
        "knowledge_compression_score": compression_score,
        "utility_per_kilobyte": finite_number(pos_utility / (float(artifact_bytes) / 1024.0)),
        "utility_per_encoded_number": finite_number(pos_utility / float(encoded_numbers)),
    }


def _category_bucket01(salt: bytes, level: str) -> float:
    digest = hmac.new(salt, level.encode("utf-8"), hashlib.sha256).digest()
    bucket = int.from_bytes(digest[:2], "big")
    return (bucket + 0.5) / float(CATEGORICAL_BUCKETS)


def _scale_numeric_column_01(fit_col: list[float], eval_col: list[float], mode: str) -> list[float]:
    if not fit_col:
        raise ValueError("empty fit column")
    for v in (*fit_col, *eval_col):
        finite_number(v)
    if mode == "minmax_01":
        lo, hi = min(fit_col), max(fit_col)
        span = hi - lo
        if span <= 1e-12:
            return [0.5 for _ in eval_col]
        return [_clamp01((v - lo) / span) for v in eval_col]
    if mode == "zscore_std_01":
        mean = math.fsum(fit_col) / float(len(fit_col))
        var = math.fsum((v - mean) ** 2 for v in fit_col) / float(len(fit_col))
        std = math.sqrt(var)
        if std <= 1e-12:
            return [0.5 for _ in eval_col]
        # Divide by training std (z-score) and map smoothly onto [0, 1] via Gaussian CDF Phi(z).
        return [_clamp01(0.5 * (1.0 + math.erf(((v - mean) / std) / math.sqrt(2.0)))) for v in eval_col]
    if mode == "midrank_quantile_01":
        ordered = sorted(fit_col)
        n = float(len(ordered))
        out = []
        for v in eval_col:
            less = sum(1 for x in ordered if x < v)
            right = sum(1 for x in ordered if x <= v)
            if right == 0:
                out.append(0.5 / n)
            elif less == len(ordered):
                out.append((n - 0.5) / n)
            elif less == right:
                x0, x1 = ordered[less - 1], ordered[less]
                left_img, right_img = (less - 0.5) / n, (less + 0.5) / n
                t = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)
                out.append(_clamp01(left_img + t * (right_img - left_img)))
            else:
                out.append(_clamp01(sum((idx + 0.5) / n for idx in range(less, right)) / float(right - less)))
        return out
    raise ValueError("unknown numeric scaling mode")


def standardize_numeric_table_01(
    fit_rows: list[list[float | int | str]],
    eval_rows: list[list[float | int | str]] | None = None,
    *,
    column_kinds: list[str] | tuple[str, ...],
    numeric_mode: str = "zscore_std_01",
    salt: bytes = b"dope-standard-01-v1",
) -> dict:
    """Standardize a real/synthetic table into purely numeric [0, 1] columns fit on real training rows only.

    Categorical columns are HMAC-encoded into 65,536 deterministic buckets in (0, 1).
    Numeric columns are scaled to [0, 1] using training-only parameters via:
    - 'zscore_std_01': divide by training std ((x - mean) / std) and map onto [0, 1] via Phi(z)
    - 'minmax_01': training min-max projection clamped to [0, 1]
    - 'midrank_quantile_01': empirical mid-rank quantile map onto (0, 1)
    No source headers, category strings, or raw extrema appear in the returned public metadata.
    """
    if not fit_rows or not column_kinds:
        raise ValueError("nonempty fit_rows and column_kinds required")
    if type(salt) is not bytes or not salt:
        raise ValueError("nonempty bytes salt required")
    width = len(column_kinds)
    target_rows = fit_rows if eval_rows is None else eval_rows
    if not target_rows or any(len(row) != width for row in (*fit_rows, *target_rows)):
        raise ValueError("row width mismatch")
    transformed_cols = []
    for col_idx, kind in enumerate(column_kinds):
        if kind == "categorical":
            col_vals = []
            for row in target_rows:
                val = row[col_idx]
                if type(val) is not str or not val:
                    raise ValueError("categorical column requires nonempty string tokens")
                col_vals.append(_category_bucket01(salt, f"c{col_idx}:{val}"))
            transformed_cols.append(col_vals)
        elif kind == "numeric":
            fit_col = [finite_number(row[col_idx]) for row in fit_rows]
            eval_col = [finite_number(row[col_idx]) for row in target_rows]
            transformed_cols.append(_scale_numeric_column_01(fit_col, eval_col, numeric_mode))
        else:
            raise ValueError("column_kinds entries must be 'numeric' or 'categorical'")
    rows_out = [[transformed_cols[c][r] for c in range(width)] for r in range(len(target_rows))]
    return {
        "format": "dope-numeric-01-table-v1",
        "numeric_mode": numeric_mode,
        "columns": width,
        "rows": rows_out,
        "positional_headers": [f"f{i}" for i in range(width)],
        "cleartext_headers_retained": False,
        "raw_extrema_retained": False,
    }
