"""Historical MFS-v4 primitives. Preserve semantics; do not extend."""
from __future__ import annotations

import math
from research.transfer.losses import finite_number

def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, finite_number(value)))


def _fsum(values) -> float:
    try:
        return finite_number(math.fsum(values))
    except OverflowError as exc:
        raise ValueError("non-finite number") from exc


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
    if not isinstance(real_importances, (list, tuple)) or not isinstance(synthetic_importances, (list, tuple)):
        raise ValueError("invalid driver importance vector")
    real_imp = [finite_number(v) for v in real_importances]
    syn_imp = [finite_number(v) for v in synthetic_importances]
    features = len(real_imp)
    if features == 0 or features != len(syn_imp) or any(v < 0.0 for v in (*real_imp, *syn_imp)):
        raise ValueError("invalid driver importance vector")
    real_total = _fsum(real_imp)
    syn_total = _fsum(syn_imp)
    active_real_features = sum(1 for v in real_imp if v > 1e-12)
    if real_total <= 1e-12 or active_real_features == 0:
        raise ValueError("degenerate real driver importance vector")
    if real_signs is None or synthetic_signs is None:
        raise ValueError("driver sign vectors are required")
    if not isinstance(real_signs, (list, tuple)) or not isinstance(synthetic_signs, (list, tuple)):
        raise ValueError("invalid driver sign vector")
    real_s = list(real_signs)
    syn_s = list(synthetic_signs)
    if len(real_s) < features or len(real_s) != len(syn_s):
        raise ValueError("driver sign vector length mismatch")
    if any(type(s) is not int or isinstance(s, bool) or s not in (-1, 0, 1) for s in (*real_s, *syn_s)):
        raise ValueError("invalid driver sign value")
    if any(
        (imp > 1e-12 and sign == 0) or (imp <= 1e-12 and sign != 0)
        for imp, sign in zip(real_imp, real_s)
    ) or any(imp > 1e-12 and sign == 0 for imp, sign in zip(syn_imp, syn_s)):
        raise ValueError("missing or spurious directional sign for active driver")
    active = [(idx, r, s) for idx, (r, s) in enumerate(zip(real_s, syn_s)) if r != 0]
    if not active:
        raise ValueError("degenerate driver sign vector")
    sign_agreement = sum(
        1
        for idx, r, s in active
        if r == s and (idx >= features or syn_imp[idx] > 1e-12)
    ) / float(len(active))
    k = max(1, min(features, active_real_features, min(20, max(5, math.ceil(features * 0.10)))))
    if syn_total <= 1e-12:
        ndcg, jaccard, overlap, l1_agree = (0.0, 0.0, 0.0, 0.0)
    else:
        real_order = sorted(range(features), key=lambda idx: (-real_imp[idx], idx))
        syn_order = sorted(range(features), key=lambda idx: (-syn_imp[idx], idx))
        real_top = {idx for idx in real_order[:k] if real_imp[idx] > 1e-12}
        syn_top = {idx for idx in syn_order[:k] if syn_imp[idx] > 1e-12}
        inter = len(real_top & syn_top)
        union = max(1, len(real_top | syn_top))
        overlap = inter / float(k)
        jaccard = inter / float(union)
        idcg = _fsum(real_imp[idx] / math.log2(rank + 2.0) for rank, idx in enumerate(real_order[:k]))
        dcg = _fsum(
            real_imp[idx] / math.log2(rank + 2.0)
            for rank, idx in enumerate(syn_order[:k])
            if syn_imp[idx] > 1e-12
        )
        ndcg = 0.0 if idcg <= 1e-12 else _clamp01(dcg / idcg)
        tv = 0.5 * _fsum(abs(real_imp[i] / real_total - syn_imp[i] / syn_total) for i in range(features))
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
    compression_score = (
        _clamp01(0.40 * bpc_score + 0.35 * sparsity_score + 0.25 * log_byte_score)
        if utility > 0.0 and artifact_bytes <= byte_limit
        else 0.0
    )
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


