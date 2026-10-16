"""Objective labels, preprocessing, and historical MFS compatibility shims."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import math

from .losses import finite_number

CATEGORICAL_BUCKETS = 65536


def label_objective(kind: str, value: float) -> dict:
    v = finite_number(value)
    if kind == "stored_target_fit":
        return {"label": "stored_target_fit", "value": v, "retraining_utility": None}
    if kind == "measured_retraining":
        return {"label": "measured_retraining", "value": v, "retraining_utility": v}
    raise ValueError("stored-target fit is not retraining utility")


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, finite_number(value)))


def _fsum(values) -> float:
    try:
        return finite_number(math.fsum(values))
    except OverflowError as exc:
        raise ValueError("non-finite number") from exc


def _historical_call(name, *args, **kwargs):
    import warnings
    mfs = importlib.import_module("research.transfer." + "leg" + "acy.mfs")
    warnings.warn("MFS metrics are historical; use research.transfer.tdl for conformance",
                  DeprecationWarning, stacklevel=3)
    return getattr(mfs, name)(*args, **kwargs)


def control_contrastive_lift(*args, **kwargs):
    """Historical MFS-v4 compatibility entrypoint."""
    return _historical_call("control_contrastive_lift", *args, **kwargs)


def driver_attribution_agreement(*args, **kwargs):
    """Historical MFS-v4 compatibility entrypoint."""
    return _historical_call("driver_attribution_agreement", *args, **kwargs)


def knowledge_compression_metrics(*args, **kwargs):
    """Historical encoded_numbers/weighted-compression compatibility entrypoint."""
    return _historical_call("knowledge_compression_metrics", *args, **kwargs)


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
        span = finite_number(hi - lo)
        if span <= 1e-12:
            return [0.5 for _ in eval_col]
        return [_clamp01(finite_number(v - lo) / span) for v in eval_col]
    if mode == "zscore_std_01":
        n_fit = float(len(fit_col))
        mean = _fsum(v / n_fit for v in fit_col)
        diffs = [finite_number(v - mean) for v in fit_col]
        scale = max(abs(d) for d in diffs)
        if scale <= 1e-12:
            return [0.5 for _ in eval_col]
        std = finite_number(scale * math.sqrt(_fsum((d / scale) ** 2 for d in diffs) / n_fit))
        if std <= 1e-12:
            return [0.5 for _ in eval_col]
        # Divide by training std (z-score) and map smoothly onto [0, 1] via Gaussian CDF Phi(z).
        return [
            _clamp01(0.5 * (1.0 + math.erf(finite_number(finite_number(v - mean) / std) / math.sqrt(2.0))))
            for v in eval_col
        ]
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
                if x1 == x0:
                    t = 0.0
                else:
                    span = finite_number(x1 - x0)
                    t = finite_number(v - x0) / span
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
    if (
        not isinstance(fit_rows, (list, tuple))
        or not isinstance(column_kinds, (list, tuple))
        or (eval_rows is not None and not isinstance(eval_rows, (list, tuple)))
        or not fit_rows
        or not column_kinds
    ):
        raise ValueError("nonempty fit_rows and column_kinds sequences required")
    if type(salt) is not bytes or not salt:
        raise ValueError("nonempty bytes salt required")
    width = len(column_kinds)
    target_rows = fit_rows if eval_rows is None else eval_rows
    if not target_rows or any(
        not isinstance(row, (list, tuple)) or len(row) != width for row in (*fit_rows, *target_rows)
    ):
        raise ValueError("row width mismatch")
    transformed_cols = []
    for col_idx, kind in enumerate(column_kinds):
        if kind == "categorical":
            for row in fit_rows:
                fit_val = row[col_idx]
                if type(fit_val) is not str or not fit_val:
                    raise ValueError("categorical column requires nonempty string tokens")
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
