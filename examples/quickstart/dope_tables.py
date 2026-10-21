"""Reversible encoding between a real table and DOPE's unit-interval matrix.

DOPE's compiler (`dope-kernel compile --dataset-dir`) reads a headerless numeric
``train.csv`` whose features and target all lie in [0, 1], with the target in the
last column and blanks for missing feature values. Real tables have strings,
skewed amounts, outliers, integers, dates and missing cells. ``TableEncoder``
bridges the two in both directions:

* ``fit(df)`` learns one small transform per column from the fit rows only;
* ``transform(df)`` maps rows into the unit-interval matrix DOPE compiles;
* ``inverse_transform(matrix)`` maps DOPE's samples back to the original schema,
  with original column names, category labels, integer types and dates;
* ``to_bytes()`` writes the transform state as a compact binary blob, so the bytes
  the decoder needs can be charged next to the compiled artifact.

Default encodings, chosen by measurement (see examples/quickstart/README.md):

``numeric="robust"``
    Apply ``log1p`` to non-negative columns whose skewness exceeds 2, clip at
    Tukey's far-out fences (quartiles -/+ 3 IQR, never tighter than the 1%/99%
    quantiles; 0.1%/99.9% quantiles when the IQR is zero), then min-max to
    [0, 1]. Outliers no longer
    squash the bulk of a column into a sliver of the interval, and heavy tails
    become close to linear, which helps the compact target model. Plain
    ``"minmax"`` and a 256-knot ``"quantile"`` map are available for comparison.

``categorical="target_ordered"``
    One column per categorical feature. Levels are sorted by their smoothed
    (empirical-Bayes) mean target on the fit rows and placed in [0, 1] in
    proportion to those means, so the code is a monotone, linear carrier of the
    category's main effect. DOPE stores the at most 64 distinct codes as an exact
    categorical grid marginal, so category frequencies are reproduced exactly and
    decoding is lossless. ``"onehot"`` gives pandas ``get_dummies(drop_first=True)``
    columns (the most frequent level is the dropped reference, because a full set of
    dummies is collinear with an intercept), decoded by arg-max; ``"auto"`` uses one-hot
    only for categoricals with at most ``onehot_max_levels`` levels. The published
    benchmark projection used one-hot for every categorical column.

The target of a regression task is mapped by a plain fit-row min-max (an affine
map, so retention, which is a ratio of loss differences, is unchanged by it). A
two-class target becomes 0/1. Multi-class targets are refused: DOPE's compiler
supports regression and binary tasks.

Only the fit rows are ever used to fit a transform. Nothing in this module reads
a holdout or test split.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ENCODER_MAGIC = b"DOPEENC1"
MISSING = "__missing__"
OTHER = "__other__"
_KINDS = ("numeric", "binary", "ordered", "onehot", "datetime")


@dataclass
class ColumnTransform:
    """The fitted transform of one source column."""

    name: str
    kind: str
    lo: float = 0.0
    hi: float = 1.0
    log: bool = False
    integer: bool = False
    levels: list[str] = field(default_factory=list)
    codes: list[float] = field(default_factory=list)
    boolean: bool = False
    knots: list[float] = field(default_factory=list)

    @property
    def width(self) -> int:
        # One-hot drops the most frequent (reference) level, like get_dummies(drop_first=True):
        # a full set of dummies sums to one and is collinear with any intercept.
        return len(self.levels) - 1 if self.kind == "onehot" else 1


class TableEncoder:
    """Fit-row-only, reversible encoder from a DataFrame to DOPE's [0, 1] matrix."""

    def __init__(
        self,
        target: str,
        *,
        task: str = "auto",
        numeric: str = "robust",
        categorical: str = "target_ordered",
        onehot_max_levels: int = 4,
        clip_quantile: float = 0.001,
        skew_threshold: float = 2.0,
        smoothing: float = 10.0,
        max_levels: int = 64,
        min_level_share: float = 0.002,
        drop: Iterable[str] = (),
        positive: object | None = None,
    ):
        if numeric not in ("robust", "minmax", "quantile"):
            raise ValueError("numeric must be 'robust', 'minmax' or 'quantile'")
        if categorical not in ("auto", "target_ordered", "onehot"):
            raise ValueError("categorical must be 'auto', 'target_ordered' or 'onehot'")
        if not 2 <= max_levels <= 64:
            raise ValueError("max_levels must lie in [2, 64]; DOPE keeps at most 64 exact grid levels")
        self.target = target
        self.task = task
        self.numeric = numeric
        self.categorical = categorical
        self.onehot_max_levels = onehot_max_levels
        self.clip_quantile = clip_quantile
        self.skew_threshold = skew_threshold
        self.smoothing = smoothing
        self.max_levels = max_levels
        self.min_level_share = min_level_share
        self.drop = tuple(drop)
        self.positive = positive
        self.columns: list[ColumnTransform] = []
        self.classes: list[str] = []
        self.target_lo = 0.0
        self.target_hi = 1.0
        self.source_columns: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------ fitting

    def fit(self, df: pd.DataFrame) -> "TableEncoder":
        if self.target not in df.columns:
            raise KeyError(f"target column {self.target!r} is not in the table")
        y = df[self.target]
        if y.isna().any():
            raise ValueError("the target has missing values; drop or impute those rows first")
        self.task = self._resolve_task(y)
        y_numeric = self._fit_target(y)
        self.source_columns = [c for c in df.columns if c != self.target and c not in self.drop]
        if not self.source_columns:
            raise ValueError("no feature columns remain")
        self.columns = [self._fit_column(df[name], y_numeric) for name in self.source_columns]
        self._fitted = True
        return self

    def _resolve_task(self, y: pd.Series) -> str:
        distinct = y.nunique()
        if self.task == "auto":
            numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
            if distinct == 2:
                return "binary"
            if not numeric:
                raise ValueError(f"target has {distinct} classes; DOPE supports regression and binary targets")
            return "regression"
        if self.task in ("binary", "classification"):
            if distinct != 2:
                raise ValueError(f"a binary task needs exactly 2 target classes, found {distinct}")
            return "binary"
        if self.task == "regression":
            return "regression"
        raise ValueError("task must be 'auto', 'regression' or 'binary'")

    def _fit_target(self, y: pd.Series) -> np.ndarray:
        if self.task == "binary":
            labels = sorted(pd.unique(y).tolist(), key=lambda v: (str(type(v)), str(v)))
            if self.positive is not None:
                if self.positive not in labels:
                    raise ValueError("positive class is not a target value")
                labels = [v for v in labels if v != self.positive] + [self.positive]
            self.classes = labels
            return (y == labels[1]).to_numpy(dtype=float)
        values = y.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("a regression target must be finite")
        self.target_lo, self.target_hi = float(values.min()), float(values.max())
        if self.target_hi <= self.target_lo:
            raise ValueError("the regression target is constant")
        return values

    def _fit_column(self, column: pd.Series, y: np.ndarray) -> ColumnTransform:
        name = str(column.name)
        if pd.api.types.is_datetime64_any_dtype(column):
            days = _datetime_days(column)
            spec = ColumnTransform(name, "datetime")
            self._fit_numeric(spec, days[np.isfinite(days)])
            return spec
        observed_values = column.dropna()
        boolean = bool(len(observed_values)) and all(isinstance(v, (bool, np.bool_)) for v in observed_values)
        if boolean:
            column = column.astype("object")
        if pd.api.types.is_numeric_dtype(column):
            values = column.to_numpy(dtype=float)
            observed = values[np.isfinite(values)]
            if observed.size == 0:
                raise ValueError(f"column {name!r} has no observed values")
            spec = ColumnTransform(name, "numeric", integer=bool(np.all(np.mod(observed, 1) == 0)))
            if np.unique(observed).size <= 2:
                # Indicator-like numeric column: an exact affine map keeps both values.
                spec.lo, spec.hi = float(observed.min()), float(observed.max())
                if not spec.hi > spec.lo:
                    spec.hi = spec.lo + 1.0
            else:
                self._fit_numeric(spec, observed)
            return spec
        spec = self._fit_categorical(name, column, y)
        spec.boolean = boolean
        return spec

    def _fit_categorical(self, name: str, column: pd.Series, y: np.ndarray) -> ColumnTransform:
        keys = _category_keys(column)
        counts = keys.value_counts()
        if len(counts) == 1:
            return ColumnTransform(name, "binary", levels=[counts.index[0], counts.index[0]])
        levels = self._pooled_levels(counts)
        keys = keys.where(keys.isin(levels), OTHER)
        if len(levels) == 2 and OTHER not in levels:
            ordered = sorted(levels, key=lambda lv: (-int(counts.get(lv, 0)), lv))
            return ColumnTransform(name, "binary", levels=ordered)
        if self.categorical == "onehot" or (self.categorical == "auto" and len(levels) <= self.onehot_max_levels):
            ordered = sorted(levels, key=lambda lv: (-int((keys == lv).sum()), lv))
            return ColumnTransform(name, "onehot", levels=ordered)
        return self._fit_ordered(name, keys, levels, y)

    def _fit_numeric(self, spec: ColumnTransform, observed: np.ndarray) -> None:
        if self.numeric == "quantile":
            spec.knots = np.quantile(observed, np.linspace(0.0, 1.0, 257)).tolist()
            return
        if self.numeric == "robust":
            skew = float(pd.Series(observed).skew()) if observed.size > 2 else 0.0
            spec.log = bool(observed.min() >= 0 and math.isfinite(skew) and skew > self.skew_threshold)
            warped = np.log1p(observed) if spec.log else observed
            q1, q3 = np.quantile(warped, [0.25, 0.75])
            span = q3 - q1
            q_lo, q_hi = np.quantile(warped, [self.clip_quantile, 1.0 - self.clip_quantile])
            if span > 0:
                # Tukey "far out" fences, never tighter than the 1%/99% quantiles.
                p_lo, p_hi = np.quantile(warped, [0.01, 0.99])
                q_lo = min(q1 - 3.0 * span, p_lo)
                q_hi = max(q3 + 3.0 * span, p_hi)
            lo, hi = max(q_lo, warped.min()), min(q_hi, warped.max())
        else:
            lo, hi = observed.min(), observed.max()
        spec.lo, spec.hi = float(lo), float(hi)
        if not spec.hi > spec.lo:
            spec.hi = spec.lo + 1.0

    def _pooled_levels(self, counts: pd.Series) -> list[str]:
        total = float(counts.sum())
        keep = [lv for lv, c in counts.items() if c / total >= self.min_level_share]
        if len(keep) == len(counts) and len(keep) <= self.max_levels:
            return keep
        return keep[: self.max_levels - 1] + [OTHER]

    def _fit_ordered(self, name: str, keys: pd.Series, levels: list[str], y: np.ndarray) -> ColumnTransform:
        prior = float(y.mean())
        frame = pd.DataFrame({"key": keys.to_numpy(), "y": y})
        stats = frame.groupby("key")["y"].agg(["sum", "count"])
        smoothed = {}
        for level in levels:
            s, c = (stats.loc[level, "sum"], stats.loc[level, "count"]) if level in stats.index else (0.0, 0)
            smoothed[level] = (float(s) + self.smoothing * prior) / (float(c) + self.smoothing)
        order = sorted(levels, key=lambda lv: (smoothed[lv], lv))
        means = np.array([smoothed[lv] for lv in order])
        span = means[-1] - means[0]
        rank = np.linspace(0.0, 1.0, len(order))
        position = (means - means[0]) / span if span > 0 else rank
        # A small rank term keeps every code distinct, so decoding is exact.
        position = 0.9 * position + 0.1 * rank
        position = (position - position[0]) / (position[-1] - position[0])
        return ColumnTransform(name, "ordered", levels=order, codes=[float(round(p, 6)) for p in position])

    # ------------------------------------------------------------------ mapping

    @property
    def encoded_columns(self) -> list[str]:
        """Names of the matrix columns, target last."""
        self._require_fit()
        names = []
        for spec in self.columns:
            if spec.kind == "onehot":
                names.extend(f"{spec.name}={level}" for level in spec.levels[1:])
            else:
                names.append(spec.name)
        return names + [self.target]

    @property
    def feature_groups(self) -> dict[str, list[int]]:
        """Matrix column indices that belong to each source feature."""
        self._require_fit()
        groups, index = {}, 0
        for spec in self.columns:
            groups[spec.name] = list(range(index, index + spec.width))
            index += spec.width
        return groups

    def transform(self, df: pd.DataFrame, *, with_target: bool = True) -> np.ndarray:
        self._require_fit()
        blocks = [self._encode(spec, df[spec.name]) for spec in self.columns]
        if with_target:
            blocks.append(self.encode_target(df[self.target])[:, None])
        return np.hstack(blocks).astype(np.float64)

    def encode_target(self, y: pd.Series) -> np.ndarray:
        if self.task == "binary":
            return (y == self.classes[1]).to_numpy(dtype=float)
        values = y.to_numpy(dtype=float)
        return np.clip((values - self.target_lo) / (self.target_hi - self.target_lo), 0.0, 1.0)

    def _encode(self, spec: ColumnTransform, column: pd.Series) -> np.ndarray:
        if spec.kind in ("numeric", "datetime"):
            values = _datetime_days(column) if spec.kind == "datetime" else column.to_numpy(dtype=float)
            return self._forward_numeric(spec, values)[:, None]
        keys = _category_keys(column)
        if spec.kind == "binary":
            return (keys == spec.levels[1]).to_numpy(dtype=float)[:, None]
        known = set(spec.levels)
        fallback = OTHER if OTHER in known else (MISSING if MISSING in known else spec.levels[0])
        keys = keys.where(keys.isin(known), fallback)
        if spec.kind == "onehot":
            return np.column_stack([(keys == level).to_numpy(dtype=float) for level in spec.levels[1:]])
        lookup = dict(zip(spec.levels, spec.codes))
        return keys.map(lookup).to_numpy(dtype=float)[:, None]

    def _forward_numeric(self, spec: ColumnTransform, values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, np.nan)
        ok = np.isfinite(values)
        if spec.knots:
            knots = np.asarray(spec.knots)
            out[ok] = np.interp(values[ok], knots, np.linspace(0.0, 1.0, knots.size))
            return out
        warped = np.log1p(np.clip(values[ok], 0.0, None)) if spec.log else values[ok]
        out[ok] = np.clip((warped - spec.lo) / (spec.hi - spec.lo), 0.0, 1.0)
        return out

    def _inverse_numeric(self, spec: ColumnTransform, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        if spec.knots:
            knots = np.asarray(spec.knots)
            values = np.interp(u, np.linspace(0.0, 1.0, knots.size), knots)
        else:
            warped = spec.lo + u * (spec.hi - spec.lo)
            values = np.expm1(warped) if spec.log else warped
        if spec.integer:
            values = np.round(values)
        return values

    def inverse_transform(self, matrix: np.ndarray, *, seed: int = 0) -> pd.DataFrame:
        """Map DOPE samples (features then target, NaN for missing) to the source schema."""
        self._require_fit()
        matrix = np.asarray(matrix, dtype=float)
        expected = sum(spec.width for spec in self.columns) + 1
        if matrix.ndim != 2 or matrix.shape[1] != expected:
            raise ValueError(f"expected {expected} columns, found {matrix.shape[-1]}")
        rng = np.random.default_rng(seed)
        out, index = {}, 0
        for spec in self.columns:
            block = matrix[:, index:index + spec.width]
            index += spec.width
            out[spec.name] = self._decode(spec, block, rng)
        y = matrix[:, -1]
        if self.task == "binary":
            labels = np.array(self.classes, dtype=object)
            out[self.target] = labels[(np.nan_to_num(y, nan=0.0) >= 0.5).astype(int)]
        else:
            out[self.target] = self.target_lo + np.clip(y, 0.0, 1.0) * (self.target_hi - self.target_lo)
        return pd.DataFrame(out)[self.source_columns + [self.target]]

    def _decode(self, spec: ColumnTransform, block: np.ndarray, rng) -> pd.Series:
        if spec.kind in ("numeric", "datetime"):
            values = self._inverse_numeric(spec, block[:, 0])
            if spec.kind == "datetime":
                return pd.Series(pd.to_datetime(values, unit="D", origin="unix", errors="coerce"))
            if spec.integer and not np.isnan(values).any():
                return pd.Series(values.astype(np.int64))
            return pd.Series(values)
        if spec.kind == "binary":
            labels = np.array(spec.levels, dtype=object)
            picks = labels[(np.nan_to_num(block[:, 0], nan=0.0) >= 0.5).astype(int)]
        elif spec.kind == "onehot":
            scores = np.nan_to_num(block, nan=0.0)
            best = (scores + rng.random(scores.shape) * 1e-6).argmax(axis=1) + 1
            best[scores.max(axis=1) < 0.5] = 0
            picks = np.array(spec.levels, dtype=object)[best]
        else:
            codes = np.asarray(spec.codes)
            u = np.nan_to_num(block[:, 0], nan=0.5)
            picks = np.array(spec.levels, dtype=object)[np.abs(u[:, None] - codes[None, :]).argmin(axis=1)]
        series = pd.Series(picks, dtype=object)
        if spec.boolean:
            series = series.map({"True": True, "False": False, MISSING: None})
        return series.where(series != MISSING, None)

    # ------------------------------------------------------------------ state

    def to_bytes(self, *, labels: bool = True) -> bytes:
        """Compact binary state: everything ``inverse_transform`` needs.

        Layout: magic, task byte, target bounds (2 x float32), column count, then per
        column a kind byte, flags and float32 parameters, or a level count and
        uint16 codes. Names and category labels follow as one zlib-compressed,
        NUL-separated block; ``labels=False`` omits them (decoded labels then
        become ``level_<i>`` and columns ``x<i>``).
        """
        self._require_fit()
        head = bytearray(ENCODER_MAGIC)
        head += struct.pack("<BffH", 1 if self.task == "binary" else 0, self.target_lo, self.target_hi,
                            len(self.columns))
        text: list[str] = [self.target] + [str(c) for c in self.classes]
        for spec in self.columns:
            kind = _KINDS.index(spec.kind)
            if spec.kind in ("numeric", "datetime"):
                flags = int(spec.log) | int(spec.integer) << 1 | int(bool(spec.knots)) << 2
                head += struct.pack("<BB", kind, flags)
                if spec.knots:
                    head += struct.pack("<H", len(spec.knots)) + np.asarray(spec.knots, "<f4").tobytes()
                else:
                    head += struct.pack("<ff", spec.lo, spec.hi)
            else:
                head += struct.pack("<BB", kind, len(spec.levels))
                if spec.kind == "ordered":
                    head += np.round(np.asarray(spec.codes) * 65535).astype("<u2").tobytes()
            text.append(spec.name)
            text.extend(str(level) for level in spec.levels)
        block = zlib.compress("\0".join(text).encode("utf-8"), 9) if labels else b""
        return bytes(head) + struct.pack("<I", len(block)) + block

    def byte_breakdown(self) -> dict[str, int]:
        full = len(self.to_bytes(labels=True))
        bare = len(self.to_bytes(labels=False))
        return {"transform_bytes": bare, "label_bytes": full - bare, "total_bytes": full}

    def to_json(self) -> dict:
        self._require_fit()
        return {
            "format": "dope-table-encoder-v1",
            "target": self.target, "task": self.task, "classes": [str(c) for c in self.classes],
            "target_range": [self.target_lo, self.target_hi] if self.task == "regression" else None,
            "options": {"numeric": self.numeric, "categorical": self.categorical,
                        "onehot_max_levels": self.onehot_max_levels,
                        "clip_quantile": self.clip_quantile, "skew_threshold": self.skew_threshold,
                        "smoothing": self.smoothing, "max_levels": self.max_levels},
            "columns": [
                {k: v for k, v in spec.__dict__.items() if v not in ([], None) and not (k in ("knots", "boolean") and not v)}
                for spec in self.columns
            ],
            "encoded_columns": self.encoded_columns,
        }

    def write_dataset_dir(self, df: pd.DataFrame, out_dir: str | Path) -> Path:
        """Write ``train.csv`` (headerless, target last, blanks for missing) plus the encoder state."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        matrix = self.transform(df)
        with (out / "train.csv").open("w") as stream:
            for row in matrix:
                stream.write(",".join("" if not np.isfinite(v) else f"{v:.9g}" for v in row) + "\n")
        (out / "encoder.bin").write_bytes(self.to_bytes())
        (out / "encoder.json").write_text(json.dumps(self.to_json(), indent=1, default=str) + "\n")
        return out

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() first")


def _category_keys(column: pd.Series) -> pd.Series:
    keys = column.astype(object).where(column.notna(), MISSING)
    return keys.map(lambda v: v if isinstance(v, str) else (str(v) if v is not None else MISSING))


def _datetime_days(column: pd.Series) -> np.ndarray:
    """Days since the Unix epoch as floats, independent of the datetime resolution."""
    stamps = pd.to_datetime(column, errors="coerce")
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert(None)
    days = (stamps - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    return days.to_numpy(dtype=float)


def read_dope_samples(path: str | Path) -> np.ndarray:
    """Read the headerless CSV written by ``dope-kernel sample`` (``nan`` marks missing)."""
    return pd.read_csv(path, header=None, na_values=["nan", "NaN", ""]).to_numpy(dtype=float)
