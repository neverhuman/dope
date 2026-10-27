"""Optional / progressive CV landmarks (LandmarkBundle v1) for the MLC-v3 conditioning path.

Offline research reference; the Rust ``dope_core::landmark::bundle`` port mirrors it.
Nothing here changes the strict fit-only path in ``landmark_features`` (which still
refuses NaN and empty masks); this module is the explicit, non-raising sibling.

Contract (shared with Rust, see ``/home/ubuntu/jope_plan/comms.md``):

* ``values``: 23 x f32 SQUASHED gains ``s = g/(1+|g|)``. The squash is evaluated in
  float32 arithmetic on ``f32(g)`` exactly like ``landmark_features.feature_tensor``,
  so ``features46()`` of a full bundle is bit-identical to the strict 46-d tensor.
  Missing slots are exactly ``+0.0``.
* ``mask``: 23 x bool. Non-finite (or f32-overflowing) inputs become ``+0.0``/missing.
* ``status``: derived from the sanitized mask: ``"none"`` | ``{"partial": n}`` | ``"full"``.
* ``subsample_n``: ``int`` or ``None`` (full training partition).
* ``source_digest``: ``"zero"`` for the zero bundle, otherwise a sha256 hex string.
* ``provenance``: free-form string or ``None``.
* ``conditioning_digest`` = sha256(b"dope.landmark-bundle/v1\\0" || b"landmarks-native-v1\\0"
  || 23 x f32 LE values || 23 x u8 mask || u64 LE subsample_n (u64::MAX for None)).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .descriptors import FINE_ADDED_INDICES, LANDMARK_NAMES, LARGER_ADDED_INDICES, MACRO_INDICES

LARGER_INDICES = LARGER_ADDED_INDICES
FINE_INDICES = FINE_ADDED_INDICES

PANEL = "landmarks-native-v1"
LANDMARK_SLOTS = 23
ZERO_SOURCE_DIGEST = "zero"
DIGEST_DOMAIN = b"dope.landmark-bundle/v1\0"
DIGEST_PANEL = PANEL.encode() + b"\0"
SUBSAMPLE_NONE_SENTINEL = 0xFFFFFFFFFFFFFFFF
DEFAULT_PROGRESSIVE_SUBSAMPLE = 300
LANDMARK_MODES = ("none", "fast", "progressive", "full")
F32_MAX = 3.4028234663852886e38

# Progressive groups are the descriptors.py tier constants (0-based slots), in order.
PROGRESSIVE_GROUPS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("macro", tuple(MACRO_INDICES)),
    ("larger", tuple(LARGER_ADDED_INDICES)),
    ("fine", tuple(FINE_ADDED_INDICES)),
)
assert sorted(s for _, g in PROGRESSIVE_GROUPS for s in g) == list(range(LANDMARK_SLOTS))


class LandmarkScopeError(ValueError):
    """Raised when merging bundles measured on incompatible scopes."""


def squash_f32(gain: float) -> Optional[np.float32]:
    """f32 squash identical to ``landmark_features.feature_tensor`` for ``|g| <= F32_MAX``.

    Finite gains with ``|g| > F32_MAX`` squash in f64 before converting to f32 so
    finite observations stay present. Non-finite inputs return ``None``.
    """
    if isinstance(gain, bool):
        return None
    try:
        g = float(gain)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(g):
        return None
    if abs(g) > F32_MAX:
        s = np.float32(g / (1.0 + abs(g)))
    else:
        v = np.float32(g)
        if not np.isfinite(v):
            s = np.float32(g / (1.0 + abs(g)))
        else:
            s = np.float32(v / (np.float32(1.0) + np.abs(v)))
    return np.float32(0.0) if s == 0.0 else s


def unsquash(value: float) -> float:
    """Float64 inverse of the squash used when synthesizing a report from a bundle.

    ``|s| >= 1`` (only reachable through f32 rounding of huge gains) maps to +-F32_MAX.
    """
    s = float(value)
    if abs(s) >= 1.0:
        return math.copysign(F32_MAX, s)
    return s / (1.0 - abs(s))


def _status_of(mask: Sequence[bool]) -> Any:
    n = int(sum(bool(m) for m in mask))
    if n == 0:
        return "none"
    if n == LANDMARK_SLOTS:
        return "full"
    return {"partial": n}


@dataclass(frozen=True)
class LandmarkBundle:
    """Immutable sanitized landmark bundle. Construct through the classmethods."""

    values: Tuple[float, ...]  # 23 squashed f32 values stored as exact Python floats
    mask: Tuple[bool, ...]
    subsample_n: Optional[int] = None
    source_digest: str = ZERO_SOURCE_DIGEST
    provenance: Optional[str] = None
    panel: str = field(default=PANEL)

    # ----------------------------------------------------------------- constructors
    @classmethod
    def zero(cls) -> "LandmarkBundle":
        return cls(values=(0.0,) * LANDMARK_SLOTS, mask=(False,) * LANDMARK_SLOTS)

    @classmethod
    def _build(cls, pairs: Sequence[Tuple[Optional[np.float32], bool]], *, subsample_n: Optional[int],
               source_digest: Optional[str], provenance: Optional[str], panel: str = PANEL) -> "LandmarkBundle":
        if panel != PANEL:
            raise LandmarkScopeError("landmark_bundle_panel")
        if len(pairs) != LANDMARK_SLOTS:
            raise ValueError("landmark_bundle_shape")
        if subsample_n is not None and (type(subsample_n) is not int or subsample_n < 1):
            raise ValueError("landmark_bundle_subsample_n")
        values: List[float] = []
        mask: List[bool] = []
        for v, present in pairs:
            ok = (bool(present) and v is not None and not isinstance(v, bool)
                  and bool(np.isfinite(v)) and abs(float(v)) <= 1.0)
            values.append(float(np.float32(v)) + 0.0 if ok else 0.0)
            mask.append(ok)
        if source_digest is None:
            if not any(mask) and subsample_n is None:
                source_digest = ZERO_SOURCE_DIGEST
            else:
                source_digest = _values_digest(values, mask, subsample_n)
        else:
            source_digest = str(source_digest)
            if not source_digest.strip():
                raise ValueError("landmark_bundle_source_digest")
            if source_digest == ZERO_SOURCE_DIGEST and (any(mask) or subsample_n is not None):
                raise ValueError("landmark_bundle_zero_digest")
        return cls(values=tuple(values), mask=tuple(mask), subsample_n=subsample_n,
                   source_digest=str(source_digest), provenance=provenance, panel=panel)

    @classmethod
    def from_raw_gains(cls, gains: Sequence[Optional[float]], *, subsample_n: Optional[int] = None,
                       source_digest: Optional[str] = None, provenance: Optional[str] = None,
                       panel: str = PANEL) -> "LandmarkBundle":
        """Raw CV gains (``None``/NaN/inf = missing) -> squashed bundle."""
        if panel != PANEL:
            raise LandmarkScopeError("landmark_bundle_panel")
        if len(gains) != LANDMARK_SLOTS:
            raise ValueError("landmark_bundle_shape")
        pairs = []
        for g in gains:
            s = None if g is None else squash_f32(g)
            pairs.append((s, s is not None))
        return cls._build(pairs, subsample_n=subsample_n, source_digest=source_digest,
                          provenance=provenance, panel=panel)

    @classmethod
    def from_squashed(cls, values: Sequence[float], mask: Sequence[bool], *, subsample_n: Optional[int] = None,
                      source_digest: Optional[str] = None, provenance: Optional[str] = None,
                      panel: str = PANEL) -> "LandmarkBundle":
        """Already-squashed values. Non-finite or ``|s| > 1`` -> missing; missing -> +0.0."""
        if panel != PANEL:
            raise LandmarkScopeError("landmark_bundle_panel")
        if len(values) != LANDMARK_SLOTS or len(mask) != LANDMARK_SLOTS:
            raise ValueError("landmark_bundle_shape")
        pairs = []
        for v, m in zip(values, mask):
            if isinstance(v, bool) or v is None:
                pairs.append((None, False))
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                pairs.append((None, False))
                continue
            if not math.isfinite(f) or abs(f) > 1.0:
                pairs.append((None, False))
                continue
            fv = np.float32(f)
            if not np.isfinite(fv) or abs(float(fv)) > 1.0:
                pairs.append((None, False))
                continue
            pairs.append((np.float32(0.0) if fv == 0.0 else fv, bool(m)))
        return cls._build(pairs, subsample_n=subsample_n, source_digest=source_digest,
                          provenance=provenance, panel=panel)

    @classmethod
    def from_report(cls, report: Dict[str, Any], subsample_n: Optional[int] = None, *,
                    source_digest: Optional[str] = None, provenance: Optional[str] = "native-v1",
                    include_uninformative: bool = True, complete_only: bool = False) -> "LandmarkBundle":
        """Non-raising view of a native ``landmarks real`` report (any subset of cells).

        By default (``include_uninformative=True``, ``complete_only=False``), a cell is
        present iff ``state in ("complete", "uninformative")`` and its gain is finite, so
        finite measured ``0.0`` gains marked ``"uninformative"`` remain present.
        Pass ``complete_only=True`` (or ``include_uninformative=False``) for strict
        ``FEATURE_RECIPE`` (``state == "complete"`` only) parity. Unknown ids are ignored.
        ``source_digest`` defaults to sha256 of the canonical report JSON.
        """
        panel = report.get("panel", PANEL)
        if panel != PANEL:
            raise LandmarkScopeError("landmark_bundle_panel")
        admit_uninformative = bool(include_uninformative) and not bool(complete_only)
        states = ("complete", "uninformative") if admit_uninformative else ("complete",)
        gains: List[Optional[float]] = [None] * LANDMARK_SLOTS
        for cell in report.get("cells") or []:
            try:
                idx = int(cell.get("id")) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= idx < LANDMARK_SLOTS or cell.get("state") not in states:
                continue
            g = cell.get("gain")
            if isinstance(g, bool) or not isinstance(g, (int, float)):
                continue
            gains[idx] = float(g)
        if source_digest is None:
            source_digest = hashlib.sha256(
                json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()
        return cls.from_raw_gains(gains, subsample_n=subsample_n, source_digest=source_digest,
                                  provenance=provenance, panel=panel)

    @classmethod
    def from_report_complete(cls, report: Dict[str, Any], subsample_n: Optional[int] = None,
                             **kwargs: Any) -> "LandmarkBundle":
        """Strict ``complete``-only view of a native report (``FEATURE_RECIPE`` admission)."""
        return cls.from_report(report, subsample_n, complete_only=True, **kwargs)

    # ----------------------------------------------------------------- derived
    @property
    def status(self) -> Any:
        return _status_of(self.mask)

    @property
    def present(self) -> int:
        return int(sum(self.mask))

    def is_zero(self) -> bool:
        return (
            not any(self.mask)
            and self.subsample_n is None
            and self.source_digest == ZERO_SOURCE_DIGEST
        )

    def is_full(self) -> bool:
        return all(self.mask)

    def same_landmarks(self, other: "LandmarkBundle") -> bool:
        """Values (bitwise as f32) and mask equality; ignores scope/provenance."""
        return self.mask == other.mask and _f32_bytes(self.values) == _f32_bytes(other.values)

    def features46(self, device: Any = None):
        """cat(values, mask as 0/1) -> np.float32[46], or a torch FP32 tensor on ``device``."""
        arr = np.concatenate([np.asarray(self.values, dtype=np.float32),
                              np.asarray(self.mask, dtype=np.float32)]).astype(np.float32)
        if device is None:
            return arr
        import torch
        return torch.from_numpy(arr.copy()).to(device=device, dtype=torch.float32)

    def conditioning_digest(self) -> str:
        h = hashlib.sha256()
        h.update(DIGEST_DOMAIN)
        h.update(DIGEST_PANEL)
        h.update(_f32_bytes(self.values))
        h.update(bytes(1 if m else 0 for m in self.mask))
        h.update(struct.pack("<Q", SUBSAMPLE_NONE_SENTINEL if self.subsample_n is None else int(self.subsample_n)))
        return h.hexdigest()

    def sanitize(self) -> "LandmarkBundle":
        """Sanitize values and mask (non-finite, |v|>1 or masked-out -> +0.0/False) and validate scope."""
        return self.from_squashed(
            self.values, self.mask, subsample_n=self.subsample_n,
            source_digest=self.source_digest, provenance=self.provenance, panel=self.panel,
        )

    def validate(self) -> "LandmarkBundle":
        if self.panel != PANEL or len(self.values) != LANDMARK_SLOTS or len(self.mask) != LANDMARK_SLOTS:
            raise ValueError("landmark_bundle_shape")
        for v, m in zip(self.values, self.mask):
            if type(m) is not bool:
                raise ValueError("landmark_bundle_mask_type")
            if isinstance(v, bool):
                raise ValueError("landmark_bundle_value")
            f = float(v)
            if not math.isfinite(f) or abs(f) > 1.0 or float(np.float32(f)) != f:
                raise ValueError("landmark_bundle_value")
            if f == 0.0 and math.copysign(1.0, f) < 0:
                raise ValueError("landmark_bundle_negative_zero")
            if not m and f != 0.0:
                raise ValueError("landmark_bundle_missing_value_nonzero")
        if self.subsample_n is not None and (type(self.subsample_n) is not int or self.subsample_n < 1):
            raise ValueError("landmark_bundle_subsample_n")
        if not isinstance(self.source_digest, str) or not self.source_digest.strip():
            raise ValueError("landmark_bundle_source_digest")
        if self.source_digest == ZERO_SOURCE_DIGEST and (any(self.mask) or self.subsample_n is not None):
            raise ValueError("landmark_bundle_zero_digest")
        return self

    # ----------------------------------------------------------------- merge
    def merge(self, incoming: "LandmarkBundle") -> "LandmarkBundle":
        """Per slot incoming wins where present, else base; mask = union; never downgrades.

        Both operands are sanitized first. Compatible scope: same panel, same
        ``subsample_n`` and same ``source_digest``. Only the explicit zero sentinel
        (``is_zero()``) is a universal initializer and adopts the other side's scope;
        scoped empty bundles still enforce scope compatibility.
        """
        if not isinstance(incoming, LandmarkBundle):
            raise TypeError("landmark_bundle_required")
        if self.panel != PANEL or incoming.panel != PANEL:
            raise LandmarkScopeError("landmark_bundle_panel")
        base = self.sanitize()
        inc = incoming.sanitize()
        if inc.is_zero():
            return base
        if base.is_zero():
            return inc
        if base.subsample_n != inc.subsample_n:
            raise LandmarkScopeError("landmark_bundle_subsample_scope")
        if base.source_digest != inc.source_digest:
            raise LandmarkScopeError("landmark_bundle_source_scope")
        values = tuple(iv if im else bv for bv, bm, iv, im in zip(base.values, base.mask, inc.values, inc.mask))
        mask = tuple(bm or im for bm, im in zip(base.mask, inc.mask))
        if base.provenance == inc.provenance:
            prov = base.provenance
        else:
            parts = set()
            for p in (base.provenance, inc.provenance):
                if p:
                    parts.update(p.split("+"))
            prov = "+".join(sorted(parts)) or None
        return LandmarkBundle(values=values, mask=mask, subsample_n=base.subsample_n,
                              source_digest=base.source_digest, provenance=prov, panel=base.panel)

    # ----------------------------------------------------------------- serde
    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": [float(v) for v in self.values],
            "mask": [bool(m) for m in self.mask],
            "status": self.status,
            "subsample_n": self.subsample_n,
            "source_digest": self.source_digest,
            "provenance": self.provenance,
        }

    def to_json(self) -> Dict[str, Any]:
        return self.to_dict()

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, obj: Any) -> "LandmarkBundle":
        if not isinstance(obj, dict) or set(obj) - {
            "values", "mask", "status", "subsample_n", "source_digest", "provenance", "panel"
        }:
            raise ValueError("landmark_bundle_json_fields")
        panel = obj.get("panel", PANEL)
        if panel != PANEL:
            raise LandmarkScopeError("landmark_bundle_panel")
        if not isinstance(obj.get("values"), list) or not isinstance(obj.get("mask"), list):
            raise ValueError("landmark_bundle_json_fields")
        if any(type(m) is not bool for m in obj["mask"]):
            raise ValueError("landmark_bundle_mask_type")
        sub = obj.get("subsample_n")
        return cls.from_squashed(
            obj["values"], obj["mask"], subsample_n=sub,
            source_digest=obj.get("source_digest"), provenance=obj.get("provenance"),
            panel=panel,
        )

    @classmethod
    def from_json(cls, obj: Any) -> "LandmarkBundle":
        if isinstance(obj, (str, bytes, bytearray)):
            obj = json.loads(obj)
        return cls.from_dict(obj)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _values_digest(values: Sequence[float], mask: Sequence[bool], subsample_n: Optional[int]) -> str:
    h = hashlib.sha256(b"dope.landmark-bundle-values/v1\0")
    h.update(_f32_bytes(values))
    h.update(bytes(1 if m else 0 for m in mask))
    h.update(struct.pack("<Q", SUBSAMPLE_NONE_SENTINEL if subsample_n is None else int(subsample_n)))
    return h.hexdigest()


# --------------------------------------------------------------------------- report synthesis
def report_task(task: str) -> str:
    """Native report task string for an encode task name."""
    if task == "regression":
        return "regression"
    if task in ("classification", "binary", "multiclass"):
        return "classification"
    raise ValueError(f"unknown_task_{task}")


def table_classes(y: np.ndarray, task: str) -> int:
    """Native ``classes``: 0 regression, 2 binary, k unique labels for multiclass."""
    if task == "regression":
        return 0
    if task in ("classification", "binary"):
        return 2
    return int(len(np.unique(np.rint(np.asarray(y, dtype=np.float64).reshape(-1)))))


def table_target_stats(y: Any, task: str) -> List[Optional[float]]:
    """Native ``target_stats`` formula (dope-core landmark/signature.rs) evaluated on ``y``.

    Used only when no native report is available. NOTE: when ``y`` is the scaled
    regression target (8*tanh(z/8)) skew/kurtosis are those of the scaled target.
    """
    if isinstance(y, np.ndarray) and y.dtype.kind in ("f", "i", "u"):
        arr = np.asarray(y, dtype=np.float64).reshape(-1)
    else:
        from .scaling import _parse_float
        arr = np.asarray([_parse_float(v) for v in y], dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return [None, None, None]
    if task == "regression":
        m = float(np.sum(arr) / n)
        d = arr - m
        v = float(np.sum(d ** 2) / n)
        if v <= 1e-24:
            return [None, None, None]
        return [float(np.sum(d ** 3) / n / v ** 1.5), float(np.sum(d ** 4) / n / v ** 2 - 3.0), None]
    labels = np.rint(arr).astype(np.int64)
    classes = [0, 1] if task in ("classification", "binary") else sorted(set(int(v) for v in labels))
    slot = {c: i for i, c in enumerate(classes)}
    counts = np.zeros(len(classes), dtype=np.float64)
    for lab in labels:  # accumulate 1/rows like the native loop
        if int(lab) in slot:
            counts[slot[int(lab)]] += 1.0 / n
    if counts.size < 2:
        return [None, None, None]
    pos = counts[counts > 0.0]
    ent = -float(np.sum(pos * np.log(pos))) / math.log(counts.size)
    return [ent, float(np.min(counts)), None]


def bundle_to_report(bundle: LandmarkBundle, *, rows: int, features: int, task: str, classes: int,
                     base_report: Optional[Dict[str, Any]] = None,
                     target_stats: Optional[Sequence[Optional[float]]] = None) -> Dict[str, Any]:
    """Native-shaped report for the optional conditioning path.

    * If ``base_report`` is a native report whose landmarks equal the (full) bundle and
      whose rows/features match the table, it is returned UNCHANGED (bit-identity).
    * Otherwise a new report: rows/features/task/classes from the TABLE; cells ONLY for
      present slots (state ``"complete"``). A present slot reuses the native cell from
      ``base_report`` verbatim when its gain squashes to the bundle value (keeps se,
      stable, lambda_indices, train/query rows); else it gets ``gain = unsquash(value)``
      and the documented defaults ``se=None, stable=False, lambda_indices=[],
      train_rows=[], query_rows=0`` (the native "not evaluated" defaults).
    * spectrum: ``base_report["spectrum"]`` if present, else ``[]``.
    * target_stats: ``base_report["target_stats"]`` if present, else ``target_stats``
      (table-derived), else ``[None, None, None]``.
    """
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
    if base_report is not None and base_report.get("target_stats") is not None:
        t_stats = list(base_report["target_stats"])
    elif target_stats is not None:
        t_stats = list(target_stats)
    else:
        t_stats = [None, None, None]
    report: Dict[str, Any] = {
        "panel": PANEL, "kind": "real", "privacy": "private", "task": report_task(task),
        "classes": int(classes), "rows": rows, "features": features, "cells": cells,
        "target_stats": t_stats,
        "spectrum": list(base_report.get("spectrum") or []) if base_report is not None else [],
    }
    if base_report is not None and "options" in base_report:
        report["options"] = base_report["options"]
    return report


# --------------------------------------------------------------------------- subsampling
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
        return z ^ (z >> np.uint64(31))


def _row_keys(n_rows: int, seed: int, row_ids: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """key(i) = splitmix64(splitmix64(seed) XOR id_i) with u64 wrap-around."""
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
    """Integer apportionment of ``total`` by ``weights``; ties -> lower stratum index."""
    w_sum = int(sum(weights))
    if w_sum <= 0:
        return [0] * len(weights)
    base = [total * int(w) // w_sum for w in weights]
    rem = [total * int(w) % w_sum for w in weights]
    left = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-rem[i], i))
    for i in order[:left]:
        base[i] += 1
    return base


def stratified_subsample(y: Sequence[float], task: str, n: int, seed: int = 0,
                         row_ids: Optional[Sequence[int]] = None) -> np.ndarray:
    """Deterministic stratified row subsample; returns sorted int64 indices.

    * ``n >= rows`` -> all rows.
    * regression: rows ranked by (y, index) (stable), split into ``B = min(10, n)``
      contiguous equal-size rank bins (first ``rows % B`` bins one larger); quotas by
      largest remainder of ``n`` over bin sizes.
    * classification/binary/multiclass: strata = unique labels (sorted); each class
      gets 2 then the remaining ``n - 2k`` is apportioned by largest remainder over
      ``count_c - 2``. Raises ``ValueError`` if any class has < 2 rows or ``n < 2k``.
    * within a stratum rows are taken in ascending ``(key(i), id(i), i)`` with
      ``key(i) = splitmix64(splitmix64(seed) ^ id(i))`` (u64), so Rust can mirror it exactly.
    """
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
        b = min(10, n)
        strata = [s.astype(np.int64) for s in np.array_split(order, b)]
        quotas = _largest_remainder(n, [len(s) for s in strata])
    elif task in ("classification", "binary", "multiclass"):
        labels = np.unique(arr)
        strata = [np.flatnonzero(arr == lab).astype(np.int64) for lab in labels]
        counts = [len(s) for s in strata]
        if min(counts) < 2 or n < 2 * len(strata):
            raise ValueError("subsample_class_too_small")
        extra = _largest_remainder(n - 2 * len(strata), [c - 2 for c in counts])
        quotas = [2 + e for e in extra]
    else:
        raise ValueError(f"unknown_task_{task}")
    picked = []
    for idx, q in zip(strata, quotas):
        if q <= 0:
            continue
        k = keys[idx]
        sel = np.lexsort((idx, rids[idx], k))[:q]
        picked.append(idx[sel])
    return np.sort(np.concatenate(picked)).astype(np.int64)


# --------------------------------------------------------------------------- progressive
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


def progressive_steps(compute_group: Callable[[Tuple[int, ...]], Dict[str, Any]], *,
                      subsample_n: Optional[int] = DEFAULT_PROGRESSIVE_SUBSAMPLE,
                      source_digest: Optional[str] = None, provenance: Optional[str] = None,
                      groups: Sequence[Tuple[str, Tuple[int, ...]]] = PROGRESSIVE_GROUPS,
                      start: Optional[LandmarkBundle] = None,
                      ) -> Iterator[Tuple[int, str, LandmarkBundle, Optional[Dict[str, Any]]]]:
    """Yield ``(seq, group_name, bundle, merged_native_report)``: start (zero), then +each group.

    ``compute_group(slots)`` returns a native report for those 0-based slots. Only
    cells inside the requested group are admitted from each group report. With a cached
    ``start`` bundle, groups whose slots are all already present are not recomputed.
    """
    if provenance is None:
        provenance = "native-v1" if subsample_n is None else f"subsample-{subsample_n}-stratified"
    digest = source_digest or hashlib.sha256(
        f"dope.progressive/v1\0{subsample_n}".encode()).hexdigest()
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


def progressive_bundles(compute_group: Callable[[Tuple[int, ...]], Dict[str, Any]], **kwargs) -> Iterator[LandmarkBundle]:
    """zero -> +macro -> +larger -> +fine bundles (see ``progressive_steps``)."""
    for _, _, bundle, _ in progressive_steps(compute_group, **kwargs):
        yield bundle


def native_group_runner(engine: str | Path, csv_path: str | Path, *, target: int, task: str,
                        subsample_n: Optional[int] = DEFAULT_PROGRESSIVE_SUBSAMPLE, seed: int = 0,
                        options: Optional[Dict[str, Any]] = None, threads: int = 4,
                        wall_seconds: int = 300) -> Callable[[Tuple[int, ...]], Dict[str, Any]]:
    """Thin default ``compute_group``: ``dope landmarks real ... --ids <group> [--subsample N]``.

    GUARDED: ``--ids`` / ``--subsample`` / ``--subsample-seed`` are being added to the native
    CLI by the ol-contract lane; an engine without them fails with a non-zero exit, which
    is raised as ``RuntimeError("native_group_runner_failed")``. Tests inject fake runners.
    """
    from .landmark_features import OPTIONS
    opts = dict(OPTIONS if options is None else options)
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
    """Canonical CSV (``c0..c{p-1},target``) used for native runs and source digests."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    X = np.asarray(X, dtype=np.float64)
    writer.writerow([f"c{i}" for i in range(X.shape[1])] + ["target"])
    for row, t in zip(X, np.asarray(y, dtype=np.float64).reshape(-1)):
        writer.writerow([repr(float(v)) for v in row] + [repr(float(t))])
    return stream.getvalue().encode()


# --------------------------------------------------------------------------- observed-only conditioning
OBSERVED_RECIPE = "observed-only-v1"
LEGACY_FULL_RECIPE = "legacy-full-v1"
CONDITIONING_DOMAIN = b"dope.conditioning/v1\0"


def observed_target_stats(
    y: np.ndarray,
    task: str,
    *,
    y_raw: Optional[np.ndarray] = None,
    target_stats: Optional[Sequence[Optional[float]]] = None,
) -> List[Optional[float]]:
    """target_stats for none/progressive.

    * If ``target_stats`` is explicitly provided (e.g. cached on ``PreparedTable``), returns it.
    * If ``y_raw`` is provided (unscaled training target), computes the exact native formula
      (skewness and excess kurtosis for regression, entropy/min-share for classification).
    * Classification/binary/multiclass on ``y`` (scaled labels): native entropy/min-share
      formula on the label indices (label-permutation invariant, so exact on scaled labels).
    * Regression when only tanh-scaled ``y`` is passed (``y_raw is None`` and
      ``target_stats is None``): ``[None, None, None]`` (-> 0).
    """
    if target_stats is not None:
        ts = list(target_stats)
        return (ts + [None, None, None])[:3]
    if y_raw is not None:
        return table_target_stats(y_raw, task)
    if task == "regression":
        return [None, None, None]
    return table_target_stats(y, task)


def observed_report_descriptors(bundle: LandmarkBundle, *, rows: int, features: int, classes: int, task: str,
                                target_stats: Sequence[Optional[float]], level: str = "high") -> Dict[str, Any]:
    """Observed-only replacement for ``extract_report_descriptors`` (same keys).

    present = bundle.mask & level_landmark_mask(level); s = f64(bundle value) where present.
    raw = unsquash(s) (f64 ``s/(1-|s|)``) where present. vector116 layout:

    * [0:23]  s (0 where absent)          * [23:46] present (0/1)
    * [46:69] midranks of raw among present (legacy formula)
    * [69:85] compute_level_contrasts(raw, present, level) (missing -> NaN -> 0, legacy)
    * [85:87] squash(target_stats[0:2]) (None/non-finite -> 0)
    * [87:91] shape: is_clf, log10(rows)/6, log2(feats)/12, log10(feats/rows)/4 (table shape)
    * [91:107] spectrum summary: 0 (no spectrum measured)
    * [107:116] group: max s over present in macro / larger (high+) / fine (ultra);
      MEAN s over PRESENT slots in macro / larger (high+) / fine (ultra) (0 if none);
      mean SE, stability, lambda: 0 (unmeasured).
    column_spectrum: zeros (feats, 9); column_prior_importance: uniform 1/feats.
    """
    from .descriptors import (
        FINE_ADDED_INDICES as FINE, LARGER_ADDED_INDICES as LARGER, MACRO_INDICES as MACRO,
        compute_level_contrasts, level_landmark_mask, squash,
    )
    lvl_mask = level_landmark_mask(level)
    present = np.asarray(bundle.mask, dtype=bool) & lvl_mask
    s = np.where(present, np.asarray(bundle.values, dtype=np.float32).astype(np.float64), 0.0)
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
    shape_vec = np.array([is_clf, math.log10(rows_c) / 6.0, math.log2(feats_c) / 12.0,
                          math.log10(feats_c / float(rows_c)) / 4.0], dtype=np.float64)

    def _gmax(idxs, gate):
        sel = [s[i] for i in idxs if present[i]]
        return float(max(sel)) if gate and sel else 0.0

    def _gmean(idxs, gate):
        sel = [s[i] for i in idxs if present[i]]
        return float(np.mean(sel)) if gate and sel else 0.0

    larger_gate, fine_gate = level in ("high", "ultra"), level == "ultra"
    group = np.array([
        _gmax(MACRO, True), _gmax(LARGER, larger_gate), _gmax(FINE, fine_gate),
        _gmean(MACRO, True), _gmean(LARGER, larger_gate), _gmean(FINE, fine_gate),
        0.0, 0.0, 0.0,
    ], dtype=np.float64)
    spec_summary = np.zeros(16, dtype=np.float64)
    vector = np.concatenate([s, present.astype(np.float64), midranks, contrasts, t_stats, shape_vec,
                             spec_summary, group])
    return {
        "level": level, "gains23": s, "ses23": np.zeros(23), "lambdas23": np.zeros(23), "mask23": present,
        "midranks23": midranks, "contrasts16": contrasts, "target_stats2": t_stats, "shape4": shape_vec,
        "spectrum_summary16": spec_summary, "group_summary9": group,
        "vector116": vector.astype(np.float32),
        "column_spectrum": np.zeros((feats_c, 9), dtype=np.float32),
        "column_prior_importance": np.full(feats_c, 1.0 / float(feats_c), dtype=np.float32),
    }


def build_observed_conditioning_bundle(
    bundle: LandmarkBundle,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    *,
    task: str,
    level: str = "high",
    y_train_raw: Optional[np.ndarray] = None,
    target_stats: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    """Observed-only V3 conditioning (modes none/fast/progressive); same keys as the legacy bundle.

    vec256 = [observed vector116 | table_vec24 (T0) | observed inter48 | bb68 = 0];
    col_landmarks24 = [col_spec9 = 0 | col_desc8 (T0) | col_bb7 = 0];
    v3_driver_weights uses the legacy formula with uniform report prior and zero bb.
    """
    from .building_block_landmarks import compute_observed_inter_landmark_stats, zero_building_block_landmarks
    from .descriptors import compute_table_descriptors
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
    """sha256(b"dope.conditioning/v1\\0" || recipe || 0x00 || level || 0x00 || vec256 f32 LE || col 64x17 f32 LE)."""
    vec = np.asarray(vec256, dtype="<f4").reshape(256)
    cols = np.asarray(col_landmarks, dtype="<f4").reshape(64, 17)
    h = hashlib.sha256(CONDITIONING_DOMAIN)
    h.update(recipe.encode() + b"\0" + level.encode() + b"\0")
    h.update(vec.tobytes())
    h.update(cols.tobytes())
    return h.hexdigest()
