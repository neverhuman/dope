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


from .observed_conditioning import (  # noqa: E402,F401
    CONDITIONING_DOMAIN,
    LEGACY_FULL_RECIPE,
    MLC_V3_TRAINING_WORK_BUDGET,
    OBSERVED_RECIPE,
    build_conditioning,
    build_conditioning_inputs,
    build_observed_conditioning_bundle,
    bundle_to_report,
    compute_observed_inter_landmark_stats,
    effective_conditioning_digest,
    encode_with_landmarks,
    merge_reports,
    mlc_v3_training_options,
    native_group_runner,
    observed_report_descriptors,
    observed_target_stats,
    progressive_bundles,
    progressive_steps,
    stratified_subsample,
    table_csv_bytes,
)

