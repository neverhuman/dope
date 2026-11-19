"""Frozen full629 token instrument for the B00 control and the B03 init.

The three direct DOPE slots (2, 4, 5) use the Stage-1B standardize, mask, and
project formula. B03 starts from the affine map obtained by folding that
formula, before either clamp. Inside the standardize clamp the folded map
reproduces those slots; the output clamp is applied on both paths. Landmark
slots 0/1/7 and mixed slot 9 are copied from the control. Mask slots 3/6/8
stay zero. This module does not train, score a panel, or load a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence

SCHEMA = "dope.transfer-projection/v1"
LAYOUT_SCHEMA = "dope.projector-normalizer-token-layout/v1"
DOPE_WIDTH = 629
TOKEN_WIDTH = 32
TOKEN_COUNT = 10
DIRECT_SLOTS = (2, 4, 5)
LANDMARK_SLOTS = (0, 1, 7)
ZERO_SLOTS = (3, 6, 8)
MIXED_SLOT = 9
STANDARDIZE_CLAMP = 8.0
OUTPUT_CLAMP = 6.0
AFFINE_PARAMETER_COUNT = DOPE_WIDTH * (TOKEN_WIDTH * len(DIRECT_SLOTS)) + TOKEN_WIDTH * len(DIRECT_SLOTS)
_MATCH_ABS = 1e-9
_MATCH_REL = 1e-9


class TransferProjectionError(ValueError):
    """Refusal whose text is a static code."""


@dataclass(frozen=True)
class Arm:
    arm_id: str
    intervention: str
    constructs_tokens: bool
    trains: bool
    reports_downstream_decisions: bool
    changes_encoder_weights: bool


_ARMS = {
    "B00": Arm("B00", "current SVD projection of frozen full629 into slots 2/4/5", True, False, False, False),
    "B01": Arm("B01", "shuffle task embeddings and retrain only the consumer", False, False, False, False),
    "B02": Arm("B02", "descriptor and landmark features with a matched consumer", False, False, False, False),
    "B03": Arm("B03", "affine 629-to-96 map into slots 2/4/5 with every other slot fixed", True, False, False, False),
}


@dataclass(frozen=True)
class SvdSlot:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    active: tuple[float, ...]
    projection: tuple[float, ...]

    def checked(self) -> SvdSlot:
        width = DOPE_WIDTH
        if (
            len(self.mean) != width
            or len(self.std) != width
            or len(self.active) != width
            or len(self.projection) != width * TOKEN_WIDTH
        ):
            raise TransferProjectionError("transfer_projection_shape")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.std):
            raise TransferProjectionError("transfer_projection_slot")
        if any(not math.isfinite(value) for value in self.mean):
            raise TransferProjectionError("transfer_projection_slot")
        if any(not math.isfinite(value) for value in self.active):
            raise TransferProjectionError("transfer_projection_slot")
        if any(not math.isfinite(value) for value in self.projection):
            raise TransferProjectionError("transfer_projection_slot")
        return self


@dataclass(frozen=True)
class AffineMap:
    """Row-major 96x629 map. Rows are slots 2, then 4, then 5."""

    weight: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    @property
    def parameter_count(self) -> int:
        return AFFINE_PARAMETER_COUNT


def arm(arm_id: str) -> Arm:
    try:
        return _ARMS[arm_id]
    except KeyError:
        raise TransferProjectionError("transfer_projection_arm") from None


def fold_slots(slots: Mapping[int, SvdSlot]) -> AffineMap:
    _require_direct_slots(slots)
    rows: list[tuple[float, ...]] = []
    bias: list[float] = []
    for slot_id in DIRECT_SLOTS:
        slot_rows, slot_bias = _fold_one(slots[slot_id].checked())
        rows.extend(slot_rows)
        bias.extend(slot_bias)
    return AffineMap(tuple(rows), tuple(bias))


def project_control_slots(features: Sequence[float], slots: Mapping[int, SvdSlot]) -> dict[int, list[float]]:
    """B00 direct slots, including both clamps."""
    clean = _clean_features(features)
    _require_direct_slots(slots)
    return {slot_id: _project_svd(clean, slots[slot_id].checked()) for slot_id in DIRECT_SLOTS}


def project_folded_slots(features: Sequence[float], mapping: AffineMap) -> dict[int, list[float]]:
    clean = _clean_features(features)
    if (
        len(mapping.weight) != TOKEN_WIDTH * len(DIRECT_SLOTS)
        or len(mapping.bias) != len(mapping.weight)
        or any(len(row) != DOPE_WIDTH for row in mapping.weight)
    ):
        raise TransferProjectionError("transfer_projection_shape")
    outputs: dict[int, list[float]] = {}
    for offset, slot_id in enumerate(DIRECT_SLOTS):
        start = offset * TOKEN_WIDTH
        outputs[slot_id] = [
            _clamp(_dot(clean, mapping.weight[start + column]) + mapping.bias[start + column], OUTPUT_CLAMP)
            for column in range(TOKEN_WIDTH)
        ]
    return outputs


def initialization_reproduces_control(features: Sequence[float], slots: Mapping[int, SvdSlot]) -> bool:
    """True when the untrained folded map matches B00 on these features."""
    if not _inside_standardize_clamp(features, slots):
        return False
    control = project_control_slots(features, slots)
    folded = project_folded_slots(features, fold_slots(slots))
    for slot_id in DIRECT_SLOTS:
        for left, right in zip(control[slot_id], folded[slot_id]):
            if not math.isclose(left, right, rel_tol=_MATCH_REL, abs_tol=_MATCH_ABS):
                return False
    return True


def b03_tokens(
    features: Sequence[float],
    slots: Mapping[int, SvdSlot],
    control_tokens: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Replace slots 2/4/5 with the folded init and keep every other slot."""
    return _assemble(control_tokens, project_folded_slots(features, fold_slots(slots)))


def projector_normalizer_and_token_layout_sha256(slots: Mapping[int, SvdSlot]) -> str:
    """Digest for the WP01 `projector_normalizer_and_token_layout_sha256` field."""
    _require_direct_slots(slots)
    document = {
        "schema": LAYOUT_SCHEMA,
        "direct_slots": list(DIRECT_SLOTS),
        "landmark_slots": list(LANDMARK_SLOTS),
        "mixed_slot": MIXED_SLOT,
        "mixed_slot_policy": "preserve-control",
        "output_clamp": OUTPUT_CLAMP,
        "standardize_clamp": STANDARDIZE_CLAMP,
        "token_count": TOKEN_COUNT,
        "token_width": TOKEN_WIDTH,
        "zero_slots": list(ZERO_SLOTS),
        "slot_sha256": {str(slot_id): _slot_digest(slots[slot_id].checked()) for slot_id in DIRECT_SLOTS},
    }
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_direct_slots(slots: Mapping[int, SvdSlot]) -> None:
    if set(slots) != set(DIRECT_SLOTS):
        raise TransferProjectionError("transfer_projection_shape")


def _clean_features(features: Sequence[float]) -> list[float]:
    if len(features) != DOPE_WIDTH:
        raise TransferProjectionError("transfer_projection_shape")
    return [value if math.isfinite(value) else 0.0 for value in features]


def _inside_standardize_clamp(features: Sequence[float], slots: Mapping[int, SvdSlot]) -> bool:
    clean = _clean_features(features)
    _require_direct_slots(slots)
    for slot in slots.values():
        checked = slot.checked()
        for index, value in enumerate(clean):
            if checked.active[index] == 0.0:
                continue
            standardized = (value - checked.mean[index]) / checked.std[index]
            if abs(standardized) > STANDARDIZE_CLAMP:
                return False
    return True


def _project_svd(clean: Sequence[float], slot: SvdSlot) -> list[float]:
    output = [0.0] * TOKEN_WIDTH
    for index, value in enumerate(clean):
        standardized = _clamp((value - slot.mean[index]) / slot.std[index], STANDARDIZE_CLAMP)
        standardized *= slot.active[index]
        base = index * TOKEN_WIDTH
        for column in range(TOKEN_WIDTH):
            output[column] += standardized * slot.projection[base + column]
    return [_clamp(value, OUTPUT_CLAMP) for value in output]


def _fold_one(slot: SvdSlot) -> tuple[list[tuple[float, ...]], list[float]]:
    rows = [[0.0] * DOPE_WIDTH for _ in range(TOKEN_WIDTH)]
    bias = [0.0] * TOKEN_WIDTH
    for index in range(DOPE_WIDTH):
        scale = slot.active[index] / slot.std[index]
        base = index * TOKEN_WIDTH
        for column in range(TOKEN_WIDTH):
            coefficient = scale * slot.projection[base + column]
            rows[column][index] = coefficient
            bias[column] -= slot.mean[index] * coefficient
    return [tuple(row) for row in rows], bias


def _assemble(control_tokens: Sequence[Sequence[float]], direct: Mapping[int, Sequence[float]]) -> list[list[float]]:
    if len(control_tokens) != TOKEN_COUNT or set(direct) != set(DIRECT_SLOTS):
        raise TransferProjectionError("transfer_projection_shape")
    assembled: list[list[float]] = []
    for index, row in enumerate(control_tokens):
        if len(row) != TOKEN_WIDTH:
            raise TransferProjectionError("transfer_projection_shape")
        if index in ZERO_SLOTS:
            if any(value != 0.0 for value in row):
                raise TransferProjectionError("transfer_projection_mask")
            assembled.append([0.0] * TOKEN_WIDTH)
            continue
        source = direct[index] if index in DIRECT_SLOTS else row
        if len(source) != TOKEN_WIDTH or any(not math.isfinite(value) for value in source):
            raise TransferProjectionError("transfer_projection_value")
        assembled.append([float(value) for value in source])
    return assembled


def _slot_digest(slot: SvdSlot) -> str:
    payload = struct.pack(
        f"<{DOPE_WIDTH * 3 + DOPE_WIDTH * TOKEN_WIDTH}d",
        *slot.mean,
        *slot.std,
        *slot.active,
        *slot.projection,
    )
    return hashlib.sha256(payload).hexdigest()


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(value * weight for value, weight in zip(left, right))


def _clamp(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value
