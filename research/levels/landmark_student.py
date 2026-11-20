"""Landmark student report contract.

A predicted slot stays predicted when its number matches a measured slot.
None, partial, and full modes share one table and target. Synthetic
retention is not a decision result. The known dropout checkpoint stays
unregistered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

SCHEMA = "dope-landmark-student/v1"
SLOTS = 23
MODES = ("none", "partial", "full")
_COST_NAMES = frozenset({"prepare", "parse", "normalize", "forward", "backward"})
_MAX_CANDIDATES = 32
_MAX_STEPS = 64
_UNREGISTERED_ASSET = "ecded0225fca7c6e13c413b0e39b1040bf95f7694940b2809f1ce2ce2abf7bd9"


class StudentError(ValueError):
    """Static refusal. The message is a code, not a source value."""


@dataclass(frozen=True)
class Slot:
    index: int
    provenance: str
    value: float | None


@dataclass(frozen=True)
class Report:
    mode: str
    support: tuple[int, ...]
    slots: tuple[Slot, ...]


@dataclass(frozen=True)
class Budget:
    table_id: str
    target_id: str
    mask: tuple[bool, ...]
    support: tuple[int, ...]


def _finite(value: object) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise StudentError("student_shape")
    number = float(value)
    if not math.isfinite(number):
        raise StudentError("student_nonfinite")
    return number


def observed_support(mask: tuple[bool, ...]) -> tuple[int, ...]:
    if not isinstance(mask, tuple) or len(mask) != SLOTS or any(type(bit) is not bool for bit in mask):
        raise StudentError("student_shape")
    return tuple(index for index, bit in enumerate(mask) if bit)


def mode_of(support: tuple[int, ...]) -> str:
    if support == ():
        return "none"
    if support == tuple(range(SLOTS)):
        return "full"
    return "partial"


def measured_report(mask: tuple[bool, ...], values: tuple[object, ...]) -> Report:
    support = observed_support(mask)
    if not isinstance(values, tuple) or len(values) != SLOTS:
        raise StudentError("student_shape")
    slots: list[Slot] = []
    for index, (bit, value) in enumerate(zip(mask, values)):
        if bit:
            slots.append(Slot(index, "measured", _finite(value)))
        elif value is not None:
            raise StudentError("student_provenance")
        else:
            slots.append(Slot(index, "absent", None))
    return Report(mode_of(support), support, tuple(slots))


def predict(report: Report, fills: dict[int, object]) -> Report:
    if not isinstance(report, Report) or not isinstance(fills, dict) or not fills:
        raise StudentError("student_shape")
    if any(type(index) is not int or isinstance(index, bool) for index in fills):
        raise StudentError("student_shape")
    slots: list[Slot] = []
    seen: set[int] = set()
    for slot in report.slots:
        if slot.index not in fills:
            slots.append(slot)
            continue
        if slot.provenance != "absent":
            raise StudentError("student_provenance")
        slots.append(Slot(slot.index, "predicted", _finite(fills[slot.index])))
        seen.add(slot.index)
    if seen != set(fills):
        raise StudentError("student_shape")
    return Report(report.mode, report.support, tuple(slots))


def measured_support(report: Report) -> tuple[int, ...]:
    if not isinstance(report, Report):
        raise StudentError("student_shape")
    return tuple(slot.index for slot in report.slots if slot.provenance == "measured")


def require_measured(report: Report) -> Report:
    if not isinstance(report, Report) or any(slot.provenance != "measured" for slot in report.slots):
        raise StudentError("student_provenance")
    return report


def matched_modes(table_id: str, target_id: str, masks: dict[str, tuple[bool, ...]]) -> dict[str, Budget]:
    if type(table_id) is not str or type(target_id) is not str or not table_id or not target_id:
        raise StudentError("student_budget")
    if not isinstance(masks, dict) or set(masks) != set(MODES):
        raise StudentError("student_mode")
    budgets: dict[str, Budget] = {}
    for mode in MODES:
        support = observed_support(masks[mode])
        if mode_of(support) != mode:
            raise StudentError("student_mode")
        budgets[mode] = Budget(table_id, target_id, masks[mode], support)
    return budgets


def require_same_context(left: Budget, right: Budget) -> None:
    if not isinstance(left, Budget) or not isinstance(right, Budget):
        raise StudentError("student_budget")
    if left.table_id != right.table_id or left.target_id != right.target_id:
        raise StudentError("student_budget")


def distill(teacher: dict[str, object], *, steps: int = 32) -> dict[str, object]:
    if type(steps) is not int or steps < 1 or steps > _MAX_STEPS:
        raise StudentError("student_limit")
    if not isinstance(teacher, dict) or not teacher or len(teacher) > _MAX_CANDIDATES:
        raise StudentError("student_support")
    names = tuple(teacher)
    if any(type(name) is not str or name == "" for name in names):
        raise StudentError("student_support")
    weights: list[float] = []
    for name in names:
        value = teacher[name]
        if isinstance(value, bool) or type(value) not in (int, float):
            raise StudentError("student_support")
        number = float(value)
        if not math.isfinite(number):
            raise StudentError("student_nonfinite")
        if number < 0:
            raise StudentError("student_support")
        weights.append(number)
    total = sum(weights)
    if total == 0:
        raise StudentError("student_support")
    target = torch.tensor(weights, dtype=torch.float64) / total
    logits = torch.nn.Parameter(torch.zeros(len(names), dtype=torch.float64))
    optimizer = torch.optim.AdamW([logits], lr=0.2)
    initial = torch.nn.functional.kl_div(torch.log_softmax(logits, dim=0), target, reduction="sum")
    last = initial.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.kl_div(torch.log_softmax(logits, dim=0), target, reduction="sum")
        if not bool(torch.isfinite(loss)):
            raise StudentError("student_nonfinite")
        loss.backward()
        optimizer.step()
        last = loss.detach()
    student = torch.softmax(logits.detach(), dim=0)
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(last)):
        raise StudentError("student_nonfinite")
    return {
        "schema": SCHEMA,
        "candidates": names,
        "teacher": tuple(float(item) for item in target),
        "student": tuple(float(item) for item in student),
        "initial_loss": float(initial.detach()),
        "loss": float(last),
        "promoted": False,
    }


def evidence(kind: str) -> dict[str, object]:
    if kind == "jope_decision":
        return {"schema": SCHEMA, "kind": kind, "promoted": False, "status": "unmeasured"}
    if kind in ("synthetic_retention", "embedding_cosine"):
        return {"schema": SCHEMA, "kind": kind, "promoted": False, "status": "not_a_decision"}
    raise StudentError("student_evidence")


def cost_record(parts: dict[str, object]) -> dict[str, object]:
    if not isinstance(parts, dict) or not parts:
        raise StudentError("student_evidence")
    clean: dict[str, float] = {}
    for key, value in parts.items():
        if type(key) is not str or key not in _COST_NAMES:
            raise StudentError("student_evidence")
        if isinstance(value, bool) or type(value) not in (int, float):
            raise StudentError("student_evidence")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise StudentError("student_nonfinite")
        clean[key] = number
    return {
        "schema": SCHEMA,
        "parts": tuple(sorted(clean)),
        "complete": "prepare" in clean and "forward" in clean,
        "quality_claim": False,
    }


def recover_candidate(asset: str | None = None) -> dict[str, object]:
    if asset is not None and type(asset) is not str:
        raise StudentError("student_unregistered")
    return {
        "schema": SCHEMA,
        "recovered": False,
        "asset": _UNREGISTERED_ASSET if asset is None else asset,
        "code": "student_unregistered",
    }
