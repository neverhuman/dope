"""Transfer diagnostics and scheduled reliability, with bytes as a constraint.

BoundEvidence is a local PR01/PR02 interface stub. A trusted verifier must be
injected to count supported coverage; this module neither verifies receipts nor
authorizes release. Missing verification defaults to observed-only evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math

from .losses import PairedLosses, TrainingScale, finite_number


@dataclass(frozen=True)
class GuardedRatio:
    value: float | None
    status: str


def loss_ratio(real: float, synthetic: float) -> GuardedRatio:
    real, synthetic = finite_number(real), finite_number(synthetic)
    if min(real, synthetic) < 0:
        raise ValueError("losses must be nonnegative")
    if real == 0:
        return GuardedRatio(1.0, "both_zero") if synthetic == 0 else GuardedRatio(None, "unbounded")
    ratio = synthetic / real
    return GuardedRatio(ratio, "finite") if math.isfinite(ratio) else GuardedRatio(None, "unbounded")


def loss_summary(pairs: PairedLosses, scale: TrainingScale) -> dict:
    real, synthetic, null = pairs.means
    gap = finite_number(synthetic - real)
    denominator, numerator = null - real, null - synthetic
    informative = denominator > 0 and denominator >= 0.01 * abs(null)
    normalized = finite_number(gap / scale.value) if scale.value is not None else None
    ratio = loss_ratio(real, synthetic)
    return {"loss_space": pairs.loss_space.value, "real_loss": real, "synthetic_loss": synthetic,
            "null_loss": null, "gap": gap, "normalized_gap": normalized,
            "scale_status": scale.status.value if scale.status else "positive_fit_scale",
            "null_normalized_gap": finite_number(gap / null) if null > 0 else None,
            "retention_numerator": numerator, "retention_denominator": denominator,
            "informative": informative,
            "retention": finite_number(numerator / denominator) if informative else None,
            "ratio": ratio.value, "ratio_status": ratio.status}


@dataclass(frozen=True)
class Margin:
    relative: float
    absolute_scaled: float
    absolute_raw: float | None = None

    def __post_init__(self):
        values = (self.relative, self.absolute_scaled) + (() if self.absolute_raw is None else (self.absolute_raw,))
        if any(finite_number(v) < 0 for v in values):
            raise ValueError("negative practical margin")

    def tolerance(self, real: float, scale: TrainingScale) -> float:
        if scale.value is None:
            if self.absolute_raw is None:
                raise ValueError("degenerate scale requires a declared absolute rule")
            return self.absolute_raw
        return finite_number(self.relative * finite_number(real) + self.absolute_scaled * scale.value)

    def query_z(self, pairs: PairedLosses, scale: TrainingScale) -> tuple[float, ...]:
        """Resampling these pairs recomputes estimated-reference tolerance."""
        return tuple(finite_number(s - r - self.tolerance(r, scale))
                     for r, s in zip(pairs.real, pairs.synthetic))


def weighted_cvar(values, weights, *, alpha: float = 0.9) -> float:
    """Upper (1-alpha) mass, with fractional weight at the boundary; no clipping."""
    alpha = finite_number(alpha)
    if not 0 <= alpha < 1:
        raise ValueError("invalid tail level")
    values, weights = tuple(values), tuple(weights)
    if not values or len(values) != len(weights):
        raise ValueError("tail evidence lengths differ")
    rows = sorted(((finite_number(v), finite_number(w)) for v, w in zip(values, weights)), reverse=True)
    if any(w < 0 for _, w in rows) or not math.isfinite(total := math.fsum(w for _, w in rows)) or total <= 0:
        raise ValueError("invalid tail weights")
    remaining = 1 - alpha
    parts = []
    for value, weight in rows:
        take = min(weight / total, remaining)
        parts.append(value * take)
        remaining -= take
        if remaining <= 0:
            break
    return finite_number(math.fsum(parts) / (1 - alpha))


@dataclass(frozen=True)
class BoundEvidence:
    upper_z: float
    contrast_id: str
    protocol_id: str
    receipt_id: str

    def __post_init__(self):
        finite_number(self.upper_z)
        if not all(isinstance(v, str) and v for v in (self.contrast_id, self.protocol_id, self.receipt_id)):
            raise ValueError("bound identity is incomplete")


@dataclass(frozen=True)
class CellResult:
    status: str = "unresolved"
    pairs: PairedLosses | None = field(default=None, repr=False)
    scale: TrainingScale | None = None
    margin: Margin | None = None
    bound: BoundEvidence | None = None

    def __post_init__(self):
        if self.status not in ("measured", "technical_failure", "unsupported", "unresolved"):
            raise ValueError("unknown cell disposition")
        if self.status == "measured":
            if self.pairs is None or self.scale is None or self.margin is None:
                raise ValueError("measured cell requires paired evidence and margin")
            self.margin.query_z(self.pairs, self.scale)
        elif any(v is not None for v in (self.pairs, self.scale, self.margin, self.bound)):
            raise ValueError("failed cells cannot carry fabricated losses")

    @property
    def z(self) -> float | None:
        if self.status != "measured":
            return None
        real, synthetic, _ = self.pairs.means
        return finite_number(synthetic - real - self.margin.tolerance(real, self.scale))


@dataclass(frozen=True)
class TaskEvaluation:
    family_id: str
    task_id: str
    required_cells: tuple[str, ...]
    cells: dict[str, CellResult] = field(repr=False)
    artifact_bytes: int
    byte_budget: int
    resources_within_budget: bool

    def __post_init__(self):
        object.__setattr__(self, "required_cells", tuple(self.required_cells))
        object.__setattr__(self, "cells", dict(self.cells))
        if not self.family_id or not self.task_id or not self.required_cells or len(set(self.required_cells)) != len(self.required_cells):
            raise ValueError("invalid scheduled task")
        if set(self.cells) - set(self.required_cells):
            raise ValueError("unscheduled cell")
        if any(type(v) is not int or v < 0 for v in (self.artifact_bytes, self.byte_budget)) or type(self.resources_within_budget) is not bool:
            raise ValueError("invalid resource accounting")


def _coverage(tasks, *, supported: bool, verify_bound=None) -> dict:
    tasks = tuple(tasks)
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("empty or duplicate task schedule")
    families = defaultdict(list)
    counts = Counter({k: 0 for k in ("qualified", "observed_only", "harmful", "technical_failure", "unsupported", "unresolved")})
    for task in tasks:
        feasible = task.artifact_bytes <= task.byte_budget and task.resources_within_budget
        passed, potential = feasible, feasible
        for key in task.required_cells:
            cell = task.cells.get(key, CellResult())
            if cell.status != "measured":
                counts[cell.status] += 1
                passed = False
                potential &= cell.status == "unresolved"
                continue
            observed = cell.z <= 0
            qualified = (feasible and observed and cell.bound is not None and cell.bound.upper_z <= 0
                         and verify_bound is not None and verify_bound(task, key, cell.bound) is True)
            counts["qualified" if qualified else "observed_only" if observed else "harmful"] += 1
            passed &= qualified if supported else observed
            potential &= qualified if supported else observed
        families[task.family_id].append((int(passed), int(potential)))
    lower = math.fsum(math.fsum(v[0] for v in members) / len(members) for members in families.values()) / len(families)
    upper = math.fsum(math.fsum(v[1] for v in members) / len(members) for members in families.values()) / len(families)
    return {"coverage": lower, "lower": lower, "upper": upper, "families": len(families),
            "tasks": len(tasks), "scheduled_cells": sum(len(t.required_cells) for t in tasks),
            "cell_counts": dict(counts), "byte_infeasible_tasks": sum(t.artifact_bytes > t.byte_budget for t in tasks),
            "resource_infeasible_tasks": sum(not t.resources_within_budget for t in tasks)}


def observed_coverage(tasks) -> dict:
    return _coverage(tasks, supported=False)


def qualified_coverage(tasks, *, verify_bound=None) -> dict:
    return _coverage(tasks, supported=True, verify_bound=verify_bound)


def harm_summary(tasks, *, catastrophic_threshold: float) -> dict:
    """Complete-case sensitivity with equal families/tasks/required-cell weights.

    Failed and degenerate cells remain explicitly counted; no loss is imputed.
    Positive-risk quantiles include zero mass for improvements.
    """
    catastrophic_threshold = finite_number(catastrophic_threshold)
    if catastrophic_threshold <= 0:
        raise ValueError("catastrophic threshold must be positive")
    tasks = tuple(tasks)
    coverage = observed_coverage(tasks)
    families = Counter(t.family_id for t in tasks)
    values, weights, worst = [], [], []
    degenerate = 0
    for task in tasks:
        task_values = []
        for key in task.required_cells:
            cell = task.cells.get(key, CellResult())
            if cell.status != "measured":
                continue
            value = loss_summary(cell.pairs, cell.scale)["normalized_gap"]
            if value is None:
                degenerate += 1
                continue
            values.append(value)
            weights.append(1 / (len(families) * families[task.family_id] * len(task.required_cells)))
            task_values.append(value)
        if task_values:
            worst.append(max(task_values))
    result = {"estimand": "complete_case_sensitivity", "measured_normalized_cells": len(values),
              "degenerate_cells": degenerate, "cell_counts": coverage["cell_counts"],
              "mean_gap": None, "median_gap": None, "positive_p90": None, "positive_p95": None,
              "positive_cvar90": None, "worst_required_learner_gap": max(worst) if worst else None,
              "catastrophic_rate": None}
    if values:
        total = math.fsum(weights)
        def quantile(numbers, probability):
            cumulative = 0.0
            for value, weight in sorted(zip(numbers, weights)):
                cumulative += weight / total
                if cumulative >= probability:
                    return value
            return max(numbers)
        positive = [max(v, 0) for v in values]
        result.update(mean_gap=math.fsum(v * w / total for v, w in zip(values, weights)),
                      median_gap=quantile(values, 0.5), positive_p90=quantile(positive, 0.9),
                      positive_p95=quantile(positive, 0.95), positive_cvar90=weighted_cvar(positive, weights),
                      catastrophic_rate=math.fsum(w / total for v, w in zip(values, weights) if v > catastrophic_threshold))
    return result
