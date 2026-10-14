"""Transfer diagnostics and scheduled reliability, with bytes as a constraint.

BoundEvidence is the typed bound-evidence record consumed alongside verified
PR01/PR02 evaluation receipts. A supported bound needs an owner digest or a
paired recomputation over the live learner, fit, artifact, scale, margin,
paired losses, schedule, code, RNG, and result. A pin_source string is not
that binding. Missing verification stays observed-only. These diagnostics do
not authorize release.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import math

from .evidence import digest
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
    lower_z: float | None = None

    def __post_init__(self):
        finite_number(self.upper_z)
        if self.lower_z is not None:
            finite_number(self.lower_z)
            if self.lower_z > self.upper_z:
                raise ValueError("reversed bound")
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
            raise ValueError("failed cells require absent loss fields")

    @property
    def z(self) -> float | None:
        if self.status != "measured":
            return None
        real, synthetic, _ = self.pairs.means
        return finite_number(synthetic - real - self.margin.tolerance(real, self.scale))


@dataclass(frozen=True)
class FitArtifact:
    cell_id: str
    fit_id: str
    artifact_bytes: int
    content: bytes | None = None

    def __post_init__(self):
        if not isinstance(self.cell_id, str) or not self.cell_id or not isinstance(self.fit_id, str) or not self.fit_id:
            raise ValueError("invalid fit artifact")
        if type(self.artifact_bytes) is not int or self.artifact_bytes < 0:
            raise ValueError("invalid fit artifact")
        if self.content is not None and (type(self.content) is not bytes or len(self.content) != self.artifact_bytes):
            raise ValueError("invalid fit artifact")


@dataclass(frozen=True)
class TaskEvaluation:
    family_id: str
    task_id: str
    required_cells: tuple[str, ...]
    cells: dict[str, CellResult] = field(repr=False)
    artifact_bytes: int
    byte_budget: int
    resources_within_budget: bool
    artifacts: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "required_cells", tuple(self.required_cells))
        object.__setattr__(self, "cells", dict(self.cells))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if not self.family_id or not self.task_id or not self.required_cells or len(set(self.required_cells)) != len(self.required_cells):
            raise ValueError("invalid scheduled task")
        if set(self.cells) - set(self.required_cells):
            raise ValueError("unscheduled cell")
        if any(type(v) is not int or v < 0 for v in (self.artifact_bytes, self.byte_budget)) or type(self.resources_within_budget) is not bool:
            raise ValueError("invalid resource accounting")
        for artifact in self.artifacts:
            if not isinstance(artifact, FitArtifact) or artifact.cell_id not in self.required_cells:
                raise ValueError("unscheduled artifact")


_BOUND_ENDPOINTS = frozenset({"gap", "normalized_gap", "z", "tolerance_adjusted_z_over_fit_scale"})
_PAIRED_MEAN_PROCEDURE = {
    "format": "dope-bound-procedure",
    "version": 1,
    "id": "paired-mean-query-z-v1",
    "calibration": "unmeasured",
}
PAIRED_MEAN_CODE_DIGEST = digest(_PAIRED_MEAN_PROCEDURE)


def query_z_mean(cell) -> float:
    """Mean of per-query tolerance-adjusted Z. This is not a calibrated upper bound."""
    values = cell.margin.query_z(cell.pairs, cell.scale)
    return finite_number(math.fsum(values) / len(values))


def _paired_digest(pairs: PairedLosses) -> str:
    return digest({
        "query_ids": list(pairs.query_ids),
        "cluster_ids": list(pairs.cluster_ids),
        "real": list(pairs.real),
        "synthetic": list(pairs.synthetic),
        "null": list(pairs.null),
        "loss_space": pairs.loss_space.value,
    })


def _binding_text(value):
    if type(value) is not str or not value:
        raise ValueError("incomplete_bound_binding")
    return value


def _binding_digest_text(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("incomplete_bound_binding")
    return value


def bound_binding_record(task, cell_id, bound, analysis, *, version: int = 1) -> dict:
    """Live cell, fit artifact, and declared analysis identity. The digest covers this record.

    Version 1 commits the artifact byte count. Version 2 commits content bytes
    when they are present and records unverified_byte_count when they are not.
    Neither version carries a calibration result.
    """
    if bound is None or not isinstance(analysis, dict) or cell_id != bound.contrast_id:
        raise ValueError("incomplete_bound_binding")
    cell = task.cells.get(cell_id)
    if cell is None or cell.status != "measured" or cell.pairs is None:
        raise ValueError("incomplete_bound_binding")
    matches = [artifact for artifact in task.artifacts if artifact.cell_id == cell_id]
    if len(matches) != 1:
        raise ValueError("unbound_artifact")
    artifact = matches[0]
    endpoint = _binding_text(analysis.get("endpoint"))
    if endpoint not in _BOUND_ENDPOINTS:
        raise ValueError("incomplete_bound_binding")
    draws = analysis.get("draws")
    if type(draws) is not int or isinstance(draws, bool) or draws < 1:
        raise ValueError("incomplete_bound_binding")
    if version not in (1, 2):
        raise ValueError("unknown_bound_binding_version")
    artifact_id = digest({
        "cell_id": artifact.cell_id,
        "fit_id": artifact.fit_id,
        "artifact_bytes": artifact.artifact_bytes,
    })
    record = {
        "format": "dope-bound-binding",
        "version": version,
        "task_id": task.task_id,
        "learner_id": _binding_text(analysis.get("learner_id")),
        "fit_id": artifact.fit_id,
        "artifact_id": artifact_id,
        "contrast_id": bound.contrast_id,
        "protocol_id": bound.protocol_id,
        "receipt_id": bound.receipt_id,
        "endpoint": endpoint,
        "scale": cell.scale.value,
        "scale_status": cell.scale.status.value if cell.scale.status else "positive_fit_scale",
        "margin": {
            "relative": cell.margin.relative,
            "absolute_scaled": cell.margin.absolute_scaled,
            "absolute_raw": cell.margin.absolute_raw,
        },
        "paired_digest": _paired_digest(cell.pairs),
        "point_z": cell.z,
        "query_z_mean": query_z_mean(cell),
        "schedule_id": _binding_text(analysis.get("schedule_id")),
        "code_digest": _binding_digest_text(analysis.get("code_digest")),
        "rng_id": _binding_text(analysis.get("rng_id")),
        "draws": draws,
        "multiplicity": _binding_text(analysis.get("multiplicity")),
        "upper_z": bound.upper_z,
        "lower_z": bound.lower_z,
    }
    if version == 2:
        record["artifact_id"] = None if artifact.content is None else hashlib.sha256(artifact.content).hexdigest()
        record["artifact_commitment"] = "unverified_byte_count" if artifact.content is None else "content_sha256"
    return record


def authenticate_bound(task, cell_id, bound, expected) -> bool:
    """Owner digest or paired recomputation. A pin_source string is not a binding.

    A true result identifies the observed record. It does not qualify coverage.
    """
    if bound is None or not isinstance(expected, dict) or expected.get("trust") not in ("owner", "recomputed"):
        return False
    if expected.get("pin_source") not in (None, "owner"):
        return False
    version = expected.get("version", 1)
    if type(version) is not int or version not in (1, 2):
        return False
    try:
        record = bound_binding_record(task, cell_id, bound, expected, version=version)
    except ValueError:
        return False
    if expected.get("trust") == "recomputed":
        cell = task.cells.get(cell_id)
        if (
            record["endpoint"] != "z"
            or record["code_digest"] != PAIRED_MEAN_CODE_DIGEST
            or record["rng_id"] != "deterministic-paired-mean"
            or record["draws"] != len(cell.pairs.real)
            or bound.lower_z is not None
            or bound.upper_z != record["query_z_mean"]
        ):
            return False
    if any(expected.get(key) != record[key] for key in record):
        return False
    if set(expected) - set(record) - {"trust", "binding_digest", "pin_source"}:
        return False
    return expected.get("binding_digest") == digest(record)


def bound_qualification(task, cell_id, bound, expected) -> dict:
    """Authentication is not calibration.

    No bound on this wire qualifies coverage. A missing, null, unmeasured, or
    self-asserted calibration flag stays observed-only. Content bytes do not
    create a calibration manifest.
    """
    authenticated = authenticate_bound(task, cell_id, bound, expected)
    return {
        "authenticated": authenticated,
        "qualifies": False,
        "disposition": "observed_only" if authenticated else "unauthenticated",
        "calibration": "unmeasured",
        "uncertainty": "absent",
    }


def _task_feasible(task) -> bool:
    if task.artifact_bytes > task.byte_budget or not task.resources_within_budget:
        return False
    if not task.artifacts:
        return True
    seen = set()
    for artifact in task.artifacts:
        if artifact.artifact_bytes > task.byte_budget:
            return False
        seen.add(artifact.cell_id)
    return set(task.required_cells) <= seen


def _coverage(tasks, *, supported: bool, verify_bound=None) -> dict:
    tasks = tuple(tasks)
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("empty or duplicate task schedule")
    families = defaultdict(list)
    counts = Counter({k: 0 for k in ("qualified", "observed_only", "observed_outside", "harmful",
                                     "technical_failure", "unsupported", "unresolved")})
    for task in tasks:
        feasible = _task_feasible(task)
        passed, potential = feasible, feasible
        for key in task.required_cells:
            cell = task.cells.get(key, CellResult())
            if cell.status != "measured":
                counts[cell.status] += 1
                passed = False
                potential &= cell.status == "unresolved"
                continue
            observed = cell.z <= 0
            authenticated = False
            if cell.bound is not None and verify_bound is not None:
                decision = verify_bound(task, key, cell.bound)
                if isinstance(decision, dict):
                    authenticated = decision.get("authenticated") is True
                else:
                    authenticated = decision is True
            inferential_harm = authenticated and cell.bound.lower_z is not None and cell.bound.lower_z > 0
            qualified = False
            if qualified:
                counts["qualified"] += 1
            elif inferential_harm:
                counts["harmful"] += 1
            elif observed:
                counts["observed_only"] += 1
            else:
                counts["observed_outside"] += 1
            passed &= qualified if supported else observed
            potential &= qualified if supported else observed
        families[task.family_id].append((int(passed), int(potential)))
    lower = math.fsum(math.fsum(v[0] for v in members) / len(members) for members in families.values()) / len(families)
    upper = math.fsum(math.fsum(v[1] for v in members) / len(members) for members in families.values()) / len(families)
    if lower > upper:
        raise ValueError("completion bounds are reversed")
    return {"coverage": lower, "lower": lower, "upper": upper, "families": len(families),
            "tasks": len(tasks), "scheduled_cells": sum(len(t.required_cells) for t in tasks),
            "cell_counts": dict(counts), "byte_infeasible_tasks": sum(t.artifact_bytes > t.byte_budget for t in tasks),
            "resource_infeasible_tasks": sum(not t.resources_within_budget for t in tasks),
            "meaning": "completion", "confidence_interval": None}


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


def tail_regression(tasks, *, catastrophic_threshold: float) -> dict:
    """A nonpositive mean does not hide a positive tail. Tails stay disclosed."""
    report = harm_summary(tasks, catastrophic_threshold=catastrophic_threshold)
    mean, tail = report["mean_gap"], report["positive_p95"]
    hidden = mean is not None and tail is not None and mean <= 0 and tail > 0
    return {**report, "mean_hides_tail": hidden,
            "tail_claim": "disclosed" if tail is not None else "not_evidenced"}
