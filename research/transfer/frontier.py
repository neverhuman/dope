"""Descriptive tested-budget frontier. No coined density scalar and no interpolation.

Family weights are frozen with the schedule. Pending work refuses B*_tested.
Terminal failures stay in the denominator. Only a measured, qualified, attempted
artifact at or under B enters the numerator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .losses import finite_number


_TERMINAL = {"measured", "technical_failure", "unsupported", "unresolved", "unfitted"}


@dataclass(frozen=True)
class ScheduledTask:
    family_id: str
    task_id: str
    family_weight: float
    status: str
    artifact_bytes: int | None
    qualified: bool
    attempted: bool

    def __post_init__(self):
        if not self.family_id or not self.task_id:
            raise ValueError("invalid frontier task")
        finite_number(self.family_weight)
        if self.family_weight <= 0:
            raise ValueError("invalid family weight")
        if self.status not in _TERMINAL | {"pending"}:
            raise ValueError("unknown frontier disposition")
        if self.artifact_bytes is not None and (type(self.artifact_bytes) is not int or self.artifact_bytes < 0):
            raise ValueError("invalid artifact bytes")
        if type(self.qualified) is not bool or type(self.attempted) is not bool:
            raise ValueError("invalid frontier flags")
        if self.status == "pending" and self.attempted:
            raise ValueError("pending work is not attempted")


def _shares(tasks):
    counts = Counter(task.family_id for task in tasks)
    weights = {}
    for task in tasks:
        seen = weights.setdefault(task.family_id, task.family_weight)
        if seen != task.family_weight:
            raise ValueError("family weight changed inside the schedule")
    return {task.task_id: weights[task.family_id] / counts[task.family_id] for task in tasks}


def tested_budget_frontier(tasks, budgets, *, measured_points=None) -> dict:
    tasks = tuple(tasks)
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("empty or duplicate frontier schedule")
    budgets = tuple(budgets)
    if not budgets or any(type(b) is not int or b <= 0 for b in budgets) or list(budgets) != sorted(set(budgets)):
        raise ValueError("invalid tested budget grid")
    shares = _shares(tasks)
    denominator = sum(shares.values())
    refused = any(task.status == "pending" or not task.attempted for task in tasks)
    measured_bytes = [task.artifact_bytes for task in tasks if task.status == "measured" and task.artifact_bytes is not None]
    censored = any(size < budgets[0] for size in measured_bytes)
    non_monotone = False
    if measured_points is not None:
        rates = [finite_number(rate) for _, rate in measured_points]
        non_monotone = any(later < earlier for earlier, later in zip(rates, rates[1:]))
    if refused:
        return {"status": "refused", "reason": "incomplete_schedule", "tested": None,
                "denominator_weight": denominator, "censored_below_grid": censored,
                "no_feasible_tested_artifact": None, "non_monotone": non_monotone,
                "interpolation": False, "lower_bound_claim": False}
    tested = {}
    for budget in budgets:
        numerator = 0.0
        for task in tasks:
            if (task.status == "measured" and task.qualified and task.artifact_bytes is not None
                    and task.artifact_bytes <= budget):
                numerator += shares[task.task_id]
        tested[str(budget)] = {"numerator_weight": numerator, "denominator_weight": denominator,
                               "rate": numerator / denominator}
    rates = [item["rate"] for item in tested.values()]
    return {"status": "complete", "reason": None, "tested": tested,
            "denominator_weight": denominator, "censored_below_grid": censored,
            "no_feasible_tested_artifact": rates[-1] == 0, "non_monotone": non_monotone or any(later < earlier for earlier, later in zip(rates, rates[1:])),
            "interpolation": False, "lower_bound_claim": False}
