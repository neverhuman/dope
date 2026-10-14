"""Descriptive tested-budget frontier. No coined density scalar and no interpolation.

Family weights are frozen with the schedule. Pending work refuses B*_tested.
Terminal failures stay in the denominator. Only a measured, qualified, attempted,
complete artifact at or under B enters the numerator.

`tested_budget_frontier` keeps `censored_below_grid` for a measured artifact
smaller than the first grid point. `b_star_tested` uses the frozen state
`censored_below_grid` only when coverage at the first grid point already
reaches the quality floor. Those two fields are not the same claim.
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
    complete: bool = True
    trivial_target: bool = False
    empty_generator: bool = False
    external_checkpoint: bool = False
    outcomes: tuple | None = None

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
        flags = (self.qualified, self.attempted, self.complete, self.trivial_target,
                 self.empty_generator, self.external_checkpoint)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("invalid frontier flags")
        if self.status == "pending" and self.attempted:
            raise ValueError("pending work is not attempted")
        if self.outcomes is not None:
            normalized = []
            for item in self.outcomes:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("invalid budget outcome")
                budget, flag = item
                if type(budget) is not int or budget <= 0 or type(flag) is not bool:
                    raise ValueError("invalid budget outcome")
                normalized.append((budget, flag))
            if len({budget for budget, _ in normalized}) != len(normalized):
                raise ValueError("invalid budget outcome")
            object.__setattr__(self, "outcomes", tuple(normalized))


def _shares(tasks):
    counts = Counter(task.family_id for task in tasks)
    weights = {}
    for task in tasks:
        seen = weights.setdefault(task.family_id, task.family_weight)
        if seen != task.family_weight:
            raise ValueError("family weight changed inside the schedule")
    return {task.task_id: weights[task.family_id] / counts[task.family_id] for task in tasks}


def _qualified_at(task, budget) -> bool:
    if task.empty_generator or task.trivial_target or task.external_checkpoint or not task.complete:
        return False
    if task.status != "measured" or not task.attempted:
        return False
    if task.outcomes is not None:
        matched = [flag for point, flag in task.outcomes if point == budget]
        if len(matched) != 1 or type(matched[0]) is not bool:
            raise ValueError("invalid budget outcome")
        return matched[0]
    return bool(task.qualified and task.artifact_bytes is not None and task.artifact_bytes <= budget)


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
            if _qualified_at(task, budget):
                numerator += shares[task.task_id]
        tested[str(budget)] = {"numerator_weight": numerator, "denominator_weight": denominator,
                               "rate": numerator / denominator}
    rates = [item["rate"] for item in tested.values()]
    return {"status": "complete", "reason": None, "tested": tested,
            "denominator_weight": denominator, "censored_below_grid": censored,
            "no_feasible_tested_artifact": rates[-1] == 0, "non_monotone": non_monotone or any(later < earlier for earlier, later in zip(rates, rates[1:])),
            "interpolation": False, "lower_bound_claim": False}


def qualified_learning_density(*, single_artifact: bool, fixed_workload: bool, verified_coverage: bool,
                               eligible: bool) -> dict:
    """QLD is unadopted. No coverage-per-byte ratio is returned."""
    flags = (single_artifact, fixed_workload, verified_coverage, eligible)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("invalid learning density")
    if not eligible or not single_artifact or not fixed_workload:
        status, reason = "ineligible", "ineligible_domain"
    elif not verified_coverage:
        status, reason = "unmeasured", "coverage_unverified"
    else:
        status, reason = "unmeasured", "unadopted"
    return {"adopted": False, "status": status, "reason": reason, "value": None,
            "aggregate": False, "headline": False, "release": False}


def deployed_artifact(candidates) -> dict:
    """One deployed artifact. Losses are not used to pick a seed."""
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("missing deployed artifact")
    seeds = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or type(candidate.get("seed")) is not str or not candidate["seed"]:
            raise ValueError("invalid deployed artifact")
        if "loss" not in candidate:
            raise ValueError("invalid deployed artifact")
        finite_number(candidate["loss"])
        seeds.append(candidate["seed"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate deployed seed")
    if len(candidates) != 1:
        return {"status": "refused", "reason": "best_of_k_forbidden", "selected_seed": None,
                "seeds": tuple(sorted(seeds)), "best_of_k": False, "selected_by_loss": False}
    return {"status": "deployed", "reason": None, "selected_seed": seeds[0],
            "seeds": (seeds[0],), "best_of_k": False, "selected_by_loss": False}


def b_star_tested(tasks, budgets, *, quality: float) -> dict:
    """One frozen B*_tested state. Pending refuses. QLD stays unadopted.

    Coverage is qualified family weight over the full scheduled weight.
    `censored_below_grid` here means C(B_1) already reaches q. It does not
    mean a measured artifact is smaller than the first grid point.
    """
    if type(quality) is bool:
        raise ValueError("invalid quality floor")
    quality = finite_number(quality)
    if not 0 < quality <= 1:
        raise ValueError("invalid quality floor")
    tasks = tuple(tasks)
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("empty or duplicate frontier schedule")
    budgets = tuple(budgets)
    if (not budgets or any(type(point) is not int or point <= 0 for point in budgets)
            or list(budgets) != sorted(set(budgets))):
        raise ValueError("invalid tested budget grid")
    shares = _shares(tasks)
    denominator = sum(shares.values())
    eligible = not any(
        task.trivial_target or task.empty_generator or task.external_checkpoint or not task.complete
        for task in tasks
    )
    qld = qualified_learning_density(
        single_artifact=len(tasks) == 1, fixed_workload=True, verified_coverage=False, eligible=eligible,
    )
    base = {"format": "dope-b-star-tested", "version": 1, "budget": None, "at_most_budget": None,
            "through_budget": None, "points": None, "first_crossing": None, "minimum_lies_in": None,
            "denominator_weight": denominator, "interpolation": False, "lower_bound_claim": False,
            "qld": qld}
    if any(task.status == "pending" or not task.attempted for task in tasks):
        return {**base, "state": "pending"}
    points = []
    for budget in budgets:
        numerator = sum(shares[task.task_id] for task in tasks if _qualified_at(task, budget))
        points.append({"budget": budget, "coverage": numerator / denominator})
    rates = [point["coverage"] for point in points]
    decreases = any(later < earlier for earlier, later in zip(rates, rates[1:]))
    crossing = next((point["budget"] for point in points if point["coverage"] >= quality), None)
    report = {**base, "points": tuple(points), "first_crossing": crossing}
    if decreases:
        return {**report, "state": "non_monotone"}
    if rates[0] >= quality:
        return {**report, "state": "censored_below_grid", "at_most_budget": budgets[0]}
    if crossing is not None:
        previous = budgets[budgets.index(crossing) - 1]
        return {**report, "state": "measured", "budget": crossing,
                "minimum_lies_in": {"lower_exclusive": previous, "upper_inclusive": crossing}}
    return {**report, "state": "no_feasible_tested_artifact", "through_budget": budgets[-1]}
