"""Full-roster coverage and minimum observed qualified bytes.

Frontier trial qualification is an upstream evaluator-owned conjunction of
T90, D80 and all required gates for one frozen method/seed. It is not inferred
from descriptive scores here. Neither arithmetic nor fixture parity certifies
those decisions or establishes a global minimum. Legacy cell coverage retains
its measured-calibration requirement and family weighting.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math

from .calibration_receipt import verify_calibration_receipt


def _scheduled_tasks(tasks, admitted_tasks):
    tasks = tuple(tasks)
    if admitted_tasks is None:
        return tasks  # Historical API: the caller supplies the complete schedule.
    roster = tuple(admitted_tasks)
    if not roster or len({t.task_id for t in roster}) != len(roster):
        raise ValueError("invalid_admitted_roster")
    indexed = {t.task_id: t for t in tasks}
    if len(indexed) != len(tasks) or set(indexed) - {t.task_id for t in roster}:
        raise ValueError("invalid_task_schedule")
    scheduled = []
    for template in roster:
        task = indexed.get(template.task_id)
        if task is None:
            task = replace(template, cells={}, artifacts=(), artifact_bytes=0, resources_within_budget=True)
        if (task.family_id, task.required_cells, task.byte_budget) != (
                template.family_id, template.required_cells, template.byte_budget):
            raise ValueError("task_schedule_mismatch")
        scheduled.append(task)
    return tuple(scheduled)


def _admissible_uncertainty(decision) -> bool:
    """Caller flags are not a calibration manifest.

    Both inferential directions stay closed unless a receipt verifies as
    measured N13b. The development protocol does not.
    """
    if not isinstance(decision, dict):
        return False
    receipt = decision.get("calibration_receipt")
    if not isinstance(receipt, dict):
        return False
    try:
        verified = verify_calibration_receipt(receipt)
    except ValueError:
        return False
    return (
        verified.get("n13b") is True
        and verified.get("qualifies") is True
        and verified.get("calibration") == "measured"
    )


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


def _coverage(tasks, *, supported: bool, verify_bound=None, admitted_tasks=None) -> dict:
    from .metrics import CellResult

    tasks = _scheduled_tasks(tasks, admitted_tasks)
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
            admissible = False
            if cell.bound is not None and verify_bound is not None:
                admissible = _admissible_uncertainty(verify_bound(task, key, cell.bound))
            inferential_harm = (
                admissible and cell.bound is not None and cell.bound.lower_z is not None and cell.bound.lower_z > 0
            )
            inferential_benefit = (
                admissible and observed and cell.bound is not None and cell.bound.upper_z <= 0 and not inferential_harm
            )
            if inferential_benefit:
                counts["qualified"] += 1
            elif inferential_harm:
                counts["harmful"] += 1
            elif observed:
                counts["observed_only"] += 1
            else:
                counts["observed_outside"] += 1
            passed &= inferential_benefit if supported else observed
            potential &= inferential_benefit if supported else observed
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


def observed_coverage(tasks, *, admitted_tasks=None) -> dict:
    return _coverage(tasks, supported=False, admitted_tasks=admitted_tasks)


def qualified_coverage(tasks, *, verify_bound=None, admitted_tasks=None) -> dict:
    return _coverage(tasks, supported=True, verify_bound=verify_bound, admitted_tasks=admitted_tasks)


@dataclass(frozen=True)
class ArtifactTrial:
    task_index: int
    search_cap_bytes: int
    artifact_bytes: int | None
    qualified: bool

    def __post_init__(self):
        if (type(self.task_index) is not int or self.task_index < 0
                or type(self.search_cap_bytes) is not int or self.search_cap_bytes <= 0
                or type(self.qualified) is not bool):
            raise ValueError("invalid_frontier_evidence")
        if self.artifact_bytes is not None and (
                type(self.artifact_bytes) is not int or not 0 < self.artifact_bytes <= self.search_cap_bytes):
            raise ValueError("invalid_frontier_evidence")
        if self.qualified and self.artifact_bytes is None:
            raise ValueError("invalid_frontier_evidence")


def non_monotone(trials, task_index):
    caps = {}
    for trial in trials:
        if trial.task_index == task_index:
            caps[trial.search_cap_bytes] = caps.get(trial.search_cap_bytes, False) or trial.qualified
    seen_success = False
    for _, qualified in sorted(caps.items()):
        if seen_success and not qualified:
            return True
        seen_success |= qualified
    return False


def minimum_qualified_bytes(trials, task_index, byte_cap, require_monotone=False):
    """B*(m,s,t): minimum charged bytes, including artifacts discovered at larger caps."""
    trials = tuple(trials)
    if (type(task_index) is not int or task_index < 0 or type(byte_cap) is not int or byte_cap < 0
            or type(require_monotone) is not bool or any(not isinstance(t, ArtifactTrial) for t in trials)):
        raise ValueError("invalid_frontier_evidence")
    if require_monotone and non_monotone(trials, task_index):
        return None
    return min((t.artifact_bytes for t in trials if t.task_index == task_index and t.qualified
                and t.artifact_bytes <= byte_cap), default=None)


def roster_coverage(admitted_tasks, trials, byte_cap, require_monotone=False):
    """C(B) uses the precommitted roster count; absent/failed tasks contribute zero."""
    trials = tuple(trials)
    if (type(admitted_tasks) is not int or admitted_tasks <= 0
            or any(not isinstance(t, ArtifactTrial) or t.task_index >= admitted_tasks for t in trials)):
        raise ValueError("invalid_frontier_evidence")
    minimum = [minimum_qualified_bytes(trials, task, byte_cap, require_monotone)
               for task in range(admitted_tasks)]
    qualified = sum(size is not None for size in minimum)
    return {"minimum_bytes": minimum, "qualified_tasks": qualified,
            "admitted_tasks": admitted_tasks, "coverage": qualified / admitted_tasks}


def evaluate_frontier_v2(data):
    """Source-free projection of prequalified, precommitted artifact trials.

    `status=complete` denotes this calculation, not scientific certification or
    campaign completeness. Sorted opaque roster IDs define output indices.
    """
    from .tdl.common import MetricError, count, failure, require

    try:
        require(type(data) is dict)
        roster = data.get("admitted_tasks")
        require(type(roster) is list and bool(roster)
                and all(type(task) is str and task for task in roster), "invalid_roster")
        require(len(set(roster)) == len(roster), "duplicate_task")
        indexed = {task: index for index, task in enumerate(sorted(roster))}
        for key in ("method_id", "seed_id"):
            require(type(data.get(key)) is str and bool(data[key]), "invalid_frontier_identity")
        grid = data.get("byte_grid")
        require(type(grid) is list and bool(grid), "invalid_byte_grid")
        grid = [count(cap, 1) for cap in grid]
        require(len(set(grid)) == len(grid), "duplicate_budget")
        monotone = data.get("require_monotone", False)
        require(type(monotone) is bool, "invalid_monotone_policy")
        cells = data.get("cells")
        require(type(cells) is list, "invalid_cells")
        trials, identities = [], set()
        states = {"qualified", "unqualified", "failed", "crashed", "technical_failure", "incomplete",
                  "timeout", "timed_out", "oom", "unsupported", "unevaluated", "unresolved"}
        for cell in cells:
            require(type(cell) is dict, "invalid_cells")
            for key in ("method_id", "seed_id"):
                require(cell.get(key) == data[key], "frontier_identity_mismatch")
            task = cell.get("task_id")
            require(type(task) is str, "invalid_roster_task")
            require(task in indexed, "unscheduled_task")
            candidate = cell.get("candidate_id")
            require(type(candidate) is str and candidate, "invalid_candidate_identity")
            identity = (indexed[task], candidate)
            require(identity not in identities, "duplicate_candidate")
            identities.add(identity)
            state = cell.get("status")
            require(type(state) is str and state in states, "invalid_cell_status")
            cap = count(cell.get("search_cap_bytes"), 1)
            size = None if cell.get("artifact_bytes") is None else count(cell["artifact_bytes"], 1)
            require(size is None or size <= cap, "artifact_exceeds_budget")
            require(state != "qualified" or size is not None, "missing_qualified_artifact")
            trials.append(ArtifactTrial(indexed[task], cap, size, state == "qualified"))
        rows = []
        for cap in sorted(grid):
            coverage = roster_coverage(len(roster), trials, cap, monotone)
            rows.append({"budget_bytes": cap, **{k: coverage[k] for k in ("qualified_tasks", "admitted_tasks", "coverage")}})
        minimum = roster_coverage(len(roster), trials, (1 << 64) - 1, monotone)["minimum_bytes"]
        return {"protocol_version": 2, "status": "complete", "reason": None,
                "admitted_tasks": len(roster), "minimum_qualified_bytes": minimum,
                "frontier": rows, "require_monotone": monotone,
                "non_monotone_tasks": sum(non_monotone(trials, task) for task in range(len(roster))),
                "global_optimum_claimed": False}
    except MetricError as error:
        return failure(str(error))
