"""Family exposure for one durable schedule.

Selection families stay exposed. This report does not qualify a learner.
"""

from research.transfer.learner_roles import assign_learner_roles
from research.transfer.scheduler import DurableSchedule, ExecutionError


class ResearchExecutor:
    """Bind one schedule and report whether a learner family was held out."""

    def __init__(self, schedule):
        if type(schedule) is not DurableSchedule:
            raise ExecutionError("invalid_execution_plan")
        self.schedule = schedule

    def report_family_exposure(self, roles, family_id, learner_id, *, claimed_held_out=False):
        assigned = assign_learner_roles(roles)
        if type(family_id) is not str or not family_id or type(learner_id) is not str or not learner_id:
            raise ExecutionError("invalid_family")
        selection = family_id in assigned["selection_families"]
        held = family_id in assigned["held_out_families"]
        if selection:
            if claimed_held_out:
                raise ExecutionError("selection_not_held_out")
            return {
                "family_id": family_id,
                "learner_id": learner_id,
                "held_out": False,
                "reason": "selection_family",
            }
        return {
            "family_id": family_id,
            "learner_id": learner_id,
            "held_out": held,
            "reason": "held_out_family" if held else "not_held_out",
        }
