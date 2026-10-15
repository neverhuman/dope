"""Transfer-side allocation checks and provenance. Native search stays with the Rust owner.

A candidate whose estimate is inside the cap still fails when its actual bytes
are over the cap or unmeasured. Search order cannot drop a scheduled family.
Allocation gain is attributable only for a paired measured comparison.
"""

from __future__ import annotations


from .losses import finite_number


def admit_candidate(*, estimated_bytes: int, actual_bytes: int | None, cap: int) -> dict:
    if any(type(value) is not int or value < 0 for value in (estimated_bytes, cap)):
        raise ValueError("invalid byte accounting")
    if actual_bytes is not None and (type(actual_bytes) is not int or actual_bytes < 0):
        raise ValueError("invalid byte accounting")
    if estimated_bytes > cap:
        return {"admitted": False, "reason": "estimate_over_cap"}
    if actual_bytes is None:
        return {"admitted": False, "reason": "actual_bytes_unmeasured"}
    if actual_bytes > cap:
        return {"admitted": False, "reason": "actual_over_cap"}
    return {"admitted": True, "reason": None}


def allocation_gain(baseline_loss, allocated_loss) -> dict:
    if baseline_loss is None or allocated_loss is None:
        return {"gain": None, "attributable": False, "reason": "unmeasured"}
    baseline, allocated = finite_number(baseline_loss), finite_number(allocated_loss)
    return {"gain": finite_number(baseline - allocated), "attributable": True, "reason": "paired_comparison"}


def search_coverage(families_by_order) -> dict:
    orders = tuple(tuple(order) for order in families_by_order)
    if not orders or any(not order for order in orders):
        raise ValueError("empty search order")
    universe = set().union(*orders)
    excluded = sum(set(order) != universe for order in orders)
    return {"families": tuple(sorted(universe)), "complete": excluded == 0, "excluded_orders": excluded}


def provenance_row(*, candidate_id: str, parent_id: str | None, estimated_bytes: int,
                   actual_bytes: int | None, cost: float, rejection: str | None) -> dict:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("missing candidate")
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        raise ValueError("invalid parent candidate")
    if rejection is not None and (not isinstance(rejection, str) or not rejection):
        raise ValueError("invalid rejection reason")
    cost_val = finite_number(cost)
    if cost_val < 0:
        raise ValueError("invalid candidate cost")
    admit_candidate(estimated_bytes=estimated_bytes, actual_bytes=actual_bytes, cap=max(estimated_bytes, 0))
    return {"candidate_id": candidate_id, "parent_id": parent_id, "estimated_bytes": estimated_bytes,
            "actual_bytes": actual_bytes, "cost": cost_val, "rejection": rejection}
