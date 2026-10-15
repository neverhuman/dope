"""One development selection. Held-out oracle scores never select.

Failed risk sketches keep the cheaper admissible candidate. The return keeps
qualification false and power_measured false, and it includes no regret value.
"""

from __future__ import annotations

from .allocation import admit_candidate
from .losses import finite_number


def select_once(candidates, *, cap: int, frozen: bool = False) -> dict:
    if not isinstance(frozen, bool):
        raise ValueError("invalid frozen flag")
    if frozen:
        raise ValueError("selection already frozen")
    candidates = tuple(candidates)
    for candidate in candidates:
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("invalid candidate_id")
        if candidate.get("risk_sketch") not in ("ok", "failed", "absent"):
            raise ValueError("invalid risk sketch")
    if len({candidate["candidate_id"] for candidate in candidates}) != len(candidates):
        raise ValueError("duplicate candidate_id")
    considered, rejections, eligible = [], [], []
    sketches_failed = any(candidate["risk_sketch"] == "failed" for candidate in candidates)
    for candidate in candidates:
        considered.append(candidate["candidate_id"])
        if candidate.get("held_out_oracle") is not None:
            rejections.append({"candidate_id": candidate["candidate_id"], "reason": "held_out_oracle"})
            continue
        decision = admit_candidate(estimated_bytes=candidate["estimated_bytes"],
                                   actual_bytes=candidate["actual_bytes"], cap=cap)
        if not decision["admitted"]:
            rejections.append({"candidate_id": candidate["candidate_id"], "reason": decision["reason"]})
            continue
        if candidate["risk_sketch"] == "failed":
            rejections.append({"candidate_id": candidate["candidate_id"], "reason": "sketch_failed"})
            continue
        if candidate["risk_sketch"] == "absent":
            rejections.append({"candidate_id": candidate["candidate_id"], "reason": "sketch_absent"})
            continue
        dev_loss = finite_number(candidate["development_loss"])
        if dev_loss < 0:
            raise ValueError("negative development loss")
        eligible.append(candidate)
    pool = [candidate for candidate in eligible if candidate.get("cheaper") is True] if sketches_failed else eligible
    if sketches_failed and not pool:
        pool = []
    chosen = None if not pool else min(
        pool,
        key=lambda candidate: (
            finite_number(candidate["development_loss"]),
            candidate["actual_bytes"],
            candidate["estimated_bytes"],
            candidate["candidate_id"],
        ),
    )
    return {"selected": None if chosen is None else chosen["candidate_id"],
            "selected_once": True, "retraining_utility": None,
            "stored_target_label": "stored_target_fit",
            "considered": tuple(considered), "rejections": tuple(rejections),
            "provenance": tuple({"candidate_id": candidate["candidate_id"], "parent_id": candidate.get("parent_id"),
                                 "estimated_bytes": candidate["estimated_bytes"], "actual_bytes": candidate["actual_bytes"],
                                 "cost": candidate.get("cost")} for candidate in candidates),
            "qualification": False, "power_measured": False}
