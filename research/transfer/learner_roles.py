"""Successor learner roles. Historical v1 required/held-out contracts stay unchanged.

selection_families choose a development learner. held_out_families cannot win
that choice and must be evaluation_required_families when claimed. This record
does not freeze a protocol, open a campaign, or issue native certification.
"""

from __future__ import annotations

from types import MappingProxyType

from .losses import finite_number


FORMAT = "dope-learner-roles"
VERSION = 2


def _families(value, code):
    if type(value) is not list or not value or len(set(value)) != len(value):
        raise ValueError(code)
    if any(type(item) is not str or not item for item in value):
        raise ValueError(code)
    return tuple(value)


def assign_learner_roles(record) -> MappingProxyType:
    """Partition one version-2 role record. A v1 required/held-out object is refused."""
    if not isinstance(record, dict):
        raise ValueError("invalid_learner_roles")
    if "required_families" in record or record.get("version") == 1:
        raise ValueError("legacy_role_identity")
    allowed = {
        "format", "version", "selection_families", "evaluation_required_families", "held_out_families",
    }
    if set(record) != allowed or record.get("format") != FORMAT or record.get("version") != VERSION:
        raise ValueError("invalid_learner_roles")
    selection = _families(record["selection_families"], "invalid_selection_families")
    evaluation = _families(record["evaluation_required_families"], "invalid_evaluation_families")
    held_out = record["held_out_families"]
    held = _families(held_out, "invalid_held_out_families") if held_out else ()
    if held_out != list(held):
        raise ValueError("invalid_held_out_families")
    if set(selection) & set(held):
        raise ValueError("held_out_selected")
    if set(held) - set(evaluation):
        raise ValueError("held_out_not_evaluated")
    return MappingProxyType({
        "format": FORMAT,
        "version": VERSION,
        "selection_families": selection,
        "evaluation_required_families": evaluation,
        "held_out_families": held,
        "selection_not_evaluated": tuple(sorted(set(selection) - set(evaluation))),
        "protocol_frozen": False,
        "confirmation_allowed": False,
        "campaign": "unmeasured",
        "native_certification": False,
        "qualification": False,
    })


def select_development_learner(candidates, roles) -> MappingProxyType:
    """Minimize development loss inside selection families. Held-out families are not scored."""
    assigned = assign_learner_roles(roles)
    held = set(assigned["held_out_families"])
    selection = set(assigned["selection_families"])
    best = None
    chosen = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid_learner_candidate")
        family = candidate.get("family_id")
        learner = candidate.get("learner_id")
        if type(family) is not str or type(learner) is not str or not family or not learner:
            raise ValueError("invalid_learner_candidate")
        if family in held or family not in selection:
            continue
        loss = finite_number(candidate.get("development_loss"))
        rank = (loss, learner)
        if best is None or rank < best:
            best = rank
            chosen = learner
    return MappingProxyType({
        **dict(assigned),
        "selected_learner": chosen,
        "held_out_excluded_from_selection": True,
    })
