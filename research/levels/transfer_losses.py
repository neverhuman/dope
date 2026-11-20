"""Decision losses for measured response scores.

The baseline objective in ``losses.py`` is not called. Query values are
refused. ``top_choice_indicator`` is nondifferentiable and is not on the
gradient path. Optional TSTR is not connected to ``v4_objectives``.
"""

import torch
from torch.nn import functional as F

TSTR_INTEGRATION = "not_connected"
NONDIFFERENTIABLE = ("top_choice_indicator",)


class DecisionLossError(ValueError):
    """Static refusal. The message is a code, not a source value."""


def pairwise_logistic(scores, outcomes, mask, *, higher_is_better, query_values=None):
    """Logistic loss on strict measured utility gaps.

    The denominator is the number of ordered pairs with a positive utility
    gap. Masked rows do not enter that count. A nonfinite active row is refused.
    """
    _refuse_query(query_values)
    scores, outcomes, valid = _vectors(scores, outcomes, mask)
    utility = _utility(outcomes, higher_is_better)
    gap = utility[:, None] - utility[None, :]
    pair = valid[:, None] & valid[None, :] & (gap > 0)
    count = int(pair.sum().item())
    if count == 0:
        return _zero(scores), _record(0, "pairwise")
    score_gap = scores[:, None] - scores[None, :]
    safe_gap = torch.where(pair, score_gap, torch.zeros((), device=scores.device, dtype=scores.dtype))
    loss = (F.softplus(-safe_gap) * pair.to(scores.dtype)).sum() / count
    return _finite(loss), _record(count, "pairwise")


def masked_calibration(scores, outcomes, mask, *, higher_is_better, query_values=None):
    """Huber loss against detached measured utilities."""
    _refuse_query(query_values)
    scores, outcomes, valid = _vectors(scores, outcomes, mask)
    target = _utility(outcomes, higher_is_better)
    count = int(valid.sum().item())
    if count == 0:
        return _zero(scores), _record(0, "calibration")
    loss = F.huber_loss(scores[valid], target[valid])
    return _finite(loss), _record(count, "calibration")


def validity_bce(logits, labels, mask, *, query_values=None):
    """Binary validity loss. Labels must be zero or one on the mask."""
    _refuse_query(query_values)
    logits, labels, valid = _vectors(logits, labels, mask)
    count = int(valid.sum().item())
    if count == 0:
        return _zero(logits), _record(0, "validity")
    target = labels[valid]
    if bool(((target != 0) & (target != 1)).any()):
        raise DecisionLossError("decision_loss_label")
    loss = F.binary_cross_entropy_with_logits(logits[valid], target)
    return _finite(loss), _record(count, "validity")


def teacher_distillation(student, teacher, mask, *, query_values=None):
    """Match detached teacher scores. The teacher receives no gradient."""
    _refuse_query(query_values)
    student, teacher, valid = _vectors(student, teacher, mask)
    count = int(valid.sum().item())
    if count == 0:
        return _zero(student), _record(0, "teacher")
    loss = F.mse_loss(student[valid], teacher[valid])
    return _finite(loss), _record(count, "teacher")


def reduce_tasks(losses, pair_counts, *, mode, family_ids=None):
    """Reduce per-task losses. Zero-pair tasks stay out of every denominator."""
    if mode not in {"uniform", "pair_weighted", "family_balanced"}:
        raise DecisionLossError("decision_loss_reduction")
    if not torch.is_tensor(losses) or losses.ndim != 1 or losses.numel() == 0:
        raise DecisionLossError("decision_loss_shape")
    counts = torch.as_tensor(pair_counts, dtype=losses.dtype, device=losses.device)
    if counts.shape != losses.shape or counts.ndim != 1:
        raise DecisionLossError("decision_loss_shape")
    eligible = counts > 0
    if bool((~torch.isfinite(losses) & eligible).any()):
        raise DecisionLossError("decision_loss_nonfinite")
    if mode == "uniform":
        chosen = losses[eligible]
        if chosen.numel() == 0:
            return _zero(losses), _record(0, "uniform")
        return _finite(chosen.mean()), _record(int(chosen.numel()), "uniform")
    safe = torch.where(eligible, losses, torch.zeros((), dtype=losses.dtype, device=losses.device))
    if mode == "pair_weighted":
        weights = torch.where(eligible, counts, torch.zeros((), dtype=losses.dtype, device=losses.device))
        total = int(weights.sum().item())
        if total == 0:
            return _zero(losses), _record(0, "pair_weighted")
        return _finite((safe * weights).sum() / total), _record(total, "pair_weighted")
    if family_ids is None or len(family_ids) != int(losses.shape[0]):
        raise DecisionLossError("decision_loss_reduction")
    groups = {}
    for index, family in enumerate(family_ids):
        if bool(eligible[index]):
            groups.setdefault(family, []).append(index)
    if not groups:
        return _zero(losses), _record(0, "family_balanced")
    weight = torch.zeros_like(losses)
    share = 1.0 / len(groups)
    for members in groups.values():
        piece = share / len(members)
        for index in members:
            weight[index] = piece
    return _finite((safe * weight).sum()), _record(len(groups), "family_balanced")


def top_choice_indicator(scores, mask):
    """Argmax over the finite mask. This result is not differentiable."""
    if not torch.is_tensor(scores) or scores.ndim != 1 or mask.shape != scores.shape or mask.dtype != torch.bool:
        raise DecisionLossError("decision_loss_shape")
    if bool((mask & ~torch.isfinite(scores)).any()):
        raise DecisionLossError("decision_loss_nonfinite")
    valid = mask
    choice = None
    if bool(valid.any()):
        filled = scores.detach().masked_fill(~valid, float("-inf"))
        choice = int(torch.argmax(filled).item())
    return {"choice": choice, "differentiable": False, "component": "top_choice_indicator"}


def _refuse_query(query_values):
    if query_values is not None:
        raise DecisionLossError("decision_loss_query")


def _utility(outcomes, higher_is_better):
    directed = outcomes if higher_is_better else -outcomes
    return directed.detach()


def _vectors(left, right, mask):
    if (
        not torch.is_tensor(left)
        or not torch.is_tensor(right)
        or not torch.is_tensor(mask)
        or left.ndim != 1
        or left.shape != right.shape
        or left.shape != mask.shape
        or mask.dtype != torch.bool
    ):
        raise DecisionLossError("decision_loss_shape")
    active_left = ~torch.isfinite(left) & mask
    active_right = ~torch.isfinite(right) & mask
    if bool(active_left.any()) or bool(active_right.any()):
        raise DecisionLossError("decision_loss_nonfinite")
    return left, right.detach(), mask


def _zero(reference):
    safe = torch.where(
        torch.isfinite(reference),
        reference,
        torch.zeros((), dtype=reference.dtype, device=reference.device),
    )
    return safe.sum() * 0


def _finite(loss):
    if not bool(torch.isfinite(loss).all()):
        raise DecisionLossError("decision_loss_nonfinite")
    return loss


def _record(count, kind):
    return {
        "denominator": count,
        "kind": kind,
        "nondifferentiable": NONDIFFERENTIABLE,
        "tstr_integration": TSTR_INTEGRATION,
    }
