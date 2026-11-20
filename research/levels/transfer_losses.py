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
    gap. Masked and nonfinite rows do not enter that count.
    """
    _refuse_query(query_values)
    scores, outcomes, valid = _vectors(scores, outcomes, mask)
    utility = _utility(outcomes, higher_is_better)
    gap = utility[:, None] - utility[None, :]
    pair = valid[:, None] & valid[None, :] & (gap > 0)
    count = int(pair.sum().item())
    if count == 0:
        return scores.sum() * 0, _record(0, "pairwise")
    score_gap = scores[:, None] - scores[None, :]
    safe_gap = torch.where(pair, score_gap, torch.zeros((), device=scores.device, dtype=scores.dtype))
    loss = (F.softplus(-safe_gap) * pair.to(scores.dtype)).sum() / count
    return loss, _record(count, "pairwise")


def masked_calibration(scores, outcomes, mask, *, higher_is_better, query_values=None):
    """Mean squared error against detached measured utilities."""
    _refuse_query(query_values)
    scores, outcomes, valid = _vectors(scores, outcomes, mask)
    target = _utility(outcomes, higher_is_better)
    count = int(valid.sum().item())
    if count == 0:
        return scores.sum() * 0, _record(0, "calibration")
    loss = F.mse_loss(scores[valid], target[valid])
    return loss, _record(count, "calibration")


def validity_bce(logits, labels, mask, *, query_values=None):
    """Binary validity loss. Labels must be zero or one on the mask."""
    _refuse_query(query_values)
    logits, labels, valid = _vectors(logits, labels, mask)
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0, _record(0, "validity")
    target = labels[valid]
    if bool(((target != 0) & (target != 1)).any()):
        raise DecisionLossError("decision_loss_label")
    loss = F.binary_cross_entropy_with_logits(logits[valid], target)
    return loss, _record(count, "validity")


def teacher_distillation(student, teacher, mask, *, query_values=None):
    """Match detached teacher scores. The teacher receives no gradient."""
    _refuse_query(query_values)
    student, teacher, valid = _vectors(student, teacher, mask)
    count = int(valid.sum().item())
    if count == 0:
        return student.sum() * 0, _record(0, "teacher")
    loss = F.mse_loss(student[valid], teacher[valid])
    return loss, _record(count, "teacher")


def reduce_tasks(losses, pair_counts, *, mode):
    """Reduce per-task losses. ``uniform`` and ``pair_weighted`` stay distinct."""
    if mode not in {"uniform", "pair_weighted"}:
        raise DecisionLossError("decision_loss_reduction")
    if not torch.is_tensor(losses) or losses.ndim != 1:
        raise DecisionLossError("decision_loss_shape")
    counts = torch.as_tensor(pair_counts, dtype=losses.dtype, device=losses.device)
    if counts.shape != losses.shape or counts.ndim != 1:
        raise DecisionLossError("decision_loss_shape")
    if mode == "uniform":
        if losses.numel() == 0:
            raise DecisionLossError("decision_loss_shape")
        return losses.mean(), _record(int(losses.numel()), "uniform")
    weights = counts.clamp_min(0)
    total = int(weights.sum().item())
    if total == 0:
        return losses.sum() * 0, _record(0, "pair_weighted")
    return (losses * weights).sum() / total, _record(total, "pair_weighted")


def top_choice_indicator(scores, mask):
    """Argmax over the finite mask. This result is not differentiable."""
    if not torch.is_tensor(scores) or scores.ndim != 1 or mask.shape != scores.shape or mask.dtype != torch.bool:
        raise DecisionLossError("decision_loss_shape")
    valid = mask & torch.isfinite(scores)
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
    valid = mask & torch.isfinite(left) & torch.isfinite(right)
    return left, right.detach(), valid


def _record(count, kind):
    return {
        "denominator": count,
        "kind": kind,
        "nondifferentiable": NONDIFFERENTIABLE,
        "tstr_integration": TSTR_INTEGRATION,
    }
