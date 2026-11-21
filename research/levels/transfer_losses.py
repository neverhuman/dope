"""Decision losses for measured response scores.

The baseline objective in ``losses.py`` is not called. Query values are
refused on score-panel losses. ``top_choice_indicator`` is nondifferentiable
and is not on the gradient path. Coupled TSTR is opt-in for ``v4_objectives``.
"""

import math
import torch
from torch.nn import functional as F

TSTR_INTEGRATION = "opt_in_v4_objectives"
NONDIFFERENTIABLE = ("top_choice_indicator",)


class DecisionLossError(ValueError):
    """Static refusal. The message is a code, not a source value."""

    purpose = "compute decision-supervised and coupled TSTR losses over measured panels"
    reason = "an input tensor shape, mask, finiteness invariant, or query boundary was violated"
    common_fixes = "pass finite matching tensors, boolean masks, and disjoint FIT and query arrays"
    docs_url = "docs/testing.md#local-repair"
    repair_hint = "fix the offending caller tensor shapes or masks and rerun test_transfer_losses"


def pairwise_logistic(
    scores,
    outcomes,
    mask,
    *,
    higher_is_better,
    query_values=None,
    margin=0.0,
    temperature=1.0,
    pair_weights=None,
):
    """Logistic loss on strict measured utility gaps with optional margin, temperature, and pair weights.

    The denominator is the number (or total weight) of ordered pairs with
    utility gap strictly exceeding ``margin``. Masked rows do not enter that
    count. A nonfinite active row is refused.
    """
    _refuse_query(query_values)
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(margin)
        or margin < 0.0
    ):
        raise DecisionLossError("decision_loss_value")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise DecisionLossError("decision_loss_value")
    scores, outcomes, valid = _vectors(scores, outcomes, mask)
    utility = _utility(outcomes, higher_is_better)
    gap = utility[:, None] - utility[None, :]
    pair = valid[:, None] & valid[None, :] & (gap > float(margin))
    count = int(pair.sum().item())
    if count == 0:
        return _zero(scores), _record(0, "pairwise")
    score_gap = (scores[:, None] - scores[None, :]) / float(temperature)
    safe_gap = torch.where(pair, score_gap, torch.zeros((), device=scores.device, dtype=scores.dtype))
    if pair_weights is None:
        loss = (F.softplus(-safe_gap) * pair.to(scores.dtype)).sum() / count
    else:
        if (
            not torch.is_tensor(pair_weights)
            or pair_weights.shape != pair.shape
            or not pair_weights.is_floating_point()
            or pair_weights.device != scores.device
        ):
            raise DecisionLossError("decision_loss_shape")
        weights = pair_weights.detach().to(dtype=scores.dtype)
        active_w = torch.where(pair, weights, torch.zeros((), device=scores.device, dtype=scores.dtype))
        if bool((~torch.isfinite(active_w)).any()) or bool((active_w < 0).any()):
            raise DecisionLossError("decision_loss_value")
        denom = active_w.sum()
        if float(denom.item()) <= 0.0:
            return _zero(scores), _record(0, "pairwise")
        loss = (F.softplus(-safe_gap) * active_w).sum() / denom
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
    if not bool(torch.isfinite(counts).all()):
        raise DecisionLossError("decision_loss_nonfinite")
    if bool((counts < 0).any()):
        raise DecisionLossError("decision_loss_weights")
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
        weight_total = weights.sum()
        weight_value = float(weight_total.detach().item())
        if weight_value <= 0.0:
            return _zero(losses), _record(0, "pair_weighted")
        denom = int(round(weight_value)) if bool((weights == weights.round()).all()) else weight_value
        return _finite((safe * weights).sum() / weight_total), _record(denom, "pair_weighted")
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


def coupled_tstr_loss(fit_x, fit_y, query_x, query_y, synth_x, synth_y, *, ridge_lambda=1e-2):
    """Differentiable FP32 ridge learner on synthetic rows scored by Huber loss on disjoint query rows."""
    if isinstance(ridge_lambda, bool) or not isinstance(ridge_lambda, (int, float)):
        raise DecisionLossError("decision_loss_ridge")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
        raise DecisionLossError("decision_loss_ridge")
    tensors = (fit_x, fit_y, query_x, query_y, synth_x, synth_y)
    if any(not torch.is_tensor(t) or not t.is_floating_point() for t in tensors):
        raise DecisionLossError("decision_loss_shape")
    if (
        fit_x.ndim != 2
        or query_x.ndim != 2
        or synth_x.ndim != 2
        or fit_y.ndim != 1
        or query_y.ndim != 1
        or synth_y.ndim != 1
        or fit_x.shape[0] < 1
        or query_x.shape[0] < 1
        or synth_x.shape[0] < 1
        or fit_x.shape[1] < 1
        or fit_x.shape[1] != query_x.shape[1]
        or fit_x.shape[1] != synth_x.shape[1]
        or fit_y.shape != (fit_x.shape[0],)
        or query_y.shape != (query_x.shape[0],)
        or synth_y.shape != (synth_x.shape[0],)
        or any(t.device != fit_x.device for t in tensors[1:])
    ):
        raise DecisionLossError("decision_loss_shape")
    if any(bool((~torch.isfinite(t)).any()) for t in tensors):
        raise DecisionLossError("decision_loss_nonfinite")
    if fit_x.data_ptr() == query_x.data_ptr() or fit_y.data_ptr() == query_y.data_ptr():
        raise DecisionLossError("decision_loss_query")
    with torch.autocast(device_type=fit_x.device.type, enabled=False):
        fx = fit_x.detach().to(torch.float32)
        qx = query_x.detach().to(torch.float32)
        qy = query_y.detach().to(torch.float32)
        sx = synth_x.to(torch.float32)
        sy = synth_y.to(torch.float32)
        mu = fx.mean(dim=0, keepdim=True)
        scale = fx.std(dim=0, correction=0, keepdim=True).clamp_min(1e-6)
        n_synth, width = sx.shape
        ones_s = torch.ones((n_synth, 1), dtype=torch.float32, device=sx.device)
        ones_q = torch.ones((qx.shape[0], 1), dtype=torch.float32, device=sx.device)
        design_s = torch.cat([ones_s, (sx - mu) / scale], dim=1)
        design_q = torch.cat([ones_q, (qx - mu) / scale], dim=1)
        reg = torch.full((width + 1,), float(ridge_lambda), dtype=torch.float32, device=sx.device)
        reg[0] = 1e-6
        gram = (design_s.T @ design_s) / float(n_synth) + torch.diag(reg)
        rhs = (design_s.T @ sy.unsqueeze(1)) / float(n_synth)
        try:
            weights = torch.linalg.solve(gram, rhs).squeeze(1)
        except RuntimeError:
            raise DecisionLossError("decision_loss_singular") from None
        pred_q = design_q @ weights
        loss = F.huber_loss(pred_q, qy)
    return _finite(loss), _record(int(qy.shape[0]), "coupled_tstr")


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
