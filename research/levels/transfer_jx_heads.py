"""Decision-supervised DOPE-JX nested prefix heads and invariance losses.

Nested prefix heads read ``task_latent[:, :prefix]`` zero-padded outside
``:prefix`` without SVD. Only ``pair_noise`` consumes ``dist_latent``.
Preference-reversal supervision cancels task-additive scalar offsets and
dataset-invariant program priors identically while rewarding bilinear
dataset-by-program interaction.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "dope-jx-heads/v1"
LATENT_WIDTH = 128
PREFIXES = (32, 64, 96, 128)
HEAD_KINDS = {
    "rank_utility": "huber",
    "calibration": "huber",
    "validity": "bce",
    "pair_noise": "huber",
}


class JXHeadError(ValueError):
    """Static refusal. The message is a fixed error code, never a source value."""

    purpose = "evaluate nested DOPE-JX prefix heads and invariance losses"
    reason = "an input latent width, prefix, mask, label, margin, or norm contract failed"
    common_fixes = "pass finite 128-wide latents, legal prefixes (32, 64, 96, 128), and nonzero row norms"
    docs_url = "docs/testing.md#local-repair"
    repair_hint = "correct the tensor shapes or active masks and rerun test_transfer_jx_heads"


class JXPrefixHeads(nn.Module):
    """Latent-only Matryoshka prefix heads; only pair_noise reads dist_latent."""

    def __init__(self, *, program_slots: int = 1, pair_noise_dim: int = 1) -> None:
        super().__init__()
        if type(program_slots) is not int or type(pair_noise_dim) is not int or program_slots < 1 or pair_noise_dim < 1:
            raise JXHeadError("jx_head_shape")
        self.program_slots = program_slots
        self.pair_noise_dim = pair_noise_dim
        self.rank_utility = nn.Linear(LATENT_WIDTH, program_slots)
        self.calibration = nn.Linear(LATENT_WIDTH, program_slots)
        self.validity = nn.Linear(LATENT_WIDTH, program_slots)
        self.pair_noise = nn.Linear(2 * LATENT_WIDTH, pair_noise_dim)

    @staticmethod
    def mask_prefix(latent: torch.Tensor, prefix: int) -> torch.Tensor:
        if type(prefix) is not int or prefix not in PREFIXES:
            raise JXHeadError("jx_head_prefix")
        if prefix == LATENT_WIDTH:
            return latent
        tail = torch.zeros((latent.shape[0], LATENT_WIDTH - prefix), dtype=latent.dtype, device=latent.device)
        return torch.cat([latent[:, :prefix], tail], dim=-1)

    def forward(
        self, task_latent: torch.Tensor, dist_latent: torch.Tensor, *, prefix: int = LATENT_WIDTH
    ) -> dict[str, torch.Tensor]:
        _check_latent_pair(task_latent, dist_latent, LATENT_WIDTH)
        if any(param.device != task_latent.device for param in self.parameters()):
            raise JXHeadError("jx_head_shape")
        task_pad = self.mask_prefix(task_latent, prefix)
        dist_pad = self.mask_prefix(dist_latent, prefix)
        return {
            "rank_utility": self.rank_utility(task_pad),
            "calibration": self.calibration(task_pad),
            "validity": self.validity(task_pad),
            "pair_noise": self.pair_noise(torch.cat([task_pad, dist_pad], dim=-1)),
        }


def matryoshka_prefix_loss(
    heads: JXPrefixHeads,
    task_latent: torch.Tensor,
    dist_latent: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    weights: Mapping[int, float] | Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Nested prefix Huber/BCE loss across (32, 64, 96, 128) with masked nonfinite refusal."""
    if not isinstance(heads, JXPrefixHeads):
        raise JXHeadError("jx_head_shape")
    _check_latent_pair(task_latent, dist_latent, LATENT_WIDTH)
    if not isinstance(targets, Mapping) or not isinstance(masks, Mapping) or not targets:
        raise JXHeadError("jx_head_shape")
    if set(targets.keys()) != set(masks.keys()) or not set(targets.keys()).issubset(HEAD_KINDS):
        raise JXHeadError("jx_head_shape")
    prefix_weights = _normalize_prefix_weights(weights)
    total = _zero(task_latent) + _zero(dist_latent)
    per_prefix: dict[int, float] = {}
    active_total = 0
    for prefix in PREFIXES:
        preds = heads(task_latent, dist_latent, prefix=prefix)
        prefix_loss, prefix_terms = _zero(task_latent), 0
        for name, target in targets.items():
            aligned_pred, aligned_target, valid = _align_head_tensors(preds[name], target, masks[name])
            count = int(valid.sum().item())
            if prefix == PREFIXES[0]:
                active_total += count
            if count == 0:
                continue
            prefix_terms += 1
            if HEAD_KINDS[name] == "bce":
                chosen_t = aligned_target[valid]
                if bool(((chosen_t != 0) & (chosen_t != 1)).any()):
                    raise JXHeadError("jx_head_label")
                term = F.binary_cross_entropy_with_logits(aligned_pred[valid], chosen_t)
            else:
                term = F.huber_loss(aligned_pred[valid], aligned_target[valid])
            prefix_loss = prefix_loss + _finite(term)
        if prefix_terms > 0:
            prefix_loss = prefix_loss / prefix_terms
        per_prefix[prefix] = float(prefix_loss.detach().item())
        if prefix_weights[prefix] > 0.0:
            total = total + prefix_weights[prefix] * prefix_loss
    return _finite(total), {
        "schema": SCHEMA,
        "prefixes": PREFIXES,
        "per_prefix_loss": per_prefix,
        "active_entries": active_total,
    }


def preference_reversal_loss(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    outcomes_a: torch.Tensor,
    outcomes_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    *,
    margin: float = 0.05,
    temperature: float = 1.0,
    higher_is_better: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Pair-of-pairs logistic loss on verified cross-dataset preference reversals."""
    _require_scalar(margin, min_val=0.0, strict_min=False, code="jx_head_margin")
    _require_scalar(temperature, min_val=0.0, strict_min=True, code="jx_head_temperature")
    if type(higher_is_better) is not bool:
        raise JXHeadError("jx_head_shape")
    sa, ua, va = _score_vector(scores_a, outcomes_a, mask_a, higher_is_better)
    sb, ub, vb = _score_vector(scores_b, outcomes_b, mask_b, higher_is_better)
    if sa.shape != sb.shape or sa.device != sb.device or sa.dtype != sb.dtype:
        raise JXHeadError("jx_head_shape")
    both_valid = va & vb
    reversals = both_valid[:, None] & both_valid[None, :] & ((ua[:, None] - ua[None, :]) > margin) & ((ub[:, None] - ub[None, :]) < -margin)
    count = int(reversals.sum().item())
    if count == 0:
        return _zero(sa) + _zero(sb), {"schema": SCHEMA, "reversal_pairs": 0, "margin": float(margin)}
    adv = ((sa[:, None] - sa[None, :]) - (sb[:, None] - sb[None, :])) / float(temperature)
    safe_adv = torch.where(reversals, adv, torch.zeros((), dtype=sa.dtype, device=sa.device))
    loss = (F.softplus(-safe_adv) * reversals.to(sa.dtype)).sum() / count
    mean_adv = float((safe_adv.detach() * reversals.to(sa.dtype)).sum().item() / count)
    return _finite(loss), {"schema": SCHEMA, "reversal_pairs": count, "mean_advantage": mean_adv, "margin": float(margin)}


def task_additive_offset_control(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    program_vectors: torch.Tensor,
    outcomes_a: torch.Tensor,
    outcomes_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    *,
    offset_a: float = 0.75,
    offset_b: float = -0.5,
    program_prior: torch.Tensor | None = None,
    margin: float = 0.05,
    temperature: float = 1.0,
) -> dict[str, float | int | str]:
    """Verify task-only scalar a(D) + prior b_p cancels on reversals while bilinear z_D @ v_p learns."""
    if (
        not torch.is_tensor(z_a)
        or not torch.is_tensor(z_b)
        or not torch.is_tensor(program_vectors)
        or z_a.ndim != 1
        or z_a.shape != z_b.shape
        or program_vectors.ndim != 2
        or program_vectors.shape[1] != z_a.shape[0]
        or z_a.device != z_b.device
        or z_a.device != program_vectors.device
    ):
        raise JXHeadError("jx_head_shape")
    if any(bool((~torch.isfinite(t)).any()) for t in (z_a, z_b, program_vectors)):
        raise JXHeadError("jx_head_nonfinite")
    p_count = program_vectors.shape[0]
    a_var = torch.tensor(float(offset_a), dtype=z_a.dtype, device=z_a.device, requires_grad=True)
    b_var = torch.tensor(float(offset_b), dtype=z_a.dtype, device=z_a.device, requires_grad=True)
    if program_prior is None:
        prior_var = torch.linspace(-0.3, 0.3, p_count, dtype=z_a.dtype, device=z_a.device, requires_grad=True)
    elif torch.is_tensor(program_prior) and program_prior.shape == (p_count,):
        prior_var = program_prior.detach().to(dtype=z_a.dtype, device=z_a.device).requires_grad_(True)
    else:
        raise JXHeadError("jx_head_shape")
    add_loss, add_rec = preference_reversal_loss(
        a_var + prior_var, b_var + prior_var, outcomes_a, outcomes_b, mask_a, mask_b, margin=margin, temperature=temperature
    )
    add_grads = torch.autograd.grad(add_loss, (a_var, b_var, prior_var))
    za_var, zb_var = z_a.detach().clone().requires_grad_(True), z_b.detach().clone().requires_grad_(True)
    vp_var = program_vectors.detach().clone().requires_grad_(True)
    bi_loss, bi_rec = preference_reversal_loss(
        vp_var @ za_var, vp_var @ zb_var, outcomes_a, outcomes_b, mask_a, mask_b, margin=margin, temperature=temperature
    )
    bi_grads = torch.autograd.grad(bi_loss, (za_var, zb_var, vp_var))
    return {
        "schema": SCHEMA,
        "reversal_pairs": int(bi_rec["reversal_pairs"]),
        "additive_loss": float(add_loss.detach().item()),
        "additive_advantage": float(add_rec.get("mean_advantage", 0.0)),
        "additive_offset_grad_norm": float((add_grads[0].abs() + add_grads[1].abs()).item()),
        "static_prior_grad_norm": float(add_grads[2].abs().sum().item()),
        "bilinear_loss": float(bi_loss.detach().item()),
        "bilinear_advantage": float(bi_rec.get("mean_advantage", 0.0)),
        "bilinear_grad_norm": float(sum(g.norm().item() for g in bi_grads)),
    }


def split_invariance_infonce(
    z_a: torch.Tensor, z_b: torch.Tensor, family_ids: Sequence[str], *, temperature: float = 0.1
) -> tuple[torch.Tensor, dict[str, object]]:
    """Symmetric InfoNCE masking same_family & ~eye to -inf so sibling tables are not negatives."""
    _require_scalar(temperature, min_val=0.0, strict_min=True, code="jx_head_temperature")
    if not torch.is_tensor(z_a) or not torch.is_tensor(z_b) or z_a.ndim != 2 or z_a.shape != z_b.shape:
        raise JXHeadError("jx_head_shape")
    batch, width = z_a.shape
    if batch < 2 or width < 1 or z_a.device != z_b.device or not z_a.is_floating_point() or z_a.dtype != z_b.dtype:
        raise JXHeadError("jx_head_shape")
    if bool((~torch.isfinite(z_a)).any()) or bool((~torch.isfinite(z_b)).any()):
        raise JXHeadError("jx_head_nonfinite")
    if (
        not isinstance(family_ids, Sequence)
        or isinstance(family_ids, (str, bytes))
        or len(family_ids) != batch
        or any(not isinstance(fid, str) or not fid for fid in family_ids)
    ):
        raise JXHeadError("jx_head_shape")
    if bool((z_a.norm(dim=-1) <= 1e-12).any()) or bool((z_b.norm(dim=-1) <= 1e-12).any()):
        raise JXHeadError("jx_head_zero_norm")
    logits_ab = (F.normalize(z_a, p=2, dim=-1) @ F.normalize(z_b, p=2, dim=-1).T) / float(temperature)
    same_family = torch.tensor(
        [[family_ids[i] == family_ids[j] for j in range(batch)] for i in range(batch)],
        dtype=torch.bool,
        device=z_a.device,
    )
    sibling_mask = same_family & ~torch.eye(batch, dtype=torch.bool, device=z_a.device)
    targets = torch.arange(batch, dtype=torch.long, device=z_a.device)
    loss = 0.5 * (
        F.cross_entropy(logits_ab.masked_fill(sibling_mask, float("-inf")), targets)
        + F.cross_entropy(logits_ab.T.masked_fill(sibling_mask.T, float("-inf")), targets)
    )
    return _finite(loss), {
        "schema": SCHEMA,
        "batch": batch,
        "masked_sibling_pairs": int(sibling_mask.sum().item()),
        "eligible_negative_pairs": int((~same_family).sum().item()),
    }


def target_swap_disentangle_loss(
    task_a: torch.Tensor, dist_a: torch.Tensor, task_b: torch.Tensor, dist_b: torch.Tensor, *, margin: float = 0.5
) -> tuple[torch.Tensor, dict[str, object]]:
    """Pull shared-X dist latents together (1 - cos) while pushing swapped-y task latents apart."""
    _require_scalar(margin, min_val=0.0, strict_min=True, code="jx_head_margin")
    if margin > 2.0:
        raise JXHeadError("jx_head_margin")
    tensors = (task_a, dist_a, task_b, dist_b)
    if any(not torch.is_tensor(t) or t.ndim != 2 or not t.is_floating_point() for t in tensors):
        raise JXHeadError("jx_head_shape")
    if (
        task_a.shape != task_b.shape
        or dist_a.shape != dist_b.shape
        or task_a.shape[0] < 1
        or task_a.shape[1] < 1
        or dist_a.shape[1] < 1
        or task_a.shape[0] != dist_a.shape[0]
        or any(t.device != task_a.device or t.dtype != task_a.dtype for t in tensors[1:])
    ):
        raise JXHeadError("jx_head_shape")
    if any(bool((~torch.isfinite(t)).any()) for t in tensors):
        raise JXHeadError("jx_head_nonfinite")
    if any(bool((t.norm(dim=-1) <= 1e-12).any()) for t in tensors):
        raise JXHeadError("jx_head_zero_norm")
    dist_pull = (1.0 - F.cosine_similarity(dist_a, dist_b, dim=-1, eps=1e-12)).mean()
    task_push = F.relu(F.cosine_similarity(task_a, task_b, dim=-1, eps=1e-12) - (1.0 - float(margin))).mean()
    return _finite(dist_pull + task_push), {
        "schema": SCHEMA,
        "batch": int(task_a.shape[0]),
        "dist_pull": float(dist_pull.detach().item()),
        "task_push": float(task_push.detach().item()),
        "margin": float(margin),
    }


def _require_scalar(value: object, *, min_val: float, strict_min: bool, code: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise JXHeadError(code)
    if (value <= min_val) if strict_min else (value < min_val):
        raise JXHeadError(code)


def _check_latent_pair(task_latent: torch.Tensor, dist_latent: torch.Tensor, width: int) -> None:
    if (
        not torch.is_tensor(task_latent)
        or not torch.is_tensor(dist_latent)
        or task_latent.ndim != 2
        or task_latent.shape != dist_latent.shape
        or task_latent.shape[0] < 1
        or task_latent.shape[1] != width
        or not task_latent.is_floating_point()
        or task_latent.dtype != dist_latent.dtype
        or task_latent.device != dist_latent.device
    ):
        raise JXHeadError("jx_head_shape")
    if bool((~torch.isfinite(task_latent)).any()) or bool((~torch.isfinite(dist_latent)).any()):
        raise JXHeadError("jx_head_nonfinite")


def _normalize_prefix_weights(weights: Mapping[int, float] | Sequence[float] | None) -> dict[int, float]:
    if weights is None:
        return {p: 0.25 for p in PREFIXES}
    if isinstance(weights, Mapping) and set(weights.keys()) == set(PREFIXES):
        out = {p: float(weights[p]) for p in PREFIXES}
    elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)) and len(weights) == len(PREFIXES):
        out = {p: float(w) for p, w in zip(PREFIXES, weights)}
    else:
        raise JXHeadError("jx_head_weights")
    if any(not math.isfinite(v) or v < 0.0 for v in out.values()) or sum(out.values()) <= 0.0:
        raise JXHeadError("jx_head_weights")
    return out


def _align_head_tensors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(target) or not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise JXHeadError("jx_head_shape")
    if target.ndim == 1 and pred.shape == (target.shape[0], 1) and mask.shape == target.shape:
        pred = pred.squeeze(-1)
    if pred.shape != target.shape or pred.shape != mask.shape or pred.device != target.device or pred.device != mask.device:
        raise JXHeadError("jx_head_shape")
    if bool((~torch.isfinite(pred) & mask).any()) or bool((~torch.isfinite(target) & mask).any()):
        raise JXHeadError("jx_head_nonfinite")
    return pred, target.detach().to(dtype=pred.dtype), mask


def _score_vector(scores: torch.Tensor, outcomes: torch.Tensor, mask: torch.Tensor, higher_is_better: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        not torch.is_tensor(scores)
        or not torch.is_tensor(outcomes)
        or not torch.is_tensor(mask)
        or scores.ndim != 1
        or scores.numel() < 1
        or scores.shape != outcomes.shape
        or scores.shape != mask.shape
        or mask.dtype != torch.bool
        or not scores.is_floating_point()
        or scores.device != outcomes.device
        or scores.device != mask.device
    ):
        raise JXHeadError("jx_head_shape")
    if bool((~torch.isfinite(scores) & mask).any()) or bool((~torch.isfinite(outcomes) & mask).any()):
        raise JXHeadError("jx_head_nonfinite")
    utility = outcomes.detach() if higher_is_better else -outcomes.detach()
    return scores, utility.to(dtype=scores.dtype), mask


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(reference), reference, torch.zeros((), dtype=reference.dtype, device=reference.device)).sum() * 0.0


def _finite(loss: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(loss).all()):
        raise JXHeadError("jx_head_nonfinite")
    return loss
