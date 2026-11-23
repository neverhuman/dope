"""Unit tests for KPI-aligned V8 losses, Gumbel-Rao K=4 variance reduction, and PCGrad-Lite."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from research.levels.s3_v8_losses import (
    compute_v8_episode_loss,
    loss_copula_corr,
    loss_pit_cvm,
    loss_privacy_dcr_hinge,
    pcgrad_lite_combine,
)
from research.levels.s3_v8_priors import sample_contrastive_xor_pair, sample_v8_prior_episode
from research.levels.v8_single_encoding import entmax_15_torch, fsq_4bit_ste, gumbel_rao_ste


@pytest.mark.parametrize("task_id,n_classes", [(0, 1), (1, 2), (2, 4)])
def test_v8_losses_finite_and_differentiable(task_id: int, n_classes: int) -> None:
    torch.manual_seed(11 + task_id)
    rng = np.random.default_rng(22 + task_id)
    ep = sample_v8_prior_episode(rng, n_fit=120, n_qry=96, p=6, task_id=task_id)
    Xf = torch.from_numpy(ep["X_fit"])
    yf = torch.from_numpy(ep["y_fit"])
    Xq = torch.from_numpy(ep["X_qry"])
    yq = torch.from_numpy(ep["y_qry"])

    x_syn = torch.randn(96, 6, requires_grad=True)
    logits = torch.randn(96, max(n_classes, 4), requires_grad=True)
    y_reg = torch.randn(96, requires_grad=True)
    resid_sd = torch.tensor(0.5, requires_grad=True)

    ld = compute_v8_episode_loss(
        x_syn, logits, y_reg, resid_sd, Xf, yf, Xq, yq, task_id, ep["n_classes"]
    )
    for k, v in ld.items():
        assert torch.isfinite(v), f"non-finite loss term {k}: {v}"
    ld["total"].backward()
    assert x_syn.grad is not None and torch.all(torch.isfinite(x_syn.grad))
    assert float(x_syn.grad.abs().sum()) > 0.0
    if task_id == 0:
        assert y_reg.grad is not None and float(y_reg.grad.abs().sum()) > 0.0
    else:
        assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0


def test_gumbel_rao_fsq_entmax_and_pcgrad() -> None:
    torch.manual_seed(7)
    # 1. entmax_1.5 sparsity & exact simplex
    p = entmax_15_torch(torch.tensor([[3.0, 1.0, -5.0, -5.0]]))
    assert abs(float(p.sum()) - 1.0) < 1e-5
    assert float(p[0, 2]) == 0.0

    # 2. FSQ 4-bit STE preserves gradients & snaps forward values to 16 uniform levels
    z = torch.linspace(-1.5, 1.5, 16, requires_grad=True)
    zq = fsq_4bit_ste(z)
    assert torch.all((zq >= -1.0 - 1e-5) & (zq <= 1.0 + 1e-5))
    zq.sum().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0.0

    # 3. Gumbel-Rao K=4 emits exact one-hot forward with finite gradients
    lg = torch.randn(32, 4, requires_grad=True)
    y = gumbel_rao_ste(lg, tau=0.8, k_mc=4)
    assert torch.allclose(y.sum(dim=-1), torch.ones(32))
    assert torch.all((y == 0.0) | (y == 1.0))
    (y * torch.arange(4, dtype=torch.float32)).sum().backward()
    assert lg.grad is not None and torch.all(torch.isfinite(lg.grad))

    # 4. PCGrad-Lite cancels directly opposing gradients!
    w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    l1 = w[0] + w[1]
    l2 = -w[0] + w[1]
    comb, tel = pcgrad_lite_combine([l1, l2], [w])
    assert tel["min_grad_cos"] <= 0.1
    assert float(comb[0][1]) > 0.0  # cooperative component preserved!

    # 5. Closed-form PIT/copula/DCR privacy hinge sanity
    x = torch.randn(64, 4)
    assert float(loss_pit_cvm(x, x)) < 1e-6
    assert float(loss_copula_corr(x, x)) < 1e-6
    # Copying X_fit verbatim MUST trigger a positive privacy hinge penalty vs disjoint X_qry!
    x_qry = torch.randn(64, 4) + 2.0
    assert float(loss_privacy_dcr_hinge(x, x, x_qry)) > 0.5

    # 6. Contrastive XOR vs XNOR pair has identical X and opposite Y
    ep_a, ep_b = sample_contrastive_xor_pair(np.random.default_rng(5))
    assert np.array_equal(ep_a["X_fit"], ep_b["X_fit"])
    assert np.allclose(ep_a["y_fit"] + ep_b["y_fit"], 1.0)
