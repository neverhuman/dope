"""`SingleEncV8(code_dim=...)`: the wire default is unchanged and a wider research code starts at analytic.

The capacity x channel probe trains decoders that read a wider learned code than the 64 FSQ values (32 B)
the DSE8 wire carries. These tests pin that (1) the default model keeps today's parameter shapes, so
existing checkpoints and the Rust receiver are unaffected, and (2) any code width or model width starts
with every residual output exactly zero, so each probe arm begins at the analytic packet.
"""

from __future__ import annotations

import pytest
import torch

from research.levels.v8_evidence import COL_DIM as EV_COL_DIM
from research.levels.v8_evidence import EDGE_DIM as EV_EDGE_DIM
from research.levels.v8_evidence import GLOB_DIM as EV_GLOB_DIM
from research.levels.v8_packet import DEC_CODE_DIM, DEC_COL_DIM, DEC_EDGE_DIM, DEC_GLOB_DIM
from research.levels.v8_single_encoding import SingleEncV8

RESIDUAL_OUTPUTS = ("delta_knots", "delta_L", "delta_W", "node_leaves", "bilin_u", "bilin_v", "bilin_w")


def _inputs(b: int = 2, p: int = 5) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    return {
        "ev_col": torch.randn(b, p, EV_COL_DIM, generator=g),
        "ev_edge": torch.randn(b, p, p, EV_EDGE_DIM, generator=g),
        "ev_glob": torch.randn(b, EV_GLOB_DIM, generator=g),
        "dec_col": torch.randn(b, p, DEC_COL_DIM, generator=g),
        "dec_edge": torch.randn(b, p, p, DEC_EDGE_DIM, generator=g),
        "dec_glob": torch.randn(b, DEC_GLOB_DIM, generator=g),
        "mask": torch.ones(b, p),
    }


def test_default_code_dim_keeps_wire_shapes() -> None:
    torch.manual_seed(0)
    default = SingleEncV8()
    torch.manual_seed(0)
    explicit = SingleEncV8(code_dim=DEC_CODE_DIM)
    a, b = default.state_dict(), explicit.state_dict()
    assert DEC_CODE_DIM == 64
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a), "default init must stay bit-identical"
    assert a["dec_code_in.weight"].shape == (default.d_model, 64)
    assert a["enc_code_head.weight"].shape == (64, default.d_model)


@pytest.mark.parametrize(("d_model", "code_dim"), [(128, 64), (128, 1024), (256, 64), (256, 1024)])
def test_probe_arms_start_at_analytic(d_model: int, code_dim: int) -> None:
    torch.manual_seed(1)
    model = SingleEncV8(d_model=d_model, n_heads=4, n_enc_layers=4, n_dec_layers=2, code_dim=code_dim).eval()
    x = _inputs()
    with torch.no_grad():
        z, _ = model.encode_evidence(x["ev_col"], x["ev_edge"], x["ev_glob"], col_mask=x["mask"])
        assert z.shape == (2, code_dim)
        plan = model.decode_plan(x["dec_col"], x["dec_edge"], x["dec_glob"], z, col_mask=x["mask"])
    for name in RESIDUAL_OUTPUTS:
        assert torch.count_nonzero(plan[name]) == 0, name


def test_decoder_only_code_matches_width() -> None:
    model = SingleEncV8(code_dim=1024, with_owner_encoder=False)
    x = _inputs()
    z, g_pool = model.encode_evidence(x["ev_col"], x["ev_edge"], x["ev_glob"])
    assert z.shape == (2, 1024) and torch.count_nonzero(z) == 0
    assert g_pool.shape == (2, model.d_model)


def test_param_counts_of_the_probe_ladder() -> None:
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    small = count(SingleEncV8(d_model=128, code_dim=64))
    wide_code = count(SingleEncV8(d_model=128, code_dim=1024))
    big = count(SingleEncV8(d_model=256, code_dim=64))
    # The wider code adds only its two projections; d256 roughly quadruples the trunk.
    assert wide_code - small == 2 * 128 * (1024 - 64) + (1024 - 64)
    assert 3.0 < big / small < 4.5
