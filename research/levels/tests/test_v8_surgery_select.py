"""V7 -> V8 model surgery: exact trunk equivalence at equal width, step-0 identity, release weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from research.levels import v8_surgery as sg
from research.levels.v7_single_encoding import SingleEncV7
from research.levels.v8_single_encoding import SingleEncV8

ROOT = Path(__file__).resolve().parents[3]
RELEASE_V7 = ROOT / "models/single-enc-v7/model.safetensors"


def _manual_v7(m: SingleEncV7, col: torch.Tensor, glob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h = (m.col_in(col) + m.glob_in(glob).unsqueeze(1)) * mask.unsqueeze(-1)  # V7 refine_packet trunk
    for blk in m.blocks:
        h = blk(h, mask)
    return m.final_norm(h) * mask.unsqueeze(-1)


def _loaded(tensors: dict[str, torch.Tensor], seed: int = 3) -> SingleEncV8:
    torch.manual_seed(seed)
    v8 = SingleEncV8().eval()
    missing, unexpected = v8.load_state_dict(tensors, strict=False)
    assert not unexpected
    assert all(k.startswith(("enc_", "aux_ev_head.")) for k in missing)
    return v8


def _inputs(b: int = 3, p: int = 7, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    col7 = torch.randn(b, p, 46, generator=g)
    col7[..., 37] = 0.0  # is_const and the V7.2 nonlinear residuals have no V8 slot
    col7[..., 42:] = 0.0
    glob7 = torch.randn(b, 16, generator=g)
    glob7[:, 3] = 0.0
    col8 = torch.randn(b, p, 48, generator=g)  # V8-only slots stay random: their weights must be zero
    for i7, i8 in sg.V7_TO_V8_COL.items():
        col8[..., i8] = col7[..., i7]
    glob8 = torch.randn(b, 32, generator=g)
    for i7, i8 in sg.V7_TO_V8_GLOB.items():
        glob8[:, i8] = glob7[:, i7] / sg.GLOB_SCALE.get(i8, 1.0)
    mask = torch.ones(b, p)
    mask[0, -2:] = 0.0
    edge = torch.randn(b, p, p, 4, generator=g)
    code = torch.randn(b, 64, generator=g)
    return col7, glob7, col8, glob8, edge, code, mask


def test_equal_width_surgery_reproduces_the_v7_trunk_exactly():
    torch.manual_seed(11)
    v7 = SingleEncV7(d_model=128, n_heads=4, n_layers=2).eval()
    tensors, report = sg.transplant(v7.state_dict(), SingleEncV8())
    assert report["source_width"] == report["target_width"] == 128
    v8 = _loaded(tensors)
    col7, glob7, col8, glob8, edge, code, mask = _inputs()
    with torch.no_grad():
        h7 = _manual_v7(v7, col7, glob7, mask)
        h8 = v8.decode_plan(col8, edge, glob8, code, mask)["h_dec"]
    torch.testing.assert_close(h8, h7, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not RELEASE_V7.is_file(), reason="release V7 weights are committed with the repository")
def test_release_surgery_keeps_the_analytic_packet_at_step_zero():
    from safetensors.torch import load_file

    tensors, report = sg.transplant(load_file(str(RELEASE_V7)), SingleEncV8())
    assert (report["source_width"], report["target_width"], report["layers_used"]) == (160, 128, [0, 1])
    assert 0 < report["params_carried_from_v7"] < report["decoder_params"]
    v8 = _loaded(tensors)
    _, _, col8, glob8, edge, code, mask = _inputs()
    with torch.no_grad():
        plan = v8.decode_plan(col8, edge, glob8, code, mask)
    assert float(plan["h_dec"].abs().sum()) > 0.0
    for key in ("delta_knots", "delta_L", "delta_W", "node_leaves", "bilin_u", "bilin_v", "bilin_w", "flow_s", "flow_t"):
        assert torch.count_nonzero(plan[key]) == 0, key
    torch.testing.assert_close(plan["alpha_table"], torch.full_like(plan["alpha_table"], torch.sigmoid(torch.tensor(-2.0)).item()))
    torch.testing.assert_close(plan["resid_sd_mult"], torch.ones_like(plan["resid_sd_mult"]))

    with_w, rep_w = sg.transplant(load_file(str(RELEASE_V7)), SingleEncV8(), with_delta_w=True)
    assert rep_w["params_carried_from_v7"] == report["params_carried_from_v7"] + with_w["delta_W.weight"].numel() + 8
    with torch.no_grad():
        assert torch.count_nonzero(_loaded(with_w).decode_plan(col8, edge, glob8, code, mask)["delta_W"]) > 0


def test_selection_rules():
    assert sg.even(5, 5) == [0, 1, 2, 3, 4]
    assert sg.even(160, 128)[:6] == [0, 1, 2, 3, 5, 6] and len(set(sg.even(160, 128))) == 128
    assert sg.per_head(160, 128, 4)[32] == 40  # head 1 starts at its own slice
    with pytest.raises(ValueError, match="surgery_target_wider_than_source"):
        sg.even(4, 8)


@pytest.mark.skipif(not RELEASE_V7.is_file(), reason="release V7 weights are committed with the repository")
def test_cli_is_deterministic_and_reports_sources(tmp_path: Path):
    shas = []
    for i in range(2):
        out, rep = tmp_path / f"s{i}.safetensors", tmp_path / f"s{i}.json"
        assert sg.main(["--v7", str(RELEASE_V7), "--out", str(out), "--report", str(rep)]) == 0
        report = json.loads(rep.read_text())
        assert report["out_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        assert report["source_sha256"] == hashlib.sha256(RELEASE_V7.read_bytes()).hexdigest()
        shas.append(report["out_sha256"])
    assert shas[0] == shas[1]
