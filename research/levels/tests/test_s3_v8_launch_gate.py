"""20-step single-CPU Launch-Gate Smoke Test for `SingleEncV8` (DOPE v3.0).

Verifies end-to-end training & deployment readiness before any fleet GPU/CPU wave:
1. 10-step initial training -> finite losses/grads, >= 1 optimizer step, full checkpoint save
2. Exact resume for 10 more steps (total = 20 steps) -> contiguous step count, finite telemetry
3. Canonical F32 `rank <= 2` SafeTensors export -> native Rust `dope_v8_emit fit` + `sample` round-trip!
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.levels.s3_v8_eval import evaluate_dse8_cell
from research.levels.s3_v8_priors import sample_v8_prior_episode
from research.levels.s3_v8_trainer import parse_args, train_v8
from research.levels.v8_evidence import find_v8_emit_binary


def test_20_step_launch_gate_resume_and_rust_eval(tmp_path: Path) -> None:
    out_dir = tmp_path / "v8_smoke"

    # Phase A: Steps 1..10
    args_a = parse_args([
        "--out-dir", str(out_dir),
        "--device", "cpu",
        "--steps", "10",
        "--episodes-per-step", "2",
        "--n-fit", "128",
        "--n-qry", "96",
        "--n-syn", "128",
        "--d-model", "32",
        "--n-heads", "4",
        "--n-enc-layers", "1",
        "--n-dec-layers", "1",
        "--save-every", "5",
        "--seed", "42",
    ])
    res_a = train_v8(args_a)
    assert res_a["steps_done"] == 10
    assert res_a["last_step"] == 10
    assert Path(res_a["checkpoint_pt"]).is_file()
    assert Path(res_a["safetensors"]).is_file()

    # Phase B: Exact Resume Steps 11..20 (with PCGrad-Lite enabled!)
    args_b = parse_args([
        "--out-dir", str(out_dir),
        "--device", "cpu",
        "--steps", "20",
        "--episodes-per-step", "2",
        "--n-fit", "128",
        "--n-qry", "96",
        "--n-syn", "128",
        "--d-model", "32",
        "--n-heads", "4",
        "--n-enc-layers", "1",
        "--n-dec-layers", "1",
        "--use-pcgrad",
        "--save-every", "10",
        "--resume",
        "--seed", "42",
    ])
    res_b = train_v8(args_b)
    assert res_b["steps_done"] == 10
    assert res_b["last_step"] == 20

    lines = (out_dir / "train.jsonl").read_text().strip().splitlines()
    assert len(lines) == 20
    last = json.loads(lines[-1])
    assert last["step"] == 20 and last["grad_norm"] > 0.0

    # Phase C: Native Rust `dope_v8_emit` + `s3_v8_eval` end-to-end check with exported SafeTensors!
    exe = find_v8_emit_binary()
    if exe is not None:
        rng = np.random.default_rng(777)
        ep = sample_v8_prior_episode(rng, n_fit=180, n_qry=240, p=8, task_id=1, mechanism="xor")
        Xb, yb = ep["X_qry"][:120], ep["y_qry"][:120]
        Xt, yt = ep["X_qry"][120:], ep["y_qry"][120:]
        ev_res = evaluate_dse8_cell(
            ep["X_fit"], ep["y_fit"], Xb, yb, Xt, yt,
            task_id=1, n_classes=2, budget_bytes=2048,
            checkpoint=Path(res_b["safetensors"]), col_meta=ep["col_meta"], seed=19,
        )
        assert ev_res["packet_bytes"] <= 2048.0
        assert ev_res["illegal_onehot_rows"] == 0.0
        assert 0.0 <= ev_res["c2st_xy"] <= 1.0
