"""Fail-closed trained-checkpoint SafeTensors & DML3/CSC1 cross-runtime parity verification for DOPE-MLC-v3."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .budget_planner import (
    TIER_BYTE_CAPS,
    TIER_COORD_WIDTHS,
    deserialize_dope_v3_envelope,
    export_v3_checkpoint_to_safetensors,
    serialize_dope_v3_envelope,
)
from .multiclass_head import (
    build_enriched_numpy_basis,
    erf_approx,
    eval_vector_leaf_oblivious_trees,
)
from .multilevel_model import MultiLevelLandmarkModel, _eval_oblivious_trees
from .scaling import FitOnlyTableTransform
from .strata_audit import serialize_dope_latent_wire_bytes

TRAINED_PT_PATH = Path("target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.pt")
TRAINED_ST_PATH = Path("target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.safetensors")
PARITY_FIXTURES_DIR = Path("target/s3-multilevel-training-20261023/v3/rust_parity_fixtures")
PARITY_RECEIPT_PATH = Path("target/s3-multilevel-training-20261023/v3/dope_mlc_v3_rust_parity_receipt.json")
EXPECTED_TRAINED_PT_SHA256 = "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952"
EXPECTED_TRAINED_ST_SHA256 = "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c"
LEVELS = ("small", "medium", "high", "ultra")
TASKS = ("regression", "classification", "multiclass")


def resolve_dope_binary(build_if_missing: bool = False) -> Optional[Path]:
    for cand in (Path("engine/target/debug/dope"), Path("engine/target/release/dope")):
        if cand.is_file():
            return cand
    if build_if_missing and subprocess.run(["cargo", "build", "--manifest-path", "engine/Cargo.toml", "--locked", "--bin", "dope"], capture_output=True).returncode == 0:
        return Path("engine/target/debug/dope") if Path("engine/target/debug/dope").is_file() else None
    return None


def _max_abs_diff_1d(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length_mismatch_{len(a)}_vs_{len(b)}")
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def _max_abs_diff_2d(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    if len(a) != len(b):
        raise ValueError(f"row_length_mismatch_{len(a)}_vs_{len(b)}")
    return max((_max_abs_diff_1d(ra, rb) for ra, rb in zip(a, b)), default=0.0)


def _make_fixed_draws(enc: Dict[str, Any], n_rows: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    reg_w = np.asarray(enc["regime_weights"], dtype=np.float64)
    reg_w = reg_w / np.sum(reg_w)
    p_use, rank = int(enc["p_use"]), int(np.asarray(enc["chol_factor"]).shape[1])
    tb = enc["target_bundle"]
    is_mc = enc["task"] == "multiclass" or tb.get("task_kind") == "multiclass"
    k_cls = int(tb.get("k_classes", 3)) if is_mc else 1
    return {
        "reg_idx": [int(x) for x in rng.choice(len(reg_w), size=n_rows, p=reg_w)],
        "eps_chol": rng.standard_normal((n_rows, rank)).tolist(),
        "eps_diag": rng.standard_normal((n_rows, p_use)).tolist(),
        "eps_target": rng.standard_normal((n_rows, k_cls)).tolist(),
    }


@torch.no_grad()
def python_sample_with_fixed_draws(
    model: MultiLevelLandmarkModel,
    encoding: Dict[str, Any],
    scaler: Optional[FitOnlyTableTransform],
    draws: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic Python reference execution of generate_synthetic + inverse_transform given FixedSampleDraws."""
    p_use, level, task = int(encoding["p_use"]), str(encoding["level"]), str(encoding["task"])
    reg_idx = np.asarray(draws["reg_idx"], dtype=np.int64)
    eps_chol = np.asarray(draws["eps_chol"], dtype=np.float64)
    eps_diag = np.asarray(draws["eps_diag"], dtype=np.float64)
    eps_target = np.asarray(draws["eps_target"], dtype=np.float64)
    n = len(reg_idx)

    reg_m = np.asarray(encoding["regime_means"], dtype=np.float64)
    reg_s = np.asarray(encoding["regime_scales"], dtype=np.float64)
    chol = np.asarray(encoding["chol_factor"], dtype=np.float64)
    diag_std = np.asarray(encoding["diag_res_std"], dtype=np.float64)

    z_syn = reg_m[reg_idx] + reg_s[reg_idx] * (eps_chol @ chol.T + eps_diag * diag_std[None, :])
    z_pad = torch.zeros((n, model.max_cols), dtype=torch.float32)
    z_pad[:, :p_use] = torch.tensor(z_syn, dtype=torch.float32)
    col_mask = torch.zeros(model.max_cols, dtype=torch.float32)
    col_mask[:p_use] = 1.0
    dist_lat = torch.tensor(np.asarray(encoding["dist_latent"], dtype=np.float32))
    z_ref = model.cov_net(z_pad, dist_lat, col_mask)[:, :p_use].detach().cpu().numpy().astype(np.float64)
    z_syn = 0.94 * z_syn + 0.06 * z_ref
    z_syn = (z_syn - np.mean(z_syn, axis=0)) / np.maximum(np.std(z_syn, axis=0), 1e-6)

    u_syn = np.clip(0.5 * (1.0 + erf_approx(z_syn / math.sqrt(2.0))), 0.001, 0.999)
    knot_grid, knots = np.asarray(encoding["knot_grid"], dtype=np.float64), np.asarray(encoding["col_knots"], dtype=np.float64)
    is_bin, bin_p, bin_lh = np.asarray(encoding["is_binary_col"], dtype=bool), np.asarray(encoding["binary_probs"], dtype=np.float64), np.asarray(encoding["binary_low_high"], dtype=np.float64)
    disc_sup = encoding.get("discrete_support", {})

    x_syn = np.zeros((n, p_use), dtype=np.float64)
    for j in range(p_use):
        if is_bin[j]:
            x_syn[:, j] = np.where(u_syn[:, j] >= (1.0 - bin_p[j]), bin_lh[j, 1], bin_lh[j, 0])
        elif j in disc_sup:
            u_vals, cum_p = disc_sup[j]
            idx = np.clip(np.searchsorted(np.asarray(cum_p, dtype=np.float64), u_syn[:, j]), 0, len(u_vals) - 1)
            x_syn[:, j] = np.asarray(u_vals, dtype=np.float64)[idx]
        else:
            x_syn[:, j] = np.interp(u_syn[:, j], knot_grid, knots[j])

    tb = encoding["target_bundle"]
    phi = build_enriched_numpy_basis(
        x_syn, tb.get("gates", {}), tb.get("pairs", []),
        level=level, include_v3_primitives=bool(tb.get("include_v3_primitives", False)),
    )
    if task == "multiclass" or tb.get("task_kind") == "multiclass":
        k_cls = int(tb.get("k_classes", 3))
        priors = np.maximum(np.asarray(tb["class_priors"], dtype=np.float64), 1e-4)
        priors = priors / np.sum(priors)
        u_b, v_b = np.asarray(tb["U_basis"], dtype=np.float64), np.asarray(tb["V_basis"], dtype=np.float64)
        m = min(phi.shape[1], u_b.shape[0])
        scores = priors[None, :] + phi[:, :m] @ (u_b[:m] @ v_b.T) + eval_vector_leaf_oblivious_trees(x_syn, tb.get("trees", []), k_cls)
        noise_amp = float(tb.get("noise_scale", 0.15)) * (0.08 if level in ("low", "small", "medium") else 0.03)
        noisy = scores + eps_target[:, :k_cls] * noise_amp
        y_syn = np.argmax(noisy, axis=1).astype(np.float64)
        if n >= k_cls:
            used_rows = set()
            for c_idx in np.argsort(priors):
                if priors[c_idx] <= 1e-5:
                    continue
                present = np.where(y_syn == c_idx)[0]
                if present.size == 0:
                    for cand in np.argsort(noisy[:, c_idx])[::-1]:
                        if int(cand) not in used_rows:
                            y_syn[int(cand)] = float(c_idx)
                            used_rows.add(int(cand))
                            break
                else:
                    used_rows.add(int(present[0]))
    else:
        w_basis = np.asarray(tb["w_basis"], dtype=np.float64)
        m = min(phi.shape[1], w_basis.size)
        pred = phi[:, :m] @ w_basis[:m]
        x_pad = torch.zeros((n, model.max_cols), dtype=torch.float32)
        x_pad[:, :p_use] = torch.tensor(x_syn, dtype=torch.float32)
        task_lat = torch.tensor(np.asarray(encoding["task_latent"], dtype=np.float32))
        res_net = model.target_net(x_pad, task_lat, col_mask).detach().cpu().numpy().astype(np.float64)
        pred = pred + float(tb["neural_w"]) * (res_net - float(tb["res_net_mean"])) + _eval_oblivious_trees(x_syn, tb["trees"])
        if task in ("classification", "binary"):
            prior = float(np.clip(tb["class_prior"], 0.01, 0.99))
            n_mult = 0.12 if level in ("low", "small", "medium") else 0.06 if level == "high" else 0.03
            score = pred + eps_target[:, 0] * float(tb["noise_scale"]) * n_mult
            y_syn = (score >= float(np.quantile(score, 1.0 - prior))).astype(np.float64)
            if n >= 2 and len(np.unique(y_syn)) < 2:
                y_syn[int(np.argmax(score))], y_syn[int(np.argmin(score))] = 1.0, 0.0
        else:
            n_mult = 0.42 if level in ("low", "small", "medium") else 0.22 if level == "high" else 0.10
            sigma = float(tb["residual_std"]) * float(tb["noise_scale"]) * n_mult
            y_syn = np.clip(pred + eps_target[:, 0] * sigma, float(tb["y_min"]) - 0.5, float(tb["y_max"]) + 0.5)

    x_raw, y_raw = scaler.inverse_transform(x_syn, y_syn) if scaler is not None else (x_syn.copy(), y_syn.copy())
    return {
        "x_scaled": x_syn.tolist(),
        "y_scaled": y_syn.tolist(),
        "x_raw": x_raw.tolist(),
        "y_raw": np.asarray(y_raw, dtype=np.float64).tolist(),
    }


def _build_realistic_raw_dataset(task: str, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = 96
    x_raw = np.column_stack([
        rng.normal(120.0, 25.0, size=n),
        np.exp(rng.normal(2.0, 0.6, size=n)) * 10.0,
        rng.choice([-5.0, 15.0], size=n, p=[0.35, 0.65]),
        rng.choice([10.0, 20.0, 30.0, 40.0], size=n),
        rng.normal(-40.0, 8.0, size=n),
        rng.uniform(100.0, 500.0, size=n),
        rng.normal(0.0, 3.5, size=n),
        rng.normal(50.0, 12.0, size=n),
    ])
    z0 = (x_raw[:, 0] - 120.0) / 25.0
    z1 = (np.log1p(np.maximum(x_raw[:, 1], 0.0)) - 4.3) / 0.6
    z2 = (x_raw[:, 2] > 0.0).astype(np.float64)
    if task == "regression":
        y_raw = 250.0 + 45.0 * z0 - 30.0 * np.tanh(z1) + 25.0 * z2 + rng.normal(0.0, 4.0, size=n)
    elif task in ("classification", "binary"):
        logit = 1.4 * z0 - 1.1 * z1 + 0.9 * (z2 - 0.5)
        y_raw = (logit + rng.normal(0.0, 0.3, size=n) > 0.0).astype(np.float64)
    else:
        logits = np.column_stack([
            1.8 * z0 - 0.8 * z1,
            -1.8 * z0 + 0.8 * z1,
            1.8 * z1 - 0.8 * z0,
            -1.8 * z1 - 0.8 * z0,
        ])
        cls_idx = np.argmax(logits + rng.normal(0.0, 0.2, size=logits.shape), axis=1)
        y_raw = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)[cls_idx]
    cells = [
        {
            "id": i + 1,
            "state": "complete",
            "gain": 0.12 + 0.015 * (i % 7),
            "se": 0.01,
            "stable": True,
            "lambda_indices": [1, 2],
            "top_features": [{"col": 0, "weight": 0.4}, {"col": 1, "weight": 0.3}, {"col": 2, "weight": 0.2}],
        }
        for i in range(23)
    ]
    return x_raw, y_raw, {"cells": cells, "features": 8}


def run_rust_v3_parity_verification(
    *,
    dope_bin: Path,
    pt_path: Path,
    st_path: Path,
    fixtures_dir: Path,
    receipt_path: Optional[Path] = None,
    require_trained_hashes: bool = True,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Execute fail-closed Rust vs PyTorch parity across all 4 tiers and 3 tasks and write durable fixtures/receipt."""
    if not dope_bin.is_file():
        raise RuntimeError(f"missing_required_dope_binary:{dope_bin}")
    if not pt_path.is_file():
        raise RuntimeError(f"missing_required_v3_pt_checkpoint:{pt_path}")

    export_info = export_v3_checkpoint_to_safetensors(pt_path, st_path)
    if require_trained_hashes and (
        export_info["source_pt_sha256"] != EXPECTED_TRAINED_PT_SHA256
        or export_info["safetensors_sha256"] != EXPECTED_TRAINED_ST_SHA256
    ):
        raise AssertionError(f"unexpected_trained_sha256:{export_info}")

    inspect_w = json.loads(
        subprocess.check_output([str(dope_bin), "mlc-v3", "inspect-weights", "--weights", str(st_path)], text=True)
    )
    if (
        inspect_w["tensor_count"] != 56
        or inspect_w["parameter_count"] != 444168
        or inspect_w["sha256"] != export_info["safetensors_sha256"]
        or inspect_w["byte_len"] != export_info["safetensors_bytes"]
    ):
        raise AssertionError(f"weight_inventory_mismatch:{inspect_w}")

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = MultiLevelLandmarkModel().eval()
    model.load_state_dict(state_dict, strict=True)

    fixtures_dir.mkdir(parents=True, exist_ok=True)
    x_raw_probe, y_raw_probe, rep_probe = _build_realistic_raw_dataset("regression", seed=20261024)
    sc_probe = FitOnlyTableTransform.fit(x_raw_probe, y_raw_probe, task="regression")
    x_tr_probe, y_tr_probe = sc_probe.transform(x_raw_probe, y_raw_probe)
    p_use, n_batch = x_tr_probe.shape[1], 16
    x_pad_np = np.zeros((n_batch, 64), dtype=np.float32)
    x_pad_np[:, :p_use] = x_tr_probe[:n_batch].astype(np.float32)
    x_pad_t = torch.tensor(x_pad_np, dtype=torch.float32)
    col_mask_t = torch.zeros(64, dtype=torch.float32)
    col_mask_t[:p_use] = 1.0

    forward_input_path = fixtures_dir / "forward_check_input.json"
    forward_errors: Dict[str, Dict[str, float]] = {}
    forward_refs: Dict[str, Any] = {}

    for lvl in LEVELS:
        vec256_t, _, tab_d = model.build_conditioning_vector(rep_probe, x_tr_probe, y_tr_probe, level=lvl, device=torch.device("cpu"))
        col_lm_np = np.asarray(tab_d["col_landmarks24"][:p_use, :17], dtype=np.float32)
        col_lm_t = torch.tensor(col_lm_np, dtype=torch.float32)
        fwd_lvl_path = fixtures_dir / f"forward_input_{lvl}.json"
        fwd_lvl_path.write_text(
            json.dumps({"p_use": int(p_use), "x_pad": x_pad_np.tolist(), "col_landmarks": col_lm_np.tolist()}),
            encoding="utf-8",
        )
        cond_path = fixtures_dir / f"cond_{lvl}.json"
        cond_path.write_text(json.dumps(vec256_t.tolist()), encoding="utf-8")

        with torch.no_grad():
            py_r = model.router(vec256_t, level=lvl)
            py_cg_lm = model.target_net.compute_column_gates(py_r["task_latent"], col_lm_t, X_pad=x_pad_t).tolist()
            py_cg_xp = model.target_net.compute_column_gates(py_r["task_latent"], None, X_pad=x_pad_t).tolist()
            py_tn_lm = model.target_net(x_pad_t, py_r["task_latent"], col_mask_t, col_landmarks=col_lm_t).tolist()
            py_tn_xp = model.target_net(x_pad_t, py_r["task_latent"], col_mask_t, col_landmarks=None).tolist()
            py_cov = model.cov_net(x_pad_t, py_r["dist_latent"], col_mask_t).tolist()

        rs_out = json.loads(
            subprocess.check_output(
                [str(dope_bin), "mlc-v3", "route", "--weights", str(st_path), "--level", lvl, "--conditioning-json", str(cond_path), "--forward-input-json", str(fwd_lvl_path)],
                text=True,
            )
        )
        py_gates = {k: float(v.item()) for k, v in py_r["gates"].items()}
        errs = {
            "gates_max_abs_err": max(abs(py_gates[k] - float(rs_out["gates"][k])) for k in py_gates),
            "task_latent_max_abs_err": _max_abs_diff_1d(py_r["task_latent"].tolist(), rs_out["task_latent"]),
            "dist_latent_max_abs_err": _max_abs_diff_1d(py_r["dist_latent"].tolist(), rs_out["dist_latent"]),
            "landmark_pred_max_abs_err": _max_abs_diff_1d(py_r["landmark_pred"].tolist(), rs_out["landmark_pred"]),
            "bb_pred_max_abs_err": _max_abs_diff_1d(py_r["bb_pred"].tolist(), rs_out["bb_pred"]),
            "col_gates_landmarks_max_abs_err": _max_abs_diff_1d(py_cg_lm, rs_out["col_gates_from_landmarks"]),
            "col_gates_xpad_max_abs_err": _max_abs_diff_1d(py_cg_xp, rs_out["col_gates_from_xpad"]),
            "target_net_with_landmarks_max_abs_err": _max_abs_diff_1d(py_tn_lm, rs_out["target_net_out_with_landmarks"]),
            "target_net_from_xpad_max_abs_err": _max_abs_diff_1d(py_tn_xp, rs_out["target_net_out_from_xpad"]),
            "cov_net_max_abs_err": _max_abs_diff_2d(py_cov, rs_out["cov_net_out"]),
        }
        for metric_name, err_val in errs.items():
            if not math.isfinite(err_val) or err_val >= tolerance:
                raise AssertionError(f"forward_parity_exceeded:{lvl}:{metric_name}={err_val}")
        forward_errors[lvl] = errs
        forward_refs[lvl] = {
            "conditioning_vec256": vec256_t.tolist(),
            "forward_input": {"p_use": int(p_use), "x_pad": x_pad_np.tolist(), "col_landmarks": col_lm_np.tolist()},
            "expected": rs_out,
        }

    packet_cells: Dict[str, Dict[str, Any]] = {}
    n_sample_rows = 36
    for idx_t, task in enumerate(TASKS):
        x_raw, y_raw, rep = _build_realistic_raw_dataset(task, seed=3000 + idx_t)
        sc = FitOnlyTableTransform.fit(x_raw, y_raw, task=task)
        x_tr, y_tr = sc.transform(x_raw, y_raw)
        for idx_l, lvl in enumerate(LEVELS):
            cell_key = f"{lvl}_{task}"
            enc = model.encode_table(x_tr, y_tr, rep, task=task, level=lvl, adaptive_budget=True, include_v3_primitives=True)
            wire_bytes = serialize_dope_v3_envelope(enc, scaler=sc, compress=True)
            if len(wire_bytes) > TIER_BYTE_CAPS[lvl]:
                raise AssertionError(f"wire_exceeds_cap:{cell_key}:{len(wire_bytes)}")
            wire_path = fixtures_dir / f"packet_{cell_key}.dml3"
            wire_path.write_bytes(wire_bytes)

            wire_info = json.loads(subprocess.check_output([str(dope_bin), "mlc-v3", "inspect-wire", "--wire", str(wire_path)], text=True))
            if (
                wire_info["wire_format"] != "DML3"
                or not wire_info["within_tier_cap"]
                or not wire_info["has_csc1_scaler"]
                or wire_info["level"] != lvl
                or wire_info["task"] != task
                or wire_info["code_width"] != TIER_COORD_WIDTHS[lvl]
            ):
                raise AssertionError(f"wire_inspect_mismatch:{cell_key}:{wire_info}")

            dec_enc, dec_sc = deserialize_dope_v3_envelope(wire_bytes)
            draws = _make_fixed_draws(dec_enc, n_rows=n_sample_rows, seed=4000 + idx_t * 10 + idx_l)
            draws_path = fixtures_dir / f"draws_{cell_key}.json"
            draws_path.write_text(json.dumps(draws), encoding="utf-8")

            py_fixed = python_sample_with_fixed_draws(model, dec_enc, dec_sc, draws)
            rs_json_path, rs_csv_path = fixtures_dir / f"rs_fixed_{cell_key}.json", fixtures_dir / f"rs_stat_{cell_key}.csv"
            subprocess.check_call([
                str(dope_bin), "mlc-v3", "sample", "--weights", str(st_path), "--wire", str(wire_path),
                "--rows", str(n_sample_rows), "--draws-json", str(draws_path), "--json-out", str(rs_json_path),
                "--out", str(fixtures_dir / f"rs_fixed_{cell_key}.csv"),
            ])
            rs_fixed = json.loads(rs_json_path.read_text(encoding="utf-8"))
            x_sc_err = _max_abs_diff_2d(py_fixed["x_scaled"], rs_fixed["x_scaled"])
            y_sc_err = _max_abs_diff_1d(py_fixed["y_scaled"], rs_fixed["y_scaled"])
            x_raw_err = _max_abs_diff_2d(py_fixed["x_raw"], rs_fixed["x_raw"])
            y_raw_err = _max_abs_diff_1d(py_fixed["y_raw"], rs_fixed["y_raw"])
            for m_name, err_v in (("x_scaled", x_sc_err), ("y_scaled", y_sc_err), ("x_raw", x_raw_err), ("y_raw", y_raw_err)):
                if not math.isfinite(err_v) or err_v >= tolerance:
                    raise AssertionError(f"fixed_draw_sample_parity_exceeded:{cell_key}:{m_name}={err_v}")

            subprocess.check_call([
                str(dope_bin), "mlc-v3", "sample", "--weights", str(st_path), "--wire", str(wire_path),
                "--rows", str(n_sample_rows), "--seed", str(20261023 + idx_t * 7 + idx_l), "--out", str(rs_csv_path),
            ])
            with rs_csv_path.open("r", encoding="utf-8", newline="") as f:
                rows_csv = list(csv.reader(f))
            if rows_csv[0] != [f"f_{j:03d}" for j in range(p_use)] + ["target"] or len(rows_csv) != n_sample_rows + 1:
                raise AssertionError(f"csv_schema_mismatch:{cell_key}")
            mat_raw = np.array([[float(v) for v in r] for r in rows_csv[1:]], dtype=np.float64)
            if not np.all(np.isfinite(mat_raw)):
                raise AssertionError(f"nonfinite_sampled_values:{cell_key}")
            col0_mean = float(np.mean(mat_raw[:, 0]))
            if not (60.0 <= col0_mean <= 180.0):
                raise AssertionError(f"raw_domain_feature0_not_restored:{cell_key}:{col0_mean}")
            y_col = mat_raw[:, -1]
            if task == "classification" and sorted(set(np.round(y_col, 4).tolist())) != [0.0, 1.0]:
                raise AssertionError(f"binary_label_support_mismatch:{cell_key}")
            if task == "multiclass" and sorted(set(np.round(y_col, 4).tolist())) != [10.0, 20.0, 30.0, 40.0]:
                raise AssertionError(f"multiclass_raw_label_domain_mismatch:{cell_key}")
            if task == "regression" and not (100.0 <= float(np.mean(y_col)) <= 400.0):
                raise AssertionError(f"regression_raw_target_not_restored:{cell_key}")

            packet_cells[cell_key] = {
                "wire_bytes": len(wire_bytes),
                "tier_cap_bytes": TIER_BYTE_CAPS[lvl],
                "crc32": wire_info["crc32"],
                "fixed_draw_x_scaled_max_abs_err": x_sc_err,
                "fixed_draw_y_scaled_max_abs_err": y_sc_err,
                "fixed_draw_x_raw_max_abs_err": x_raw_err,
                "fixed_draw_y_raw_max_abs_err": y_raw_err,
                "statistical_raw_col0_mean": round(col0_mean, 4),
            }

    enc_dml2 = model.encode_table(x_tr_probe, y_tr_probe, rep_probe, task="regression", level="medium")
    for ver in (1, 2):
        dml2_path = fixtures_dir / f"compat_v{ver}.dml2"
        dml2_path.write_bytes(serialize_dope_latent_wire_bytes(enc_dml2, version=ver))
        dml2_info = json.loads(subprocess.check_output([str(dope_bin), "mlc-v3", "inspect-wire", "--wire", str(dml2_path)], text=True))
        if dml2_info["wire_format"] != "DML2" or dml2_info["level"] != "medium":
            raise AssertionError(f"dml2_v{ver}_compat_failed:{dml2_info}")

    valid_wire = (fixtures_dir / "packet_small_regression.dml3").read_bytes()
    corrupt_zlib_body = b"\x78\x9c\xff\xff\xff"
    corrupt_zlib_crc = zlib.crc32(corrupt_zlib_body) & 0xFFFFFFFF
    tampered_cases = {
        "bad_magic": b"DML9" + valid_wire[4:],
        "bad_version": b"DML3" + struct.pack("<BBI", 4, valid_wire[5], struct.unpack_from("<I", valid_wire, 6)[0]) + valid_wire[10:],
        "bad_comp_flag": b"DML3" + struct.pack("<BBI", 3, 3, struct.unpack_from("<I", valid_wire, 6)[0]) + valid_wire[10:],
        "crc_mismatch": valid_wire[:-1] + bytes([valid_wire[-1] ^ 0x55]),
        "truncated_header": valid_wire[:7],
        "corrupt_zlib_stream": b"DML3" + struct.pack("<BBI", 3, 1, corrupt_zlib_crc) + corrupt_zlib_body,
        "oversized_envelope": b"DML3" + struct.pack("<BBI", 3, 0, 0) + (b"\x00" * 270000),
    }
    tamper_results: Dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, bad_bytes in tampered_cases.items():
            p_bad = tmp / f"{name}.dml3"
            p_bad.write_bytes(bad_bytes)
            if subprocess.run([str(dope_bin), "mlc-v3", "inspect-wire", "--wire", str(p_bad)], capture_output=True).returncode == 0:
                raise AssertionError(f"tampered_wire_did_not_fail_closed:{name}")
            tamper_results[name] = True
        bad_st = tmp / "bad.safetensors"
        bad_st.write_bytes(b"not_a_safetensors_file")
        if subprocess.run([str(dope_bin), "mlc-v3", "inspect-weights", "--weights", str(bad_st)], capture_output=True).returncode == 0:
            raise AssertionError("tampered_safetensors_did_not_fail_closed")
        tamper_results["corrupt_safetensors"] = True

    ref_bundle = {"forward_by_level": forward_refs}
    (fixtures_dir / "reference_bundle.json").write_text(json.dumps(ref_bundle, indent=2), encoding="utf-8")

    max_forward_err = max(max(v.values()) for v in forward_errors.values())
    max_sample_err = max(
        max(c["fixed_draw_x_scaled_max_abs_err"], c["fixed_draw_y_scaled_max_abs_err"], c["fixed_draw_x_raw_max_abs_err"], c["fixed_draw_y_raw_max_abs_err"])
        for c in packet_cells.values()
    )
    receipt = {
        "schema": "dope.mlc_v3.rust_safetensors_parity_receipt.v1",
        "dope_binary": str(dope_bin),
        "dope_binary_sha256": hashlib.sha256(dope_bin.read_bytes()).hexdigest(),
        "weights_binding": export_info,
        "tolerance": tolerance,
        "max_observed_forward_abs_err": max_forward_err,
        "max_observed_fixed_draw_sample_abs_err": max_sample_err,
        "all_checks_under_tolerance": bool(max_forward_err < tolerance and max_sample_err < tolerance),
        "levels_verified": list(LEVELS),
        "tasks_verified": list(TASKS),
        "forward_parity_by_level": forward_errors,
        "dml3_csc1_cells": packet_cells,
        "dml2_backward_compat_versions_verified": [1, 2],
        "fail_closed_tamper_checks": tamper_results,
        "qualification_scope_note": (
            "Proves < 1e-5 numerical parity between the 100% pure-Rust dope-core/dope-cli SafeTensors engine "
            "and the PyTorch MultiLevelLandmarkModel reference across all 4 tiers (small, medium, high, ultra) "
            "and 3 task families (regression, binary, multiclass) on router, column-gate, target-net, cov-net, "
            "and fixed-draw DML3/CSC1 decoding and sampling. The 1,000-dataset DOPE-MLC-v2 and 1,069-dataset "
            "(4,276-cell) DOPE-MLC-v3 downstream auditor Elo benchmarks were measured in the Python evaluation pipeline."
        ),
    }
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    bin_path = resolve_dope_binary()
    if bin_path is None:
        raise SystemExit("FAIL: dope binary not built at engine/target/{debug,release}/dope")
    rec = run_rust_v3_parity_verification(bin_path, TRAINED_PT_PATH, TRAINED_ST_PATH, PARITY_FIXTURES_DIR, PARITY_RECEIPT_PATH, True)
    print(json.dumps({"status": "PASS", "max_observed_forward_abs_err": rec["max_observed_forward_abs_err"], "max_observed_fixed_draw_sample_abs_err": rec["max_observed_fixed_draw_sample_abs_err"], "receipt_path": str(PARITY_RECEIPT_PATH)}, indent=2))
