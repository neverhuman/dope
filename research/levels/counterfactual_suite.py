"""Deterministic 7-case counterfactual sensitivity suite (Spec 3 section 6.4).

Exercises FIT-only preparation, deterministic landmark probes, ``DopeEncoder629``,
and ``TransferReadout`` across seven matched table pairs where naive summary
statistics collapse while decision-relevant structure changes.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .building_block_landmarks import build_v3_conditioning_bundle
from .encoder import DOPE_FEATURES_629_DIM, DopeEncoder629
from .multilevel_model import MAX_COLS
from .observed_conditioning import bundle_to_report
from .optional_landmarks import LandmarkBundle
from .transfer_encoder import COLUMN_WIDTH, candidate, encode, realize

SCHEMA = "dope-counterfactual-suite/v1"
CASE_NAMES = (
    "interaction_vs_shuffled_target",
    "linear_vs_xor_product",
    "homoscedastic_vs_rare_heteroscedastic",
    "smooth_trend_vs_high_freq_periodic",
    "task_budget_and_metric_shift",
    "informative_vs_ignorable_missingness",
    "wide_columns_beyond_64_and_reordering",
)


class CounterfactualSuiteError(ValueError):
    """Static refusal code without source values."""

    purpose = "evaluate deterministic Spec 3 section 6.4 counterfactual sensitivity pairs"
    reason = "an unknown counterfactual case name, invalid row count, or degenerate target was supplied"
    common_fixes = "select a registered CASE_NAMES entry with integer seed and rows >= 64"
    docs_url = "docs/testing.md#local-repair"
    repair_hint = "supply valid case parameters and rerun test_counterfactual_suite"


def _standardize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = arr - float(np.mean(arr))
    scale = float(np.std(centered))
    if scale <= 1e-12:
        raise CounterfactualSuiteError("counterfactual_degenerate_target")
    return centered / scale


def _prepare_nc1_fit(x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x_raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 16 or arr.shape[1] < 1:
        raise CounterfactualSuiteError("counterfactual_shape")
    miss = ~np.isfinite(arr)
    filled = np.zeros_like(arr, dtype=np.float64)
    z_cols = np.zeros_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        obs = ~miss[:, j]
        mu = float(np.mean(arr[obs, j])) if np.any(obs) else 0.0
        col = np.where(obs, arr[:, j], mu)
        filled[:, j] = col
        sd = float(np.std(col))
        if sd > 1e-9:
            z_cols[:, j] = np.clip((col - float(np.mean(col))) / sd, -8.0, 8.0)
    augmented = (
        np.column_stack([filled, miss[:, np.any(miss, axis=0)].astype(np.float64)])
        if np.any(miss)
        else filled
    )
    return augmented, z_cols, miss


def _abs_corr(col: np.ndarray, y_z: np.ndarray) -> float:
    c = np.asarray(col, dtype=np.float64).reshape(-1)
    sd = float(np.std(c))
    if sd <= 1e-9:
        return 0.0
    val = abs(float(np.mean(((c - float(np.mean(c))) / sd) * y_z)))
    return val if math.isfinite(val) else 0.0


def _probe_report(x_aug: np.ndarray, z_cols: np.ndarray, miss: np.ndarray, y_z: np.ndarray) -> dict[str, object]:
    n, p = z_cols.shape
    lin = np.array([_abs_corr(z_cols[:, j], y_z) for j in range(p)])
    sat = np.array([_abs_corr(np.tanh(1.5 * z_cols[:, j]), y_z) for j in range(p)])
    absv = np.array([_abs_corr(np.abs(z_cols[:, j]), y_z) for j in range(p)])
    per = np.array(
        [max(_abs_corr(np.sin(12.0 * z_cols[:, j]), y_z), _abs_corr(np.cos(12.0 * z_cols[:, j]), y_z)) for j in range(p)]
    )
    step = np.array([_abs_corr((z_cols[:, j] > 0.0).astype(np.float64), y_z) for j in range(p)])
    hinge = np.array([_abs_corr(np.maximum(0.0, z_cols[:, j]), y_z) for j in range(p)])
    miss_c = (
        np.array([_abs_corr(miss[:, j].astype(np.float64), y_z) for j in range(p)])
        if np.any(miss)
        else np.zeros(p)
    )
    top_cols = np.argsort(lin + per + absv)[::-1][: min(p, 12)]
    cand = np.unique(np.concatenate([top_cols, np.arange(max(0, p - 8), p, dtype=np.int64)]))
    pair_best = 0.0
    for i_idx in range(cand.size):
        for j_idx in range(i_idx + 1, cand.size):
            pair_best = max(pair_best, _abs_corr(z_cols[:, int(cand[i_idx])] * z_cols[:, int(cand[j_idx])], y_z))
    best_lin = float(np.max(lin))
    slots = {
        0: best_lin ** 2,
        4: float(np.max(sat)) ** 2,
        5: float(np.max(absv)) ** 2,
        6: float(np.max(hinge)) ** 2,
        7: float(np.max(step)) ** 2,
        8: float(np.max(per)) ** 2,
        11: pair_best ** 2,
        13: float(np.max(miss_c)) ** 2,
        14: max(best_lin, float(np.max(sat))) ** 2,
        16: max(pair_best, best_lin, float(np.max(per))) ** 2,
    }
    gains: list[float | None] = [None] * 23
    for slot_idx, gain in slots.items():
        gains[slot_idx] = max(0.0, min(0.999, float(gain)))
    bundle = LandmarkBundle.from_raw_gains(gains, subsample_n=n)
    return bundle_to_report(bundle, rows=n, features=x_aug.shape[1], task="regression", classes=0)


def _column_tokens_128(z_cols: np.ndarray, miss: np.ndarray, y_z: np.ndarray) -> torch.Tensor:
    n, p = z_cols.shape
    tokens = np.zeros((p, COLUMN_WIDTH), dtype=np.float32)
    y_rank = np.argsort(np.argsort(y_z)).astype(np.float64)
    y_rank_z = _standardize_1d(y_rank)
    for j in range(p):
        zj = z_cols[:, j]
        prods = zj[:, None] * z_cols
        prods_c = prods - np.mean(prods, axis=0, keepdims=True)
        prods_s = np.std(prods_c, axis=0)
        valid = prods_s > 1e-9
        valid[j] = False
        pair_max = (
            float(np.max(np.abs(np.mean((prods_c[:, valid] / prods_s[valid]) * y_z[:, None], axis=0))))
            if np.any(valid)
            else 0.0
        )
        rj = _standardize_1d(np.argsort(np.argsort(zj)).astype(np.float64)) if np.std(zj) > 1e-9 else np.zeros(n)
        xi = max(0.0, 1.0 - 3.0 * float(np.mean(np.abs(np.diff(y_rank[np.argsort(zj)])))) / float(n + 1))
        q05, q25, q50, q75, q95 = np.quantile(zj, (0.05, 0.25, 0.50, 0.75, 0.95))
        base = np.array(
            [
                _abs_corr(zj, y_z),
                abs(float(np.mean(rj * y_rank_z))),
                _abs_corr(zj ** 2 - 1.0, y_z),
                _abs_corr(np.tanh(1.5 * zj), y_z),
                _abs_corr(np.sin(12.0 * zj), y_z),
                _abs_corr(np.cos(12.0 * zj), y_z),
                xi,
                pair_max,
                float(np.mean(miss[:, j])),
                _abs_corr(miss[:, j].astype(np.float64), y_z),
                float(np.mean(zj ** 3)) / 6.0,
                float(np.clip(np.mean(zj ** 4) - 3.0, -3.0, 20.0)) / 10.0,
                float(q05),
                float(q25),
                float(q50),
                float(q75),
            ],
            dtype=np.float32,
        )
        tokens[j, :16] = base
        tokens[j, 16:32] = np.tanh(2.0 * base)
        tokens[j, 32:48] = base ** 2
        tokens[j, 48:64] = np.sin(math.pi * base)
        tokens[j, 64:96] = np.concatenate([base, np.cos(math.pi * base)]) * float(q95 - q05) * 0.25
        tokens[j, 96:128] = np.concatenate([np.abs(base), np.arcsinh(base)])
    return torch.from_numpy(tokens).unsqueeze(0)


def _distances(vec_a: torch.Tensor, vec_b: torch.Tensor) -> tuple[float, float]:
    va = vec_a.detach().reshape(-1).to(torch.float64)
    vb = vec_b.detach().reshape(-1).to(torch.float64)
    l2 = float(torch.linalg.vector_norm(va - vb).item())
    denom = float((torch.linalg.vector_norm(va) * torch.linalg.vector_norm(vb)).item())
    cos_dist = 0.0 if denom <= 1e-12 else float(1.0 - torch.dot(va, vb).item() / denom)
    return l2, max(0.0, cos_dist)


def _pair_metrics(f_a: torch.Tensor, g_a: torch.Tensor, f_b: torch.Tensor, g_b: torch.Tensor) -> dict[str, float]:
    l2_629, cos_629 = _distances(f_a, f_b)
    l2_512, cos_512 = _distances(g_a, g_b)
    return {
        "full629_l2": l2_629,
        "full629_cosine_distance": cos_629,
        "transfer512_l2": l2_512,
        "transfer512_cosine_distance": cos_512,
    }


class _SuiteHarness:
    def __init__(self) -> None:
        with torch.random.fork_rng(devices=()):
            torch.manual_seed(6400)
            self.encoder629 = DopeEncoder629().eval()
            for param in self.encoder629.parameters():
                param.requires_grad_(False)
            self.readout_a = realize("A", column_tokens=False).eval()
            self.readout_wide = realize("A", column_tokens=True).eval()

    def encode_table(self, x_raw: np.ndarray, y_raw: np.ndarray, *, task_id: str = "rank") -> tuple[torch.Tensor, torch.Tensor]:
        x_aug, z_cols, miss = _prepare_nc1_fit(x_raw)
        y_z = _standardize_1d(y_raw)
        report = _probe_report(x_aug, z_cols, miss, y_z)
        cond = build_v3_conditioning_bundle(report, x_aug, y_z, level="high", building_blocks=True)
        vec256 = torch.from_numpy(cond["vec256"]).unsqueeze(0).to(torch.float32)
        col_lm = np.zeros((1, MAX_COLS, 17), dtype=np.float32)
        raw_lm = cond["col_landmarks24"]
        take = min(MAX_COLS, raw_lm.shape[0])
        col_lm[0, :take, :] = raw_lm[:take, :17]
        with torch.no_grad():
            full629 = self.encoder629(vec256, torch.from_numpy(col_lm))
            pkt = encode(self.readout_a, "A", full629, task_id=task_id, claimed_identity=candidate("A").identity)
        return full629, pkt["global"]

    def encode_wide(self, x_raw: np.ndarray, y_raw: np.ndarray, *, pad_unobserved: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        _, truncated64 = self.encode_table(x_raw[:, :MAX_COLS], y_raw, task_id="rank")
        _, z_cols, miss = _prepare_nc1_fit(x_raw)
        y_z = _standardize_1d(y_raw)
        tokens = _column_tokens_128(z_cols, miss, y_z)
        p = z_cols.shape[1]
        observed = torch.ones((1, p), dtype=torch.bool)
        if pad_unobserved > 0:
            tokens = torch.cat([tokens, torch.full((1, pad_unobserved, COLUMN_WIDTH), 9.0, dtype=torch.float32)], dim=1)
            observed = torch.cat([observed, torch.zeros((1, pad_unobserved), dtype=torch.bool)], dim=1)
        global_in = torch.zeros((1, DOPE_FEATURES_629_DIM), dtype=torch.float32)
        obs_tokens = tokens[0, :p]
        global_in[0, :COLUMN_WIDTH] = obs_tokens.mean(dim=0)
        global_in[0, COLUMN_WIDTH : 2 * COLUMN_WIDTH] = 4.0 * obs_tokens.max(dim=0).values
        col_ids = tuple(f"col_{idx}" for idx in range(tokens.shape[1]))
        with torch.no_grad():
            pkt = encode(
                self.readout_wide,
                "A",
                global_in,
                task_id="rank",
                claimed_identity=candidate("A").identity,
                column_ids=col_ids,
                column_tokens=tokens,
                observed=observed,
                table_rows=int(x_raw.shape[0]),
            )
        return truncated64, pkt["global"]


def evaluate_counterfactual_case(case_name: str, *, seed: int = 20261120, rows: int = 384) -> dict[str, object]:
    """Generate and evaluate one Spec 3 section 6.4 counterfactual pair."""
    if case_name not in CASE_NAMES:
        raise CounterfactualSuiteError("counterfactual_unknown_case")
    if type(seed) is not int or type(rows) is not int or rows < 64:
        raise CounterfactualSuiteError("counterfactual_shape")
    rng = np.random.default_rng(seed + CASE_NAMES.index(case_name) * 97)
    harness = _SuiteHarness()
    base = {"schema": SCHEMA, "case": case_name}

    if case_name == "interaction_vs_shuffled_target":
        x = rng.standard_normal((rows, 8))
        y_a = _standardize_1d(x[:, 0] * x[:, 1])
        y_b = y_a[rng.permutation(rows)]
        metrics = _pair_metrics(*harness.encode_table(x, y_a), *harness.encode_table(x.copy(), y_b))
        return {**base, "x_max_abs_diff": 0.0, "y_sorted_max_abs_diff": float(np.max(np.abs(np.sort(y_a) - np.sort(y_b)))), **metrics}

    if case_name == "linear_vs_xor_product":
        x = rng.standard_normal((rows, 8))
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        y_a = _standardize_1d(0.85 * x[:, 0] + 0.52 * rng.standard_normal(rows))
        raw_xor = x[:, 0] * x[:, 1]
        design = np.column_stack([np.ones(rows), x])
        y_b = _standardize_1d(raw_xor - design @ np.linalg.lstsq(design, raw_xor, rcond=None)[0])
        metrics = _pair_metrics(*harness.encode_table(x, y_a), *harness.encode_table(x, y_b))
        return {
            **base,
            "arm_a_max_univariate_corr": max(_abs_corr(x[:, j], y_a) for j in range(8)),
            "arm_b_max_univariate_corr": max(_abs_corr(x[:, j], y_b) for j in range(8)),
            "arm_b_product_corr": _abs_corr(x[:, 0] * x[:, 1], y_b),
            **metrics,
        }

    if case_name == "homoscedastic_vs_rare_heteroscedastic":
        x = rng.standard_normal((rows, 8))
        eps = rng.standard_normal(rows)
        y_a = _standardize_1d(1.1 * x[:, 0] + 0.65 * eps)
        sigma = np.where(x[:, 1] > np.quantile(x[:, 1], 0.90), 2.05, 0.10)
        y_b = _standardize_1d(1.1 * x[:, 0] + sigma * eps)
        metrics = _pair_metrics(*harness.encode_table(x, y_a), *harness.encode_table(x, y_b))
        return {**base, "mean_abs_diff": abs(float(np.mean(y_a) - np.mean(y_b))), "std_abs_diff": abs(float(np.std(y_a) - np.std(y_b))), **metrics}

    if case_name == "smooth_trend_vs_high_freq_periodic":
        x = rng.standard_normal((rows, 8))
        y_a, y_b = _standardize_1d(np.tanh(1.5 * x[:, 0])), _standardize_1d(np.sin(12.0 * x[:, 0]))
        metrics = _pair_metrics(*harness.encode_table(x, y_a), *harness.encode_table(x, y_b))
        return {**base, "var_abs_diff": abs(float(np.var(y_a) - np.var(y_b))), **metrics}

    if case_name == "task_budget_and_metric_shift":
        x = rng.standard_normal((rows, 8))
        y = _standardize_1d(0.9 * x[:, 0] - 0.5 * x[:, 1] ** 2)
        metrics = _pair_metrics(*harness.encode_table(x, y, task_id="rank"), *harness.encode_table(x, y, task_id="calibrate"))
        return {**base, **metrics}

    if case_name == "informative_vs_ignorable_missingness":
        x_base = rng.standard_normal((rows, 8))
        k_miss = rows // 4
        miss_inf, miss_mcar = np.zeros(rows, dtype=bool), np.zeros(rows, dtype=bool)
        miss_inf[np.argsort(x_base[:, 0] + 0.35 * rng.standard_normal(rows))[-k_miss:]] = True
        miss_mcar[rng.choice(rows, size=k_miss, replace=False)] = True
        x_a, x_b = x_base.copy(), x_base.copy()
        x_a[miss_inf, 0], x_b[miss_mcar, 0] = np.nan, np.nan
        shared = 0.4 * x_base[:, 1]
        y_a = _standardize_1d(1.6 * miss_inf.astype(np.float64) + shared)
        y_b = _standardize_1d(1.6 * rng.permutation(miss_mcar).astype(np.float64) + shared)
        metrics = _pair_metrics(*harness.encode_table(x_a, y_a), *harness.encode_table(x_b, y_b))
        return {**base, "missing_rate_a": float(np.mean(np.isnan(x_a[:, 0]))), "missing_rate_b": float(np.mean(np.isnan(x_b[:, 0]))), **metrics}

    x_wide = rng.standard_normal((rows, 72))
    y_wide = _standardize_1d(1.1 * x_wide[:, 66] - 0.9 * x_wide[:, 67] + 1.2 * x_wide[:, 68] * x_wide[:, 69])
    x_null = x_wide.copy()
    for col_idx in range(66, 70):
        x_null[:, col_idx] = x_null[rng.permutation(rows), col_idx]
    trunc_sig, wide_sig = harness.encode_wide(x_wide, y_wide, pad_unobserved=4)
    trunc_null, wide_null = harness.encode_wide(x_null, y_wide, pad_unobserved=4)
    perm = rng.permutation(72)
    reordered = np.concatenate([np.array([66, 67, 68, 69]), np.setdiff1d(perm, [66, 67, 68, 69], assume_unique=True)])
    _, wide_reordered = harness.encode_wide(x_wide[:, reordered], y_wide, pad_unobserved=4)
    trunc_l2, _ = _distances(trunc_sig, trunc_null)
    wide_l2, wide_cos = _distances(wide_sig, wide_null)
    reorder_l2, reorder_cos = _distances(wide_sig, wide_reordered)
    return {
        **base,
        "total_columns": 72,
        "signal_columns": (66, 67, 68, 69),
        "truncated64_l2": trunc_l2,
        "wide_readout_l2": wide_l2,
        "wide_readout_cosine_distance": wide_cos,
        "reordering_l2": reorder_l2,
        "reordering_cosine_distance": reorder_cos,
    }


def evaluate_counterfactual_suite(*, seed: int = 20261120, rows: int = 384) -> dict[str, object]:
    """Run all 7 counterfactual pairs and return keyed receipts."""
    return {
        "schema": SCHEMA,
        "case_count": len(CASE_NAMES),
        "cases": {name: evaluate_counterfactual_case(name, seed=seed, rows=rows) for name in CASE_NAMES},
    }
