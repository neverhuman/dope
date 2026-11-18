"""Research view beside the pinned high encoder (`WP1` Ultra + `WP2` 196D Packet).

``research-encoder-view/v1`` is not a product recipe and not a weight release.
``DopeEncoder629.forward`` stays untouched at level ``high``. Ultra requires
explicit ``conditioning_level="ultra"`` and finite inputs. ``dope.packet-features/v1``
and ``dope.packet-global/v1`` omit target extrema (``y_min``/``y_max``) and match
``dope_core::mlc_v3::packet_features``.
"""
from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .encoder import CANONICAL_GATE_KEYS_14, DopeEncoder629

RESEARCH_ENCODER_VIEW_RECIPE = "research-encoder-view/v1"
PACKET_GLOBAL_SCHEMA = "dope.packet-global/v1"
PACKET_GLOBAL_WIDTH = 28
PACKET_FEATURES_SCHEMA = "dope.packet-features/v1"
PACKET_FEATURES_WIDTH = 196

PACKET_GLOBAL_FIELDS: tuple[str, ...] = (
    *CANONICAL_GATE_KEYS_14,
    "residual_std_scaled", "neural_w_active", "res_net_mean_tanh",
    "is_regression", "is_binary", "is_multiclass",
    "log2_k_classes_norm", "class_entropy_norm",
    "log10_n_train_norm", "log2_p_use_norm", "aspect_ratio_tanh",
    "n_trees_norm", "n_pairs_norm", "level_code",
)

LEVEL_CODE = {"high": 0.0, "ultra": 1.0}
FORBIDDEN_QUERY_KEYS = (
    "X_query", "y_query", "query_rows", "query_features",
    "X_test", "y_test", "X_qry", "y_qry", "X_val", "y_val",
    "X_tune", "y_tune", "query_targets", "test_rows",
)


def encode_research_view(
    encoder: DopeEncoder629,
    dataset_features: torch.Tensor,
    col_landmarks: torch.Tensor,
    *,
    level: str,
    conditioning_level: str | None = None,
) -> torch.Tensor:
    """Return ``[B, 629]``. ``high`` delegates to ``DopeEncoder629.forward``.

    ``ultra`` requires ``conditioning_level == "ultra"``, eval-mode ``DopeEncoder629``,
    and finite FP32 inputs so LayerNorm cannot silently collapse a row to zeros.
    """
    if not isinstance(encoder, DopeEncoder629):
        raise ValueError("encode_research_view requires a DopeEncoder629 instance")
    if level == "high":
        if conditioning_level not in (None, "high"):
            raise ValueError("high research view refuses non-high conditioning_level")
        return encoder.forward(dataset_features, col_landmarks)
    if level != "ultra":
        raise ValueError("research encoder view accepts only high or ultra")
    if conditioning_level != "ultra":
        raise ValueError("ultra research view requires conditioning_level='ultra'")
    if encoder.training:
        raise ValueError("ultra research view requires encoder.eval()")
    _check_ultra_tensors(encoder, dataset_features, col_landmarks)
    with torch.no_grad():
        routed = encoder.router(dataset_features, level="ultra")
        task = routed["task_latent"]
        columns = torch.cat([task.unsqueeze(1).expand(-1, 64, -1), col_landmarks], dim=-1)
        col_gates = 1.0 + 0.25 * torch.tanh(encoder.col_gate_head(columns).squeeze(-1))
        out = torch.cat([
            dataset_features, task, routed["dist_latent"],
            torch.stack([routed["gates"][k] for k in CANONICAL_GATE_KEYS_14], dim=-1),
            routed["landmark_pred"], routed["bb_pred"], col_gates,
        ], dim=-1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def packet_global_v1(encoding: Mapping[str, object], *, include_target_bundle: bool = True) -> tuple[float, ...]:
    """28 scale-invariant fit-row globals (zero target extrema, deduplicated gates)."""
    return packet_features_v1(encoding, include_target_bundle=include_target_bundle)[:PACKET_GLOBAL_WIDTH]


packet_global_v0 = packet_global_v1


def packet_features_v1(encoding: Mapping[str, object], *, include_target_bundle: bool = True) -> tuple[float, ...]:
    """196-D scale-invariant, column-permutation-invariant fitted packet vector (`WP2`)."""
    if not isinstance(encoding, Mapping):
        raise ValueError("packet_features_v1 requires an encode_table mapping")
    for key in FORBIDDEN_QUERY_KEYS:
        if key in encoding:
            raise ValueError(f"packet_features_v1 refuses {key}")
    role = encoding.get("role")
    if role is not None and role != "fit":
        raise ValueError("packet_features_v1 requires fit role")
    level = encoding.get("level")
    if level not in LEVEL_CODE:
        raise ValueError("packet_features_v1 requires level high or ultra")
    n_train = _count("n_train", encoding.get("n_train"))
    p_use = _count("p_use", encoding.get("p_use"))
    if p_use > 64:
        raise ValueError("packet_features_v1 requires p_use <= 64")

    gates = encoding.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(CANONICAL_GATE_KEYS_14):
        raise ValueError("packet_features_v1 requires the 14 canonical gates")
    bundle = encoding.get("target_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("packet_features_v1 requires target_bundle")

    task = str(encoding.get("task", "regression"))
    b_kind = str(bundle.get("task_kind", task))
    is_mc = task == "multiclass" or b_kind == "multiclass"
    is_bin = (not is_mc) and task in ("classification", "binary")
    is_reg = (not is_mc) and (not is_bin) and task in ("regression", "")
    if not (is_reg or is_bin or is_mc):
        raise ValueError("packet_features_v1 unsupported task")

    out = [0.0] * PACKET_FEATURES_WIDTH
    for idx, key in enumerate(CANONICAL_GATE_KEYS_14):
        out[idx] = _finite(f"gates.{key}", gates[key])

    trees_raw = _sequence("target_bundle.trees", bundle.get("trees", ()))
    mc_trees_raw = _sequence("target_bundle.mc_trees", bundle.get("mc_trees", ()))
    active_trees = mc_trees_raw if len(mc_trees_raw) > 0 else trees_raw
    n_trees = len(active_trees)
    pairs = _sequence("target_bundle.pairs", bundle.get("pairs", ()))
    priors_seq = bundle.get("class_priors", ())
    k_cls = (
        max(int(bundle.get("k_classes", 3)), len(priors_seq) if hasattr(priors_seq, "__len__") else 0, 3)
        if is_mc else (2 if is_bin else 1)
    )

    if include_target_bundle:
        for req in ("residual_std", "neural_w", "res_net_mean", "class_prior"):
            if req not in bundle:
                raise ValueError(f"packet_features_v1 requires target_bundle.{req}")
        res_std = _finite("target_bundle.residual_std", bundle["residual_std"])
        neural_w = _finite("target_bundle.neural_w", bundle["neural_w"])
        res_mean = _finite("target_bundle.res_net_mean", bundle["res_net_mean"])
        cls_prior = _finite("target_bundle.class_prior", bundle["class_prior"])
        if not is_mc:
            out[14] = min(max(res_std, 0.0), 4.0) / 4.0
            out[15] = min(max(neural_w, 0.0), 2.0) / 2.0
            out[16] = math.tanh(res_mean)
        if is_bin:
            p = min(max(cls_prior, 1e-6), 1.0 - 1e-6)
            out[21] = (-p * math.log(p) - (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)
        elif is_mc and priors_seq is not None and len(priors_seq) > 0:
            p_arr = [max(_finite("target_bundle.class_priors", x), 1e-6) for x in priors_seq]
            s_p = sum(p_arr)
            ent = sum(-(x / s_p) * math.log(x / s_p) for x in p_arr)
            out[21] = min(max(ent / math.log(max(len(p_arr), 2)), 0.0), 1.0)
        out[25] = min(max(n_trees / 56.0, 0.0), 2.0)
        out[26] = min(max(len(pairs) / 28.0, 0.0), 2.0)

    out[17] = 1.0 if is_reg else 0.0
    out[18] = 1.0 if is_bin else 0.0
    out[19] = 1.0 if is_mc else 0.0
    out[20] = min(max(math.log2(k_cls) / 8.0, 0.0), 2.0)
    out[22] = min(max(math.log10(n_train) / 6.0, 0.0), 2.0)
    out[23] = min(max(math.log2(p_use) / 6.0, 0.0), 2.0)
    out[24] = math.tanh(p_use / float(n_train))
    out[27] = LEVEL_CODE[level]

    reg_w = np.asarray(encoding.get("regime_weights", ()), dtype=np.float64).reshape(-1)
    if reg_w.size > 0:
        if not np.isfinite(reg_w).all():
            raise ValueError("packet_features_v1 non-finite regime_weights")
        rw = np.maximum(reg_w, 0.0)
        rw = np.sort(rw / max(float(np.sum(rw)), 1e-8))[::-1]
        for k in range(min(8, rw.size)):
            out[28 + k] = float(rw[k])

    col_knots = _col_matrix(encoding, "col_knots", p_use)
    chol = _col_matrix(encoding, "chol_factor", p_use)
    diag_res = _col_vec(encoding, "diag_res_std", p_use, default=1.0)
    is_bin_col = [bool(x) for x in _col_list(encoding, "is_binary_col", p_use, default=False)]
    bin_probs = _col_vec(encoding, "binary_probs", p_use, default=0.0)
    disc_map = encoding.get("discrete_support") if isinstance(encoding.get("discrete_support"), Mapping) else {}
    reg_means = np.asarray(encoding.get("regime_means", np.zeros((0, p_use))), dtype=np.float64)
    reg_scales = np.asarray(encoding.get("regime_scales", np.zeros((0, p_use))), dtype=np.float64)

    chol_sq = [float(np.sum(np.square(chol[j]))) for j in range(p_use)]
    chol_top1 = [float(np.max(np.abs(chol[j]))) if len(chol[j]) > 0 else 0.0 for j in range(p_use)]
    out[36], out[37], out[38] = _mean_std_max(chol_sq)
    out[39] = float(sum(diag_res[:p_use])) / float(p_use)

    basis_rows = _extract_basis_rows(bundle, is_mc) if include_target_bundle else []
    if include_target_bundle and basis_rows:
        l1 = sum(basis_rows)
        l2 = math.sqrt(sum(v * v for v in basis_rows))
        top4 = sum(sorted(basis_rows, reverse=True)[:4])
        out[40] = math.tanh(math.log1p(l2))
        out[41] = math.tanh(l1 / float(len(basis_rows)))
        out[42] = max(basis_rows) / (l2 + 1e-8)
        out[43] = top4 / (l1 + 1e-8)

    pair_deg = [0] * p_use
    uniq_pair_cols: set[int] = set()
    if include_target_bundle:
        for pair in pairs:
            a, c = int(pair[0]), int(pair[1])
            if 0 <= a < p_use:
                pair_deg[a] += 1
                uniq_pair_cols.add(a)
            if 0 <= c < p_use:
                pair_deg[c] += 1
                uniq_pair_cols.add(c)

    split_cnt, depth_w, root_cnt, col_thresh, all_thresh, all_leaves, depth_sum, first_rms, last_rms = (
        _scan_trees(active_trees, p_use) if include_target_bundle else (
            [0] * p_use, [0.0] * p_use, [0] * p_use, [[] for _ in range(p_use)], [], [], 0, 0.0, 0.0
        )
    )
    tot_splits = sum(split_cnt)
    tot_dw = sum(depth_w)
    sum_col_b = sum(
        _at(basis_rows, 1 + j) + _at(basis_rows, 1 + p_use + j) + _at(basis_rows, 1 + 2 * p_use + j)
        for j in range(p_use)
    )

    col_tokens = [[0.0] * 32 for _ in range(p_use)]
    activities = [0.0] * p_use
    for j in range(p_use):
        knots9 = _resample_knots_9(col_knots[j])
        tok = [0.0] * 32
        tok[:9] = knots9
        denom = max(abs((knots9[6] - knots9[2]) * 6.0), 1e-4)
        tok[9] = knots9[6] - knots9[2]
        tok[10] = math.tanh(((knots9[6] + knots9[2] - 2.0 * knots9[4]) * 6.0) / denom)
        tok[11] = math.tanh((((knots9[8] - knots9[0]) * 6.0) / denom) / 4.0)
        tok[12] = 1.0 if is_bin_col[j] else 0.0
        if is_bin_col[j]:
            pb = min(max(bin_probs[j], 0.0), 1.0)
            tok[13] = 4.0 * pb * (1.0 - pb)
        if j in disc_map:
            u_vals = disc_map[j][0]
            tok[14] = 1.0
            tok[15] = min(max(len(u_vals) / 32.0, 0.0), 2.0)
        tok[16] = min(max(chol_sq[j], 0.0), 2.0)
        tok[17] = min(max(chol_top1[j], 0.0), 2.0)
        tok[18] = min(max(diag_res[j], 0.0), 2.0)
        if reg_means.ndim == 2 and reg_means.shape[0] > 0 and reg_means.shape[1] > j:
            tok[19] = math.tanh(_mean_std_max(reg_means[:, j].tolist())[1])
        if reg_scales.ndim == 2 and reg_scales.shape[0] > 0 and reg_scales.shape[1] > j:
            tok[20] = min(max(float(np.mean(reg_scales[:, j])), 0.0), 4.0) / 4.0
        if include_target_bundle:
            w_l = _at(basis_rows, 1 + j)
            w_q = _at(basis_rows, 1 + p_use + j)
            w_a = _at(basis_rows, 1 + 2 * p_use + j)
            sp_sh = split_cnt[j] / float(max(tot_splits, 1))
            bs_sh = (w_l + w_q + w_a) / (sum_col_b + 1e-8)
            act = 0.5 * bs_sh + 0.5 * sp_sh
            tok[21] = pair_deg[j] / float(max(len(pairs), 1))
            tok[22] = sp_sh
            tok[23] = depth_w[j] / max(tot_dw, 1e-8)
            tok[24] = root_cnt[j] / float(max(n_trees, 1))
            if col_thresh[j]:
                tm, ts, _ = _mean_std_max(col_thresh[j])
                tok[25] = math.tanh(tm / 4.0)
                tok[26] = math.tanh(ts / 4.0) if len(col_thresh[j]) > 1 else 0.0
            tok[27] = math.tanh(w_l)
            tok[28] = math.tanh(w_q)
            tok[29] = math.tanh(w_a)
            tok[30] = bs_sh
            tok[31] = act
            activities[j] = act
        col_tokens[j] = tok

    max_act = max(activities) if activities else 0.0
    exp_w = [math.exp(4.0 * (a - max_act)) for a in activities]
    exp_sum = max(sum(exp_w), 1e-8)
    for d in range(32):
        vals = [col_tokens[j][d] for j in range(p_use)]
        m, s, mx = _mean_std_max(vals)
        out[44 + d] = m
        out[76 + d] = s
        out[108 + d] = mx
        out[140 + d] = sum((exp_w[j] / exp_sum) * col_tokens[j][d] for j in range(p_use))

    if include_target_bundle:
        if n_trees > 0:
            out[172] = min(max((depth_sum / float(n_trees)) / 4.0, 0.0), 2.0)
            if all_leaves:
                abs_l = [abs(v) for v in all_leaves]
                lm_abs, _, lmax_abs = _mean_std_max(abs_l)
                out[173] = math.tanh(lm_abs)
                out[174] = math.tanh(_mean_std_max(all_leaves)[1])
                out[175] = math.tanh(lmax_abs)
            out[176] = first_rms / (first_rms + last_rms + 1e-8)
            out[177] = sum(1 for c in split_cnt if c > 0) / float(p_use)
            if all_thresh:
                out[178] = math.tanh(_mean_std_max([abs(v) for v in all_thresh])[0] / 4.0)
                out[179] = math.tanh(_mean_std_max(all_thresh)[1] / 4.0)
        out[180] = len(uniq_pair_cols) / float(p_use)
        out[181] = max(pair_deg) / float(max(len(pairs), 1))
        t_start = min(1 + 3 * p_use, len(basis_rows))
        tail = basis_rows[t_start:]
        out[182] = math.tanh(_rms(tail))
        tot_l1 = sum(basis_rows) + 1e-8
        out[183] = sum(basis_rows[: min(1 + p_use, len(basis_rows))]) / tot_l1
        out[184] = sum(basis_rows[min(1 + p_use, len(basis_rows)) : min(1 + 2 * p_use, len(basis_rows))]) / tot_l1
        out[185] = sum(basis_rows[min(1 + 2 * p_use, len(basis_rows)) : min(1 + 3 * p_use, len(basis_rows))]) / tot_l1
        out[186] = sum(tail) / tot_l1
        out[187] = 1.0 if bool(bundle.get("include_v3_primitives", False)) else 0.0

    tl = [float(x) for x in np.asarray(encoding.get("task_latent", ()), dtype=np.float32).reshape(-1)]
    dl = [float(x) for x in np.asarray(encoding.get("dist_latent", ()), dtype=np.float32).reshape(-1)]
    out[188:192] = _latent_stats(tl)
    out[192:196] = _latent_stats(dl)

    return tuple(float(np.float32(v if math.isfinite(v) else 0.0)) for v in out)


def _check_ultra_tensors(encoder: DopeEncoder629, features: object, columns: object) -> None:
    if not isinstance(features, torch.Tensor) or not isinstance(columns, torch.Tensor):
        raise ValueError("ultra view requires dense FP32 tensors")
    if features.ndim != 2 or features.shape[1] != 256 or features.shape[0] < 1:
        raise ValueError("ultra view requires [B, 256] dataset_features with B >= 1")
    if columns.ndim != 3 or columns.shape != (features.shape[0], 64, 17):
        raise ValueError("ultra view requires matching [B, 64, 17] col_landmarks")
    if (features.dtype != torch.float32 or columns.dtype != torch.float32
            or features.layout != torch.strided or columns.layout != torch.strided):
        raise ValueError("ultra view requires strided FP32 tensors")
    dev = features.device
    if dev.type not in ("cpu", "cuda") or columns.device != dev:
        raise ValueError("ultra view requires matching CPU or CUDA device")
    if any(p.dtype != torch.float32 or p.device != dev for p in encoder.parameters()):
        raise ValueError("ultra view requires FP32 encoder parameters on input device")
    if torch.is_autocast_enabled(dev.type):
        raise ValueError("ultra view requires autocast disabled")
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(columns).all()):
        raise ValueError("ultra view refuses non-finite conditioning tensors")


def _extract_basis_rows(bundle: Mapping[str, object], is_mc: bool) -> list[float]:
    if is_mc and "U_basis" in bundle and bundle["U_basis"] is not None:
        u = np.asarray(bundle["U_basis"], dtype=np.float64)
        if not np.isfinite(u).all():
            raise ValueError("packet_features_v1 non-finite U_basis")
        if u.ndim == 2 and u.size > 0:
            return [float(math.sqrt(np.sum(np.square(row)))) for row in u]
    w = np.asarray(bundle.get("w_basis", ()), dtype=np.float64).reshape(-1)
    if not np.isfinite(w).all():
        raise ValueError("packet_features_v1 non-finite w_basis")
    return [float(abs(v)) for v in w]


def _scan_trees(trees: Sequence[object], p_use: int):
    split_cnt = [0] * p_use
    depth_w = [0.0] * p_use
    root_cnt = [0] * p_use
    col_thresh: list[list[float]] = [[] for _ in range(p_use)]
    all_thresh: list[float] = []
    all_leaves: list[float] = []
    depth_sum = 0
    first_rms, last_rms = 0.0, 0.0
    for ti, tree in enumerate(trees):
        if not isinstance(tree, Mapping):
            continue
        feats = [int(f) for f in tree.get("features", ())]
        ths = [_finite("tree.thresholds", th) for th in tree.get("thresholds", ())]
        depth_sum += len(feats)
        for d, (f, th) in enumerate(zip(feats, ths)):
            all_thresh.append(th)
            if 0 <= f < p_use:
                split_cnt[f] += 1
                depth_w[f] += 2.0 ** (-d)
                if d == 0:
                    root_cnt[f] += 1
                col_thresh[f].append(th)
        lv_arr = np.asarray(tree.get("leaf_values", ()), dtype=np.float64).reshape(-1)
        if not np.isfinite(lv_arr).all():
            raise ValueError("packet_features_v1 non-finite leaf_values")
        lv = [float(x) for x in lv_arr]
        rms = _rms(lv)
        if ti == 0:
            first_rms = rms
        if ti + 1 == len(trees):
            last_rms = rms
        all_leaves.extend(lv)
    return split_cnt, depth_w, root_cnt, col_thresh, all_thresh, all_leaves, depth_sum, first_rms, last_rms


def _col_matrix(encoding: Mapping[str, object], key: str, p_use: int) -> list[list[float]]:
    raw = encoding.get(key)
    if raw is None:
        return [[0.0] for _ in range(p_use)]
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < p_use or not np.isfinite(arr[:p_use]).all():
        raise ValueError(f"packet_features_v1 invalid {key}")
    return [arr[j].tolist() for j in range(p_use)]


def _col_vec(encoding: Mapping[str, object], key: str, p_use: int, *, default: float) -> list[float]:
    raw = encoding.get(key)
    if raw is None:
        return [default] * p_use
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < p_use or not np.isfinite(arr[:p_use]).all():
        raise ValueError(f"packet_features_v1 invalid {key}")
    return [float(x) for x in arr[:p_use]]


def _col_list(encoding: Mapping[str, object], key: str, p_use: int, *, default: object) -> list[object]:
    raw = encoding.get(key)
    if raw is None:
        return [default] * p_use
    seq = list(raw)
    if len(seq) < p_use:
        raise ValueError(f"packet_features_v1 invalid {key}")
    return seq[:p_use]


def _resample_knots_9(knots: Sequence[float]) -> list[float]:
    if not knots:
        return [0.0] * 9
    if len(knots) == 1:
        return [min(max(float(knots[0]), -6.0), 6.0) / 6.0] * 9
    last = float(len(knots) - 1)
    res = []
    for m in range(9):
        pos = (m / 8.0) * last
        lo = min(int(math.floor(pos)), len(knots) - 1)
        hi = min(lo + 1, len(knots) - 1)
        frac = pos - lo
        v = float(knots[lo]) * (1.0 - frac) + float(knots[hi]) * frac
        res.append(min(max(v, -6.0), 6.0) / 6.0)
    return res


def _latent_stats(vals: list[float]) -> list[float]:
    if not vals:
        return [0.0, 0.0, 0.0, 0.0]
    abs_v = [abs(v) for v in vals]
    m_abs, _, max_abs = _mean_std_max(abs_v)
    return [_rms(vals), m_abs, max_abs, _mean_std_max(vals)[1]]


def _rms(vals: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vals) / float(len(vals))) if vals else 0.0


def _mean_std_max(vals: Sequence[float]) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    n = float(len(vals))
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / n
    mx = max(vals)
    return m, math.sqrt(max(var, 0.0)), mx if math.isfinite(mx) else 0.0


def _at(vals: Sequence[float], idx: int) -> float:
    return vals[idx] if 0 <= idx < len(vals) else 0.0


def _count(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"packet_features_v1: {name} must be an integer count >= 1")
    try:
        n = operator.index(value)
    except TypeError:
        raise ValueError(f"packet_features_v1: {name} must be an integer count >= 1") from None
    if n < 1:
        raise ValueError(f"packet_features_v1: {name} must be an integer count >= 1")
    return n


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"packet_features_v1: {name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"packet_features_v1: {name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"packet_features_v1: {name} must be a finite number")
    return number


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"packet_features_v1 requires {name}")
    return value
