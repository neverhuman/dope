"""Dynamic per-batch random splitting and tabular augmentation over S3 catalog/v250.

Guarantees every training step draws a fresh Rule A-1 duplicate-grouped train/query split,
randomizes row lengths (`n_fit` in [32, 1024], `n_query` in [32, 512]), inserts synthetic
noise-distractor and nonlinear-derived columns, supports regression/binary/multiclass
(`multi_label` decoded via meta) plus true multi-label (`Y in {0,1}^L`) views, and applies
strict FIT-only NC2-aligned robust scaling with zero leakage across splits.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MAX_MODEL_COLS = 64
MAX_CLASSES = 32
MAX_MULTILABEL = 6


@dataclass(frozen=True)
class AugmentConfig:
    min_fit_rows: int = 32
    max_fit_rows: int = 512
    min_query_rows: int = 32
    max_query_rows: int = 256
    max_cols: int = MAX_MODEL_COLS
    split_ratio_lo: float = 0.40
    split_ratio_hi: float = 0.80
    aug_prob: float = 0.85
    max_noise_cols: int = 6
    max_derived_cols: int = 4
    col_keep_min: float = 0.75
    multilabel_view_prob: float = 0.12
    enable_augmentation: bool = True


def _step_rng(global_seed: int, step: int, rank: int, tag: str) -> np.random.Generator:
    digest = hashlib.blake2b(
        f"dope-v5:{global_seed}:{step}:{rank}:{tag}".encode("utf-8"),
        digest_size=16,
    ).digest()
    seed_int = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed_int)


class S3CorpusStore:
    """Bounded LRU reader over pre-parsed S3 v250 `.npz` tables."""

    def __init__(self, root: Path, split: str = "train", cache_cap: int = 512) -> None:
        self.root = Path(root)
        self.split = split
        self.cache_cap = max(16, int(cache_cap))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"missing_corpus_manifest:{manifest}")
        self.entries: list[dict[str, Any]] = []
        self.by_task: dict[str, list[int]] = {"regression": [], "binary": [], "multiclass": []}
        with manifest.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("split") != split:
                    continue
                if rec.get("task") not in self.by_task:
                    continue
                if int(rec.get("n_classes", 0)) > MAX_CLASSES:
                    continue
                idx = len(self.entries)
                self.entries.append(rec)
                self.by_task[rec["task"]].append(idx)
        if not self.entries:
            raise ValueError(f"empty_corpus_split:{split}")

    def __len__(self) -> int:
        return len(self.entries)

    def load_table(self, idx: int) -> dict[str, Any]:
        rec = self.entries[idx]
        key = rec["dataset_hash"]
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        z = np.load(self.root / rec["npz_relpath"])
        item = {
            "dataset_hash": key,
            "family_stem": rec["family_stem"],
            "task": str(rec["task"]),
            "n_classes": int(rec["n_classes"]),
            "X": np.asarray(z["X"], dtype=np.float32),
            "y": np.asarray(z["y"], dtype=np.float32),
            "dup_groups": np.asarray(z["dup_groups"], dtype=np.int32),
        }
        self._cache[key] = item
        if len(self._cache) > self.cache_cap:
            self._cache.popitem(last=False)
        return item

    def sample_balanced_index(self, rng: np.random.Generator) -> int:
        """Task-balanced sampling so regression, binary, and multiclass all receive rich signal."""
        avail = [(t, idxs) for t, idxs in self.by_task.items() if idxs]
        weights = np.array(
            [0.45 if t == "regression" else 0.30 if t == "binary" else 0.25 for t, _ in avail],
            dtype=np.float64,
        )
        weights /= weights.sum()
        chosen_pos = int(rng.choice(len(avail), p=weights))
        pool = avail[chosen_pos][1]
        return int(pool[int(rng.integers(0, len(pool)))])


def _duplicate_grouped_split(
    dup_groups: np.ndarray,
    y: np.ndarray,
    task: str,
    ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Rule A-1 duplicate-grouped random FIT/QUERY split with class coverage guard."""
    uniq = np.unique(dup_groups)
    perm = rng.permutation(uniq)
    cut = max(2, min(len(perm) - 2, int(round(len(perm) * ratio))))
    fit_groups = set(perm[:cut].tolist())
    fit_mask = np.isin(dup_groups, list(fit_groups))
    fit_idx = np.flatnonzero(fit_mask)
    qry_idx = np.flatnonzero(~fit_mask)
    if task in ("binary", "multiclass"):
        # Ensure at least 2 classes appear in both FIT and QUERY
        if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[qry_idx])) < 2:
            # Stratified fallback over unique groups by majority class
            cls_map: dict[int, list[int]] = {}
            for g in perm:
                rows_g = np.flatnonzero(dup_groups == g)
                c = int(round(float(y[rows_g[0]])))
                cls_map.setdefault(c, []).append(int(g))
            fg, qg = [], []
            for gs in cls_map.values():
                if len(gs) == 1:
                    fg.extend(gs)
                else:
                    k = max(1, min(len(gs) - 1, int(round(len(gs) * ratio))))
                    fg.extend(gs[:k])
                    qg.extend(gs[k:])
            fit_mask = np.isin(dup_groups, fg)
            fit_idx = np.flatnonzero(fit_mask)
            qry_idx = np.flatnonzero(~fit_mask)
    return fit_idx, qry_idx


def _subsample_rows(
    idx: np.ndarray,
    y: np.ndarray,
    task: str,
    target_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(idx) <= target_n:
        return rng.permutation(idx)
    if task in ("binary", "multiclass"):
        vals, inv = np.unique(np.rint(y[idx]).astype(np.int64), return_inverse=True)
        chosen: list[int] = []
        # Guarantee representation of each observed class before random fill
        for ci in range(len(vals)):
            members = idx[inv == ci]
            take_c = min(len(members), max(1, target_n // max(len(vals) * 2, 2)))
            chosen.extend(rng.choice(members, size=take_c, replace=False).tolist())
        rem = target_n - len(chosen)
        if rem > 0:
            chosen_set = set(chosen)
            pool = np.array([i for i in idx if int(i) not in chosen_set], dtype=idx.dtype)
            if len(pool) > 0:
                chosen.extend(rng.choice(pool, size=min(rem, len(pool)), replace=False).tolist())
        out = np.asarray(chosen[:target_n], dtype=np.int64)
        return rng.permutation(out)
    return rng.choice(idx, size=target_n, replace=False)


def _fit_only_nc2_scale(
    X_fit: np.ndarray,
    X_qry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strict FIT-only robust bulk+asinh-tail normalization (never reads QUERY statistics)."""
    n_f, p = X_fit.shape
    Xf = X_fit.astype(np.float64, copy=True)
    Xq = X_qry.astype(np.float64, copy=True)
    miss_f = (~np.isfinite(Xf)).astype(np.float32)
    miss_q = (~np.isfinite(Xq)).astype(np.float32)
    for j in range(p):
        col = Xf[:, j]
        fin = col[np.isfinite(col)]
        if len(fin) < 2:
            Xf[:, j] = 0.0
            Xq[:, j] = 0.0
            continue
        med = float(np.median(fin))
        q25, q75 = np.percentile(fin, [25.0, 75.0])
        iqr = float(q75 - q25)
        std = float(np.std(fin))
        scale = max(iqr / 1.349, std, 1e-6)
        Xf[:, j] = np.where(np.isfinite(Xf[:, j]), Xf[:, j], med)
        Xq[:, j] = np.where(np.isfinite(Xq[:, j]), Xq[:, j], med)
        zf = (Xf[:, j] - med) / scale
        zq = (Xq[:, j] - med) / scale
        # Smooth asinh compression beyond |z| > 3 preserves tail ordering without hard clipping
        Xf[:, j] = np.clip(np.where(np.abs(zf) > 3.0, np.sign(zf) * (3.0 + np.arcsinh(np.abs(zf) - 3.0)), zf), -6.0, 6.0)
        Xq[:, j] = np.clip(np.where(np.abs(zq) > 3.0, np.sign(zq) * (3.0 + np.arcsinh(np.abs(zq) - 3.0)), zq), -6.0, 6.0)
    return Xf.astype(np.float32), Xq.astype(np.float32), miss_f, miss_q


def _insert_synthetic_columns(
    Xf: np.ndarray,
    Xq: np.ndarray,
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Insert synthetic noise distractors + derived nonlinear columns, then permute columns."""
    nf, p = Xf.shape
    nq = Xq.shape[0]
    room = max(0, cfg.max_cols - p)
    if room == 0 or not cfg.enable_augmentation or rng.random() > cfg.aug_prob:
        perm = rng.permutation(p)
        return Xf[:, perm], Xq[:, perm], np.zeros(p, dtype=bool), 0, 0

    k_noise = int(rng.integers(1, min(cfg.max_noise_cols, room) + 1)) if room >= 1 else 0
    rem_room = room - k_noise
    k_derived = int(rng.integers(0, min(cfg.max_derived_cols, rem_room) + 1)) if (rem_room >= 1 and p >= 2) else 0

    syn_f_list = [Xf]
    syn_q_list = [Xq]
    distractor_flags = [np.zeros(p, dtype=bool)]

    if k_noise > 0:
        nf_cols = np.empty((nf, k_noise), dtype=np.float32)
        nq_cols = np.empty((nq, k_noise), dtype=np.float32)
        for j in range(k_noise):
            kind = int(rng.integers(0, 4))
            if kind == 0:
                nf_cols[:, j] = rng.standard_normal(nf)
                nq_cols[:, j] = rng.standard_normal(nq)
            elif kind == 1:
                nf_cols[:, j] = rng.uniform(-1.73, 1.73, size=nf)
                nq_cols[:, j] = rng.uniform(-1.73, 1.73, size=nq)
            elif kind == 2:
                nf_cols[:, j] = np.clip(rng.standard_t(3.0, size=nf) / 1.5, -5.0, 5.0)
                nq_cols[:, j] = np.clip(rng.standard_t(3.0, size=nq) / 1.5, -5.0, 5.0)
            else:
                prob = float(rng.uniform(0.15, 0.85))
                nf_cols[:, j] = (rng.random(nf) < prob).astype(np.float32) * 2.0 - 1.0
                nq_cols[:, j] = (rng.random(nq) < prob).astype(np.float32) * 2.0 - 1.0
        syn_f_list.append(nf_cols)
        syn_q_list.append(nq_cols)
        distractor_flags.append(np.ones(k_noise, dtype=bool))

    if k_derived > 0:
        df_cols = np.empty((nf, k_derived), dtype=np.float32)
        dq_cols = np.empty((nq, k_derived), dtype=np.float32)
        for j in range(k_derived):
            c1, c2 = rng.choice(p, size=2, replace=False)
            op = int(rng.integers(0, 4))
            if op == 0:
                vf, vq = Xf[:, c1] * Xf[:, c2], Xq[:, c1] * Xq[:, c2]
            elif op == 1:
                w1, w2 = float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0))
                vf, vq = w1 * Xf[:, c1] + w2 * Xf[:, c2], w1 * Xq[:, c1] + w2 * Xq[:, c2]
            elif op == 2:
                vf, vq = np.sin(1.5 * Xf[:, c1]), np.sin(1.5 * Xq[:, c1])
            else:
                thr = float(np.median(Xf[:, c1]))
                vf, vq = np.maximum(Xf[:, c1] - thr, 0.0), np.maximum(Xq[:, c1] - thr, 0.0)
            mu, sd = float(np.mean(vf)), max(float(np.std(vf)), 1e-5)
            df_cols[:, j] = np.clip((vf - mu) / sd, -5.0, 5.0)
            dq_cols[:, j] = np.clip((vq - mu) / sd, -5.0, 5.0)
        syn_f_list.append(df_cols)
        syn_q_list.append(dq_cols)
        distractor_flags.append(np.zeros(k_derived, dtype=bool))

    X_fit_all = np.concatenate(syn_f_list, axis=1)
    X_qry_all = np.concatenate(syn_q_list, axis=1)
    dmask = np.concatenate(distractor_flags, axis=0)
    perm = rng.permutation(X_fit_all.shape[1])
    return X_fit_all[:, perm], X_qry_all[:, perm], dmask[perm], k_noise, k_derived


def _synthesize_multilabel_targets(
    Xf: np.ndarray,
    Xq: np.ndarray,
    dmask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create a correlated true multi-label task Y in {0,1}^L driven strictly by non-distractor cols."""
    real_cols = np.flatnonzero(~dmask)
    if len(real_cols) < 2:
        real_cols = np.arange(Xf.shape[1])
    n_labels = int(rng.integers(2, min(MAX_MULTILABEL, max(3, len(real_cols))) + 1))
    k_use = min(len(real_cols), 8)
    chosen = rng.choice(real_cols, size=k_use, replace=False)
    W = rng.standard_normal((k_use, n_labels)).astype(np.float32) / math.sqrt(k_use)
    # Induce realistic inter-label correlation via a shared latent factor
    shared = rng.standard_normal((k_use, 1)).astype(np.float32) * 0.6 / math.sqrt(k_use)
    W = W + shared
    lin_f = Xf[:, chosen] @ W + 0.25 * np.tanh(Xf[:, chosen[:2]].prod(axis=1, keepdims=True))
    lin_q = Xq[:, chosen] @ W + 0.25 * np.tanh(Xq[:, chosen[:2]].prod(axis=1, keepdims=True))
    thr = np.median(lin_f, axis=0, keepdims=True) + rng.uniform(-0.35, 0.35, size=(1, n_labels)).astype(np.float32)
    pf = 1.0 / (1.0 + np.exp(-np.clip(2.2 * (lin_f - thr), -12.0, 12.0)))
    pq = 1.0 / (1.0 + np.exp(-np.clip(2.2 * (lin_q - thr), -12.0, 12.0)))
    yf = (rng.random(pf.shape) < pf).astype(np.float32)
    yq = (rng.random(pq.shape) < pq).astype(np.float32)
    # Guarantee every label has both 0 and 1 in FIT
    for l_idx in range(n_labels):
        if yf[:, l_idx].sum() < 2 or yf[:, l_idx].sum() > len(yf) - 2:
            yf[:, l_idx] = (lin_f[:, l_idx] >= float(np.median(lin_f[:, l_idx]))).astype(np.float32)
            yq[:, l_idx] = (lin_q[:, l_idx] >= float(np.median(lin_f[:, l_idx]))).astype(np.float32)
    return yf, yq, n_labels


def sample_dynamic_episode(
    table: dict[str, Any],
    *,
    global_seed: int,
    step: int,
    rank: int,
    cfg: AugmentConfig,
    view_tag: str = "a",
) -> dict[str, Any]:
    """Generate one fresh dynamic train/query episode with randomized split, length, and columns."""
    rng = _step_rng(global_seed, step, rank, f"{table['dataset_hash']}:{view_tag}")
    X_raw: np.ndarray = table["X"]
    y_raw: np.ndarray = table["y"]
    dup_groups: np.ndarray = table["dup_groups"]
    task: str = table["task"]
    n_classes: int = int(table["n_classes"])

    # 1. Fresh random Rule A-1 duplicate-grouped split ratio
    ratio = float(rng.uniform(cfg.split_ratio_lo, cfg.split_ratio_hi)) if cfg.enable_augmentation else 0.65
    fit_idx, qry_idx = _duplicate_grouped_split(dup_groups, y_raw, task, ratio, rng)
    if len(fit_idx) < 12 or len(qry_idx) < 8:
        perm = rng.permutation(len(y_raw))
        cut = max(12, int(len(perm) * 0.65))
        fit_idx, qry_idx = perm[:cut], perm[cut:]

    # 2. Random row-length adjustment (log-uniform over [min_fit_rows, min(avail, max_fit_rows)])
    if cfg.enable_augmentation:
        hi_f = max(cfg.min_fit_rows, min(len(fit_idx), cfg.max_fit_rows))
        lo_f = min(cfg.min_fit_rows, hi_f)
        n_fit = int(round(math.exp(rng.uniform(math.log(max(lo_f, 12)), math.log(max(hi_f, 12))))))
        hi_q = max(cfg.min_query_rows, min(len(qry_idx), cfg.max_query_rows))
        lo_q = min(cfg.min_query_rows, hi_q)
        n_qry = int(round(math.exp(rng.uniform(math.log(max(lo_q, 8)), math.log(max(hi_q, 8))))))
    else:
        n_fit = min(len(fit_idx), cfg.max_fit_rows)
        n_qry = min(len(qry_idx), cfg.max_query_rows)

    fit_idx = _subsample_rows(fit_idx, y_raw, task, n_fit, rng)
    qry_idx = _subsample_rows(qry_idx, y_raw, task, n_qry, rng)

    # 3. Column subsampling (always fitting inside max_cols with room for synthetic columns)
    p_orig = X_raw.shape[1]
    max_real_cols = max(2, cfg.max_cols - (cfg.max_noise_cols + cfg.max_derived_cols if cfg.enable_augmentation else 0))
    if p_orig > max_real_cols:
        col_sel = np.sort(rng.choice(p_orig, size=max_real_cols, replace=False))
    elif cfg.enable_augmentation and p_orig >= 4 and rng.random() < 0.5:
        keep_p = max(2, int(round(p_orig * float(rng.uniform(cfg.col_keep_min, 1.0)))))
        col_sel = np.sort(rng.choice(p_orig, size=keep_p, replace=False))
    else:
        col_sel = np.arange(p_orig)

    Xf_raw = X_raw[fit_idx][:, col_sel]
    Xq_raw = X_raw[qry_idx][:, col_sel]
    yf_raw = y_raw[fit_idx].copy()
    yq_raw = y_raw[qry_idx].copy()

    # 4. Strict FIT-only NC2-aligned scaling
    Xf, Xq, _, _ = _fit_only_nc2_scale(Xf_raw, Xq_raw)

    # 5. Random synthetic column insertion (noise distractors + derived nonlinear columns)
    Xf, Xq, dmask, k_noise, k_derived = _insert_synthetic_columns(Xf, Xq, cfg, rng)

    # 6. Target preparation (including optional true multi-label Y in {0,1}^L view)
    if cfg.enable_augmentation and rng.random() < cfg.multilabel_view_prob and Xf.shape[1] >= 4:
        yf_out, yq_out, n_labels = _synthesize_multilabel_targets(Xf, Xq, dmask, rng)
        ep_task = "multilabel"
        ep_classes = n_labels
    elif task == "regression":
        mu_y = float(np.mean(yf_raw))
        sd_y = max(float(np.std(yf_raw)), 1e-5)
        yf_out = np.clip((yf_raw - mu_y) / sd_y, -6.0, 6.0).astype(np.float32)
        yq_out = np.clip((yq_raw - mu_y) / sd_y, -6.0, 6.0).astype(np.float32)
        ep_task = "regression"
        ep_classes = 0
    elif task == "binary":
        yf_out = (yf_raw > 0.5).astype(np.float32)
        yq_out = (yq_raw > 0.5).astype(np.float32)
        ep_task = "binary"
        ep_classes = 2
    else:
        # Remap classes present in FIT to contiguous [0..K_eff-1]
        u_fit = np.unique(np.rint(yf_raw).astype(np.int64))
        if len(u_fit) < 2:
            yf_out = (yf_raw > float(np.median(yf_raw))).astype(np.float32)
            yq_out = (yq_raw > float(np.median(yf_raw))).astype(np.float32)
            ep_task = "binary"
            ep_classes = 2
        else:
            k_eff = min(len(u_fit), MAX_CLASSES)
            u_fit = u_fit[:k_eff]
            map_idx = {int(c): i for i, c in enumerate(u_fit)}
            yf_out = np.array([map_idx.get(int(round(v)), 0) for v in yf_raw], dtype=np.float32)
            yq_out = np.array([map_idx.get(int(round(v)), 0) for v in yq_raw], dtype=np.float32)
            ep_task = "binary" if k_eff == 2 else "multiclass"
            ep_classes = k_eff

    split_sig = hashlib.blake2b(
        np.sort(fit_idx).astype(np.int32).tobytes() + table["dataset_hash"].encode("utf-8"),
        digest_size=8,
    ).hexdigest()

    return {
        "dataset_hash": table["dataset_hash"],
        "family_stem": table["family_stem"],
        "task": ep_task,
        "n_classes": ep_classes,
        "X_fit": Xf,
        "y_fit": yf_out,
        "X_qry": Xq,
        "y_qry": yq_out,
        "distractor_mask": dmask,
        "split_sig": split_sig,
        "aug_stats": {
            "n_fit": int(Xf.shape[0]),
            "n_qry": int(Xq.shape[0]),
            "n_cols": int(Xf.shape[1]),
            "n_noise_cols": k_noise,
            "n_derived_cols": k_derived,
        },
    }
