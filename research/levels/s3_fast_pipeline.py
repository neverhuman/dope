"""High-speed 100% unique-per-batch RAM-resident data & curriculum pipeline for DOPE v2.0.0.

Guarantees:
- Distinct datasets without replacement inside every batch (0 duplicate datasets per batch).
- Fresh Rule A-1 duplicate-grouped random FIT/QUERY split with jittered ratio [0.45, 0.78] every time.
- Rich multi-layer augmentation: continuous micro-jitter, synthetic noise & interaction columns,
  random column permutation, class-ID permutation, MCAR/MAR missingness with preserved masks,
  raw heavy-tail/lattice feature warps, target-swap real views (~10x task expansion),
  correlated multi-label views, and infinite on-the-fly synthetic SCM/tree/interaction prior tables.
- 3-stage progressive data-size curriculum (small/fast -> medium -> full resolution) + JopeDime
  regret-weighted hard-table sampling (up to 6x boost with 35% replay floor).
- Background CPU prefetcher queue so GPU workers never stall on data preparation.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
from typing import Any
import numpy as np

from .s3_dynamic_augment import (
    S3CorpusStore,
    _duplicate_grouped_split,
    _fit_only_nc2_scale,
    _insert_synthetic_columns,
    _step_rng,
    _synthesize_multilabel_targets,
)


@dataclass(frozen=True)
class ProgressiveStage:
    stage_id: int
    min_fit: int
    max_fit: int
    min_qry: int
    max_qry: int
    max_cols: int


def curriculum_stage_for_step(step: int) -> ProgressiveStage:
    """3-stage progressive data-size ramp: fast small tables early -> full resolution later."""
    if step < 2500:
        return ProgressiveStage(1, 32, 128, 32, 96, 32)
    if step < 10000:
        return ProgressiveStage(2, 64, 288, 48, 160, 56)
    return ProgressiveStage(3, 96, 512, 64, 256, 80)


class RamCorpusStore:
    """RAM-resident preloaded S3 v250 store with JopeDime regret-weighted sampling."""

    def __init__(self, corpus_dir: str | Path, split: str = "train", preload: bool = True) -> None:
        self.base = S3CorpusStore(corpus_dir, split=split, cache_cap=4096 if preload else 512)
        self.tables: list[dict[str, Any] | None] = [None] * len(self.base.rows)
        self.regret_weights = np.ones(len(self.base.rows), dtype=np.float64)
        self.hash_to_idx = {r["dataset_hash"]: i for i, r in enumerate(self.base.rows)}
        if preload:
            for i in range(len(self.base.rows)):
                self.tables[i] = self.base.load_table(i)

    def __len__(self) -> int:
        return len(self.base.rows)

    def get(self, idx: int) -> dict[str, Any]:
        tbl = self.tables[idx]
        if tbl is None:
            tbl = self.base.load_table(idx)
            self.tables[idx] = tbl
        return tbl

    def update_regret_from_file(self, regret_path: Path, replay_floor: float = 0.35, max_mult: float = 6.0) -> float:
        """Apply JopeDime regret-weighted multipliers with guaranteed replay floor."""
        if not regret_path.is_file():
            return 1.0
        try:
            payload = json.loads(regret_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 1.0
        task_mult = payload.get("task_multipliers", {})
        hash_mult = payload.get("hash_multipliers", {})
        w = np.ones(len(self.base.rows), dtype=np.float64)
        for i, row in enumerate(self.base.rows):
            tm = float(task_mult.get(row["task"], 1.0))
            hm = float(hash_mult.get(row["dataset_hash"], 1.0))
            raw_m = min(max(tm * hm, 1.0), max_mult)
            w[i] = replay_floor * 1.0 + (1.0 - replay_floor) * raw_m
        self.regret_weights = w
        return float(np.mean(w))

    def sample_unique_indices(self, batch_size: int, rng: np.random.Generator) -> list[int]:
        """Sample `batch_size` DISTINCT table indices across all tasks without replacement."""
        chosen: list[int] = []
        used: set[int] = set()
        task_cycle = ["regression", "binary", "multiclass", "regression", "binary", "multiclass"]
        for slot in range(batch_size):
            pref = task_cycle[slot % len(task_cycle)]
            pool = self.base.by_task.get(pref) or list(range(len(self.base.rows)))
            avail = [i for i in pool if i not in used]
            if not avail:
                avail = [i for i in range(len(self.base.rows)) if i not in used]
            if not avail:
                break
            w = self.regret_weights[avail]
            p = w / max(float(w.sum()), 1e-12)
            pick = int(rng.choice(avail, p=p))
            used.add(pick)
            chosen.append(pick)
        return chosen


def synthesize_scm_prior_table(rng: np.random.Generator, stage: ProgressiveStage) -> dict[str, Any]:
    """Generate a brand-new synthetic Structural Causal Model / Interaction / Tree prior table.

    Provides mathematically infinite unique datasets with exact known nonlinear/interaction/regime
    structure across regression, binary, and multiclass (`K=3..12`) tasks (`tips/*.md` prior).
    """
    n_tot = int(rng.integers(stage.min_fit + stage.min_qry + 32, stage.max_fit + stage.max_qry + 64))
    p = int(rng.integers(4, max(6, min(stage.max_cols - 6, 36))))
    n_regimes = int(rng.integers(1, 4))
    reg_id = rng.integers(0, n_regimes, size=n_tot)

    # Generate correlated multi-regime latent factors + nonlinear SCM edge propagation
    r_dim = int(rng.integers(2, min(8, p + 1)))
    X = np.zeros((n_tot, p), dtype=np.float32)
    for r in range(n_regimes):
        mask = reg_id == r
        nr = int(mask.sum())
        if nr == 0:
            continue
        load = rng.normal(0.0, 0.7, size=(r_dim, p)).astype(np.float32)
        shift = rng.normal(0.0, 0.8, size=(1, p)).astype(np.float32)
        lat = rng.normal(size=(nr, r_dim)).astype(np.float32)
        X[mask] = lat @ load + shift + 0.45 * rng.normal(size=(nr, p)).astype(np.float32)

    # Inject nonlinear DAG causal edges + occasional discrete/quantized columns
    for j in range(1, p):
        if rng.random() < 0.45:
            pa, kind = int(rng.integers(0, j)), int(rng.integers(0, 4))
            if kind == 0:
                X[:, j] += 0.6 * np.tanh(X[:, pa])
            elif kind == 1:
                X[:, j] += 0.35 * (X[:, pa] ** 2 - 1.0)
            elif kind == 2:
                X[:, j] += 0.5 * np.maximum(X[:, pa], 0.0)
            else:
                X[:, j] += 0.4 * X[:, pa] * X[:, int(rng.integers(0, j))]
        if rng.random() < 0.25:
            bins = int(rng.integers(2, 9))
            qs = np.quantile(X[:, j], np.linspace(0.0, 1.0, bins + 1)[1:-1])
            X[:, j] = np.digitize(X[:, j], qs).astype(np.float32)

    k_drv = int(rng.integers(2, min(8, p + 1)))
    drv = rng.choice(p, size=k_drv, replace=False)
    task_roll = float(rng.random())
    if task_roll < 0.38:
        task, n_classes = "regression", 1
        sig = X[:, drv] @ rng.normal(0.0, 1.0, size=k_drv).astype(np.float32) + 0.5 * np.sin(1.4 * X[:, drv[0]]) + 0.45 * (X[:, drv[0]] * X[:, drv[-1]])
        if k_drv >= 3:
            sig = sig + 0.4 * ((X[:, drv[1]] > 0.0).astype(np.float32) * X[:, drv[2]])
        y = (sig + rng.normal(0.0, float(rng.uniform(0.15, 0.55)) * max(float(np.std(sig)), 1e-4), size=n_tot)).astype(np.float32)
    elif task_roll < 0.68:
        task, n_classes = "binary", 2
        lg = X[:, drv] @ rng.normal(0.0, 1.2, size=k_drv).astype(np.float32) + 0.7 * (X[:, drv[0]] * X[:, drv[-1]]) - 0.5 * np.abs(X[:, drv[0]])
        lg = (lg - np.median(lg)) / max(float(np.std(lg)), 1e-4) * float(rng.uniform(1.2, 2.8))
        y = (rng.random(n_tot) < (1.0 / (1.0 + np.exp(-np.clip(lg, -12.0, 12.0))))).astype(np.float32)
    else:
        task, n_classes = "multiclass", int(rng.integers(3, 11))
        logits = X[:, drv] @ rng.normal(0.0, 1.1, size=(k_drv, n_classes)).astype(np.float32)
        for c in range(n_classes):
            a, b = drv[c % k_drv], drv[(c + 1) % k_drv]
            logits[:, c] += 0.55 * (X[:, a] * X[:, b]) + 0.4 * np.tanh(X[:, a])
        logits = (logits - logits.mean(axis=0, keepdims=True)) / max(float(logits.std()), 1e-4) * float(rng.uniform(1.6, 3.2))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        y = (rng.random((n_tot, 1)) > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, n_classes - 1).astype(np.float32)

    h = hashlib.sha256(rng.bytes(16)).hexdigest()[:16]
    return {
        "dataset_hash": f"scm_{h}", "family_stem": f"scm_prior_{task}_{p}d", "task": task, "n_classes": n_classes,
        "X": np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), "y": y.astype(np.float32),
        "dup_groups": np.arange(n_tot, dtype=np.int32),
    }


def _apply_target_swap(table: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Turn a non-constant feature x_j into target y (~10x expansion of real S3 tasks)."""
    X = table["X"]
    if X.shape[1] < 4:
        return table
    cand = [j for j in range(X.shape[1]) if float(np.nanstd(X[:, j])) > 1e-5]
    if len(cand) < 3:
        return table
    tgt_j = int(rng.choice(cand))
    y_new = X[:, tgt_j].copy()
    keep = [j for j in range(X.shape[1]) if j != tgt_j]
    X_new = X[:, keep].copy()
    uniq = np.unique(y_new[np.isfinite(y_new)])
    if len(uniq) == 2:
        task, k = "binary", 2
        y_new = (y_new == uniq[1]).astype(np.float32)
    elif 3 <= len(uniq) <= 12:
        task, k = "multiclass", len(uniq)
        mp = {float(v): i for i, v in enumerate(uniq)}
        y_new = np.array([mp.get(float(v), 0) for v in y_new], dtype=np.float32)
    else:
        task, k = "regression", 1
    return {
        "dataset_hash": f"{table['dataset_hash']}_sw{tgt_j}",
        "family_stem": table["family_stem"],
        "task": task,
        "n_classes": k,
        "X": X_new,
        "y": y_new,
        "dup_groups": table["dup_groups"],
    }


def _apply_raw_feature_warps_and_missingness(
    Xf: np.ndarray,
    Xq: np.ndarray,
    rng: np.random.Generator,
    warp_prob: float = 0.25,
    miss_prob: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inject realistic raw-scale heavy-tail/skew warps, micro-jitter, and MCAR/MAR missingness."""
    Xf = Xf.copy()
    Xq = Xq.copy()
    n_f, p = Xf.shape
    n_q = Xq.shape[0]
    miss_f = np.zeros((n_f, p), dtype=np.float32)
    miss_q = np.zeros((n_q, p), dtype=np.float32)

    do_warp = float(rng.random()) < warp_prob
    for j in range(p):
        u_cnt = len(np.unique(Xf[:, j]))
        if u_cnt > 16:
            # Slight continuous micro-jitter (0.2% - 2.5% of IQR) so every view is strictly fresh
            q25, q75 = np.percentile(Xf[:, j], [25.0, 75.0])
            iqr = max(float(q75 - q25), float(np.std(Xf[:, j])), 1e-4)
            sig = float(rng.uniform(0.002, 0.025)) * iqr
            Xf[:, j] += rng.normal(0.0, sig, size=n_f).astype(np.float32)
            Xq[:, j] += rng.normal(0.0, sig, size=n_q).astype(np.float32)
            if do_warp and rng.random() < 0.35:
                kind = int(rng.integers(0, 3))
                if kind == 0:
                    # Heavy-tail exponential/power skew warp (tests NC2 tail handling)
                    alpha = float(rng.uniform(1.3, 2.8))
                    Xf[:, j] = np.sign(Xf[:, j]) * (np.abs(Xf[:, j]) ** alpha)
                    Xq[:, j] = np.sign(Xq[:, j]) * (np.abs(Xq[:, j]) ** alpha)
                elif kind == 1:
                    # Decimal rounding / lattice heaping
                    step = float(rng.choice([0.05, 0.1, 0.25]))
                    Xf[:, j] = np.round(Xf[:, j] / step) * step
                    Xq[:, j] = np.round(Xq[:, j] / step) * step
                else:
                    # Affine unit rescaling
                    sc = float(rng.uniform(0.2, 50.0))
                    sh = float(rng.uniform(-20.0, 20.0))
                    Xf[:, j] = Xf[:, j] * sc + sh
                    Xq[:, j] = Xq[:, j] * sc + sh

    # Realistic MCAR / MAR missingness injection (since 0/300 raw S3 tables have NaNs!)
    if float(rng.random()) < miss_prob and p >= 3:
        n_mcols = int(rng.integers(1, max(2, min(6, p // 2 + 1))))
        mcols = rng.choice(p, size=n_mcols, replace=False)
        for j in mcols:
            rate = float(rng.uniform(0.04, 0.20))
            if rng.random() < 0.5 or p < 2:
                mf = rng.random(n_f) < rate
                mq = rng.random(n_q) < rate
            else:
                # MAR: missingness depends on another column's upper tail
                driver = int((j + 1) % p)
                thr = float(np.quantile(Xf[:, driver], 1.0 - min(rate * 1.8, 0.45)))
                mf = (Xf[:, driver] >= thr) & (rng.random(n_f) < 0.65)
                mq = (Xq[:, driver] >= thr) & (rng.random(n_q) < 0.65)
            miss_f[:, j] = mf.astype(np.float32)
            miss_q[:, j] = mq.astype(np.float32)
            Xf[mf, j] = np.nan
            Xq[mq, j] = np.nan

    return Xf, Xq, miss_f, miss_q


def build_unique_episode(
    store: RamCorpusStore,
    tbl_idx: int,
    *,
    global_seed: int,
    step: int,
    rank: int,
    slot: int,
    view_tag: str = "a",
    scm_prob: float = 0.18,
    target_swap_prob: float = 0.18,
    multilabel_prob: float = 0.12,
    aug_prob: float = 0.90,
) -> dict[str, Any]:
    """Build a guaranteed-unique episode with progressive size, fresh split, and rich augmentation."""
    stage = curriculum_stage_for_step(step)
    rng = _step_rng(global_seed, step, rank, f"slot{slot}:{tbl_idx}:{view_tag}")

    source_kind = "s3_real"
    if float(rng.random()) < scm_prob and view_tag == "a":
        table = synthesize_scm_prior_table(rng, stage)
        source_kind = "scm_prior"
    else:
        table = store.get(tbl_idx)
        if float(rng.random()) < target_swap_prob and view_tag == "a":
            table = _apply_target_swap(table, rng)
            source_kind = "s3_target_swap"

    X_raw = table["X"]
    y_raw = table["y"]
    dg = table["dup_groups"]
    task = table["task"]
    n_classes = int(table["n_classes"])
    n_tot, p_raw = X_raw.shape

    # Column cap + random column subset on wide tables
    p_cap = max(2, stage.max_cols - 8)
    if p_raw > p_cap:
        cols = np.sort(rng.choice(p_raw, size=p_cap, replace=False))
        X_raw = X_raw[:, cols]

    # Jittered random split ratio every single episode + Rule A-1 duplicate grouping
    train_frac = float(rng.uniform(0.45, 0.78))
    fit_idx, qry_idx = _duplicate_grouped_split(dg, y_raw, task, train_frac, rng)
    if len(fit_idx) < 16 or len(qry_idx) < 12:
        perm = rng.permutation(n_tot)
        cut = max(16, int(n_tot * 0.65))
        fit_idx, qry_idx = perm[:cut], perm[cut:]

    # Progressive random row-length resampling
    t_fit = int(rng.integers(stage.min_fit, stage.max_fit + 1))
    t_qry = int(rng.integers(stage.min_qry, stage.max_qry + 1))
    if len(fit_idx) > t_fit:
        fit_idx = rng.choice(fit_idx, size=t_fit, replace=False)
    if len(qry_idx) > t_qry:
        qry_idx = rng.choice(qry_idx, size=t_qry, replace=False)

    Xf_raw, Xq_raw = X_raw[fit_idx].copy(), X_raw[qry_idx].copy()
    yf, yq = y_raw[fit_idx].copy(), y_raw[qry_idx].copy()

    # Raw warps, micro-jitter, and MCAR/MAR missingness BEFORE fit-only NC2 scaling
    Xf_w, Xq_w, miss_f_inj, miss_q_inj = _apply_raw_feature_warps_and_missingness(Xf_raw, Xq_raw, rng)
    Xf, Xq, mf_obs, mq_obs = _fit_only_nc2_scale(Xf_w, Xq_w)
    miss_f = np.maximum(miss_f_inj, mf_obs).astype(np.float32)
    miss_q = np.maximum(miss_q_inj, mq_obs).astype(np.float32)

    # Synthetic noise + derived interaction columns + column permutation
    if float(rng.random()) < aug_prob:
        Xf, Xq, dmask, n_noise, n_derived = _insert_synthetic_columns(
            Xf, Xq, rng, max_noise=6, max_derived=4, max_total_cols=stage.max_cols
        )
        pad_c = Xf.shape[1] - miss_f.shape[1]
        if pad_c > 0:
            miss_f = np.concatenate([miss_f, np.zeros((Xf.shape[0], pad_c), dtype=np.float32)], axis=1)
            miss_q = np.concatenate([miss_q, np.zeros((Xq.shape[0], pad_c), dtype=np.float32)], axis=1)
    else:
        dmask, n_noise, n_derived = np.zeros(Xf.shape[1], dtype=bool), 0, 0

    cperm = rng.permutation(Xf.shape[1])
    Xf, Xq, miss_f, miss_q, dmask = Xf[:, cperm], Xq[:, cperm], miss_f[:, cperm], miss_q[:, cperm], dmask[cperm]

    if task == "regression" and float(rng.random()) < multilabel_prob and Xf.shape[1] >= 4:
        yf, yq, n_classes = _synthesize_multilabel_targets(Xf, Xq, dmask, rng)
        task = "multilabel"
    elif task == "regression":
        mu, sd = float(np.nanmean(yf)), max(float(np.nanstd(yf)), 1e-5)
        yf = np.clip(np.nan_to_num((yf - mu) / sd, nan=0.0), -6.0, 6.0).astype(np.float32)
        yq = np.clip(np.nan_to_num((yq - mu) / sd, nan=0.0), -6.0, 6.0).astype(np.float32)
        n_classes = 1
    elif task == "binary":
        yf, yq, n_classes = (yf > 0.5).astype(np.float32), (yq > 0.5).astype(np.float32), 2
        if float(rng.random()) < 0.5:
            yf, yq = 1.0 - yf, 1.0 - yq
    else:
        u_cls = np.unique(np.rint(yf).astype(np.int64))
        if len(u_cls) < 2:
            yf, yq, task, n_classes = (rng.random(len(yf)) > 0.5).astype(np.float32), (rng.random(len(yq)) > 0.5).astype(np.float32), "binary", 2
        else:
            perm_ids = rng.permutation(len(u_cls))
            mp = {int(c): int(perm_ids[i]) for i, c in enumerate(u_cls)}
            yf = np.array([mp.get(int(round(v)), 0) for v in yf], dtype=np.float32)
            yq = np.array([mp.get(int(round(v)), 0) for v in yq], dtype=np.float32)
            n_classes = len(u_cls)

    h = hashlib.blake2b(digest_size=10)
    h.update(f"{table['dataset_hash']}:{step}:{rank}:{slot}:{view_tag}:{Xf.shape}".encode())
    h.update(np.sort(fit_idx).astype(np.int32).tobytes())
    return {
        "dataset_hash": table["dataset_hash"], "family_stem": table["family_stem"], "source_kind": source_kind,
        "task": task, "n_classes": int(n_classes), "stage_id": stage.stage_id,
        "X_fit": Xf.astype(np.float32), "y_fit": yf.astype(np.float32), "miss_fit": miss_f.astype(np.float32),
        "X_qry": Xq.astype(np.float32), "y_qry": yq.astype(np.float32), "miss_qry": miss_q.astype(np.float32),
        "distractor_mask": dmask, "split_sig": h.hexdigest(),
        "aug_stats": {
            "n_fit": int(Xf.shape[0]), "n_qry": int(Xq.shape[0]), "n_cols": int(Xf.shape[1]),
            "n_noise_cols": int(n_noise), "n_derived_cols": int(n_derived),
            "miss_rate": float(miss_f.mean()), "train_frac": round(train_frac, 3), "stage": stage.stage_id,
        },
    }


class AsyncBatchPrefetcher:
    """Background thread-pool prefetcher producing guaranteed-unique step batches at RAM speed."""

    def __init__(
        self,
        store: RamCorpusStore,
        *,
        batch_size: int,
        global_seed: int,
        rank: int,
        start_step: int = 1,
        num_workers: int = 4,
        queue_size: int = 12,
        regret_path: Path | None = None,
    ) -> None:
        self.store = store
        self.batch_size = batch_size
        self.global_seed = global_seed
        self.rank = rank
        self.regret_path = regret_path
        self._step = start_step
        self._stop = threading.Event()
        self._q: queue.Queue[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]] = queue.Queue(maxsize=queue_size)
        self._pool = ThreadPoolExecutor(max_workers=max(1, num_workers), thread_name_prefix="dope_prep")
        self._rng = np.random.default_rng(global_seed + rank * 100_003)
        self._prod = threading.Thread(target=self._run, daemon=True)
        self._prod.start()

    def _build_one_step(self, step: int, idxs: list[int]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
        futs_a = [
            self._pool.submit(
                build_unique_episode, self.store, idx,
                global_seed=self.global_seed, step=step, rank=self.rank, slot=slot, view_tag="a",
            )
            for slot, idx in enumerate(idxs)
        ]
        # Two paired independent split views for InfoNCE split invariance
        futs_b = [
            self._pool.submit(
                build_unique_episode, self.store, idxs[s],
                global_seed=self.global_seed, step=step, rank=self.rank, slot=s, view_tag="b",
                scm_prob=0.0, target_swap_prob=0.0,
            )
            for s in range(min(2, len(idxs)))
        ]
        eps_a = [f.result() for f in futs_a]
        eps_b = [f.result() for f in futs_b]
        return step, eps_a, eps_b

    def _run(self) -> None:
        pending: deque[Any] = deque()
        while not self._stop.is_set():
            while len(pending) < 4 and not self._stop.is_set():
                step = self._step
                self._step += 1
                if self.regret_path is not None and step % 50 == 0:
                    self.store.update_regret_from_file(self.regret_path)
                idxs = self.store.sample_unique_indices(self.batch_size, self._rng)
                pending.append(self._pool.submit(self._build_one_step, step, idxs))
            if pending:
                item = pending.popleft().result()
                while not self._stop.is_set():
                    try:
                        self._q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue

    def next_batch(self) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
        return self._q.get()

    def close(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False)
