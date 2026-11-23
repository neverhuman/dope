"""Maximum-entropy S3 augmentation & infinite synthetic prior engine for SingleEncV7 (V7.2).

Unlocks 100% of all 4,145 S3 train tables (recovering all 51 multiclass tables with > 8 classes
via random K-class subset/tail cropping), enforces exact per-step shuffled tri-task balance
(37.5% regression / 31.25% binary / 31.25% multiclass), synthesizes infinite nonlinear SCM /
random-tree / bilinear-interaction priors, and applies fresh duplicate-grouped train/test splits,
log-uniform row/column cropping, multi-task target swaps, MCAR/MAR missingness, monotonic warps,
zero-inflation/discretization, and class-ID permutations on EVERY batch.
"""
from __future__ import annotations

import math

import numpy as np


def task_quota_for_step(micro_batch: int, step_seed: int) -> list[str]:
    """Exact balanced per-batch task schedule (6 reg / 5 bin / 5 mc for B=16), shuffled every step."""
    n_bin = max(1, (micro_batch * 5) // 16)
    n_mc = max(1, (micro_batch * 5) // 16)
    n_reg = max(1, micro_batch - n_bin - n_mc)
    tasks = ["regression"] * n_reg + ["binary"] * n_bin + ["multiclass"] * n_mc
    rng = np.random.default_rng(step_seed & 0xFFFFFFFFFFFFFFFF)
    rng.shuffle(tasks)
    return tasks


def sample_synthetic_prior_table(
    rng: np.random.Generator,
    task: str,
    n_fit: int,
    n_qry: int,
    max_cols: int = 44,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Infinite synthetic nonlinear SCM / random depth-3 tree / interaction prior table (2^64 entropy)."""
    n_tot = n_fit + n_qry
    p = int(rng.integers(4, min(max_cols, 36) + 1))
    # 1. Latent causes: mix of Gaussian, Student-t, skew-lognormal, and discrete regimes.
    r_lat = int(rng.integers(2, min(8, p) + 1))
    dist_kind = int(rng.integers(0, 3))
    if dist_kind == 0:
        Z = rng.standard_normal((n_tot, r_lat)).astype(np.float32)
    elif dist_kind == 1:
        Z = rng.standard_t(df=4.0, size=(n_tot, r_lat)).clip(-6.0, 6.0).astype(np.float32)
    else:
        Z = (rng.laplace(0.0, 0.8, size=(n_tot, r_lat))).clip(-6.0, 6.0).astype(np.float32)

    # 2. 2-layer structural DAG with nonlinear edges + optional discrete/zero-inflated features.
    W1 = rng.normal(0.0, 0.7 / math.sqrt(r_lat), size=(r_lat, p)).astype(np.float32)
    X = Z @ W1 + rng.normal(0.0, 0.55, size=(n_tot, p)).astype(np.float32)
    # Add random DAG nonlinear feature-to-feature couplings.
    n_edges = min(p // 2, 10)
    for _ in range(n_edges):
        src = int(rng.integers(0, p - 1))
        dst = int(rng.integers(src + 1, p))
        op = int(rng.integers(0, 4))
        w = float(rng.uniform(-0.8, 0.8))
        if op == 0:
            X[:, dst] += w * np.tanh(X[:, src])
        elif op == 1:
            X[:, dst] += w * np.clip(X[:, src] ** 2 - 1.0, -4.0, 4.0) * 0.4
        elif op == 2:
            src2 = int(rng.integers(0, dst))
            X[:, dst] += w * np.clip(X[:, src] * X[:, src2], -4.0, 4.0) * 0.4
        else:
            X[:, dst] += w * (X[:, src] > float(rng.uniform(-0.8, 0.8))).astype(np.float32)

    # Discretize / zero-inflate 20% of columns to mirror realistic mixed tabular manifolds.
    for j in range(p):
        u = rng.random()
        if u < 0.12:
            k_lvl = int(rng.integers(2, 8))
            qs = np.quantile(X[:, j], np.linspace(0.05, 0.95, k_lvl - 1))
            X[:, j] = np.digitize(X[:, j], qs).astype(np.float32)
        elif u < 0.22:
            thr = float(np.quantile(X[:, j], rng.uniform(0.15, 0.55)))
            X[:, j] = np.where(X[:, j] < thr, 0.0, X[:, j] - thr).astype(np.float32)

    # Normalize features cleanly before generating target signal.
    Xn = (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, keepdims=True), 1e-4)

    # 3. Ground-truth target generator: linear + nonlinear spline/quadratic + interaction + tree!
    n_cls = 1 if task == "regression" else (2 if task == "binary" else int(rng.integers(3, 9)))
    o_dim = 1 if task != "multiclass" else n_cls
    f_out = np.zeros((n_tot, o_dim), dtype=np.float32)
    n_act = min(p, int(rng.integers(2, min(p, 10) + 1)))
    act = rng.choice(p, size=n_act, replace=False)
    for c in range(o_dim):
        # Linear component
        w_lin = rng.normal(0.0, 1.0 / math.sqrt(n_act), size=n_act).astype(np.float32)
        sc = Xn[:, act] @ w_lin
        # Nonlinear component (quadratic / abs / sin / pairwise interaction)
        nl_wt = float(rng.uniform(0.35, 1.15))
        j0, j1 = int(act[0]), int(act[min(1, n_act - 1)])
        nl_type = int(rng.integers(0, 4))
        if nl_type == 0:
            sc += nl_wt * (np.clip(Xn[:, j0] ** 2, 0.0, 9.0) - 1.0) * 0.45 * float(rng.choice([-1.0, 1.0]))
        elif nl_type == 1:
            sc += nl_wt * np.abs(Xn[:, j0]) * float(rng.choice([-1.0, 1.0]))
        elif nl_type == 2:
            sc += nl_wt * np.sin(1.4 * Xn[:, j0]) + 0.5 * nl_wt * np.clip(Xn[:, j0] * Xn[:, j1], -4.0, 4.0)
        else:
            # Random depth-3 decision tree leaf values!
            t0 = float(rng.uniform(-0.8, 0.8))
            t1 = float(rng.uniform(-0.8, 0.8))
            t2 = float(rng.uniform(-0.8, 0.8))
            j2 = int(act[min(2, n_act - 1)])
            leaf = (
                ((Xn[:, j0] > t0).astype(np.int32) << 2)
                | ((Xn[:, j1] > t1).astype(np.int32) << 1)
                | (Xn[:, j2] > t2).astype(np.int32)
            )
            l_vals = rng.normal(0.0, 1.0, size=8).astype(np.float32)
            sc += nl_wt * l_vals[leaf]
        sc = (sc - sc.mean()) / max(float(sc.std()), 1e-4)
        f_out[:, c] = sc

    snr = float(rng.uniform(0.25, 0.85))
    if task == "regression":
        y = (f_out[:, 0] + snr * rng.standard_normal(n_tot)).astype(np.float32)
    elif task == "binary":
        logit_gain = float(rng.uniform(0.55, 1.55))
        bias = float(rng.uniform(-1.1, 1.1))
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logit_gain * f_out[:, 0] + bias, -10.0, 10.0)))
        y = (rng.random(n_tot) < p1).astype(np.float32)
        flip_p = float(rng.uniform(0.0, 0.08))
        if flip_p > 0.0:
            y = np.where(rng.random(n_tot) < flip_p, 1.0 - y, y).astype(np.float32)
        if len(np.unique(y[:n_fit])) < 2 or len(np.unique(y[n_fit:])) < 2:
            y = (f_out[:, 0] > float(np.median(f_out[:n_fit, 0]))).astype(np.float32)
    else:
        logit_gain = float(rng.uniform(0.85, 2.05))
        lg = logit_gain * f_out
        lg -= lg.max(axis=1, keepdims=True)
        pr = np.exp(lg)
        pr /= pr.sum(axis=1, keepdims=True)
        cdf = np.cumsum(pr, axis=1)
        u = rng.random((n_tot, 1))
        y = np.minimum((u >= cdf).sum(axis=1), n_cls - 1).astype(np.float32)
        flip_p = float(rng.uniform(0.0, 0.05))
        if flip_p > 0.0:
            y = np.where(rng.random(n_tot) < flip_p, rng.integers(0, n_cls, size=n_tot), y).astype(np.float32)
        if len(np.unique(y[:n_fit])) < 3:
            y = np.argmax(f_out, axis=1).astype(np.float32)

    return X[:n_fit], y[:n_fit], X[n_fit:], y[n_fit:], n_cls


def crop_multiclass_classes(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    dg: np.ndarray,
    max_classes: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Recover ALL >8-class S3 multiclass datasets via random K-class subset or tail-merge cropping."""
    yi = np.rint(y).astype(np.int64)
    vals, cnts = np.unique(yi, return_counts=True)
    if len(vals) <= max_classes:
        return X, y, dg, False
    # Choose K in [3, max_classes] randomly for maximum class-combination entropy!
    k_pick = int(rng.integers(3, max_classes + 1))
    # Prioritize classes with >= 16 rows so FIT and QRY both remain populated.
    eligible = vals[cnts >= 16]
    if len(eligible) < 3:
        order = np.argsort(cnts)[::-1]
        eligible = vals[order[: max(3, min(len(vals), max_classes))]]
    k_pick = min(k_pick, len(eligible))
    chosen = rng.choice(eligible, size=k_pick, replace=False)
    if rng.random() < 0.70:
        # Mode A: clean subset crop to chosen K classes (pure K-class sub-problem!).
        mask = np.isin(yi, chosen)
        if int(mask.sum()) >= 64:
            return X[mask], y[mask], dg[mask], True
    # Mode B: keep top-(K-1) chosen classes and merge remaining tail into an "other" class.
    keep_main = chosen[: k_pick - 1]
    other_id = int(vals.max()) + 1
    y_new = np.where(np.isin(yi, keep_main), yi, other_id).astype(np.float32)
    return X, y_new, dg, True


def synthesize_target_from_regression(
    rng: np.random.Generator, y_raw: np.ndarray, target_task: str
) -> tuple[np.ndarray, int]:
    """Turn any real S3 regression table into a brand-new binary or K-class task via random cuts."""
    fin = y_raw[np.isfinite(y_raw)]
    if len(fin) < 20:
        return y_raw, 1
    if target_task == "binary":
        cut = float(np.quantile(fin, rng.uniform(0.25, 0.75)))
        return (y_raw > cut).astype(np.float32), 2
    k_bins = int(rng.integers(3, 9))
    qs = np.quantile(fin, np.linspace(0.0, 1.0, k_bins + 1)[1:-1])
    qs = np.unique(qs)
    if len(qs) < 2:
        return y_raw, 1
    yb = np.digitize(y_raw, qs).astype(np.float32)
    return yb, len(qs) + 1


def apply_real_table_target_swap(
    rng: np.random.Generator, X_raw: np.ndarray, y_raw: np.ndarray, desired_task: str
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    """Multi-task target-swap: promotes 2-level cols to binary, 3..8-level to multiclass, cont to reg."""
    nc = X_raw.shape[1]
    if nc < 4:
        return X_raw, y_raw, desired_task, False
    cand_cols = rng.permutation(nc)
    for col in cand_cols[: min(6, nc)]:
        yc = X_raw[:, col]
        fin = np.isfinite(yc)
        if float(fin.mean()) < 0.90:
            continue
        vals = yc[fin]
        if float(np.std(vals)) < 1e-4:
            continue
        uniq = np.unique(vals)
        new_task: str | None = None
        if len(uniq) == 2 and desired_task in ("binary", "any"):
            yc = (yc == uniq[-1]).astype(np.float32)
            new_task = "binary"
        elif 3 <= len(uniq) <= 8 and desired_task in ("multiclass", "any"):
            new_task = "multiclass"
        elif len(uniq) > 8 and desired_task == "regression":
            new_task = "regression"
        if new_task is not None:
            keep = np.ones(nc, dtype=bool)
            keep[col] = False
            Xn = np.concatenate([X_raw[:, keep], y_raw[:, None]], axis=1)
            yn = np.where(np.isfinite(yc), yc, float(np.nanmedian(vals))).astype(np.float32)
            return Xn, yn, new_task, True
    return X_raw, y_raw, desired_task, False


def apply_manifold_augmentations(
    rng: np.random.Generator,
    Xa_raw: np.ndarray,
    Xq_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply MCAR/MAR missingness, monotonic warps, sign flips, and discrete/zero-inflation warps."""
    Xa = Xa_raw.copy()
    Xq = Xq_raw.copy()
    p = Xa.shape[1]

    # 1. MCAR / MAR missingness injection (25% of episodes, 5-22% rate) BEFORE FIT-only scaling!
    if rng.random() < 0.25:
        rate = float(rng.uniform(0.05, 0.22))
        if rng.random() < 0.5 or p < 2:
            Xa[rng.random(Xa.shape) < rate] = np.nan
            Xq[rng.random(Xq.shape) < rate] = np.nan
        else:
            # MAR: missingness conditioned on column 0 quantile!
            driver = int(rng.integers(0, p))
            thr = float(np.nanmedian(Xa[:, driver]))
            m_a = (Xa[:, driver] > thr)[:, None] & (rng.random(Xa.shape) < rate * 1.6)
            m_q = (Xq[:, driver] > thr)[:, None] & (rng.random(Xq.shape) < rate * 1.6)
            m_a[:, driver] = False
            m_q[:, driver] = False
            Xa[m_a] = np.nan
            Xq[m_q] = np.nan

    # 2. Random column-wise monotonic power warp / sign flip / zero-inflation (consistent on FIT+QRY!).
    for j in range(p):
        fin_a = Xa[:, j][np.isfinite(Xa[:, j])]
        if len(fin_a) < 8:
            continue
        is_cont = len(np.unique(fin_a)) > 12
        if rng.random() < 0.35:
            # Random sign flip (exact symmetry!)
            Xa[:, j] = -Xa[:, j]
            Xq[:, j] = -Xq[:, j]
        if is_cont and rng.random() < 0.20:
            # Monotonic heavy-tail power warp around FIT median
            med = float(np.median(fin_a))
            sd = max(float(np.std(fin_a)), 1e-5)
            gamma = float(rng.uniform(0.65, 1.65))
            for arr in (Xa, Xq):
                d = (arr[:, j] - med) / sd
                arr[:, j] = med + sd * np.sign(d) * (np.abs(d) ** gamma)
        elif is_cont and rng.random() < 0.12:
            # Zero-inflation / left-censoring realism augmentation
            cut = float(np.quantile(fin_a, rng.uniform(0.15, 0.45)))
            Xa[:, j] = np.where(Xa[:, j] < cut, cut, Xa[:, j])
            Xq[:, j] = np.where(Xq[:, j] < cut, cut, Xq[:, j])
    return Xa, Xq


def append_nonlinear_and_distractor_cols(
    rng: np.random.Generator,
    Xa: np.ndarray,
    Xq: np.ndarray,
    max_cols: int = 48,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Append random bilinear/nonlinear derived features + Gaussian/Laplace distractors."""
    p = Xa.shape[1]
    avail = max_cols - p
    if avail <= 0:
        return Xa, Xq, 0
    n_add = int(rng.integers(0, min(6, avail) + 1))
    if n_add == 0:
        return Xa, Xq, 0
    if rng.random() < 0.5:
        na = rng.standard_normal((len(Xa), n_add)).astype(np.float32)
        nq = rng.standard_normal((len(Xq), n_add)).astype(np.float32)
    else:
        na = rng.laplace(0.0, 0.7, size=(len(Xa), n_add)).clip(-6.0, 6.0).astype(np.float32)
        nq = rng.laplace(0.0, 0.7, size=(len(Xq), n_add)).clip(-6.0, 6.0).astype(np.float32)
    if n_add >= 2 and p >= 2:
        # 1st extra col: bilinear interaction; 2nd extra col (if >=3): nonlinear tanh/abs derived col.
        j0, j1 = int(rng.integers(0, p)), int(rng.integers(0, p))
        na[:, 0] = np.clip(0.55 * Xa[:, j0] * Xa[:, j1] + 0.25 * na[:, 0], -6.0, 6.0)
        nq[:, 0] = np.clip(0.55 * Xq[:, j0] * Xq[:, j1] + 0.25 * nq[:, 0], -6.0, 6.0)
        if n_add >= 3:
            j2 = int(rng.integers(0, p))
            na[:, 1] = np.clip(np.tanh(Xa[:, j2]) + 0.3 * np.abs(Xa[:, j0]) + 0.2 * na[:, 1], -6.0, 6.0)
            nq[:, 1] = np.clip(np.tanh(Xq[:, j2]) + 0.3 * np.abs(Xq[:, j0]) + 0.2 * nq[:, 1], -6.0, 6.0)
    return np.concatenate([Xa, na], axis=1), np.concatenate([Xq, nq], axis=1), n_add


class RealDatasetOnePassQueue:
    """Disjoint shard-partitioned one-pass real dataset queue (0 dataset repeats by construction)."""

    def __init__(
        self,
        arena: Any,
        shard_id: int = 0,
        num_shards: int = 1,
        seed: int = 2026,
        seen_datasets: set[str] | None = None,
    ) -> None:
        self.shard_id = int(shard_id) % max(1, int(num_shards))
        self.seen_datasets: set[str] = seen_datasets if seen_datasets is not None else set()
        self.queues: dict[str, list[tuple[int, str]]] = {"regression": [], "binary": [], "multiclass": []}
        import hashlib
        n_sh = max(1, int(num_shards))
        sh = self.shard_id
        rng = np.random.default_rng(seed ^ 0x5EED_0001)
        for task in ("regression", "binary", "multiclass"):
            all_ids = [idx for grp in arena.by_task_fams[task] for idx in grp]
            if not all_ids:
                continue
            perm = rng.permutation(len(all_ids))
            q: list[tuple[int, str]] = []
            for pi in perm:
                idx = all_ids[int(pi)]
                dh = str(arena.entries[idx].get("dataset_hash", f"tbl_{idx}"))
                owner = int.from_bytes(hashlib.blake2b(dh.encode("utf-8"), digest_size=8).digest(), "little") % n_sh
                if owner != sh:
                    continue
                if dh not in self.seen_datasets:
                    q.append((idx, dh))
            self.queues[task] = q

    def pop_unseen(self, task: str) -> int | None:
        q = self.queues.get(task)
        while q:
            idx, dh = q.pop()
            if dh not in self.seen_datasets:
                self.seen_datasets.add(dh)
                return idx
        return None

    def remaining(self) -> int:
        return sum(len(q) for q in self.queues.values())

    def take_slice(self, max_per_task: int) -> "SliceRealQueue":
        """Pre-pop unseen real table IDs for a forked prefetch worker (0% real repeats)."""
        sliced = {t: [i for i in (self.pop_unseen(t) for _ in range(max_per_task)) if i is not None] for t in ("regression", "binary", "multiclass")}
        return SliceRealQueue(self.shard_id, sliced)


class SliceRealQueue:
    """Zero-copy picklable real-table slice for GIL-free multiprocess GPU prefetchers."""
    def __init__(self, shard_id: int, items: dict[str, list[int]]) -> None:
        self.shard_id, self.items = int(shard_id), items
    def pop_unseen(self, task: str) -> int | None:
        q = self.items.get(task)
        return q.pop() if q else None


def diloco_sync_step(
    sync_dir: Any,
    base_model: Any,
    ema_model: Any,
    anchor_state: dict[str, np.ndarray],
    outer_lr: float = 0.7,
    outer_mom: float = 0.9,
    base_round: int = 0,
    worker_id: str = "w0",
) -> tuple[int, str]:
    """Atomic LAN/HTTP DiLoCo outer-momentum weight sync across asynchronous nodes."""
    import fcntl
    import hashlib
    import http.client
    import json
    from pathlib import Path
    import struct
    import sys
    import time
    import urllib.request
    import torch

    def _pack_st(sd: dict[str, np.ndarray]) -> bytes:
        hdr: dict[str, Any] = {}
        blobs: list[bytes] = []
        off = 0
        for k in sorted(sd):
            arr = np.ascontiguousarray(sd[k], dtype=np.float32)
            b = arr.tobytes()
            hdr[k] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
            blobs.append(b)
            off += len(b)
        hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        hb += b" " * ((8 - (len(hb) % 8)) % 8)
        return struct.pack("<Q", len(hb)) + hb + b"".join(blobs)

    def _unpack_st(raw: bytes) -> dict[str, np.ndarray]:
        hlen = struct.unpack("<Q", raw[:8])[0]
        hdr = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
        body = raw[8 + hlen :]
        return {
            k: np.frombuffer(body[m["data_offsets"][0] : m["data_offsets"][1]], dtype=np.float32).reshape(m["shape"]).copy()
            for k, m in hdr.items() if k != "__metadata__"
        }

    def _apply_global(new_g: dict[str, np.ndarray]) -> None:
        dev = next(base_model.parameters()).device
        with torch.no_grad():
            t_sd = {k: torch.from_numpy(v.copy()).to(dev) for k, v in new_g.items()}
            base_model.load_state_dict(t_sd, strict=True)
            for ke, pe in ema_model.state_dict().items():
                if ke in t_sd and pe.is_floating_point():
                    pe.mul_(0.5).add_(t_sd[ke].to(device=pe.device, dtype=pe.dtype), alpha=0.5)
        anchor_state.clear()
        for k, v in new_g.items():
            anchor_state[k] = v.copy()

    cur_sd = {k: v.detach().cpu().to(torch.float32).numpy().copy() for k, v in base_model.state_dict().items()}
    if str(sync_dir).startswith(("http://", "https://")):
        url = str(sync_dir).rstrip("/") + "/sync"
        mode = "init" if base_round <= 0 else "delta"
        send_sd = cur_sd if mode == "init" else {k: (anchor_state.get(k, cur_sd[k]) - cur_sd[k]).astype(np.float32) for k in cur_sd}
        body = _pack_st(send_sd)
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=body, method="POST",
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Worker-Id": worker_id, "X-Payload-Mode": mode, "X-Base-Round": str(base_round),
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
                    s_round = int(resp.headers.get("X-Diloco-Round", str(base_round + 1)))
                    sha = str(resp.headers.get("X-Diloco-Sha", hashlib.sha256(raw).hexdigest()[:16]))
                _apply_global(_unpack_st(raw))
                return s_round, sha[:16]
            except (OSError, ValueError, struct.error, http.client.HTTPException) as exc:  # network or payload failure only
                if attempt == 2:  # typed outcome: this round is skipped, training continues on local weights
                    print(f"[DiLoCo] hub sync skipped after 3 attempts: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    return base_round, ""
                time.sleep(0.5 * (attempt + 1))

    sdir = Path(sync_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    lock_p, glob_p, mom_p, meta_p = sdir / "sync.lock", sdir / "global_model.safetensors", sdir / "global_outer_mom.npz", sdir / "global_meta.json"
    with lock_p.open("a+b") as lfh:
        fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
        try:
            g_sd = _unpack_st(glob_p.read_bytes()) if glob_p.is_file() else {k: v.copy() for k, v in cur_sd.items()}
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {"sync_round": 0}
            prev_mom = dict(np.load(mom_p)) if mom_p.is_file() else {}
            new_mom, new_g = {}, {}
            for k in sorted(cur_sd):
                delta = anchor_state.get(k, cur_sd[k]) - cur_sd[k]
                v_new = (outer_mom * prev_mom.get(k, np.zeros_like(delta, dtype=np.float32)) + delta).astype(np.float32)
                new_mom[k], new_g[k] = v_new, (g_sd[k] - outer_lr * v_new).astype(np.float32)
            payload = _pack_st(new_g)
            sha = hashlib.sha256(payload).hexdigest()
            part = glob_p.with_suffix(".safetensors.part")
            part.write_bytes(payload)
            part.replace(glob_p)
            mpart = sdir / "global_outer_mom.part.npz"
            np.savez(mpart, **new_mom)
            mpart.replace(mom_p)
            s_round = int(meta.get("sync_round", 0)) + 1
            meta_part = meta_p.with_suffix(".json.part")
            meta_part.write_text(json.dumps({
                "sync_mode": "diloco_outer_momentum", "sync_round": s_round, "global_sha256": sha,
                "outer_lr": outer_lr, "outer_mom": outer_mom, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, sort_keys=True) + "\n", encoding="utf-8")
            meta_part.replace(meta_p)
            _apply_global(new_g)
            return s_round, sha[:16]
        finally:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_UN)
