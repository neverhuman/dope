"""Vectorised view builder of the batched S3 training core (plan M3, stage C2).

``build_episode(corpus, spec, cfg)`` turns one ``EpisodeSpec`` of the counter sampler into the
episode dict the existing losses consume (the keys of ``s3_fast_pipeline.build_unique_episode``)
plus the identity of the view. It is a pure function of the spec, the config and the corpus:
no global RNG, clock, rank or worker count. Row and column choices are hashed orders; values
(noise, synthetic columns, prior tables) come from one ``numpy`` generator seeded by the view
seed, so the value stream is tied to the numpy version, the structure is not.

Order of operations:

1. source: the stored table, a target-swap view of it, or a synthetic prior table;
2. column subset;
3. Rule A-1 groups on that final view: rows equal on the selected columns (rounded to 1e-5)
   form one group. ``dup_key="row"`` also requires an equal target, the reading of Rule A-1
   used by the Rust arena; it is not the default because it lets rows with identical features
   and different targets sit on both sides (measured on s3-v250: 17 % of views, 13-37 % of
   their QUERY rows), which the stored feature-only groups never allowed;
4. group-respecting split at the spec ratio, then a prior-preserving subsample to
   ``n_fit`` / ``n_qry``;
5. raw augmentation: lattice-aware jitter, warps, MCAR / MAR missingness;
6. FIT-only scaling (``nc2_compat`` or ``robust_v2``) and the missing masks;
7. synthetic noise and derived columns, then one seeded column permutation of every array;
8. targets: standardised regression, binary 0/1, multiclass remapped to the classes of the FIT
   half under a seeded class permutation, or a synthetic multilabel view.

A view that cannot be built honestly (no group boundary fits the sizes, one class or a constant
target in the FIT half, no swappable column, too few scorable QUERY rows) is never repaired with
invented labels: ``build_episode`` returns ``RetryEpisode`` and the caller asks the sampler for
``attempt + 1``. ``build_slot`` runs that loop. There is no Python loop over table columns; the
only column loop runs over the handful of synthetic columns being appended.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .s3_counter_sampler import CounterSampler, EpisodeSpec, view_columns
from .s3_view_corpus import ViewCorpus, ViewError
from .s3_view_rows import plan_rows, row_hashes, signatures
from .s3_view_scale import raw_augment, scale_stacked
from .s3_view_sources import scm_prior_table, swap_target

__all__ = ["DEFAULT_VIEW", "RetryEpisode", "ViewConfig", "ViewError", "build_episode", "build_slot",
           "multilabel_targets", "synthetic_columns"]

SOURCE_KIND = {"real": "s3_real", "target_swap": "s3_target_swap", "scm_prior": "scm_prior"}
_F32 = np.float32


@dataclass(frozen=True)
class ViewConfig:
    """Plain data. Probabilities are per episode unless a name says otherwise."""

    scaler: str = "nc2_compat"  # or "robust_v2"
    # "scaled": real columns leave scaled by the FIT statistics. "raw": real columns leave with their
    # augmented raw values (non-finite -> NaN) for a native stage that fits its own FIT-only scale
    # card, as serving does. Splits, targets, synthetic columns and RNG streams are the same in both.
    emit: str = "scaled"
    dup_key: str = "features"  # Rule A-1 key on the view: "features", or "row" = features + target (see module doc)
    jitter_prob: float = 0.30
    jitter_lo: float = 0.002  # sigma as a share of the FIT IQR
    jitter_hi: float = 0.025
    lattice_max_distinct: int = 64
    lattice_max_grid: int = 4096
    lattice_probe_rows: int = 256  # FIT rows read by the lattice test and the jitter scale
    warp_prob: float = 0.25
    warp_col_prob: float = 0.35  # per continuous column of a warped episode
    miss_prob: float = 0.30
    binary_flip_prob: float = 0.50
    permute_classes: bool = True
    permute_columns: bool = True
    swap_max_distinct: int = 12
    prior_margin: float = 0.25  # extra rows of a multiclass prior table, as a share of n_qry (+ 32)


DEFAULT_VIEW = ViewConfig()


@dataclass(frozen=True)
class RetryEpisode:
    """The view could not be built; ``reason`` is a static code. Re-plan the slot with attempt + 1."""

    reason: str
    spec: EpisodeSpec


def synthetic_columns(z: np.ndarray, n_fit: int, n_noise: int, n_derived: int,
                      rng: np.random.Generator) -> list[np.ndarray]:
    """``n_noise`` distractors then ``n_derived`` derived columns for the scaled view ``z`` (p, rows).

    Noise: normal, uniform, heavy-tailed (t3) or +-1 Bernoulli, independent of everything.
    Derived: product, random linear combination, sine or hinge of two real columns, standardised
    with FIT statistics. The laws are those of ``s3_dynamic_augment._insert_synthetic_columns``.
    The loops run over these few synthetic columns (at most 10), never over table columns.
    """
    p, n = z.shape
    u = rng.random((2, n_noise)).tolist()
    out = []
    for j in range(n_noise):
        kind = int(u[0][j] * 4)
        if kind == 0:
            col = rng.standard_normal(n, dtype=_F32)
        elif kind == 1:
            col = rng.random(n, dtype=_F32) * _F32(3.46) - _F32(1.73)
        elif kind == 2:
            col = np.clip(rng.standard_t(3.0, n) / 1.5, -5.0, 5.0)
        else:
            col = np.where(rng.random(n, dtype=_F32) < 0.15 + 0.7 * u[1][j], _F32(1.0), _F32(-1.0))
        out.append(col)
    u = rng.random((5, n_derived)).tolist()
    for j in range(n_derived):
        c1, op = int(u[0][j] * p), int(u[2][j] * 4)
        a = z[c1]
        if op == 0:
            v = a * z[(c1 + 1 + int(u[1][j] * (p - 1))) % p]
        elif op == 1:
            v = (2.0 * u[3][j] - 1.0) * a + (2.0 * u[4][j] - 1.0) * z[(c1 + 1 + int(u[1][j] * (p - 1))) % p]
        elif op == 2:
            v = np.sin(1.5 * a)
        else:
            v = np.maximum(a, 0.0)  # the hinge sits at the FIT median, which is 0 for every scaled column
        head = v[:n_fit]  # a view: centred with the rest
        v -= head.sum() / n_fit
        v /= max(math.sqrt(float(np.dot(head, head)) / n_fit), 1e-5)
        out.append(np.clip(v, -5.0, 5.0))
    return out


def multilabel_targets(x: np.ndarray, n_fit: int, distractor: np.ndarray, n_labels: int,
                       rng: np.random.Generator) -> np.ndarray | None:
    """Correlated labels in {0, 1}^L driven by non-distractor columns (stacked rows, float32).

    The law of ``s3_dynamic_augment._synthesize_multilabel_targets`` with L fixed by the spec.
    ``None`` when a label stays constant in the FIT half.
    """
    real = np.flatnonzero(~distractor)
    if real.size < 2:
        real = np.arange(x.shape[1])
    k = min(real.size, 8)
    c = x[:, real[rng.permutation(real.size)[:k]]].astype(np.float64)
    w = (rng.standard_normal((k, n_labels)) + 0.6 * rng.standard_normal((k, 1))) / math.sqrt(k)
    lin = c @ w + 0.25 * np.tanh(c[:, 0] * c[:, 1 % k])[:, None]
    centre = np.median(lin[:n_fit], axis=0)
    prob = 1.0 / (1.0 + np.exp(-np.clip(2.2 * (lin - (centre + rng.uniform(-0.35, 0.35, n_labels))), -12.0, 12.0)))
    y = rng.random(prob.shape) < prob
    ones = y[:n_fit].sum(axis=0)
    thin = (ones < 2) | (ones > n_fit - 2)
    if thin.any():  # a label almost constant in FIT falls back to the deterministic median cut
        y[:, thin] = lin[:, thin] >= centre[thin]
        ones = y[:n_fit].sum(axis=0)
        if ((ones == 0) | (ones == n_fit)).any():
            return None
    return y.astype(_F32)


def build_episode(corpus: ViewCorpus, spec: EpisodeSpec, cfg: ViewConfig = DEFAULT_VIEW) -> dict[str, Any] | RetryEpisode:
    """One episode for ``spec``: a pure function of ``(spec, cfg, corpus)``."""
    seed, n_fit, n_qry = spec.view_seed, spec.n_fit, spec.n_qry
    multilabel = spec.task == "multilabel"
    target_col, valid, labels = -1, None, None
    if spec.kind == "scm_prior":
        base = "regression" if multilabel else spec.task
        spare = int(cfg.prior_margin * n_qry) + 32 if base == "multiclass" else 0
        view, y, labels = scm_prior_table(seed, n_fit + n_qry + spare, spec.p_view, base, spec.n_classes)
        cols, dataset_hash = np.arange(spec.p_view), f"scm_{seed:016x}"
    else:
        table = corpus.table(spec.table_id)
        src, dataset_hash = table["X"], table["dataset_hash"]
        if spec.kind == "target_swap":
            swap = swap_target(src, spec, cfg.swap_max_distinct)
            if isinstance(swap, str):
                return RetryEpisode(swap, spec)
            target_col, base, y, labels, valid = swap.column, swap.task, swap.y, swap.labels, swap.valid
            others = np.delete(np.arange(src.shape[1]), target_col)
            if spec.p_view > others.size:
                raise ViewError("spec_columns")
            cols = others[view_columns(seed, others.size, spec.p_view)]
            dataset_hash = f"{dataset_hash}_sw{target_col}"
        elif spec.kind == "real":
            base, y = table["task"], table["y"]
            if spec.p_view > src.shape[1] or (spec.task != base and not (multilabel and base == "regression")):
                raise ViewError("spec_table_mismatch")
            cols = view_columns(seed, src.shape[1], spec.p_view)
            if not np.isfinite(y).all():
                valid = np.isfinite(y)
            if base == "binary":
                labels = (y > 0.5).astype(np.int64)
            elif base == "multiclass":
                labels = np.rint(y if valid is None else np.where(valid, y, 0.0)).astype(np.int64)
                if int(labels.min()) < 0:
                    raise ViewError("class_index")
        else:
            raise ViewError("spec_kind")
        view = src if cols.size == src.shape[1] else np.take(src, cols, axis=1)

    # Rule A-1 on the final view, split, subsample. A multilabel view has no stored target to key on.
    if cfg.dup_key not in ("features", "row"):
        raise ViewError("dup_key")
    hashed = row_hashes(view, y if cfg.dup_key == "row" and not multilabel else None)
    plan = plan_rows(hashed, seed, n_fit, n_qry, spec.split_ratio, labels, valid)
    if isinstance(plan, str):
        return RetryEpisode(plan, spec)
    rows = np.concatenate([plan.fit, plan.qry])
    n = rows.size
    flipped, y_out = False, None
    rng = np.random.Generator(np.random.PCG64(seed))
    if multilabel:
        pass  # labels are synthesised from the finished view below; the stored target is not part of it
    elif base == "regression":
        target = np.take(y, rows).astype(np.float64)
        head = target[:n_fit]  # a view: centred together with the rest below
        if float(head.min()) == float(head.max()):
            return RetryEpisode("constant_target", spec)
        target -= head.sum() / n_fit
        target /= max(math.sqrt(float(np.dot(head, head)) / n_fit), 1e-5)
        y_out = np.clip(target, -6.0, 6.0).astype(_F32)
        task, n_classes = "regression", 1
    elif base == "binary":
        y_out = np.take(labels, rows).astype(_F32)
        flipped = bool(rng.random() < cfg.binary_flip_prob)
        if flipped:
            y_out = _F32(1.0) - y_out
        task, n_classes = "binary", 2
    else:
        present = np.flatnonzero(plan.fit_counts > 0)
        remap = np.full(plan.fit_counts.size, -1, dtype=np.int64)
        remap[present] = rng.permutation(present.size) if cfg.permute_classes else np.arange(present.size)
        y_out = remap[np.take(labels, rows)].astype(_F32)
        task, n_classes = "multiclass", int(present.size)

    # Values, column-major: (columns, rows) float64, FIT rows first.
    x = np.take(view, rows, axis=0).T.astype(np.float64, order="C")
    aug = raw_augment(x, n_fit, rng, cfg)
    if cfg.emit not in ("scaled", "raw"):
        raise ViewError("emit")
    raw = np.where(np.isfinite(x), x, np.nan) if cfg.emit == "raw" else None
    miss, constant = scale_stacked(x, n_fit, cfg.scaler)
    p, kn, kd = x.shape[0], spec.n_noise, spec.n_derived
    total = p + kn + kd
    order = rng.permutation(total) if cfg.permute_columns else np.arange(total)  # base column shown at each position
    place = np.empty(total, dtype=np.intp)
    place[order] = np.arange(total)
    stacked = np.empty((n, total), dtype=_F32)
    out = stacked.T  # (columns, rows) view of the row-major result
    out[place[:p]] = x
    if kn or kd:
        for position, column in zip(place[p:].tolist(), synthetic_columns(x, n_fit, kn, kd, rng)):
            out[position] = column
    source = np.empty(total, dtype=np.int64)  # source column, -1 noise, -2 derived
    source[place[:p]], source[place[p:p + kn]], source[place[p + kn:]] = cols, -1, -2
    is_const = np.zeros(total, dtype=bool)
    is_const[place[:p]] = constant
    missing = np.zeros((n, total), dtype=_F32)
    if miss is not None:
        missing.T[place[:p]] = miss
    distractor = source == -1
    if multilabel:
        y_out = multilabel_targets(stacked, n_fit, distractor, spec.n_classes, rng)
        if y_out is None:
            return RetryEpisode("constant_target", spec)
        task, n_classes = "multilabel", int(spec.n_classes)
    if raw is not None:  # after every draw that reads the scaled values
        out[place[:p]] = raw

    view_sig, split_sig = signatures(spec, cols, plan.fit, plan.qry, target_col)
    if plan.fit_counts is not None:  # class counts of the FIT pool and of the FIT half, by source class id
        aug["pool_counts"], aug["fit_counts"] = plan.pool_counts.tolist(), plan.fit_counts.tolist()
    return {
        "dataset_hash": dataset_hash, "family_stem": spec.family_id, "source_kind": SOURCE_KIND[spec.kind],
        "task": task, "n_classes": n_classes, "stage_id": 0,
        "X_fit": stacked[:n_fit], "y_fit": y_out[:n_fit], "miss_fit": missing[:n_fit],
        "X_qry": stacked[n_fit:], "y_qry": y_out[n_fit:], "miss_qry": missing[n_fit:],
        "distractor_mask": distractor, "split_sig": f"{split_sig:016x}", "view_sig": view_sig,
        "aug_stats": {
            "n_fit": n_fit, "n_qry": n_qry, "n_cols": total, "n_noise_cols": kn, "n_derived_cols": kd,
            "miss_rate": float(miss[:, :n_fit].mean()) * p / total if miss is not None else 0.0,
            "train_frac": round(spec.split_ratio, 3), "stage": 0,
            "fit_pool": plan.fit_pool, "qry_pool": plan.qry_pool, "dup_rows": plan.dup_rows,
            "qry_dropped": plan.qry_dropped, "const_cols": int(constant.sum()), "flipped": flipped, **aug,
        },
        "kind": spec.kind, "table_id": spec.table_id, "step": spec.step, "slot": spec.slot, "attempt": spec.attempt,
        "target_col": target_col, "cols": cols, "fit_rows": plan.fit, "qry_rows": plan.qry,
        "col_source": source, "constant_mask": is_const,
    }


def build_slot(corpus: ViewCorpus, sampler: CounterSampler, seed: int, spec: EpisodeSpec,
               cfg: ViewConfig = DEFAULT_VIEW, counters: Any = None, max_tries: int = 64,
               ) -> tuple[dict[str, Any], EpisodeSpec, list[str]]:
    """Build a slot, re-planning it through ``sampler.retry`` until a view is built.

    Returns (episode, the spec that was built, reasons of the failed attempts). ``counters`` is
    an optional ``HonestCounters``: failed attempts and the built view are reported to it.
    """
    reasons: list[str] = []
    for _ in range(max_tries):
        episode = build_episode(corpus, spec, cfg)
        if not isinstance(episode, RetryEpisode):
            if counters is not None:
                counters.observe_built(spec, episode["cols"], episode["fit_rows"], episode["qry_rows"], episode["target_col"])
            return episode, spec, reasons
        reasons.append(episode.reason)
        if counters is not None:
            counters.observe_failed(spec)
        spec = sampler.retry(seed, spec)
    raise ViewError("slot_never_built")
