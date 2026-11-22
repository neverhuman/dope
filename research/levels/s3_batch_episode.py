"""Episode batches for the batched S3 training core (plan M3, stage C2).

``collate`` pads a list of episode dicts (``s3_view_builder.build_episode`` or the legacy
``build_unique_episode``) into plain numpy arrays with masks; ``uncollate`` is its exact inverse.
There is no torch import: the training process converts with ``torch.from_numpy``.

One target layout serves all four tasks, ``Y`` of shape (B, rows, C):

==========  ==========================  =====
task        columns used                n_out
==========  ==========================  =====
regression  0: standardised value       1
binary      0: label in {0, 1}          1
multiclass  0..K-1: one-hot class       K
multilabel  0..L-1: label in {0, 1}     L
==========  ==========================  =====

Padding is zero everywhere and is described by four masks: ``row_fit`` / ``row_qry`` (real
rows), ``col`` (real or synthetic feature columns of the episode) and ``out_mask`` (target
columns in use). The bucket key is the sampler's: ``(C in {1, 8, 32}, ceil8(P), n_fit, n_qry)``.
Episodes of one exact bucket collate with no row padding; any other mix is legal and padded.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Sequence

import numpy as np

__all__ = ["BatchError", "EpisodeBatch", "TASK_IDS", "bucket_key", "collate", "out_bucket", "out_width", "uncollate"]

TASK_IDS = {"regression": 0, "binary": 1, "multiclass": 2, "multilabel": 3}
TASK_NAMES = tuple(TASK_IDS)
MAX_OUT = 32
_COLLATED = ("X_fit", "X_qry", "miss_fit", "miss_qry", "y_fit", "y_qry", "distractor_mask", "constant_mask",
             "task", "n_classes", "view_sig")


class BatchError(ValueError):
    """Static refusal of collate. The message is a code, never a value."""


def out_width(task: str, n_classes: int) -> int:
    """Target columns an episode really uses."""
    if task in ("regression", "binary"):
        return 1
    if task not in TASK_IDS or not 1 <= int(n_classes) <= MAX_OUT:
        raise BatchError("task_width")
    return int(n_classes)


def out_bucket(width: int) -> int:
    """Padded output width: 1, 8 or 32 (``EpisodeSpec.c_out``)."""
    return 1 if width <= 1 else 8 if width <= 8 else MAX_OUT


def bucket_key(episode: dict[str, Any]) -> tuple[int, int, int, int]:
    """(C, ceil8(P), n_fit, n_qry) of a built episode, on the grid of ``EpisodeSpec.bucket_key``."""
    n_fit, p = episode["X_fit"].shape
    return (out_bucket(out_width(episode["task"], episode["n_classes"])), -(-p // 8) * 8, n_fit,
            episode["X_qry"].shape[0])


@dataclass
class EpisodeBatch:
    """B padded episodes. Float arrays are float32, masks are bool; ``meta`` keeps everything else."""

    X_fit: np.ndarray  # (B, Nf, P)
    X_qry: np.ndarray  # (B, Nq, P)
    miss_fit: np.ndarray  # (B, Nf, P) 1.0 where the value was missing before imputation
    miss_qry: np.ndarray  # (B, Nq, P)
    Y_fit: np.ndarray  # (B, Nf, C)
    Y_qry: np.ndarray  # (B, Nq, C)
    row_fit: np.ndarray  # (B, Nf) real FIT rows
    row_qry: np.ndarray  # (B, Nq) real QUERY rows
    col: np.ndarray  # (B, P) columns of the episode
    out_mask: np.ndarray  # (B, C) target columns in use
    dmask: np.ndarray  # (B, P) noise distractor columns
    const_mask: np.ndarray  # (B, P) columns constant in the FIT half
    task_id: np.ndarray  # (B,) int8, index into TASK_NAMES
    n_out: np.ndarray  # (B,) int32
    n_classes: np.ndarray  # (B,) int32, as the losses expect it (1 regression, 2 binary, K, L)
    n_fit: np.ndarray  # (B,) int32
    n_qry: np.ndarray  # (B,) int32
    p_true: np.ndarray  # (B,) int32
    view_sig: np.ndarray  # (B,) uint64, 0 when the episode carries none
    split_sig: np.ndarray  # (B,) uint64: the first 64 bits of the episode's hex ``split_sig``
    meta: list[dict[str, Any]]  # per episode: the keys that are not arrays above

    def __len__(self) -> int:
        return int(self.task_id.size)

    def arrays(self) -> dict[str, np.ndarray]:
        """Every array field by name (for ``torch.from_numpy`` or shared memory)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "meta"}


def _targets(y: np.ndarray, task: str, width: int, rows: int) -> np.ndarray:
    """(rows, width) float32 block of one half in the common layout."""
    if task == "multiclass":
        cls = np.asarray(y).astype(np.intp)
        if cls.shape != (rows,) or (cls.size and (int(cls.min()) < 0 or int(cls.max()) >= width)):
            raise BatchError("class_index")
        block = np.zeros((rows, width), dtype=np.float32)
        block[np.arange(rows), cls] = 1.0
        return block
    block = np.asarray(y, dtype=np.float32).reshape(rows, -1)
    if block.shape[1] != width:
        raise BatchError("target_shape")
    return block


def collate(episodes: Sequence[dict[str, Any]], n_fit: int | None = None, n_qry: int | None = None,
            p: int | None = None, c: int | None = None) -> EpisodeBatch:
    """Pad ``episodes`` to common shapes.

    Defaults: the largest ``n_fit`` and ``n_qry`` of the list, the largest column count rounded
    up to a multiple of 8, and the output bucket (1, 8 or 32) of the widest target. An explicit
    size must not be smaller than what the episodes need.
    """
    if not episodes:
        raise BatchError("empty_batch")
    b = len(episodes)
    fits = np.array([ep["X_fit"].shape[0] for ep in episodes], dtype=np.int32)
    qrys = np.array([ep["X_qry"].shape[0] for ep in episodes], dtype=np.int32)
    cols = np.array([ep["X_fit"].shape[1] for ep in episodes], dtype=np.int32)
    outs = np.array([out_width(ep["task"], ep["n_classes"]) for ep in episodes], dtype=np.int32)
    nf = int(fits.max()) if n_fit is None else int(n_fit)
    nq = int(qrys.max()) if n_qry is None else int(n_qry)
    pp = -(-int(cols.max()) // 8) * 8 if p is None else int(p)
    cc = out_bucket(int(outs.max())) if c is None else int(c)
    if nf < fits.max() or nq < qrys.max() or pp < cols.max() or cc < outs.max():
        raise BatchError("pad_too_small")
    f32 = np.float32
    out = EpisodeBatch(
        X_fit=np.zeros((b, nf, pp), f32), X_qry=np.zeros((b, nq, pp), f32),
        miss_fit=np.zeros((b, nf, pp), f32), miss_qry=np.zeros((b, nq, pp), f32),
        Y_fit=np.zeros((b, nf, cc), f32), Y_qry=np.zeros((b, nq, cc), f32),
        row_fit=np.arange(nf) < fits[:, None], row_qry=np.arange(nq) < qrys[:, None],
        col=np.arange(pp) < cols[:, None], out_mask=np.arange(cc) < outs[:, None],
        dmask=np.zeros((b, pp), bool), const_mask=np.zeros((b, pp), bool),
        task_id=np.array([TASK_IDS[ep["task"]] for ep in episodes], dtype=np.int8), n_out=outs,
        n_classes=np.array([ep["n_classes"] for ep in episodes], dtype=np.int32),
        n_fit=fits, n_qry=qrys, p_true=cols,
        view_sig=np.array([ep.get("view_sig", 0) for ep in episodes], dtype=np.uint64),
        split_sig=np.array([int(str(ep.get("split_sig", "0"))[:16], 16) for ep in episodes], dtype=np.uint64),
        meta=[{k: v for k, v in ep.items() if k not in _COLLATED} for ep in episodes],
    )
    for i, ep in enumerate(episodes):
        f, q, k, w = int(fits[i]), int(qrys[i]), int(cols[i]), int(outs[i])
        if ep["X_qry"].shape[1] != k or ep["distractor_mask"].shape != (k,):
            raise BatchError("column_mismatch")
        out.X_fit[i, :f, :k], out.X_qry[i, :q, :k] = ep["X_fit"], ep["X_qry"]
        if "miss_fit" in ep:
            out.miss_fit[i, :f, :k], out.miss_qry[i, :q, :k] = ep["miss_fit"], ep["miss_qry"]
        out.Y_fit[i, :f, :w] = _targets(ep["y_fit"], ep["task"], w, f)
        out.Y_qry[i, :q, :w] = _targets(ep["y_qry"], ep["task"], w, q)
        out.dmask[i, :k] = ep["distractor_mask"]
        if "constant_mask" in ep:
            out.const_mask[i, :k] = ep["constant_mask"]
    return out


def uncollate(batch: EpisodeBatch) -> list[dict[str, Any]]:
    """The episodes ``collate`` was given: same keys, same dtypes, same values (arrays are copies).

    ``miss_*``, ``constant_mask`` and ``view_sig`` come back for every episode, zero-filled when
    the input episode did not carry them.
    """
    episodes = []
    for i, meta in enumerate(batch.meta):
        f, q, k, w = int(batch.n_fit[i]), int(batch.n_qry[i]), int(batch.p_true[i]), int(batch.n_out[i])
        task = TASK_NAMES[int(batch.task_id[i])]
        halves = []
        for block in (batch.Y_fit[i, :f, :w], batch.Y_qry[i, :q, :w]):
            if task == "multiclass":
                halves.append(block.argmax(axis=1).astype(np.float32))
            else:
                halves.append(block.copy() if task == "multilabel" else block[:, 0].copy())
        episodes.append({
            **meta, "task": task, "n_classes": int(batch.n_classes[i]),
            "X_fit": batch.X_fit[i, :f, :k].copy(), "y_fit": halves[0], "miss_fit": batch.miss_fit[i, :f, :k].copy(),
            "X_qry": batch.X_qry[i, :q, :k].copy(), "y_qry": halves[1], "miss_qry": batch.miss_qry[i, :q, :k].copy(),
            "distractor_mask": batch.dmask[i, :k].copy(), "constant_mask": batch.const_mask[i, :k].copy(),
            "view_sig": int(batch.view_sig[i]),
        })
    return episodes
