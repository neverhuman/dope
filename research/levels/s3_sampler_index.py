"""Corpus index view and counter-hash primitives of the S3 counter sampler (plan M3, stage C1).

The sampler sees a corpus only through this module: the arena ``index.jsonl`` (formats.md
section 1) and, when present, the lineage ``families.jsonl`` (section 2). Nothing here opens a
feature, target or duplicate-group array. Every random choice of a plan is a blake2b key
stretched by splitmix64, so a plan needs no RNG object and is identical in every process.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .s3_dynamic_augment import MAX_CLASSES

TASKS = ("regression", "binary", "multiclass")
STRATA = (*TASKS, "target_swap", "scm_prior")
# File-name stems that say nothing about lineage: without families.jsonl each table is its own family.
GENERIC_STEMS = frozenset({"rows", "export", "unknown", "en", "csv_layers", "data", "train", "test", "dataset"})
M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
_MIX_A, _MIX_B = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


class SamplerError(ValueError):
    """Static refusal. The message is a code, never a source value."""


def key64(*parts: Any) -> int:
    """64-bit blake2b of a tagged tuple; the only source of randomness in a plan."""
    text = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8, person=b"dope-s3-plan/v1").digest(), "little")


def mix64(x: int) -> int:
    x &= M64
    x = ((x ^ (x >> 30)) * _MIX_A) & M64
    x = ((x ^ (x >> 27)) * _MIX_B) & M64
    return x ^ (x >> 31)


def unit(key: int, k: int) -> float:
    """k-th uniform in [0, 1) of the stream keyed by ``key`` (splitmix64, 53 bits)."""
    return (mix64(key + (k + 1) * GOLD) >> 11) * (1.0 / (1 << 53))


def hash_perm(key: int, n: int) -> np.ndarray:
    """Permutation of ``range(n)``: the order of splitmix64 keys. No numpy RNG involved."""
    x = np.arange(1, n + 1, dtype=np.uint64) * np.uint64(GOLD) + np.uint64(key & M64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(_MIX_A)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(_MIX_B)
    return np.argsort(x ^ (x >> np.uint64(31)), kind="stable")


class CorpusIndex:
    """Sampler's view of ``index.jsonl``: eligible train tables, their shapes and families."""

    def __init__(self, rows: Sequence[dict[str, Any]], families: dict[str, str] | None = None,
                 *, min_rows: int = 24, max_classes: int = MAX_CLASSES) -> None:
        n = len(rows)
        self.n_tables, self.min_rows, self.max_classes = n, int(min_rows), int(max_classes)
        self.dataset_hash = [str(rec["dataset_hash"]) for rec in rows]
        self.task = np.full(n, -1, dtype=np.int8)
        self.n_classes = np.zeros(n, dtype=np.int32)
        self.n_rows = np.zeros(n, dtype=np.int64)
        self.n_cols = np.zeros(n, dtype=np.int32)
        self.table_family = np.full(n, -1, dtype=np.int64)
        seen: dict[str, int] = {}
        for t, rec in enumerate(rows):
            if int(rec["table_id"]) != t:
                raise SamplerError("index_table_order")
            task = str(rec["task"])
            self.n_classes[t], self.n_rows[t], self.n_cols[t] = int(rec["n_classes"]), int(rec["n_rows"]), int(rec["n_cols"])
            if (rec["split"] != "train" or task not in TASKS or self.n_classes[t] > max_classes
                    or self.n_rows[t] < min_rows or self.n_cols[t] < 1):
                continue
            self.task[t] = TASKS.index(task)
            if families is not None:
                fid = families.get(self.dataset_hash[t])
                if fid is None:
                    raise SamplerError("families_missing_table")
            else:
                stem = str(rec["family_stem"])
                fid = f"{stem}#{self.dataset_hash[t]}" if stem in GENERIC_STEMS else stem
            self.table_family[t] = seen.setdefault(fid, len(seen))
        if not seen:
            raise SamplerError("index_no_train_tables")
        self.family_ids = list(seen)
        self.n_families = len(seen)
        self.eligible = np.flatnonzero(self.table_family >= 0)

    @classmethod
    def load(cls, arena: str | Path, families: str | Path | None = None, **limits: int) -> "CorpusIndex":
        """Read ``<arena>/index.jsonl`` (or that file itself) and an optional ``families.jsonl``."""
        path = Path(arena)
        path = path / "index.jsonl" if path.is_dir() else path
        with path.open("r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        lineage = None
        if families:
            with Path(families).open("r", encoding="utf-8") as fh:
                lineage = {rec["dataset_hash"]: str(rec["family_id"]) for rec in map(json.loads, filter(str.strip, fh))}
        return cls(rows, lineage, **limits)

    def members(self, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """CSR (ptr, table ids) of the eligible tables of each family, table id ascending."""
        tables = self.eligible if mask is None else self.eligible[mask[self.eligible]]
        fam = self.table_family[tables]
        ptr = np.concatenate([[0], np.cumsum(np.bincount(fam, minlength=self.n_families))])
        return ptr, tables[np.argsort(fam, kind="stable")]
