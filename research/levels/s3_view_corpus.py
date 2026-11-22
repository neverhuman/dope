"""Corpus accessors for the vectorised S3 view builder (plan M3, stage C2).

The view builder sees a corpus only through ``table(table_id)``, which returns zero-copy,
read-only views::

    {"X": (n, p) float32, "y": (n,) float32, "dup": (n,) int32,
     "task": str, "n_classes": int, "dataset_hash": str, "family_id": str}

``ArenaCorpus`` maps the flat arena of ``docs/plans/v2-gap-closure/formats.md`` section 1
(``index.jsonl``, ``X.f32``, ``y.f32``, ``dup.i32``) with ``numpy.memmap``; pages are shared by
every worker process and nothing is copied until a view gathers its rows. ``MemoryCorpus`` holds
the same tables in RAM for tests, and ``write_flat_arena`` writes them in the arena layout.
Errors carry a static code, never a source value.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

__all__ = ["ArenaCorpus", "MemoryCorpus", "ViewCorpus", "ViewError", "write_flat_arena"]


class ViewError(ValueError):
    """Static refusal of the view builder. The message is a code, never a source value."""


class ViewCorpus(Protocol):
    def table(self, table_id: int) -> dict[str, Any]: ...


class ArenaCorpus:
    """Read-only memory map of a ``dope.flat-corpus/v1`` arena."""

    def __init__(self, arena: str | Path) -> None:
        root = Path(arena)
        with (root / "index.jsonl").open("r", encoding="utf-8") as fh:
            self.rows = [json.loads(line) for line in fh if line.strip()]
        if any(int(rec["table_id"]) != t for t, rec in enumerate(self.rows)):
            raise ViewError("arena_table_order")
        self.n_rows = np.array([rec["n_rows"] for rec in self.rows], dtype=np.int64)
        self.n_cols = np.array([rec["n_cols"] for rec in self.rows], dtype=np.int64)
        self.x_off = np.array([rec["x_off"] for rec in self.rows], dtype=np.int64)
        self.row_off = np.array([rec["row_off"] for rec in self.rows], dtype=np.int64)
        cells, total = int((self.n_rows * self.n_cols).sum()), int(self.n_rows.sum())
        self._maps = (np.memmap(root / "X.f32", dtype="<f4", mode="r"), np.memmap(root / "y.f32", dtype="<f4", mode="r"),
                      np.memmap(root / "dup.i32", dtype="<i4", mode="r"))
        if self._maps[0].size != cells or self._maps[1].size != total or self._maps[2].size != total:
            raise ViewError("arena_size")
        # Base-class views of the maps: same pages, but slicing skips the memmap subclass overhead.
        self._x, self._y, self._dup = (np.asarray(m) for m in self._maps)

    def __len__(self) -> int:
        return len(self.rows)

    def table(self, table_id: int) -> dict[str, Any]:
        rec = self.rows[table_id]
        n, p = int(self.n_rows[table_id]), int(self.n_cols[table_id])
        x0, r0 = int(self.x_off[table_id]), int(self.row_off[table_id])
        return {"X": self._x[x0:x0 + n * p].reshape(n, p), "y": self._y[r0:r0 + n], "dup": self._dup[r0:r0 + n],
                "task": rec["task"], "n_classes": int(rec["n_classes"]), "dataset_hash": rec["dataset_hash"],
                "family_id": rec["family_stem"]}


class MemoryCorpus:
    """The same accessor over tables held in RAM (tests, small experiments)."""

    def __init__(self, tables: Sequence[dict[str, Any]]) -> None:
        self.tables = []
        for t, src in enumerate(tables):
            X = np.array(src["X"], dtype=np.float32, order="C")
            y = np.array(src["y"], dtype=np.float32)
            dup = np.array(src.get("dup", np.arange(len(y))), dtype=np.int32)
            if X.ndim != 2 or y.shape != (X.shape[0],) or dup.shape != y.shape:
                raise ViewError("memory_table_shape")
            for arr in (X, y, dup):
                arr.setflags(write=False)
            self.tables.append({"X": X, "y": y, "dup": dup, "task": str(src["task"]), "n_classes": int(src["n_classes"]),
                                "dataset_hash": str(src.get("dataset_hash", f"{t:016x}")),
                                "family_id": str(src.get("family_id", f"fam{t}")), "split": str(src.get("split", "train"))})

    def __len__(self) -> int:
        return len(self.tables)

    def table(self, table_id: int) -> dict[str, Any]:
        return self.tables[table_id]

    def index_rows(self) -> list[dict[str, Any]]:
        """``index.jsonl`` records of these tables (formats.md section 1), for ``CorpusIndex``."""
        rows, x_off, row_off = [], 0, 0
        for t, tbl in enumerate(self.tables):
            n, p = tbl["X"].shape
            rows.append({"table_id": t, "dataset_hash": tbl["dataset_hash"], "canonical_sha256": tbl["dataset_hash"] * 4,
                         "family_stem": tbl["family_id"], "task": tbl["task"], "n_classes": tbl["n_classes"],
                         "n_rows": n, "n_cols": p, "split": tbl["split"], "x_off": x_off, "row_off": row_off})
            x_off, row_off = x_off + n * p, row_off + n
        return rows


def write_flat_arena(corpus: MemoryCorpus, arena: str | Path) -> Path:
    """Write ``corpus`` in the flat arena layout (no META.json checksums: a fixture, not a pack)."""
    root = Path(arena)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "index.jsonl").open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n" for rec in corpus.index_rows())
    for name, key, dtype in (("X.f32", "X", "<f4"), ("y.f32", "y", "<f4"), ("dup.i32", "dup", "<i4")):
        with (root / name).open("wb") as fh:
            for tbl in corpus.tables:
                fh.write(np.ascontiguousarray(tbl[key], dtype=dtype).tobytes())
    return root
