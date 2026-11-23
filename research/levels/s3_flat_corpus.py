"""Flat float32 corpus arena (`dope.flat-corpus/v1`) for training workers and the scoreboard.

Layout and field meanings are fixed by docs/plans/v2-gap-closure/formats.md section 1:
`META.json`, `index.jsonl`, `X.f32`, `y.f32`, `dup.i32` (all little-endian, tables concatenated
in manifest order with every split). `pack` writes an arena once, `attach` memory-maps it
read-only with zero-copy table views, `verify` recomputes every checksum.

Errors carry a code and a table position only; they never echo cell values or headers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

SCHEMA = "dope.flat-corpus/v1"
TASKS = ("regression", "binary", "multiclass")
DATA_FILES = {"x_sha256": "X.f32", "y_sha256": "y.f32", "dup_sha256": "dup.i32", "index_sha256": "index.jsonl"}
INDEX_KEYS = (
    "canonical_sha256", "dataset_hash", "family_stem", "n_classes", "n_cols", "n_rows",
    "row_off", "split", "table_id", "task", "x_off",
)
_CHUNK = 1 << 22


def _dumps(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(_CHUNK)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _HashedWriter:
    """Append-only binary file with a running sha256 and element count."""

    def __init__(self, path: Path) -> None:
        self._stream = path.open("wb")
        self._digest = hashlib.sha256()

    def write(self, data: bytes) -> None:
        self._digest.update(data)
        self._stream.write(data)

    def close(self) -> str:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        return self._digest.hexdigest()

    def abort(self) -> None:
        self._stream.close()


def _load_manifest_table(corpus: Path, position: int, rec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def bad(code: str) -> ValueError:
        return ValueError(f"flat_corpus_table_invalid:{code}:table_{position}")

    task = rec.get("task")
    if task not in TASKS:
        raise bad("task")
    n_classes = int(rec.get("n_classes", -1))
    if (task == "regression" and n_classes != 0) or (task == "binary" and n_classes != 2) or (
        task == "multiclass" and n_classes < 2
    ):
        raise bad("n_classes")
    for key in ("dataset_hash", "canonical_sha256", "family_stem", "split", "npz_relpath"):
        if not isinstance(rec.get(key), str) or not rec[key]:
            raise bad(key)
    with np.load(corpus / rec["npz_relpath"], allow_pickle=False) as npz:
        x = np.ascontiguousarray(np.asarray(npz["X"], dtype=np.float32).astype("<f4", copy=False))
        y = np.ascontiguousarray(np.asarray(npz["y"], dtype=np.float32).astype("<f4", copy=False))
        dup = np.ascontiguousarray(np.asarray(npz["dup_groups"], dtype=np.int32).astype("<i4", copy=False))
        stored_classes = int(npz["n_classes"]) if "n_classes" in npz.files else n_classes
        stored_task = str(npz["task"]) if "task" in npz.files else task
    if x.ndim != 2 or y.shape != (x.shape[0],) or dup.shape != (x.shape[0],):
        raise bad("shape")
    if "n_rows" in rec and int(rec["n_rows"]) != x.shape[0]:
        raise bad("n_rows")
    if "n_features" in rec and int(rec["n_features"]) != x.shape[1]:
        raise bad("n_features")
    if stored_classes != n_classes or stored_task != task:
        raise bad("manifest_npz_disagree")
    return x, y, dup


def pack(corpus_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Pack every manifest table (all splits, manifest order) into a new arena; returns META."""
    corpus, out = Path(corpus_dir), Path(out_dir)
    manifest_path = corpus / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError("flat_corpus_missing_manifest")
    if out.exists():
        raise FileExistsError("flat_corpus_out_exists")
    manifest_bytes = manifest_path.read_bytes()
    records = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("flat_corpus_empty_manifest")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    writers: dict[str, _HashedWriter] = {}
    try:
        for name in DATA_FILES.values():
            writers[name] = _HashedWriter(staging / name)
        seen: set[str] = set()
        x_off = row_off = 0
        for table_id, rec in enumerate(records):
            x, y, dup = _load_manifest_table(corpus, table_id, rec)
            if rec["dataset_hash"] in seen:
                raise ValueError(f"flat_corpus_table_invalid:duplicate_dataset_hash:table_{table_id}")
            seen.add(rec["dataset_hash"])
            writers["X.f32"].write(x.tobytes(order="C"))
            writers["y.f32"].write(y.tobytes(order="C"))
            writers["dup.i32"].write(dup.tobytes(order="C"))
            writers["index.jsonl"].write(_dumps({
                "table_id": table_id,
                "dataset_hash": rec["dataset_hash"],
                "canonical_sha256": rec["canonical_sha256"],
                "family_stem": rec["family_stem"],
                "task": rec["task"],
                "n_classes": int(rec["n_classes"]),
                "n_rows": int(x.shape[0]),
                "n_cols": int(x.shape[1]),
                "split": rec["split"],
                "x_off": x_off,
                "row_off": row_off,
            }))
            x_off += int(x.size)
            row_off += int(x.shape[0])
        if x_off == 0 or row_off == 0:
            raise ValueError("flat_corpus_no_cells")
        hashes = {name: writer.close() for name, writer in writers.items()}
        meta = {
            "schema": SCHEMA,
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "tables": len(records),
            "rows": row_off,
            "cells": x_off,
            **{key: hashes[name] for key, name in DATA_FILES.items()},
        }
        with (staging / "META.json").open("wb") as stream:
            stream.write((json.dumps(meta, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(staging, 0o755)
        _fsync_dir(staging)
        os.rename(staging, out)
        _fsync_dir(out.parent)
    except BaseException:
        for writer in writers.values():
            writer.abort()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return meta


class FlatCorpus:
    """Read-only memory-mapped arena. `X`, `y`, `dup` are whole-arena memmaps (mode r)."""

    def __init__(self, root: Path, meta: dict[str, Any], index: list[dict[str, Any]]) -> None:
        self.root = root
        self.meta = meta
        self.index = index
        self.X = np.memmap(root / "X.f32", dtype="<f4", mode="r")
        self.y = np.memmap(root / "y.f32", dtype="<f4", mode="r")
        self.dup = np.memmap(root / "dup.i32", dtype="<i4", mode="r")
        if self.X.shape[0] != meta["cells"] or self.y.shape[0] != meta["rows"] or self.dup.shape[0] != meta["rows"]:
            raise ValueError("flat_corpus_size_mismatch")
        self._by_hash = {entry["dataset_hash"]: entry["table_id"] for entry in index}

    def __len__(self) -> int:
        return len(self.index)

    def table_id(self, dataset_hash: str) -> int:
        return self._by_hash[dataset_hash]

    def table(self, i: int) -> dict[str, Any]:
        """Zero-copy views of table `i` plus its index fields (same keys as S3CorpusStore.load_table)."""
        entry = self.index[i]
        rows, cols, x_off, row_off = entry["n_rows"], entry["n_cols"], entry["x_off"], entry["row_off"]
        return {
            **{key: entry[key] for key in ("table_id", "dataset_hash", "canonical_sha256", "family_stem", "task", "n_classes", "split")},
            "X": self.X[x_off : x_off + rows * cols].reshape(rows, cols),
            "y": self.y[row_off : row_off + rows],
            "dup_groups": self.dup[row_off : row_off + rows],
        }


def _read_meta(root: Path) -> dict[str, Any]:
    meta = json.loads((root / "META.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
        raise ValueError("flat_corpus_schema_mismatch")
    for key in ("source_manifest_sha256", "tables", "rows", "cells", *DATA_FILES):
        if key not in meta:
            raise ValueError(f"flat_corpus_meta_missing:{key}")
    return meta


def _read_index(root: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    index = [json.loads(line) for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    x_off = row_off = 0
    for position, entry in enumerate(index):
        if not isinstance(entry, dict) or tuple(sorted(entry)) != INDEX_KEYS:
            raise ValueError(f"flat_corpus_index_invalid:keys:table_{position}")
        if entry["table_id"] != position or entry["x_off"] != x_off or entry["row_off"] != row_off:
            raise ValueError(f"flat_corpus_index_invalid:offsets:table_{position}")
        if entry["task"] not in TASKS or entry["n_rows"] < 0 or entry["n_cols"] < 0:
            raise ValueError(f"flat_corpus_index_invalid:fields:table_{position}")
        x_off += entry["n_rows"] * entry["n_cols"]
        row_off += entry["n_rows"]
    if (len(index), row_off, x_off) != (meta["tables"], meta["rows"], meta["cells"]):
        raise ValueError("flat_corpus_index_invalid:totals")
    return index


def attach(arena_dir: str | Path) -> FlatCorpus:
    """Attach an arena read-only. Structural checks only; run `verify` for checksums."""
    root = Path(arena_dir)
    meta = _read_meta(root)
    return FlatCorpus(root, meta, _read_index(root, meta))


def verify(arena_dir: str | Path) -> dict[str, Any]:
    """Recompute all four checksums, then the size and index invariants; raises on any mismatch."""
    root = Path(arena_dir)
    meta = _read_meta(root)
    for key, name in DATA_FILES.items():
        if _file_sha256(root / name) != meta[key]:
            raise ValueError(f"flat_corpus_checksum_mismatch:{name}")
    sizes = {"X.f32": 4 * meta["cells"], "y.f32": 4 * meta["rows"], "dup.i32": 4 * meta["rows"]}
    for name, expected in sizes.items():
        if (root / name).stat().st_size != expected:
            raise ValueError(f"flat_corpus_size_mismatch:{name}")
    _read_index(root, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flat float32 corpus arena (dope.flat-corpus/v1)")
    commands = parser.add_subparsers(dest="command", required=True)
    pack_cmd = commands.add_parser("pack", help="pack a corpus directory (manifest.jsonl + tables/*.npz)")
    pack_cmd.add_argument("--corpus", required=True)
    pack_cmd.add_argument("--out", required=True)
    verify_cmd = commands.add_parser("verify", help="recompute checksums of an arena")
    verify_cmd.add_argument("--arena", required=True)
    args = parser.parse_args(argv)
    try:
        meta = pack(args.corpus, args.out) if args.command == "pack" else verify(args.arena)
    except (OSError, ValueError, KeyError) as error:
        code = str(error) if str(error).startswith("flat_corpus_") else type(error).__name__
        print(json.dumps({"status": "failed", "error": code}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "command": args.command, "meta": meta}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
