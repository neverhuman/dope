"""Tests for the flat float32 corpus arena (`dope.flat-corpus/v1`, formats.md section 1)."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from research.levels import s3_flat_corpus as flat
from research.levels.s3_dynamic_augment import S3CorpusStore

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_FIELDS = {
    "table_id", "dataset_hash", "canonical_sha256", "family_stem", "task", "n_classes",
    "n_rows", "n_cols", "split", "x_off", "row_off",
}
META_FIELDS = {
    "schema", "source_manifest_sha256", "tables", "rows", "cells",
    "x_sha256", "y_sha256", "dup_sha256", "index_sha256",
}


def make_table(task: str, n: int, p: int, seed: int, n_classes: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, 1] = np.round(X[:, 1])  # a discrete column
    for i in range(0, 12, 2):  # exact duplicate rows share a duplicate group
        X[i + 1] = X[i]
    dup = np.arange(n, dtype=np.int32)
    for i in range(0, 12, 2):
        dup[i + 1] = dup[i]
    X[rng.random((n, p)) < 0.03] = np.nan
    signal = np.nan_to_num(1.6 * X[:, 0] - 1.1 * X[:, 2])
    if task == "regression":
        y, k = (1000.0 + 50.0 * signal + rng.standard_normal(n)).astype(np.float32), 0
    elif task == "binary":
        y, k = (signal > float(np.median(signal))).astype(np.float32), 2
    else:
        cuts = np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1])
        y, k = np.digitize(signal, cuts).astype(np.float32), n_classes
    return {"dataset_hash": f"{task[:3]}{seed:013d}", "family_stem": f"fam_{task}_{seed}", "task": task,
            "n_classes": k, "X": X, "y": y, "dup_groups": dup}


def write_corpus(root: Path, tables: list[tuple[dict, str]]) -> None:
    (root / "tables").mkdir(parents=True)
    rows = []
    for table, split in tables:
        rel = f"tables/{table['dataset_hash']}.npz"
        np.savez_compressed(root / rel, X=table["X"], y=table["y"], dup_groups=table["dup_groups"],
                            n_classes=np.int32(table["n_classes"]), task=np.array(table["task"]))
        rows.append({
            "canonical_sha256": hashlib.sha256(table["dataset_hash"].encode()).hexdigest(),
            "dataset_hash": table["dataset_hash"], "family_stem": table["family_stem"],
            "n_classes": table["n_classes"], "n_features": int(table["X"].shape[1]),
            "n_rows": int(table["X"].shape[0]), "npz_relpath": rel, "raw_task": table["task"],
            "split": split, "task": table["task"],
        })
    (root / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def run_main(argv: list[str]) -> int:
    """`s3_flat_corpus.main` with its JSON report captured."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return flat.main(argv)


def small_corpus(root: Path) -> list[tuple[dict, str]]:
    tables = [
        (make_table("regression", 150, 6, 1), "train"),
        (make_table("binary", 90, 4, 2), "val"),
        (make_table("multiclass", 130, 9, 3, n_classes=5), "train"),
        (make_table("regression", 64, 3, 4), "val"),
        (make_table("multiclass", 80, 5, 5, n_classes=3), "val"),
        (make_table("binary", 200, 12, 6), "train"),
    ]
    write_corpus(root, tables)
    return tables


class FlatCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="dope-flat-test-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.corpus = self.root / "corpus"
        self.tables = small_corpus(self.corpus)
        self.arena = self.root / "arena"
        self.meta = flat.pack(self.corpus, self.arena)

    def test_round_trip_equals_store_for_every_table(self) -> None:
        fc = flat.attach(self.arena)
        self.assertEqual(len(fc), len(self.tables))
        # manifest order, all splits, table_id == arena position
        self.assertEqual([e["dataset_hash"] for e in fc.index], [t["dataset_hash"] for t, _ in self.tables])
        self.assertEqual([e["split"] for e in fc.index], [s for _, s in self.tables])
        seen = 0
        for split in ("train", "val"):
            store = S3CorpusStore(self.corpus, split=split, cache_cap=16)
            for i in range(len(store)):
                ref = store.load_table(i)
                got = fc.table(fc.table_id(ref["dataset_hash"]))
                self.assertEqual(got["split"], split)
                for key in ("dataset_hash", "family_stem", "task", "n_classes"):
                    self.assertEqual(got[key], ref[key])
                for key, dtype in (("X", np.float32), ("y", np.float32), ("dup_groups", np.int32)):
                    self.assertEqual(got[key].dtype, dtype)
                    self.assertEqual(got[key].shape, ref[key].shape)
                    self.assertTrue(got[key].tobytes() == ref[key].tobytes(), key)  # bit-exact, NaN included
                self.assertTrue(np.isnan(got["X"]).any())
                seen += 1
        self.assertEqual(seen, len(self.tables))

    def test_cell_addressing_and_zero_copy_read_only_views(self) -> None:
        fc = flat.attach(self.arena)
        raw = np.frombuffer((self.arena / "X.f32").read_bytes(), dtype="<f4")
        for entry in fc.index:
            table = fc.table(entry["table_id"])
            r, c = entry["n_rows"] - 1, entry["n_cols"] - 1
            np.testing.assert_array_equal(raw[entry["x_off"] + r * entry["n_cols"] + c], table["X"][r, c])
            for key, arena in (("X", fc.X), ("y", fc.y), ("dup_groups", fc.dup)):
                self.assertTrue(np.shares_memory(table[key], arena))
                self.assertFalse(table[key].flags.writeable)
        self.assertIsInstance(fc.X, np.memmap)
        self.assertEqual(fc.X.mode, "r")
        with self.assertRaises(ValueError):
            fc.table(0)["X"][0, 0] = 1.0

    def test_meta_index_schema_and_checksums(self) -> None:
        meta = json.loads((self.arena / "META.json").read_text(encoding="utf-8"))
        self.assertEqual(meta, self.meta)
        self.assertEqual(set(meta), META_FIELDS)
        self.assertEqual(meta["schema"], "dope.flat-corpus/v1")
        self.assertEqual(meta["tables"], len(self.tables))
        self.assertEqual(meta["rows"], sum(t["X"].shape[0] for t, _ in self.tables))
        self.assertEqual(meta["cells"], sum(t["X"].size for t, _ in self.tables))
        self.assertEqual(meta["source_manifest_sha256"], hashlib.sha256((self.corpus / "manifest.jsonl").read_bytes()).hexdigest())
        for key, name in (("x_sha256", "X.f32"), ("y_sha256", "y.f32"), ("dup_sha256", "dup.i32"), ("index_sha256", "index.jsonl")):
            self.assertEqual(meta[key], hashlib.sha256((self.arena / name).read_bytes()).hexdigest())
        self.assertEqual((self.arena / "X.f32").stat().st_size, 4 * meta["cells"])
        self.assertEqual((self.arena / "y.f32").stat().st_size, 4 * meta["rows"])
        self.assertEqual((self.arena / "dup.i32").stat().st_size, 4 * meta["rows"])
        x_off = row_off = 0
        for position, line in enumerate((self.arena / "index.jsonl").read_text(encoding="utf-8").splitlines()):
            entry = json.loads(line)
            self.assertEqual(set(entry), INDEX_FIELDS)
            self.assertEqual(list(entry), sorted(entry))  # key order sorted
            self.assertEqual((entry["table_id"], entry["x_off"], entry["row_off"]), (position, x_off, row_off))
            x_off += entry["n_rows"] * entry["n_cols"]
            row_off += entry["n_rows"]
        self.assertEqual(flat.verify(self.arena), meta)

    def test_flipped_byte_fails_verification(self) -> None:
        for name in ("X.f32", "y.f32", "dup.i32", "index.jsonl"):
            broken = self.root / f"broken-{name}"
            shutil.copytree(self.arena, broken)
            data = bytearray((broken / name).read_bytes())
            data[len(data) // 2] ^= 0x01
            (broken / name).write_bytes(bytes(data))
            with self.assertRaises(ValueError) as caught:
                flat.verify(broken)
            self.assertEqual(str(caught.exception), f"flat_corpus_checksum_mismatch:{name}")
            self.assertEqual(run_main(["verify", "--arena", str(broken)]), 1)
        truncated = self.root / "truncated"
        shutil.copytree(self.arena, truncated)
        (truncated / "y.f32").write_bytes((truncated / "y.f32").read_bytes()[:-4])
        with self.assertRaises(ValueError):
            flat.verify(truncated)
        with self.assertRaises(ValueError):
            flat.attach(truncated)

    def test_pack_is_deterministic_atomic_and_refuses_overwrite(self) -> None:
        second = self.root / "arena-2"
        self.assertEqual(run_main(["pack", "--corpus", str(self.corpus), "--out", str(second)]), 0)
        for name in ("META.json", "index.jsonl", "X.f32", "y.f32", "dup.i32"):
            self.assertTrue((second / name).read_bytes() == (self.arena / name).read_bytes(), name)
        with self.assertRaises(FileExistsError):
            flat.pack(self.corpus, second)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["arena", "arena-2", "corpus"])
        done = subprocess.run(
            [sys.executable, "-B", "-m", "research.levels.s3_flat_corpus", "verify", "--arena", str(second)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["meta"], self.meta)

    def test_schema_drift_is_rejected_without_leaving_partial_output(self) -> None:
        cases = {
            "n_rows": lambda rec: rec.update(n_rows=rec["n_rows"] + 1),
            "n_features": lambda rec: rec.update(n_features=rec["n_features"] + 1),
            "task": lambda rec: rec.update(task="ranking"),
            "n_classes": lambda rec: rec.update(n_classes=7),
            "duplicate_dataset_hash": None,
        }
        lines = (self.corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        for code, mutate in cases.items():
            drifted = self.root / f"drift-{code}"
            shutil.copytree(self.corpus, drifted)
            records = [json.loads(line) for line in lines]
            if mutate is None:
                records.append(dict(records[0]))
            else:
                mutate(records[1])
            (drifted / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            out = self.root / f"out-{code}"
            with self.assertRaises(ValueError) as caught:
                flat.pack(drifted, out)
            self.assertTrue(str(caught.exception).startswith("flat_corpus_table_invalid:"), str(caught.exception))
            self.assertFalse(out.exists())
            self.assertEqual([p.name for p in self.root.iterdir() if ".partial-" in p.name], [])
            shutil.rmtree(drifted)


if __name__ == "__main__":
    unittest.main()
