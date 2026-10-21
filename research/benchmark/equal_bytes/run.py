"""Run the equal-byte study on prepared lineage directories.

Arms, all scored by the v2 auditors (linear, one-hidden-layer MLP, CatBoost;
seed 1729) on the same untouched test rows:

* ``data``: the largest uniform subset of real fit rows whose quantized (8 or
  16 bits), row-sorted, column-major encoding fits the budget under the best of
  LZMA2 (extreme), bzip2 and raw deflate. Three subset seeds per budget. Budgets
  are the power-of-two grid plus each DOPE rung's own median bytes, so the
  equal-byte comparison is exact. The MLP keeps its better test loss of
  training on the subset as given or resampled to the DOPE arm's synthetic size
  (``MATCHED_FACTOR`` times the fit rows), so its epoch budget does not starve
  small subsets of optimizer steps; the whole table is scored as the reference.
* ``dope``: a ladder of DOPE programs (packed values, natively trained
  residual), three fit seeds each, sampled at four times the fit rows.
* ``dope_cpu_default``: the historical CPU build default, for reference.

The real reference's MLP takes the same better-of-two, so retention compares
MLPs trained on equally many rows. Every arm is charged the same 8 bytes per column for the fit-row min-max
projection it needs to return original units; the historical JSON projection
size is recorded beside it.

    python3 -B -m research.benchmark.equal_bytes.run --binary BIN --out DIR LINEAGE_DIR...
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import os
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

GRID = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BITS = (8, 16)
SUBSET_SEEDS = (0, 1, 2)
FIT_SEEDS = (11, 23, 37)
AUDITOR_SEED = 1729
AUDITORS = ("linear", "mlp", "catboost")
HEADLINE_RUNG = "L4"
MATCHED_FACTOR = 4  # the DOPE arm samples this many times the fit rows
MLP_EPOCHS, MLP_PATIENCE = 200, 15  # the auditor MLP's epoch budget and early-stopping patience
# name, candidate, native target profile, packed values
LADDER = (
    ("L1", "symbolic_k1_d2_t5_q6", "features6_width8_steps2048", True),
    ("L2", "symbolic_k1_d2_t5_q6", "features12_steps2048", True),
    ("L3", "symbolic_k2_d2_t5_q6", "features12_steps2048", True),
    ("L4", "symbolic_k2_d2_t5_q8", "features12_steps2048", True),
    ("L5", "symbolic_k3_d2_t5_q8", "features24_steps2048", True),
)
CPU_DEFAULT = ("cpu_default", "compact_neural_residual_symbolic", None, False)
# Ablation rungs, run on demand: the headline rung with the exact Gaussian copula.
ABLATIONS = (
    ("G4", "symbolic_k2_d7_t5_q8", "features12_steps2048", True),
)


def auditor(name: str, rows: int):
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    if name == "linear":
        return LinearRegression()
    if name == "mlp":
        # The v2 evaluator's MLP; early stopping needs a validation split.
        return MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, random_state=AUDITOR_SEED,
                            early_stopping=rows >= 20, n_iter_no_change=15)
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=100, depth=6, learning_rate=0.05, random_seed=AUDITOR_SEED,
                             verbose=False, thread_count=1, allow_writing_files=False)


def losses(x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    out = {}
    for name in AUDITORS:
        model = auditor(name, len(y)).fit(x, y)
        out[name] = float(np.mean((model.predict(x_test) - y_test) ** 2))
    return out


def resampled_mlp_loss(x, y, x_test, y_test, size: int, seed: int) -> float:
    """Test loss of the MLP trained on the rows resampled to `size`.

    A tenth of the distinct rows (at least two, when there are twenty or more)
    is held out before resampling and drives early stopping, so validation never
    sees a duplicate of a training row; the best epoch's weights are kept.
    """
    import copy
    from sklearn.neural_network import MLPRegressor
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    hold = max(2, len(y) // 10) if len(y) >= 20 else 0
    held, kept = order[:hold], order[hold:]
    index = kept[rng.integers(0, len(kept), size)]
    # partial_fit runs one shuffled epoch per call and keeps the Adam state between calls.
    model = MLPRegressor(hidden_layer_sizes=(32,), random_state=AUDITOR_SEED)
    best, state, stale = np.inf, None, 0
    for _ in range(MLP_EPOCHS):
        model.partial_fit(x[index], y[index])
        if not hold:
            continue
        loss = float(np.mean((model.predict(x[held]) - y[held]) ** 2))
        if loss < best:
            best, state, stale = loss, (copy.deepcopy(model.coefs_), copy.deepcopy(model.intercepts_)), 0
        else:
            stale += 1
            if stale >= MLP_PATIENCE:
                break
    if state is not None:
        model.coefs_, model.intercepts_ = state
    return float(np.mean((model.predict(x_test) - y_test) ** 2))


def matched_losses(x, y, x_test, y_test, size: int, seed: int, plain: dict | None = None) -> tuple[dict, dict]:
    """Losses with the MLP's better of the rows as given or resampled to `size` rows.

    The MLP has an epoch budget, so its optimizer steps scale with the rows it
    sees; least squares and a fixed number of boosting rounds do not. `plain`
    reuses losses already measured on the rows as given.
    """
    plain = plain or losses(x, y, x_test, y_test)
    if len(y) >= size:
        return plain, plain
    resampled = resampled_mlp_loss(x, y, x_test, y_test, size, seed)
    return {**plain, "mlp": min(plain["mlp"], resampled)}, plain


def is_packed(artifact: Path) -> bool:
    """Whether the written artifact carries the packed-values header flag (bit 1 of the u16 at byte 10)."""
    head = artifact.read_bytes()[:12]
    return head[:4] == b"DPK3" and bool(int.from_bytes(head[10:12], "little") & 2)


def encoded_rows(rows: np.ndarray, bits: int) -> int:
    """Smallest lossless encoding of quantized rows, with a 4-byte header."""
    codes = np.round(rows * (2 ** bits - 1)).astype(np.uint8 if bits == 8 else np.uint16)
    codes = codes[np.lexsort(codes.T[::-1])]
    payload = np.ascontiguousarray(codes.T).tobytes()
    lzma_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
    best = min(len(lzma.compress(payload, format=lzma.FORMAT_RAW, filters=lzma_filters)),
               len(bz2.compress(payload, 9)),
               len(zlib.compress(payload, 9)) - 6)
    return best + 4


def largest_subset(order: np.ndarray, rows: np.ndarray, bits: int, budget: int, fixed: int) -> int:
    if encoded_rows(rows[order], bits) + fixed <= budget:
        return len(order)
    low, high = 0, len(order)
    while high - low > 1:
        middle = (low + high) // 2
        if encoded_rows(rows[order[:middle]], bits) + fixed <= budget:
            low = middle
        else:
            high = middle
    return low


def projection_json_bytes(meta: dict) -> int:
    """Size of the historical harness's projection.json for this lineage."""
    columns = meta["columns"]
    features = [{"kind": "numeric", "max": hi, "min": lo, "missing": "0.5", "name": name}
                for name, lo, hi in zip(columns[:-1], meta["low"][:-1], meta["high"][:-1])]
    mapping = {"version": 1, "target": columns[-1], "task": "regression", "input_columns": columns,
               "features": features, "output_features": len(features),
               "target_map": {"min": meta["low"][-1], "max": meta["high"][-1]},
               "fit_partition": "train", "unknown_category_rule": "all_zero",
               "numeric_rule": "train_minmax_clip_0_1", "missing_numeric_rule": "0.5"}
    return len(json.dumps(mapping, sort_keys=True, indent=2)) + 1


def compile_program(binary: str, lineage: Path, rung: tuple, seed: int, out: Path) -> dict:
    name, candidate, profile, packed = rung
    command = [binary, "compile", "--dataset-dir", str(lineage), "--task", "regression",
               "--candidate", candidate, "--tier", "l1", "--seed", str(seed), "--out", str(out)]
    if profile:
        command += ["--native-target-profile", profile]
    if packed:
        command.append("--packed-codec")
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr[-300:]}
    return {"artifact_bytes": out.stat().st_size, "compile_seconds": time.time() - started,
            "is_packed": is_packed(out)}


def dope_arm(binary: str, lineage: Path, rungs, train, test, projection: int) -> list[dict]:
    x_test, y_test = test[:, :-1], test[:, -1]
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for rung in rungs:
            for seed in FIT_SEEDS:
                artifact = Path(scratch) / f"{rung[0]}-{seed}.dpk"
                record = {"rung": rung[0], "candidate": rung[1], "profile": rung[2],
                          "packed": rung[3], "fit_seed": seed}
                record.update(compile_program(binary, lineage, rung, seed, artifact))
                if "error" in record:
                    rows.append(record)
                    continue
                if rung[0] == HEADLINE_RUNG:
                    # Same program written with the historical value codec: the
                    # samples are identical, only the bytes differ.
                    plain = Path(scratch) / "plain.dpk"
                    record["historical_codec_bytes"] = compile_program(
                        binary, lineage, (rung[0], rung[1], rung[2], False), seed, plain
                    ).get("artifact_bytes")
                record["bytes"] = record["artifact_bytes"] + projection
                synthetic = Path(scratch) / "sample.csv"
                subprocess.run([binary, "sample", "--kernel", str(artifact), "--rows",
                                str(4 * len(train)), "--seed", str(100 + seed), "--out", str(synthetic)],
                               check=True, capture_output=True)
                values = pd.read_csv(synthetic, header=None).values
                record["loss"] = losses(values[:, :-1], values[:, -1], x_test, y_test)
                rows.append(record)
    return rows


def program_budgets(record: dict) -> list[int]:
    """Each DOPE rung's median charged bytes: the exact equal-byte budgets."""
    rungs: dict = {}
    for row in record.get("dope", []):
        if "bytes" in row:
            rungs.setdefault(row["rung"], []).append(row["bytes"])
    return sorted({int(np.median(sizes)) for sizes in rungs.values()})


def data_arm(train, test, projection: int, reference: dict, budgets=(), cache=()) -> list[dict]:
    """`cache` holds earlier rows of this arm; their losses on the rows as given are reused."""
    x_test, y_test = test[:, :-1], test[:, -1]
    known = {(r["seed"], r["bits"], r["k"]): r["loss_as_given"] for r in cache if "loss_as_given" in r}
    rows = []
    for seed in SUBSET_SEEDS:
        order = np.random.default_rng(seed).permutation(len(train))
        for bits in BITS:
            previous = None
            for budget in sorted(set(GRID) | set(budgets)):
                k = largest_subset(order, train, bits, budget, projection)
                if k < 2:
                    continue
                entry = {"seed": seed, "bits": bits, "budget": budget, "k": k, "exact": budget not in GRID,
                         "bytes": encoded_rows(train[order[:k]], bits) + projection}
                if k == len(train):
                    entry["loss"], entry["anchored"] = dict(reference), True
                elif previous is not None and previous["k"] == k:
                    entry["loss"], entry["loss_as_given"] = previous["loss"], previous["loss_as_given"]
                else:
                    subset = np.round(train[order[:k]] * (2 ** bits - 1)) / (2 ** bits - 1)
                    entry["loss"], entry["loss_as_given"] = matched_losses(
                        subset[:, :-1], subset[:, -1], x_test, y_test, MATCHED_FACTOR * len(train), seed,
                        known.get((seed, bits, k)))
                rows.append(entry)
                previous = entry
                if k == len(train):
                    break
    return rows


def run(binary: str, lineage: Path, arms: set[str], rungs: tuple = LADDER, base: dict | None = None) -> dict:
    """One lineage's record; `base` is an earlier record whose other arms are kept."""
    train = pd.read_csv(lineage / "train.csv", header=None).values
    test = pd.read_csv(lineage / "test.csv", header=None).values
    meta = json.loads((lineage / "meta.json").read_text())
    projection = 8 * train.shape[1]
    record = dict(base or {})
    record.update({"dataset": lineage.name, "n_fit": len(train), "n_test": len(test),
                   "d": train.shape[1] - 1, "rows_available": meta.get("rows_available"),
                   "projection_bytes": projection, "projection_json_bytes": projection_json_bytes(meta),
                   "null": float(np.mean((test[:, -1] - train[:, -1].mean()) ** 2)),
                   "full_table_bytes": {str(b): encoded_rows(train, b) + projection for b in BITS}})
    real = losses(train[:, :-1], train[:, -1], test[:, :-1], test[:, -1])
    if base is not None and any(abs(base["real"][a] - real[a]) > 1e-9 * max(1.0, abs(real[a])) for a in AUDITORS):
        raise ValueError(f"{lineage.name}: the lineage split differs from the reused record")
    record["real"] = real
    record["real_matched"], _ = matched_losses(train[:, :-1], train[:, -1], test[:, :-1], test[:, -1],
                                                  MATCHED_FACTOR * len(train), AUDITOR_SEED)
    record["protocol"] = "equal-bytes-v3"
    if "dope" in arms:
        record["dope"] = dope_arm(binary, lineage, rungs, train, test, projection)
    if "cpu_default" in arms:
        record["dope_cpu_default"] = dope_arm(binary, lineage, (CPU_DEFAULT,), train, test, projection)
    if "data" in arms:
        if not program_budgets(record):
            raise ValueError(f"{lineage.name}: the data arm needs DOPE rows for exact budgets (--arms dope or --reuse)")
        record["data"] = data_arm(train, test, projection, record["real_matched"], program_budgets(record),
                                  (base or {}).get("data", ()))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="dope,cpu_default,data")
    parser.add_argument("--rungs", default=",".join(rung[0] for rung in LADDER))
    parser.add_argument("--reuse", type=Path, help="directory of earlier records whose other arms are kept")
    parser.add_argument("lineages", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for lineage in args.lineages:
        target = args.out / f"{lineage.name}.json"
        if target.exists():
            continue
        started = time.time()
        rungs = tuple(rung for rung in LADDER + ABLATIONS if rung[0] in args.rungs.split(","))
        base = json.loads((args.reuse / target.name).read_text()) if args.reuse else None
        record = run(args.binary, lineage, set(args.arms.split(",")), rungs, base)
        target.write_text(json.dumps(record, indent=1))
        print(lineage.name, f"{time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
