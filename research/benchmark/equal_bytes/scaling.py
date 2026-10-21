"""How program bytes and data bytes scale with the number of fit rows.

For a large lineage the first 2,000 shuffled rows are a fixed test set and fit
views of n = 250 ... 16,000 rows are nested prefixes of the rest. For each n we
compile the headline DOPE rung, sample 4n rows (at most 32,000) and record each
auditor's test loss. Separately, a learning curve of real-row subsets (8- and
16-bit codes, best lossless codec) gives the bytes of real rows needed to match
the DOPE-trained learners' test loss on every informative auditor.

    python3 -B -m research.benchmark.equal_bytes.scaling --binary BIN --sources DIR --out DIR NAME...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.benchmark.equal_bytes.run import AUDITORS, LADDER, HEADLINE_RUNG, encoded_rows, losses

SIZES = (250, 500, 1000, 2000, 4000, 8000, 16000)
TEST_ROWS, SPLIT_SEED, FIT_SEED = 2000, 1729, 11
SUBSET_SEEDS = (0, 1)


def load(source: Path):
    frame = pd.read_csv(source, sep="\t").dropna()
    frame = frame.iloc[np.random.default_rng(SPLIT_SEED).permutation(len(frame))].reset_index(drop=True)
    return frame.iloc[:TEST_ROWS], frame.iloc[TEST_ROWS:]


def project(fit: pd.DataFrame, test: pd.DataFrame):
    low, high = fit.min(), fit.max()
    spread = (high - low).replace(0, 1)
    scale = lambda f: ((f - low) / spread).clip(0, 1).to_numpy()
    return scale(fit), scale(test)


def dope_losses(binary: str, train: np.ndarray, test: np.ndarray) -> dict:
    rung = next(r for r in LADDER if r[0] == HEADLINE_RUNG)
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        np.savetxt(work / "train.csv", train, delimiter=",", fmt="%.8f")
        started = time.time()
        subprocess.run([binary, "compile", "--dataset-dir", str(work), "--task", "regression",
                        "--candidate", rung[1], "--tier", "l1", "--seed", str(FIT_SEED),
                        "--native-target-profile", rung[2], "--packed-codec", "--out", str(work / "m.dpk")],
                       check=True, capture_output=True)
        seconds = time.time() - started
        rows = min(4 * len(train), 32000)
        subprocess.run([binary, "sample", "--kernel", str(work / "m.dpk"), "--rows", str(rows),
                        "--seed", "111", "--out", str(work / "s.csv")], check=True, capture_output=True)
        synthetic = pd.read_csv(work / "s.csv", header=None).to_numpy()
        return {"artifact_bytes": (work / "m.dpk").stat().st_size, "compile_seconds": seconds,
                "synthetic_rows": rows,
                "loss": losses(synthetic[:, :-1], synthetic[:, -1], test[:, :-1], test[:, -1])}


def run_lineage(binary: str, source: Path) -> dict:
    test_frame, pool = load(source)
    sizes = [n for n in SIZES if n <= len(pool)]
    record = {"dataset": source.name.replace(".tsv.gz", ""), "pool_rows": len(pool), "sizes": []}
    for n in sizes:
        fit = pool.iloc[:n]
        train, test = project(fit, test_frame)
        projection = 8 * train.shape[1]
        entry = {"n": n, "projection_bytes": projection,
                 "null": float(np.mean((test[:, -1] - train[:, -1].mean()) ** 2)),
                 "real": losses(train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]),
                 "full_table_bytes": min(encoded_rows(train, b) for b in (8, 16)) + projection,
                 "dope": dope_losses(binary, train, test)}
        entry["dope"]["bytes"] = entry["dope"]["artifact_bytes"] + projection
        # Learning curve of real-row subsets inside this fit view.
        curve = []
        k = 16
        while k <= n:
            for bits in (8, 16):
                values = []
                for seed in SUBSET_SEEDS:
                    order = np.random.default_rng(seed).permutation(n)[:k]
                    subset = np.round(train[order] * (2 ** bits - 1)) / (2 ** bits - 1)
                    values.append({"bytes": encoded_rows(train[order], bits) + projection,
                                   "loss": losses(subset[:, :-1], subset[:, -1], test[:, :-1], test[:, -1])})
                curve.append({"k": k, "bits": bits, "bytes": float(np.median([v["bytes"] for v in values])),
                              "loss": {a: float(np.mean([v["loss"][a] for v in values])) for a in AUDITORS}})
            k = n if k < n and 2 * k > n else 2 * k
        entry["data_curve"] = curve
        record["sizes"].append(entry)
        print(record["dataset"], n, f"dope {entry['dope']['bytes']} B", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        target = args.out / f"{name}.json"
        if not target.exists():
            target.write_text(json.dumps(run_lineage(args.binary, args.sources / f"{name}.tsv.gz"), indent=1))


if __name__ == "__main__":
    main()
