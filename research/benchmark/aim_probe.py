"""Bounded AIM dependency and artifact probe on a fixed public domain.

The shared train-fitted projection prevents a formal end-to-end DP claim.
This probe is not a locked comparison adapter or a privacy certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import time
from itertools import combinations
from pathlib import Path

ROOT = Path("/mnt/fast-scratch/dope-benchmark")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("Adult", "California", "News"))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    import jax
    import numpy as np
    from aim import AIM
    from mbi import Dataset, Domain

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    worker = ROOT / "public-prepared" / "worker" / args.dataset
    manifest = json.loads((worker / "worker-manifest.json").read_text())
    if sha256(worker / "train.csv") != manifest["projected_hashes"]["train"]:
        raise ValueError("train hash mismatch")
    table = np.loadtxt(worker / "train.csv", delimiter=",", ndmin=2)
    if not np.isfinite(table).all() or (table < 0).any() or (table > 1).any():
        raise ValueError("AIM fixed-domain input must be finite in [0,1]")
    # Bins and cardinalities are public constants, independent of this table.
    bins = 8
    codes = np.minimum((table * bins).astype(np.int32), bins - 1)
    names = tuple(f"c{i}" for i in range(table.shape[1]))
    domain = Domain(names, (bins,) * len(names))
    data = Dataset({name: codes[:, i] for i, name in enumerate(names)}, domain)
    workload = [(pair, 1.0) for pair in combinations(names, 2)]
    epsilon = 1.0
    delta = min(1e-5, 1 / len(table) ** 2)
    mechanism = AIM(epsilon, delta, prng=np.random.RandomState(args.seed),
                    max_model_size=80, max_iters=1000)
    start = time.perf_counter()
    model, _ = mechanism.run(data, workload, num_synth_rows=100)
    fit_seconds = time.perf_counter() - start
    probe = ROOT / "pilot-4h" / "aim-probe" / f"{args.dataset}-{args.seed}"
    if probe.exists():
        raise FileExistsError("probe artifact already exists")
    artifact = probe / "artifact"
    artifact.mkdir(parents=True)
    with (artifact / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(worker / "projection.json", artifact / "projection.json")
    sample_start = time.perf_counter()
    with (artifact / "model.pkl").open("rb") as stream:
        loaded = pickle.load(stream)
    np.random.seed(101)
    sample = loaded.synthetic_data(rows=100)
    result = np.column_stack([sample.data[name] for name in names])
    np.savetxt(probe / "sample-100.csv", (result + 0.5) / bins,
               delimiter=",", fmt="%.17g")
    receipt = {"format": "dope-aim-fixed-domain-probe", "dataset": args.dataset,
               "fit_seed": args.seed, "epsilon": epsilon, "delta": delta,
               "domain_bins": bins, "workload_pairs": len(workload),
               "fit_seconds": fit_seconds,
               "sample_100_seconds": time.perf_counter() - sample_start,
               "artifact_bytes": sum(p.stat().st_size for p in artifact.iterdir()),
               "train_sha256": manifest["projected_hashes"]["train"],
               "formal_dp_claim": False,
               "reason": "Shared projection min/max is train-fitted and unaudited for DP accounting"}
    (probe / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
