"""Run validation-only descriptive metrics for a completed pilot sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pilot_metrics import measure
from .score import sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark")


def find_sample(dataset: str, method: str, fit_seed: int, sample_seed: int,
                multiplier: int, results_root: Path | None = None) -> tuple[Path, dict]:
    train_rows = json.loads((ROOT / "public-prepared" / "worker" / dataset / "worker-manifest.json").read_text())["train_rows"]
    results_root = results_root or ROOT / "pilot-4h" / "results"
    for fit_path in results_root.glob("*/fit-receipt.json"):
        fit = json.loads(fit_path.read_text())
        identity = fit.get("fit_identity", {})
        if (fit.get("status") != "ok" or identity.get("dataset") != dataset
                or identity.get("method") != method or identity.get("fit_seed") != fit_seed):
            continue
        for receipt_path in fit_path.parent.glob("*.receipt.json"):
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("status") == "ok" and receipt.get("sample_seed") == sample_seed
                    and receipt.get("row_count") == train_rows * multiplier):
                sample = fit_path.parent / f"{receipt['run_key']}.csv"
                if sha256(sample) != receipt["sample_sha256"]:
                    raise ValueError("sample hash mismatch")
                return sample, fit
    raise FileNotFoundError("matching completed pilot sample is unavailable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("Adult", "California", "News"))
    parser.add_argument("method")
    parser.add_argument("--fit-seed", type=int, default=11)
    parser.add_argument("--sample-seed", type=int, default=101)
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--results-root", type=Path, default=ROOT / "pilot-4h" / "results")
    args = parser.parse_args()
    sample, fit = find_sample(args.dataset, args.method, args.fit_seed,
                              args.sample_seed, args.multiplier, args.results_root)
    output = ROOT / "pilot-4h" / "validation-metrics"
    output.mkdir(exist_ok=True)
    destination = output / f"{args.dataset}-{args.method}-{args.fit_seed}-{args.sample_seed}-x{args.multiplier}.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["fit_key"] != fit["fit_key"] or previous["sample_sha256"] != sha256(sample):
            raise ValueError("existing metric identity differs")
        print(destination)
        return
    worker = ROOT / "public-prepared" / "worker" / args.dataset
    projection = json.loads((worker / "projection.json").read_text())
    report = measure(worker / "train.csv", worker / "validation.csv", sample,
                     projection["task"])
    report.update({"dataset": args.dataset, "method": args.method,
                   "fit_seed": args.fit_seed, "sample_seed": args.sample_seed,
                   "multiplier": args.multiplier, "fit_key": fit["fit_key"],
                   "sample_sha256": sha256(sample)})
    destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
