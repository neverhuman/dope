#!/usr/bin/env python3
"""Which encoding and which backend? A small, reproducible CPU ablation.

Runs the quickstart pipeline over a grid of encoder options and DOPE backends on two
open UCI tables and five seeds, and writes ``results/ablation.json`` and
``results/ablation.md``. Every configuration sees the same fit rows and the same
holdout rows for a given seed; the seed changes the fit subsample (Adult) or the
80/20 split (Abalone) and the compile seed.

    python3 examples/quickstart/ablation.py            # about 15 minutes on 2 CPU cores

Evidence status: local engineering ablation on two tables. It chose the quickstart
defaults; it is not the paper's benchmark and makes no claim beyond these tables.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dope_eval import retention_report, worst_retention  # noqa: E402
from dope_select import select_backend  # noqa: E402
from dope_tables import TableEncoder, read_dope_samples  # noqa: E402
from quickstart import REPO, find_kernel, load_dataset, split  # noqa: E402

CONFIGS = [
    # (label, numeric, categorical, backend)
    ("one-hot · robust · chow_liu", "robust", "onehot", "chow_liu"),
    ("target-ordered · robust · chow_liu", "robust", "target_ordered", "chow_liu"),
    ("auto (one-hot if at most 4 levels) · robust · chow_liu", "robust", "auto", "chow_liu"),
    ("target-ordered · min-max · chow_liu", "minmax", "target_ordered", "chow_liu"),
    ("target-ordered · quantile · chow_liu", "quantile", "target_ordered", "chow_liu"),
    ("target-ordered · robust · default search", "robust", "target_ordered", "search"),
    ("target-ordered · robust · sparse_gaussian_copula", "robust", "target_ordered", "sparse_gaussian_copula"),
    ("target-ordered · robust · compact_neural_residual", "robust", "target_ordered",
     "compact_neural_residual_symbolic"),
    ("target-ordered · robust · transfer-in-the-loop (K=3)", "robust", "target_ordered", "auto"),
]
SEEDS = (11, 23, 37, 53, 71)
ADULT_FIT_ROWS = 8000


def one_run(kernel: str, fit, holdout, target, positive, numeric, categorical, backend, seed, work: Path):
    encoder = TableEncoder(target, numeric=numeric, categorical=categorical, positive=positive).fit(fit)
    chosen = None
    if backend == "auto":
        chosen = select_backend(kernel, fit, target, positive=encoder.classes[1] if encoder.task == "binary"
                                else None, encoder_options={"numeric": numeric, "categorical": categorical},
                                seed=seed)
        backend = chosen.backend
    encoder.write_dataset_dir(fit, work)
    command = [kernel, "compile", "--dataset-dir", str(work), "--task", encoder.task, "--out",
               str(work / "model.dpk"), "--report", str(work / "report.json"), "--seed", str(seed),
               "--deadline-seconds", "120"]
    if backend != "search":
        command += ["--candidate", backend]
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run([kernel, "sample", "--kernel", str(work / "model.dpk"), "--rows", str(len(fit)),
                    "--out", str(work / "samples.csv"), "--seed", "101"], check=True, capture_output=True)
    synthetic = encoder.inverse_transform(read_dope_samples(work / "samples.csv"), seed=seed)
    results = retention_report(fit, holdout, synthetic, target, encoder.task, positive=positive)
    report = json.loads((work / "report.json").read_text())
    return {
        "retention": {r.auditor: r.retention for r in results},
        "worst": worst_retention(results),
        "model_bytes": (work / "model.dpk").stat().st_size,
        "encoder_bytes": len(encoder.to_bytes()),
        "candidate": report.get("selected_candidate"),
        "selected_backend": backend if chosen is not None else None,
        "selection_scores": chosen.scores if chosen is not None else None,
    }


def summarize(runs: list[dict]) -> list[dict]:
    rows = []
    for dataset in sorted({r["dataset"] for r in runs}):
        for label, *_ in CONFIGS:
            cell = [r for r in runs if r["dataset"] == dataset and r["config"] == label]
            if not cell:
                continue
            auditors = sorted(cell[0]["retention"])
            rows.append({
                "dataset": dataset, "config": label, "runs": len(cell),
                "worst_mean": statistics.mean(r["worst"] for r in cell),
                "worst_sd": statistics.pstdev(r["worst"] for r in cell),
                **{f"{a}_mean": statistics.mean(r["retention"][a] for r in cell) for a in auditors},
                "state_bytes_mean": statistics.mean(r["model_bytes"] + r["encoder_bytes"] for r in cell),
            })
    return rows


def markdown(rows: list[dict]) -> str:
    auditors = [k[:-5] for k in rows[0] if k.endswith("_mean") and k not in ("worst_mean", "state_bytes_mean")]
    head = ("| Table | Encoding · backend | Worst-auditor retention | "
            + " | ".join(a for a in auditors) + " | State bytes |")
    lines = [head, "| --- | --- | ---: | " + " | ".join("---:" for _ in auditors) + " | ---: |"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['config']} | {row['worst_mean']:.3f} ± {row['worst_sd']:.3f} | "
                     + " | ".join(f"{row[a + '_mean']:.3f}" for a in auditors)
                     + f" | {row['state_bytes_mean']:,.0f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kernel")
    parser.add_argument("--datasets", default="adult,abalone")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    kernel = find_kernel(args.kernel)
    cache = REPO / "target" / "quickstart" / "cache"
    runs = []
    started = time.time()
    for dataset in args.datasets.split(","):
        fit_all, holdout_all, target, positive, title = load_dataset(dataset, cache)
        for seed in SEEDS:
            if dataset == "adult":
                fit = fit_all.sample(n=ADULT_FIT_ROWS, random_state=seed).reset_index(drop=True)
                holdout = holdout_all
            else:
                pooled = pd.concat([fit_all, holdout_all], ignore_index=True)
                fit, holdout = split(pooled, 0.2, 2026 + seed)
            for label, numeric, categorical, backend in CONFIGS:
                with tempfile.TemporaryDirectory() as tmp:
                    outcome = one_run(kernel, fit, holdout, target, positive, numeric, categorical, backend,
                                      seed, Path(tmp))
                outcome.update({"dataset": dataset, "seed": seed, "config": label, "fit_rows": len(fit),
                                "holdout_rows": len(holdout)})
                runs.append(outcome)
                print(f"[{time.time() - started:6.0f}s] {dataset:8s} seed {seed} {label:52s} "
                      f"worst {outcome['worst']:.3f}  bytes {outcome['model_bytes'] + outcome['encoder_bytes']:,}",
                      flush=True)
    rows = summarize(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"format": "dope-quickstart-ablation-v1", "evidence_status": "local_engineering_ablation",
               "seeds": list(SEEDS), "adult_fit_rows": ADULT_FIT_ROWS, "synthetic_size": "1n",
               "summary": rows, "runs": runs}
    (args.out / "ablation.json").write_text(json.dumps(payload, indent=1, default=float) + "\n")
    (args.out / "ablation.md").write_text(markdown(rows))
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
