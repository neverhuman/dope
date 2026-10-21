#!/usr/bin/env python3
"""DOPE on a real, open table: encode, select, compile, inspect, sample, evaluate.

    cargo build --release --bin dope-kernel
    python3 examples/quickstart/quickstart.py --dataset adult

or on your own CSV:

    python3 examples/quickstart/quickstart.py --csv my_table.csv --target churned

The script downloads the UCI Adult table (CC BY 4.0, DOI 10.24432/C5XW20) once into a
local cache, verifies its SHA-256, and writes everything it produces under ``--out``:
the encoder state, the compiled artifact, decoded synthetic rows in the original
schema, and ``report.json`` with bytes and retention per auditor. Compilation and
sampling read only the fit rows and the artifact respectively; the holdout rows are
read only by the evaluation step.

This is a local engineering demonstration, not the paper's benchmark: one table, one
split, small auditor recipes. It makes no privacy claim.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from dope_eval import default_auditors, retention_report, worst_retention  # noqa: E402
from dope_select import PORTFOLIO, select_backend  # noqa: E402
from dope_tables import TableEncoder, read_dope_samples  # noqa: E402

ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                 "hours_per_week", "native_country", "income"]
DATASETS = {
    "adult": {
        "title": "UCI Adult (census income), CC BY 4.0, DOI 10.24432/C5XW20",
        "url": "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "sha256": "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
        "target": "income", "positive": ">50K",
    },
    "abalone": {
        "title": "UCI Abalone, CC BY 4.0, DOI 10.24432/C55C7W",
        "url": "https://archive.ics.uci.edu/static/public/1/abalone.zip",
        "sha256": "755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4",
        "target": "rings", "positive": None,
    },
}
MAX_SAMPLE_ROWS = 100_000


def say(step: str, text: str) -> None:
    print(f"  {step:<9} {text}", flush=True)


def cached_archive(name: str, cache: Path) -> bytes:
    spec = DATASETS[name]
    path = cache / Path(spec["url"]).name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"  download  {spec['url']}", flush=True)
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            payload = response.read()
        path.write_bytes(payload)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != spec["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest} does not match the pinned {spec['sha256']}")
    return payload


def load_dataset(name: str, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, object, str]:
    """Return (fit rows, holdout rows, target, positive class, title)."""
    spec = DATASETS[name]
    archive = zipfile.ZipFile(io.BytesIO(cached_archive(name, cache)))
    if name == "adult":
        read = lambda member, skip: pd.read_csv(  # noqa: E731
            io.BytesIO(archive.read(member)), header=None, names=ADULT_COLUMNS, skipinitialspace=True,
            na_values="?", skiprows=skip)
        fit, holdout = read("adult.data", 0), read("adult.test", 1)
        holdout["income"] = holdout["income"].str.rstrip(".")
        return fit, holdout, spec["target"], spec["positive"], spec["title"]
    columns = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
               "viscera_weight", "shell_weight", "rings"]
    table = pd.read_csv(io.BytesIO(archive.read("abalone.data")), header=None, names=columns)
    fit, holdout = split(table, 0.2, 2026)
    return fit, holdout, spec["target"], spec["positive"], spec["title"]


def split(table: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = np.random.default_rng(seed).permutation(len(table))
    cut = int(round(len(table) * fraction))
    return (table.iloc[order[cut:]].reset_index(drop=True), table.iloc[order[:cut]].reset_index(drop=True))


def find_kernel(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("DOPE_KERNEL"),
                  str(REPO / "target" / "release" / "dope-kernel"),
                  str(REPO / "target" / "debug" / "dope-kernel"), shutil.which("dope-kernel")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise SystemExit("dope-kernel not found: run `cargo build --release --bin dope-kernel` "
                     "or pass --kernel /path/to/dope-kernel")


def run(command: list[str]) -> dict:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"command failed: {' '.join(command[:2])}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        return json.loads(result.stdout) if result.stdout.strip().startswith("{") and len(lines) > 1 \
            else json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        return {}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} GB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    source.add_argument("--csv", type=Path, help="your own headered CSV")
    parser.add_argument("--target", help="target column of --csv")
    parser.add_argument("--holdout", type=Path, help="holdout CSV for --csv (default: a seeded 20%% split)")
    parser.add_argument("--positive", help="positive class of a binary --csv target")
    parser.add_argument("--drop", nargs="*", default=[], help="columns to leave out (IDs, free text, ...)")
    parser.add_argument("--backend", default="auto",
                        help="'auto' (default) picks from a small portfolio by K-fold transfer on the fit rows; "
                             "or name a frozen backend such as chow_liu, or 'search' for the built-in tournament")
    parser.add_argument("--numeric", choices=("robust", "minmax", "quantile"), default="robust")
    parser.add_argument("--categorical", choices=("target_ordered", "onehot", "auto"), default="target_ordered")
    parser.add_argument("--multiplier", type=float, default=1.0, help="synthetic rows per fit row (paper: 4)")
    parser.add_argument("--fit-rows", type=int, default=None, help="subsample the fit rows (faster)")
    parser.add_argument("--auditors", default=",".join(default_auditors()))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--kernel", help="path to the dope-kernel binary")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=REPO / "target" / "quickstart" / "cache")
    args = parser.parse_args(argv)
    if args.csv is None and args.dataset is None:
        args.dataset = "adult"

    kernel = find_kernel(args.kernel)
    if args.csv is not None:
        if not args.target:
            parser.error("--csv needs --target")
        table = pd.read_csv(args.csv)
        if args.holdout:
            fit, holdout = table, pd.read_csv(args.holdout)
        else:
            fit, holdout = split(table, 0.2, 2026)
        target, positive, title = args.target, args.positive, args.csv.name
        name = args.csv.stem
    else:
        fit, holdout, target, positive, title = load_dataset(args.dataset, args.cache)
        name = args.dataset
    if args.fit_rows and args.fit_rows < len(fit):
        fit = fit.sample(n=args.fit_rows, random_state=args.seed).reset_index(drop=True)
    out = (args.out or REPO / "target" / "quickstart" / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nDOPE quickstart · {title}")
    print(f"  {len(fit):,} fit rows × {fit.shape[1]} columns · target '{target}' · "
          f"{len(holdout):,} untouched holdout rows\n")

    # 1. Encode: fit-row-only, reversible, every column into [0, 1].
    timer = time.perf_counter()
    encoder = TableEncoder(target, numeric=args.numeric, categorical=args.categorical,
                           drop=args.drop, positive=positive).fit(fit)
    encoder.write_dataset_dir(fit, out)
    enc_bytes = encoder.byte_breakdown()
    kinds = {}
    for spec in encoder.columns:
        kinds[spec.kind] = kinds.get(spec.kind, 0) + 1
    say("1 encode", f"{len(encoder.columns)} features ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}) "
        f"+ {encoder.task} target → [0,1]  ·  encoder.bin {enc_bytes['total_bytes']:,} B "
        f"({enc_bytes['transform_bytes']} B transforms + {enc_bytes['label_bytes']} B names/labels)  "
        f"[{time.perf_counter() - timer:.1f}s]")

    # 2. Compile: the native compiler reads only train.csv and writes the artifact.
    timer = time.perf_counter()
    selection = None
    if args.backend == "auto":
        selection = select_backend(kernel, fit, target, positive=encoder.classes[1] if encoder.task == "binary"
                                   else None, encoder_options={"numeric": args.numeric,
                                                               "categorical": args.categorical,
                                                               "drop": args.drop}, seed=args.seed)
        args.backend = selection.backend
        ranked = sorted(selection.scores.items(), key=lambda kv: -kv[1])
        say("2 select", "3-fold transfer on the fit rows: " + ", ".join(f"{b} {v:.3f}" for b, v in ranked)
            + f"  [{time.perf_counter() - timer:.1f}s]")
        timer = time.perf_counter()
    command = [kernel, "compile", "--dataset-dir", str(out), "--task", encoder.task,
               "--out", str(out / "model.dpk"), "--report", str(out / "compile-report.json"),
               "--seed", str(args.seed), "--deadline-seconds", "120"]
    if args.backend != "search":
        command += ["--candidate", args.backend]
    run(command)
    model_bytes = (out / "model.dpk").stat().st_size
    report = json.loads((out / "compile-report.json").read_text())
    say("3 compile", f"dope-kernel compile → model.dpk {model_bytes:,} B  ·  candidate "
        f"{report.get('selected_candidate')}  [{time.perf_counter() - timer:.1f}s]")

    # 4. Inspect: the artifact is a small, readable program.
    inspection = run([kernel, "inspect", "--kernel", str(out / "model.dpk")])
    (out / "inspect.json").write_text(json.dumps(inspection, indent=1) + "\n")
    say("4 inspect", f"marginals ×{len(inspection.get('marginals', []))} · dependence "
        f"{inspection.get('dependence')} · target {inspection.get('target')} · noise {inspection.get('noise')} · "
        f"{inspection.get('quantization_bits')}-bit")

    # 5. Sample from the artifact alone, then decode to the original schema.
    rows = int(min(MAX_SAMPLE_ROWS, max(2, round(args.multiplier * len(fit)))))
    timer = time.perf_counter()
    run([kernel, "sample", "--kernel", str(out / "model.dpk"), "--rows", str(rows),
         "--out", str(out / "samples.csv"), "--seed", "101"])
    synthetic = encoder.inverse_transform(read_dope_samples(out / "samples.csv"), seed=args.seed)
    synthetic.to_csv(out / "synthetic.csv", index=False)
    say("5 sample", f"{rows:,} rows from the artifact alone → synthetic.csv (original columns and labels)  "
        f"[{time.perf_counter() - timer:.1f}s]")

    # 6. Evaluate: train on real vs synthetic rows, score both on the untouched holdout.
    timer = time.perf_counter()
    results = retention_report(fit, holdout, synthetic, target, encoder.task, positive=positive,
                               auditors=[a.strip() for a in args.auditors.split(",") if a.strip()])
    say("6 evaluate", f"{len(results)} auditors trained on real rows and on synthetic rows, "
        f"scored on {len(holdout):,} holdout rows  [{time.perf_counter() - timer:.1f}s]\n")
    loss = "log loss" if encoder.task == "binary" else "MSE"
    print(f"  {'auditor':<10}{'null ' + loss:>16}{'real-trained':>15}{'synthetic-trained':>20}{'retention':>12}")
    for r in results:
        value = "uninformative" if r.retention is None else f"{r.retention:.3f}"
        print(f"  {r.auditor:<10}{r.null_loss:>16.4g}{r.real_loss:>15.4g}{r.synthetic_loss:>20.4g}{value:>12}")
    worst = worst_retention(results)

    raw_csv = fit.to_csv(index=False).encode()
    gz = len(gzip.compress(raw_csv, 9, mtime=0))
    state = model_bytes + enc_bytes["total_bytes"]
    print(f"\n  worst-auditor retention {worst:.3f}" if worst is not None else "\n  no informative auditor")
    print(f"  dataset-specific state {state:,} B (model {model_bytes:,} + encoder {enc_bytes['total_bytes']:,})"
          f"  ·  fit table as CSV {human(len(raw_csv))} ({len(raw_csv) / state:,.0f}× larger), "
          f"gzip -9 {human(gz)} ({gz / state:,.0f}× larger)")
    summary = {
        "format": "dope-quickstart-v1", "dataset": title, "task": encoder.task, "target": target,
        "fit_rows": len(fit), "holdout_rows": len(holdout), "synthetic_rows": rows,
        "backend": args.backend, "selected_candidate": report.get("selected_candidate"),
        "selection": None if selection is None else {"portfolio": list(PORTFOLIO), "folds": 3,
                                                       "scores": selection.scores},
        "encoding": {"numeric": args.numeric, "categorical": args.categorical},
        "bytes": {"model": model_bytes, "encoder": enc_bytes, "state": state,
                  "fit_csv": len(raw_csv), "fit_csv_gzip9": gz},
        "auditors": [{"auditor": r.auditor, "null_loss": r.null_loss, "real_loss": r.real_loss,
                      "synthetic_loss": r.synthetic_loss, "retention": r.retention} for r in results],
        "worst_retention": worst, "program": inspection.get("sexpr"),
        "evidence_status": "local_engineering_demo",
    }
    (out / "report.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\n  outputs in {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
