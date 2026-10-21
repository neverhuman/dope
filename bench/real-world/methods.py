"""Synthetic-data methods compared in the benchmark, with byte accounting.

* ``dope``: the Rust CLI; bytes = artifact file size (no extra compression).
* ``kernel``: the research kernel's table workflow (``dope-kernel fit-table`` and
  ``sample-table`` at their defaults); bytes = artifact file size. It accepts
  regression and binary targets only, so a multiclass table is a failed cell.
* ``rows``: the *information-compression* control. The largest prefix of a
  fixed random permutation of real training rows whose CSV, compressed with
  xz (preset 9e), fits the byte budget. Auditors train on those rows as is.
* ``copula`` / ``ctgan`` / ``tvae``: SDV synthesizers at their defaults.
* ``arf``: adversarial random forests (arfpy) at its defaults.
* ``bootstrap``: real rows resampled with replacement (a control, not a
  generator); bytes = the xz-compressed training table.

Generator bytes are the xz-compressed pickle of the fitted object, which is
generous to the baselines (their uncompressed checkpoints are larger).
"""
from __future__ import annotations

import io
import lzma
import os
import pickle
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

DOPE = os.environ.get("DOPE_BIN", str(Path(__file__).resolve().parents[2] / "engine/target/release/dope"))
KERNEL = os.environ.get("DOPE_KERNEL_BIN", str(Path(__file__).resolve().parents[2] / "target/release/dope-kernel"))


def xz(data: bytes) -> int:
    return len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def dope(train_csv: Path, target: str, n: int, seed: int, budget: str, extra: tuple[str, ...] = ()) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        model, out = Path(tmp) / "m.dope", Path(tmp) / "s.csv"
        report = Path(tmp) / "r.json"
        t0 = time.perf_counter()
        subprocess.run([DOPE, "fit", str(train_csv), "--target", target, "--budget", budget, "--seed", str(seed),
                        "--device", "cpu", "-o", str(model), "--report", str(report), "-q", *extra], check=True)
        t1 = time.perf_counter()
        subprocess.run([DOPE, "sample", str(model), "-n", str(n), "--seed", str(seed + 1000), "--device", "cpu",
                        "-o", str(out)], check=True)
        t2 = time.perf_counter()
        import json
        return {"bytes": model.stat().st_size, "synthetic": pd.read_csv(out), "fit_s": t1 - t0,
                "sample_s": t2 - t1, "report": json.loads(report.read_text())}


def kernel(train_csv: Path, target: str, n: int, seed: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        model, out = Path(tmp) / "m.dope", Path(tmp) / "s.csv"
        t0 = time.perf_counter()
        subprocess.run([KERNEL, "fit-table", "--csv", str(train_csv), "--target", target, "--seed", str(seed),
                        "--out", str(model)], check=True, stdout=subprocess.DEVNULL)
        t1 = time.perf_counter()
        subprocess.run([KERNEL, "sample-table", "--model", str(model), "--rows", str(n), "--seed", str(seed + 1000),
                        "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
        t2 = time.perf_counter()
        return {"bytes": model.stat().st_size, "synthetic": pd.read_csv(out), "fit_s": t1 - t0, "sample_s": t2 - t1}


def rows(train: pd.DataFrame, budget: int, seed: int) -> dict:
    perm = np.random.default_rng(seed).permutation(len(train))
    lo, hi = 0, len(train)
    size = lambda r: xz(csv_bytes(train.iloc[perm[:r]])) if r else 0
    if size(hi) <= budget:
        lo = hi
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if size(mid) <= budget else (lo, mid)
    return {"bytes": size(lo), "synthetic": train.iloc[perm[:lo]].reset_index(drop=True), "fit_s": 0.0,
            "sample_s": 0.0, "rows": lo}


def bootstrap(train: pd.DataFrame, n: int, seed: int) -> dict:
    idx = np.random.default_rng(seed).integers(0, len(train), n)
    return {"bytes": xz(csv_bytes(train)), "synthetic": train.iloc[idx].reset_index(drop=True), "fit_s": 0.0,
            "sample_s": 0.0}


def _metadata(train: pd.DataFrame):
    from sdv.metadata import Metadata
    md = Metadata.detect_from_dataframes(data={"t": train})
    for c in train.columns:
        sdtype = "categorical" if train[c].dtype == object else "numerical"
        md.update_column(column_name=c, sdtype=sdtype, table_name="t")
    return md


def sdv(kind: str, train: pd.DataFrame, n: int, seed: int) -> dict:
    import torch
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
    np.random.seed(seed)
    torch.manual_seed(seed)
    cls = {"copula": GaussianCopulaSynthesizer, "ctgan": CTGANSynthesizer, "tvae": TVAESynthesizer}[kind]
    synth = cls(_metadata(train))
    t0 = time.perf_counter()
    synth.fit({"t": train})
    t1 = time.perf_counter()
    syn = synth.sample(num_rows=n)
    if isinstance(syn, dict):
        syn = next(iter(syn.values()))
    t2 = time.perf_counter()
    buf = io.BytesIO()
    pickle.dump(synth, buf)
    return {"bytes": xz(buf.getvalue()), "synthetic": syn, "fit_s": t1 - t0, "sample_s": t2 - t1}


def arf(train: pd.DataFrame, n: int, seed: int) -> dict:
    if not hasattr(np, "in1d"):  # arfpy predates NumPy 2.4's removal of in1d
        np.in1d = np.isin
    from arfpy import arf as arfmod
    np.random.seed(seed)
    # arfpy keeps internal columns (e.g. "cnt") in the same frame; positional names avoid collisions.
    df = train.copy()
    names = list(df.columns)
    df.columns = [f"v{i}" for i in range(len(names))]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("NA").astype("category")
        else:
            df[c] = df[c].fillna(df[c].median())
    t0 = time.perf_counter()
    model = arfmod.arf(x=df, random_state=seed)
    model.forde()
    t1 = time.perf_counter()
    syn = model.forge(n=n)
    syn.columns = [names[int(c[1:])] for c in syn.columns]
    t2 = time.perf_counter()
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return {"bytes": xz(buf.getvalue()), "synthetic": syn, "fit_s": t1 - t0, "sample_s": t2 - t1}
