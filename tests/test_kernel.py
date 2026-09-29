from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dope_kernel.kernel import fit_kernel_from_dir, kernel_to_canonical_json, sample_kernel, save_kernel


def write_dataset(path: Path, task: str = "regression", rows: int = 80) -> np.ndarray:
    rng = np.random.default_rng(123)
    x0 = rng.beta(2.0, 5.0, rows)
    x1 = np.clip(0.65 * x0 + 0.35 * rng.random(rows), 0.0, 1.0)
    x2 = rng.choice([0.0, 0.35, 0.7, 1.0], size=rows, p=[0.1, 0.4, 0.3, 0.2])
    if task == "binary":
        logits = -1.0 + 4.0 * x0 - 2.0 * x2
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(rows) < prob).astype(float)
    else:
        y = np.clip(0.2 + 0.5 * x0 + 0.25 * x1 * x2 + rng.normal(0, 0.04, rows), 0.0, 1.0)
    data = np.column_stack([x0, x1, x2, y])
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path / "train.csv", header=False, index=False)
    pd.DataFrame(data[:20]).to_csv(path / "test.csv", header=False, index=False)
    (path / "meta.json").write_text(json.dumps({"columns": ["secret_name", "leaky_feature", "source_system"]}))
    return data


def test_kernel_is_anonymous_and_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    write_dataset(dataset)

    k1 = fit_kernel_from_dir(dataset, "regression", seed=99)
    k2 = fit_kernel_from_dir(dataset, "regression", seed=99)
    text1 = kernel_to_canonical_json(k1)
    text2 = kernel_to_canonical_json(k2)

    assert text1 == text2
    assert k1["kernel_bytes"] == k2["kernel_bytes"]
    assert k1["description_bits"] == k2["description_bits"]
    assert "secret_name" not in text1
    assert "leaky_feature" not in text1
    assert k1["metadata_policy"]["source_metadata_stored"] is False
    assert k1["metadata_policy"]["feature_order_stored"] is False


def test_sampler_values_and_regression_guard(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    write_dataset(dataset)
    kernel = fit_kernel_from_dir(dataset, "regression", seed=777)

    a = sample_kernel(kernel, 15)
    b = sample_kernel(kernel, 15)

    assert np.array_equal(a, b)
    assert a.shape == (15, 4)
    assert np.all((a >= 0.0) & (a <= 1.0))


def test_cli_sample_writes_headerless_target_last(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    write_dataset(dataset, task="binary")
    kernel_path = tmp_path / "kernel.dk.json"
    synth_path = tmp_path / "synth.csv"

    kernel = fit_kernel_from_dir(dataset, "binary", seed=42)
    save_kernel(kernel, kernel_path)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dope_kernel",
            "sample",
            "--kernel",
            str(kernel_path),
            "--rows",
            "12",
            "--out",
            str(synth_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )

    first_line = synth_path.read_text().splitlines()[0]
    assert not any(ch.isalpha() for ch in first_line)
    data = pd.read_csv(synth_path, header=None).to_numpy()
    assert data.shape == (12, 4)
    assert set(np.unique(data[:, -1])).issubset({0.0, 1.0})
    assert np.all((data >= 0.0) & (data <= 1.0))


def test_cli_fit_eval_and_gp_search_smoke(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    write_dataset(dataset, task="regression", rows=70)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    kernel_path = tmp_path / "fit.dk.json"
    report_path = tmp_path / "report.json"
    best_path = tmp_path / "best.dk.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "dope_kernel",
            "fit",
            "--dataset-dir",
            str(dataset),
            "--task",
            "regression",
            "--out",
            str(kernel_path),
        ],
        check=True,
        cwd=root,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dope_kernel",
            "eval",
            "--real-dir",
            str(dataset),
            "--kernel",
            str(kernel_path),
            "--out",
            str(report_path),
        ],
        check=True,
        cwd=root,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dope_kernel",
            "gp-search",
            "--dataset-dir",
            str(dataset),
            "--budget-secs",
            "0",
            "--out",
            str(best_path),
        ],
        check=True,
        cwd=root,
        env=env,
    )

    report = json.loads(report_path.read_text())
    assert set(report["baselines"]) == {"independent_marginal", "gaussian_copula", "row_bootstrap"}
    assert "rbcs" in report["kernel"]
    best = json.loads(best_path.read_text())
    assert best["format"] == "dope-kernel"
    assert best["gp_search"]["selection"] == "typed_tournament_mutation_v1"
