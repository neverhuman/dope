"""Offline fit/sample adapters. External methods are added only after source audit."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def fit(method: str, train: Path, metadata: dict, config: dict, seed: int,
        artifact_dir: Path, binary: Path | None = None) -> list[str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if method == "dope":
        if binary is None:
            raise ValueError("DOPE binary is required")
        cmd = [str(binary), "compile", "--dataset-dir", str(train.parent),
               "--task", metadata["task"], "--out", str(artifact_dir / "model.dpk"),
               "--tier", config["tier"], "--seed", str(seed),
               "--deadline-seconds", str(config["deadline_seconds"])]
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       timeout=config["deadline_seconds"] + 60)
        evidence_dir = artifact_dir.parent / "fit_evidence"
        for sidecar in artifact_dir.glob("model.dpk.*"):
            evidence_dir.mkdir(exist_ok=True)
            sidecar.replace(evidence_dir / sidecar.name)
        return ["model.dpk"]
    if method == "independent_marginals":
        import numpy as np
        table = np.loadtxt(train, delimiter=",", ndmin=2)
        bins = config["bins"]
        if bins not in (8, 16, 32):
            raise ValueError("bins outside locked search space")
        columns = []
        for index in range(table.shape[1]):
            if index == table.shape[1] - 1 and metadata["task"] == "binary":
                columns.append({"kind": "binary", "p": float(table[:, index].mean())})
            else:
                counts, _ = np.histogram(table[:, index], bins=bins, range=(0.0, 1.0))
                columns.append({"kind": "histogram", "p": (counts / counts.sum()).tolist()})
        (artifact_dir / "model.json").write_text(json.dumps({"columns": columns, "bins": bins},
                                                      sort_keys=True, separators=(",", ":")))
        return ["model.json"]
    if method == "GaussianCopula":
        import pandas as pd
        from copulas.multivariate import GaussianMultivariate
        from copulas.univariate import GaussianUnivariate, Univariate
        distribution = {"Univariate": Univariate, "GaussianUnivariate": GaussianUnivariate}.get(config["distribution"])
        if distribution is None:
            raise ValueError("copula distribution outside locked search space")
        table = pd.read_csv(train, header=None)
        table.columns = [f"c{i}" for i in range(table.shape[1])]
        model = GaussianMultivariate(distribution=distribution, random_state=seed)
        model.fit(table)
        (artifact_dir / "model.json").write_text(json.dumps(
            {"model": model.to_dict(), "task": metadata["task"]},
            sort_keys=True, separators=(",", ":")))
        return ["model.json"]
    raise ValueError("method lacks an audited adapter")


def sample(method: str, artifact_dir: Path, row_count: int, seed: int,
           output: Path, binary: Path | None = None) -> None:
    if method == "dope":
        if binary is None:
            raise ValueError("DOPE binary is required")
        subprocess.run([str(binary), "sample", "--kernel", str(artifact_dir / "model.dpk"),
                        "--rows", str(row_count), "--seed", str(seed), "--out", str(output)],
                       check=True, capture_output=True, text=True, timeout=600)
        return
    if method == "independent_marginals":
        import numpy as np
        model = json.loads((artifact_dir / "model.json").read_text())
        rng = np.random.default_rng(seed)
        result = np.empty((row_count, len(model["columns"])))
        for index, column in enumerate(model["columns"]):
            if column["kind"] == "binary":
                result[:, index] = rng.binomial(1, column["p"], size=row_count)
            else:
                bins = rng.choice(model["bins"], size=row_count, p=column["p"])
                result[:, index] = (bins + rng.random(row_count)) / model["bins"]
        np.savetxt(output, result, fmt="%.17g", delimiter=",")
        return
    if method == "GaussianCopula":
        import numpy as np
        from copulas.multivariate import GaussianMultivariate
        artifact = json.loads((artifact_dir / "model.json").read_text())
        model = GaussianMultivariate.from_dict(artifact["model"])
        model.random_state = np.random.RandomState(seed)
        result = np.clip(model.sample(row_count).to_numpy(dtype=float), 0.0, 1.0)
        if artifact["task"] == "binary":
            result[:, -1] = (result[:, -1] >= 0.5).astype(float)
        np.savetxt(output, result, fmt="%.17g", delimiter=",")
        return
    raise ValueError("method lacks an audited adapter")
