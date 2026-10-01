"""Author CTGAN/TVAE adapters and author-library regression efficacy.

Only common-numeric training/validation inputs enter this research adapter.
The sampling artifact contains the fitted transformer and generator; CTGAN's
training row indices and the loss history are removed before serialization.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .score import sha256


VERSIONS = {"ctgan": "0.12.1", "rdt": "1.22.0", "sdmetrics": "0.32.0"}


def check_versions() -> None:
    from importlib.metadata import version
    if any(version(name) != expected for name, expected in VERSIONS.items()):
        raise ValueError("SDV dependency version differs from source audit")


def load_table(path: Path):
    import numpy as np
    import pandas as pd
    if path.name == "test.csv" or "evaluator" in path.parts:
        raise ValueError("validation adapter cannot read sealed test input")
    table = pd.read_csv(path, header=None)
    if table.shape[1] < 2 or len(table) < 2 or not np.isfinite(table.to_numpy()).all():
        raise ValueError("invalid common-numeric input")
    table.columns = [f"c{i}" for i in range(table.shape[1] - 1)] + ["target"]
    return table


def fit(method: str, train: Path, config: dict, seed: int, artifact: Path) -> dict:
    import numpy as np
    import torch
    from ctgan import CTGAN, TVAE
    check_versions()
    if method not in ("CTGAN", "TVAE") or config.get("enable_gpu") is not True:
        raise ValueError("expected a GPU CTGAN or TVAE configuration")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    table = load_table(train)
    if not ((table >= 0) & (table <= 1)).all().all():
        raise ValueError("training input is outside common-numeric support")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(16)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(16 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    model = {"CTGAN": CTGAN, "TVAE": TVAE}[method](**config)
    model.set_random_state(seed)
    discrete = [c for c in table if table[c].isin([0.0, 1.0]).all()]
    model.fit(table, discrete_columns=discrete)
    # Neither container is used by upstream's sample() path. The current
    # DataSampler has no retained data matrix; fail closed if that changes.
    if method == "CTGAN":
        if hasattr(model._data_sampler, "_data"):
            raise ValueError("upstream sampler unexpectedly retains source rows")
        model._data_sampler._rid_by_cat_cols = []
    model.loss_values = None
    artifact.mkdir(parents=True, exist_ok=True)
    model.save(artifact / "model.pt")
    metadata = {"method": method, "columns": len(table.columns), "version": 1,
                "sampling_device": "cpu", "support": "clip_to_unit_interval",
                "removed": ["loss_values", "training_row_indices"] if method == "CTGAN"
                else ["loss_values"]}
    (artifact / "model.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
    return {"files": ["model.pt", "model.json"], "training_rows": len(table),
            "gpu": torch.cuda.get_device_name(0),
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()}


def sample(artifact: Path, rows: int, seed: int, output: Path) -> dict:
    import numpy as np
    import torch
    check_versions()
    if rows < 1:
        raise ValueError("sample row count must be positive")
    metadata = json.loads((artifact / "model.json").read_text())
    # Load only a hash-verified locally produced artifact, never user pickle.
    model = torch.load(artifact / "model.pt", map_location="cpu", weights_only=False)
    model.set_device(torch.device("cpu"))
    torch.set_num_threads(1)
    if metadata["method"] == "CTGAN" and (
            hasattr(model._data_sampler, "_data") or model._data_sampler._rid_by_cat_cols):
        raise ValueError("sampling artifact retains training-row containers")
    model.set_random_state(seed)
    values = model.sample(rows).to_numpy(dtype=float)
    if values.shape != (rows, metadata["columns"]) or not np.isfinite(values).all():
        raise ValueError("invalid generated table")
    clipped = int(((values < 0) | (values > 1)).sum())
    np.savetxt(output, np.clip(values, 0, 1), delimiter=",", fmt="%.17g")
    return {"rows": rows, "clipped_values": clipped, "sha256": sha256(output)}


def efficacy(synthetic: Path, validation: Path, seed: int = 1729) -> dict:
    import numpy as np
    from sdmetrics.single_table import LinearRegression, MLPRegressor
    check_versions()
    train, valid = load_table(synthetic), load_table(validation)
    components = {}
    for metric in (LinearRegression, MLPRegressor):
        np.random.seed(seed)
        random.seed(seed)
        components[metric.__name__] = float(metric.compute(
            train_data=train, test_data=valid, target="target"))
    if not all(np.isfinite(value) for value in components.values()):
        raise ValueError("undefined native efficacy")
    return {"objective": "sdmetrics_mean_regression_r2", "direction": "maximize",
            "components": components, "value": sum(components.values()) / len(components),
            "partition": "validation", "seed": seed,
            "synthetic_sha256": sha256(synthetic), "validation_sha256": sha256(validation),
            "implementation_sha256": sha256(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    action = request.pop("action")
    for key in ("train", "artifact", "output", "synthetic", "validation"):
        if key in request:
            request[key] = Path(request[key])
    result = {"fit": fit, "sample": sample, "efficacy": efficacy}[action](**request)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
