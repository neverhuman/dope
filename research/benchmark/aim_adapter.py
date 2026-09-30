"""AIM common-numeric pilot adapter with fixed public eight-bin domains."""

from __future__ import annotations

import json
import pickle
from itertools import combinations
from pathlib import Path


def fit(train: Path, metadata: dict, config: dict, seed: int, artifact_dir: Path) -> list[str]:
    import jax
    import numpy as np
    from aim import AIM
    from mbi import Dataset, Domain

    if config["bins"] != 8 or config["max_model_size"] != 80 or config["max_iters"] != 1000:
        raise ValueError("AIM pilot configuration differs from author defaults and fixed domain")
    if config["epsilon"] not in (1.0, 4.0, 10.0):
        raise ValueError("AIM epsilon outside registered grid")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    table = np.loadtxt(train, delimiter=",", ndmin=2)
    if (not np.isfinite(table).all() or (table < 0).any() or (table > 1).any()
            or table.shape[1] != metadata["output_features"] + 1):
        raise ValueError("AIM expects finite common-numeric train rows")
    bins = config["bins"]
    codes = np.minimum((table * bins).astype(np.int32), bins - 1)
    names = tuple(f"c{i}" for i in range(table.shape[1]))
    domain = Domain(names, (bins,) * len(names))
    data = Dataset({name: codes[:, i] for i, name in enumerate(names)}, domain)
    workload = [(pair, 1.0) for pair in combinations(names, 2)]
    delta = min(1e-5, 1 / len(table) ** 2)
    model, _ = AIM(config["epsilon"], delta, prng=np.random.RandomState(seed),
                   max_model_size=80, max_iters=1000).run(data, workload, num_synth_rows=1)
    with (artifact_dir / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (artifact_dir / "model-meta.json").write_text(json.dumps(
        {"bins": bins, "columns": len(names), "task": metadata["task"],
         "epsilon": config["epsilon"], "delta": delta,
         "formal_dp_claim": False, "workload_pairs": len(workload)},
        sort_keys=True, separators=(",", ":")))
    return ["model.pkl", "model-meta.json"]


def sample(artifact_dir: Path, row_count: int, seed: int, output: Path) -> None:
    import numpy as np

    meta = json.loads((artifact_dir / "model-meta.json").read_text())
    with (artifact_dir / "model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    np.random.seed(seed)
    generated = model.synthetic_data(rows=row_count)
    codes = np.column_stack([generated.data[f"c{i}"] for i in range(meta["columns"])])
    values = (codes + 0.5) / meta["bins"]
    if meta["task"] == "binary":
        values[:, -1] = (codes[:, -1] >= meta["bins"] // 2).astype(float)
    np.savetxt(output, values, delimiter=",", fmt="%.17g")
