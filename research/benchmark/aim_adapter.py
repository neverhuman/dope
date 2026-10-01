"""AIM common-numeric pilot adapter with fixed public eight-bin domains."""

from __future__ import annotations

import json
import hashlib
from itertools import combinations
from pathlib import Path


def save_model(model: object, artifact_dir: Path) -> None:
    """Store only numeric MBI factors; never persist executable Python objects."""
    import numpy as np

    if model.constraints:
        raise ValueError("AIM constraints need an audited numeric codec")
    domain = model.potentials.domain
    attributes = list(domain.attributes)
    shape = [int(card) for card in domain.shape]
    if not (1 <= len(attributes) <= 128 and len(attributes) == len(shape)
            and attributes == [f"c{i}" for i in range(len(attributes))]
            and all(1 <= card <= 256 for card in shape)):
        raise ValueError("AIM model domain is outside the fixed numeric contract")
    arrays = {}
    vectors = {}
    for kind in ("potentials", "marginals"):
        vector = getattr(model, kind)
        if vector.domain != domain or len(vector.cliques) > 4096:
            raise ValueError("AIM factor domain or count is invalid")
        cliques = []
        for index, clique in enumerate(vector.cliques):
            if len(clique) != len(set(clique)) or not set(clique) <= set(attributes):
                raise ValueError("AIM factor clique is invalid")
            values = np.asarray(vector.tables[clique].values, dtype=np.float64)
            expected = tuple(shape[attributes.index(name)] for name in clique)
            if values.shape != expected or values.size > 2_000_000 or not np.isfinite(values).all():
                raise ValueError("AIM factor shape or values are invalid")
            arrays[f"{kind}_{index}"] = values
            cliques.append(list(clique))
        vectors[kind] = cliques
    total = float(model.total)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("AIM model total is invalid")
    numeric_path = artifact_dir / "model.npz"
    np.savez_compressed(numeric_path, **arrays)
    metadata = {"format": "dope-aim-numeric-v1", "attributes": attributes,
                "shape": shape, "vectors": vectors, "total": total,
                "model_sha256": hashlib.sha256(numeric_path.read_bytes()).hexdigest()}
    (artifact_dir / "model-structure.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")))


def load_model(artifact_dir: Path) -> object:
    import jax.numpy as jnp
    import numpy as np
    from mbi import CliqueVector, Domain, Factor, MarkovRandomField

    structure = json.loads((artifact_dir / "model-structure.json").read_text())
    if structure.get("format") != "dope-aim-numeric-v1":
        raise ValueError("AIM numeric artifact format mismatch")
    attributes, shape = structure["attributes"], structure["shape"]
    if not (isinstance(attributes, list) and isinstance(shape, list)
            and 1 <= len(attributes) <= 128 and len(attributes) == len(shape)
            and attributes == [f"c{i}" for i in range(len(attributes))]
            and all(type(card) is int and 1 <= card <= 256 for card in shape)):
        raise ValueError("AIM numeric artifact domain is invalid")
    numeric_path = artifact_dir / "model.npz"
    if (numeric_path.stat().st_size > 512_000_000 or
            hashlib.sha256(numeric_path.read_bytes()).hexdigest() != structure["model_sha256"]):
        raise ValueError("AIM numeric artifact digest or size mismatch")
    domain = Domain(tuple(attributes), tuple(shape))
    vectors = {}
    with np.load(numeric_path, allow_pickle=False) as archive:
        expected_keys = set()
        for kind in ("potentials", "marginals"):
            cliques = structure["vectors"][kind]
            if not isinstance(cliques, list) or len(cliques) > 4096:
                raise ValueError("AIM numeric artifact cliques are invalid")
            tables = {}
            for index, names in enumerate(cliques):
                if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
                        or len(names) != len(set(names)) or not set(names) <= set(attributes)):
                    raise ValueError("AIM numeric artifact clique is invalid")
                clique = tuple(names)
                key = f"{kind}_{index}"
                expected_keys.add(key)
                values = archive[key]
                expected = tuple(shape[attributes.index(name)] for name in clique)
                if (values.dtype != np.float64 or values.shape != expected
                        or values.size > 2_000_000 or not np.isfinite(values).all()):
                    raise ValueError("AIM numeric artifact factor is invalid")
                tables[clique] = Factor(domain.project(clique), jnp.asarray(values))
            vectors[kind] = CliqueVector(domain, tuple(tuple(names) for names in cliques), tables)
        if set(archive.files) != expected_keys:
            raise ValueError("AIM numeric artifact has unexpected arrays")
    total = structure["total"]
    if not isinstance(total, (float, int)) or not np.isfinite(total) or total <= 0:
        raise ValueError("AIM numeric artifact total is invalid")
    return MarkovRandomField(potentials=vectors["potentials"],
                             marginals=vectors["marginals"], total=total)


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
    save_model(model, artifact_dir)
    (artifact_dir / "model-meta.json").write_text(json.dumps(
        {"bins": bins, "columns": len(names), "task": metadata["task"],
         "epsilon": config["epsilon"], "delta": delta,
         "formal_dp_claim": False, "workload_pairs": len(workload)},
        sort_keys=True, separators=(",", ":")))
    return ["model.npz", "model-structure.json", "model-meta.json"]


def sample(artifact_dir: Path, row_count: int, seed: int, output: Path) -> None:
    import numpy as np

    meta = json.loads((artifact_dir / "model-meta.json").read_text())
    model = load_model(artifact_dir)
    np.random.seed(seed)
    generated = model.synthetic_data(rows=row_count)
    codes = np.column_stack([generated.data[f"c{i}"] for i in range(meta["columns"])])
    values = (codes + 0.5) / meta["bins"]
    if meta["task"] == "binary":
        values[:, -1] = (codes[:, -1] >= meta["bins"] // 2).astype(float)
    np.savetxt(output, values, delimiter=",", fmt="%.17g")
