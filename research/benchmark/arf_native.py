"""Held-out FORDE density and artifact-only FORGE for the pinned arfpy source.

The author implementation supplies the fitted forest, leaf parameters, and
sampler. This study implementation evaluates its published leaf-mixture
density on training-derived validation rows. The serialized sampler is a
restricted research artifact; it is not a public or release-safe sidecar.
"""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path


LOG_FLOOR = math.log(1e-300)


def prepare_frames(train, validation):
    """Use numeric c0... names and train-observed binary category types."""
    import numpy as np
    import pandas as pd

    if train.shape[1] != validation.shape[1] or train.shape[1] < 2:
        raise ValueError("ARF training/validation width mismatch")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("ARF input is nonfinite")
    names = [f"c{i}" for i in range(train.shape[1])]
    fit = pd.DataFrame(train, columns=names)
    holdout = pd.DataFrame(validation, columns=names)
    categories = []
    for index, name in enumerate(names):
        values = np.unique(train[:, index])
        if len(values) == 2 and np.array_equal(values, [0.0, 1.0]):
            fit[name] = pd.Categorical(fit[name], categories=[0.0, 1.0])
            holdout[name] = pd.Categorical(holdout[name], categories=[0.0, 1.0])
            categories.append(name)
    return fit, holdout, categories


def heldout_mean_log_density(model, validation) -> float:
    """FORDE coverage-weighted product density averaged over forest trees.

    Each validation row visits one leaf per tree. Continuous factors use the
    author's fitted truncated-normal parameters; category factors use its
    leaf probability table. Zero probabilities are floored at 1e-300 before
    the row log is averaged. A factor with invalid fitted parameters makes
    that leaf's contribution zero.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import truncnorm

    names = list(model.orig_colnames)
    if (not isinstance(validation, pd.DataFrame) or list(validation) != names
            or len(validation) == 0 or model.num_trees <= 0):
        raise ValueError("ARF validation shape or forest is invalid")
    numeric = validation.copy()
    for name in names:
        if bool(model.factor_cols[name]):
            codes = pd.Categorical(validation[name], categories=model.levels[name]).codes
            numeric[name] = codes.astype(float)
        else:
            numeric[name] = validation[name].astype(float)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("ARF validation has an unseen category or nonfinite value")
    leaves = model.clf.apply(numeric)
    if leaves.shape != (len(validation), model.num_trees):
        raise ValueError("ARF forest leaf assignment changed")

    coverage = model.bnds.drop_duplicates(["tree", "nodeid"]).set_index(
        ["tree", "nodeid"])["cvg"].to_dict()
    continuous = {(int(tree), int(node), name): row for (tree, node, name), row in
                  model.params.set_index(["tree", "nodeid", "variable"]).iterrows()}
    categories = {}
    for row in model.class_probs.itertuples(index=False):
        key = (int(row.tree), int(row.nodeid), row.variable)
        categories.setdefault(key, {})[float(row.value)] = float(row.prob)
    result = np.full(len(validation), -np.inf)
    for tree in range(model.num_trees):
        component = np.full(len(validation), -np.inf)
        for node in np.unique(leaves[:, tree]):
            cover = coverage.get((tree, int(node)), 0.0)
            if not math.isfinite(cover) or cover <= 0:
                continue
            index = np.flatnonzero(leaves[:, tree] == node)
            contribution = np.full(len(index), math.log(cover / model.num_trees))
            for column, name in enumerate(names):
                key = (tree, int(node), name)
                if bool(model.factor_cols[name]):
                    probabilities = categories.get(key, {})
                    mass = np.array([probabilities.get(value, 0.0)
                                     for value in values[index, column]])
                    with np.errstate(divide="ignore"):
                        contribution += np.log(mass)
                else:
                    row = continuous.get(key)
                    if row is None:
                        contribution[:] = -np.inf
                        break
                    mean, sd, lower, upper = (float(row[field]) for field in
                                              ("mean", "sd", "min", "max"))
                    if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
                        contribution[:] = -np.inf
                        break
                    contribution += truncnorm.logpdf(values[index, column],
                                                     (lower - mean) / sd,
                                                     (upper - mean) / sd,
                                                     loc=mean, scale=sd)
            component[index] = contribution
        result = np.logaddexp(result, component)
    if np.isnan(result).any() or np.isposinf(result).any():
        raise ValueError("ARF native density is invalid")
    value = float(np.maximum(result, LOG_FLOOR).mean())
    if not math.isfinite(value):
        raise ValueError("ARF native density is undefined")
    return value


def save_sampler(model, artifact_dir: Path) -> list[str]:
    """Keep only fitted FORDE factors needed by the author's FORGE sampler."""
    import pandas as pd

    if any(name not in model.__dict__ for name in
           ("x_real", "clf", "bnds", "params", "class_probs")):
        raise ValueError("ARF author model structure changed")
    artifact_dir.mkdir(exist_ok=False)
    frames = {"bounds.csv": model.bnds,
              "continuous.csv": (model.params if len(model.params.columns) else
                                 pd.DataFrame(columns=["tree", "nodeid", "variable",
                                                       "mean", "sd", "min", "max"])),
              "categories.csv": (model.class_probs if len(model.class_probs.columns) else
                                 pd.DataFrame(columns=["tree", "nodeid", "variable",
                                                       "value", "prob"]))}
    files = {}
    for name, frame in frames.items():
        path = artifact_dir / name
        frame.to_csv(path, index=False, float_format="%.17g")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"format": "dope-arf-forge-factors-v1",
                "names": list(model.orig_colnames),
                "factor_columns": [bool(model.factor_cols[name]) for name in model.orig_colnames],
                "object_columns": [bool(model.object_cols[name]) for name in model.orig_colnames],
                "levels": {name: [float(value) for value in model.levels[name]]
                           for name in model.levels},
                "num_trees": int(model.num_trees), "dist": model.dist,
                "files": files, "source_rows_required": False,
                "restricted_research_artifact": True}
    (artifact_dir / "model.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return ["model.json", *frames]


def load_sampler(artifact_dir: Path):
    """Rebuild the author's FORGE sampler from numeric factors only."""
    import pandas as pd
    from arfpy.arf import arf

    metadata_path = artifact_dir / "model.json"
    if not metadata_path.is_file() or metadata_path.stat().st_size > 1_000_000:
        raise ValueError("ARF artifact is absent or too large")
    metadata = json.loads(metadata_path.read_text())
    names = metadata.get("names")
    expected = {"bounds.csv", "continuous.csv", "categories.csv"}
    if (metadata.get("format") != "dope-arf-forge-factors-v1"
            or metadata.get("source_rows_required") is not False
            or metadata.get("restricted_research_artifact") is not True
            or not isinstance(names, list) or not names
            or names != [f"c{i}" for i in range(len(names))]
            or set(metadata.get("files", {})) != expected
            or len(metadata.get("factor_columns", [])) != len(names)
            or len(metadata.get("object_columns", [])) != len(names)
            or not isinstance(metadata.get("num_trees"), int)
            or not 1 <= metadata["num_trees"] <= 100
            or metadata.get("dist") != "truncnorm"):
        raise ValueError("ARF numeric artifact metadata is invalid")
    frames = {}
    for name in expected:
        path = artifact_dir / name
        if (not path.is_file() or path.stat().st_size > 512_000_000
                or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["files"][name]):
            raise ValueError("ARF numeric artifact file changed")
        frames[name] = pd.read_csv(path, float_precision="round_trip")
    model = arf.__new__(arf)
    model.p = len(names)
    model.orig_colnames = names
    model.num_trees = metadata["num_trees"]
    model.factor_cols = pd.Series(metadata["factor_columns"], index=names, dtype=bool)
    model.object_cols = pd.Series(metadata["object_columns"], index=names, dtype=bool)
    model.levels = metadata["levels"]
    model.dist = metadata["dist"]
    model.bnds = frames["bounds.csv"]
    model.params = frames["continuous.csv"]
    model.class_probs = frames["categories.csv"]
    return model
