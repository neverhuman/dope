"""Original MIT ForestDiffusion API with charged, row-independent artifacts.

The study separately implements the author's four-model, five-seed ML-efficacy
objective. No repository-root experiment code with unresolved license coverage
is imported. Common validation outcomes never select the native configuration.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import pickle
import shutil
import sys
import time

SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe(path):
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(SCRATCH):
        raise ValueError("research path outside verified local scratch")
    return path


def read(path):
    return json.loads(Path(path).read_text())


def runtime(path, expected):
    safe(path)
    if sha(path) != expected:
        raise ValueError("ForestDiffusion runtime identity changed")
    lock = read(path)
    if (
        lock["format"] != "dope-forestdiffusion-author-runtime"
        or lock["adapter_sha256"] != sha(__file__)
        or lock["version"] != 1
    ):
        raise ValueError("ForestDiffusion adapter identity changed")
    if {name: version(name) for name in lock["environment"]} != lock["environment"]:
        raise ValueError("ForestDiffusion dependency version changed")
    for root_key, files_key in [
        ("package_root", "author_package_files"),
        ("dependency_root", "dependency_files"),
    ]:
        root = safe(lock[root_key])
        for name, h in lock[files_key].items():
            if sha(safe(root / name)) != h:
                raise ValueError("ForestDiffusion source or dependency changed")
    sys.path.insert(0, lock["package_root"])
    return lock


def table(path, expected):
    import numpy as np
    import pandas as pd

    safe(path)
    if (
        path.name not in ("train.csv", "validation.csv")
        or "evaluator" in path.resolve().parts
        or (path.parent / "test.csv").exists()
        or sha(path) != expected
    ):
        raise ValueError("input is not verified training-derived data")
    try:
        values = pd.read_csv(path, header=None).to_numpy(dtype=float)
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("invalid common-numeric input") from None
    if (
        values.ndim != 2
        or min(values.shape) < 2
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("invalid common-numeric input")
    return values


def inventory(root):
    safe(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("artifact contains a symlink")
        if p.is_file():
            rows.append(
                {
                    "path": p.relative_to(root).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": sha(p),
                }
            )
    return rows


def clean_rows(model):
    import numpy as np

    model.X1 = np.empty((0, model.c))
    model.mask_y = {}
    if model.label_y is not None:
        model.label_y = np.empty(0)
    check_rows(model)


def check_rows(model):
    if (
        model.X1.size
        or model.mask_y
        or model.X_covs is not None
        or (model.label_y is not None and model.label_y.size)
    ):
        raise ValueError("ForestDiffusion artifact retains training rows")


def generate(model, rows, seed):
    import numpy as np

    if type(rows) is not int or rows < 1 or type(seed) is not int:
        raise ValueError("invalid sample request")
    np.random.seed(seed)
    samples = model.generate(batch_size=rows)
    if (
        samples.ndim != 2
        or samples.shape[0] != rows
        or not np.isfinite(samples).all()
        or ((samples < 0) | (samples > 1)).any()
    ):
        raise ValueError("invalid original ForestDiffusion samples")
    return samples


def load_model(request):
    runtime(Path(request["runtime_lock"]), request["runtime_lock_sha256"])
    artifact = safe(request["artifact"])
    actual = inventory(artifact)
    if actual != request["expected_inventory"]:
        raise ValueError("ForestDiffusion artifact inventory changed")
    meta = read(artifact / "adapter.json")
    if (
        meta["runtime_lock_sha256"] != request["runtime_lock_sha256"]
        or meta["source_rows_required"] is not False
    ):
        raise ValueError("ForestDiffusion artifact runtime identity changed")
    # Unpickle only this locally produced model after full byte/hash validation.
    with (artifact / "sampler.pkl").open("rb") as stream:
        model = pickle.load(stream)
    check_rows(model)
    return model, meta


def fit(request):
    runtime(Path(request["runtime_lock"]), request["runtime_lock_sha256"])
    import numpy as np
    from ForestDiffusion import ForestDiffusionModel

    train, valid = Path(request["train"]), Path(request["validation"])
    x = table(train, request["train_sha256"])
    validation = table(valid, request["validation_sha256"])
    if (
        train.parent != valid.parent
        or x.shape[1] != validation.shape[1]
        or set(map(tuple, x)) & set(map(tuple, validation))
    ):
        raise ValueError("training/validation shape or row groups changed")
    projection = safe(request["projection"])
    if (
        projection.name != "projection.json"
        or projection.parent != train.parent
        or sha(projection) != request["projection_sha256"]
    ):
        raise ValueError("frozen projection changed")
    task = read(projection)["task"]
    if task not in ("binary", "regression"):
        raise ValueError("inapplicable target")
    config = request["config"]
    if (
        set(config) != {"n_t", "duplicate_K", "max_depth", "n_estimators"}
        or any(type(v) is not int or v < 1 for v in config.values())
        or config["n_t"] > 50
        or config["duplicate_K"] > 100
        or config["max_depth"] > 7
        or config["n_estimators"] > 100
    ):
        raise ValueError("configuration outside frozen author parameter subset")
    features = x[:, :-1] if task == "binary" else x
    binary_columns = [
        i
        for i in range(features.shape[1])
        if np.array_equal(np.unique(features[:, i]), [0.0, 1.0])
    ]
    if task == "binary" and not np.array_equal(np.unique(x[:, -1]), [0.0, 1.0]):
        raise ValueError("incompatible binary target")
    started = time.monotonic()
    model = ForestDiffusionModel(
        features,
        label_y=x[:, -1] if task == "binary" else None,
        bin_indexes=binary_columns,
        n_jobs=1,
        n_batch=1,
        diffusion_type="flow",
        gpu_hist=True,
        seed=request["seed"],
        **config,
    )
    fit_seconds = time.monotonic() - started
    def booster_devices(value):
        if isinstance(value, list):
            return [device for child in value for device in booster_devices(child)]
        booster = value.get_booster() if hasattr(value, "get_booster") else value
        return [json.loads(booster.save_config())["learner"]["generic_param"]["device"]]
    trained_devices = booster_devices(model.regr)
    if not trained_devices or any(not device.startswith("cuda") for device in trained_devices):
        raise ValueError("ForestDiffusion GPU training was not retained")
    before = generate(model, 64, 101)
    clean_rows(model)
    if not np.array_equal(before, generate(model, 64, 101)):
        raise ValueError("removing row containers changed author samples")
    artifact = safe(request["artifact"])
    artifact.mkdir()
    with (artifact / "sampler.pkl").open("xb") as stream:
        pickle.dump(model, stream, protocol=5)
    shutil.copyfile(projection, artifact / "projection.json")
    metadata = {
        "format": "dope-forestdiffusion-author-sampler",
        "version": 1,
        "task": task,
        "features": x.shape[1],
        "config": config,
        "seed": request["seed"],
        "runtime_lock_sha256": request["runtime_lock_sha256"],
        "source_rows_required": False,
        "restricted_research_artifact": True,
        "resource_overrides": {"n_jobs": 1, "gpu_hist": True},
        "row_removal_parity": "exact",
    }
    (artifact / "adapter.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
    with (artifact / "sampler.pkl").open("rb") as stream:
        loaded = pickle.load(stream)
    check_rows(loaded)
    if not np.array_equal(before, generate(loaded, 64, 101)):
        raise ValueError("serialized original samples changed")
    files = inventory(artifact)
    return {
        "format": "dope-forestdiffusion-author-fit-receipt",
        "status": "ok",
        "fit_seconds": fit_seconds,
        "artifact_bytes": sum(r["bytes"] for r in files),
        "artifact_inventory": files,
        "original_sample_parity": "exact",
        "trained_booster_devices": trained_devices,
        "retained_training_row_containers": False,
        "runtime_lock_sha256": request["runtime_lock_sha256"],
    }


def sample(request):
    model, meta = load_model(request)
    samples = generate(model, request["rows"], request["seed"])
    if samples.shape[1] != meta["features"]:
        raise ValueError("original sampler width changed")
    import numpy as np

    output = safe(request["output"])
    if output.exists():
        raise ValueError("sample output already exists")
    np.savetxt(output, samples, delimiter=",", fmt="%.17g")
    return {"status": "ok", "sample_sha256": sha(output), "rows": len(samples)}


def native(request):
    """Study implementation of author mean ML efficacy, four models × five seeds."""
    runtime(Path(request["runtime_lock"]), request["runtime_lock_sha256"])
    import numpy as np
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import f1_score, r2_score
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier, XGBRegressor

    model, meta = load_model(request)
    validation = table(Path(request["validation"]), request["validation_sha256"])
    synthetic = generate(model, request["rows"], request["seed"])
    x, y = synthetic[:, :-1], synthetic[:, -1]
    vx, vy = validation[:, :-1], validation[:, -1]
    binary = meta["task"] == "binary"
    scores = {name: [] for name in ("linear", "adaboost", "random_forest", "xgboost")}
    if binary and (
        len(np.unique(y)) < 2 or not np.isin(np.unique(vy), np.unique(y)).all()
    ):
        scores = {name: [0.0] * 5 for name in scores}
    else:
        encoder = LabelEncoder().fit(y) if binary else None
        fitted_y = encoder.transform(y) if binary else y
        for seed in range(5):
            models = (
                [
                    LogisticRegression(
                        penalty=None, solver="lbfgs", max_iter=500, random_state=seed
                    ),
                    AdaBoostClassifier(random_state=seed),
                    RandomForestClassifier(max_depth=28, random_state=seed, n_jobs=1),
                    XGBClassifier(reg_lambda=0.0, random_state=seed, n_jobs=1),
                ]
                if binary
                else [
                    LinearRegression(),
                    AdaBoostRegressor(random_state=seed),
                    RandomForestRegressor(max_depth=28, random_state=seed, n_jobs=1),
                    XGBRegressor(
                        objective="reg:squarederror",
                        reg_lambda=0.0,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ]
            )
            for name, estimator in zip(scores, models):
                prediction = estimator.fit(x, fitted_y).predict(vx)
                if binary:
                    prediction = encoder.inverse_transform(prediction.astype(int))
                value = (
                    f1_score(vy, prediction, average="macro")
                    if binary
                    else r2_score(vy, prediction)
                )
                if not math.isfinite(value):
                    raise ValueError("native ML efficacy is undefined")
                scores[name].append(float(value))
    return {
        "name": "author_mean_ml_macro_f1" if binary else "author_mean_ml_r2",
        "implementation": "study_implemented_author_four_model_five_seed_formula",
        "direction": "maximize",
        "value": float(np.mean([np.mean(v) for v in scores.values()])),
        "auditor_seed_scores": scores,
        "auditor_fit_seeds": list(range(5)),
        "shared_kpi_used_for_selection": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["fit", "sample", "native"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = globals()[args.operation](read(safe(args.request)))
    with safe(args.receipt).open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
