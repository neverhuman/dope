"""Pinned author TabPC APIs for validation research, outside the product runtime.

The Apache-2.0 author implementation imports a GPL-3.0-or-later Cirkit runtime.
The frozen runtime applies only the published CPU memory-stat compatibility
patch. Fit uses the protocol's external validation partition and fits every
preprocessor and the circuit structure on training rows only. Native selection
uses author transformed-space validation mean NLL, never the shared KPI.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from .score import artifact_inventory, sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")


def scratch_path(path: Path) -> Path:
    if path.is_symlink() or not path.resolve().is_relative_to(SCRATCH.resolve()):
        raise ValueError("research path is outside verified local scratch")
    return path


def check_runtime(runtime_lock: Path, runtime_lock_sha256: str) -> dict:
    scratch_path(runtime_lock)
    if sha256(runtime_lock) != runtime_lock_sha256:
        raise ValueError("TabPC runtime lock changed")
    lock = json.loads(runtime_lock.read_text())
    if (lock["format"] != "dope-tabpc-author-runtime" or lock["version"] != 1
            or lock["adapter_sha256"] != sha256(Path(__file__))
            or lock["patch_sha256"] != sha256(Path(__file__).with_name("tabpc-cpu-sampling.patch"))):
        raise ValueError("TabPC adapter or compatibility patch changed")
    if {name: version(name) for name in lock["environment"]} != lock["environment"]:
        raise ValueError("TabPC dependency environment changed")
    for name, expected in lock["source_files"].items():
        path = scratch_path(Path(lock["audit_root"]) / name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("TabPC source or runtime evidence changed")
    for name, expected in lock["dependency_files"].items():
        if sha256(scratch_path(Path(name))) != expected:
            raise ValueError("TabPC dependency source changed")
    sys.path.insert(0, str(scratch_path(Path(lock["cirkit_source"]))))
    sys.path.insert(0, str(scratch_path(Path(lock["author_runtime"]))))
    return lock


def load_table(path: Path, expected_sha256: str):
    import numpy as np
    import pandas as pd
    scratch_path(path)
    if (path.name not in ("train.csv", "validation.csv") or "evaluator" in path.resolve().parts
            or (path.parent / "test.csv").exists() or sha256(path) != expected_sha256):
        raise ValueError("TabPC input is not a verified training partition")
    try:
        frame = pd.read_csv(path, header=None)
        values = frame.to_numpy(dtype=float)
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("invalid common-numeric input") from None
    if (values.shape[0] < 2 or values.shape[1] < 2
            or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError("invalid common-numeric input")
    frame.columns = [f"c{i}" for i in range(frame.shape[1] - 1)] + ["target"]
    return frame


def assert_no_row_container(model) -> None:
    if hasattr(model, "df_train") or model.data.numel() != 0:
        raise ValueError("TabPC artifact retains training-row containers")


def load_artifact(artifact: Path, expected_inventory: list[dict], runtime_lock: Path,
                  runtime_lock_sha256: str):
    check_runtime(runtime_lock, runtime_lock_sha256)
    scratch_path(artifact)
    inventory, _ = artifact_inventory(artifact, [entry["path"] for entry in expected_inventory])
    if inventory != expected_inventory:
        raise ValueError("TabPC locally produced artifact hash changed")
    metadata = json.loads((artifact / "adapter.json").read_text())
    if (metadata["runtime_lock_sha256"] != runtime_lock_sha256
            or metadata["removed_row_tensor"] is not True):
        raise ValueError("TabPC artifact runtime identity changed")
    from src.models.probabilistic_circuit import ProbabilisticCircuit
    # Author pickle loading is allowed only after the complete local artifact
    # inventory has been checked, including preprocessors and circuit weights.
    model = ProbabilisticCircuit.init_from_folder(artifact / "model")
    model.load_(artifact)
    assert_no_row_container(model)
    return model, metadata


def fit(train: Path, validation: Path, projection: Path, train_sha256: str,
        validation_sha256: str, projection_sha256: str, config: dict, seed: int,
        artifact: Path, runtime_lock: Path, runtime_lock_sha256: str) -> dict:
    check_runtime(runtime_lock, runtime_lock_sha256)
    import numpy as np
    import torch
    from src.models.probabilistic_circuit import ProbabilisticCircuit
    from src.nn.optimization import Optimization
    from src.preprocessors.df_type_conversion import TypeConversion
    from src.preprocessors.nan_handler import NanHandler
    from src.preprocessors.df_dequantization import Dequantization
    from src.preprocessors.df_quantile_normalizer import QuantileNormalizer
    from src.preprocessors.string_to_int import StringToInt
    from src.preprocessors.to_tensor import ToTensor
    from src.util import set_seeds
    if not torch.cuda.is_available():
        raise ValueError("TabPC fit requires an admitted GPU")
    if (set(config) != {"epochs", "num_units", "batch_size", "learning_rate"}
            or any(type(config[k]) is not int or config[k] < 1
                   for k in ("epochs", "num_units", "batch_size"))
            or not 0 < config["learning_rate"] <= 1):
        raise ValueError("invalid TabPC configuration")
    training = load_table(train, train_sha256)
    valid = load_table(validation, validation_sha256)
    if (train.parent != validation.parent or list(training.columns) != list(valid.columns)
            or set(map(tuple, training.to_numpy())) & set(map(tuple, valid.to_numpy()))):
        raise ValueError("TabPC fit/validation row groups overlap or differ in shape")
    if (scratch_path(projection).parent != train.parent or projection.name != "projection.json"
            or sha256(projection) != projection_sha256):
        raise ValueError("TabPC projection differs from frozen training inputs")
    torch.set_num_threads(16)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(16 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    set_seeds(seed)
    model = ProbabilisticCircuit(
        optimization=Optimization(
            epochs=config["epochs"], batch_size=config["batch_size"], compile_nn=False,
            lr_scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
            lr_scheduler_params={"patience": 0, "factor": 0.85, "min_lr": 2e-4},
            optimizer_class=torch.optim.RAdam,
            optimizer_hyperparameters={"lr": config["learning_rate"]}, patience=10),
        num_input_units=config["num_units"], num_sum_units=config["num_units"],
        region_graph="chow-liu-tree", bin_for_mi=15, sum_product_layer="cp", seed=seed,
        validation_size=0.0, fit_preprocessor_only_on_train=True,
        preprocessors=[TypeConversion(cat_max_states=50, int_max_states=None),
                       NanHandler(handle_inflated_values=True, handle_missing_values=False),
                       Dequantization(dequantize=True, dequantize_all_floats=False),
                       QuantileNormalizer(), StringToInt(), ToTensor()])
    model.original_size = len(training)
    model.original_pandas_types = training.dtypes.to_dict()
    x_train = model.dataset_to_device(model.preprocessor.fit_transform(training), "cuda:0")
    x_valid = model.dataset_to_device(model.preprocessor.transform(valid), "cuda:0")
    trained = model._train(X_train=x_train, X_val=x_valid)
    _, _, nll = model.compute_log_likelihood(x_valid, "cuda:0", config["batch_size"])
    value = float(nll.item())
    if not np.isfinite(value):
        raise ValueError("undefined author validation likelihood")
    native = {"objective": "author_transformed_validation_mean_nll", "direction": "minimize",
              "value": value, "partition": "validation", "validation_sha256": validation_sha256,
              "preprocessing_fit_partition": "train_only", "fit_seed": seed,
              "not_comparable_across_methods": True}
    peak = {"peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()}
    model.prepare_to_store()
    def sample_probe():
        set_seeds(101)
        return model.generate(num_to_generate=32, device="cpu", batch_size=16, backward=True).to_numpy(dtype=float)
    before = sample_probe()
    metadata = model.data.metadata
    model.data = torch.empty((0, model.num_features))
    model.data.metadata = metadata
    if hasattr(model, "df_train"):
        del model.df_train
    assert_no_row_container(model)
    if not np.array_equal(before, sample_probe()):
        raise ValueError("row-container removal changes author samples")
    scratch_path(artifact)
    artifact.mkdir(parents=True, exist_ok=False)
    model.store(artifact)
    (artifact / "projection.json").write_bytes(projection.read_bytes())
    (artifact / "adapter.json").write_text(json.dumps({
        "method": "TabPC", "columns": len(training.columns), "removed_row_tensor": True,
        "runtime_lock_sha256": runtime_lock_sha256, "sampling_device": "cpu",
        "support": "clip_to_unit_interval", "compatibility_patch": "cuda_memory_stats_guard_only"}, sort_keys=True) + "\n")
    files = [p.relative_to(artifact).as_posix() for p in sorted(artifact.rglob("*")) if p.is_file()]
    inventory, charged = artifact_inventory(artifact, files)
    loaded, _ = load_artifact(artifact, inventory, runtime_lock, runtime_lock_sha256)
    set_seeds(101)
    replay = loaded.generate(num_to_generate=32, device="cpu", batch_size=16, backward=True).to_numpy(dtype=float)
    if not np.array_equal(before, replay):
        raise ValueError("serialized author sample replay differs")
    return {"native_kpi": native, "artifact_inventory": inventory, "artifact_bytes": charged,
            "training_rows": len(training), "validation_rows": len(valid),
            "best_epoch_validation_nll": trained["best_validation_loss"],
            "row_container_removal_sample_replay": "exact", "serialized_sample_replay": "exact",
            "gpu": torch.cuda.get_device_name(0), **peak}


def sample(artifact: Path, expected_inventory: list[dict], rows: int, seed: int, output: Path,
           runtime_lock: Path, runtime_lock_sha256: str) -> dict:
    model, metadata = load_artifact(artifact, expected_inventory, runtime_lock, runtime_lock_sha256)
    import numpy as np
    import torch
    from src.util import set_seeds
    if type(rows) is not int or rows < 1:
        raise ValueError("invalid sample row count")
    scratch_path(output)
    torch.set_num_threads(1)
    set_seeds(seed)
    values = model.generate(num_to_generate=rows, device="cpu", batch_size=1000, backward=True).to_numpy(dtype=float)
    if values.shape != (rows, metadata["columns"]) or not np.isfinite(values).all():
        raise ValueError("invalid generated table")
    clipped = int(((values < 0) | (values > 1)).sum())
    with output.open("x") as stream:
        np.savetxt(stream, np.clip(values, 0, 1), delimiter=",", fmt="%.17g")
    return {"rows": rows, "clipped_values": clipped, "sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    request = json.loads(parser.parse_args().request.read_text())
    action = request.pop("action")
    for key in ("train", "validation", "projection", "artifact", "output", "runtime_lock"):
        if key in request:
            request[key] = Path(request[key])
    result = {"fit": fit, "sample": sample}[action](**request)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
