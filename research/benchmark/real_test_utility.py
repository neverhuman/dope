"""Sealed-test TSTR/TRTR utility on real held-out rows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .admission import assess
from .manifest import digest
from .pilot_metrics import _loss, _model
from .score import artifact_inventory, sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark")


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_numeric(path: Path):
    import numpy as np

    try:
        table = np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError):
        raise ValueError("numeric table could not be loaded") from None
    if table.ndim != 2 or len(table) == 0 or not np.isfinite(table).all():
        raise ValueError("numeric table is empty or nonfinite")
    return table


def retention(null_loss: float, trtr_loss: float, tstr_loss: float) -> dict:
    informative = null_loss - trtr_loss >= 0.01 * abs(null_loss)
    return {"null_loss": null_loss, "trtr_loss": trtr_loss, "tstr_loss": tstr_loss,
            "informative": informative,
            "retention": ((null_loss - tstr_loss) / (null_loss - trtr_loss)
                          if informative and null_loss > trtr_loss else None),
            "low_signal_noninferior": (tstr_loss <= trtr_loss + 0.01 * abs(null_loss)
                                         if not informative else None)}


def evaluate(repo_root: Path, lock_root: Path, worker_dir: Path, evaluator_dir: Path,
             samples: dict[int, tuple[Path, Path]], output: Path, seed: int = 1729) -> dict:
    decision = assess(repo_root, lock_root)
    if not decision["admitted"]:
        raise ValueError("final benchmark admission blocked")
    if set(samples) != {1, 4}:
        raise ValueError("n and 4n sample cells are required")
    if not output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("metric receipts must stay on benchmark scratch")
    worker_manifest = json.loads((worker_dir / "worker-manifest.json").read_text())
    test_manifest = json.loads((evaluator_dir / "manifest.json").read_text())
    frozen = json.loads((lock_root / "datasets.lock.json").read_text())["datasets"]
    locked = next((row for row in frozen if row["id"] == test_manifest["id"]), None)
    if locked is None or any(locked.get(key) != test_manifest.get(key) for key in
                             ("source_row_hash", "projection_sha256", "split", "projected_files")):
        raise ValueError("evaluator manifest differs from frozen dataset lock")
    if (worker_manifest["dataset_id"] != test_manifest["id"]
            or worker_manifest["projection_sha256"] != test_manifest["projection_sha256"]
            or worker_manifest["split_hashes"] != {name: test_manifest["split"]["hashes"][name]
                                                  for name in ("train", "validation")}
            or sha256(worker_dir / "train.csv") != test_manifest["projected_files"]["train"]
            or sha256(evaluator_dir / "test.csv") != test_manifest["projected_files"]["test"]):
        raise ValueError("frozen train/test manifest digest mismatch")
    projection = json.loads((worker_dir / "projection.json").read_text())
    task = projection["task"]
    if task not in ("binary", "regression"):
        raise ValueError("unsupported task")
    train, test = load_numeric(worker_dir / "train.csv"), load_numeric(evaluator_dir / "test.csv")
    if train.shape[1] != test.shape[1] or len(train) != worker_manifest["train_rows"]:
        raise ValueError("real train/test shape mismatch")
    import numpy as np
    import sklearn
    import catboost
    from sklearn.metrics import log_loss, mean_squared_error

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    if task == "binary" and not set(np.unique(y_train)).issubset({0.0, 1.0}):
        raise ValueError("invalid binary target")
    null_prediction = float(y_train.mean())
    null_loss = (float(log_loss(y_test, np.full(len(y_test),
                       np.clip(null_prediction, 1e-9, 1 - 1e-9)), labels=[0, 1]))
                 if task == "binary" else
                 float(mean_squared_error(y_test, np.full(len(y_test), null_prediction))))
    real_models = {}
    for auditor in ("catboost", "linear", "mlp"):
        try:
            model = _model(auditor, task, seed).fit(x_train, y_train)
            real_models[auditor] = _loss(model, x_test, y_test, task)
        except (ValueError, RuntimeError):
            real_models[auditor] = None
    report = {"format": "dope-benchmark-real-test-utility", "version": 1,
              "dataset": test_manifest["id"], "task": task,
              "implementation_sha256": sha256(Path(__file__)),
              "auditor_implementation_sha256": sha256(Path(__file__).with_name("pilot_metrics.py")),
              "dependency_versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__,
                                      "catboost": catboost.__version__},
              "train_sha256": sha256(worker_dir / "train.csv"),
              "test_sha256": sha256(evaluator_dir / "test.csv"),
              "projection_sha256": worker_manifest["projection_sha256"],
              "rows": {"train": len(train), "test": len(test)},
              "auditor_seed": seed, "auditors": {}, "null_loss": null_loss}
    fit_keys = set()
    sample_seeds = set()
    for multiplier in (1, 4):
        sample_path, receipt_path = samples[multiplier]
        receipt = json.loads(receipt_path.read_text())
        expected_rows = len(train) * multiplier
        fit_dir = receipt_path.parent
        fit_receipt = json.loads((fit_dir / "fit-receipt.json").read_text())
        fit_identity = fit_receipt["fit_identity"]
        inventory, artifact_bytes = artifact_inventory(fit_dir / "artifact",
                                                       fit_receipt["artifact_files"])
        if (fit_receipt.get("status") != "ok"
                or fit_receipt.get("fit_key") != digest(fit_identity)
                or fit_dir.name != fit_receipt["fit_key"]
                or inventory != fit_receipt["artifact_inventory"]
                or artifact_bytes != fit_receipt["artifact_bytes"]
                or fit_identity.get("dataset") != test_manifest["id"]
                or fit_identity.get("projection") != test_manifest["projection_sha256"]
                or fit_identity.get("split") != worker_manifest["split_hashes"]
                or fit_identity.get("sample_seeds") != [101, 211, 307]
                or fit_identity.get("size_multipliers") != [1, 2, 4, 8]
                or receipt.get("fit_key") != fit_receipt["fit_key"]
                or receipt.get("run_key") != digest({**fit_identity,
                    "sample_seed": receipt.get("sample_seed"), "row_count": expected_rows})
                or receipt_path != fit_dir / f"{receipt['run_key']}.receipt.json"
                or sample_path != fit_dir / f"{receipt['run_key']}.csv"
                or receipt.get("artifact_sampling_verified") is not True):
            raise ValueError("sample is not bound to frozen fit and artifact")
        if (receipt.get("status") != "ok" or receipt.get("row_count") != expected_rows
                or receipt.get("sample_sha256") != sha256(sample_path)):
            raise ValueError("sample receipt digest or row count mismatch")
        fit_keys.add(receipt["fit_key"])
        sample_seeds.add(receipt["sample_seed"])
        synthetic = load_numeric(sample_path)
        if synthetic.shape != (expected_rows, train.shape[1]):
            raise ValueError("synthetic shape mismatch")
        if task == "binary" and not set(np.unique(synthetic[:, -1])).issubset({0.0, 1.0}):
            raise ValueError("invalid synthetic binary target")
        x_synthetic, y_synthetic = synthetic[:, :-1], synthetic[:, -1]
        values = {}
        for auditor in ("catboost", "linear", "mlp"):
            if real_models[auditor] is None:
                values[auditor] = {"status": "real_model_failed", "retention": None}
                continue
            try:
                model = _model(auditor, task, seed).fit(x_synthetic, y_synthetic)
                tstr = _loss(model, x_test, y_test, task)
                if not math.isfinite(tstr):
                    raise ValueError("nonfinite model loss")
                values[auditor] = {"status": "ok", **retention(null_loss, real_models[auditor], tstr)}
            except (ValueError, RuntimeError):
                values[auditor] = {"status": "synthetic_model_failed", "retention": None}
        report["auditors"][str(multiplier)] = {
            "sample_sha256": receipt["sample_sha256"], "receipt_sha256": sha256(receipt_path),
            "sample_seed": receipt["sample_seed"], "rows": expected_rows, "metrics": values}
    if len(fit_keys) != 1 or len(sample_seeds) != 1:
        raise ValueError("n and 4n came from different fits or sample seeds")
    report["fit_key"] = fit_keys.pop()
    if output.exists():
        if json.loads(output.read_text()) != report:
            raise ValueError("existing utility receipt differs")
    else:
        write_once(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker_dir", type=Path)
    parser.add_argument("evaluator_dir", type=Path)
    parser.add_argument("sample_n", type=Path)
    parser.add_argument("receipt_n", type=Path)
    parser.add_argument("sample_4n", type=Path)
    parser.add_argument("receipt_4n", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lock-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = evaluate(args.repo_root, args.lock_root, args.worker_dir, args.evaluator_dir,
                      {1: (args.sample_n, args.receipt_n),
                       4: (args.sample_4n, args.receipt_4n)}, args.output)
    print(json.dumps({"output": str(args.output), "fit_key": result["fit_key"]}, sort_keys=True))


if __name__ == "__main__":
    main()
