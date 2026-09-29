"""Offline external-learner evidence for one artifact and one sealed split.

This program does not participate in Rust certification or release gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

# Set before importing NumPy or any learner backend.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402 - thread limits must be set before import
from external_learners import MAX_FI_HOLDOUT_ROWS, evaluate_model  # noqa: E402


MODEL_IDS = ("sklearn_linear", "random_forest", "xgboost", "lightgbm")
MODEL_LIBRARIES = {
    "sklearn_linear": "scikit-learn",
    "random_forest": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}
IMPLEMENTATION_VERSION = 1
MAX_INPUT_ROWS = 32_768
MAX_FEATURES = 2_000


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_numeric_csv(path: Path, task: str) -> tuple[np.ndarray, np.ndarray]:
    records: list[list[float]] = []
    width = None
    with path.open(newline="", encoding="utf-8") as source:
        for row_number, fields in enumerate(csv.reader(source), start=1):
            if row_number > MAX_INPUT_ROWS:
                raise ValueError("numeric CSV exceeds the external row limit")
            if width is None:
                width = len(fields)
                if width < 2:
                    raise ValueError(
                        "numeric CSV needs at least one feature and a target"
                    )
                if width - 1 > MAX_FEATURES:
                    raise ValueError("numeric CSV exceeds the external feature limit")
            if len(fields) != width:
                raise ValueError(
                    f"row {row_number}: column count differs from first row"
                )
            parsed = []
            for column_number, field in enumerate(fields, start=1):
                try:
                    value = (
                        math.nan
                        if field == "" and column_number < width
                        else float(field)
                    )
                except ValueError as error:
                    raise ValueError(
                        f"row {row_number}, column {column_number}: numeric value required"
                    ) from error
                if math.isinf(value) or (
                    math.isfinite(value) and not 0.0 <= value <= 1.0
                ):
                    raise ValueError(
                        f"row {row_number}, column {column_number}: normalize to [0,1]"
                    )
                if column_number == width and (
                    not math.isfinite(value)
                    or (task == "binary" and value not in (0.0, 1.0))
                ):
                    raise ValueError(
                        f"row {row_number}, column {column_number}: invalid target"
                    )
                parsed.append(value)
            records.append(parsed)
    if not records:
        raise ValueError("numeric CSV has no rows")
    matrix = np.asarray(records, dtype=np.float64)
    return matrix[:, :-1], matrix[:, -1]


def validate_split_manifest(path: Path, real_train_hash: str, holdout_hash: str) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "dope-external-split" or manifest.get("version") != 1:
        raise ValueError("split manifest format or version is invalid")
    if (
        manifest.get("real_train_sha256") != real_train_hash
        or manifest.get("real_holdout_sha256") != holdout_hash
    ):
        raise ValueError(
            "split manifest does not match the supplied real train and holdout files"
        )
    return sha256_bytes(canonical_bytes(manifest))


def run(args: argparse.Namespace) -> dict:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("external validation requires Python 3.12")
    for path in (
        args.real_train,
        args.real_holdout,
        args.synthetic_train,
        args.artifact,
        args.split_manifest,
    ):
        if not path.is_file():
            raise ValueError("required input file is missing")
    hashes = {
        "real_train_sha256": sha256_file(args.real_train),
        "real_holdout_sha256": sha256_file(args.real_holdout),
        "synthetic_train_sha256": sha256_file(args.synthetic_train),
        "artifact_sha256": sha256_file(args.artifact),
    }
    manifest_hash = validate_split_manifest(
        args.split_manifest, hashes["real_train_sha256"], hashes["real_holdout_sha256"]
    )
    split_hash = sha256_bytes(
        canonical_bytes(
            {
                "manifest": manifest_hash,
                "real_train": hashes["real_train_sha256"],
                "real_holdout": hashes["real_holdout_sha256"],
            }
        )
    )
    implementation_hash = sha256_bytes(
        canonical_bytes(
            {
                "runner": sha256_file(Path(__file__)),
                "learners": sha256_file(
                    Path(__file__).with_name("external_learners.py")
                ),
            }
        )
    )
    lock_hash = sha256_file(Path(__file__).with_name("uv.lock"))
    real_train = load_numeric_csv(args.real_train, args.task)
    holdout = load_numeric_csv(args.real_holdout, args.task)
    synthetic_train = load_numeric_csv(args.synthetic_train, args.task)
    widths = {real_train[0].shape[1], holdout[0].shape[1], synthetic_train[0].shape[1]}
    if len(widths) != 1:
        raise ValueError("real, synthetic, and sealed holdout feature counts differ")
    models = []
    for model_id in MODEL_IDS:
        if model_id not in args.models:
            continue
        try:
            model = evaluate_model(
                model_id,
                args.task,
                args.seed,
                real_train,
                synthetic_train,
                holdout,
                args.top_k,
            )
            canonical_bytes(model)
        except (ImportError, ModuleNotFoundError) as error:
            model = {
                "status": "missing_library",
                "model_id": model_id,
                "error_type": type(error).__name__,
            }
        except Exception as error:
            model = {
                "status": "failed",
                "model_id": model_id,
                "error_type": type(error).__name__,
            }
        model["evidence_sha256"] = sha256_bytes(
            canonical_bytes(
                {
                    "artifact_sha256": hashes["artifact_sha256"],
                    "dataset_split_sha256": split_hash,
                    "synthetic_train_sha256": hashes["synthetic_train_sha256"],
                    "seed": args.seed,
                    "implementation_sha256": implementation_hash,
                    "dependency_lock_sha256": lock_hash,
                    "model": model,
                }
            )
        )
        models.append(model)
    model_by_id = {model["model_id"]: model for model in models}
    selected_by_library = {
        library: [
            model_id for model_id in args.models if MODEL_LIBRARIES[model_id] == library
        ]
        for library in set(MODEL_LIBRARIES.values())
    }
    supported_libraries = sorted(
        library
        for library, model_ids in selected_by_library.items()
        if model_ids
        and all(model_by_id[model_id]["status"] == "measured" for model_id in model_ids)
    )
    result = {
        "format": "dope-external-learner-evidence",
        "version": 1,
        "implementation_version": IMPLEMENTATION_VERSION,
        "python_version": platform.python_version(),
        "implementation_sha256": implementation_hash,
        "dependency_lock_sha256": lock_hash,
        "task": args.task,
        "seed": args.seed,
        "threads": 1,
        "permutation_repeats": 3,
        "fi_holdout_row_limit": MAX_FI_HOLDOUT_ROWS,
        "dataset_split_sha256": split_hash,
        "split_manifest_sha256": manifest_hash,
        "input_hashes": hashes,
        "rows": {
            "real_train": len(real_train[1]),
            "synthetic_train": len(synthetic_train[1]),
            "sealed_holdout": len(holdout[1]),
        },
        "features": int(real_train[0].shape[1]),
        "models": models,
        "supported_exact_model_claims": [
            model["model_id"] for model in models if model["status"] == "measured"
        ],
        "supported_exact_fi_claims": [
            model["model_id"]
            for model in models
            if model["status"] == "measured"
            and model["feature_importance"]["applicable"]
            and model["feature_importance"]["spearman"] is not None
        ],
        "supported_exact_library_claims": supported_libraries,
        "withheld_model_claims": [
            model["model_id"] for model in models if model["status"] != "measured"
        ],
        "rust_certification_effect": "none",
    }
    result["evidence_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument("--real-holdout", type=Path, required=True)
    parser.add_argument("--synthetic-train", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=("binary", "regression"), required=True)
    parser.add_argument("--seed", type=int, default=57721)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS)
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        evidence = run(args)
        payload = canonical_bytes(evidence) + b"\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + f".tmp-{os.getpid()}")
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            f"external validation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
