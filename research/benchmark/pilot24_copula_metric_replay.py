"""Recompute every published Copula metric from sealed-validation samples."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from . import pilot_metrics
from .manifest import digest
from .score import sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
RUN = ROOT / "copula-matrix-v1"
OUT = RUN / "metric-replay-v1"
LOCK = OUT / "round.lock.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def payload(metric: dict) -> dict:
    if not isinstance(metric.get("metric_seconds"), (int, float)):
        raise ValueError("Copula metric timing missing")
    return {key: value for key, value in metric.items() if key != "metric_seconds"}


def freeze() -> Path:
    validation_path = RUN / "validation-v2.lock.json"
    validation = read(validation_path)
    if (validation["script_sha256"] != sha256(
            Path(__file__).with_name("pilot24_copula_validate_v2.py"))
            or validation["metric_sha256"] != sha256(Path(pilot_metrics.__file__))
            or validation["official_tests_opened"] is not False):
        raise ValueError("Copula validation source changed")
    cells = []
    for metric_path in sorted((RUN / "validation-metrics").glob("*/*.json")):
        receipt = read(metric_path)
        dataset = receipt["dataset"]
        worker = ROOT / "prepared-v2/worker" / dataset
        sample_receipt_path = Path(receipt["sample_receipt_path"])
        sample_path = sample_receipt_path.with_suffix("").with_suffix(".csv")
        if (dataset not in ("Adult", "California", "News")
                or (worker / "test.csv").exists()
                or not sample_path.resolve().is_relative_to((RUN / "results").resolve())
                or not metric_path.resolve().is_relative_to(
                    (RUN / "validation-metrics").resolve())
                or receipt["official_tests_opened"] is not False
                or receipt["mfs_v2"] is not None or receipt["ptf_v1"] is not None
                or receipt["metric_source_sha256"] != sha256(
                    Path(pilot_metrics.__file__))
                or receipt["sample_receipt_sha256"] != sha256(sample_receipt_path)
                or read(sample_receipt_path)["sample_sha256"] != sha256(sample_path)):
            raise ValueError("Copula metric input lineage changed")
        manifest = read(worker / "worker-manifest.json")
        if (manifest["dataset_id"] != dataset
                or any(sha256(worker / f"{part}.csv")
                       != manifest["projected_hashes"][part]
                       for part in ("train", "validation"))):
            raise ValueError("Copula worker input changed")
        cells.append({"dataset": dataset,
                      "metric_path": str(metric_path),
                      "metric_sha256": sha256(metric_path),
                      "sample_path": str(sample_path),
                      "sample_sha256": sha256(sample_path),
                      "train_sha256": manifest["projected_hashes"]["train"],
                      "validation_sha256": manifest["projected_hashes"]["validation"],
                      "task": read(worker / "projection.json")["task"],
                      "seed": receipt["sample_seed"]})
    if len(cells) != 24 or len({digest(cell) for cell in cells}) != 24:
        raise ValueError("Copula replay matrix is incomplete")
    lock = {"format": "dope-copula-validation-metric-replay", "version": 1,
            "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "metric_source_sha256": sha256(Path(pilot_metrics.__file__)),
            "validation_v2_lock_sha256": sha256(validation_path),
            "comparison_policy": "exact_metric_payload_except_elapsed_metric_seconds",
            "cells": cells, "official_tests_opened": False,
            "mfs_v2": None, "ptf_v1": None, "production_certified": False}
    write_once(LOCK, lock)
    return LOCK


def replay(lock_path: Path = LOCK) -> dict:
    lock = read(lock_path)
    if (lock["format"] != "dope-copula-validation-metric-replay"
            or lock["source_sha256"] != sha256(Path(__file__))
            or lock["metric_source_sha256"] != sha256(Path(pilot_metrics.__file__))
            or lock["validation_v2_lock_sha256"] != sha256(
                RUN / "validation-v2.lock.json")
            or lock["comparison_policy"]
            != "exact_metric_payload_except_elapsed_metric_seconds"
            or len(lock["cells"]) != 24
            or lock["official_tests_opened"] is not False):
        raise ValueError("Copula replay lock changed")
    if not set(range(16, 32)).issubset(os.sched_getaffinity(0)):
        raise ValueError("Copula replay escaped reserved CPU slot")
    receipts = []
    for cell in lock["cells"]:
        worker = ROOT / "prepared-v2/worker" / cell["dataset"]
        metric_path = Path(cell["metric_path"])
        sample_path = Path(cell["sample_path"])
        if ((worker / "test.csv").exists()
                or sha256(metric_path) != cell["metric_sha256"]
                or sha256(sample_path) != cell["sample_sha256"]
                or sha256(worker / "train.csv") != cell["train_sha256"]
                or sha256(worker / "validation.csv") != cell["validation_sha256"]):
            raise ValueError("Copula replay input changed")
        path = OUT / "cells" / f"{digest(cell)}.json"
        original = read(metric_path)["metrics"]
        if path.exists():
            prior = read(path)
            if (prior["metric_sha256"] != cell["metric_sha256"]
                    or prior["metric_payload"] != payload(original)
                    or prior["round_lock_sha256"] != sha256(lock_path)):
                raise ValueError("Copula replay receipt changed")
        else:
            started = time.monotonic()
            repeated = pilot_metrics.measure(worker / "train.csv",
                                             worker / "validation.csv",
                                             sample_path, cell["task"], cell["seed"])
            if payload(original) != payload(repeated):
                raise ValueError("Copula validation metric does not replay exactly")
            prior = {"format": "dope-copula-validation-metric-replay-cell",
                     "version": 1, "cell": cell,
                     "round_lock_sha256": sha256(lock_path),
                     "metric_sha256": cell["metric_sha256"],
                     "metric_payload": payload(original),
                     "replay_seconds": time.monotonic() - started,
                     "status": "exact", "official_tests_opened": False,
                     "mfs_v2": None, "ptf_v1": None}
            write_once(path, prior)
        receipts.append({"cell_digest": digest(cell), "receipt_sha256": sha256(path),
                         "metric_sha256": cell["metric_sha256"]})
        print(json.dumps({"dataset": cell["dataset"],
                          "cell_digest": digest(cell), "status": "exact"}), flush=True)
    report = {"format": "dope-copula-validation-metric-replay-manifest",
              "version": 1, "round_lock_sha256": sha256(lock_path),
              "source_sha256": sha256(Path(__file__)),
              "cells": receipts, "exact_cells": len(receipts),
              "official_tests_opened": False, "mfs_v2": None, "ptf_v1": None}
    manifest = OUT / "manifest.json"
    if manifest.exists():
        if read(manifest) != report:
            raise ValueError("Copula replay manifest changed")
    else:
        write_once(manifest, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "replay"))
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze())
    else:
        print(json.dumps({"exact_cells": replay()["exact_cells"]}))
