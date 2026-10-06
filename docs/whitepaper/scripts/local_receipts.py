#!/usr/bin/env python3
"""Copy local receipt facts into committed JSON extracts.

Scratch is not on every CI runner. When the directories below are mounted,
this refreshes the extracts. When they are absent, the committed files stay.
Test-file contents are not read. Only file names are counted.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "generated"
SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
REPLAY = SCRATCH / "dope-s3-loss-log-v1" / "replay"
BEYOND = SCRATCH / "beyondarena-prepared-v1"
S3_WORKERS = SCRATCH / "s3-v1" / "prepared" / "worker"


def _write(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def replay():
    if not REPLAY.is_dir():
        return
    payload = {}
    for profile in ("features12_steps2048", "features12_steps8192"):
        values = []
        for path in sorted((REPLAY / profile).glob("*/receipt.json")):
            document = json.loads(path.read_text())
            values.append(float(document["elapsed_seconds"]))
        values.sort()
        payload[profile] = {
            "max": values[-1],
            "median": statistics.median(values),
            "min": values[0],
            "n": len(values),
            "sum": sum(values),
        }
    _write("replay-cost.json", payload)


def beyond():
    fits = BEYOND / "fits"
    if not fits.is_dir():
        return
    families = []
    for name in sorted(path.name for path in (fits / "features12_steps2048").iterdir()):
        slots = {}
        fit_rows = None
        for profile, key in (
            ("features12_steps2048", "steps2048"),
            ("features12_steps8192", "steps8192"),
        ):
            directory = fits / profile / name
            receipt = json.loads((directory / "receipt.json").read_text())
            report = json.loads((directory / "report.json").read_text())
            last = None
            with (directory / "loss.tsv").open() as handle:
                for line in handle:
                    if line.strip():
                        last = line.rstrip("\n").split("\t")
            fit_rows = report["rows"]
            slots[key] = {
                "bytes": receipt["artifact_bytes"],
                "elapsed_seconds": receipt["elapsed_seconds"],
                "official_test_opened": receipt["official_test_opened"],
                "status": receipt["status"],
                "validation_loss": float(last[2]),
                "within_l3_cap": receipt["within_l3_cap"],
            }
        families.append({"fit_rows": fit_rows, "name": name, **slots})
    _write(
        "beyond-fit.json",
        {
            "families": families,
            "source": "beyondarena-prepared-v1/fits receipt.json, report.json, and the final loss.tsv row",
            "validation_column": "third TSV field, the hidden-basis validation loss",
        },
    )


def counts():
    if not S3_WORKERS.is_dir():
        return
    workers = [path for path in S3_WORKERS.iterdir() if path.is_dir()]
    beyond_workers = [path for path in (BEYOND / "worker").iterdir() if path.is_dir()]
    evaluators = [path for path in (BEYOND / "evaluator").iterdir() if path.is_dir()]
    _write(
        "provenance-counts.json",
        {
            "beyond_evaluator_test_csv": sum((path / "test.csv").exists() for path in evaluators),
            "beyond_worker_test_csv": sum((path / "test.csv").exists() for path in beyond_workers),
            "beyond_workers": len(beyond_workers),
            "note": "File-name counts only. Test file contents were not read.",
            "s3_train_csv": sum((path / "train.csv").exists() for path in workers),
            "s3_validation_csv": sum((path / "validation.csv").exists() for path in workers),
            "s3_worker_test_csv": sum((path / "test.csv").exists() for path in workers),
            "s3_workers": len(workers),
        },
    )


if __name__ == "__main__":
    replay()
    beyond()
    counts()
    print("local receipt extracts refreshed where scratch was mounted")
