"""The interim report fails closed and regenerates its committed tables."""

import json
from pathlib import Path

import pytest

from research.benchmark import pilot24_interim


def test_measure_rejects_claimed_gate() -> None:
    metric = {"mfs_v2": 0.9, "gate_profile_complete": False,
              "utility": {"catboost": {"retention": 0.5}},
              "null_loss": 1, "c2st_auc": 0.5,
              "copy_counts": {"exact": 0},
              "real_vs_real_control_counts": {"exact": 0}}
    with pytest.raises(ValueError, match="claimed gated score"):
        pilot24_interim.measure(metric)
    metric["mfs_v2"] = None
    metric["gate_profile_complete"] = True
    with pytest.raises(ValueError, match="claimed gated score"):
        pilot24_interim.measure(metric)


def test_receipt_path_cannot_open_sealed_or_external_data(tmp_path: Path) -> None:
    worker = tmp_path / "gpu-v3/fits"
    worker.mkdir(parents=True)
    receipt = worker / "attempt-0001.json"
    receipt.write_text("{}")
    assert pilot24_interim.within(tmp_path, "gpu-v3/fits", str(receipt)) == receipt
    with pytest.raises(ValueError, match="training-only scratch"):
        pilot24_interim.within(tmp_path, "gpu-v3/fits", str(tmp_path / "evaluator/test.csv"))
    with pytest.raises(ValueError, match="training-only scratch"):
        pilot24_interim.within(tmp_path, "gpu-v3/fits", "/etc/passwd")


def test_paired_summary_requires_every_frozen_replicate() -> None:
    dope = {(dataset, fit, sample, size): {"retention": 0.75}
            for dataset in pilot24_interim.DATASETS
            for fit in pilot24_interim.SEEDS
            for sample in pilot24_interim.SAMPLE_SEEDS
            for size in pilot24_interim.SIZES}
    compact = {(dataset, method, kind, fit, sample, size): {"retention": 0.5}
               for dataset in pilot24_interim.DATASETS
               for method in pilot24_interim.METHODS
               for kind in ("default", "tuned")
               for fit in pilot24_interim.SEEDS
               for sample in pilot24_interim.SAMPLE_SEEDS
               for size in pilot24_interim.SIZES}
    rows = pilot24_interim.paired_summary(dope, compact)
    assert len(rows) == 24
    assert all(row["paired_replicates"] == 4
               and row["median_paired_difference"] == 0.25 for row in rows)
    del compact["Adult", "Chow-Liu", "default", 11, 101, 1]
    with pytest.raises(KeyError):
        pilot24_interim.paired_summary(dope, compact)


def test_committed_tables_regenerate_from_committed_json() -> None:
    root = Path(__file__).parents[1]
    results = root / "results"
    report = json.loads((results / "pilot24-interim.json").read_text())
    assert report["source_sha256"] == pilot24_interim.sha(root / "pilot24_interim.py")
    assert report["ptf_v1"] is None and report["mfs_v2"] is None
    assert report["official_tests_opened"] is False
    assert pilot24_interim.render_table(report) == (results / "pilot24-interim.md").read_text()
    assert pilot24_interim.render_paired_csv(report) == (
        results / "pilot24-interim-paired.csv").read_text()
