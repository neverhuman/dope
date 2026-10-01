"""Prove the News target-weight confirmation is byte-identical to q10."""

from __future__ import annotations

import json
from pathlib import Path

from research.benchmark.manifest import digest
from research.benchmark.score import sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
ROUND = ROOT / "dope-news-autoreg-confirm-v2"
Q10 = ROOT / "dope-news-q10-v1"
PACKED = ROOT / "dope-news-q10-packed-v2"
OUT = ROUND / "identity.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def build() -> dict:
    round_lock_path = ROUND / "round.lock.json"
    dispatch_path = ROUND / "dispatch-receipt.json"
    q10_lock_path = Q10 / "round.lock.json"
    q10_fit_path = Q10 / "fit.json"
    q10_sample_lock_path = Q10 / "sample.lock.json"
    packed_lock_path = PACKED / "round.lock.json"
    packed_receipt_path = PACKED / "receipt.json"
    current_lock = read(round_lock_path)
    dispatch = read(dispatch_path)
    old_lock = read(q10_lock_path)
    old_fit = read(q10_fit_path)
    q10_samples = read(q10_sample_lock_path)
    packed_lock = read(packed_lock_path)
    packed_receipt = read(packed_receipt_path)
    current_fit_path = Path(dispatch["fit_receipt_path"])
    old_fit_path = Path(old_fit["fit_receipt_path"])
    current_fit = read(current_fit_path)
    prior_fit = read(old_fit_path)
    worker = ROOT / "prepared-v2/worker/News"
    if (current_lock["format"] != "dope-news-autoregressive-confirmation-round"
            or current_lock["source_sha256"] != sha256(Path(__file__).with_name("fit_v2.py"))
            or dispatch["round_lock_sha256"] != sha256(round_lock_path)
            or dispatch["fit_receipt_sha256"] != sha256(current_fit_path)
            or dispatch["status"] != "ok"
            or current_fit["status"] != "ok"
            or current_fit["identity"]["target_weight"] != 4.0
            or current_fit["identity"]["seed"] != 23
            or old_lock["format"] != "dope-pilot-overcap-news-q10-confirmation-fit"
            or old_fit["round_sha256"] != sha256(q10_lock_path)
            or old_fit["fit_receipt_sha256"] != sha256(old_fit_path)
            or prior_fit["status"] != "ok"
            or prior_fit["identity"]["target_weight"] != 2.0
            or prior_fit["identity"]["seed"] != 23
            or current_fit["identity"]["candidate"]
            != prior_fit["identity"]["candidate"]
            or current_fit["identity"]["structural_penalty"]
            != prior_fit["identity"]["structural_penalty"]
            or current_fit["identity"]["binary_sha256"]
            != prior_fit["identity"]["binary_sha256"]
            or current_fit["identity"]["projection_sha256"]
            != prior_fit["identity"]["projection_sha256"]
            or current_fit["identity"]["train_sha256"]
            != prior_fit["identity"]["train_sha256"]
            or current_fit["identity"]["validation_sha256"]
            != prior_fit["identity"]["validation_sha256"]
            or current_fit["artifact_sha256"]
            != prior_fit["artifact_sha256"]
            or current_fit["artifact_bytes"] != prior_fit["artifact_bytes"]
            or sha256(current_fit_path.with_suffix(".dpk"))
            != current_fit["artifact_sha256"]
            or sha256(old_fit_path.with_suffix(".dpk"))
            != prior_fit["artifact_sha256"]
            or q10_samples["source_sha256"] != sha256(Q10 / "sample.py")
            or packed_lock["fit_receipt_sha256"] != sha256(old_fit_path)
            or packed_receipt["round_lock_sha256"] != sha256(packed_lock_path)
            or packed_receipt["artifact_bytes"] != 9_205
            or len(packed_receipt["sample_checks"]) != 12
            or packed_receipt["sample_checks_exact"] != 12
            or sha256(worker / "projection.json")
            != current_fit["identity"]["projection_sha256"]
            or (worker / "test.csv").exists()
            or current_lock["official_tests_opened"] is not False
            or q10_samples["official_tests_opened"] is not False
            or packed_receipt["official_tests_opened"] is not False):
        raise ValueError("News autoregressive/q10 identity evidence changed")
    for cell, row in zip(packed_lock["sample_cells"],
                         packed_receipt["sample_checks"]):
        path = PACKED / "sample-checks" / f"{digest(cell)}.json"
        check = read(path)
        if (sha256(path) != row["sample_check_sha256"]
                or check["status"] != "exact"
                or check["cell"] != cell
                or check["sample_sha256"] != row["reference_sample_sha256"]):
            raise ValueError("News q10 deterministic sample replay changed")
    return {"format": "dope-news-autoregressive-q10-artifact-identity",
            "version": 1, "source_sha256": sha256(Path(__file__)),
            "confirmation_round_sha256": sha256(round_lock_path),
            "confirmation_dispatch_sha256": sha256(dispatch_path),
            "confirmation_fit_receipt_sha256": sha256(current_fit_path),
            "q10_round_sha256": sha256(q10_lock_path),
            "q10_fit_receipt_sha256": sha256(old_fit_path),
            "q10_sample_round_sha256": sha256(q10_sample_lock_path),
            "packed_round_sha256": sha256(packed_lock_path),
            "packed_receipt_sha256": sha256(packed_receipt_path),
            "fitted_model_sha256": current_fit["artifact_sha256"],
            "raw_artifact_bytes": current_fit["artifact_bytes"],
            "packed_q10_artifact_bytes": packed_receipt["artifact_bytes"],
            "conclusion": "target_weights_2_and_4_produced_identical_seed23_model_bytes",
            "additional_sample_jobs_skipped": 6,
            "reason": "same model, binary, projection, fit seed, and deterministic sample seeds; q10 samples already replayed",
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


if __name__ == "__main__":
    evidence = build()
    if OUT.exists():
        if read(OUT) != evidence:
            raise ValueError("prior News artifact identity changed")
    else:
        with OUT.open("x") as stream:
            json.dump(evidence, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"conclusion": evidence["conclusion"],
                      "identity_receipt_sha256": sha256(OUT)}, sort_keys=True))
