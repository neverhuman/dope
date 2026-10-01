"""Publish the News target-weight confirmation without duplicate outcomes."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from .pilot_sources.dope_news_autoreg.identity import build as verify_identity
from .score import sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
RESULT = HERE / "results/pilot24-news-autoreg-identity.json"
SOURCES = HERE / "pilot_sources/dope_news_autoreg"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def build() -> dict:
    identity_path = ROOT / "dope-news-autoreg-confirm-v2/identity.json"
    identity = verify_identity()
    if identity != read(identity_path):
        raise ValueError("News artifact identity receipt changed")
    v1_lock_path = ROOT / "dope-news-autoreg-confirm-v1/round.lock.json"
    v1_receipt_path = ROOT / "dope-news-autoreg-confirm-v1/dispatch-receipt.json"
    v2_lock_path = ROOT / "dope-news-autoreg-confirm-v2/round.lock.json"
    v2_receipt_path = ROOT / "dope-news-autoreg-confirm-v2/dispatch-receipt.json"
    v1_lock = read(v1_lock_path)
    v1_receipt = read(v1_receipt_path)
    v2_lock = read(v2_lock_path)
    v2_receipt = read(v2_receipt_path)
    q10_path = HERE / "results/pilot24-news-q10-packed.json"
    q10 = read(q10_path)
    if (v1_lock["source_sha256"] != sha256(SOURCES / "fit.py")
            or v2_lock["source_sha256"] != sha256(SOURCES / "fit_v2.py")
            or identity["source_sha256"] != sha256(SOURCES / "identity.py")
            or v1_receipt["round_lock_sha256"] != sha256(v1_lock_path)
            or v1_receipt["status"] != "admission_blocked"
            or v1_receipt["fit_receipt_sha256"] is not None
            or v2_lock["predecessor_lock_sha256"] != sha256(v1_lock_path)
            or v2_lock["predecessor_failure_receipt_sha256"]
            != sha256(v1_receipt_path)
            or v2_receipt["round_lock_sha256"] != sha256(v2_lock_path)
            or v2_receipt["status"] != "ok"
            or identity["confirmation_dispatch_sha256"] != sha256(v2_receipt_path)
            or q10["dataset"] != "News" or q10["fit_seed"] != 23
            or q10["packed_q10_artifact_bytes"] != identity["packed_q10_artifact_bytes"]
            or q10["packed_sample_checks_exact"] != 12
            or q10["official_tests_opened"] is not False
            or q10["mfs_v2"] is not None or q10["ptf_v1"] is not None):
        raise ValueError("News confirmation source, receipt, or q10 result changed")
    q10_fit = read(Path(ROOT / "dope-news-q10-v1/fit.json"))
    q10_receipt = read(Path(q10_fit["fit_receipt_path"]))
    v2_fit_receipt = read(Path(v2_receipt["fit_receipt_path"]))
    if (v2_fit_receipt["artifact_sha256"] != q10_receipt["artifact_sha256"]
            or v2_fit_receipt["artifact_sha256"] != identity["fitted_model_sha256"]
            or v2_fit_receipt["artifact_bytes"] != q10_receipt["artifact_bytes"]
            or v2_fit_receipt["status"] != "ok"
            or q10_receipt["status"] != "ok"):
        raise ValueError("News seed23 fitted models differ")
    fits = [{"target_weight": weight,
             "fit_receipt_sha256": sha256(path),
             "fit_elapsed_seconds": receipt["elapsed_seconds"],
             "gpu_peak_used_mib": receipt["peak_gpu_used_mib"],
             "gpu_energy_joules_estimate": receipt["energy_joules_estimate"],
             "model_sha256": receipt["artifact_sha256"],
             "raw_artifact_bytes": receipt["artifact_bytes"]}
            for weight, path, receipt in (
                (2.0, Path(q10_fit["fit_receipt_path"]), q10_receipt),
                (4.0, Path(v2_receipt["fit_receipt_path"]), v2_fit_receipt))]
    return {"format": "dope-news-target-weight-identity-validation",
            "version": 1, "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "source_snapshots": {name: sha256(SOURCES / name)
                                 for name in ("fit.py", "fit_v2.py", "identity.py")},
            "receipt_hashes": {"first_admission_failure": sha256(v1_receipt_path),
                               "repaired_round": sha256(v2_lock_path),
                               "repaired_dispatch": sha256(v2_receipt_path),
                               "artifact_identity": sha256(identity_path),
                               "q10_packed_validation_report": sha256(q10_path)},
            "dataset": "News", "candidate": "symbolic_autoregressive_residual",
            "fit_seed": 23, "fits": fits,
            "fitted_model_identical": True,
            "packed_q10_artifact_bytes": identity["packed_q10_artifact_bytes"],
            "additional_sample_jobs_skipped": identity["additional_sample_jobs_skipped"],
            "shared_validation_reference": "pilot24-news-q10-packed.json",
            "notes": ["The seed-11 target-weight-4 result motivated a single frozen seed-23 confirmation fit.",
                      "The first attempt was blocked before fitting by CPU-affinity admission; the versioned repair retained the 600-second and 16-GiB GPU limits.",
                      "At seed 23, target weights 2 and 4 emitted identical model bytes with the same binary and projection, so the existing q10 deterministic samples and validation metrics are reused without counting an independent result.",
                      "The 9,205-byte packed projection remains restricted research evidence; public sidecar safety and release gates are incomplete."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-news-target-weight-identity-validation"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("News target-weight report changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=("target_weight", "fit_elapsed_seconds",
                            "gpu_peak_used_mib", "gpu_energy_joules_estimate",
                            "model_sha256", "raw_artifact_bytes"),
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["fits"])
    RESULT.with_suffix(".csv").write_text(output.getvalue())
    lines = ["# News DOPE target-weight confirmation", "",
             "A frozen seed-23 GPU fit tested target weight 4 after a seed-11 "
             "validation lead. The earlier target-weight-2 q10 fit used the "
             "same candidate, binary, projection, and fit seed. The two model "
             "files have the same SHA-256 digest. Their raw charged artifacts "
             f"are both {report['fits'][0]['raw_artifact_bytes']:,} bytes; "
             "the existing lossless packed q10 artifact is "
             f"{report['packed_q10_artifact_bytes']:,} bytes.", "",
             "| Target weight | Fit seconds | Peak GPU MiB | Raw charged bytes |",
             "|---:|---:|---:|---:|"]
    for fit in report["fits"]:
        lines.append(f"| {fit['target_weight']:.0f} | "
                     f"{fit['fit_elapsed_seconds']:.2f} | "
                     f"{fit['gpu_peak_used_mib']:,} | "
                     f"{fit['raw_artifact_bytes']:,} |")
    lines += ["", "The first confirmation attempt was blocked at admission "
              "before GPU fitting; its receipt and the corrected round source "
              "are retained. Six duplicate n/4n samples were skipped because "
              "the model bytes are identical. The [existing q10 validation "
              "comparison](pilot24-news-q10-packed.md) supplies the shared "
              "outcomes; this confirmation adds no independent DOPE result. "
              "Official tests remain sealed, and MFS-v2, PTF-v1, and production "
              "certification remain null.", ""]
    RESULT.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"identical": report["fitted_model_identical"],
                      "fits": len(report["fits"])}, sort_keys=True))


if __name__ == "__main__":
    main()
