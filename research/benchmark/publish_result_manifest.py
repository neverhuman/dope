"""Bind published validation results to immutable receipts and verification."""

from __future__ import annotations

import json
from pathlib import Path

from .manifest import digest
from .score import sha256
from .sdv_round import ROOT


HERE = Path(__file__).parent
RESULTS = HERE / "results"
NAME = "gpu-native-20261001.manifest"


def build() -> dict:
    audit_path = ROOT / "reproduction/audit.lock.json"
    audit = json.loads(audit_path.read_text())
    repeats = []
    for item in audit["jobs"]:
        p = ROOT / "reproduction" / digest(item["job"]) / "receipt.json"
        r = json.loads(p.read_text())
        if (r["audit_lock_sha256"] != sha256(audit_path) or r["job"] != item["job"]
                or r["original_sha256"] != item["original_sha256"]):
            raise ValueError("reproduction identity changed")
        for name, expected in r["evidence_files"].items():
            if sha256(p.parent / name) != expected:
                raise ValueError("reproduction evidence changed")
        repeats.append({"path": str(p), "sha256": sha256(p), "dataset": r["job"]["dataset"],
                        "method": r["job"]["method"], "passed": r["passed"],
                        "exact_artifact_hashes": r["exact_artifact_hashes"],
                        "exact_sample_hash": r["exact_sample_hash"],
                        "native_kpi_absolute_difference": r["native_kpi_absolute_difference"],
                        "fit_seconds": r["fit_seconds"]})
    if len(repeats) != 4:
        raise ValueError("reproduction subset incomplete")
    cross_path = ROOT / "cross-host/verification.json"
    cross = json.loads(cross_path.read_text())
    dope_path = ROOT.parent / "conditional-readout-v1/reproduction/verification.json"
    dope = json.loads(dope_path.read_text())
    for item in dope["checks"]:
        if sha256(Path(item["attempt_path"])) != item["attempt_sha256"]:
            raise ValueError("DOPE reproduction attempt changed")
    verification = []
    archive = ROOT / "published-verification"
    for name in ("just-fast-readout-20261001.log", "just-check-completed-20261001.log",
                 "benchmark-tests-completed-20261001.log", "conditional-readout-gpu-test-20261001.log",
                 "result-verification-20261001.json"):
        source = Path("target") / name
        destination = archive / name
        if destination.exists():
            if sha256(source) != sha256(destination):
                raise ValueError("published verification already exists with different content")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(source.read_bytes())
        verification.append({"path": str(destination), "sha256": sha256(destination)})
    artifacts = {}
    for basename in ("sdv-native-validation", "conditional-gpu-validation"):
        for suffix in (".json", ".schema.json", ".csv", ".svg", ".pdf"):
            p = RESULTS / (basename + suffix)
            artifacts[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return {"format": "dope-validation-publication-manifest", "version": 1,
            "scope": "validation_research_only", "date": "2026-10-01", "artifacts": artifacts,
            "reproduction_lock_path": str(audit_path), "reproduction_lock_sha256": sha256(audit_path),
            "reproductions": repeats, "cross_host_path": str(cross_path),
            "cross_host_sha256": sha256(cross_path), "cross_host_checks": cross,
            "dope_reproduction_path": str(dope_path), "dope_reproduction_sha256": sha256(dope_path),
            "dope_reproduction": dope,
            "verification": verification, "publisher_sha256": sha256(Path(__file__)),
            "ptf_v1": None, "mfs_v2": None, "campaign_complete": False,
            "official_tests_opened": False, "production_certified": False}


def main() -> None:
    import jsonschema
    result = build()
    schema = json.loads((RESULTS / f"{NAME}.schema.json").read_text())
    jsonschema.validate(result, schema)
    path = RESULTS / f"{NAME}.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifacts": len(result["artifacts"]), "reproductions": len(result["reproductions"]),
                      "passed": sum(r["passed"] for r in result["reproductions"])}))


if __name__ == "__main__":
    main()
