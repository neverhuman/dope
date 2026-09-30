"""Coordinator-only, selected-object JopeDime fetch with SHA-256 checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .score import sha256


BUCKET = "veox-jopedime-datasets-use2"
CATALOG_SHA256 = "ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120"
LIMIT = 200_000_000_000


def used_bytes(root: Path) -> int:
    return sum((Path(directory) / name).stat().st_size
               for directory, _, files in os.walk(root) for name in files)


def fetch_object(key: str, destination: Path, profile: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    subprocess.run(["aws", "--profile", profile, "s3", "cp",
                    f"s3://{BUCKET}/{key}", str(destination), "--only-show-errors"],
                   check=True, capture_output=True, text=True, timeout=300)


def fetch(dataset_hash: str, root: Path, profile: str) -> dict:
    if root.resolve().is_relative_to(Path(__file__).resolve().parents[2]) or root.resolve().is_relative_to(Path("/tmp")):
        raise ValueError("data must live outside worktrees and /tmp")
    catalog_path = root / "catalog/v1/catalog.jsonl"
    fetch_object("catalog/v1/catalog.jsonl", catalog_path, profile)
    if sha256(catalog_path) != CATALOG_SHA256:
        raise ValueError("catalog digest mismatch")
    selected = None
    for line in catalog_path.open():
        row = json.loads(line)
        if row["dataset_hash"] == dataset_hash:
            selected = row
            break
    if selected is None:
        raise ValueError("dataset hash absent from frozen catalog")
    license_info = selected["provenance"]["license"]
    if (selected["origin"] != "real" or selected["task"] not in ("binary", "regression")
            or license_info.get("status") != "recorded" or license_info.get("spdx") != "MIT"
            or not license_info.get("evidence_url")):
        raise ValueError("dataset is outside rights-cleared supplement")
    if used_bytes(root) + sum(file["bytes"] for file in selected["files"].values()) > LIMIT:
        raise ValueError("coordinator scratch ceiling reached")
    destination = root / "datasets" / dataset_hash
    source_manifest = destination / "manifest.json"
    fetch_object(f"datasets/{dataset_hash}/manifest.json", source_manifest, profile)
    manifest = json.loads(source_manifest.read_text())
    if manifest["dataset_hash"] != dataset_hash or manifest["files"] != selected["files"]:
        raise ValueError("dataset manifest disagrees with frozen catalog")
    files = {}
    for kind, metadata in selected["files"].items():
        path = destination / ("meta.json" if kind == "meta" else f"{kind}.csv")
        fetch_object(f"blobs/sha256/{metadata['sha256']}", path, profile)
        if sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
            raise ValueError("blob digest or byte count mismatch")
        files[kind] = str(path)
    return {"format": "dope-benchmark-source-fetch", "dataset_hash": dataset_hash,
            "catalog_sha256": CATALOG_SHA256, "license": license_info,
            "files": {kind: {"path": files[kind], "sha256": selected["files"][kind]["sha256"]}
                      for kind in files}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_hash")
    parser.add_argument("root", type=Path)
    parser.add_argument("--profile", default="veox-jepson-prod")
    args = parser.parse_args()
    print(json.dumps(fetch(args.dataset_hash, args.root, args.profile), sort_keys=True))


if __name__ == "__main__":
    main()
