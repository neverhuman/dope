"""Coordinator-only, selected-object JopeDime fetch with SHA-256 checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from .score import sha256


CATALOG_SHA256 = "ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120"
LIMIT = 200_000_000_000


def validate_bucket(bucket: str | None) -> str:
    """Accept an explicit general-purpose bucket name, never a URI or default.

    This validates supported name syntax; existence, access and dataset rights
    still require their independent checks.
    """
    if (not isinstance(bucket, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None
            or any(part in bucket for part in ("..", ".-", "-."))
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+){3}", bucket)
            or bucket.startswith(("xn--", "sthree-", "amzn-s3-demo-"))
            or bucket.endswith(("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3"))):
        raise ValueError("explicit supported S3 bucket name required")
    return bucket


def used_bytes(root: Path) -> int:
    if not root.is_dir():
        raise NotADirectoryError(root)

    def fail_scan(error: OSError) -> None:
        raise error

    total = 0
    for directory, directories, files in os.walk(root, followlinks=False, onerror=fail_scan):
        for name in files:
            # lstat charges the link itself without following archived evidence
            # into a removed worktree or outside the benchmark scratch root.
            total += (Path(directory) / name).lstat().st_size
        for name in directories:
            path = Path(directory) / name
            if path.is_symlink():
                total += path.lstat().st_size
    return total


def fetch_object(key: str, destination: Path, profile: str | None,
                 *, bucket: str | None = None) -> None:
    bucket = validate_bucket(bucket)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    command = ["aws"] + (["--profile", profile] if profile else [])
    subprocess.run(command + ["s3", "cp", f"s3://{bucket}/{key}", str(destination), "--only-show-errors"],
                   check=True, capture_output=True, text=True, timeout=300)


def fetch(dataset_hash: str, root: Path, profile: str | None,
          *, bucket: str | None = None) -> dict:
    bucket = validate_bucket(bucket)
    if root.resolve().is_relative_to(Path(__file__).resolve().parents[2]) or root.resolve().is_relative_to(Path("/tmp")):
        raise ValueError("data must live outside worktrees and /tmp")
    catalog_path = root / "catalog/v1/catalog.jsonl"
    fetch_object("catalog/v1/catalog.jsonl", catalog_path, profile, bucket=bucket)
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
    fetch_object(f"datasets/{dataset_hash}/manifest.json", source_manifest, profile, bucket=bucket)
    manifest = json.loads(source_manifest.read_text())
    if manifest["dataset_hash"] != dataset_hash or manifest["files"] != selected["files"]:
        raise ValueError("dataset manifest disagrees with frozen catalog")
    files = {}
    for kind, metadata in selected["files"].items():
        path = destination / ("meta.json" if kind == "meta" else f"{kind}.csv")
        fetch_object(f"blobs/sha256/{metadata['sha256']}", path, profile, bucket=bucket)
        if sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
            raise ValueError("blob digest or byte count mismatch")
        files[kind] = str(path)
    return {"format": "dope-benchmark-source-fetch", "dataset_hash": dataset_hash,
            "catalog_sha256": CATALOG_SHA256, "manifest_sha256": sha256(source_manifest),
            "license": license_info,
            "files": {kind: {"path": files[kind], "sha256": selected["files"][kind]["sha256"]}
                      for kind in files}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_hash")
    parser.add_argument("root", type=Path)
    parser.add_argument("--bucket", required=True, help="explicit source bucket name; no default")
    parser.add_argument("--profile", help="optional AWS profile; otherwise use the standard AWS credential chain")
    args = parser.parse_args()
    print(json.dumps(fetch(args.dataset_hash, args.root, args.profile, bucket=args.bucket), sort_keys=True))


if __name__ == "__main__":
    main()
