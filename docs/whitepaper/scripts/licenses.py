#!/usr/bin/env python3
"""Read pinned LICENSE files into a committed license ledger.

The paper paragraph is generated from this JSON. A missing LICENSE file stays
absent. The script does not copy a lock SPDX string in place of the file, and
it does not open an official test.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "whitepaper" / "generated" / "licenses.json"
SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
LOCK_PATH = REPO / "research" / "benchmark" / "methods.lock.json"
DATA_LOCK = REPO / "research" / "benchmark" / "results" / "s3-data.lock.json"

# Local pins. Keys are methods.lock method ids except the shared CTGAN tree,
# which is also the TVAE pin. Paths are the unpacked snapshots already audited.
PINNED = {
    "CTGAN": SCRATCH / "source-snapshots/ctgan-826da23f/sdv-dev-CTGAN-826da23/LICENSE",
    "TVAE": SCRATCH / "source-snapshots/ctgan-826da23f/sdv-dev-CTGAN-826da23/LICENSE",
    "TabSyn": SCRATCH / "method-source-audits/tabsyn-v1/source/LICENSE",
    "TabDDPM": SCRATCH / "method-source-audits/tabddpm-v1/source/LICENSE.md",
    "ForestDiffusion/Forest-Flow": (
        SCRATCH / "method-source-audits/forestdiffusion-v1/source/Python-Package/base-ForestDiffusion/LICENSE.txt"
    ),
    "ARF": SCRATCH / "source-snapshots/arfpy-unpacked/arfpy-8b63c1b3999981125b4af2828ff52cba8e29169d/LICENSE",
}


def _lock_methods():
    document = json.loads(LOCK_PATH.read_text())
    return document["methods"]


def _identify(text):
    """Heading, version, licensor, licensed work, and copyright line as written."""
    lines = [line.strip() for line in text.splitlines()]
    filled = [line for line in lines if line]
    if not filled:
        raise ValueError("LICENSE file is empty")
    heading = filled[0]
    version = None
    licensor = None
    licensed_work = None
    copyright_line = None
    for line in filled[:40]:
        if version is None and line.startswith("Version "):
            version = line.split(",", 1)[0].removeprefix("Version ").strip()
        if line.startswith("Licensor:"):
            licensor = line.split(":", 1)[1].strip()
        if line.startswith("Licensed Work:"):
            licensed_work = line.split(":", 1)[1].strip()
        if copyright_line is None and line.startswith("Copyright"):
            copyright_line = line
    if heading.startswith("Business Source License"):
        name = heading
    elif heading == "MIT License":
        name = "MIT License"
    elif heading == "Apache License":
        name = "Apache License" if version is None else f"Apache License {version}"
    else:
        raise ValueError(f"unrecognized LICENSE heading: {heading!r}")
    return {
        "name": name,
        "licensor": licensor,
        "licensed_work": licensed_work,
        "copyright": copyright_line,
    }


def _entry_from_bytes(body, evidence, lock_sha):
    """Hash the file bytes. Text mode would rewrite CRLF and miss the lock hash."""
    if isinstance(body, str):
        raise TypeError("LICENSE bytes are required so the hash matches the pin")
    digest = hashlib.sha256(body).hexdigest()
    if lock_sha and digest != lock_sha:
        raise ValueError(f"LICENSE hash {digest} disagrees with the lock for {evidence}")
    found = _identify(body.decode("utf-8"))
    found["sha256"] = digest
    found["lock_sha256"] = lock_sha
    found["evidence"] = evidence
    return found


def _pmlb():
    document = json.loads(DATA_LOCK.read_text())
    entries = document.get("entries") or []
    if not entries:
        raise ValueError("data lock has no dataset entries")
    notes = set()
    for entry in entries:
        license_block = entry.get("license") or {}
        if license_block.get("spdx") != "MIT":
            raise ValueError(f"dataset {entry.get('id')} license spdx is not MIT")
        note = license_block.get("note")
        if note != "source_license: MIT (PMLB)":
            raise ValueError(f"dataset {entry.get('id')} license note is not the PMLB MIT note")
        notes.add(note)
    return {"spdx": "MIT", "entries": len(entries), "note": "source_license: MIT (PMLB)"}


def _repository():
    body = (REPO / "LICENSE").read_bytes()
    found = _identify(body.decode("utf-8"))
    if found["name"] != "MIT License":
        raise ValueError("repository LICENSE heading is not MIT License")
    found["sha256"] = hashlib.sha256(body).hexdigest()
    found["evidence"] = "LICENSE"
    return found


def _fetch(url):
    raw = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    request = urllib.request.Request(raw, headers={"User-Agent": "dope-paper-licenses"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    if b"Business Source License" not in body and b"MIT License" not in body and b"Apache License" not in body:
        raise ValueError(f"fetched file is not a recognized license: {raw}")
    return body, raw


def _kept(existing, key):
    methods = (existing or {}).get("methods") or {}
    entry = methods.get(key)
    if not entry or not entry.get("sha256") or not entry.get("name"):
        return None
    return entry


def build():
    existing = json.loads(OUT.read_text()) if OUT.is_file() else None
    methods = _lock_methods()
    recorded = {}
    for key, path in PINNED.items():
        lock = methods[key]
        evidence = lock.get("license_evidence")
        lock_sha = lock.get("license_sha256")
        if path.is_file():
            recorded[key] = _entry_from_bytes(path.read_bytes(), evidence, lock_sha)
            continue
        kept = _kept(existing, key)
        if kept is None:
            raise FileNotFoundError(f"pinned LICENSE missing and no committed entry: {key}")
        if lock_sha and kept.get("sha256") != lock_sha:
            raise ValueError(f"committed {key} hash disagrees with the lock")
        if kept.get("evidence") != evidence:
            raise ValueError(f"committed {key} evidence URL changed")
        recorded[key] = kept
    copula = methods["GaussianCopula"]
    evidence = copula["license_evidence"]
    try:
        body, raw = _fetch(evidence)
        recorded["GaussianCopula"] = _entry_from_bytes(body, evidence, copula.get("license_sha256"))
        recorded["GaussianCopula"]["fetched"] = raw
    except Exception as exc:
        kept = _kept(existing, "GaussianCopula")
        if kept is None or kept.get("evidence") != evidence:
            raise RuntimeError(f"Copulas LICENSE could not be read: {exc}") from exc
        recorded["GaussianCopula"] = kept
    payload = {
        "format": "dope-paper-licenses",
        "version": 1,
        "official_tests_opened": False,
        "pmlb": _pmlb(),
        "repository": _repository(),
        "sdv": {
            "name": None,
            "note": "pinned source snapshot has no SDV LICENSE file",
        },
        "methods": recorded,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    pins_mounted = all(path.is_file() for path in PINNED.values())
    if OUT.is_file() and OUT.read_text() == text:
        print("licenses.json unchanged")
        return 0
    if not pins_mounted:
        raise RuntimeError("licenses.json would change and a pinned LICENSE file is not mounted")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
