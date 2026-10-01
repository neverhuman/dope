"""Reconstruct the exact source used by the frozen ARF short-grid round."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .score import sha256


BASE = Path(__file__).with_name("arf_round.py")
PATCH = Path(__file__).parent / "patches/arf_short_round_v1.patch"
BASE_SHA256 = "ca39ab58e922787f7248e6426f5eb0f111aa6823f468d6f6ebcf9c5a5dfef2f9"
PATCH_SHA256 = "3bca2669e651b61448c11cb66f7870c20f8862777c77de3cff38c23f603a72ba"
SOURCE_SHA256 = "7e7a5dc98c5ec13539e6e037c4739c04a6ee0d394b2158b035b6d636017b6b89"


def materialize(output: Path) -> str:
    if sha256(BASE) != BASE_SHA256 or sha256(PATCH) != PATCH_SHA256:
        raise ValueError("ARF short-grid source inputs changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        subprocess.run(["patch", "--silent", "-o", str(output), str(BASE), str(PATCH)],
                       check=True, timeout=10)
    if sha256(output) != SOURCE_SHA256:
        raise ValueError("ARF short-grid source differs from its frozen hash")
    return SOURCE_SHA256


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(materialize(args.output))
