#!/usr/bin/env python3
"""Generated README surfaces of the DOPE engine benchmark.

The README carries three generated blocks, each a byte-for-byte copy of a file
written by a generator, and generated hero figures:

* ``DOPE1 SCOREBOARD``: ``bench/real-world/results/scoreboard/v2.0.0/readme-block.md``
  (``dope_kpi render-readme`` from the v2.0.0 scoreboard receipts beside it);
* ``DOPE1 RESULTS``: ``bench/real-world/results/readme-block.md``
  (``bench/real-world/render.py`` from the benchmark summary);
* ``DOPE1 WALKTHROUGH``: ``examples/adult/TRANSCRIPT.md``
  (``examples/adult/walkthrough.py`` from a recorded CLI run);
* heroes: the v2.0.0 ``bench/real-world/figures/hero-v2-{light,dark}.svg``
  (``dope_kpi render-hero``), hash-listed in the scoreboard manifest next to the
  receipt digests, and the v1.0.0 ``bench/real-world/figures/hero.png``,
  hash-listed in the render manifest next to the summary digest.

``--write`` splices the files into README.md; ``--check`` (and the public
claims gate) refuses drift, so README numbers are never typed by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOREBOARD = "bench/real-world/results/scoreboard/v2.0.0"
BLOCKS = {
    "DOPE1 SCOREBOARD": f"{SCOREBOARD}/readme-block.md",
    "DOPE1 RESULTS": "bench/real-world/results/readme-block.md",
    "DOPE1 WALKTHROUGH": "examples/adult/TRANSCRIPT.md",
}
HERO = "bench/real-world/figures/hero.png"
MANIFEST = "bench/real-world/results/render-manifest.json"
SUMMARY = "bench/real-world/results/confirm/summary.json"
HEROES_V2 = ("bench/real-world/figures/hero-v2-light.svg", "bench/real-world/figures/hero-v2-dark.svg")
SCOREBOARD_MANIFEST = f"{SCOREBOARD}/render-manifest.json"


def markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"


def _span(readme: str, name: str) -> tuple[int, int] | None:
    begin, end = markers(name)
    if readme.count(begin) != 1 or readme.count(end) != 1 or readme.index(begin) > readme.index(end):
        return None
    return readme.index(begin), readme.index(end) + len(end)


def expected(root: Path, name: str) -> str | None:
    path = root / BLOCKS[name]
    if not path.is_file():
        return None
    text = path.read_text().strip()
    begin, end = markers(name)
    return text if text.startswith(begin) and text.endswith(end) else None


def splice(readme: str, root: Path = ROOT) -> str:
    for name in BLOCKS:
        span, block = _span(readme, name), expected(root, name)
        if span and block:
            readme = readme[:span[0]] + block + readme[span[1]:]
    return readme


def strip(readme: str) -> str:
    """README prose with the generated DOPE1 blocks removed (for language lint)."""
    for name in BLOCKS:
        begin, end = markers(name)
        readme = re.sub(re.escape(begin) + ".*?" + re.escape(end), "", readme, flags=re.S)
    return readme


def failures(readme: str, root: Path = ROOT) -> list[str]:
    out = []
    for name in BLOCKS:
        span = _span(readme, name)
        if span is None:
            out.append(f"README: {name} markers must be present, unique and ordered")
            continue
        block = expected(root, name)
        if block is None:
            out.append(f"README: {name} source {BLOCKS[name]} is missing or malformed")
        elif readme[span[0]:span[1]] != block:
            out.append(f"README: {name} block drifted from {BLOCKS[name]}")
    try:
        manifest = json.loads((root / MANIFEST).read_text())
        if manifest["summary_sha256"] != hashlib.sha256((root / SUMMARY).read_bytes()).hexdigest():
            out.append("README: benchmark summary changed after rendering")
        for rel in (HERO, BLOCKS["DOPE1 RESULTS"]):
            if manifest["files"].get(rel) != hashlib.sha256((root / rel).read_bytes()).hexdigest():
                out.append(f"README: {rel} differs from the render manifest")
    except (OSError, KeyError, ValueError):
        out.append("README: benchmark render manifest is missing or invalid")
    # The scoreboard block names each receipt by its SHA-256; the manifest binds both to the files.
    try:
        board = json.loads((root / SCOREBOARD_MANIFEST).read_text())
        block = (root / BLOCKS["DOPE1 SCOREBOARD"]).read_text()
        if not board["receipts"]:
            out.append("README: the scoreboard render manifest lists no receipt")
        for rel, digest in board["receipts"].items():
            if (not rel.startswith(SCOREBOARD + "/") or digest not in block
                    or hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest):
                out.append(f"README: scoreboard receipt {rel} differs from the scoreboard manifest or block")
        for rel in (BLOCKS["DOPE1 SCOREBOARD"], *HEROES_V2):
            if board["files"].get(rel) != hashlib.sha256((root / rel).read_bytes()).hexdigest():
                out.append(f"README: {rel} differs from the scoreboard render manifest")
    except (OSError, KeyError, ValueError, AttributeError, TypeError):
        out.append("README: scoreboard render manifest is missing or invalid")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="splice generated blocks into README.md")
    parser.add_argument("--check", action="store_true", help="fail on drift")
    args = parser.parse_args()
    path = ROOT / "README.md"
    if args.write:
        path.write_text(splice(path.read_text()))
    found = failures(path.read_text())
    print("\n".join(found) if found else "DOPE1 README surfaces are current")
    return int(bool(found)) if args.check or args.write else 0


if __name__ == "__main__":
    sys.exit(main())
