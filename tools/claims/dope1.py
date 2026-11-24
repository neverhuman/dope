#!/usr/bin/env python3
"""Generated landing-page and historical engine surfaces of the DOPE benchmark.

``README.md`` carries the current v3 scoreboard block and hero figures:

* ``DOPE1 SCOREBOARD``: ``bench/real-world/results/scoreboard/v3.0.0/readme-block.md``
  (``dope_kpi render-readme`` from the v3.0.0 scoreboard receipts beside it);
* heroes: ``bench/real-world/figures/hero-v3-methods-{light,dark}.svg``
  (``dope_kpi render-hero``), hash-listed in the scoreboard manifest next to the
  receipt digests.

``HISTORY.md`` carries the archived v1/MLC generated engine blocks:

* ``DOPE1 RESULTS``: ``bench/real-world/results/readme-block.md``
  (``bench/real-world/render.py`` from the benchmark summary);
* ``DOPE1 WALKTHROUGH``: ``examples/adult/TRANSCRIPT.md``
  (``examples/adult/walkthrough.py`` from a recorded CLI run);
* historical hero: ``bench/real-world/figures/hero.png``, hash-listed in the
  render manifest next to the summary digest.

``--write`` splices each block into its target document; ``--check`` (and the
public claims gate) refuses drift and forbids historical blocks in ``README.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOREBOARD = "bench/real-world/results/scoreboard/v3.0.0"
README_BLOCKS = {
    "DOPE1 SCOREBOARD": f"{SCOREBOARD}/readme-block.md",
}
HISTORY_BLOCKS = {
    "DOPE1 RESULTS": "bench/real-world/results/readme-block.md",
    "DOPE1 WALKTHROUGH": "examples/adult/TRANSCRIPT.md",
}
BLOCKS = {**README_BLOCKS, **HISTORY_BLOCKS}
HERO = "bench/real-world/figures/hero.png"
MANIFEST = "bench/real-world/results/render-manifest.json"
SUMMARY = "bench/real-world/results/confirm/summary.json"
HEROES = ("bench/real-world/figures/hero-v3-methods-light.svg", "bench/real-world/figures/hero-v3-methods-dark.svg")
SCOREBOARD_MANIFEST = f"{SCOREBOARD}/render-manifest.json"


def markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"


def _span(text: str, name: str) -> tuple[int, int] | None:
    begin, end = markers(name)
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        return None
    return text.index(begin), text.index(end) + len(end)


def expected(root: Path, name: str) -> str | None:
    path = root / BLOCKS[name]
    if not path.is_file():
        return None
    text = path.read_text().strip()
    begin, end = markers(name)
    return text if text.startswith(begin) and text.endswith(end) else None


def splice(text: str, root: Path = ROOT, names: tuple[str, ...] | None = None) -> str:
    for name in (names if names is not None else tuple(BLOCKS)):
        span, block = _span(text, name), expected(root, name)
        if span and block:
            text = text[:span[0]] + block + text[span[1]:]
    return text


def strip(text: str, names: tuple[str, ...] | None = None) -> str:
    """Document prose with generated DOPE1 blocks removed (for language lint)."""
    for name in (names if names is not None else tuple(BLOCKS)):
        begin, end = markers(name)
        text = re.sub(re.escape(begin) + ".*?" + re.escape(end), "", text, flags=re.S)
    return text


def _check_blocks(text: str, label: str, names: dict[str, str], root: Path) -> list[str]:
    out = []
    for name in names:
        span = _span(text, name)
        if span is None:
            out.append(f"{label}: {name} markers must be present, unique and ordered")
            continue
        block = expected(root, name)
        if block is None:
            out.append(f"{label}: {name} source {BLOCKS[name]} is missing or malformed")
        elif text[span[0]:span[1]] != block:
            out.append(f"{label}: {name} block drifted from {BLOCKS[name]}")
    return out


def failures(readme: str, root: Path = ROOT, history: str | None = None) -> list[str]:
    out = _check_blocks(readme, "README", README_BLOCKS, root)
    for name in HISTORY_BLOCKS:
        begin, end = markers(name)
        if begin in readme or end in readme:
            out.append(f"README: historical {name} block belongs in HISTORY.md")
    if history is None:
        history_path = root / "HISTORY.md"
        history = history_path.read_text() if history_path.is_file() else ""
    out.extend(_check_blocks(history, "HISTORY.md", HISTORY_BLOCKS, root))
    for name in README_BLOCKS:
        begin, end = markers(name)
        if begin in history or end in history:
            out.append(f"HISTORY.md: {name} block belongs in README.md")
    try:
        manifest = json.loads((root / MANIFEST).read_text())
        if manifest["summary_sha256"] != hashlib.sha256((root / SUMMARY).read_bytes()).hexdigest():
            out.append("HISTORY.md: benchmark summary changed after rendering")
        for rel in (HERO, BLOCKS["DOPE1 RESULTS"]):
            if manifest["files"].get(rel) != hashlib.sha256((root / rel).read_bytes()).hexdigest():
                out.append(f"HISTORY.md: {rel} differs from the render manifest")
    except (OSError, KeyError, ValueError):
        out.append("HISTORY.md: benchmark render manifest is missing or invalid")
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
        for rel in (BLOCKS["DOPE1 SCOREBOARD"], *HEROES):
            if board["files"].get(rel) != hashlib.sha256((root / rel).read_bytes()).hexdigest():
                out.append(f"README: {rel} differs from the scoreboard render manifest")
    except (OSError, KeyError, ValueError, AttributeError, TypeError):
        out.append("README: scoreboard render manifest is missing or invalid")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="splice generated blocks into README.md and HISTORY.md")
    parser.add_argument("--check", action="store_true", help="fail on drift")
    args = parser.parse_args()
    readme_path = ROOT / "README.md"
    history_path = ROOT / "HISTORY.md"
    if args.write:
        readme_path.write_text(splice(readme_path.read_text(), ROOT, tuple(README_BLOCKS)))
        if history_path.is_file():
            history_path.write_text(splice(history_path.read_text(), ROOT, tuple(HISTORY_BLOCKS)))
    found = failures(readme_path.read_text(), ROOT, history_path.read_text() if history_path.is_file() else "")
    print("\n".join(found) if found else "DOPE1 README and HISTORY surfaces are current")
    return int(bool(found)) if args.check or args.write else 0


if __name__ == "__main__":
    sys.exit(main())
