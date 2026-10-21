"""Authenticate the README blocks and paper TeX added with the compression frontier.

claim_registry.public_failures calls these helpers. Each generated surface must equal
its renderer's output from a committed input, so a number can only reach the README or
the paper through a generator:

* the COMPRESSION block and paper/generated/compression-{macros,table}.tex render
  from paper/generated/compression-frontier.json (itself recomputed and compared by
  `compression_frontier.py --check` in ops/ci/cpu.sh);
* the QUICKSTART block and paper/generated/quickstart-macros.tex render from
  examples/quickstart/results/{adult-reference,ablation}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import compression_frontier_render as frontier
import quickstart_block as quickstart

GENERATED_TEX = {
    "paper/generated/compression-macros.tex": frontier.tex_macros,
    "paper/generated/compression-table.tex": frontier.tex_table,
}


def _frontier_data(root: Path) -> dict | None:
    path = root / frontier.DATA_JSON
    return json.loads(path.read_text()) if path.is_file() else None


def block_failures(readme: str, root: Path) -> list[str]:
    """A generated block is optional, but when present it must appear once and match its renderer."""
    failures = []
    if frontier.README_BEGIN in readme or frontier.README_END in readme:
        data = _frontier_data(root)
        if data is None:
            failures.append(f"{frontier.DATA_JSON}: missing")
        else:
            failures.extend(frontier.readme_failures(readme, data))
    if quickstart.BEGIN in readme or quickstart.END in readme:
        if not (root / quickstart.REPORT).is_file():
            failures.append(f"{quickstart.REPORT}: missing")
        else:
            failures.extend(quickstart.failures(readme, root))
    return failures


def strip_blocks(readme: str) -> str:
    """README prose without the generated blocks (they are authenticated, not linted)."""
    return quickstart.strip(frontier.strip_readme_block(readme))


def verified_tex(root: Path) -> tuple[set[str], list[str]]:
    """Generated TeX files that equal their rendering, plus failures for any drift."""
    verified, failures = set(), []
    data = _frontier_data(root)
    rendered = {}
    if data is not None:
        rendered.update({name: renderer(data) for name, renderer in GENERATED_TEX.items()})
    if (root / quickstart.REPORT).is_file() and (root / quickstart.ABLATION).is_file():
        rendered[quickstart.MACROS] = quickstart.tex_macros(quickstart.load_report(root),
                                                             quickstart.load_ablation(root))
        rendered[quickstart.ABLATION_TEX] = quickstart.ablation_tex(quickstart.load_ablation(root))
    for name, text in rendered.items():
        path = root / name
        if not path.is_file():
            continue
        if path.read_text() == text:
            verified.add(name)
        else:
            failures.append(f"{name}: generated TeX drift")
    return verified, failures
