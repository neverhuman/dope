#!/usr/bin/env python3
"""Refuse a manuscript whose Figure 1 is not a generated hero figure on page 1.

\\GeneratedFigure typesets a visible pending box instead of failing when a generated
hero figure (paper/generated/compression-frontier.pdf, hero-frontier.pdf, or dope1-hero.pdf
for the DOPE 1.0 manuscript) is absent,
and LaTeX may float the figure to a later page. After latexmk, paper/build.sh runs this check on the auxiliary file (the label's
figure number and page) and on the log (the graphic pdfTeX actually included).

Usage: figure_one.py [BUILD_DIR] [DOC ...]   (defaults: target/paper, main)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LABEL = "fig:frontier"
# The transfer-per-byte frontier is the hero; the foundation-model bars remain accepted.
GRAPHICS = ("generated/transfer-per-byte.pdf", "generated/hero-frontier.pdf", "generated/compression-frontier.pdf",
            "generated/dope1-hero.pdf")
GRAPHIC = GRAPHICS[1]
# TeX hard-wraps log lines at max_print_line (79 characters by default).
LOG_WRAP = 79


def failures(aux: str, log: str) -> list[str]:
    found = []
    match = re.search(r"\\newlabel\{" + re.escape(LABEL) + r"\}\{\{([^{}]*)\}\{([^{}]*)\}", aux)
    if match is None:
        found.append(f"{LABEL} has no label in main.aux")
    else:
        number, page = match.groups()
        if number != "1":
            found.append(f"{LABEL} is not Figure 1")
        if page != "1":
            found.append(f"{LABEL} is not on page 1")
    unwrapped = re.sub(r"(?m)^(.{%d})\n" % LOG_WRAP, r"\1", log)
    if not any(f"<use {graphic}>" in unwrapped for graphic in GRAPHICS):
        found.append("no generated hero figure was included; Figure 1 would be a pending placeholder")
    return found


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "target" / "paper"
    status = 0
    for doc in args[1:] or ["main"]:
        try:
            aux = (out / f"{doc}.aux").read_text(errors="replace")
            log = (out / f"{doc}.log").read_text(errors="replace")
        except OSError:
            print(f"figure one: {doc}.aux or {doc}.log is missing; build {doc}.tex first", file=sys.stderr)
            return 1
        found = failures(aux, log)
        for failure in found:
            print(f"figure one: {doc}: {failure}", file=sys.stderr)
        if found:
            status = 1
        else:
            print(f"Figure 1 of {doc} is a generated hero figure on page 1")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
