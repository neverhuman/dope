#!/usr/bin/env python3
"""Refuse a manuscript whose Figure 1 is not the generated hero figure on page 1.

\\GeneratedFigure typesets a visible pending box instead of failing when
paper/generated/hero-frontier.pdf is absent, and LaTeX may float the figure to a later
page. After latexmk, paper/build.sh runs this check on the auxiliary file (the label's
figure number and page) and on the log (the graphic pdfTeX actually included).

Usage: figure_one.py [BUILD_DIR]   (default: target/paper)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LABEL = "fig:frontier"
GRAPHIC = "generated/hero-frontier.pdf"
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
    if f"<use {GRAPHIC}>" not in unwrapped:
        found.append(f"{GRAPHIC} was not included; Figure 1 would be a pending placeholder")
    return found


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "target" / "paper"
    try:
        aux = (out / "main.aux").read_text(errors="replace")
        log = (out / "main.log").read_text(errors="replace")
    except OSError:
        print("figure one: main.aux or main.log is missing; build main.tex first", file=sys.stderr)
        return 1
    found = failures(aux, log)
    for failure in found:
        print(f"figure one: {failure}", file=sys.stderr)
    if not found:
        print("Figure 1 is the generated hero figure on page 1")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
