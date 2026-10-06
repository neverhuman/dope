#!/usr/bin/env python3
"""Fail if the paper types a measured decimal by hand, a figure embeds a Type 3 font, or the third-pass build log has a box, float, or undefined-reference warning."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "dope-mfs.tex"
NUMBERS = ROOT / "generated" / "numbers.tex"
FIGURES = (
    ROOT / "figures" / "retention-bytes.pdf",
    ROOT / "figures" / "paired-cdf.pdf",
    ROOT / "figures" / "loss-curves.pdf",
)
# A measurement typed into the prose. Integers and one-decimal illustration
# values such as the MFS arithmetic example stay in the generator instead.
RAW_DECIMAL = re.compile(r"(?<![\w.\\])\d+\.\d{4,}(?![\w])")
MACRO_DEF = re.compile(r"\\(?:newcommand|renewcommand)\*?\{\\(\w+)\}")
MACRO_USE = re.compile(r"\\([A-Z][A-Za-z]+)")
# Third-pass logs only. A first pass reports undefined references on purpose.
LOGS = (Path("/tmp/dope-mfs-3.log"), Path("/tmp/supplement-3.log"))
LATEX_WARNING = re.compile(
    r"^(?:Overfull \\[hv]box|Underfull \\[hv]box"
    r"|LaTeX Warning: Float too large"
    r"|LaTeX Warning: Reference .+ undefined"
    r"|LaTeX Warning: Citation .+ undefined"
    r"|LaTeX Warning: There were undefined references"
    r"|! Undefined control sequence)"
)


def latex_warnings(text: str) -> list[str]:
    """Lines a submission log must not contain."""
    return [line.strip() for line in text.splitlines() if LATEX_WARNING.match(line)]


def strip_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def main() -> int:
    failures = []
    tex = strip_comments(TEX.read_text())
    # Drop verbatim-ish inputs of generated tables; those files are the generator.
    body = "\n".join(line for line in tex.splitlines() if "\\input{generated/" not in line)
    leaked = RAW_DECIMAL.findall(body)
    if leaked:
        failures.append(f"hand-typed decimals in dope-mfs.tex: {leaked[:12]}")
    if NUMBERS.exists():
        defined = set(MACRO_DEF.findall(NUMBERS.read_text()))
        # Also accept commands defined with \def\Name
        defined.update(re.findall(r"\\def\\(\w+)", NUMBERS.read_text()))
        used = set(MACRO_USE.findall(body))
        missing = sorted(
            name for name in used
            if name not in defined and name not in _LATEX and not name.startswith("IEEE")
        )
        if missing:
            failures.append("macros used but not defined in numbers.tex: " + ", ".join(missing))
    else:
        failures.append("generated/numbers.tex is missing")
    for path in FIGURES:
        if not path.exists():
            failures.append(f"missing figure {path.name}")
            continue
        listed = subprocess.check_output(["pdffonts", str(path)], text=True, errors="replace")
        if "Type 3" in listed:
            failures.append(f"Type 3 font in {path.name}")
    for log in LOGS:
        if not log.is_file():
            failures.append(f"missing build log {log}")
            continue
        hits = latex_warnings(log.read_text(errors="replace"))
        if hits:
            failures.append(log.name + ": " + " | ".join(hits[:6]))
    if failures:
        print("\n".join(failures))
        return 1
    print("paper check passed")
    return 0


_LATEX = {
    "Abstract",
    "Appendices",
    "Begin",
    "Bottomrule",
    "Caption",
    "Centering",
    "Cite",
    "Document",
    "End",
    "Eqref",
    "Figure",
    "Hline",
    "Input",
    "Label",
    "Maketitle",
    "Midrule",
    "Paragraph",
    "Ref",
    "Section",
    "Subsection",
    "Table",
    "Toprule",
    "Url",
    "UrlBreaks",
}


if __name__ == "__main__":
    sys.exit(main())
