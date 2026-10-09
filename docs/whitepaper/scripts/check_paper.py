#!/usr/bin/env python3
"""Fail if the paper types a measured decimal by hand, a built PDF embeds a Type 3 font, or the final build log contains a warning, an undefined reference, or an overfull or underfull box."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

# Loaded by file path from the benchmark suite, which does not put this
# directory on sys.path.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from public_hardware import banned_hits

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "dope-mfs.tex"
NUMBERS = ROOT / "generated" / "numbers.tex"
FIGURES = (
    ROOT / "dope-mfs.pdf",
    ROOT / "dope-mfs-anonymous.pdf",
    ROOT / "supplement.pdf",
    ROOT / "figures" / "retention-bytes.pdf",
    ROOT / "figures" / "retention-bars.pdf",
    ROOT / "figures" / "paired-cdf.pdf",
    ROOT / "figures" / "loss-curves.pdf",
    ROOT / "figures" / "mfs-v3-retention-bars.pdf",
    ROOT / "figures" / "mfs-v3-paired.pdf",
    ROOT / "figures" / "retained-matched-eight.pdf",
    ROOT / "figures" / "published-baseline-cohorts.pdf",
    # Drawn at the 5.5in text width. The other plots stay 7.16in.
    ROOT / "figures" / "teaser.pdf",
)
# A measurement typed into the prose. Integers and one-decimal illustration
# values such as the MFS arithmetic example stay in the generator instead.
RAW_DECIMAL = re.compile(r"(?<![\w.\\])\d+\.\d{4,}(?![\w])")
MACRO_DEF = re.compile(r"\\(?:newcommand|renewcommand)\*?\{\\(\w+)\}")
MACRO_USE = re.compile(r"\\([A-Z][A-Za-z]+)")
# Final logs from the latexmk run. A first pass reports undefined references on purpose.
LOGS = (
    ROOT.parents[1] / "target/paper-build/dope-mfs-3.log",
    ROOT.parents[1] / "target/paper-build/dope-mfs-anonymous-3.log",
    ROOT.parents[1] / "target/paper-build/supplement-3.log",
)
LOG_HITS = ("Warning", "undefined", "Undefined", "Overfull", "Underfull")


def latex_warnings(text: str) -> list[str]:
    """Lines a submission log must not contain."""
    hits = []
    for line in text.splitlines():
        if any(token in line for token in LOG_HITS):
            hits.append(line.strip())
    return hits


def strip_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def main() -> int:
    failures = []
    raw_tex = TEX.read_text()
    if "D.O.P.E." in raw_tex:
        failures.append("D.O.P.E. remains in dope-mfs.tex")
    if r"\documentclass{article}" not in raw_tex.split(r"\begin{document}", 1)[0]:
        failures.append("dope-mfs.tex is not an article-class manuscript")
    if "IEEEtran" in raw_tex.split(r"\begin{document}", 1)[0]:
        failures.append("dope-mfs.tex preamble still uses IEEEtran")
    if r"\fancyhead{}" not in raw_tex or r"\headrulewidth" not in raw_tex:
        failures.append("the class running header is not cleared")
    if "conference paper at ICLR" in raw_tex:
        failures.append("dope-mfs.tex claims an ICLR venue")
    if "figures/teaser.pdf" not in raw_tex:
        failures.append("dope-mfs.tex does not include figures/teaser.pdf")
    if r"\IfFileExists" in raw_tex or "The page-1 value figure is" in raw_tex:
        failures.append("the teaser include has a missing-file fallback")
    tex = strip_comments(raw_tex)
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
    if NUMBERS.exists() and "\\newcommand{\\FigWidth}{7.16in}" not in NUMBERS.read_text():
        failures.append("FigWidth is not 7.16in")
    hash_path = ROOT / "generated" / "figure-hashes.json"
    recorded_hashes = {}
    if hash_path.is_file():
        recorded_hashes = json.loads(hash_path.read_text()).get("pdf_sha256") or {}
    else:
        failures.append("generated/figure-hashes.json is missing")
    retained_hash_path = ROOT / "generated/retained-figure-hashes.json"
    if retained_hash_path.is_file():
        recorded_hashes.update(json.loads(retained_hash_path.read_text()).get("pdf_sha256") or {})
    else:
        failures.append("generated/retained-figure-hashes.json is missing")
    published_hash_path = ROOT / "generated" / "published-baseline-figure-hashes.json"
    if published_hash_path.is_file():
        recorded_hashes.update(json.loads(published_hash_path.read_text()).get("pdf_sha256") or {})
    else:
        failures.append("generated/published-baseline-figure-hashes.json is missing")
    mfs_hash_path = ROOT / "generated" / "mfs-v3-figure-hashes.json"
    if mfs_hash_path.is_file():
        recorded_hashes.update(json.loads(mfs_hash_path.read_text()).get("pdf_sha256") or {})
    else:
        failures.append("generated/mfs-v3-figure-hashes.json is missing")
    teaser_hash_path = ROOT / "generated" / "teaser-figure-hashes.json"
    if teaser_hash_path.is_file():
        recorded_hashes.update(json.loads(teaser_hash_path.read_text()).get("pdf_sha256") or {})
    else:
        failures.append("generated/teaser-figure-hashes.json is missing")
    for path in FIGURES:
        if not path.exists():
            failures.append(f"missing figure {path.name}")
            continue
        listed = subprocess.check_output(["pdffonts", str(path)], text=True, errors="replace")
        # pdffonts writes a font-type mismatch on stderr. Type 3 is a stdout row.
        if "Type 3" in listed:
            failures.append(f"Type 3 font in {path.name}")
        if path.parent.name == "figures":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if recorded_hashes.get(path.name) != digest:
                failures.append(f"{path.name} does not match figure-hashes.json")
            info = subprocess.check_output(["pdfinfo", str(path)], text=True, errors="replace")
            width = None
            for line in info.splitlines():
                if line.startswith("Page size:"):
                    width = float(line.split()[2]) / 72.0
            expected = 5.5 if path.name == "teaser.pdf" else 7.16
            if width is None or abs(width - expected) > 0.02:
                failures.append(f"{path.name} width is not {expected:.2f}in")
    manuscript_texts = [TEX, ROOT / "supplement.tex"]
    manuscript_texts.extend(sorted((ROOT / "generated").glob("*.tex")))
    for path in manuscript_texts:
        if not path.is_file():
            failures.append(f"missing manuscript text {path.name}")
            continue
        hits = banned_hits(path.read_text(errors="replace"))
        if hits:
            failures.append(f"{path.name} contains a private token: {hits[0]}")
    for path in FIGURES[:3]:
        if not path.exists():
            continue
        try:
            extracted = subprocess.check_output(
                ["pdftotext", str(path), "-"], text=True, errors="replace"
            )
        except FileNotFoundError:
            failures.append("pdftotext is not installed")
            break
        except subprocess.CalledProcessError as exc:
            failures.append(f"pdftotext failed for {path.name}: {exc.returncode}")
            continue
        hits = banned_hits(extracted)
        if hits:
            failures.append(f"{path.name} text contains a private token: {hits[0]}")
        if path.name != "supplement.pdf":
            info = subprocess.check_output(["pdfinfo", str(path)], text=True, errors="replace")
            if "612 x 792" not in info:
                failures.append(f"{path.name} is not US Letter")
            if "conference paper at ICLR" in extracted or "Under review as a conference paper" in extracted:
                failures.append(f"{path.name} claims an ICLR venue")
        if path.name == "dope-mfs-anonymous.pdf":
            if "Jepson" in extracted or "NEVERHUMAN" in extracted or "Alton" in extracted:
                failures.append("anonymous PDF still names an author")
        if path.name == "dope-mfs.pdf" and "DOPE" not in extracted:
            failures.append("journal PDF does not show the DOPE title")
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
