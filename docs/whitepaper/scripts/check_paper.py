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
    ROOT / "supplement-anonymous.pdf",
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
    ROOT / "figures" / "architecture.pdf",
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
    ROOT.parents[1] / "target/paper-build/supplement-anonymous-3.log",
    ROOT.parents[1] / "target/paper-build/paper-architecture.log",
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


def strip_tex_comments(text: str) -> str:
    """Drop TeX comments while keeping an escaped percent sign."""
    lines = []
    for line in text.splitlines():
        kept = []
        index = 0
        while index < len(line):
            if line[index] == "%" and (index == 0 or line[index - 1] != "\\"):
                break
            kept.append(line[index])
            index += 1
        lines.append("".join(kept))
    return "\n".join(lines)


def anonymous_leaks(text: str) -> list[str]:
    """Scan visible text and PDF metadata, including identifying repository URLs."""
    patterns = (r"\bJepson\b", r"\bAlton\b", r"\bNEVERHUMAN\b", r"jepsontaylor",
                r"https?://[^\s<>]*(?:neverhuman|jepsontaylor)")
    return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]


def presentation_failures(layout: str) -> list[str]:
    """Frozen rubric limits on the main PDF's layout extraction."""
    failures = []
    reference_start = None
    for page_index, page in enumerate(layout.split("\f")):
        lines = page.splitlines()
        for line_index, line in enumerate(lines):
            # pdfTeX can split a small-cap heading into spaced letters.
            if re.sub(r"[^A-Za-z]", "", line).upper() == "REFERENCES":
                prefix = "\n".join(lines[:line_index]).strip()
                reference_start = page_index + bool(prefix)
                break
        if reference_start is not None:
            break
    if reference_start is None:
        failures.append("main PDF has no References heading")
    elif reference_start > 10:
        failures.append("main body before References exceeds 10 pages")
    # Match the frozen grep -o ' not ' / wc -w metric exactly.
    words = int(subprocess.check_output(["wc", "-w"], input=layout, text=True))
    if words == 0:
        failures.append("main PDF layout extraction is empty")
    elif 1000 * layout.count(" not ") > 5 * words:
        failures.append("main PDF negation density exceeds 5 per 1000 words")
    return failures


def _environments(text: str, kind: str) -> list[str]:
    """Return each table or longtable body, without crossing a nested environment."""
    opener = "\\begin{" + kind + "}"
    closer = "\\end{" + kind + "}"
    bodies = []
    start = 0
    while True:
        begin = text.find(opener, start)
        if begin < 0:
            return bodies
        end = text.find(closer, begin)
        if end < 0:
            bodies.append(text[begin:])
            return bodies
        bodies.append(text[begin:end])
        start = end + len(closer)


def _caption_bodies(environment: str) -> list[str]:
    captions = []
    token = "\\caption{"
    start = 0
    while True:
        begin = environment.find(token, start)
        if begin < 0:
            return captions
        index = begin + len(token)
        depth = 1
        while index < len(environment) and depth:
            if environment[index] == "{":
                depth += 1
            elif environment[index] == "}":
                depth -= 1
            index += 1
        captions.append(environment[begin + len(token):index - 1])
        start = index


def table_family_failures(text: str) -> list[str]:
    """Every table states a Holm family size or says that it has no Holm family."""
    failures = []
    for kind in ("table", "table*", "longtable"):
        for environment in _environments(text, kind):
            captions = _caption_bodies(environment)
            if not captions or not any("Holm" in caption for caption in captions):
                preview = " ".join(environment.split())[:80]
                failures.append("table environment lacks a Holm family declaration: " + preview)
    return failures


def main() -> int:
    failures = []
    raw_tex = TEX.read_text()
    supplement_source = (ROOT / "supplement.tex").read_text()
    if (r"\documentclass{article}" not in supplement_source
            or r"\usepackage{iclr2027_conference,times}" not in supplement_source
            or "IEEEtran" in supplement_source):
        failures.append("supplement is not in the main ICLR article style")
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
    for source in (raw_tex, supplement_source):
        failures.extend(table_family_failures(strip_tex_comments(source)))
    for path in sorted((ROOT / "generated").glob("*.tex")):
        failures.extend(table_family_failures(strip_tex_comments(path.read_text())))
    tex = strip_comments(raw_tex)
    # Drop verbatim-ish inputs of generated tables; those files are the generator.
    body = "\n".join(line for line in tex.splitlines() if "\\input{generated/" not in line)
    leaked = RAW_DECIMAL.findall(body)
    if leaked:
        failures.append(f"hand-typed decimals in dope-mfs.tex: {leaked[:12]}")
    if NUMBERS.exists():
        number_sources = NUMBERS.read_text()
        wave2_numbers = ROOT / "generated/review-wave2-numbers.tex"
        if wave2_numbers.is_file():
            number_sources += "\n" + wave2_numbers.read_text()
        defined = set(MACRO_DEF.findall(number_sources))
        # Also accept commands defined with \def\Name
        defined.update(re.findall(r"\\def\\(\w+)", number_sources))
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
    architecture_hash_path = ROOT / "generated" / "architecture-figure-hashes.json"
    if architecture_hash_path.is_file():
        architecture_hashes = json.loads(architecture_hash_path.read_text())
        recorded_hashes.update(architecture_hashes.get("pdf_sha256") or {})
        for name in ("figures/architecture.tex", "generated/numbers.tex", "../../ops/ci/paper-architecture.tex"):
            source = ROOT / name
            if (not source.is_file() or architecture_hashes.get("source_sha256", {}).get(name)
                    != hashlib.sha256(source.read_bytes()).hexdigest()):
                failures.append("architecture input hash mismatch: " + name)
    else:
        failures.append("generated/architecture-figure-hashes.json is missing")
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
            expected = 5.5 if path.name in ("teaser.pdf", "architecture.pdf") else 7.16
            if width is None or abs(width - expected) > 0.02:
                failures.append(f"{path.name} width is not {expected:.2f}in")
    manuscript_texts = [TEX, ROOT / "supplement.tex"]
    manuscript_texts.append(ROOT / "figures" / "architecture.tex")
    manuscript_texts.extend(sorted((ROOT / "generated").glob("*.tex")))
    for path in manuscript_texts:
        if not path.is_file():
            failures.append(f"missing manuscript text {path.name}")
            continue
        hits = banned_hits(path.read_text(errors="replace"))
        if hits:
            failures.append(f"{path.name} contains a private token: {hits[0]}")
    for path in FIGURES[:4]:
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
        info = subprocess.check_output(["pdfinfo", str(path)], text=True, errors="replace")
        if "612 x 792" not in info:
            failures.append(f"{path.name} is not US Letter")
        if "conference paper at ICLR" in extracted or "Under review as a conference paper" in extracted:
            failures.append(f"{path.name} claims an ICLR venue")
        if path.name.endswith("-anonymous.pdf"):
            info = subprocess.check_output(["pdfinfo", str(path)], text=True, errors="replace")
            links = subprocess.check_output(
                ["pdftohtml", "-i", "-xml", "-stdout", str(path)], text=True, errors="replace"
            )
            if anonymous_leaks(extracted + "\n" + info + "\n" + links):
                failures.append(path.name + " contains an identifying name or URL")
        if path.name in ("dope-mfs.pdf", "dope-mfs-anonymous.pdf"):
            layout = subprocess.check_output(
                ["pdftotext", "-layout", str(path), "-"], text=True, errors="replace"
            )
            failures.extend(path.name + ": " + item for item in presentation_failures(layout))
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
    "Sigma",
    "Subsection",
    "Table",
    "Toprule",
    "Url",
    "UrlBreaks",
}


if __name__ == "__main__":
    sys.exit(main())
