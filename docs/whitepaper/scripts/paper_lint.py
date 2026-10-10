#!/usr/bin/env python3
"""Prose and layout lint for the manuscript and the supplement.

Pure functions over LaTeX source, with ``\\input`` and ``\\fittowidthfile``
expanded relative to docs/whitepaper, and over ``pdftotext -layout`` text.
Every rule returns human-readable violations. Counting rules read the PDF
body before References when a PDF text is given, and the expanded source
otherwise. Not yet called by check_paper.py; paper_v2_guards.run_v2_lint is
the hook.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
STAY = re.compile(r"\bstay(?:s|ed|ing)?\b", re.IGNORECASE)
BANNED_PHRASES = (
    "stays unclaimed", "stays null", "stays sealed", "stays outside", "stays distinct",
    "stays unpooled", "wave 2", "in progress", "prints no number", "a retention leaves",
)
UN_HEDGES = (
    "unclaimed", "unpooled", "unmade", "unmeasured", "unscored", "unsealed", "unscanned",
    "unemitted", "unshown", "unrecorded", "unstored", "unreported", "unauthorized",
)
SUPPLEMENT_COMMANDS = (
    "python3 -m research.benchmark.fetch_pmlb --list",
    "python3 -m research.benchmark.fetch_pmlb DATASET_ID OUTPUT_DIRECTORY",
)
TABULARS = ("tabular", "tabular*", "tabularx", "longtable")
FLOATS = ("table", "table*")
DROPPED_ENVIRONMENTS = (
    "figure", "figure*", "table", "table*", "equation", "equation*", "align", "align*",
    "gather", "gather*", "multline", "multline*", "tikzpicture",
)
_INPUT = re.compile(r"\\(?:input|include|fittowidthfile)\s*\{([^{}#]+)\}")
_FILE_MARK = re.compile(r"\\lint(begin|end)file\{([^{}]*)\}")
_ENV = re.compile(r"\\(begin|end)\{([^{}]+)\}|\\(begingroup|endgroup)\b|\\let\\tabular\\longtable\b")
_HEADING = re.compile(r"\\(section|subsection|subsubsection)\*?\s*(?:\[[^\]]*\])?\s*\{")
_LEVEL = {"section": 1, "subsection": 2, "subsubsection": 3}
_DROP_ARG = re.compile(
    r"\\(?:label|ref|cref|Cref|eqref|pageref|includegraphics|bibliography|bibliographystyle|"
    r"vspace|hspace|setlength|input|begin|end|newcommand|renewcommand)\*?(?:\[[^\]]*\])?\{[^{}]*\}"
)
_CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{[^{}]*\}")
_WORD = re.compile(r"[a-z0-9]+")
_DEFINE = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\\[A-Za-z@]+\}?\s*(?:\[[^\]]*\])*\s*\{")


def strip_comments(text: str) -> str:
    """Drop TeX comments. A percent sign after an odd run of backslashes is escaped."""
    kept = []
    for line in text.splitlines():
        index = 0
        while True:
            index = line.find("%", index)
            if index < 0:
                kept.append(line)
                break
            slashes = len(line[:index]) - len(line[:index].rstrip("\\"))
            if slashes % 2 == 0:
                kept.append(line[:index])
                break
            index += 1
    return "\n".join(kept)


def _read_file(path: Path) -> str | None:
    return path.read_text(errors="replace") if path.is_file() else None


def expand_inputs(text: str, root: Path = PAPER, read=None, missing=None, depth: int = 0) -> str:
    """Inline \\input, \\include, and \\fittowidthfile arguments, comments removed.

    Each inlined file sits between \\lintbeginfile and \\lintendfile marks so a
    violation can name its source. A missing file keeps its command and is
    appended to ``missing``.
    """
    read = read or _read_file
    text = strip_comments(text)
    if depth > 8:
        raise ValueError("input nesting exceeds eight levels")

    def inline(match):
        name = match.group(1).strip()
        path = Path(root) / name
        if not path.suffix:
            path = path.with_suffix(".tex")
        body = read(path)
        if body is None:
            if missing is not None:
                missing.append(name)
            return match.group(0)
        inner = expand_inputs(body, root, read, missing, depth + 1)
        return f"\\lintbeginfile{{{name}}}{inner}\\lintendfile{{{name}}}"

    return _INPUT.sub(inline, text)


def _balanced(text: str, start: int) -> int:
    """Index just past the brace group that opens at ``start``."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and text[index - 1] != "\\":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def document_body(source: str) -> str:
    begin = source.find("\\begin{document}")
    body = source[begin + len("\\begin{document}"):] if begin >= 0 else source
    cut = len(body)
    for token in ("\\bibliographystyle", "\\bibliography{", "\\begin{thebibliography}", "\\end{document}"):
        found = body.find(token)
        if 0 <= found < cut:
            cut = found
    return body[:cut]


def environment_text(source: str, name: str) -> list[str]:
    opener, closer = "\\begin{" + name + "}", "\\end{" + name + "}"
    found, start = [], 0
    while (begin := source.find(opener, start)) >= 0:
        end = source.find(closer, begin)
        end = len(source) if end < 0 else end
        found.append(source[begin + len(opener):end])
        start = end + len(closer)
    return found


def drop_environments(source: str, names=DROPPED_ENVIRONMENTS) -> str:
    for name in names:
        pattern = re.compile(r"\\begin\{" + re.escape(name) + r"\}.*?\\end\{" + re.escape(name) + r"\}", re.S)
        source = pattern.sub(" ", source)
    return re.sub(r"\\\[.*?\\\]", " ", source, flags=re.S)


def drop_definitions(text: str) -> str:
    """Remove \\newcommand bodies, such as the generated number macros."""
    kept, position = [], 0
    while (match := _DEFINE.search(text, position)) is not None:
        kept.append(text[position:match.start()])
        position = _balanced(text, match.end() - 1)
    kept.append(text[position:])
    return "".join(kept)


def detex(source: str) -> str:
    """Approximate reading text. Tables and display math go; captions stay."""
    text = drop_definitions(_FILE_MARK.sub(" ", source))
    for name in TABULARS:
        text = re.sub(r"\\begin\{" + re.escape(name) + r"\}.*?\\end\{" + re.escape(name) + r"\}", " ", text, flags=re.S)
    text = drop_environments(text, tuple(name for name in DROPPED_ENVIRONMENTS if not name.startswith(("figure", "table"))))
    text = re.sub(r"\\(?:sub)*section\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}", r" \1. ", text)
    text = _CITE.sub(" CITE ", text)
    text = re.sub(r"\\[cC]?ref\{[^{}]*\}|\\eqref\{[^{}]*\}", " REF ", text)
    text = re.sub(r"\\(?:path|url)\{([^{}]*)\}", lambda match: match.group(1).replace(" ", ""), text)
    text = _DROP_ARG.sub(" ", text)
    text = re.sub(r"\$([^$]*)\$", lambda match: re.sub(r"[\\{}]", "", match.group(1)), text)
    text = text.replace("~", " ").replace("\\\\", " ").replace("{,}", ",")
    text = re.sub(r"\\([%&_#$])", r"\1", text)
    text = re.sub(r"\\[,;:! ]", " ", text)
    text = re.sub(r"\\(?:item|centering|noindent|newpage|par|footnotesize|scriptsize|small|tiny)\b", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?", "", text)
    text = re.sub(r"\[[a-z]+=[^\]]*\]|\[[htbpH!]{1,4}\]", " ", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def pdf_main_body(pdf_text: str) -> str:
    """PDF text before the References heading, hyphenated line ends joined, spaces collapsed."""
    lines = pdf_text.replace("\f", "\n").splitlines()
    for index, line in enumerate(lines):
        if re.sub(r"[^A-Za-z]", "", line).upper() == "REFERENCES":
            lines = lines[:index]
            break
    text = "\n".join(lines)
    text = re.sub(r"(\w)-[ \t]*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def _phrase(phrase: str) -> re.Pattern:
    return re.compile(r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b", re.IGNORECASE)


def stay_violations(text: str, threshold: int = 5) -> list[str]:
    count = len(STAY.findall(text))
    return [f"'stay' forms appear {count} times (limit {threshold})"] if count > threshold else []


def banned_phrase_hits(text: str, phrases=BANNED_PHRASES) -> dict[str, int]:
    found = {phrase: len(_phrase(phrase).findall(text)) for phrase in phrases}
    return {phrase: count for phrase, count in found.items() if count}


def un_hedge_violations(text: str, limit: int = 6, hedges=UN_HEDGES) -> list[str]:
    counts = Counter(match.lower() for match in re.findall(r"\b(" + "|".join(hedges) + r")\b", text, re.IGNORECASE))
    total = sum(counts.values())
    if total <= limit:
        return []
    listed = ", ".join(f"{word} {count}" for word, count in sorted(counts.items()))
    return [f"un- hedges appear {total} times (limit {limit}): {listed}"]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text) if part.strip()]


def normalize_sentence(sentence: str) -> str:
    """Lower case, words only. Hyphens, dashes, and apostrophes join their neighbours."""
    return " ".join(_WORD.findall(re.sub(r"[-\u2010-\u2014'\u2019]", "", sentence.lower())))


def repeated_sentences(text: str, min_words: int = 5, limit: int = 2) -> list[str]:
    counts = Counter()
    for sentence in sentences(text):
        normal = normalize_sentence(sentence)
        if len(normal.split()) >= min_words:
            counts[normal] += 1
    return [f"sentence repeated {count} times (limit {limit}): '{normal[:80]}'"
            for normal, count in sorted(counts.items()) if count > limit]


def section_bodies(source: str) -> list[tuple[str, str, int]]:
    """(kind, title, words) for each \\section and \\subsection.

    A section body runs to the next heading of the same or a higher level, so
    it includes its subsections. Floats, display math, tables, and nested
    heading titles are left out of the count.
    """
    body = document_body(source)
    heads = []
    for match in _HEADING.finditer(body):
        close = _balanced(body, match.end() - 1)
        heads.append((match.group(1), body[match.end():close - 1], match.start(), close))
    found = []
    for index, (kind, title, _start, close) in enumerate(heads):
        if kind == "subsubsection":
            continue
        end = len(body)
        for later_kind, _title, later_start, _close in heads[index + 1:]:
            if _LEVEL[later_kind] <= _LEVEL[kind]:
                end = later_start
                break
        text = drop_environments(body[close:end])
        text = re.sub(r"\\(?:sub)*section\*?\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}", " ", text)
        found.append((kind, detex(title), words(detex(text))))
    return found


def short_sections(source: str, minimum: int = 60) -> list[str]:
    return [f"{kind} '{title}' body has {count} words (minimum {minimum})"
            for kind, title, count in section_bodies(source) if count < minimum]


def abstract_violations(source: str, maximum: int = 200) -> list[str]:
    bodies = environment_text(source, "abstract")
    if not bodies:
        return ["no abstract environment"]
    count = words(detex(bodies[0]))
    return [f"abstract has {count} words (maximum {maximum})"] if count > maximum else []


def orphan_tabulars(source: str) -> list[str]:
    """Every tabular sits in a captioned table float, or is a captioned longtable.

    Run on expanded source. A tabular inside a \\begingroup that lets
    \\tabular be \\longtable counts as a longtable. A tabular nested in a table
    cell belongs to its parent, which carries the caption check.
    """
    text = strip_comments(source)
    files, groups, envs, records, closed = [], [], [], [], {}
    position, serial, global_longtable = 0, 0, False
    while True:
        mark = _FILE_MARK.search(text, position)
        env = _ENV.search(text, position)
        if mark is None and env is None:
            break
        if mark is not None and (env is None or mark.start() < env.start()):
            if mark.group(1) == "begin":
                files.append(mark.group(2))
            elif files:
                files.pop()
            position = mark.end()
            continue
        position = env.end()
        if env.group(3) == "begingroup":
            groups.append(False)
        elif env.group(3) == "endgroup":
            if groups:
                groups.pop()
        elif env.group(1) is None:
            if groups:
                groups[-1] = True
            else:
                global_longtable = True
        elif env.group(1) == "begin":
            name, index = env.group(2), serial
            serial += 1
            envs.append((name, env.start(), index))
            nested = any(item[0] in TABULARS for item in envs[:-1])
            if name in TABULARS and not nested:
                longtable = name == "longtable" or (name == "tabular" and (any(groups) or global_longtable))
                outer = next((item[2] for item in reversed(envs[:-1]) if item[0] in FLOATS), None)
                records.append((index, longtable, outer, files[-1] if files else "main source", env.end()))
        else:
            for depth in range(len(envs) - 1, -1, -1):
                if envs[depth][0] == env.group(2):
                    _name, start, index = envs[depth]
                    closed[index] = text[start:env.start()]
                    del envs[depth:]
                    break
    failures = []
    for index, longtable, outer, origin, end in records:
        preview = " ".join(text[end:end + 70].split())
        if outer is not None:
            if "\\caption" not in closed.get(outer, text[end:]):
                failures.append(f"tabular in a table float without a caption ({origin}): {preview}")
        elif longtable:
            if "\\caption" not in closed.get(index, text[end:]):
                failures.append(f"longtable without a caption ({origin}): {preview}")
        else:
            failures.append(f"tabular outside a captioned table float ({origin}): {preview}")
    return failures


def path_whitespace(source: str) -> list[str]:
    """\\path and \\url drop spaces, so a space-separated command prints joined."""
    failures = []
    for match in re.finditer(r"\\(path|url)\s*\{", source):
        argument = source[match.end():_balanced(source, match.end() - 1) - 1]
        if re.search(r"\s", argument):
            failures.append(f"whitespace inside \\{match.group(1)}{{{' '.join(argument.split())[:70]}}}")
    for match in re.finditer(r"\\(path|url)\s*([|!+])(.*?)\2", source):
        if re.search(r"\s", match.group(3)):
            failures.append(f"whitespace inside \\{match.group(1)}{match.group(2)}{match.group(3)[:70]}")
    return failures


def command_pattern(command: str) -> re.Pattern:
    """Spaces between words are required; a line break may fall inside a word."""
    def word(part):
        return r"(?:[ \t]*\n[ \t]*)?".join(re.escape(char) for char in part)
    return re.compile(r"\s+".join(word(part) for part in command.split()))


def missing_commands(pdf_text: str, commands=SUPPLEMENT_COMMANDS) -> list[str]:
    return [f"command not printed with its spaces: {command}"
            for command in commands if not command_pattern(command).search(pdf_text)]


def _prose(source: str, pdf_text: str | None) -> str:
    return pdf_main_body(pdf_text) if pdf_text else detex(document_body(source))


def lint_main(source_text: str, pdf_text: str | None, *, root: Path = PAPER, read=None,
              stay_threshold: int = 5, hedge_limit: int = 6, min_section_words: int = 60,
              max_abstract_words: int = 200, repeat_limit: int = 2) -> list[str]:
    """Every rule on the main manuscript. ``pdf_text`` may be None before a build."""
    missing: list[str] = []
    source = expand_inputs(source_text, root, read, missing)
    failures = [f"input file is missing: {name}" for name in missing]
    prose = _prose(source, pdf_text)
    failures += stay_violations(prose, stay_threshold)
    found = banned_phrase_hits(detex(document_body(source)))
    for phrase, count in banned_phrase_hits(prose).items():
        found[phrase] = max(found.get(phrase, 0), count)
    failures += [f"banned phrase '{phrase}' appears {count} times" for phrase, count in sorted(found.items())]
    failures += repeated_sentences(prose, limit=repeat_limit)
    failures += un_hedge_violations(prose, hedge_limit)
    failures += short_sections(source, min_section_words)
    failures += abstract_violations(source, max_abstract_words)
    failures += orphan_tabulars(source)
    failures += path_whitespace(source)
    return failures


def lint_supplement(source_text: str, pdf_text: str | None, *, root: Path = PAPER, read=None,
                    commands=SUPPLEMENT_COMMANDS, min_section_words: int | None = 60) -> list[str]:
    """Banned phrases, section length (None skips it), tabulars, paths, and printed commands."""
    missing: list[str] = []
    source = expand_inputs(source_text, root, read, missing)
    failures = [f"input file is missing: {name}" for name in missing]
    found = banned_phrase_hits(detex(document_body(source)))
    for phrase, count in (banned_phrase_hits(pdf_main_body(pdf_text)) if pdf_text else {}).items():
        found[phrase] = max(found.get(phrase, 0), count)
    failures += [f"banned phrase '{phrase}' appears {count} times" for phrase, count in sorted(found.items())]
    if min_section_words is not None:
        failures += short_sections(source, min_section_words)
    failures += orphan_tabulars(source)
    failures += path_whitespace(source)
    if pdf_text is None:
        failures.append("supplement PDF text is unavailable; printed commands are unchecked")
    else:
        failures += missing_commands(pdf_text, commands)
    return failures


def pdf_layout_text(path: Path) -> str | None:
    """``pdftotext -layout`` output, or None when the PDF or the tool is absent."""
    if not Path(path).is_file():
        return None
    try:
        return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True, errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", type=Path, default=PAPER / "dope-mfs.tex")
    parser.add_argument("--main-pdf", type=Path, default=PAPER / "dope-mfs.pdf")
    parser.add_argument("--supplement", type=Path, default=PAPER / "supplement.tex")
    parser.add_argument("--supplement-pdf", type=Path, default=PAPER / "supplement.pdf")
    parser.add_argument("--root", type=Path, default=PAPER)
    parser.add_argument("--stay-threshold", type=int, default=5)
    args = parser.parse_args(argv)
    failures = ["main: " + item for item in lint_main(
        args.main.read_text(), pdf_layout_text(args.main_pdf), root=args.root, stay_threshold=args.stay_threshold)]
    failures += ["supplement: " + item for item in lint_supplement(
        args.supplement.read_text(), pdf_layout_text(args.supplement_pdf), root=args.root)]
    print("\n".join(failures) if failures else "paper lint passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
