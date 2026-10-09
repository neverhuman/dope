"""The v2 prose lint and count checks as one call for check_paper.py.

check_paper.main() extends its failures with run_v2_lint(). Every module used
here reads the standard library only, like check_paper itself.
"""

from __future__ import annotations

from pathlib import Path

from paper_consistency import check_v2
from paper_lint import PAPER, lint_main, lint_supplement, pdf_layout_text


def run_v2_lint(root: Path = PAPER) -> list[str]:
    """Lint dope-mfs.tex and supplement.tex with their built PDFs under ``root``."""
    root = Path(root)
    failures = []
    main_tex = root / "dope-mfs.tex"
    supplement_tex = root / "supplement.tex"
    if not main_tex.is_file() or not supplement_tex.is_file():
        return ["v2 lint: dope-mfs.tex or supplement.tex is missing"]
    main_pdf = pdf_layout_text(root / "dope-mfs.pdf")
    if main_pdf is None:
        failures.append("v2 lint main: dope-mfs.pdf text is unavailable; counts read the source")
    failures += ["v2 lint main: " + item for item in lint_main(main_tex.read_text(), main_pdf, root=root)]
    failures += ["v2 lint supplement: " + item for item in lint_supplement(
        supplement_tex.read_text(), pdf_layout_text(root / "supplement.pdf"), root=root)]
    # The count checks read the committed v2 panel, so they run for the repository paper only.
    counts = check_v2() if root.resolve() == Path(PAPER).resolve() else []
    return failures + ["v2 counts: " + item for item in counts]
