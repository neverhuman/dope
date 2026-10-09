"""The v2 prose lint as one call for check_paper.py.

Not called yet, because the current manuscript fails it. When it passes, add
``from paper_v2_guards import run_v2_lint`` beside check_paper's
``from public_hardware import banned_hits`` and this line to check_paper.main()
just before the final ``if failures:``:

    failures.extend(run_v2_lint())

Both modules use the standard library only, like check_paper itself.
"""

from __future__ import annotations

from pathlib import Path

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
    return failures
