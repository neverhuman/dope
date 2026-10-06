"""The paper check reads the third-pass LaTeX logs, not a handwritten claim."""

import importlib.util
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "check_paper",
    _ROOT / "docs/whitepaper/scripts/check_paper.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class PaperLog(unittest.TestCase):
    def test_overfull_and_underfull_are_reported(self):
        text = "\n".join(
            [
                "Overfull \\hbox (2.0pt too wide) in paragraph at lines 1--2",
                "Underfull \\vbox (badness 1000) has occurred while \\output is active",
                "Output written on dope-mfs.pdf (8 pages, 1 bytes).",
            ]
        )
        hits = _MODULE.latex_warnings(text)
        self.assertEqual(len(hits), 2)

    def test_undefined_reference_and_float_are_reported(self):
        text = "\n".join(
            [
                "LaTeX Warning: Reference `tab:cost' on page 3 undefined on input line 294.",
                "LaTeX Warning: Float too large for page by 10.0pt on input line 40.",
            ]
        )
        self.assertEqual(len(_MODULE.latex_warnings(text)), 2)

    def test_clean_log_is_empty(self):
        text = "Output written on dope-mfs.pdf (8 pages, 1 bytes).\n"
        self.assertEqual(_MODULE.latex_warnings(text), [])


if __name__ == "__main__":
    unittest.main()
