"""One figure spec feeds the drawing code and the prose macros."""

from __future__ import annotations

import ast
import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import check_paper
import figure_spec

SCRIPTS = Path(__file__).resolve().parent
PAPER = SCRIPTS.parent
REPO = PAPER.parents[1]
DRAWING = ("render_figures.py", "v2_figures.py", "mfs_v3_table.py")


def _limit_calls(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("set_xlim", "set_ylim")):
            yield node


def _is_number(node) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _spec_imports(tree) -> set[str]:
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "figure_spec"
        for alias in node.names
    }


class FigureSpecTest(unittest.TestCase):
    def test_macros_equal_the_spec(self):
        found = figure_spec.parse_macros(figure_spec.macros_tex())
        stems = 0
        for spec in figure_spec.FIGURES.values():
            if spec.macro is None:
                continue
            stems += 1
            stem, axis = "Fig" + spec.macro, spec.axis.upper()
            self.assertEqual(found[stem + "Quantity"], spec.quantity)
            self.assertEqual(float(found[stem + axis + "Lo"]), spec.window[0])
            self.assertEqual(float(found[stem + axis + "Hi"]), spec.window[1])
        self.assertEqual(len(found), 3 * stems)
        # Fig. 2 draws retention levels clipped to [-1.5, 1.6], not paired differences.
        self.assertEqual(found["FigScatterQuantity"], "retention level")
        self.assertEqual((found["FigScatterYLo"], found["FigScatterYHi"]), ("-1.5", "1.6"))
        self.assertEqual(found["FigPairedQuantity"], "paired retention difference")
        self.assertEqual((found["FigPairedXLo"], found["FigPairedXHi"]), ("-1.5", "1.5"))
        self.assertEqual(figure_spec.macro_failures(figure_spec.macros_tex()), [])

    def test_macro_names_satisfy_the_paper_checker(self):
        names = set(figure_spec.macro_values())
        self.assertEqual(set(check_paper.MACRO_DEF.findall(figure_spec.macros_tex())), names)
        self.assertEqual(set(check_paper.MACRO_USE.findall(" ".join("\\" + n for n in names))), names)
        numbers = (PAPER / "generated" / "numbers.tex").read_text()
        for name in names:
            self.assertNotIn("\\newcommand{\\" + name + "}", numbers)
            self.assertNotIn(name, check_paper._LATEX)

    def test_check_detects_drift_and_a_missing_file(self):
        target = REPO / "target"
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            out = Path(directory)
            self.assertIn("missing", figure_spec.check_macros(out)[0])
            path = figure_spec.write_macros(out)
            self.assertEqual(path.name, "figure-spec.tex")
            self.assertEqual(figure_spec.check_macros(out), [])
            fresh = figure_spec.macros_tex()
            path.write_text(fresh.replace("{\\FigScatterYHi}{1.6}", "{\\FigScatterYHi}{1.5}"))
            self.assertEqual(figure_spec.check_macros(out), ["FigScatterYHi disagrees with figure_spec.py"])
            path.write_text(fresh + "\\newcommand{\\FigExtra}{1}\n")
            self.assertEqual(figure_spec.check_macros(out), ["FigExtra is not in figure_spec.py"])
            path.write_text(fresh.replace("\\newcommand{\\FigPairedXLo}{-1.5}\n", ""))
            self.assertEqual(figure_spec.check_macros(out), ["FigPairedXLo is missing"])
            path.write_text(fresh + "% trailing\n")
            self.assertEqual(len(figure_spec.check_macros(out)), 1)
            path.write_text(fresh)
            with patch.object(figure_spec, "GENERATED", out), redirect_stdout(io.StringIO()) as shown:
                self.assertEqual(figure_spec.main(["--check"]), 0)
                path.unlink()
                self.assertEqual(figure_spec.main(["--check"]), 1)
            self.assertIn("missing", shown.getvalue())

    def test_committed_macro_file_matches_the_spec(self):
        self.assertEqual(figure_spec.check_macros(), [])

    def test_drawing_code_reads_every_window_from_the_spec(self):
        for name in DRAWING:
            tree = ast.parse((SCRIPTS / name).read_text())
            imported = _spec_imports(tree)
            self.assertTrue(imported, name)
            calls = list(_limit_calls(tree))
            self.assertTrue(calls, name)
            for call in calls:
                self.assertFalse(any(_is_number(arg) for arg in call.args), f"{name} types a limit")
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id.endswith(("_LIM", "CLIP"))
                        for target in node.targets):
                    self.assertNotIsInstance(node.value, ast.Tuple, f"{name} redefines a window")
        render = _spec_imports(ast.parse((SCRIPTS / "render_figures.py").read_text()))
        self.assertLessEqual({"Y_LIM", "PAIRED_X_LIM", "LOSS_Y_LIM"}, render)

    def test_every_included_figure_has_a_spec(self):
        included = set()
        for name in ("dope-mfs.tex", "supplement.tex"):
            text = (PAPER / name).read_text()
            included.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", text))
        self.assertIn("retention-bytes.pdf", included)
        self.assertLessEqual(included, set(figure_spec.FIGURES))
        for spec in figure_spec.FIGURES.values():
            self.assertTrue((PAPER / spec.script).is_file() or (SCRIPTS / spec.script).is_file())
            if spec.window is not None:
                self.assertLess(spec.window[0], spec.window[1])


if __name__ == "__main__":
    unittest.main()
