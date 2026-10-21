"""Figure 1 must be the generated hero figure on page 1, not a placeholder or a floated figure."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_one as F  # noqa: E402

AUX = (r"\newlabel{fig:frontier}{{1}{1}{\textbf {Why this matters: how much learning survives at how few "
       r"bytes.} Historical CatBoost validation}{figure.1}{}}" + "\n")
LOG = ("<generated/hero-frontier.pdf, id=40, 607.068pt x 346.89601pt>\n"
       "File: generated/hero-frontier.pdf Graphic file (type pdf)\n"
       "<use generated/hero-frontier.pdf>\n")


class FigureOneTests(unittest.TestCase):
    def test_generated_hero_on_page_one_passes(self):
        self.assertEqual(F.failures(AUX, LOG), [])

    def test_pending_placeholder_is_refused(self):
        self.assertIn("no generated hero figure was included; Figure 1 would be a pending placeholder",
                      F.failures(AUX, "File: macros.tex\n"))

    def test_compression_frontier_counts_as_a_generated_hero(self):
        log = LOG.replace("hero-frontier.pdf", "compression-frontier.pdf")
        self.assertEqual(F.failures(AUX, log), [])

    def test_floated_or_renumbered_figure_is_refused(self):
        self.assertEqual(F.failures(AUX.replace("{{1}{1}", "{{1}{2}"), LOG), ["fig:frontier is not on page 1"])
        self.assertEqual(F.failures(AUX.replace("{{1}{1}", "{{2}{1}"), LOG), ["fig:frontier is not Figure 1"])
        self.assertEqual(F.failures(AUX.replace("{{1}{1}", "{{1}{11}"), LOG), ["fig:frontier is not on page 1"])

    def test_missing_label_is_refused(self):
        self.assertEqual(F.failures("", LOG), ["fig:frontier has no label in main.aux"])

    def test_log_line_wrapped_by_tex_still_counts(self):
        prefix = "x" * (F.LOG_WRAP - len("<use generated/"))
        wrapped = prefix + "<use generated/\nhero-frontier.pdf>\n"
        self.assertEqual(len(wrapped.splitlines()[0]), F.LOG_WRAP)
        self.assertEqual(F.failures(AUX, wrapped), [])

    def test_main_reads_the_build_directory_and_refuses_a_missing_build(self):
        local = Path(__file__).resolve().parents[2] / "target" / "paper-tests"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            out = Path(directory)
            self.assertEqual(F.main([str(out)]), 1)
            (out / "main.aux").write_text(AUX)
            (out / "main.log").write_text(LOG)
            self.assertEqual(F.main([str(out)]), 0)
            (out / "main.log").write_text("File: macros.tex\n")
            self.assertEqual(F.main([str(out)]), 1)


if __name__ == "__main__":
    unittest.main()
