"""Prose and layout lint rules on small fixtures."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import mfs_v3_table
import paper_lint as lint
import paper_v2_guards

ROOT = Path("/fixture")
REPO = Path(__file__).resolve().parents[3]
# The pre-split mfs-v3 emission: a sentence, then an uncaptioned centered tabular.
LEGACY_SCALAR = (
    "The receipt scores 0 of 388 method--lineage cells, so the scalar is null.\n"
    "\\begin{center}\\scriptsize\n\\begin{tabular}{@{}ll@{}}\n"
    "Kumo large & \\texttt{row\\_\\allowbreak{}project} \\\\\n\\end{tabular}\n\\end{center}\n"
)
TABLE = "\\begin{tabular}{ll}\n\\toprule\nA & B \\\\\n\\bottomrule\n\\end{tabular}\n"
PROSE = " ".join(["Each lineage keeps its own retention value under every auditor here."] * 7)


def reader(files):
    def read(path):
        return files.get(Path(path).relative_to(ROOT).as_posix())
    return read


def manuscript(body, abstract="A short abstract with plain words."):
    return (
        "\\documentclass{article}\n\\newcommand{\\fittowidthfile}[1]{\\begingroup\\input{#1}\\endgroup}\n"
        "\\begin{document}\n\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n"
        + body + "\n\\bibliography{references}\n\\end{document}\n"
    )


class SourceExpansionTests(unittest.TestCase):
    def test_comments_inputs_and_fit_to_width_expand(self):
        self.assertEqual(lint.strip_comments("50\\% kept % dropped\n\\\\% gone"), "50\\% kept \n\\\\")
        files = {"generated/a.tex": "outer \\input{generated/b} % hidden \\input{generated/c.tex}",
                 "generated/b.tex": "inner"}
        missing = []
        text = lint.expand_inputs("x \\fittowidthfile{generated/a.tex} \\input{generated/z.tex}",
                                  ROOT, reader(files), missing)
        self.assertIn("\\lintbeginfile{generated/a.tex}outer \\lintbeginfile{generated/b}inner", text)
        self.assertNotIn("hidden", text)
        self.assertEqual(missing, ["generated/z.tex"])
        self.assertEqual(lint.expand_inputs("\\input{#1}", ROOT, reader({})), "\\input{#1}")

    def test_detex_keeps_captions_and_drops_tables_math_and_definitions(self):
        text = lint.detex(
            "\\newcommand{\\Med}{0.940}\\begin{table}[t]\\caption{\\textbf{Kept caption.} No Holm family.}"
            + TABLE + "\\end{table} See \\cref{tab:x} and $\\Med$~lineages \\citep{a}.\n"
            "\\begin{equation}x=1\\end{equation} 10{,}240 bytes \\path{a/b.json}."
        )
        self.assertEqual(text, "Kept caption. No Holm family. See REF and Med lineages CITE . 10,240 bytes a/b.json.")


class OrphanTabularTests(unittest.TestCase):
    def test_legacy_mfs_scalar_is_an_orphan_and_the_split_float_is_not(self):
        source = lint.expand_inputs("Text.\n\\input{generated/mfs-v3-scalar.tex}\n", ROOT,
                                    reader({"generated/mfs-v3-scalar.tex": LEGACY_SCALAR}))
        found = lint.orphan_tabulars(source)
        self.assertEqual(len(found), 1)
        self.assertIn("tabular outside a captioned table float (generated/mfs-v3-scalar.tex)", found[0])
        self.assertEqual(lint.orphan_tabulars(mfs_v3_table.layers_tex()), [])

    def test_floats_longtables_and_groups(self):
        captioned = "\\begin{table}[t]\\caption{C. No Holm family.}\\fittowidthfile{generated/t.tex}\\end{table}"
        bare = "\\begin{table}[t]\\centering\\fittowidthfile{generated/t.tex}\\end{table}"
        files = {"generated/t.tex": TABLE}

        def orphans(text):
            return lint.orphan_tabulars(lint.expand_inputs(text, ROOT, reader(files)))

        self.assertEqual(orphans(captioned), [])
        self.assertEqual(len(orphans(bare)), 1)
        self.assertIn("without a caption", orphans(bare)[0])
        # Two floats at the same depth keep separate captions.
        self.assertEqual(len(orphans(captioned + bare)), 1)
        self.assertEqual(len(orphans(bare + captioned)), 1)
        grouped = "\\begingroup\\let\\tabular\\longtable\\let\\endtabular\\endlongtable\\input{generated/t.tex}\\endgroup"
        self.assertIn("longtable without a caption (generated/t.tex)", orphans(grouped)[0])
        files["generated/c.tex"] = TABLE.replace("\\toprule", "\\caption{Rows. No Holm family.}\\\\\n\\toprule")
        self.assertEqual(orphans(grouped.replace("t.tex", "c.tex")), [])
        self.assertIn("outside a captioned table float", orphans("\\input{generated/t.tex}")[0])
        self.assertEqual(len(orphans("\\begin{figure}\\caption{F.}\\input{generated/t.tex}\\end{figure}")), 1)
        self.assertEqual(orphans("\\begin{longtable}{ll}\\caption{L.}\\\\ A & B\\end{longtable}"), [])
        # A value-over-interval cell nests a tabular; its parent carries the caption check.
        cell = "\\begin{tabular}[t]{@{}c@{}}$1$\\\\[0,2]\\end{tabular}"
        self.assertEqual(orphans("\\begin{longtable}{ll}\\caption{L.}\\\\ A & " + cell + "\\end{longtable}"), [])
        self.assertEqual(len(orphans("\\begin{tabular}{ll}A & " + cell + "\\end{tabular}")), 1)


class PdfTextTests(unittest.TestCase):
    def test_body_stops_at_references_and_joins_hyphenated_breaks(self):
        body = lint.pdf_main_body("A re-\n   lease gate.\fMore text.\nR EFERENCES\nStays null.")
        self.assertEqual(body, "A release gate. More text.")

    def test_printed_commands_need_their_spaces(self):
        command = "python3 -m research.benchmark.fetch_pmlb --list"
        self.assertEqual(len(lint.missing_commands("python3-mresearch.benchmark.fetch_pmlb--list", [command])), 1)
        for printed in ("run python3 -m research.benchmark.fetch_pmlb --list now",
                        "python3 -m research.bench\n      mark.fetch_pmlb --list",
                        "python3 -m\n   research.benchmark.fetch_pmlb    --list"):
            self.assertEqual(lint.missing_commands(printed, [command]), [], printed)
        self.assertEqual(len(lint.missing_commands("", lint.SUPPLEMENT_COMMANDS)), 2)

    def test_path_and_url_arguments_carry_no_whitespace(self):
        found = lint.path_whitespace("\\path{python3 -m research.x --list} \\path{docs/a.json} \\url{https://a b}")
        self.assertEqual(found, ["whitespace inside \\path{python3 -m research.x --list}",
                                 "whitespace inside \\url{https://a b}"])
        self.assertEqual(lint.path_whitespace("\\url{https://github.com/x}\\path|a b|"), ["whitespace inside \\path|a b"])


class ProseRuleTests(unittest.TestCase):
    def test_stay_count_and_banned_phrases(self):
        self.assertEqual(lint.stay_violations("stay stays stayed staying stay", 5), [])
        self.assertEqual(lint.stay_violations("Stay stays stayed staying stay stays. Stayer.", 5),
                         ["'stay' forms appear 6 times (limit 5)"])
        hits = lint.banned_phrase_hits("Equivalence Stays\n Unclaimed. Wave 2 and wave 20. Still In progress.")
        self.assertEqual(hits, {"stays unclaimed": 1, "wave 2": 1, "in progress": 1})

    def test_un_hedges_and_repeated_sentences(self):
        self.assertEqual(lint.un_hedge_violations("unclaimed unpooled unmade unshown unstored unscored"), [])
        found = lint.un_hedge_violations("Unclaimed unclaimed unpooled unmade unshown unstored unscored unclaimedness")
        self.assertEqual(found, ["un- hedges appear 7 times (limit 6): unclaimed 2, unmade 1, unpooled 1, "
                                 "unscored 1, unshown 1, unstored 1"])
        text = ("MFS-v2 and MFS-v3 stay null. Mfs-v2 and MFS–v3 stay null! MFSv2 and mfsv3 stay  null. "
                "Official tests stay sealed. Official tests stay sealed. Official tests stay sealed.")
        self.assertEqual(lint.repeated_sentences(text), ["sentence repeated 3 times (limit 2): 'mfsv2 and mfsv3 stay null'"])
        self.assertEqual(lint.repeated_sentences(text, limit=3), [])

    def test_section_bodies_include_subsections_and_skip_floats(self):
        source = manuscript(
            "\\section{Method}\n\\subsection{Short}\nOnly a few words here.\n"
            "\\begin{table}\\caption{" + PROSE + "}\\end{table}\n"
            "\\subsection{Long}\n" + PROSE + "\n\\section{Related Work}\nBrief.\n"
        )
        bodies = lint.section_bodies(source)
        self.assertEqual([(kind, title) for kind, title, _count in bodies],
                         [("section", "Method"), ("subsection", "Short"), ("subsection", "Long"), ("section", "Related Work")])
        counts = {title: count for _kind, title, count in bodies}
        self.assertEqual(counts["Short"], 5)
        self.assertEqual(counts["Long"], 77)
        self.assertEqual(counts["Method"], 82)
        self.assertEqual(lint.short_sections(source), [
            "subsection 'Short' body has 5 words (minimum 60)",
            "section 'Related Work' body has 1 words (minimum 60)",
        ])

    def test_abstract_word_limit(self):
        self.assertEqual(lint.abstract_violations(manuscript("", "word " * 200)), [])
        self.assertEqual(lint.abstract_violations(manuscript("", "word " * 201)), ["abstract has 201 words (maximum 200)"])
        self.assertEqual(lint.abstract_violations("\\begin{document}\\end{document}"), ["no abstract environment"])


class LintEntryPointTests(unittest.TestCase):
    def test_lint_main_reports_each_rule_and_passes_a_clean_paper(self):
        files = {"generated/mfs-v3-scalar.tex": LEGACY_SCALAR}
        dirty = manuscript("\\section{Results}\n" + PROSE + " Equivalence stays unclaimed.\n"
                           "\\input{generated/mfs-v3-scalar.tex}\\path{python3 -m x}\n")
        found = lint.lint_main(dirty, None, root=ROOT, read=reader(files))
        self.assertEqual(found[0], "banned phrase 'stays unclaimed' appears 1 times")
        self.assertIn("sentence repeated 7 times", found[1])
        self.assertTrue(found[2].startswith("tabular outside a captioned table float (generated/mfs-v3-scalar.tex)"))
        self.assertEqual(found[3], "whitespace inside \\path{python3 -m x}")
        self.assertEqual(len(found), 4)
        clean = manuscript("\\section{Results}\n" + " ".join(f"Lineage {n} keeps one retention." for n in range(20)))
        self.assertEqual(lint.lint_main(clean, None, root=ROOT, read=reader({})), [])
        pdf = "Stays null and stays null.\nR EFERENCES\nstays null"
        self.assertIn("banned phrase 'stays null' appears 2 times", lint.lint_main(clean, pdf, root=ROOT, read=reader({})))
        self.assertEqual(lint.lint_main(clean + "\\input{generated/gone.tex}", None, root=ROOT, read=reader({}))[0],
                         "input file is missing: generated/gone.tex")

    def test_lint_supplement_checks_printed_commands(self):
        source = manuscript("\\section{Data}\n" + PROSE + " Run \\texttt{python3 -m research.benchmark.fetch\\_pmlb --list}.")
        self.assertEqual(lint.lint_supplement(source, None, root=ROOT, read=reader({}), commands=lint.SUPPLEMENT_COMMANDS[:1]),
                         ["supplement PDF text is unavailable; printed commands are unchecked"])
        printed = "Run python3 -m research.benchmark.fetch_pmlb --list."
        self.assertEqual(lint.lint_supplement(source, printed, root=ROOT, read=reader({}),
                                              commands=lint.SUPPLEMENT_COMMANDS[:1]), [])
        self.assertEqual(len(lint.lint_supplement(source, printed, root=ROOT, read=reader({}))), 1)

    def test_cli_and_v2_hook_on_fixture_files(self):
        target = REPO / "target"
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory)
            clean = manuscript("\\section{Results}\n" + " ".join(f"Lineage {n} keeps one retention." for n in range(20)))
            (root / "dope-mfs.tex").write_text(clean)
            (root / "supplement.tex").write_text(clean)
            self.assertIsNone(lint.pdf_layout_text(root / "dope-mfs.pdf"))
            self.assertEqual(paper_v2_guards.run_v2_lint(root), [
                "v2 lint main: dope-mfs.pdf text is unavailable; counts read the source",
                "v2 lint supplement: supplement PDF text is unavailable; printed commands are unchecked",
            ])
            arguments = ["--main", str(root / "dope-mfs.tex"), "--main-pdf", str(root / "none.pdf"),
                         "--supplement", str(root / "supplement.tex"), "--supplement-pdf", str(root / "none.pdf"),
                         "--root", str(root)]
            with redirect_stdout(io.StringIO()) as shown:
                self.assertEqual(lint.main(arguments), 1)
            self.assertIn("printed commands are unchecked", shown.getvalue())
            (root / "dope-mfs.tex").unlink()
            self.assertEqual(paper_v2_guards.run_v2_lint(root), ["v2 lint: dope-mfs.tex or supplement.tex is missing"])

    def test_real_manuscript_lints_without_error(self):
        for name, run in (("dope-mfs.tex", lint.lint_main), ("supplement.tex", lint.lint_supplement)):
            found = run((lint.PAPER / name).read_text(), None)
            self.assertIsInstance(found, list)
            self.assertTrue(all(isinstance(item, str) and item for item in found))


if __name__ == "__main__":
    unittest.main()
