"""Anonymity failures and generated contract conformance examples."""
import json
import re
import unittest
from pathlib import Path

import check_paper
from research.benchmark.mfs_v3_score import _v3_contract


class PackagingTests(unittest.TestCase):
    def test_canonical_recipes_execute_one_real_replay_and_reject_the_old_path(self):
        # Import inside the test so discovery does not also collect the shared
        # PipelineTests class. Run the same real Just/reducer/PDF assertions once.
        from research.benchmark.tests.test_paper_build_pipeline import PipelineTests
        fixture = PipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.assert_canonical_recipes_execute_one_real_replay_and_reject_the_old_path()

    def test_body_limit_counts_text_before_references_on_the_same_page(self):
        first_ten = "body\f" * 10
        self.assertEqual(check_paper.presentation_failures(first_ten + 'R E F E R E N C E S\n'), [])
        self.assertIn('main body before References exceeds 10 pages',
                      check_paper.presentation_failures(first_ten + 'last body paragraph\nReferences\n'))
        self.assertIn('main PDF has no References heading',
                      check_paper.presentation_failures(first_ten))

    def test_frozen_negation_limit_rejects_six_but_allows_five_per_thousand(self):
        for count in (5, 6):
            text = 'word ' * (999 - count) + ' not ' * count + '\nReferences\n'
            self.assertEqual(len(text.split()), 1000)
            failed = 'main PDF negation density exceeds 5 per 1000 words'
            if count == 5:
                self.assertNotIn(failed, check_paper.presentation_failures(text))
            else:
                self.assertIn(failed, check_paper.presentation_failures(text))

    def test_nonprinting_pdf_glyphs_cannot_dilute_negation_density(self):
        text = 'word ' * 993 + ' not ' * 6 + '\nReferences\n'
        text = ' \x01 ' * 200 + text
        self.assertIn('main PDF negation density exceeds 5 per 1000 words',
                      check_paper.presentation_failures(text))

    def test_name_and_url_leaks_are_case_insensitive(self):
        for text in ('Jepson Taylor', 'alton alexander', 'NEVERHUMAN Research',
                     'https://github.com/neverhuman/dope', 'https://example.com/jepsontaylor'):
            self.assertTrue(check_paper.anonymous_leaks(text))
        self.assertFalse(check_paper.anonymous_leaks('Anonymous; https://github.com/EpistasisLab/pmlb'))

    def test_conformance_table_contains_every_contract_weight_and_floor(self):
        contract = _v3_contract()
        generated = check_paper.ROOT / 'generated'
        text = (generated / 'mfs-v3-conformance.tex').read_text()
        for key, weight in contract['master_fitness']['weights'].items():
            self.assertIn(key.replace('_', r'\_') + ' = ' + str(weight), text)
        self.assertIn('$q=' + str(contract['release_gates']['near_copy_quantile']) + '$', text)
        self.assertIn('linear interpolation', text)
        self.assertIn('Median linear MSE of three', text)
        self.assertIn(r'\newcommand{\MfsVThreePerfect}{100}', (generated / 'mfs-v3-examples.tex').read_text())
        supplement = (check_paper.ROOT / 'supplement.tex').read_text()
        self.assertIn(r'\input{generated/mfs-v3-conformance.tex}', supplement)
        self.assertIn(r'\input{generated/mfs-v3-examples.tex}', supplement)
        self.assertNotIn('100.0001', supplement)

    def test_a_stated_holm_family_size_names_its_members(self):
        bare = "\\begin{table}\\caption{Holm family size is nine.}\\end{table}"
        named = ("\\begin{table}\\caption{Holm family size is nine: DOPE against three comparators "
                 "under each of the three auditors.}\\end{table}")
        self.assertTrue(check_paper.table_member_failures(bare))
        self.assertEqual(check_paper.table_member_failures(named), [])
        self.assertEqual(check_paper.table_member_failures("\\begin{table}\\caption{No Holm family.}\\end{table}"), [])
        for path in (check_paper.TEX, check_paper.ROOT / "supplement.tex"):
            self.assertEqual(check_paper.table_member_failures(check_paper.strip_tex_comments(path.read_text())), [],
                             path.name)

    def test_main_text_mfs_v3_weights_and_epsilon_match_the_contract(self):
        """Rubric 9a: the prose weights and epsilon are the production contract's."""
        fitness = _v3_contract()['master_fitness']
        text = " ".join(check_paper.TEX.read_text().split())
        weights = fitness['weights']
        self.assertIn(f"utility transfer ${weights['utility_transfer']:.2f}$", text)
        same = {weights[key] for key in ('representation_closeness', 'driver_fidelity', 'distribution_fidelity')}
        self.assertEqual(len(same), 1)
        self.assertIn(f"representation closeness, driver fidelity, and distribution fidelity ${same.pop():.2f}$ each", text)
        self.assertIn(f"structure fidelity ${weights['structure_fidelity']:.2f}$", text)
        small = {weights['coverage_realism'], weights['compactness']}
        self.assertEqual(len(small), 1)
        self.assertIn(f"coverage realism and compactness ${small.pop():.2f}$ each", text)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(fitness['epsilon'], 1e-6)
        self.assertIn(r"\epsilon = 10^{-6}", text)

    def test_table_environment_must_declare_a_holm_family(self):
        self.assertEqual(check_paper.table_family_failures(
            "\\begin{table}\\caption{No Holm family.}\\end{table}"), [])
        self.assertEqual(check_paper.table_family_failures(
            "\\begin{longtable}\\caption{Holm family size is three.}\\end{longtable}"), [])
        self.assertTrue(check_paper.table_family_failures(
            "\\begin{table}\\caption{Inventory only.}\\end{table}"))
        escaped = "\\begin{table}\\caption{Median at $95\\%$ coverage. No Holm family.}\\end{table}"
        self.assertEqual(check_paper.table_family_failures(check_paper.strip_tex_comments(escaped)), [])
        for path in (check_paper.TEX, check_paper.ROOT / "supplement.tex"):
            self.assertEqual(check_paper.table_family_failures(path.read_text()), [], path.name)
        for path in (check_paper.ROOT / "generated").glob("*.tex"):
            self.assertEqual(check_paper.table_family_failures(path.read_text()), [], path.name)

    def test_titles_metadata_and_profile_selection_disclosure_agree(self):
        """One title in the paper, the supplement and CITATION.cff; the withdrawn selection claims stay out."""
        main = check_paper.TEX.read_text()
        supplement = (check_paper.ROOT / "supplement.tex").read_text()
        registry = (check_paper.ROOT / "scripts" / "method_registry.py").read_text()
        citation = (check_paper.ROOT.parents[1] / "CITATION.cff").read_text()

        def plain_title(source):
            match = re.search(r"\\title\{\\texorpdfstring\{.*?\}\{([^{}]*)\}\}", source, re.S)
            self.assertIsNotNone(match, "no \\texorpdfstring title")
            return match.group(1)

        title = plain_title(main)
        self.assertEqual(plain_title(supplement), "Supplement: " + title)
        cited = re.search(r"preferred-citation:.*?\n  title: \"([^\"]*)\"", citation, re.S)
        self.assertIsNotNone(cited)
        self.assertEqual(cited.group(1), title)
        self.assertTrue(title.startswith("DOPE: "))
        self.assertNotIn("Kilobyte", title)
        self.assertIn("Downstream Objective-Preserving Encoding", main)
        # The round records keep all four profiles and record no selection, so these statements are withdrawn.
        for text in (main, supplement, registry):
            for withdrawn in ("chosen on a six-lineage discovery panel", "checked on six disjoint confirmation",
                              "pre-registered reference", "pre-registered profile"):
                self.assertNotIn(withdrawn, text)
        for text in (main, supplement):
            self.assertIn("selection-exposed", text)

    def test_readme_citation_and_pdf_metadata_carry_the_one_title_without_kilobyte_branding(self):
        """README heading and bibtex, CITATION.cff and both pdftitle entries name the paper title; byte counts stay measurements."""
        main = check_paper.TEX.read_text()
        supplement = (check_paper.ROOT / "supplement.tex").read_text()
        readme = (check_paper.ROOT.parents[1] / "README.md").read_text()
        citation = (check_paper.ROOT.parents[1] / "CITATION.cff").read_text()
        title = re.search(r"\\title\{\\texorpdfstring\{.*?\}\{([^{}]*)\}\}", main, re.S).group(1)
        self.assertIn(r"pdftitle={" + title + "}", main)
        self.assertIn(r"pdftitle={Supplement: " + title + "}", supplement)
        self.assertEqual(readme.splitlines()[0], "# DOPE — " + title.removeprefix("DOPE: "))
        bibtex = re.search(r"@misc\{taylor2026dope,\n  title  = \{(.*)\},\n", readme)
        self.assertIsNotNone(bibtex)
        self.assertEqual(bibtex.group(1), "{DOPE}: " + title.removeprefix("DOPE: "))
        prose = readme.split("<!-- BEGIN BENCHMARK -->")[0] + readme.split("<!-- END BENCHMARK -->")[1]
        for text in (prose, citation, title):
            self.assertNotRegex(text.lower(), r"kilobyte[- ]scale|kilobyte generator")
        self.assertIn("Downstream Objective-Preserving Encoding", readme)

    def test_profile_disclosure_cites_what_each_committed_publication_records(self):
        """The supplement's account of the three profile publications must match their committed fields."""
        results = check_paper.ROOT.parents[1] / "research" / "benchmark" / "results"
        for name in ("s3-matched-discovery-validation.json", "s3-matched-confirmation-validation.json"):
            publication = json.loads((results / name).read_text())
            self.assertEqual(publication["fit_seeds"], [11], name)
            self.assertTrue(publication["scope"].endswith("All four DOPE profiles retained; production family unselected."),
                            name)
        population = json.loads((results / "dope-s3-population-validation.json").read_text())
        self.assertIs(population["global_family_selected"], False)
        self.assertEqual(population["fit_seeds"], [11])
        self.assertEqual(sorted({cell["profile"] for cell in population["cells"]}),
                         ["features12_steps2048", "features12_steps512", "features24_steps2048", "features24_steps512"])
        self.assertIn("descriptive aggregates, with no DOPE win or production family selection",
                      (results / "dope-s3-population-validation.md").read_text())
        supplement = (check_paper.ROOT / "supplement.tex").read_text()
        self.assertIn("``production family unselected''", supplement)
        self.assertIn(r"\texttt{global\_\allowbreak{}family\_\allowbreak{}selected} to false", supplement)
        self.assertNotIn("records that no global family was selected", supplement)


if __name__ == '__main__':
    unittest.main()
