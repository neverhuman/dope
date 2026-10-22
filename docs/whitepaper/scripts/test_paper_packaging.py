"""Anonymity failures and generated contract conformance examples."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import check_paper
from research.benchmark.mfs_v3_score import _v3_contract


class PackagingTests(unittest.TestCase):
    def test_a_cold_paper_recipe_creates_the_density_table_before_checking_its_numbers(self):
        """Run Just with real pinned table rendering and the real protocol validator."""
        repo = check_paper.ROOT.parents[1]
        just = shutil.which("just")
        self.assertIsNotNone(just, "canonical paper recipe regression requires Just")
        target = repo / "target/paper-pipeline-tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as folder:
            root = Path(folder)
            generated = root / "docs/whitepaper/generated"
            table = generated / "mfs-v3-density-table.tex"
            generator = "    python3 docs/whitepaper/scripts/mfs_v3_table.py\n"
            validator = "    python3 research/benchmark/verify_paper_numbers.py\n"
            current = (repo / "Justfile").read_text()
            # Reintroduce the ordering bug without changing either real Python implementation.
            late = current.replace(generator, "", 1).replace(validator, validator + generator, 1)
            runner = root / "recipe-runner.py"
            runner.write_text(f"""import os, pathlib, sys
sys.path[:0] = [{str(repo)!r}, {str(repo / 'docs/whitepaper/scripts')!r}]
args = sys.argv[1:]
if args[:1] == ['-B']:
    args = args[1:]
table = pathlib.Path({str(table)!r})
if args == ['ops/ci/paper-prepare.py']:
    table.parent.mkdir(parents=True, exist_ok=True)
elif args == ['docs/whitepaper/scripts/mfs_v3_table.py']:
    import mfs_v3_table as mfs
    text = mfs.table_tex(mfs.measure(mfs.load_receipt()))
    table.write_text('corrupt table' if os.environ.get('CORRUPT_DENSITY_TABLE') else text)
elif args == ['research/benchmark/verify_paper_numbers.py']:
    import research.benchmark.verify_paper_numbers as verify
    original = verify._must_contain
    def check(path, snippet, failures):
        original(table if path.name == table.name else path, snippet, failures)
    verify._must_contain = check
    verify.protocol_tokens()
""")
            binary = root / "bin"
            binary.mkdir()
            shim = binary / "python3"
            shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" -B "{runner}" "$@"\n')
            shim.chmod(0o755)
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
            env.pop("CORRUPT_DENSITY_TABLE", None)

            def run_paper(text, *, corrupt=False):
                shutil.rmtree(generated, ignore_errors=True)
                justfile = root / "Justfile"
                justfile.write_text(text)
                return subprocess.run(
                    [just, "--justfile", str(justfile), "--working-directory", str(root), "paper"],
                    env=dict(env, CORRUPT_DENSITY_TABLE="1") if corrupt else env,
                    capture_output=True, text=True, timeout=120,
                )

            bad = run_paper(late)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("FileNotFoundError", bad.stderr)
            self.assertIn(table.name, bad.stderr)
            good = run_paper(current)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(table.read_bytes(), (repo / table.relative_to(root)).read_bytes())
            corrupt = run_paper(current, corrupt=True)
            self.assertNotEqual(corrupt.returncode, 0)
            self.assertIn("does not contain", corrupt.stderr)

    def test_canonical_recipes_execute_one_real_replay_and_reject_the_old_path(self):
        # Import inside the test so discovery does not also collect the shared
        # PipelineTests class. Run the same real Just/reducer/PDF assertions once.
        from research.benchmark.tests.test_paper_build_pipeline import PipelineTests
        fixture = PipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.assert_canonical_recipes_execute_one_real_replay_and_reject_the_old_path()

    def test_a_cold_paper_recipe_renders_the_native_byte_paragraph_before_the_driver(self):
        """paper-clean deletes generated/; the real paper recipe must render this paragraph before the driver checks it.

        The actual Justfile runs through a python3 shim: paper-prepare only creates generated/, the shim execs
        the copied native_byte_conformance.py in its place (--write, then the default check at the driver stub)
        so the checker's own exit status is the step's status, and the other generators are traced without
        running. The same Justfile without the generator line must fail, and so must a checker that crashes.
        """
        repo = check_paper.ROOT.parents[1]
        just = shutil.which("just")
        self.assertIsNotNone(just, "canonical paper recipe regression requires Just")
        target = repo / "target/paper-pipeline-tests"
        target.mkdir(parents=True, exist_ok=True)
        folder = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        checker = "docs/whitepaper/scripts/native_byte_conformance.py"
        current = (repo / "Justfile").read_text()
        recipe = current.split("\npaper:\n", 1)[1].split("\n\n", 1)[0].splitlines()
        positions = {key: [index for index, line in enumerate(recipe) if key in line]
                     for key in ("ops/ci/paper-prepare.py", checker + " --write", "docs/whitepaper/scripts/paper_build.py")}
        self.assertEqual([len(found) for found in positions.values()], [1, 1, 1], recipe)
        prepare, generate, driver = (found[0] for found in positions.values())
        self.assertLess(prepare, generate)
        self.assertLess(generate, driver)
        missing = "".join(line for line in current.splitlines(keepends=True) if checker + " --write" not in line)
        for name in (checker, "docs/whitepaper/receipts/native-byte-conformance-v1.json", "HISTORY.md",
                     "docs/whitepaper/supplement.tex"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, root / name)
        trace = root / "commands.jsonl"
        runner = root / "recipe-runner.py"
        runner.write_text(f"""import json, os, pathlib, sys
root = pathlib.Path({str(root)!r})
args = sys.argv[1:]
if args[:1] == ["-B"]:
    args = args[1:]
with pathlib.Path({str(trace)!r}).open("a") as log:
    log.write(json.dumps(args) + "\\n")
checker = {checker!r}
real_checker = [{sys.executable!r}, "-B", str(root / checker)]
if args[:1] == ["ops/ci/paper-prepare.py"]:
    (root / "docs/whitepaper/generated").mkdir(parents=True, exist_ok=True)
elif args[:1] == [checker]:
    # Become the real checker: its own exit status is this recipe step's status.
    os.chdir(root)
    os.execv(real_checker[0], [*real_checker, *args[1:]])
elif args[:1] == ["docs/whitepaper/scripts/paper_build.py"]:
    # The canonical driver checks the captured paragraph before any numerical replay.
    os.chdir(root)
    os.execv(real_checker[0], real_checker)
""")
        binary = root / "bin"
        binary.mkdir()
        (binary / "python3").write_text(f"#!/bin/sh\nexec {sys.executable} -B {runner} \"$@\"\n")
        (binary / "python3").chmod(0o755)
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
        generated = root / "docs/whitepaper/generated/native-byte-conformance.tex"

        def run_paper(name, text):
            """Run the real recipe cold; check=True turns any failed step into CalledProcessError."""
            shutil.rmtree(root / "docs/whitepaper/generated", ignore_errors=True)
            trace.write_text("")
            (root / f"Justfile.{name}").write_text(text)
            subprocess.run([just, "--justfile", str(root / f"Justfile.{name}"), "--working-directory", str(root),
                            "paper"], env=env, capture_output=True, text=True, timeout=120, check=True)

        def commands():
            return [json.loads(line)[0] for line in trace.read_text().splitlines()]

        run_paper("current", current)
        self.assertEqual(generated.read_bytes(),
                         (repo / "docs/whitepaper/generated/native-byte-conformance.tex").read_bytes())
        self.assertLess(commands().index(checker), commands().index("docs/whitepaper/scripts/paper_build.py"))
        with self.assertRaises(subprocess.CalledProcessError):
            run_paper("missing", missing)
        self.assertFalse(generated.exists())
        self.assertEqual(commands()[-1], "docs/whitepaper/scripts/paper_build.py")
        self.assertNotIn(checker, commands())
        (root / checker).write_text('raise RuntimeError("hostile checker")\n')
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            run_paper("hostile", current)
        self.assertEqual(raised.exception.returncode, 1)
        self.assertIn("RuntimeError: hostile checker", raised.exception.stderr)
        self.assertFalse(generated.exists())
        self.assertEqual(commands()[-1], checker)

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

    def test_measured_mfs_gate_failures_are_not_called_unmeasured(self):
        for phrase in (
            "The score is null for every method in this paper because its privacy and representation gates are "
            "still unmeasured.",
            "The release score is null until its privacy and representation gates are measured.",
            "The privacy gate is unmeasured.",
        ):
            self.assertTrue(check_paper.mfs_gate_state_failures(phrase), phrase)
        for phrase in (
            "because the four prerequisites and the cleartext scan are unmeasured; measured copy, membership, and "
            "attribute gates also fail on some method--lineage cells.",
            "The release score is null because its prerequisites and cleartext scan are unmeasured.",
            "Cleartext is unscanned on every method--lineage cell.",
        ):
            self.assertEqual(check_paper.mfs_gate_state_failures(phrase), [], phrase)
        sources = [check_paper.TEX, check_paper.ROOT / "supplement.tex", check_paper.ROOT.parents[1] / "README.md"]
        sources.append(check_paper.ROOT.parents[1] / "HISTORY.md")
        sources += sorted((check_paper.ROOT / "generated").glob("*.tex"))
        for path in sources:
            self.assertEqual(check_paper.mfs_gate_state_failures(check_paper.strip_tex_comments(path.read_text())), [],
                             path.name)

    def test_the_real_row_control_is_a_diagnostic_not_a_bound(self):
        for phrase in (
            "resampled real rows remain the upper reference (Section 5).",
            "it measures what extra rows alone do and is an upper reference rather than a generator.",
            "retention mixes fidelity with row count, which the real-row control bounds.",
            "Resampled real rows are an upper bound on retention.",
        ):
            self.assertTrue(check_paper.real_row_bound_failures(phrase), phrase)
        for phrase in (
            "and is below resampled real rows, which diagnose row count rather than bound retention.",
            "it is a diagnostic of what extra rows alone do, neither a generator nor an upper bound on retention.",
            "the real-row control diagnoses that effect but does not bound it.",
            "Resampled real rows are a row-count diagnostic, neither a generator nor an upper bound.",
            "The interval's upper bound is printed beneath each median.",
        ):
            self.assertEqual(check_paper.real_row_bound_failures(phrase), [], phrase)
        sources = [check_paper.TEX, check_paper.ROOT / "supplement.tex", check_paper.ROOT.parents[1] / "README.md"]
        sources.append(check_paper.ROOT.parents[1] / "HISTORY.md")
        sources += sorted((check_paper.ROOT / "generated").glob("*.tex"))
        for path in sources:
            self.assertEqual(check_paper.real_row_bound_failures(check_paper.strip_tex_comments(path.read_text())), [],
                             path.name)

    def test_a_stated_holm_family_size_names_its_members(self):
        bare = "\\begin{table}\\caption{Holm family size is nine.}\\end{table}"
        named = ("\\begin{table}\\caption{Holm family size is nine: DOPE against three comparators "
                 "under each of the three auditors.}\\end{table}")
        self.assertTrue(check_paper.table_member_failures(bare))
        self.assertEqual(check_paper.table_member_failures(named), [])
        self.assertEqual(check_paper.table_member_failures("\\begin{table}\\caption{No Holm family.}\\end{table}"), [])
        old_table_one = ("\\begin{table}\\caption{Holm families are pre-declared per size: the primary family has "
                         "$12$ tests (author-default ARF, TabSyn, by three auditors), the density family $9$, and the "
                         "control family $6$; untested members keep their slot.}\\end{table}")
        self.assertEqual(len(check_paper.table_member_failures(old_table_one)), 2)
        sizes_only = "\\begin{table}\\caption{The block's Holm family (nine, six, and three tests).}\\end{table}"
        self.assertTrue(check_paper.table_member_failures(sizes_only))
        cluster = "\\begin{table}\\caption{Its family-cluster bootstrap 95\\% interval. No Holm family.}\\end{table}"
        self.assertEqual(check_paper.table_member_failures(cluster), [])
        sources = [check_paper.TEX, check_paper.ROOT / "supplement.tex"]
        sources += sorted((check_paper.ROOT / "generated").glob("*.tex"))
        for path in sources:
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
        self.assertTrue(title.startswith("D.O.P.E.: "))
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
        self.assertEqual(readme.splitlines()[0], "# " + title.split(" -- ")[0])
        bibtex = re.search(r"@misc\{taylor2026dope,\n  title  = \{(.*)\},\n", readme)
        self.assertIsNotNone(bibtex)
        self.assertEqual(bibtex.group(1), "{D.O.P.E.}: " + title.removeprefix("D.O.P.E.: "))
        self.assertNotIn("<!-- BEGIN BENCHMARK -->", readme)
        history = (check_paper.ROOT.parents[1] / "HISTORY.md").read_text()
        history = history.split("<!-- BEGIN BENCHMARK -->")[0] + history.split("<!-- END BENCHMARK -->")[1]
        for text in (readme, history, citation, title):
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
