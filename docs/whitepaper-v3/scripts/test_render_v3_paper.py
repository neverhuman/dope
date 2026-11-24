"""Tests for the DOPE v3.0.0 paper renderer and its modules (tables, third-party methods, final look).

Run: python3 -B -m unittest discover -s docs/whitepaper-v3/scripts -p 'test_*.py'
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_v3_paper as rp  # noqa: E402
import v3_paper_final_look as final_mod  # noqa: E402
import v3_paper_methods as methods_mod  # noqa: E402
import v3_paper_tables as tables  # noqa: E402

SCOREBOARD = rp.REPO / rp.SCOREBOARD


def load(name: str) -> dict:
    return json.loads((SCOREBOARD / name).read_text(encoding="utf-8"))


def readme() -> str:
    return (SCOREBOARD / "readme-block.md").read_text(encoding="utf-8")


class CellFormatting(unittest.TestCase):
    def test_fixed_and_signed_never_print_negative_zero(self):
        self.assertEqual(tables.fixed3(-0.0001), "0.000")
        self.assertEqual(tables.signed3(-0.0001), "0.000")
        self.assertEqual(tables.signed3(0.0604), "+0.060")
        self.assertEqual(tables.signed3(-0.0526), "-0.053")

    def test_p_value_floor(self):
        self.assertEqual(tables.p_value(0.0004), "<0.001")
        self.assertEqual(tables.p_value(0.0094), "0.009")

    def test_tex_turns_signs_into_math_minus_and_lone_dashes_into_en_dashes(self):
        self.assertEqual(tables.tex("-0.121 [-0.189, -0.049]"),
                         r"\ensuremath{-}0.121 [\ensuremath{-}0.189, \ensuremath{-}0.049]")
        self.assertEqual(tables.tex("- / - / -"), "-- / -- / --")
        self.assertEqual(tables.tex("-"), "--")
        self.assertEqual(tables.tex("dope-2.0.0-auto"), "dope-2.0.0-auto")
        self.assertEqual(tables.tex("label_shuffle"), r"label\_shuffle")
        self.assertEqual(tables.tex("13.0%"), r"13.0\%")

    def test_renderer_numbers(self):
        self.assertEqual(rp.num(-0.0004, 3, signed=True), "0.000")
        self.assertEqual(rp.num(0.06, 3, signed=True), "+0.060")
        self.assertEqual(rp.num(-0.121), r"\ensuremath{-}0.121")
        self.assertEqual(rp.grouped(10000), "10{,}000")
        self.assertEqual(rp.grouped(1817.5), "1{,}817.5")
        self.assertEqual(rp.pct(0.13), r"13.0\%")


class ReadmeCrossCheck(unittest.TestCase):
    def setUp(self):
        primary_methods = load("primary-methods.json")
        keep = {"dope-v3b-auto-v8", "arfpy-arf", "sdv-gaussian-copula", "sdv-tvae", "sdv-ctgan"}
        self.arms = [a for a in primary_methods["arms"] if a["arm_id"] in keep or a["artifact_class"] == "control"]
        self.comps = [
            load(f"evidence/paper/v3-vs-{arm}.json")["comparisons"][0]
            for arm in ("arfpy-arf", "sdv-gaussian-copula", "sdv-tvae", "sdv-ctgan")
        ]

    def test_committed_receipts_render_the_committed_readme_block(self):
        tables.cross_check(readme(), self.arms, self.comps, set())

    def test_every_receipt_row_is_a_readme_row_with_eleven_cells(self):
        arms, comps = tables.parse_readme_block(readme())
        self.assertEqual(set(arms), {a["arm_id"] for a in self.arms})
        self.assertTrue(all(len(cells) == 11 for cells in arms.values()))
        self.assertEqual(len(comps), len(self.comps))

    def test_a_changed_value_is_an_error(self):
        arms = copy.deepcopy(self.arms)
        arms[0]["primary"]["trg"] += 0.01
        with self.assertRaisesRegex(tables.TableError, "differs from its receipt"):
            tables.cross_check(readme(), arms, self.comps, set())

    def test_a_receipt_arm_absent_from_the_readme_is_an_error_unless_extra(self):
        extra = copy.deepcopy(self.arms[0])
        extra["arm_id"] = "dope-2.0.0-auto"
        arms = self.arms + [extra]
        with self.assertRaisesRegex(tables.TableError, "missing from the README"):
            tables.cross_check(readme(), arms, self.comps, set())
        tables.cross_check(readme(), arms, self.comps, {"dope-2.0.0-auto"})

    def test_a_readme_row_without_a_receipt_is_an_error(self):
        with self.assertRaisesRegex(tables.TableError, "in no receipt"):
            tables.cross_check(readme(), self.arms[1:], self.comps, set())

    def test_a_changed_comparison_is_an_error(self):
        comps = copy.deepcopy(self.comps)
        comps[0]["p_holm"] = 0.5
        with self.assertRaisesRegex(tables.TableError, "comparison rows differ"):
            tables.cross_check(readme(), self.arms, comps, set())


class ArmAndComparisonTables(unittest.TestCase):
    def setUp(self):
        self.primary, self.compare = load("primary.json"), load("compare.json")
        self.comp = self.compare["comparisons"][0]

    def test_no_third_party_receipt_renders_no_third_party_rows(self):
        text = tables.render_arm_table(self.primary["arms"], [], self.comp["id"])
        self.assertIn("% third-party baseline rows: none", text)
        rows = [line for line in text.splitlines() if line.endswith(r"\\") and r"\texttt{" in line]
        self.assertEqual(len(rows), 2 * len(self.primary["arms"]))

    def test_third_party_rows_sit_between_source_free_arms_and_controls(self):
        extra = copy.deepcopy(self.primary["arms"][0])
        extra.update(arm_id="sdv-ctgan", artifact_class="third_party")
        text = tables.render_arm_table(self.primary["arms"], [extra], self.comp["id"])
        self.assertIn("% third-party baseline rows: sdv-ctgan", text)
        panel_a = text.split(r"\textbf{(b)}")[0]
        row = panel_a.index(r"\texttt{sdv-ctgan}")
        self.assertLess(panel_a.index(r"\texttt{dse8"), row)
        self.assertLess(row, panel_a.index(r"\texttt{twin\_c}"))

    def test_candidate_row_is_bold(self):
        text = tables.render_arm_table(self.primary["arms"], [], self.comp["id"])
        self.assertIn(r"\textbf{\texttt{" + self.comp["id"] + "}}", text)

    def test_comparison_table_lists_every_condition_and_guard(self):
        text = tables.render_comparison_table(self.comp)
        decision = self.comp["decision"]
        rule_rows = [line for line in text.splitlines() if line.startswith(("condition:", "guard:"))]
        self.assertEqual(len(rule_rows), len(decision["conditions"]) + len(decision["guards"]))
        fails = sum(g["check"] == "fail" for g in decision["guards"])
        self.assertEqual(text.count(r"\textbf{FAIL}"), fails)
        self.assertIn(r"\textbf{" + decision["outcome"] + "}", text)


class DigitLint(unittest.TestCase):
    def errors(self, text: str) -> list[str]:
        return rp.lint_tex("x.tex", text)

    def test_typed_numbers_are_rejected(self):
        self.assertTrue(self.errors("TRG improves by 0.06 on the panel."))
        self.assertTrue(self.errors("We used 44 cells."))
        self.assertTrue(self.errors(r"a threshold of $0.55$ in math"))
        self.assertTrue(self.errors(r"\begin{equation} x \le 2048 \end{equation}"))

    def test_macros_references_names_and_comments_are_allowed(self):
        self.assertFalse(self.errors(r"TRG \VThree{CandTrg} (\S\ref{sec:results}, Eq.~\eqref{eq:trg2})."))
        self.assertFalse(self.errors(r"\citep{xu2019ctgan,dwork2006calibrating}; \label{app:a1}"))
        self.assertFalse(self.errors(r"The DSE8 and DPA1 formats; \path{contracts/evidence-layout-v1.json}."))
        self.assertFalse(self.errors(r"Format~1, \cli{-{}-format 1} and SHA-256. % 64 cells"))
        self.assertFalse(self.errors(r"\includegraphics[width=0.5\linewidth]{figures/hero-v3.pdf} \vspace{2pt}"))

    def test_structural_zero_and_one_in_math_are_allowed(self):
        self.assertFalse(self.errors(r"$f \sim \mathcal N(0, I_r)$ and $v = 1 - R_B$"))
        self.assertFalse(self.errors("\\begin{equation}\n\\frac{1}{|T|}\\sum_{t} x_{1}\n\\end{equation}"))
        self.assertFalse(self.errors(r"$x^{2}$ and $L_{0}$"))
        self.assertTrue(self.errors(r"outside math the digit 1 is typed"))

    def test_error_names_file_and_line(self):
        errors = self.errors("clean line\nnow 12 cells")
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("x.tex:2:"))


def methods_info(primary: dict, files: dict | None = None) -> dict | None:
    """The committed third-party receipts, with any of them replaced or removed (None) via `files`."""
    files = files or {}

    def read(rel: str) -> dict:
        name = rel.split(f"{rp.SCOREBOARD}/", 1)[1]
        return files[name] if name in files else load(name)

    def exists(rel: str) -> bool:
        name = rel.split(f"{rp.SCOREBOARD}/", 1)[1]
        return files[name] is not None if name in files else (SCOREBOARD / name).is_file()
    return methods_mod.load(rp.SCOREBOARD, read, exists, primary, rp.sha256(f"{rp.SCOREBOARD}/primary.json"),
                            rp.check_same_run)


def final_info(primary: dict, files: dict | None = None, shas: dict | None = None) -> dict:
    """The committed final look, with receipts replaced via `files` (by file name) or hashes via `shas`."""
    files, shas = files or {}, shas or {}
    roles = rp.role_ids(primary, load("compare.json")["comparisons"][0])

    def read_json(rel: str) -> dict:
        name = rel.rsplit("/", 1)[1]
        return copy.deepcopy(files[name]) if name in files else rp.read_json(rel)

    def exists(rel: str) -> bool:
        name = rel.rsplit("/", 1)[1]
        return files[name] is not None if name in files else (rp.REPO / rel).is_file()

    def sha256(rel: str) -> str:
        return shas.get(rel.rsplit("/", 1)[1]) or rp.sha256(rel)
    return final_mod.load(rp.SCOREBOARD, read_json, lambda rel: (rp.REPO / rel).read_text(encoding="utf-8"),
                          exists, sha256, primary, roles["Base"], roles["Cand"], rp.check_same_run)


def look(name: str) -> dict:
    return load(f"{final_mod.LOOK}/{name}")


def second_info(files: dict | None = None) -> dict | None:
    """The committed second look (Addendum H part (a)), with receipts replaced via `files` (by file name)."""
    files = files or {}

    def read_json(rel: str) -> dict:
        name = rel.rsplit("/", 1)[1]
        return copy.deepcopy(files[name]) if name in files else rp.read_json(rel)

    def exists(rel: str) -> bool:
        name = rel.rsplit("/", 1)[1]
        return files[name] is not None if name in files else (rp.REPO / rel).is_file()
    return methods_mod.load_second(rp.SCOREBOARD, read_json, exists, rp.sha256, look("primary.json"), rp.check_same_run)


class SecondLook(unittest.TestCase):
    def setUp(self):
        self.arf_low = methods_info(load("primary.json"))["versus"]["arfpy-arf"]["matched"]["ci_low"]

    def test_committed_second_look_binds_its_receipts_and_replicates_the_sdv_gains(self):
        info = second_info()
        self.assertEqual(sorted(info["versus"]), sorted(methods_mod.ROLES))
        self.assertEqual([ok for _, ok in methods_mod.second_claims(info, self.arf_low)], [True, True, True])
        self.assertFalse(methods_mod.second_claims(info, None)[-1][1])
        macros = dict(methods_mod.second_macros(info, rp.ci, rp.sha256, rp.SCORE_SCALE))
        self.assertTrue(macros["SecondLookArfVsPts"].startswith("+17.8"), macros)
        self.assertTrue(macros["SecondLookArfVsPtsMatched"].startswith("+33.7"), macros)
        self.assertEqual(macros["SecondLookCtganVsRule"], "PROMOTE")
        self.assertFalse(any(rp.PRIOR_MACRO.match(n) for n in macros), sorted(macros))

    def test_absent_second_look_is_none_and_its_prose_fails(self):
        self.assertIsNone(second_info({"family-h.json": None}))
        self.assertEqual([ok for _, ok in methods_mod.second_claims(None)], [False])

    def test_a_reused_row_that_differs_from_the_final_look_is_an_error(self):
        prim = rp.read_json(f"{rp.SCOREBOARD}/{methods_mod.SECOND}/primary-h.json")
        next(a for a in prim["arms"] if a["arm_id"] == "copy_a")["primary"]["trg"] += 0.001
        with self.assertRaisesRegex(methods_mod.MethodsError, "field for field"):
            second_info({"primary-h.json": prim})

    def test_an_unbound_or_changed_pair_receipt_is_an_error(self):
        fam = rp.read_json(f"{rp.SCOREBOARD}/{methods_mod.SECOND}/family-h.json")
        fam["comparisons"][0]["receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(methods_mod.MethodsError, "does not bind"):
            second_info({"family-h.json": fam})
        fam = rp.read_json(f"{rp.SCOREBOARD}/{methods_mod.SECOND}/family-h.json")
        fam["comparisons"][0]["replicates"] = not fam["comparisons"][0]["replicates"]
        with self.assertRaisesRegex(methods_mod.MethodsError, "differs from Holm"):
            second_info({"family-h.json": fam})

    def test_a_changed_second_look_outcome_invalidates_the_prose(self):
        info = copy.deepcopy(second_info())
        info["rule"]["sdv-ctgan"] = "HOLD"
        self.assertEqual([ok for _, ok in methods_mod.second_claims(info, self.arf_low)], [True, False, True])


class ProseClaims(unittest.TestCase):
    def setUp(self):
        self.primary, compare = load("primary.json"), load("compare.json")
        self.comp = compare["comparisons"][0]
        self.roles = rp.role_ids(self.primary, self.comp)
        self.methods = methods_info(self.primary)
        self.final = final_info(self.primary)

    def errors(self, comp=None, methods="committed", final=None) -> list[str]:
        info = self.methods if methods == "committed" else methods
        return rp.prose_errors(self.primary, comp or self.comp, self.roles, info, -1.0, final or self.final, second_info())

    def test_committed_receipts_support_every_claim(self):
        self.assertEqual(self.errors(), [])

    def test_a_promote_or_absent_third_party_receipts_invalidate_the_prose(self):
        comp = copy.deepcopy(self.comp)
        comp["decision"]["outcome"] = "PROMOTE"
        self.assertIn("outcome is HOLD", " ".join(self.errors(comp)))
        self.assertTrue(any("third-party receipts are committed" in e for e in self.errors(methods=None)))

    def test_a_passing_guard_invalidates_the_list_of_failures(self):
        comp = copy.deepcopy(self.comp)
        next(g for g in comp["decision"]["guards"] if g["name"] == "mlp_delta")["check"] = "pass"
        self.assertTrue(any("failing guards" in e for e in self.errors(comp)))

    def test_a_changed_third_party_outcome_invalidates_the_prose(self):
        info = copy.deepcopy(self.methods)
        next(c for c in info["comparisons"] if c["id"] == "arfpy-arf")["decision"]["outcome"] = "REJECT"
        self.assertTrue(any("ARF is HOLD" in e for e in self.errors(methods=info)))


class ThirdPartyReceipts(unittest.TestCase):
    def setUp(self):
        self.primary = load("primary.json")

    def test_committed_receipts_are_one_run_with_four_methods(self):
        info = methods_info(self.primary)
        self.assertEqual(sorted(a["arm_id"] for a in info["third"]), sorted(methods_mod.ROLES))
        self.assertTrue(all(c["holm_m"] == len(info["comparisons"]) for c in info["comparisons"]))

    def test_no_receipts_means_no_rows_but_a_partial_set_is_an_error(self):
        none = {n: None for n in (methods_mod.PRIMARY, methods_mod.COMPARE, methods_mod.MANIFEST)}
        self.assertIsNone(methods_info(self.primary, none))
        with self.assertRaisesRegex(methods_mod.MethodsError, "committed together"):
            methods_info(self.primary, {methods_mod.COMPARE: None})

    def test_a_release_arm_that_differs_from_primary_json_is_an_error(self):
        methods = load(methods_mod.PRIMARY)
        methods["arms"][0]["primary"]["trg"] += 0.001
        with self.assertRaisesRegex(methods_mod.MethodsError, "field for field"):
            methods_info(self.primary, {methods_mod.PRIMARY: methods})

    def test_a_paper_pair_must_be_the_exact_sign_reversal(self):
        name = methods_mod.VERSUS.format("arfpy-arf")
        pair = load(name)
        pair["comparisons"][0]["ci_low"] += 1e-6
        with self.assertRaisesRegex(methods_mod.MethodsError, "sign reversal"):
            methods_info(self.primary, {name: pair})
        with self.assertRaisesRegex(methods_mod.MethodsError, "missing"):
            methods_info(self.primary, {name: None})

    def test_another_run_is_an_error(self):
        compare = load(methods_mod.COMPARE)
        compare["inputs"]["schedule_sha256"] = "0" * 64
        with self.assertRaises(rp.RenderError):
            methods_info(self.primary, {methods_mod.COMPARE: compare})
        manifest = load(methods_mod.MANIFEST)
        manifest["sanity"]["committed_primary_sha256"] = "0" * 64
        with self.assertRaisesRegex(methods_mod.MethodsError, "sanity"):
            methods_info(self.primary, {methods_mod.MANIFEST: manifest})

    def test_methods_table_and_dagger_note(self):
        info = methods_info(self.primary)
        text = tables.render_methods_table(info["comparisons"])
        self.assertEqual(text.count(r"\textbf{REJECT}") + text.count(r"\textbf{HOLD}"), len(info["comparisons"]))
        self.assertIn("none (compare-methods.json absent)", tables.render_methods_table([]))
        note = tables.above_budget_note(info["third"])
        self.assertTrue(note.startswith("\u2020 Scored with packets above 2048 B"))
        self.assertEqual(note.count(" files, TRG at 2048 B -1.000"), len(info["third"]))
        self.assertEqual(tables.above_budget_note(self.primary["arms"]), "")


class FinalLook(unittest.TestCase):
    def setUp(self):
        self.primary = load("primary.json")

    def test_committed_look_is_the_frozen_run_and_the_candidate_does_not_ship(self):
        info = final_info(self.primary)
        self.assertEqual((info["outcome_a"], info["outcome_b"]), ("PROMOTE", "HOLD"))
        self.assertLess(info["b"]["delta"], 0)
        self.assertEqual(info["served_build"], "b8a0037c")
        macros = dict(final_mod.macros(info, rp.num, rp.sha256))
        self.assertEqual((macros["FlSeeds"], macros["ReportSeeds"], macros["WideSeeds"]), ("9--12", "5--8", "1--4"))

    def test_holm_and_exact_sign_flip_are_recomputed(self):
        self.assertEqual(final_mod.holm([0.04, 0.01]), [0.04, 0.02])
        self.assertEqual(final_mod.sign_flip_p([0.3, 0.0, 0.2, 0.1], 0.6), 0.125)
        self.assertEqual(final_mod.sign_flip_p([0.3, -0.2], 0.1), 0.5)

    def test_a_missing_or_altered_file_is_an_error(self):
        with self.assertRaisesRegex(final_mod.FinalLookError, "missing"):
            final_info(self.primary, files={"null-a.json": None})
        with self.assertRaisesRegex(final_mod.FinalLookError, "SHA256SUMS"):
            final_info(self.primary, shas={"compare-b.json": "0" * 64})

    def test_receipts_of_another_run_or_build_are_errors(self):
        compare = look("compare-a.json")
        compare["tool"]["git_sha"] = "0" * 40
        with self.assertRaisesRegex(final_mod.FinalLookError, "frozen dope_kpi"):
            final_info(self.primary, files={"compare-a.json": compare})
        family = look("family.json")
        family["comparisons"][0]["p_holm_family"] = family["comparisons"][0]["p_one_sided"]
        with self.assertRaisesRegex(final_mod.FinalLookError, "Holm"):
            final_info(self.primary, files={"family.json": family})
        disclosure = look("family-disclosure-rc-vs-v2.json")
        disclosure["exact_sign_flip_p"] = 0.05
        with self.assertRaisesRegex(final_mod.FinalLookError, "sign-flip"):
            final_info(self.primary, files={"family-disclosure-rc-vs-v2.json": disclosure})
        served = look("served-vs-v2-descriptive.json")
        served["arms"][0]["primary"]["trg"] += 0.001
        with self.assertRaisesRegex(final_mod.FinalLookError, "field for field"):
            final_info(self.primary, files={"served-vs-v2-descriptive.json": served})

    def test_a_candidate_at_or_above_the_released_engine_invalidates_the_prose(self):
        compare = look("compare-b.json")
        compare["comparisons"][0]["delta"] = 0.001
        with self.assertRaisesRegex(final_mod.FinalLookError, "family.json does not repeat"):
            final_info(self.primary, files={"compare-b.json": compare})
        info = copy.deepcopy(final_info(self.primary))
        info["b"]["delta"] = 0.001
        comp = load("compare.json")["comparisons"][0]
        errors = [text for text, ok in final_mod.claims(info, comp) if not ok]
        self.assertIn("the ship rule's second condition alone fails", errors)


class CommittedPaper(unittest.TestCase):
    def test_roles_resolve_to_exactly_one_arm_each(self):
        primary, compare = load("primary.json"), load("compare.json")
        roles = rp.role_ids(primary, compare["comparisons"][0])
        ids = {a["arm_id"] for a in primary["arms"]}
        self.assertTrue(set(roles.values()) <= ids)
        self.assertEqual(len(set(roles.values())), len(roles))

    def test_rust_constants_are_found_and_numeric(self):
        consts, files = rp.rust_constants()
        self.assertEqual(set(consts), set(rp.RUST))
        self.assertTrue(files)
        for name, value in consts.items():
            float(value)

    def test_rendering_matches_the_committed_generated_files(self):
        manifest = json.loads((rp.REPO / rp.OUT["manifest"]).read_text(encoding="utf-8"))
        out, prose = rp.render(manifest["hero"])
        self.assertEqual(prose, [])
        for key, text in out.items():
            self.assertEqual((rp.REPO / rp.OUT[key]).read_text(encoding="utf-8"), text, rp.OUT[key])

    def test_sections_are_digit_free_and_every_macro_is_defined(self):
        macros = (rp.REPO / rp.OUT["macros"]).read_text(encoding="utf-8")
        self.assertEqual(rp.paper_errors(macros), [])

    def test_every_section_file_is_input_by_the_main_file(self):
        main = (rp.REPO / "docs/whitepaper-v3/dope-v3.tex").read_text(encoding="utf-8")
        for path in (rp.REPO / "docs/whitepaper-v3/sections").glob("*.tex"):
            self.assertIn(r"\input{sections/" + path.stem + "}", main, path.name)

    def test_hero_manifest_points_at_the_readme_svg(self):
        manifest = json.loads((rp.REPO / rp.OUT["manifest"]).read_text(encoding="utf-8"))
        hero = manifest["hero"]
        self.assertEqual((hero["source"], hero["output"]), (rp.HERO_SVG, rp.HERO_PDF))
        self.assertIn("rsvg-convert -f pdf", hero["command"])
        scoreboard = load("render-manifest.json")["files"]
        self.assertEqual(scoreboard[rp.HERO_SVG], hero["source_sha256"])


if __name__ == "__main__":
    unittest.main()
