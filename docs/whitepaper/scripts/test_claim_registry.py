"""The abstract's claims compile from the pinned evidence, and each kind of drift is refused."""
import copy
import unittest

import claim_registry as C


class ClaimRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = C.load_inputs()

    def compile(self, **changes):
        names = ("registry", "panel", "headline", "macros", "tex", "readme", "citation", "lineages")
        values = dict(zip(names, copy.deepcopy(self.inputs)))
        values.update(changes)
        return C.compile_claims(*(values[name] for name in names))

    def failures(self, **changes):
        return "\n".join(self.compile(**changes)[1])

    def test_current_source_compiles_every_claim_without_failure(self):
        compiled, failures = self.compile()
        self.assertEqual(failures, [])
        self.assertEqual({claim["id"] for claim in compiled}, {claim["id"] for claim in self.inputs[0]["claims"]})

    def test_a_verdict_that_leaves_its_registered_state_is_refused(self):
        panel = copy.deepcopy(self.inputs[1])
        row = C.contrast_row(next(c for c in self.inputs[0]["claims"] if c["id"] == "dope-vs-arf-default-catboost-4n"),
                             panel)
        row["hl_lo"], row["verdict"] = 0.001, "above"
        self.assertIn("registered state inconclusive but the evidence now says above", self.failures(panel=panel))

    def test_a_panel_verdict_that_contradicts_its_interval_is_refused(self):
        panel = copy.deepcopy(self.inputs[1])
        row = C.contrast_row(next(c for c in self.inputs[0]["claims"] if c["id"] == "dope-vs-predictor-only-catboost-4n"),
                             panel)
        row["verdict"] = "inseparable"
        self.assertIn("disagrees with its interval", self.failures(panel=panel))

    def test_a_macro_that_drifts_from_the_evidence_is_refused(self):
        macros = dict(self.inputs[3], VDopeCbFour="0.950")
        self.assertIn("\\VDopeCbFour is '0.950'", self.failures(macros=macros))

    def test_an_abstract_number_without_a_registered_claim_is_refused(self):
        tex = self.inputs[4].replace("\\end{abstract}", "Against the copula: $\\VDiffGaussCbFour$.\n\\end{abstract}", 1)
        self.assertIn("\\VDiffGaussCbFour belongs to no registered claim", self.failures(tex=tex))

    def test_a_small_cohort_must_be_printed_as_a_separate_block_with_its_count(self):
        tex = self.inputs[4].replace("On a separate $\\VNForestCbFour$-lineage block, DOPE", "DOPE", 1)
        self.assertNotEqual(tex, self.inputs[4])
        self.assertIn("small cohort is printed without its count", self.failures(tex=tex))

    def test_a_full_panel_claim_that_loses_lineages_is_refused(self):
        panel = copy.deepcopy(self.inputs[1])
        row = C.contrast_row(next(c for c in self.inputs[0]["claims"] if c["id"] == "dope-vs-arf-default-catboost-4n"),
                             panel)
        row["n"] = C.STRONGEST_MIN_LINEAGES - 1
        self.assertIn("is registered as full", self.failures(panel=panel))

    def test_an_opened_test_split_or_a_measured_release_score_is_refused(self):
        panel = dict(copy.deepcopy(self.inputs[1]), official_tests_opened=True, mfs_v3={"DOPE": 1.0})
        failures = self.failures(panel=panel)
        self.assertIn("panel official_tests_opened is True", failures)
        self.assertIn("panel mfs_v3 is no longer null", failures)

    def test_forbidden_language_in_readme_prose_or_title_is_refused(self):
        readme = self.inputs[5].replace("small **sampling program**", "**kilobyte-scale** generator", 1)
        self.assertIn("README uses forbidden language", self.failures(readme=readme))
        title = C.title_of(self.inputs[4])
        tex = self.inputs[4].replace("}{" + title + "}}", "}{DOPE: Compressing Data into Transferable Knowledge}}", 1)
        self.assertNotEqual(tex, self.inputs[4])
        self.assertIn("title uses forbidden language", self.failures(tex=tex))


if __name__ == "__main__":
    unittest.main()
