"""The vnext draft check is unchanged, and measured mode renders only a verified claim graph.

Every claim graph here is G1's synthetic fixture or a mutation of it. Synthetic
claims prove the interface only; none is qualified or admitted to a campaign.
"""
import contextlib
import copy
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import vnext_paper as V

claims, canonical_json, assert_public_provenance, LineageError = V._transfer()
from research.transfer.evidence import parse_json  # noqa: E402  (repo root is on sys.path after _transfer)

COMPARISONS = dict(V.COMPARISONS)
SENTINEL = "zq-sentinel-header-extremum"
CAMPAIGN = "synthetic-interface"


def mutated(comparison, receipt_change=None, claim_change=None, repin=True):
    """G1's fixture with an optional change. The new pin is computed here from the new bytes."""
    fixture = claims.synthetic_fixture(comparison)
    document = parse_json(fixture["document"])
    ((receipt_id, raw),) = fixture["receipts"].items()
    receipt = parse_json(raw)
    if receipt_change is not None:
        receipt_change(receipt)
        raw = canonical_json(receipt)
    if repin:
        document["claims"][0]["references"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    if claim_change is not None:
        claim_change(document)
    return canonical_json(document), {receipt_id: raw}, fixture["expected"]


def verified(comparison="raw_gap_superiority"):
    fixture = claims.synthetic_fixture(comparison)
    return claims.verify_claim_graph(fixture["document"], fixture["receipts"], fixture["expected"])


def redigest(projection):
    body = {key: value for key, value in projection.items() if key != "projection_sha256"}
    projection["projection_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    return projection


class DraftModeTests(unittest.TestCase):
    def test_the_draft_check_and_its_output_are_unchanged(self):
        V.check()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(V.main([]), 0)
        expected = ["vnext draft check passed", "venue undecided"]
        expected += [f"{claim.claim_id} {claim.kind} unbound" for claim in V.CLAIMS]
        self.assertEqual(stdout.getvalue().splitlines(), expected)

    def test_measured_options_without_measured_mode_are_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            V.main(["--campaign-id", CAMPAIGN])
        self.assertEqual(raised.exception.code, 2)

    def test_the_ladder_matches_the_claim_interface(self):
        render = V.render_measured(verified(), CAMPAIGN)
        self.assertEqual([row["claim_id"] for row in render.receipt["ladder"]],
                         [claim.claim_id for claim in V.CLAIMS])


class MeasuredRenderTests(unittest.TestCase):
    def test_each_synthetic_comparison_renders_as_an_unqualified_value(self):
        for comparison, label in COMPARISONS.items():
            with self.subTest(comparison=comparison):
                render = V.render_measured(verified(comparison), CAMPAIGN)
                projection = parse_json(render.projection)
                receipt = render.receipt
                self.assertIs(assert_public_provenance(receipt), receipt)
                self.assertEqual(receipt["projection_sha256"], projection["projection_sha256"])
                self.assertEqual(receipt["claims_json_sha256"], hashlib.sha256(render.projection).hexdigest())
                self.assertEqual(receipt["claims_tex_sha256"],
                                 hashlib.sha256(render.macros.encode("utf-8")).hexdigest())
                self.assertEqual((receipt["bound_claims"], receipt["qualified_claims"]), (1, 0))
                self.assertIs(receipt["campaign_admitted"], False)
                self.assertIs(receipt["n21_admitted"], False)
                self.assertEqual(receipt["figure_inputs"]["state"], "unbound")
                bound = [row for row in receipt["ladder"] if row["state"] == "bound"]
                self.assertEqual(bound, [{"claim_id": "rate-risk-frontier", "state": "bound",
                                          "qualification": "synthetic"}])
                definitions = [line for line in render.macros.splitlines() if line.startswith("\\@namedef")]
                self.assertEqual(len(definitions), 1)
                self.assertTrue(definitions[0].startswith(
                    "\\@namedef{vnext@claim@rate-risk-frontier}{\\VNextUnqualified{synthetic}{mse in original\\_mse; "
                    "2 artifacts, 3 fits, 5 samples; "))
                self.assertIn(f"{label} supported", definitions[0])
                self.assertNotIn("original_mse", render.macros)
                self.assertIn("\\providecommand{\\TBD}[1]{[unbound: #1]}", render.macros)

    def test_the_three_endpoints_stay_separate(self):
        definition = V.render_measured(verified("equivalence"), CAMPAIGN).macros
        self.assertIn("raw-gap superiority not supported, interval [-0.25, 0.25], upper bound", definition)
        self.assertIn("adjusted-Z noninferiority unavailable", definition)
        self.assertIn("equivalence supported, interval [-0.25, 0.25], equivalence margin 0.5, both bounds", definition)
        definition = V.render_measured(verified("adjusted_z_noninferiority"), CAMPAIGN).macros
        self.assertIn("adjusted-Z noninferiority supported, interval [-0.5, -0.25], "
                      "margin relative 0, scaled 0, raw unavailable, upper bound", definition)
        self.assertIn("raw-gap superiority unavailable", definition)

    def test_rendered_numbers_come_from_the_bound_receipt(self):
        def change(receipt):
            receipt["lower"], receipt["upper"] = -0.875, -0.125

        render = V.verify_and_render(*mutated("raw_gap_superiority", change), CAMPAIGN)
        self.assertIn("raw-gap superiority supported, interval [-0.875, -0.125]", render.macros)
        self.assertNotIn("-0.25]", render.macros)

    def test_verifier_refusals_never_reach_a_render(self):
        def receipt(key, value):
            return lambda row: row.__setitem__(key, value)

        def claim(key, value):
            return lambda document: document["claims"][0].__setitem__(key, value)

        cases = {
            "stale pin": (("raw_gap_superiority", receipt("lower", -0.75), None, False),
                          {"pin_mismatch", "invalid_receipt"}),
            "mixed cohort": (("raw_gap_superiority", receipt("cohort_sha256", hashlib.sha256(b"other").hexdigest())),
                             {"identity_mismatch", "mixed_cohort"}),
            "label mismatch": (("raw_gap_superiority", receipt("comparison_label", "equivalence")),
                               {"mixed_label", "label_mismatch"}),
            "missing claim": (("raw_gap_superiority", None, lambda document: document.__setitem__("claims", [])),
                              {"missing_claim"}),
            "vendor not executed": (("raw_gap_superiority", receipt("vendor_executed", False)),
                                    {"vendor_not_executed"}),
            "unverified interval": (("raw_gap_superiority", receipt("lower", None)), {"unverified_bound"}),
            "sample-fit conflation": (("raw_gap_superiority", None, claim("fit_count", 5)),
                                      {"sample_fit_conflation"}),
            "self-declared verification": (("raw_gap_superiority", None, claim("verified", True)),
                                           {"self_declared_verification"}),
        }
        for name, (arguments, codes) in cases.items():
            with self.subTest(name), self.assertRaises(claims.ClaimError) as raised:
                V.verify_and_render(*mutated(*arguments), CAMPAIGN)
            self.assertIn(str(raised.exception), codes)

    def test_only_the_verified_type_is_accepted(self):
        projection = claims.public_projection(verified())
        forged = object.__new__(claims.VerifiedClaimGraph)
        for value in (projection, {"verified": True, "note": SENTINEL}, True, 0.5, forged):
            with self.subTest(type(value).__name__), self.assertRaises(V.PaperDraftError) as raised:
                V.render_measured(value, CAMPAIGN)
            self.assertNotIn(SENTINEL, str(raised.exception))

    def test_a_tampered_projection_is_refused(self):
        original = claims.public_projection(verified())

        def claim_change(key, value, digest=True):
            def change(projection):
                projection["claims"][0][key] = value
                return redigest(projection) if digest else projection
            return change

        def decision_change(key, value, digest=True):
            def change(projection):
                projection["claims"][0]["raw_gap_superiority"][key] = value
                return redigest(projection) if digest else projection
            return change

        cases = {
            "qualified": claim_change("qualified", True),
            "campaign admitted": claim_change("campaign_admitted", True),
            "n21 admitted": claim_change("n21_admitted", True),
            "campaign qualification": claim_change("qualification", "campaign"),
            "vendor not executed": claim_change("vendor_executed", False),
            "bool count": claim_change("fit_count", True),
            "unknown claim": claim_change("claim_id", "headline-retention"),
            "bare pin list": claim_change("receipts", [hashlib.sha256(b"pin").hexdigest()]),
            "unsorted pins": claim_change("receipts", [{"sha256": hashlib.sha256(name).hexdigest()}
                                                       for name in (b"b", b"a", b"c")]),
            "pin with extra key": claim_change("receipts", [{"sha256": hashlib.sha256(b"a").hexdigest(),
                                                            "note": SENTINEL}]),
            "header leak": claim_change("headers", [SENTINEL]),
            "per-query leak": claim_change("per_query_loss", [0.5, SENTINEL]),
            "stale digest": claim_change("artifact_count", 9, digest=False),
            "supported without interval": decision_change("lower", None),
            "nonfinite bound": decision_change("upper", float("inf"), digest=False),
            "qualified decision": decision_change("qualified", True),
        }
        for name, change in cases.items():
            tampered = change(copy.deepcopy(original))
            with self.subTest(name), mock.patch.object(claims, "public_projection", return_value=tampered), \
                    self.assertRaises(V.PaperDraftError) as raised:
                V.render_measured(verified(), CAMPAIGN)
            self.assertNotIn(SENTINEL, str(raised.exception))

    def test_campaign_id_is_a_lowercase_slug(self):
        for campaign in ("", "../target", "Synthetic", "a/b", "-x"):
            with self.subTest(campaign), self.assertRaises(V.PaperDraftError):
                V.render_measured(verified(), campaign)

    def test_tex_escaping_and_control_characters(self):
        self.assertEqual(V.tex_escape("a\\b{c}$d&e#f%g_h^i~j"),
                         "a\\textbackslash{}b\\{c\\}\\$d\\&e\\#f\\%g\\_h\\textasciicircum{}i\\textasciitilde{}j")
        with self.assertRaises(V.PaperDraftError) as raised:
            V.tex_escape(SENTINEL + "\n")
        self.assertNotIn(SENTINEL, str(raised.exception))
        for value in (True, float("nan"), float("inf"), None, "1"):
            with self.subTest(repr(value)), self.assertRaises(V.PaperDraftError):
                V._number(value)


class MeasuredCommandTests(unittest.TestCase):
    def run_main(self, folder, document, receipts, expected, output):
        paths = []
        for receipt_id, raw in receipts.items():
            path = folder / f"{receipt_id}.json"
            path.write_bytes(raw)
            paths += ["--receipt", f"{receipt_id}={path}"]
        (folder / "graph.json").write_bytes(document)
        (folder / "expected.json").write_bytes(canonical_json(expected))
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = V.main(["--measured", str(folder / "graph.json"), *paths, "--expected",
                           str(folder / "expected.json"), "--campaign-id", CAMPAIGN, "--output-dir", str(output)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_the_command_writes_only_its_three_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = folder / "out"
            code, stdout, _ = self.run_main(folder, *mutated("raw_gap_superiority"), output)
            self.assertEqual(code, 0)
            self.assertIn("no claim is qualified or campaign-admitted", stdout)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["claims.json", "claims.tex", "render.json"])
            receipt = parse_json((output / "render.json").read_bytes().rstrip(b"\n"))
            self.assertEqual(receipt["claims_json_sha256"],
                             hashlib.sha256((output / "claims.json").read_bytes()).hexdigest())
            self.assertEqual(receipt["claims_tex_sha256"],
                             hashlib.sha256((output / "claims.tex").read_bytes()).hexdigest())
        V.check()

    def test_a_refused_graph_writes_nothing_and_prints_only_the_code(self):
        def change(receipt):
            receipt["vendor_executed"] = False

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = folder / "out"
            code, _, stderr = self.run_main(folder, *mutated("raw_gap_superiority", change), output)
            self.assertEqual(code, 1)
            self.assertEqual(stderr, "vnext measured mode refused: vendor_not_executed\n")
            self.assertFalse(output.exists())

    def test_an_oversized_document_is_refused_by_its_budget(self):
        document, receipts, expected = mutated("raw_gap_superiority")
        oversized = document + b" " * claims.MAX_DOCUMENT_BYTES
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            code, _, stderr = self.run_main(folder, oversized, receipts, expected, folder / "out")
            self.assertEqual(code, 1)
            self.assertEqual(stderr, "vnext measured mode refused: document_budget_exceeded\n")
            self.assertEqual(len(V._bounded_read(folder / "graph.json", claims.MAX_DOCUMENT_BYTES)),
                             claims.MAX_DOCUMENT_BYTES + 1)


if __name__ == "__main__":
    unittest.main()
