"""Contract macros track the TDL contract, and every public surface must agree with its headline."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_macros as C  # noqa: E402


class ContractMacroTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(C.CONTRACT.read_text())
        self.results = json.loads(C.RESULTS.read_text())
        self.headline_macros = C.HEADLINE_MACROS.read_text()
        self.readme = C.README.read_text()

    def failures(self, contract=None, results=None, headline_macros=None, readme=None):
        return C.surface_failures(contract or self.contract, results or self.results,
                                  self.headline_macros if headline_macros is None else headline_macros,
                                  self.readme if readme is None else readme)

    def test_committed_file_and_surfaces_match_the_contract(self):
        self.assertEqual(C.OUTPUT.read_text(), C.render(self.contract))
        self.assertEqual(self.failures(), [])
        values = C.macro_values(C.OUTPUT.read_text())
        point = C.headline_point(self.contract)
        self.assertEqual(values["HeadlinePoint"], point["id"])
        self.assertEqual(values["TauT"], f"{point['transfer_threshold']:.2f}")
        self.assertEqual(values["TauD"], f"{point['driver_threshold']:.2f}")

    def test_results_threshold_drift_is_refused(self):
        results = copy.deepcopy(self.results)
        results["thresholds"]["transfer"] = 0.85
        self.assertTrue(any("transfer threshold" in f for f in self.failures(results=results)))
        results = copy.deepcopy(self.results)
        results["thresholds"]["drivers"] = 0.9
        self.assertTrue(any("driver threshold" in f for f in self.failures(results=results)))

    def test_rendered_macro_drift_is_refused(self):
        drifted = self.headline_macros.replace("\\newcommand{\\TdlTransferThreshold}{0.90}",
                                               "\\newcommand{\\TdlTransferThreshold}{0.95}")
        self.assertNotEqual(drifted, self.headline_macros)
        self.assertTrue(any("TdlTransferThreshold" in f for f in self.failures(headline_macros=drifted)))

    def test_contract_change_without_readme_update_is_refused(self):
        contract = copy.deepcopy(self.contract)
        for point in contract["operating_points"]:
            point["headline"] = point["id"] == "T95/D95"
        contract["headline_operating_point"] = "T95/D95"
        failures = self.failures(contract=contract)
        self.assertTrue(any("KiB@T95/D95" in f for f in failures))
        self.assertTrue(any("headline-results.json transfer threshold" in f for f in failures))

    def test_readme_wording_drift_is_refused(self):
        readme = self.readme.replace("nine-tenths", "most")
        self.assertTrue(any("T90" in f for f in self.failures(readme=readme)))

    def test_exactly_one_headline_operating_point(self):
        contract = copy.deepcopy(self.contract)
        for point in contract["operating_points"]:
            point["headline"] = True
        with self.assertRaises(C.ContractMismatch):
            C.headline_point(contract)


if __name__ == "__main__":
    unittest.main()
