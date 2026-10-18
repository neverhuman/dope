"""Semantic checks for the protocol-only Figure 1 renderer."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_one as F


class FigureOneTests(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads(F.LEDGER.read_text())

    def test_committed_ledger_is_protocol_only_and_complete(self):
        F.validate(self.ledger)
        self.assertEqual(self.ledger["points"], [])
        self.assertIn("failed", self.ledger["retained_states"])
        self.assertEqual(self.ledger["compressed_data_controls"], "nonrelease_only")
        self.assertIn("actual_complete_recipient_bytes", self.ledger["contract"]["x"])
        self.assertEqual(self.ledger["contract"]["thresholds"], {"transfer": .9, "drivers": .8})

    def test_illustrative_fixture_is_never_publishable(self):
        fixture = copy.deepcopy(self.ledger)
        fixture["mode"] = "illustrative_fixture"
        fixture["points"] = [{"bytes": 100, "coverage": .7, "paired": True,
                              "state": "eligible", "source_cluster": "ILLUSTRATIVE ONLY"}]
        F.validate(fixture, allow_fixture=True)
        with self.assertRaisesRegex(ValueError, "measured or fixture"):
            F.validate(fixture)

    def test_unpaired_and_unknown_states_are_rejected(self):
        fixture = copy.deepcopy(self.ledger)
        fixture["mode"] = "illustrative_fixture"
        fixture["points"] = [{"bytes": 100, "coverage": .7, "paired": False,
                              "state": "eligible", "source_cluster": "ILLUSTRATIVE ONLY"}]
        with self.assertRaisesRegex(ValueError, "unpaired"):
            F.validate(fixture, allow_fixture=True)
        fixture["points"][0]["state"] = "winner"
        with self.assertRaisesRegex(ValueError, "unknown retained state"):
            F.validate(fixture, allow_fixture=True)

    def test_rendered_schematic_has_labels_and_no_points(self):
        figure = F.draw()
        try:
            labels = [item.get_text() for axis in figure.axes
                      for item in axis.texts + axis.get_xticklabels() + axis.get_yticklabels()]
            self.assertIn("PROTOCOL SCHEMATIC", labels)
            self.assertIn("No measured points admitted; E05 confirmation pending", labels)
            self.assertTrue(any("Cross-model feature-driver preservation: not yet measured" in lbl for lbl in labels))
            self.assertEqual(len(figure.axes[0].lines), 0)
        finally:
            F.plt.close(figure)

    def test_q01_fig_provenance_audit_and_deterministic_check(self):
        prov = self.ledger.get("provenance_audit", {})
        self.assertEqual(prov.get("matched_lineages"), 97)
        self.assertEqual(prov.get("fit_seed"), 11)
        self.assertEqual(prov.get("sample_seeds"), [101, 211, 307])
        self.assertEqual(prov.get("synthetic_context"), "4n")
        self.assertIn("0.975", prov.get("driver_proxy_disclosure", ""))
        self.assertIn("0.923", prov.get("driver_proxy_disclosure", ""))
        bad = copy.deepcopy(self.ledger)
        bad["provenance_audit"]["matched_lineages"] = 100
        with self.assertRaisesRegex(ValueError, "provenance_audit"):
            F.validate(bad)


if __name__ == "__main__":
    unittest.main()
