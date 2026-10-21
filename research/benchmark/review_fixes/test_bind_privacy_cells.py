"""Privacy scalar binder on public toy fixtures; no CSV or partition is read."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes import bind_privacy_cells as privacy

ROOT = Path(__file__).resolve().parents[3]


def raw_privacy(status="ok", **overrides):
    cell = {
        "configuration": "features12_steps2048", "dataset": "ab" * 8, "formal_dp": False,
        "hipaa_deidentification": False, "method": "DOPE", "official_tests_opened": False,
        "reason": None, "sample_seed": 101, "size": 1, "status": status,
        "csv": "/scratch/toy/sample.csv",
    }
    if status == "ok":
        cell.update({
            "c2st_catboost_auc": 0.6, "c2st_rows_per_class": 1, "c2st_status": "ok",
            "dcr_fit_median": 0.1, "dcr_validation_median": 0.2, "distance_mia_auc": 0.5,
            "distance_mia_status": "ok", "nndr_fit_median": 0.8,
            "rows": {"fit": 4, "synthetic": 4, "validation": 1}, "synthetic_sha256": "cd" * 32,
        })
    else:
        cell.update(reason="sample_csv_unresolved", csv=None)
    cell.update(overrides)
    return cell


class PrivacyBinderTests(unittest.TestCase):
    def setUp(self):
        target = ROOT / "target/review-binder-tests"
        target.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "privacy"
        self.write(raw_privacy())
        self.write(raw_privacy("unavailable", method="ForestDiffusion/Forest-Flow",
                               configuration="native_selected", size=4))

    def write(self, cell, name=None):
        path = self.root / (name or privacy.expected_path({**cell}))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cell))
        return path

    def refuse(self, reason):
        with self.assertRaisesRegex(ValueError, reason):
            privacy.load_cell_dir(self.root)

    def test_public_cells_drop_the_scratch_path(self):
        cells = privacy.load_cell_dir(self.root)
        self.assertEqual(len(cells), 2)
        text = privacy.canonical_jsonl(cells)
        self.assertNotIn("/scratch", text)
        self.assertNotIn('"csv"', text)

    def test_open_or_formal_privacy_flags_are_refused(self):
        for flag in ("official_tests_opened", "formal_dp", "hipaa_deidentification"):
            for value in (True, 1, 0, None, "false"):
                with self.subTest(flag=flag, value=value):
                    path = self.write(raw_privacy(**{flag: value}))
                    self.refuse("flag is not closed")
                    path.write_text(json.dumps(raw_privacy()))

    def test_unknown_method_configuration_or_reason_is_refused(self):
        self.write(raw_privacy(method="TabSyn"), "TabSyn/features12_steps2048/" + "ab" * 8 + "/size1-seed211.json")
        self.refuse("outside the published set")

    def test_unknown_unavailable_reason_and_extra_key_are_refused(self):
        path = self.write(raw_privacy("unavailable", reason="manual_drop"))
        self.refuse("reason")
        path.write_text(json.dumps(raw_privacy(source="density.json")))
        self.refuse("public scalar set")
        path.write_text(json.dumps(raw_privacy(csv="relative/sample.csv")))
        self.refuse("scratch path")

    def test_symlink_staging_and_foreign_files_are_refused(self):
        outside = Path(self.tmp.name) / "outside.json"
        outside.write_text(json.dumps(raw_privacy(sample_seed=211)))
        link = self.root / privacy.expected_path(raw_privacy(sample_seed=211))
        link.symlink_to(outside)
        self.refuse("symlink")
        link.unlink()
        (self.root / "DOPE" / "features12_steps2048" / "linked").symlink_to(self.root, target_is_directory=True)
        self.refuse("symlink")
        (self.root / "DOPE" / "features12_steps2048" / "linked").unlink()
        tmp = self.root / "DOPE" / "cell.json.tmp"
        tmp.write_text("{}")
        self.refuse("staging file")
        tmp.unlink()
        (self.root / "DOPE" / "sample.csv").write_text("a,b\n")
        self.refuse("non-json")

    def test_misfiled_and_duplicate_identities_are_refused(self):
        self.write(raw_privacy(), "DOPE/features12_steps2048/" + "ab" * 8 + "/size1-seed307.json")
        self.refuse("filed under another identity")
        cell = privacy.public_cell(raw_privacy())
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            privacy.assert_unique_cells([cell, dict(cell)])

    def test_ledger_is_written_only_when_the_panel_matches(self):
        cells = privacy.load_cell_dir(self.root)
        panel = Path(self.tmp.name) / "panel.json"
        panel.write_text('{"predeclaration_sha256": "x"}\n')
        ledger = Path(self.tmp.name) / "scalar-cells.jsonl"
        with patch.object(privacy, "replay", return_value="different\n"):
            with self.assertRaisesRegex(ValueError, "does not match"):
                privacy.write_ledger(cells, ledger, panel)
        self.assertFalse(ledger.exists())
        with patch.object(privacy, "replay", return_value=panel.read_text()):
            digest = privacy.write_ledger(cells, ledger, panel)
        meta = json.loads(ledger.with_suffix(".meta.json").read_text())
        self.assertEqual((meta["sha256"], meta["cell_count"]), (digest, 2))
        self.assertEqual(digest, hashlib.sha256(ledger.read_bytes()).hexdigest())
        self.assertEqual(meta["reference_rows"]["fit_to_holdout_median"], 4.0)

    def test_committed_ledger_is_public_and_matches_its_meta(self):
        raw = privacy.LEDGER.read_bytes()
        cells = privacy.decode_jsonl(raw)
        privacy.assert_unique_cells(cells)
        meta = json.loads(privacy.LEDGER.with_suffix(".meta.json").read_text())
        self.assertEqual(meta["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(meta["panel_sha256"], hashlib.sha256(privacy.PANEL.read_bytes()).hexdigest())
        self.assertEqual(len(cells), json.loads(privacy.PANEL.read_text())["cells"])
        self.assertNotIn(b"/mnt/", raw)
        self.assertNotIn(b"/home/", raw)


if __name__ == "__main__":
    unittest.main()
