"""Gate checks for the one sealed run. These tests do not open a test file."""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from research.benchmark.review_fixes.sealed_once import (
    FREEZE_BLOB,
    FREEZE_SHA256,
    _control_identity,
    _csv_shape,
    _gate,
    _missing_published_grid,
    _published_problem,
    _shape_problem,
    cell_path,
    read_checkpoint,
    registry_mode,
    write_json,
)


class SealedGateTests(unittest.TestCase):
    def test_a_missing_hash_is_not_a_ready_sample(self):
        sample = {
            "method": "DOPE",
            "configuration": "features12_steps2048",
            "dataset": "aa",
            "size": 1,
            "sample_seed": 101,
            "published_status": "ok",
            "csv": "/tmp/sample.csv",
            "expected_sha256": None,
        }
        self.assertIn("no sha pin", _published_problem(sample))
        absent = {**sample, "published_status": "fit_unavailable", "csv": None}
        self.assertIsNone(_published_problem(absent))

    def test_missing_grid_cell_is_reported(self):
        manifest = [{
            "method": "DOPE",
            "configuration": "features12_steps2048",
            "dataset": "aa",
            "size": 1,
            "sample_seed": 101,
        }]
        missing = _missing_published_grid(manifest, ["aa"])
        self.assertIn("GaussianCopula aa size 1 seed 101", missing)
        self.assertGreater(len(missing), 1)

    def test_wrong_control_identity_is_rejected(self):
        cell = {
            "dataset": "bb",
            "kind": "predictor_only",
            "sample_seed": 101,
            "size": 1,
            "fit_seed": 11,
            "split_seed": None,
            "status": "ok",
        }
        self.assertEqual(_control_identity(cell, "aa", "predictor_only_fit_seed_11", 1, 101), "control_identity")
        self.assertIsNone(_control_identity(
            {**cell, "dataset": "aa"}, "aa", "predictor_only_fit_seed_11", 1, 101,
        ))
        split = {**cell, "dataset": "aa", "split_seed": 2027}
        self.assertEqual(_control_identity(split, "aa", "predictor_only_fit_seed_11", 1, 101), "control_identity")

    def test_a_second_output_directory_is_refused(self):
        args = argparse.Namespace(
            out=Path("/tmp/dope-rf-not-the-registry"),
            predeclare=Path("predeclare.json"),
            expect_git_blob="deadbeef",
            expect_sha256="deadbeef",
        )
        with self.assertRaises(SystemExit) as caught:
            _gate(args, [])
        self.assertIn("sealed output must be", str(caught.exception))

    def test_an_existing_registry_refuses_a_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "started.json").write_text("{}\n")
            args = argparse.Namespace(
                out=registry,
                predeclare=Path("predeclare.json"),
                expect_git_blob="deadbeef",
                expect_sha256="deadbeef",
            )
            with self.assertRaises(SystemExit) as caught:
                _gate(args, [{"ready": True}], registry=registry)
            self.assertIn("already exists", str(caught.exception))
            self.assertFalse((registry / "panel.json").exists())

    def test_a_wrong_freeze_pin_does_not_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            args = argparse.Namespace(
                out=registry,
                host="xbabe3",
                predeclare=Path("predeclare.json"),
                expect_git_blob="deadbeef",
                expect_sha256="deadbeef",
            )
            old = os.environ.get("DOPE_RF_UNSEAL")
            os.environ["DOPE_RF_UNSEAL"] = "1"
            try:
                with self.assertRaises(SystemExit) as caught:
                    _gate(args, [{"ready": True}], registry=registry)
                self.assertIn("original predeclare freeze", str(caught.exception))
                self.assertFalse((registry / "started.json").exists())
            finally:
                if old is None:
                    os.environ.pop("DOPE_RF_UNSEAL", None)
                else:
                    os.environ["DOPE_RF_UNSEAL"] = old

    def test_started_marker_refuses_reopen_and_keeps_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            self.assertEqual(registry_mode(registry, FREEZE_SHA256, FREEZE_BLOB), "fresh")
            started = {
                "predeclaration_sha256": FREEZE_SHA256,
                "predeclaration_git_blob": FREEZE_BLOB,
            }
            marker = registry / "started.json"
            marker.write_text(json.dumps(started) + "\n")
            before = marker.read_text()
            path = cell_path(registry, "aa", "DOPE", "features12_steps2048", 1, 101)
            write_json(path, {"dataset": "aa", "status": "ok"})
            with self.assertRaises(SystemExit) as caught:
                registry_mode(registry, FREEZE_SHA256, FREEZE_BLOB)
            self.assertIn("refusing to reopen", str(caught.exception))
            self.assertEqual(marker.read_text(), before)
            self.assertEqual(read_checkpoint(path)["status"], "ok")
            with self.assertRaises(SystemExit):
                registry_mode(registry, "0" * 64, FREEZE_BLOB)
            self.assertEqual(marker.read_text(), before)
            (registry / "panel.json").write_text("{}\n")
            with self.assertRaises(SystemExit) as caught:
                registry_mode(registry, FREEZE_SHA256, FREEZE_BLOB)
            self.assertIn("already exists", str(caught.exception))
            self.assertEqual(marker.read_text(), before)

    def test_shape_problem_rejects_a_short_sample(self):
        self.assertEqual(_shape_problem(4, 3, 4, 3, 1), None)
        self.assertEqual(_shape_problem(4, 3, 4, 3, 4), "row_count")
        self.assertEqual(_shape_problem(8, 2, 4, 3, 4), "width")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("1,2\n3,4\n")
            self.assertEqual(_csv_shape(path), (2, 2))


if __name__ == "__main__":
    unittest.main()
