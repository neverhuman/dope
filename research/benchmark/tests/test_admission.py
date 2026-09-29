import json
import tempfile
import unittest
from pathlib import Path

from research.benchmark.admission import assess


ROOT = Path(__file__).resolve().parents[3]


class AdmissionTests(unittest.TestCase):
    def test_missing_final_locks_keep_test_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            result = assess(ROOT, Path(directory))
            self.assertFalse(result["admitted"])
            self.assertIn("budget.lock.json:missing_or_invalid", result["blockers"])

    def test_incomplete_method_lock_keeps_test_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            lock_root = Path(directory)
            (lock_root / "methods.lock.json").write_text(json.dumps({
                "complete": False, "frozen_for_final_evaluation": False,
                "methods": {"dope": {"status": "locked"}}}))
            result = assess(ROOT, lock_root)
            self.assertIn("methods.lock.json:not_frozen", result["blockers"])
            self.assertIn("methods.lock.json:source_or_config_gap", result["blockers"])


if __name__ == "__main__":
    unittest.main()
