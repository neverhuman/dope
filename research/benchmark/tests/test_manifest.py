import tempfile
import unittest
from pathlib import Path

from research.benchmark.manifest import (check_nonoverlap, fit_projection, prepare, project,
                                         split_official_training, split_rows)


ROOT = Path(__file__).resolve().parents[3]


class ManifestTests(unittest.TestCase):
    def test_duplicate_rows_stay_together(self):
        rows = [[str(i % 10), str(i % 2)] for i in range(40)]
        result = split_rows(rows, 1, "binary")
        self.assertEqual(sum(len(v) for v in result.values()), len(rows))
        seen = {}
        for name, indices in result.items():
            for index in indices:
                key = tuple(rows[index])
                if key in seen:
                    self.assertEqual(seen[key], name)
                seen[key] = name

    def test_official_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            check_nonoverlap({"train": [["1", "0"]], "validation": [["1", "0"]], "test": [["2", "1"]]})

    def test_official_training_split_preserves_groups_and_test(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            train = root / "official-train.csv"
            test = root / "official-test.csv"
            train.write_text("".join(f"{i},{i * 2}\n" for i in range(20)) + "3,6\n")
            test.write_text("".join(f"{i},{i * 2}\n" for i in range(20, 25)))
            entry = {"id": "toy", "panel": "s3_matched_regression", "task": "regression",
                     "target": "c1", "header": False,
                     "license": {"status": "recorded", "spdx": "MIT", "evidence_url": "https://example.org"},
                     "official_split_id": "toy", "official_split_source": "catalog",
                     "source_identity": "toy", "source": "https://example.org",
                     "raw": {"train": str(train), "test": str(test)}}
            result = prepare(entry, root / "prepared")
            self.assertEqual(result["split"]["kind"], "official_test_grouped_training_80_20")
            self.assertEqual(result["split"]["rows"]["test"], 5)
            self.assertEqual(result["split"]["rows"]["train"] +
                             result["split"]["rows"]["validation"], 21)
            self.assertTrue(result["row_group_assignments_sha256"])
            self.assertFalse((root / "prepared/worker/toy/test.csv").exists())
            self.assertFalse((root / "prepared/worker/toy/official-test.csv").exists())
            split = split_official_training([[str(i), str(i * 2)] for i in range(20)] + [["3", "6"]])
            self.assertEqual([name for name, ids in split.items() if 3 in ids],
                             [name for name, ids in split.items() if 20 in ids])

    def test_projection_is_train_fitted(self):
        mapping = fit_projection([["0", "red", "0"], ["10", "blue", "1"]],
                                 ["x", "color", "y"], "y", "binary")
        self.assertEqual(mapping["output_features"], 3)
        self.assertEqual(project([["100", "green", "1"]], mapping), [[1.0, 0.0, 0.0, 1.0]])

    def test_rights_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            source = Path(directory) / "data.csv"
            source.write_text("x,y\n1,0\n2,1\n")
            with self.assertRaisesRegex(ValueError, "license"):
                prepare({"id": "toy", "task": "binary", "license": {"status": "unknown"},
                         "raw": str(source)}, Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()
