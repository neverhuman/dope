import tempfile
import unittest
from pathlib import Path

from research.benchmark.manifest import check_nonoverlap, fit_projection, prepare, project, split_rows


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
