import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from research.benchmark import public_sources
from research.benchmark.manifest import prepare
from research.benchmark.score import sha256


ROOT = Path(__file__).resolve().parents[3]


class PublicSourcesTests(unittest.TestCase):
    def test_adult_official_test_is_evaluator_only_after_overlap_exclusion(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            archive = root / "adult.zip"

            def row(index, dotted=False):
                cells = [str(20 + index), "Private", "100", "HS-grad", "9", "Never-married",
                         "Sales", "Not-in-family", "White", "Female", "0", "0", "40",
                         "United-States", ">50K" if index % 2 else "<=50K"]
                if dotted:
                    cells[-1] += "."
                return ", ".join(cells)

            with zipfile.ZipFile(archive, "w") as source:
                source.writestr("adult.data", "\n".join(row(i) for i in range(30)) + "\n")
                source.writestr("adult.test", "| header\n" +
                                "\n".join([row(5, True), *(row(i, True)
                                                   for i in range(30, 40))]) + "\n")
            train, test = root / "official-train.csv", root / "official-test.csv"
            source = {**public_sources.SOURCES["Adult"], "sha256": sha256(archive)}
            with patch.dict(public_sources.SOURCES, {"Adult": source}):
                entry = public_sources.transform("Adult", archive, train, test)
            self.assertEqual(entry["excluded_official_overlap_rows"], 1)
            self.assertEqual(entry["transformed_rows"], {"train": 29, "test": 11})
            with train.open(newline="") as stream:
                train_rows = list(csv.reader(stream))[1:]
            with test.open(newline="") as stream:
                test_rows = list(csv.reader(stream))[1:]
            self.assertFalse(set(map(tuple, train_rows)) & set(map(tuple, test_rows)))
            self.assertTrue(all(value[-1] in ("<=50K", ">50K") for value in test_rows))
            manifest = prepare(entry, root / "prepared")
            self.assertEqual(manifest["split"]["kind"],
                             "official_test_grouped_training_80_20")
            self.assertEqual(manifest["split"]["rows"]["test"], 11)
            self.assertEqual(manifest["split"]["rows"]["train"] +
                             manifest["split"]["rows"]["validation"], 29)
            self.assertFalse((root / "prepared/worker/Adult/test.csv").exists())
            self.assertTrue((root / "prepared/evaluator/Adult/test.csv").exists())


if __name__ == "__main__":
    unittest.main()
