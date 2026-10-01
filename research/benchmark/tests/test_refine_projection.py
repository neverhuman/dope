import json
import unittest

from research.benchmark.refine_projection import packed_projection, unpacked_projection


class ProjectionPackingTests(unittest.TestCase):
    def test_roundtrip_removes_only_redundant_column_names(self):
        projection = {"features": [{"name": "x", "kind": "numeric", "min": 1.0,
                                     "max": 9.0}], "target": "y",
                      "input_columns": ["x", "y"], "task": "regression"}
        encoded = packed_projection(projection)
        self.assertNotIn(b"input_columns", encoded)
        self.assertEqual(unpacked_projection(encoded), projection)
        self.assertEqual(encoded, packed_projection(projection))

    def test_inconsistent_or_duplicate_columns_fail_closed(self):
        projection = {"features": [{"name": "x"}], "target": "y",
                      "input_columns": ["wrong", "y"]}
        with self.assertRaises(ValueError):
            packed_projection(projection)
        projection["input_columns"] = ["x", "x"]
        projection["target"] = "x"
        with self.assertRaises(ValueError):
            packed_projection(projection)
        with self.assertRaises(ValueError):
            unpacked_projection(json.dumps({**projection,
                "input_columns": ["x", "x"]}).encode())


if __name__ == "__main__":
    unittest.main()
