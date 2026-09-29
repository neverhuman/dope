import unittest

from research.benchmark.select_extension import select


class ExtensionSelectionTests(unittest.TestCase):
    def test_rights_deduplication_hash_order_and_shortfall(self):
        def row(task_id, identity, rights=True):
            return {"official_task_id": task_id, "task": "binary", "suite": "OpenML-CC18",
                    "rows": 1000, "features": 12, "target": "label",
                    "license": {"status": "recorded" if rights else "unknown",
                                "spdx": "CC0-1.0", "evidence_url": "https://example.com"},
                    "source_identity": identity, "source_row_hash": "same"}
        result = select([row("2", "duplicate"), row("1", "duplicate"),
                         row("3", "unique"), row("4", "unknown", False)], limit=2)
        self.assertEqual(len(result["selected"]["binary"]), 2)
        self.assertEqual(result["shortfall"], {"binary": 0, "regression": 2})
        reasons = [reason for entry in result["exclusions"] for reason in entry["reasons"]]
        self.assertIn("duplicate_source_rows", reasons)
        self.assertIn("redistribution_rights", reasons)


if __name__ == "__main__":
    unittest.main()
