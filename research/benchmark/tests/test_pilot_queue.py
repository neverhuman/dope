import unittest

from research.benchmark import pilot_queue


class PilotQueueTests(unittest.TestCase):
    def test_unavailable_methods_have_no_dispatch_host(self):
        methods = {"methods": {name: {"status": "locked" if name in ("dope", "GaussianCopula")
                                       else "source_audit_pending"}
                               for name in pilot_queue.ROSTER}}
        cells = pilot_queue.matrix(methods)
        self.assertEqual(len(cells), 42)
        self.assertEqual(sum(cell["host"] is not None for cell in cells), 12)
        self.assertTrue(all(cell["host"] is None for cell in cells
                            if cell["method_status"] != "locked"))
        self.assertTrue(all(cell["host"] == "xbabe2" for cell in cells
                            if cell["method"] == "GaussianCopula"))
        self.assertTrue(all("test" not in str(cell).lower() for cell in cells))


if __name__ == "__main__":
    unittest.main()
