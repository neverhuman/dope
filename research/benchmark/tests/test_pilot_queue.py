import unittest
import queue

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

    def test_cpu_slot_is_returned_after_worker_failure(self):
        pool = queue.Queue()
        pool.put(1)
        with self.assertRaisesRegex(RuntimeError, "worker"):
            with pilot_queue.cpu_slot("xbabe2", {"xbabe2": pool}) as slot:
                self.assertEqual(slot, 1)
                raise RuntimeError("worker")
        self.assertEqual(pool.get_nowait(), 1)


if __name__ == "__main__":
    unittest.main()
