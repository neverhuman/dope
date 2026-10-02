import copy
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research.benchmark import inventory_hosts, resource_admission


class ResourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.host = {"cpu_count": 128, "allowed_cpus": list(range(128)),
                     "memory": {"MemTotal": 64 * 2**30, "MemAvailable": 60 * 2**30},
                     "scratch_disk": {"free_bytes": 80 * 2**30},
                     "load_average": [20, 20, 20], "active_gpu_processes": [],
                     "gpus": [{"memory_free_mib": 24 * 1024}]}
        self.active = [self.reservation(32, 24), self.reservation(64, 24)]

    @staticmethod
    def reservation(start, gib, gpu=False):
        return {"cpu_slot": list(range(start, start + 16)), "ram_bytes": gib * 2**30,
                "scratch_bytes": 3 * 2**30, "requires_gpu": gpu}

    def assess(self, request, active=None, allocated=120_000_000_000):
        return resource_admission.assess(self.host, allocated,
                                        self.active if active is None else active, request)

    def test_free_ram_alone_cannot_admit_overcommitted_reservations(self):
        result = self.assess(self.reservation(16, 24, True))
        self.assertGreater(self.host["memory"]["MemAvailable"], 24 * 2**30)
        self.assertEqual(result["reserved_ram_bytes"], 72 * 2**30)
        self.assertFalse(result["admitted"])
        self.assertIn("ram_reservation_capacity", result["blockers"])
        self.assertTrue(self.assess(self.reservation(16, 8, True))["admitted"])

    def test_foreign_ram_use_keeps_other_study_reservations_covered(self):
        self.host["memory"]["MemAvailable"] = 26 * 2**30
        self.assertIn("ram_reservation_capacity",
                      self.assess(self.reservation(16, 8, True))["blockers"])

    def test_slot_overlap_or_fifth_slot_rejected(self):
        for request, active in (
            (self.reservation(32, 1), self.active),
            (self.reservation(64, 1), [self.reservation(start, 1)
                                     for start in (0, 16, 32, 48)])):
            self.assertIn("cpu_reservation_conflict", self.assess(request, active)["blockers"])

    def test_scratch_accounts_for_other_reserved_outputs(self):
        result = self.assess(self.reservation(16, 8), allocated=195_000_000_000)
        self.assertIn("scratch_reservation_capacity", result["blockers"])

    def test_foreign_gpu_or_second_study_gpu_rejected(self):
        self.host["active_gpu_processes"] = ["999, 1024"]
        self.assertIn("gpu_owner_or_vram_capacity",
                      self.assess(self.reservation(16, 8, True))["blockers"])
        self.host["active_gpu_processes"] = []
        active = copy.deepcopy(self.active)
        active[0]["requires_gpu"] = True
        self.assertIn("gpu_reservation_conflict",
                      self.assess(self.reservation(16, 8, True), active)["blockers"])

    def test_failed_owner_query_never_becomes_empty_gpu_inventory(self):
        failed = subprocess.CalledProcessError(1, ["nvidia-smi"])
        disk = SimpleNamespace(total=200_000_000_000, free=100_000_000_000)
        with patch.object(inventory_hosts.shutil, "disk_usage", return_value=disk), \
                patch.object(inventory_hosts, "command", side_effect=[
                    "0, test GPU, 24576, 24000, test-driver", failed]) as query:
            with self.assertRaises(subprocess.CalledProcessError):
                inventory_hosts.local()
        self.assertEqual(query.call_count, 2)


if __name__ == "__main__":
    unittest.main()
