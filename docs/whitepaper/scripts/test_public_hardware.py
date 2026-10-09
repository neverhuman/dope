"""Public hardware sentences, origin counts, and the invented score rows."""

import json
import unittest
from pathlib import Path

import paper_emit
import public_hardware as hardware

REPO = Path(__file__).resolve().parents[3]


class PublicHardware(unittest.TestCase):
    def test_dope_sentence_names_devices_only(self):
        sentence = hardware.public_hardware(
            ["xbabe1", "xbabe3", "xbabe2"],
            {
                "xbabe1": "NVIDIA GeForce RTX 4090",
                "xbabe2": "NVIDIA GeForce RTX 3090",
                "xbabe3": "NVIDIA GeForce RTX 3090",
            },
        )
        self.assertEqual(
            sentence,
            "one NVIDIA GeForce RTX 4090 and two NVIDIA GeForce RTX 3090 GPUs",
        )
        self.assertEqual(hardware.banned_hits(sentence), [])

    def test_single_gpu_is_the_model_name(self):
        sentence = hardware.public_hardware(
            ["xbabe1"], {"xbabe1": "NVIDIA GeForce RTX 4090"}
        )
        self.assertEqual(sentence, "NVIDIA GeForce RTX 4090")

    def test_redacted_machine_id_matches_the_private_host(self):
        sentence = hardware.public_hardware(
            ["fit-machine-1"], {"fit-machine-1": "NVIDIA GeForce RTX 4090"}
        )
        self.assertEqual(sentence, "NVIDIA GeForce RTX 4090")

    def test_missing_gpu_does_not_invent_a_model(self):
        self.assertEqual(
            hardware.public_hardware(["xbabe2"], {"xbabe2": None}),
            "GPU model not recorded",
        )

    def test_no_host_stays_unrecorded(self):
        self.assertEqual(hardware.public_hardware([], {}), "host not recorded")

    def test_unknown_host_error_omits_the_name(self):
        secret = "secret-box"
        with self.assertRaises(ValueError) as caught:
            hardware.public_hardware([secret], {secret: "NVIDIA GeForce RTX 4090"})
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn("xbabe", str(caught.exception))

    def test_fit_path_hash_is_idempotent(self):
        raw = "/mnt/fast-scratch/dope-benchmark/attempts/fit.json"
        once = hardware.redact_fit_path(raw)
        self.assertNotIn("/", once)
        self.assertEqual(hardware.redact_fit_path(once), once)
        self.assertEqual(hardware.banned_hits(once), [])

    def test_origin_counts_on_the_lineage_ledger(self):
        path = REPO / "research/benchmark/results/s3-lineage-record.json"
        rows = json.loads(path.read_text())["rows"]
        counts = hardware.origin_counts(row["display_name"] for row in rows)
        self.assertEqual(
            counts,
            {"feynman": 20, "strogatz": 14, "fri": 25, "bng": 6, "other": 35},
        )
        sentence = hardware.origin_sentence(counts)
        self.assertEqual(hardware.banned_hits(sentence), [])
        self.assertIn("20", sentence)
        self.assertIn("35", sentence)


class MfsIllustration(unittest.TestCase):
    def test_one_repository_url_is_allowed(self):
        url = "https://github.com/neverhuman/dope"
        self.assertEqual(hardware.banned_hits(f"Receipts are at {url}."), [])
        self.assertIn("neverhuman", hardware.banned_hits(f"{url} and neverhuman"))
        self.assertIn("neverhuman", hardware.banned_hits(f"{url}\n{url}"))
        self.assertIn("jepsontaylor", hardware.banned_hits(f"{url} jepsontaylor"))
        self.assertIn("xbabe1", hardware.banned_hits(f"{url} on xbabe1"))
        self.assertIn("pull request number", hardware.banned_hits(f"{url}. See PR #194."))
        manuscript = (REPO / "docs/whitepaper/dope-mfs.tex").read_text()
        self.assertEqual(manuscript.count(url), 1)
        self.assertEqual(hardware.banned_hits(manuscript), [])
        anonymous = hardware.strip_journal_availability(manuscript)
        self.assertNotIn("neverhuman", anonymous)
        self.assertNotIn("BEGIN JOURNAL AVAILABILITY", anonymous)
        self.assertIn("not a utility score", anonymous)
        self.assertEqual(hardware.banned_hits(anonymous), [])

    def test_formula_rows_are_computed(self):
        shown = paper_emit.mfs_illustrations()
        self.assertEqual(shown["classic"], "82.40")
        self.assertEqual(shown["all_ones"], "100.0001")
        self.assertNotEqual(shown["zero_compactness"], "82.40")
        self.assertNotEqual(shown["zero_compactness"], "50.12")
        self.assertEqual(hardware.banned_hits(shown["zero_compactness"]), [])
        algebra = paper_emit.retention_algebra_sentence()
        self.assertIn("0.0694", algebra)
        self.assertIn("6.94", algebra)
        self.assertEqual(hardware.banned_hits(algebra), [])
