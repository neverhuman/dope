"""Reject incomparable receipt joins and prevent scheduling estimates becoming results."""
import copy
import hashlib
import gzip
import io
import json
from pathlib import Path
from statistics import mean, median
import tempfile
import unittest
from unittest.mock import patch

import jsonschema
from research.benchmark import publish_tabddpm_matched_coverage as publisher


class MatchedCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = publisher.load_sources()
        cls.panel = publisher.build(cls.docs)

    def test_hash_failure_precedes_json_decoding(self):
        with tempfile.TemporaryDirectory(dir=publisher.REPO / "target") as directory:
            root = Path(directory)
            (root / "bad.json").write_text("{broken")
            with patch.object(publisher, "REPO", root), patch.object(publisher, "SOURCES", {
                "bad": ("bad.json", "0" * 64)}), patch.object(publisher.json, "loads") as decode:
                with self.assertRaisesRegex(ValueError, "frozen_publication_changed"):
                    publisher.load_sources()
                decode.assert_not_called()

    def test_matched_partitions_and_real_controls_are_required(self):
        for mutation in ("partition", "real_control", "evaluator", "folds", "fit_type"):
            docs = copy.deepcopy(self.docs)
            cell = docs["tabddpm"]["physical_panel"]["cells"][0]
            if mutation == "partition": cell["input_hashes"]["validation_ref"] = "0" * 64
            elif mutation == "real_control":
                dataset = cell["dataset"]
                for row in docs["followon"]["rows"]:
                    if row["dataset"] == dataset and row["method"] == "ARF": row["null_loss"] += 1
            elif mutation == "evaluator":
                docs["tabsyn"]["source_refs"]["expanded_validation_metrics"]["sha256"] = "0" * 64
            elif mutation == "folds": cell["metrics"]["detection"]["catboost"]["folds"] = 3
            else: docs["tabsyn"]["cells"][0]["fit_seed"] = 11.0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): publisher.build(docs)

    def test_partial_reference_group_is_not_averaged(self):
        docs = copy.deepcopy(self.docs)
        removed = docs["tabsyn"]["cells"].pop(0)
        result = publisher.build(docs)
        matching = [r for r in result["paired_descriptive"] if r["reference_method"] == "TabSyn"
                    and r["row_multiplier"] == removed["row_multiplier"]]
        self.assertTrue(matching)
        for row in matching: self.assertNotIn(removed["dataset"], row["datasets"])

    def resolve(self, ref):
        key = next(k for k, (path, sha) in publisher.SOURCES.items() if
                   (path, sha) == (ref["path"], ref["sha256"]))
        obj = self.docs[key]
        for token in ref["pointer"].split("/")[1:]: obj = obj[int(token)] if isinstance(obj, list) else obj[token]
        return publisher.flatten(obj) if "metrics" in obj else obj

    def test_every_summary_is_paired_on_complete_sample_groups(self):
        for summary in self.panel["paired_descriptive"]:
            group = [r for r in self.panel["matched_cells"] if all(r[k] == summary[k] for k in
                     ("tabddpm_selection", "reference_method", "row_multiplier"))]
            values = {}
            for r in group:
                left, right = self.resolve(r["tabddpm_publication"]), self.resolve(r["reference_publication"])
                values.setdefault(r["dataset"], []).append((left[summary["metric"]], right[summary["metric"]]))
            usable = [(d, mean(x for x, _ in rows), mean(y for _, y in rows)) for d, rows in sorted(values.items())
                      if len(rows) == 3 and all(x is not None and y is not None for x, y in rows)]
            self.assertEqual(summary["datasets"], [d for d, _, _ in usable])
            self.assertEqual(summary["informative_lineages"], len(usable))
            self.assertEqual(summary["tabddpm_median"], median(x for _, x, _ in usable) if usable else None)
            self.assertEqual(summary["reference_median"], median(y for _, _, y in usable) if usable else None)
            self.assertEqual(summary["median_paired_difference"], median(x-y for _, x, y in usable) if usable else None)
            self.assertIsNone(summary["confidence_interval"])

    def test_missing_fit_seeds_have_only_conditional_operational_etas(self):
        panel = self.panel
        self.assertEqual(panel["coverage_by_fit_seed"], {"11": 132, "23": 0, "37": 0, "53": 0, "71": 0})
        self.assertEqual(len(panel["grid"]), 6000)
        self.assertEqual(sum(r["status"] == "measured" for r in panel["grid"]), 132)
        self.assertEqual(panel["planning"]["total_cap_minutes"], 16900)
        self.assertEqual(panel["planning"]["total_gpu_slot_hours"], 287+2/3)
        for row in panel["grid"]:
            if row["status"] == "measured":
                self.assertIsNone(row["conditional_eta_mdt"])
                self.assertEqual(self.resolve(row["measurement"])["sample_seed"], row["sample_seed"])
            else:
                self.assertIsNone(row["measurement"])
                self.assertGreater(row["eligible_gpu_minutes_to_metrics"], 0)
                self.assertTrue(row["conditional_eta_mdt"].endswith("-06:00"))
        self.assertFalse(panel["planning"]["gpu_admitted"])
        self.assertFalse(panel["planning"]["runnable_job_manifest"])
        self.assertFalse(panel["detection_protocol"]["identical_fold_membership_certified"])

    def test_committed_schema_replay_and_manifest(self):
        schema = json.loads((publisher.OUT / "panel.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
        validator.validate(self.panel)
        for name, text in publisher.render(self.panel).items(): self.assertEqual((publisher.OUT / name).read_text(), text)
        manifest = json.loads((publisher.OUT / "manifest.json").read_text())
        manifest_schema = json.loads((publisher.OUT / "manifest.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(manifest_schema)
        jsonschema.validate(manifest, manifest_schema)
        self.assertEqual(manifest["producer"]["sha256"], hashlib.sha256(Path(publisher.__file__).read_bytes()).hexdigest())
        for name, ref in manifest["files"].items():
            raw = (publisher.OUT / name).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (ref["bytes"], ref["sha256"]))
        for key in ("mfs_v2", "ptf_v1", "superiority", "release_safe_l3"):
            bad = copy.deepcopy(self.panel); bad[key] = .99
            with self.assertRaises(jsonschema.ValidationError): validator.validate(bad)

    def test_redteam_snapshot_and_all_artifact_sources_are_hash_bound(self):
        root = publisher.REPO / "research/benchmark/results/paper-receipt-redteam-20261008-v1"
        report = json.loads((root / "report.json").read_bytes())
        schema = json.loads((root / "report.schema.json").read_bytes())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(report)
        refs = list(report["manuscripts"])
        for artifact in report["artifacts"]:
            refs.extend([artifact["artifact"], *artifact["reduction_source_refs"]])
        for finding in report["findings"]: refs.extend(finding["affected_refs"])
        snapshot_raw = (root / "historical-paper.json").read_bytes()
        self.assertEqual(hashlib.sha256(snapshot_raw).hexdigest(),
                         "dbbcabc1c89a64931059e85c7139dc175100442fcbf3ce4b21da3cf4a66669aa")
        snapshot = json.loads(snapshot_raw)
        snapshot_schema = json.loads((root / "historical-paper.schema.json").read_bytes())
        jsonschema.Draft202012Validator(snapshot_schema).validate(snapshot)
        self.assertEqual(snapshot["reviewed_main_commit"], report["reviewed_main_commit"])
        self.assertEqual(snapshot["report_sha256"], hashlib.sha256((root / "report.json").read_bytes()).hexdigest())
        archived = {r["original_path"]: r for r in snapshot["sources"]}
        self.assertEqual(len(archived), len(snapshot["sources"]))
        self.assertEqual(set(archived), {r["path"] for r in refs if r["path"].startswith("docs/whitepaper/")})

        def audited_bytes(ref):
            self.assertFalse(Path(ref["path"]).is_absolute())
            if ref["path"] in archived:
                entry = archived[ref["path"]]
                self.assertEqual((entry["original_bytes"], entry["original_sha256"]), (ref["bytes"], ref["sha256"]))
                path = publisher.REPO / entry["snapshot_path"]
                self.assertEqual(path.parent, root / "historical-paper")
                compressed = path.read_bytes()
                self.assertEqual((len(compressed), hashlib.sha256(compressed).hexdigest()),
                                 (entry["snapshot_bytes"], entry["snapshot_sha256"]))
                with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as handle:
                    raw = handle.read(entry["original_bytes"] + 1)
            else:
                raw = (publisher.REPO / ref["path"]).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (ref["bytes"], ref["sha256"]))
            return raw

        for ref in refs:
            audited_bytes(ref)
        kp_ref = next(r for r in refs if r["path"] == "docs/whitepaper/generated/published-baseline-kpis.json")
        kp = json.loads(audited_bytes(kp_ref))
        self.assertEqual(report["numeric_kpi_points"], len(kp["figure_points"]))
        self.assertEqual({f["id"] for f in report["findings"]}, {"TRACE-01", "SCOPE-01"})
        self.assertFalse(report["official_tests_opened"])


if __name__ == "__main__": unittest.main()
