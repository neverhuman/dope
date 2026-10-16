"""Receipt-traceable, complete ARF reductions and partial TabSyn accounting."""
import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

from research.benchmark import publish_arf_tabsyn_followon as publisher


class FollowonPublication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = publisher.build()

    def test_exact_matrix_aliases_and_partial_groups(self):
        p = self.panel
        self.assertEqual(len(p["rows"]), 6095)
        self.assertEqual(len({r["metric_receipt_ref"]["sha256"] for r in p["rows"]}), 5892+95)
        self.assertEqual(len(p["summary"]), 400)
        for seed in publisher.FIT_SEEDS:
            self.assertEqual(sum(r["method"] == "ARF" and r["fit_seed"] == seed for r in p["rows"]), 1200)
        ts = [f for f in p["per_fit"] if f["method"] == "TabSyn"]
        self.assertEqual(sum(f["complete_sample_group"] for f in ts), 19)
        partial = [f for f in ts if not f["complete_sample_group"]]
        self.assertEqual(len(partial), 22)
        self.assertEqual(sum(len(f["source_row_indices"]) for f in partial), 38)
        self.assertTrue(all(v is None for f in partial for v in f["metrics"].values()))

    def test_summary_recomputes_sample_then_fit_means_and_sd(self):
        for s in self.panel["summary"]:
            fits = [self.panel["per_fit"][i] for i in s["source_per_fit_indices"]]
            self.assertEqual([f["fit_seed"] for f in fits], publisher.FIT_SEEDS)
            for metric, stat in s["metrics"].items():
                values = []
                for fit in fits:
                    rows = [self.panel["rows"][i] for i in fit["source_row_indices"]]
                    self.assertEqual(sorted(r["sample_seed"] for r in rows), publisher.SAMPLE_SEEDS)
                    observed = [r[metric] for r in rows]
                    values.append(mean(observed) if all(v is not None for v in observed) else None)
                self.assertEqual(stat["fit_seed_means"], values)
                self.assertEqual(stat["measured_fits"], sum(v is not None for v in values))
                self.assertEqual(stat["mean"], mean(values) if all(v is not None for v in values) else None)
                self.assertEqual(stat["fit_sample_sd"], stdev(values) if all(v is not None for v in values) else None)
                self.assertFalse(stat["sd_is_confidence_interval"])

    def test_missing_duplicate_float_seed_and_non_null_claim_fail(self):
        for name in ("missing", "duplicate", "float", "gate", "split", "job"):
            rows = copy.deepcopy(self.panel["rows"])
            if name == "missing": rows.pop(0)
            elif name == "duplicate": rows.append(copy.deepcopy(rows[0]))
            elif name == "float": rows[0]["fit_seed"] = 11.0
            elif name == "gate": rows[0]["mfs_v2"] = .99
            elif name == "split": rows[0]["input_hashes"]["validation"] = "0"*64
            else: rows[0]["metric_job_identity"]["fit_seed"] = 11.0
            with self.subTest(name=name), self.assertRaises(ValueError):
                publisher.reduce_rows(rows)

    def test_shared_physical_receipt_cannot_have_changed_metric(self):
        rows = copy.deepcopy(self.panel["rows"])
        seen = set()
        for row in rows:
            sha = row["metric_receipt_ref"]["sha256"]
            if sha in seen:
                break
            seen.add(sha)
        else:
            self.fail("expected a frozen physical alias")
        row["marginal_error_mean"] += .1
        with self.assertRaisesRegex(ValueError, "aliased_measurement_changed"):
            publisher.reduce_rows(rows)

    def test_five_fit_variance_requires_the_same_frozen_configuration(self):
        rows = copy.deepcopy(self.panel["rows"])
        chosen = next(r for r in rows if r["method"] == "ARF" and r["fit_seed"] == 23)
        for row in rows:
            if (row["dataset"], row["selection_binding"], row["fit_seed"]) == (chosen["dataset"], chosen["selection_binding"], 23):
                row["config_sha256"] = "0"*64
                row["logical_identity_sha256"] = publisher.digest({k:row[k] for k in publisher.IDENTITY})
        with self.assertRaisesRegex(ValueError, "configuration_changed_between_fits"):
            publisher.reduce_rows(rows)

    def test_public_worker_digest_restores_original_job_hash(self):
        for row in self.panel["rows"]:
            job = dict(row["metric_job_identity"])
            self.assertNotIn("worker_key", job)
            job["worker_key"] = job.pop("worker_sha256")
            self.assertEqual(publisher.digest(job), row["metric_job_sha256"])

    def test_locked_export_rejected_before_scalar_decode(self):
        Path("target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="target") as directory:
            root = Path(directory)
            (root/"inputs.lock.json").write_bytes((publisher.ROOT/"inputs.lock.json").read_bytes())
            (root/"scalar-cells.jsonl").write_bytes(b"invalid scalar JSON\n")
            with patch.object(publisher.json, "loads", wraps=json.loads) as decoder:
                with self.assertRaisesRegex(ValueError, "frozen_scalar_export_changed"):
                    publisher.build(root)
                self.assertEqual(decoder.call_count, 1)  # Import lock only.

    def test_committed_outputs_schema_and_manifest(self):
        for name, data in publisher.render(self.panel).items():
            self.assertEqual((publisher.ROOT/name).read_bytes(), data)
        schema = json.loads((publisher.ROOT/"schema.json").read_bytes())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(self.panel)
        manifest = json.loads((publisher.ROOT/"manifest.json").read_bytes())
        for name, value in (("import_lock", self.panel["inputs"]), ("manifest", manifest)):
            jsonschema.Draft202012Validator({"$ref": "#/$defs/"+name, "$defs": schema["$defs"]}).validate(value)
        for ref in manifest["files"]:
            data = (publisher.ROOT/ref["path"]).read_bytes()
            self.assertEqual(len(data), ref["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), ref["sha256"])
        source = Path(publisher.__file__).read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), manifest["publisher_sha256"])

    def test_operational_eta_recomputes_without_entering_kpi_rows(self):
        data = json.loads((publisher.ROOT/"operations-checkpoint.json").read_bytes())
        schema = json.loads((publisher.ROOT/"schema.json").read_bytes())
        jsonschema.Draft202012Validator({"$ref":"#/$defs/operations", "$defs":schema["$defs"]}).validate(data)
        start, end = data["observations"]
        elapsed = (datetime.fromisoformat(end["observed_utc"])-datetime.fromisoformat(start["observed_utc"])).total_seconds()
        rate = (end["physical_closed"]-start["physical_closed"])*3600/elapsed
        self.assertEqual(data["closures_per_hour"], rate)
        eta = datetime.fromisoformat(end["observed_utc"])+timedelta(hours=(400-end["physical_closed"])/rate)
        self.assertEqual(datetime.fromisoformat(data["fit_disposition_eta_mt"]), eta)
        self.assertEqual(datetime.fromisoformat(data["metric_publication_eta_mt"]), eta+timedelta(hours=6))
        self.assertTrue(data["forecast_only"])
        self.assertTrue(all("forecast_only" not in row for row in self.panel["rows"]))


if __name__ == "__main__":
    unittest.main()
