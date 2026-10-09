"""Small synthetic controls; no source dataset, official test or fit is opened."""

import ast
import copy
import hashlib
import json
import math
import random
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

from research.benchmark import beyondarena_train_screen as screen


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def oracle():
    """Extract only the original pure projection and split functions."""
    path = Path(__file__).resolve().parents[1] / "manifest.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "e3f44001cbc52545b698fa1d2831349fe5ea6fd4f9c0c691d0526eb4bf5f271e":
        raise AssertionError("projection oracle source changed")
    names = {"canonical", "digest", "row_digest", "split_official_training",
             "_float", "fit_projection"}
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise AssertionError("projection oracle functions missing")
    namespace = {"hashlib": hashlib, "json": json, "math": math,
                 "random": random, "defaultdict": defaultdict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def source_binding():
    return {"family": screen.OVERLAP_FAMILIES[0], "source_manifest_sha256": "1" * 64,
            "source_parquet_sha256": "2" * 64, "source_split_metadata_sha256": "3" * 64}


def intersection_receipt(blocked):
    return {**source_binding(), "format": "dope-beyond-precomputed-overlap-groups",
            "version": 1,
            "canonical_row_hash": "SHA256_JSON_strings_sortkeys_compact_UTF8_ensure_ascii_false",
            "from_existing_authenticated_evaluator_hash_catalog": True,
            "only_TRAIN_intersection_group_hashes": True,
            "official_test_rows_decoded_during_this_operation": 0,
            "official_test_indices_or_values_in_worker": False,
            "evaluator_catalog_sha256": "4" * 64,
            "TRAIN_intersection_group_hashes": blocked}


def panel():
    names = list(screen.OVERLAP_FAMILIES) + ["candidate_a", "candidate_b", "candidate_c"]
    families = [{"name": name, "source_family_id": "beyondarena:" + name,
                 "license_spdx": "MIT", "task": "binary_classification",
                 "parquet_sha256": str(i + 1) * 64, "stratum": "iid"}
                for i, name in enumerate(names)]
    return {"format": "dope-beyondarena-panel-lock", "version": 3,
            "selected_count": len(families), "datasets": families,
            "historical_pilot_families": list(screen.OVERLAP_FAMILIES)}


class BeyondArenaTrainingScreenTests(unittest.TestCase):
    def test_01_width_matches_original_projection_and_target_is_protected(self):
        rows = [["1", "2", "a", "NaN", ""],
                ["2", "", "b", "", ""], ["3", "4", "a", "NaN", ""]]
        columns = ["target", "number", "category", "nonfinite", "missing"]
        for family in screen.WIDE_FAMILIES:
            result = screen.screen_inner_fit(family, rows, columns, "target")
            indices = result["column_indices"]
            mapping = oracle()["fit_projection"](
                [[row[i] for i in indices] for row in rows],
                [columns[i] for i in indices], "target", "regression")
            self.assertEqual(mapping["output_features"], result["projected_feature_width_estimate"])
            self.assertEqual(result["feature_indices"], [1, 2, 3])
            self.assertEqual(result["target_index_in_selected"], 0)
            self.assertEqual(result["projected_row_width_estimate"], mapping["output_features"] + 1)
            self.assertEqual(result["target_output_width"], 1)
            self.assertIsNone(result["complete_generator_byte_charge"])

    def test_02_width_cap_skips_whole_input_and_keeps_later_source_positions(self):
        rows = [["level" + str(i), "1", "2", ""] for i in range(2001)]
        result = screen.screen_inner_fit("kick", rows, ["large", "target", "small", "empty"], "target")
        self.assertEqual(result["column_indices"], [1, 2])
        self.assertEqual(result["projected_feature_width_estimate"], 1)
        rows = [["1", "level" + str(i), "2", "3"] for i in range(1999)]
        result = screen.screen_inner_fit("kick", rows, ["target", "levels", "first", "last"], "target")
        self.assertEqual(result["feature_indices"], [1, 2])
        self.assertEqual(result["projected_feature_width_estimate"], 2000)
        self.assertEqual(result["projected_row_width_estimate"], 2001)
        with self.assertRaises(ValueError):
            screen.screen_inner_fit("kick", [["1", ""]], ["target", "missing"], "target")

        for target_position in (0, 13):
            columns = ["input_" + str(i) for i in range(13)]
            columns.insert(target_position, "target")
            rows = [[str(i + 1) for i in range(14)], [str(i + 2) for i in range(14)]]
            result = screen.screen_inner_fit("kick", rows, columns, "target")
            predictors = [i for i in range(14) if i != target_position]
            self.assertEqual(result["feature_indices"], predictors[:12])
            self.assertEqual(result["selected_raw_inputs"], 12)
            self.assertEqual(result["raw_input_cap"], 12)
            self.assertIn(target_position, result["column_indices"])
            self.assertEqual(result["projected_feature_width_estimate"], 12)
            self.assertEqual(result["omitted_inputs"], [
                {"column_index": predictors[12], "reason": "raw_input_cap"}])

    def test_03_screen_neither_reads_holdout_nor_uses_target_to_rank_inputs(self):
        rows = [["1", "a", "2"], ["2", "b", "3"]]
        before = copy.deepcopy(rows)
        forbidden = mock.Mock(side_effect=AssertionError("holdout reader called"))
        with mock.patch("builtins.open", side_effect=AssertionError("file opened")), \
                mock.patch.object(Path, "open", side_effect=AssertionError("partition opened")):
            result = screen.screen_inner_fit("wine_world_cost", rows, ["target", "cat", "number"], "target")
            changed = [["100", "a", "2"], ["-100", "b", "3"]]
            self.assertEqual(result["feature_indices"], screen.screen_inner_fit(
                "wine_world_cost", changed, ["target", "cat", "number"], "target")["feature_indices"])
            with self.assertRaises(TypeError):
                screen.screen_inner_fit("kick", rows, ["target", "cat", "number"], "target",
                                        holdout_reader=forbidden)
        forbidden.assert_not_called()
        self.assertEqual(rows, before)

    def test_04_evaluator_receipt_removes_all_identical_train_group_occurrences(self):
        blocked = ["same", "0"]
        rows = [blocked, ["keep1", "0"], blocked[:], ["keep2", "1"], blocked[:]]
        raw = packed(intersection_receipt([screen.row_group_sha256(blocked)]))
        kept, receipt = screen.drop_train_overlap_groups(rows, raw, hashlib.sha256(raw).hexdigest(), source_binding())
        self.assertEqual(kept, [["keep1", "0"], ["keep2", "1"]])
        self.assertEqual(receipt["removed_TRAIN_rows"], 3)
        self.assertEqual(receipt["removed_groups"], 1)
        self.assertEqual(len(rows), 5)

    def test_05_unbound_or_test_decoding_or_wrong_source_intersection_refuses(self):
        rows = [["drop", "0"], ["a", "0"], ["b", "1"]]
        valid = intersection_receipt([screen.row_group_sha256(rows[0])])
        raw = packed(valid)
        with self.assertRaises(ValueError):
            screen.drop_train_overlap_groups(rows, raw, "0" * 64, source_binding())
        for key, value in [("official_test_rows_decoded_during_this_operation", True),
                           ("official_test_rows_decoded_during_this_operation", 1),
                           ("from_existing_authenticated_evaluator_hash_catalog", False),
                           ("source_parquet_sha256", "0" * 64),
                           ("TRAIN_intersection_group_hashes", ["0" * 64])]:
            bad = {**valid, key: value}; raw = packed(bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.drop_train_overlap_groups(rows, raw, hashlib.sha256(raw).hexdigest(), source_binding())

    def test_06_group_split_equals_original_and_keeps_complete_duplicate_groups(self):
        rows = [[str(i), str(i % 2)] for i in range(10)] + [["1", "1"]] * 4
        split = screen.split_training_groups(rows)
        self.assertEqual(split, oracle()["split_official_training"](rows))
        self.assertEqual(sorted(split["train"] + split["validation"]), list(range(len(rows))))
        hashes = [{screen.row_group_sha256(rows[i]) for i in split[k]} for k in ("train", "validation")]
        self.assertTrue(hashes[0].isdisjoint(hashes[1]))
        with self.assertRaises(ValueError):
            screen.split_training_groups([["same", "0"]] * 3)

    def test_07_missing_family_choices_are_frozen_hash_order_and_separate_cohort(self):
        locked = panel(); original = copy.deepcopy(locked)
        catalog = {"datasets": [{"name": "candidate_c"}, {"name": "candidate_a"}, {"name": "candidate_b"}]}
        raw = packed(catalog)
        result = screen.plan_catalog_successors(locked, raw, hashlib.sha256(raw).hexdigest())
        expected = sorted(["candidate_a", "candidate_b", "candidate_c"],
                          key=lambda name: (hashlib.sha256(("beyondarena:" + name).encode()).hexdigest(), name))
        self.assertEqual([x["prospective_successor"] for x in result["choices"]], expected)
        self.assertTrue(all(x["separate_cohort_required"] for x in result["choices"]))
        self.assertFalse(result["original_lock_or_denominator_changed"])
        self.assertFalse(result["training_exports_verified"])
        self.assertEqual(result["prepared_families"], 0)
        self.assertIsNone(result["mfs_v2"])
        self.assertEqual(locked, original)

    def test_08_catalog_gap_is_bounded_and_present_names_are_never_declared_absent(self):
        locked = panel()
        result = screen.plan_catalog_successors(locked, None, None)
        self.assertEqual(result["status"], "input_gap_catalog_not_supplied")
        self.assertEqual(result["choices"], [])
        raw = packed({"datasets": [{"name": name} for name in screen.OVERLAP_FAMILIES]})
        result = screen.plan_catalog_successors(locked, raw, hashlib.sha256(raw).hexdigest())
        self.assertTrue(all(x["status"] == "catalog_entry_present" for x in result["choices"]))
        for raw, pin in [(raw, "0" * 64), (packed({"datasets": [{"name": "x"}] * 2}), None),
                         (b"x" * (screen.CATALOG_BYTE_CAP + 1), None)]:
            with self.assertRaises(ValueError):
                screen.plan_catalog_successors(locked, raw, pin or hashlib.sha256(raw).hexdigest())
        for key, value in [("license_spdx", None), ("license_spdx", True), ("license_spdx", 1),
                           ("license_spdx", ["MIT"]), ("license_spdx", {"license": "MIT"}),
                           ("license_spdx", "not-an-approved-license"),
                           ("stratum", None), ("stratum", {"private": "synthetic-control"}),
                           ("stratum", "not-an-approved-stratum"), ("task", "multiclass"),
                           ("source_family_id", "another-source:candidate_c"),
                           ("parquet_sha256", "not-a-digest")]:
            bad = copy.deepcopy(locked); bad["datasets"][-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.plan_catalog_successors(bad, None, None)


if __name__ == "__main__":
    unittest.main()
