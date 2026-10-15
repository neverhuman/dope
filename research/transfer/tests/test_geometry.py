import math
import unittest
from unittest.mock import patch

import numpy as np

from research.transfer.geometry import (
    closeness_qualifies, coordinate_distance, ecdf, energy_distance_unbiased, fit_minmax,
    frozen_column_digest, frozen_geometry_commitment, held_out_status, matched_neighbor_references, mmd2_unbiased,
    nearest_other_entity_distance, null_cloud,
    other_entity_row_count, real_real_calibration, shared_frozen_geometry, shared_geometry_claim,
    shared_null_realizations,
    sliced_wasserstein,
)


def bound_scaler(*columns):
    columns = tuple(columns)
    return {"frozen_id": frozen_column_digest(columns), "columns": columns}


class GeometryTests(unittest.TestCase):
    def test_spaces_nulls_mmd_and_nonqualification(self):
        left = null_cloud(12, 3, 1)
        right = null_cloud(12, 3, 2)
        self.assertEqual(left.shape, right.shape)
        distance = coordinate_distance(left[0], right[0], space="row_activation", encoder_id="enc-a")
        self.assertGreater(distance, 0)
        with self.assertRaises(ValueError):
            coordinate_distance(left[0], right[0], space="row_activation", encoder_id="enc-a",
                                other_space="action_representation")
        same = mmd2_unbiased(np.array([[0.0], [0.0]]), np.array([[0.0], [0.0]]), sigma=1.0)
        self.assertEqual(same, 0.0)
        negative = mmd2_unbiased(np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]]), sigma=0.1)
        self.assertLess(negative, 0.0)
        flipped = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        counted = other_entity_row_count(flipped, flipped, reference_entities=("a", "a", "b"),
                                          query_entities=("a", "a", "b"))
        self.assertEqual(counted["estimand"], "other_entity_row_count")
        self.assertEqual(counted["counts"], (1, 1, 2))
        self.assertIs(counted["nearest_neighbor"], False)
        self.assertIs(counted["privacy_result"], False)
        with self.assertRaises(ValueError):
            other_entity_row_count(flipped, flipped[:2], reference_entities=("a", "a", "b"),
                                   query_entities=("a", "b"))
        refit = shared_geometry_claim([fit_minmax([[0], [1]]), fit_minmax([[3], [9]])])
        self.assertFalse(refit["shared_geometry"])
        self.assertEqual(refit["reason"], "per_table_refit")
        equal_ranges = shared_geometry_claim([fit_minmax([[0], [1]]), fit_minmax([[0], [1]])])
        self.assertFalse(equal_ranges["shared_geometry"])
        self.assertEqual(equal_ranges["reason"], "equal_ranges")
        frozen = shared_geometry_claim(("fit-scale-a", "fit-scale-a"), frozen_id="fit-scale-a")
        self.assertFalse(frozen["shared_geometry"])
        self.assertEqual(frozen["reason"], "unverified_label")
        self.assertIs(frozen["label_verified"], False)
        named_but_refit = shared_geometry_claim(
            [fit_minmax([[0], [1]]), fit_minmax([[0], [1]])], frozen_id="fit-scale-a",
        )
        self.assertFalse(named_but_refit["shared_geometry"])
        self.assertEqual(named_but_refit["reason"], "equal_ranges")
        band = ecdf(left[:, 0], space="row_activation")
        other = ecdf(right[:, 0], space="action_representation")
        self.assertNotEqual(band["space"], other["space"])
        calibration = real_real_calibration(orientation="higher_farther", scale=1.0, floor=1e-6)
        self.assertFalse(calibration["qualifies_transfer"])
        self.assertFalse(closeness_qualifies(distance)["privacy"])
        self.assertFalse(closeness_qualifies(distance)["formal_dp"])
        self.assertFalse(held_out_status("learner", {"embeddings"})["held_out"])
        self.assertTrue(held_out_status("learner", ())["held_out"])

    def test_one_null_cloud_is_shared_across_comparison_roles(self):
        shared = shared_null_realizations(8, 2, 3)
        again = shared_null_realizations(8, 2, 3)
        other = shared_null_realizations(8, 2, 4)
        direct = null_cloud(8, 2, 3)
        self.assertEqual(shared["estimand"], "shared_null_realization")
        self.assertEqual(shared["roles"], ("real_real", "real_synthetic", "synthetic_synthetic"))
        self.assertEqual(shared["scale"], 1.0)
        self.assertEqual(shared["cloud"].shape, (8, 2))
        self.assertTrue(np.array_equal(shared["cloud"], again["cloud"]))
        self.assertTrue(np.array_equal(shared["cloud"], direct))
        self.assertFalse(np.array_equal(shared["cloud"], other["cloud"]))
        self.assertIs(shared["shared"], True)
        self.assertIs(shared["nearest_neighbor"], False)
        self.assertIs(shared["row_count"], False)
        self.assertIs(shared["privacy_result"], False)
        self.assertIs(shared["qualifies_transfer"], False)
        calibration = real_real_calibration(orientation="higher_farther", scale=2.0, floor=1e-6)
        self.assertNotEqual(shared["scale"], calibration["scale"])
        self.assertFalse(calibration["qualifies_transfer"])
        with self.assertRaises(ValueError) as invalid:
            shared_null_realizations(1, 2, 0)
        self.assertEqual(str(invalid.exception), "invalid null cloud")

    def test_nearest_distance_uses_the_frozen_scaler(self):
        flipped = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        entities = ("a", "a", "b")
        counted = other_entity_row_count(flipped, flipped, reference_entities=entities, query_entities=entities)
        unit = {"kind": "continuous", "minimum": 0.0, "maximum": 1.0}
        near = nearest_other_entity_distance(
            flipped, flipped, reference_entities=entities, query_entities=entities,
            frozen_scaler=bound_scaler(unit, unit),
        )
        self.assertEqual(near["estimand"], "nearest_other_entity_distance")
        self.assertEqual(near["frozen_id"], frozen_column_digest((unit, unit)))
        self.assertEqual(near["distances"], (1.0, 2.0 ** 0.5, 1.0))
        self.assertNotEqual(near["distances"], counted["counts"])
        self.assertEqual(counted["counts"], (1, 1, 2))
        self.assertIs(near["nearest_neighbor"], True)
        self.assertIs(near["row_count"], False)
        self.assertIs(near["privacy_result"], False)
        self.assertNotIn("neighbor_index", near)
        wide = dict(unit)
        wide["maximum"] = 2.0
        scaled = nearest_other_entity_distance(
            flipped, flipped, reference_entities=entities, query_entities=entities,
            frozen_scaler=bound_scaler(wide, wide),
        )
        self.assertEqual(scaled["frozen_id"], frozen_column_digest((wide, wide)))
        self.assertNotEqual(scaled["frozen_id"], near["frozen_id"])
        self.assertAlmostEqual(scaled["distances"][0], 0.5)
        self.assertAlmostEqual(scaled["distances"][1], (0.5 ** 2 + 0.5 ** 2) ** 0.5)
        tied = nearest_other_entity_distance(
            [[1.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0]],
            reference_entities=("b", "c"), query_entities=("a",),
            frozen_scaler=bound_scaler(unit, unit),
        )
        self.assertEqual(tied["statuses"], ("tie",))
        self.assertEqual(tied["tie_counts"], (2,))
        self.assertEqual(tied["unique_neighbor"], (False,))
        self.assertNotIn("neighbor_index", tied)
        missing = nearest_other_entity_distance(
            [[0.0, 0.0]], [[0.0, 0.0]], reference_entities=("a",), query_entities=("a",),
            frozen_scaler=bound_scaler(unit, unit),
        )
        self.assertEqual(missing["statuses"], ("missing_neighbor",))
        self.assertEqual(missing["distances"], (None,))
        self.assertEqual(missing["tie_counts"], (0,))
        nominal = {"kind": "nominal", "levels": ("red", "blue")}
        mixed = nearest_other_entity_distance(
            [(0.0, "red")], [(1.0, "blue")],
            reference_entities=("b",), query_entities=("a",),
            frozen_scaler=bound_scaler(unit, nominal),
        )
        self.assertAlmostEqual(mixed["distances"][0], 2.0 ** 0.5)
        self.assertEqual(mixed["statuses"], ("measured",))
        marker = "category-marker-99999"
        with self.assertRaises(ValueError) as refused:
            nearest_other_entity_distance(
                [(0.0, marker)], [(0.0, "red")],
                reference_entities=("b",), query_entities=("a",),
                frozen_scaler=bound_scaler(unit, nominal),
            )
        self.assertEqual(str(refused.exception), "nominal_outside_frozen_levels")
        self.assertNotIn(marker, str(refused.exception))
        with self.assertRaises(ValueError) as missing_value:
            nearest_other_entity_distance(
                [[float("nan"), 0.0]], [[0.0, 0.0]],
                reference_entities=("b",), query_entities=("a",),
                frozen_scaler=bound_scaler(unit, unit),
            )
        self.assertEqual(str(missing_value.exception), "missing_value")
        with self.assertRaises(ValueError) as nonfinite:
            nearest_other_entity_distance(
                [[float("inf"), 0.0]], [[0.0, 0.0]],
                reference_entities=("b",), query_entities=("a",),
                frozen_scaler=bound_scaler(unit, unit),
            )
        self.assertEqual(str(nonfinite.exception), "nonfinite_geometry")
        with self.assertRaises(ValueError) as constant:
            nearest_other_entity_distance(
                [[0.0]], [[1.0]], reference_entities=("b",), query_entities=("a",),
                frozen_scaler={"frozen_id": "fit-scale-a", "columns": (
                    {"kind": "continuous", "minimum": 0.0, "maximum": 0.0},
                )},
            )
        self.assertEqual(str(constant.exception), "constant_column")
        with self.assertRaises(ValueError) as flat_nominal:
            nearest_other_entity_distance(
                [("red",)], [("blue",)], reference_entities=("b",), query_entities=("a",),
                frozen_scaler={"frozen_id": "fit-scale-a", "columns": (
                    {"kind": "nominal", "levels": ("red",)},
                )},
            )
        self.assertEqual(str(flat_nominal.exception), "constant_column")
        with self.assertRaises(ValueError) as ordinal:
            nearest_other_entity_distance(
                [[0.0]], [[1.0]], reference_entities=("b",), query_entities=("a",),
                frozen_scaler={"frozen_id": "fit-scale-a", "columns": ({"kind": "ordinal"},)},
            )
        self.assertEqual(str(ordinal.exception), "invalid frozen scaler")
        with self.assertRaises(ValueError) as refit:
            nearest_other_entity_distance(
                flipped, flipped, reference_entities=entities, query_entities=entities,
                frozen_scaler=fit_minmax([[0.0], [1.0]]),
            )
        self.assertEqual(str(refit.exception), "missing frozen scaler")

    def test_fit_and_heldout_references_match_cardinality(self):
        flipped = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        entities = ("a", "a", "b")
        unit = {"kind": "continuous", "minimum": 0.0, "maximum": 1.0}
        scaler = bound_scaler(unit, unit)
        heldout = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        matched = matched_neighbor_references(
            flipped, heldout, flipped, fit_entities=entities, heldout_entities=entities,
            query_entities=entities, frozen_scaler=scaler,
        )
        counted = other_entity_row_count(flipped, flipped, reference_entities=entities, query_entities=entities)
        self.assertEqual(matched["estimand"], "matched_neighbor_references")
        self.assertEqual(matched["reference_rows"], 3)
        self.assertEqual(matched["state_digest"], scaler["frozen_id"])
        self.assertEqual(matched["frozen_id"], scaler["frozen_id"])
        self.assertEqual(matched["fit_distances"], (1.0, 2.0 ** 0.5, 1.0))
        self.assertEqual(matched["heldout_distances"], (2.0, 5.0 ** 0.5, 1.0))
        self.assertNotEqual(matched["fit_distances"], counted["counts"])
        self.assertIs(matched["matched"], True)
        self.assertIs(matched["nearest_neighbor"], True)
        self.assertIs(matched["row_count"], False)
        self.assertIs(matched["privacy_result"], False)
        self.assertIs(matched["qualifies_transfer"], False)
        alone = matched_neighbor_references(
            [[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]],
            fit_entities=("a",), heldout_entities=("a",), query_entities=("a",), frozen_scaler=scaler,
        )
        self.assertEqual(alone["reference_rows"], 1)
        self.assertEqual(alone["fit_statuses"], ("missing_neighbor",))
        self.assertEqual(alone["heldout_statuses"], ("missing_neighbor",))
        self.assertEqual(alone["fit_distances"], (None,))
        with self.assertRaises(ValueError) as unequal:
            matched_neighbor_references(
                flipped, flipped[:2], flipped, fit_entities=entities, heldout_entities=("a", "b"),
                query_entities=entities, frozen_scaler=scaler,
            )
        self.assertEqual(str(unequal.exception), "unequal reference cardinality")

    def test_frozen_content_not_caller_label_binds_shared_geometry(self):
        unit = {"kind": "continuous", "minimum": 0.0, "maximum": 1.0}
        wide = {"kind": "continuous", "minimum": 0.0, "maximum": 2.0}
        nominal = {"kind": "nominal", "levels": ("red", "blue")}
        swapped_levels = {"kind": "nominal", "levels": ("blue", "red")}
        left = frozen_geometry_commitment(
            frozen_id="label", encoder_id="enc-a", schema_id="schema-a", columns=(unit, nominal),
        )
        same = frozen_geometry_commitment(
            frozen_id="other-label", encoder_id="enc-a", schema_id="schema-a", columns=(unit, nominal),
        )
        relabeled = frozen_geometry_commitment(
            frozen_id="label", encoder_id="enc-a", schema_id="schema-a", columns=(wide, nominal),
        )
        reordered = frozen_geometry_commitment(
            frozen_id="label", encoder_id="enc-a", schema_id="schema-a", columns=(nominal, unit),
        )
        reordered_levels = frozen_geometry_commitment(
            frozen_id="label", encoder_id="enc-a", schema_id="schema-a", columns=(unit, swapped_levels),
        )
        other_encoder = frozen_geometry_commitment(
            frozen_id="label", encoder_id="enc-b", schema_id="schema-a", columns=(unit, nominal),
        )

        def private(encoder_id, columns):
            return {"encoder_id": encoder_id, "schema_id": "schema-a", "columns": columns}

        left_content = private("enc-a", (unit, nominal))
        self.assertEqual(set(left), {
            "format", "version", "frozen_id", "state_digest", "label_verified",
            "privacy_result", "qualifies_transfer",
        })
        shared = shared_frozen_geometry(
            left, same, left_content=left_content, right_content=left_content,
        )
        self.assertTrue(shared["shared_geometry"])
        self.assertEqual(shared["reason"], "common_frozen_state")
        self.assertIs(shared["label_verified"], False)
        self.assertIs(shared["qualifies_transfer"], False)
        self.assertNotEqual(left["frozen_id"], same["frozen_id"])
        self.assertEqual(left["state_digest"], same["state_digest"])
        self.assertNotIn("encoder_id", left)
        self.assertNotIn("schema_id", left)
        self.assertNotIn("columns", left)
        self.assertEqual(
            left["state_digest"],
            "9b0ebffefb080f65f33fd457f7a531f9997514bf2c728acebebed82659864319",
        )
        copied = {
            "format": "dope-frozen-geometry-state",
            "version": 1,
            "frozen_id": "label",
            "label_verified": False,
            "state_digest": left["state_digest"],
        }
        with self.assertRaises(ValueError) as bare:
            shared_frozen_geometry(copied, copied)
        self.assertEqual(str(bare.exception), "invalid frozen scaler")
        self.assertNotIn(left["state_digest"], str(bare.exception))
        wide_content = private("enc-a", (wide, nominal))
        with self.assertRaises(ValueError) as forged:
            shared_frozen_geometry(
                copied, copied, left_content=wide_content, right_content=wide_content,
            )
        self.assertEqual(str(forged.exception), "unverified_state_digest")
        self.assertNotIn(left["state_digest"], str(forged.exception))
        letters = dict(left)
        letters["state_digest"] = "z" * 64
        with self.assertRaises(ValueError) as nonhex:
            shared_frozen_geometry(
                letters, letters, left_content=left_content, right_content=left_content,
            )
        self.assertEqual(str(nonhex.exception), "invalid frozen scaler")
        self.assertNotIn("z" * 64, str(nonhex.exception))
        tampered = dict(left)
        tampered["state_digest"] = "a" * 64
        with self.assertRaises(ValueError) as mismatch:
            shared_frozen_geometry(
                tampered, same, left_content=left_content, right_content=left_content,
            )
        self.assertEqual(str(mismatch.exception), "unverified_state_digest")
        self.assertNotIn(tampered["state_digest"], str(mismatch.exception))
        self.assertNotIn(left["state_digest"], str(mismatch.exception))
        comparisons = (
            (relabeled, private("enc-a", (wide, nominal))),
            (reordered, private("enc-a", (nominal, unit))),
            (reordered_levels, private("enc-a", (unit, swapped_levels))),
            (other_encoder, private("enc-b", (unit, nominal))),
        )
        for other, other_content in comparisons:
            refused = shared_frozen_geometry(
                left, other, left_content=left_content, right_content=other_content,
            )
            self.assertFalse(refused["shared_geometry"])
            self.assertEqual(refused["reason"], "unverified_label")
            self.assertNotEqual(left["state_digest"], other["state_digest"])
            self.assertIs(refused["label_verified"], False)
        legacy = shared_geometry_claim(("label", "label"), frozen_id="label")
        self.assertFalse(legacy["shared_geometry"])
        self.assertEqual(legacy["reason"], "unverified_label")
        self.assertNotIn("state_digest", legacy)
        same_columns = shared_geometry_claim((bound_scaler(unit, nominal), bound_scaler(unit, nominal)))
        self.assertTrue(same_columns["shared_geometry"])
        self.assertEqual(same_columns["reason"], "common_frozen_state")
        different_columns = shared_geometry_claim((
            {"frozen_id": "frozen-1", "columns": (unit, unit)},
            {"frozen_id": "frozen-1", "columns": (wide, wide)},
        ))
        self.assertFalse(different_columns["shared_geometry"])
        self.assertEqual(different_columns["reason"], "unverified_label")
        bound_pair = shared_geometry_claim((bound_scaler(unit, unit), bound_scaler(wide, wide)))
        self.assertFalse(bound_pair["shared_geometry"])
        self.assertEqual(bound_pair["reason"], "different_frozen_state")
        rows = np.array([[0.0, 0.0], [1.0, 0.0]])
        pair = ("a", "b")
        hundred = {"kind": "continuous", "minimum": 0.0, "maximum": 100.0}
        narrow_distance = nearest_other_entity_distance(
            rows, rows, reference_entities=pair, query_entities=pair, frozen_scaler=bound_scaler(unit, unit),
        )
        wide_distance = nearest_other_entity_distance(
            rows, rows, reference_entities=pair, query_entities=pair, frozen_scaler=bound_scaler(hundred, hundred),
        )
        self.assertNotEqual(narrow_distance["frozen_id"], wide_distance["frozen_id"])
        self.assertEqual(narrow_distance["state_digest"], narrow_distance["frozen_id"])
        self.assertEqual(wide_distance["state_digest"], wide_distance["frozen_id"])
        self.assertAlmostEqual(narrow_distance["distances"][0], 1.0)
        self.assertAlmostEqual(wide_distance["distances"][0], 0.01)
        one = np.array([[0.0], [0.5]])
        ends = ("a", "b")
        unit_column = {"kind": "continuous", "minimum": 0.0, "maximum": 1.0}
        wide_column = {"kind": "continuous", "minimum": 0.0, "maximum": 100.0}
        unit_distance = nearest_other_entity_distance(
            one, one, reference_entities=ends, query_entities=ends, frozen_scaler=bound_scaler(unit_column),
        )
        wide_span = nearest_other_entity_distance(
            one, one, reference_entities=ends, query_entities=ends, frozen_scaler=bound_scaler(wide_column),
        )
        self.assertAlmostEqual(unit_distance["distances"][0], 0.5)
        self.assertAlmostEqual(wide_span["distances"][0], 0.005)
        self.assertNotEqual(unit_distance["state_digest"], wide_span["state_digest"])
        unit_matched = matched_neighbor_references(
            one, one, one, fit_entities=ends, heldout_entities=ends, query_entities=ends,
            frozen_scaler=bound_scaler(unit_column),
        )
        wide_matched = matched_neighbor_references(
            one, one, one, fit_entities=ends, heldout_entities=ends, query_entities=ends,
            frozen_scaler=bound_scaler(wide_column),
        )
        self.assertEqual(unit_matched["state_digest"], unit_distance["state_digest"])
        self.assertEqual(unit_matched["frozen_id"], unit_distance["frozen_id"])
        self.assertNotEqual(unit_matched["state_digest"], wide_matched["state_digest"])
        self.assertNotIn("columns", unit_matched)
        self.assertAlmostEqual(unit_matched["fit_distances"][0], 0.5)
        self.assertAlmostEqual(wide_matched["fit_distances"][0], 0.005)
        sentinel_level = "secret-level-99999"
        sentinel_encoder = "encoder-sentinel-99999"
        sentinel_schema = "schema-sentinel-99999"
        extreme = {"kind": "continuous", "minimum": -424242.5, "maximum": 424242.5}
        hidden = {"kind": "nominal", "levels": ("red", sentinel_level)}
        sentinel_content = {
            "encoder_id": sentinel_encoder,
            "schema_id": sentinel_schema,
            "columns": (extreme, hidden),
        }
        redacted = frozen_geometry_commitment(
            frozen_id="label", encoder_id=sentinel_encoder, schema_id=sentinel_schema,
            columns=(extreme, hidden),
        )
        again = frozen_geometry_commitment(
            frozen_id="other-label", encoder_id=sentinel_encoder, schema_id=sentinel_schema,
            columns=(extreme, hidden),
        )
        redacted_shared = shared_frozen_geometry(
            redacted, again, left_content=sentinel_content, right_content=sentinel_content,
        )
        for secret in (sentinel_level, sentinel_encoder, sentinel_schema, "424242.5"):
            self.assertNotIn(secret, repr(redacted))
            self.assertNotIn(secret, repr(redacted_shared))
        self.assertTrue(redacted_shared["shared_geometry"])
        smuggled = dict(redacted)
        smuggled["columns"] = (extreme, hidden)
        with self.assertRaises(ValueError) as exposed:
            shared_frozen_geometry(
                smuggled, again, left_content=sentinel_content, right_content=sentinel_content,
            )
        self.assertEqual(str(exposed.exception), "invalid frozen scaler")
        self.assertNotIn(sentinel_level, str(exposed.exception))
        self.assertNotIn("424242.5", str(exposed.exception))
        with self.assertRaises(ValueError) as label:
            nearest_other_entity_distance(
                rows, rows, reference_entities=pair, query_entities=pair,
                frozen_scaler={"frozen_id": "fit-scale-a", "columns": (unit, unit)},
            )
        self.assertEqual(str(label.exception), "unverified_frozen_label")
        self.assertNotIn("fit-scale-a", str(label.exception))
        marker = "secret-level-99999"
        with self.assertRaises(ValueError) as echoed:
            frozen_geometry_commitment(
                frozen_id="label", encoder_id="enc-a", schema_id="schema-a",
                columns=({"kind": "nominal", "levels": ("red", marker, "red")},),
            )
        self.assertEqual(str(echoed.exception), "invalid frozen scaler")
        self.assertNotIn(marker, str(echoed.exception))

    def test_energy_and_sliced_wasserstein_do_not_qualify(self):
        negative = energy_distance_unbiased(np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]]))
        self.assertAlmostEqual(negative, -1.0)
        separated = energy_distance_unbiased(np.array([[0.0], [0.0]]), np.array([[5.0], [5.0]]))
        self.assertGreater(separated, 0.0)
        cloud = np.array([[0.0, 0.0], [1.0, 0.0]])
        same = sliced_wasserstein(cloud, cloud, projections=4, seed=0)
        self.assertEqual(same["value"], 0.0)
        self.assertFalse(same["privacy_result"])
        self.assertFalse(same["formal_dp"])
        moved = sliced_wasserstein(cloud, cloud + 3.0, projections=4, seed=0)
        self.assertGreater(moved["value"], 0.0)
        self.assertFalse(closeness_qualifies(moved["value"])["transfer"])
        self.assertFalse(closeness_qualifies(moved["value"])["privacy"])
        with self.assertRaises(ValueError) as unequal:
            sliced_wasserstein(cloud, cloud[:1], projections=4, seed=0)
        self.assertEqual(str(unequal.exception), "unequal sliced cardinality")


def _euclid(left, right):
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def _prior_energy(left, right):
    def mean_all(one, other):
        return sum(_euclid(row, item) for row in one for item in other) / (len(one) * len(other))

    def off_mean(rows):
        count = len(rows)
        total = sum(
            _euclid(row, other)
            for index, row in enumerate(rows)
            for other_index, other in enumerate(rows)
            if index != other_index
        )
        return total / (count * (count - 1))

    return 2 * mean_all(left, right) - off_mean(left) - off_mean(right)


def _prior_mmd(left, right, sigma):
    def kernel(one, other):
        gap = _euclid(one, other)
        return math.exp(-(gap * gap) / (2 * sigma * sigma))

    def mean_all(one, other):
        return sum(kernel(row, item) for row in one for item in other) / (len(one) * len(other))

    def off_mean(rows):
        count = len(rows)
        total = sum(
            kernel(row, other)
            for index, row in enumerate(rows)
            for other_index, other in enumerate(rows)
            if index != other_index
        )
        return total / (count * (count - 1))

    return off_mean(left) + off_mean(right) - 2 * mean_all(left, right)


def _role_work(left_rows, right_rows, width):
    roles = left_rows * left_rows + right_rows * right_rows + left_rows * right_rows
    return roles * (2 * width + 1)


class MetricBudgetTests(unittest.TestCase):
    def test_small_inputs_match_the_prior_formulas(self):
        left = ((0.0, 1.0), (2.0, -1.0))
        right = ((0.5, 0.25), (1.5, 0.0), (-0.5, 2.0))
        sigma = 0.5
        self.assertEqual(_role_work(2, 3, 2), (4 + 9 + 6) * (2 * 2 + 1))
        self.assertAlmostEqual(energy_distance_unbiased(left, right), _prior_energy(left, right))
        self.assertAlmostEqual(
            energy_distance_unbiased(left, right, max_elements=_role_work(2, 3, 2)),
            _prior_energy(left, right),
        )
        self.assertAlmostEqual(mmd2_unbiased(left, right, sigma=sigma), _prior_mmd(left, right, sigma))
        self.assertAlmostEqual(
            mmd2_unbiased(left, right, sigma=sigma, max_elements=_role_work(2, 3, 2)),
            _prior_mmd(left, right, sigma),
        )
        narrow = ((0.0,), (1.0,))
        self.assertAlmostEqual(energy_distance_unbiased(narrow, narrow), -1.0)
        zeros = ((0.0,), (0.0,))
        self.assertEqual(mmd2_unbiased(zeros, zeros, sigma=1.0), 0.0)

    def test_width_and_every_role_count_toward_the_bound(self):
        pair = ((0.0, 1.0), (1.0, 0.0))
        other = ((0.0, 0.0), (1.0, 1.0))
        wide_work = _role_work(2, 2, 2)
        self.assertEqual(wide_work, 60)
        self.assertAlmostEqual(
            energy_distance_unbiased(pair, other, max_elements=wide_work),
            _prior_energy(pair, other),
        )
        self.assertAlmostEqual(
            mmd2_unbiased(pair, other, sigma=1.0, max_elements=wide_work),
            _prior_mmd(pair, other, 1.0),
        )
        narrow = ((0.0,), (1.0,))
        same = ((2.0,), (3.0,))
        narrow_work = _role_work(2, 2, 1)
        self.assertEqual(narrow_work, 36)
        self.assertAlmostEqual(energy_distance_unbiased(narrow, same, max_elements=narrow_work), _prior_energy(narrow, same))
        longer = ((2.0,), (3.0,), (4.0,))
        role_work = _role_work(2, 3, 1)
        self.assertEqual(role_work, (4 + 9 + 6) * 3)
        self.assertAlmostEqual(
            energy_distance_unbiased(narrow, longer, max_elements=role_work),
            _prior_energy(narrow, longer),
        )
        self.assertAlmostEqual(
            mmd2_unbiased(narrow, longer, sigma=0.5, max_elements=role_work),
            _prior_mmd(narrow, longer, 0.5),
        )

        def allocated(*_args, **_kwargs):
            raise AssertionError("allocated")

        sentinel = "99999.5"
        marked = ((0.0,), (float(sentinel),))
        with patch("research.transfer.geometry._pairwise_norms", allocated), patch(
            "research.transfer.geometry._rbf", allocated
        ):
            for function, payload in (
                (energy_distance_unbiased, (pair, other)),
                (mmd2_unbiased, (pair, other)),
            ):
                with self.subTest(function=function.__name__, bound=wide_work - 1):
                    kwargs = {"max_elements": wide_work - 1}
                    if function is mmd2_unbiased:
                        kwargs["sigma"] = 1.0
                    with self.assertRaises(ValueError) as refused:
                        function(*payload, **kwargs)
                    self.assertEqual(str(refused.exception), "distance_budget_exceeded")
                    self.assertNotIn(sentinel, str(refused.exception))
            for omitted in (4 * 3, 9 * 3, 6 * 3, (4 + 9) * 3, (4 + 6) * 3, (9 + 6) * 3):
                with self.subTest(omitted=omitted):
                    with self.assertRaises(ValueError) as refused:
                        energy_distance_unbiased(marked, longer, max_elements=omitted)
                    self.assertEqual(str(refused.exception), "distance_budget_exceeded")
                    self.assertNotIn(sentinel, str(refused.exception))
                    with self.assertRaises(ValueError) as refused:
                        mmd2_unbiased(marked, longer, sigma=0.5, max_elements=omitted)
                    self.assertEqual(str(refused.exception), "distance_budget_exceeded")
            with self.assertRaises(ValueError) as narrow_refused:
                energy_distance_unbiased(marked, same, max_elements=narrow_work - 1)
            self.assertEqual(str(narrow_refused.exception), "distance_budget_exceeded")
            with self.assertRaises(ValueError) as wide_as_narrow:
                energy_distance_unbiased(pair, other, max_elements=narrow_work)
            self.assertEqual(str(wide_as_narrow.exception), "distance_budget_exceeded")

    def test_malformed_budget_is_refused_before_allocation(self):
        sentinel = "99999.5"
        left = ((0.0,), (float(sentinel),))
        right = ((1.0,), (2.0,))

        def allocated(*_args, **_kwargs):
            raise AssertionError("allocated")

        budgets = (True, False, 1.5, float("nan"), float("inf"), float("-inf"), 0, -1, 10_000_001)
        with patch("research.transfer.geometry._pairwise_norms", allocated), patch(
            "research.transfer.geometry._rbf", allocated
        ):
            for budget in budgets:
                with self.subTest(budget=type(budget).__name__):
                    with self.assertRaises(ValueError) as energy:
                        energy_distance_unbiased(left, right, max_elements=budget)
                    self.assertEqual(str(energy.exception), "invalid_distance_budget")
                    self.assertNotIn(sentinel, str(energy.exception))
                    self.assertNotIn("nan", str(energy.exception))
                    with self.assertRaises(ValueError) as kernel:
                        mmd2_unbiased(left, right, sigma=1.0, max_elements=budget)
                    self.assertEqual(str(kernel.exception), "invalid_distance_budget")
                    self.assertNotIn(sentinel, str(kernel.exception))
            with self.assertRaises(ValueError) as nonfinite:
                energy_distance_unbiased(((0.0,), (float("nan"),)), right, max_elements=36)
            self.assertEqual(str(nonfinite.exception), "nonfinite cloud")
            with self.assertRaises(ValueError) as scale:
                mmd2_unbiased(left, right, sigma=float("nan"), max_elements=36)
            self.assertEqual(str(scale.exception), "invalid kernel scale")
            self.assertNotIn(sentinel, str(scale.exception))
        self.assertAlmostEqual(
            energy_distance_unbiased(left, right, max_elements=10_000_000),
            _prior_energy(left, right),
        )


if __name__ == "__main__":
    unittest.main()
