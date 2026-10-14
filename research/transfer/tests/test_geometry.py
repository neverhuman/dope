import unittest

import numpy as np

from research.transfer.geometry import (
    closeness_qualifies, coordinate_distance, ecdf, energy_distance_unbiased, fit_minmax,
    frozen_geometry_commitment, held_out_status, matched_neighbor_references, mmd2_unbiased,
    nearest_other_entity_distance, null_cloud,
    other_entity_row_count, real_real_calibration, shared_frozen_geometry, shared_geometry_claim,
    shared_null_realizations,
    sliced_wasserstein,
)


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
        self.assertTrue(frozen["shared_geometry"])
        self.assertEqual(frozen["reason"], "common_frozen_scaler")
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
            frozen_scaler={"frozen_id": "fit-scale-a", "columns": (unit, unit)},
        )
        self.assertEqual(near["estimand"], "nearest_other_entity_distance")
        self.assertEqual(near["frozen_id"], "fit-scale-a")
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
            frozen_scaler={"frozen_id": "fit-scale-a", "columns": (wide, wide)},
        )
        self.assertEqual(scaled["frozen_id"], "fit-scale-a")
        self.assertAlmostEqual(scaled["distances"][0], 0.5)
        self.assertAlmostEqual(scaled["distances"][1], (0.5 ** 2 + 0.5 ** 2) ** 0.5)
        tied = nearest_other_entity_distance(
            [[1.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0]],
            reference_entities=("b", "c"), query_entities=("a",),
            frozen_scaler={"frozen_id": "fit-scale-a", "columns": (unit, unit)},
        )
        self.assertEqual(tied["statuses"], ("tie",))
        self.assertEqual(tied["tie_counts"], (2,))
        self.assertEqual(tied["unique_neighbor"], (False,))
        self.assertNotIn("neighbor_index", tied)
        missing = nearest_other_entity_distance(
            [[0.0, 0.0]], [[0.0, 0.0]], reference_entities=("a",), query_entities=("a",),
            frozen_scaler={"frozen_id": "fit-scale-a", "columns": (unit, unit)},
        )
        self.assertEqual(missing["statuses"], ("missing_neighbor",))
        self.assertEqual(missing["distances"], (None,))
        self.assertEqual(missing["tie_counts"], (0,))
        nominal = {"kind": "nominal", "levels": ("red", "blue")}
        mixed = nearest_other_entity_distance(
            [(0.0, "red")], [(1.0, "blue")],
            reference_entities=("b",), query_entities=("a",),
            frozen_scaler={"frozen_id": "fit-scale-a", "columns": (unit, nominal)},
        )
        self.assertAlmostEqual(mixed["distances"][0], 2.0 ** 0.5)
        self.assertEqual(mixed["statuses"], ("measured",))
        marker = "category-marker-99999"
        with self.assertRaises(ValueError) as refused:
            nearest_other_entity_distance(
                [(0.0, marker)], [(0.0, "red")],
                reference_entities=("b",), query_entities=("a",),
                frozen_scaler={"frozen_id": "fit-scale-a", "columns": (unit, nominal)},
            )
        self.assertEqual(str(refused.exception), "nominal_outside_frozen_levels")
        self.assertNotIn(marker, str(refused.exception))
        with self.assertRaises(ValueError) as missing_value:
            nearest_other_entity_distance(
                [[float("nan"), 0.0]], [[0.0, 0.0]],
                reference_entities=("b",), query_entities=("a",),
                frozen_scaler={"frozen_id": "fit-scale-a", "columns": (unit, unit)},
            )
        self.assertEqual(str(missing_value.exception), "missing_value")
        with self.assertRaises(ValueError) as nonfinite:
            nearest_other_entity_distance(
                [[float("inf"), 0.0]], [[0.0, 0.0]],
                reference_entities=("b",), query_entities=("a",),
                frozen_scaler={"frozen_id": "fit-scale-a", "columns": (unit, unit)},
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
        scaler = {"frozen_id": "fit-scale-a", "columns": (unit, unit)}
        heldout = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        matched = matched_neighbor_references(
            flipped, heldout, flipped, fit_entities=entities, heldout_entities=entities,
            query_entities=entities, frozen_scaler=scaler,
        )
        counted = other_entity_row_count(flipped, flipped, reference_entities=entities, query_entities=entities)
        self.assertEqual(matched["estimand"], "matched_neighbor_references")
        self.assertEqual(matched["reference_rows"], 3)
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
        shared = shared_frozen_geometry(left, same)
        self.assertTrue(shared["shared_geometry"])
        self.assertEqual(shared["reason"], "common_frozen_state")
        self.assertIs(shared["label_verified"], False)
        self.assertIs(shared["qualifies_transfer"], False)
        self.assertNotEqual(left["frozen_id"], same["frozen_id"])
        self.assertEqual(left["state_digest"], same["state_digest"])
        for other in (relabeled, reordered, reordered_levels, other_encoder):
            refused = shared_frozen_geometry(left, other)
            self.assertFalse(refused["shared_geometry"])
            self.assertEqual(refused["reason"], "unverified_label")
            self.assertNotEqual(left["state_digest"], other["state_digest"])
            self.assertIs(refused["label_verified"], False)
        legacy = shared_geometry_claim(("label", "label"), frozen_id="label")
        self.assertTrue(legacy["shared_geometry"])
        self.assertEqual(legacy["reason"], "common_frozen_scaler")
        self.assertNotIn("state_digest", legacy)
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


if __name__ == "__main__":
    unittest.main()
