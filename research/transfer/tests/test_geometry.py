import unittest

import numpy as np

from research.transfer.geometry import (
    closeness_qualifies, coordinate_distance, ecdf, fit_minmax, held_out_status,
    mmd2_unbiased, null_cloud, other_entity_row_count, real_real_calibration,
    shared_geometry_claim,
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


if __name__ == "__main__":
    unittest.main()
