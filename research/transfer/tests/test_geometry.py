import unittest

import numpy as np

from research.transfer.geometry import (
    closeness_qualifies, coordinate_distance, ecdf, fit_minmax, held_out_status,
    mmd2_unbiased, nearest_neighbor_cardinality, null_cloud, real_real_calibration,
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
        counts = nearest_neighbor_cardinality(flipped, flipped, reference_entities=("a", "a", "b"),
                                              query_entities=("a", "a", "b"))
        self.assertEqual(counts, (1, 1, 2))
        with self.assertRaises(ValueError):
            nearest_neighbor_cardinality(flipped, flipped[:2], reference_entities=("a", "a", "b"),
                                         query_entities=("a", "b"))
        self.assertFalse(shared_geometry_claim([fit_minmax([[0], [1]]), fit_minmax([[3], [9]])])["shared_geometry"])
        self.assertTrue(shared_geometry_claim([fit_minmax([[0], [1]]), fit_minmax([[0], [1]])])["shared_geometry"])
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
