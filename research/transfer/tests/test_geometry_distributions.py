import unittest

import numpy as np

from research.transfer.geometry_distributions import (
    dependence_aware_ecdf, euclidean_cloud_report, euclidean_role_distributions,
)


class GeometryDistributionTests(unittest.TestCase):
    def test_matched_clouds_keep_three_roles_and_an_unmeasured_band(self):
        real = np.array([[0.0], [3.0]])
        synthetic = np.array([[10.0], [10.0]])
        report = euclidean_role_distributions(real, synthetic, space="row_activation", encoder_id="enc-a")
        self.assertEqual(report["roles"], ("real_real", "real_synthetic", "synthetic_synthetic"))
        self.assertEqual(report["distances"]["real_real"], (3.0,))
        self.assertEqual(report["distances"]["synthetic_synthetic"], (0.0,))
        self.assertEqual(report["distances"]["real_synthetic"], (7.0, 7.0, 10.0, 10.0))
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["pair_counts"], {"real_real": 1, "real_synthetic": 4, "synthetic_synthetic": 1})
        self.assertNotEqual(report["distances"]["real_real"][0], report["row_count"])
        self.assertIsNone(report["independent_units"])
        self.assertIsNone(report["band"])
        self.assertFalse(report["qualifies_transfer"])
        self.assertFalse(report["privacy_result"])
        self.assertFalse(report["formal_dp"])
        self.assertFalse(report["hipaa"])
        cloud = euclidean_cloud_report(real, synthetic, space="row_activation", encoder_id="enc-a", fit_count=1)
        self.assertEqual(cloud["reference_role"], "real_real")
        self.assertEqual(cloud["reference_probabilities"], (1.0,))
        self.assertEqual(cloud["ecdfs"]["real_synthetic"]["independent_units"], 1)
        self.assertEqual(cloud["ecdfs"]["real_synthetic"]["distance_count"], 4)
        self.assertEqual(cloud["ecdfs"]["real_synthetic"]["probabilities"], (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(cloud["band_status"], "unmeasured")
        self.assertIsNone(cloud["ecdfs"]["synthetic_synthetic"]["band"])
        self.assertFalse(cloud["qualifies_transfer"])
        self.assertFalse(cloud["formal_dp"])
        other_space = euclidean_cloud_report(
            real, synthetic, space="action_representation", encoder_id="enc-a", fit_count=1,
        )
        self.assertNotEqual(cloud["space"], other_space["space"])
        self.assertNotEqual(cloud["ecdfs"]["real_real"]["space"], other_space["ecdfs"]["real_real"]["space"])

    def test_cardinality_and_pair_units_fail_closed(self):
        real = np.array([[0.0], [3.0]])
        sentinel = 99999.5
        with self.assertRaises(ValueError) as unequal:
            euclidean_cloud_report(
                real, np.array([[sentinel], [sentinel], [sentinel]]),
                space="row_activation", encoder_id="enc-a", fit_count=1,
            )
        self.assertEqual(str(unequal.exception), "unequal cloud cardinality")
        self.assertNotIn("99999.5", str(unequal.exception))
        with self.assertRaises(ValueError) as width:
            euclidean_role_distributions(
                real, np.array([[0.0, 1.0], [1.0, 1.0]]), space="row_activation", encoder_id="enc-a",
            )
        self.assertEqual(str(width.exception), "invalid clouds")
        with self.assertRaises(ValueError) as pairs:
            dependence_aware_ecdf((0.0, 1.0, 2.0, 3.0, 4.0, 5.0), space="row_activation", row_count=4, fit_count=6)
        self.assertEqual(str(pairs.exception), "dependent_pairs_are_not_units")
        self.assertNotIn("5.0", str(pairs.exception))
        collapsed = euclidean_cloud_report(
            real, np.array([[sentinel], [sentinel]]), space="pooled_embedding", encoder_id="enc-a", fit_count=1,
        )
        self.assertEqual(collapsed["distances"]["synthetic_synthetic"], (0.0,))
        self.assertFalse(collapsed["qualifies_transfer"])
        self.assertFalse(collapsed["privacy_result"])
        self.assertNotIn("99999.5", collapsed["encoder_id"])
