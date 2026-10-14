import math
import unittest

import numpy as np

import research.transfer.geometry as geometry
from research.transfer.geometry_distributions import (
    ECDF_BAND_DEVELOPMENT, dependence_aware_ecdf, euclidean_cloud_report,
    euclidean_role_distributions, grouped_euclidean_bands,
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
        synthetic_ecdf = cloud["ecdfs"]["real_synthetic"]
        self.assertIsNone(synthetic_ecdf["independent_units"])
        self.assertEqual(synthetic_ecdf["unit_status"], "unverified")
        self.assertFalse(synthetic_ecdf["fit_count_verified"])
        self.assertEqual(synthetic_ecdf["caller_fit_count"], 1)
        self.assertEqual(synthetic_ecdf["distance_count"], 4)
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


def _cluster(fit_id, real, synthetic, **extra):
    record = {
        "fit_id": fit_id,
        "lineage_id": extra.pop("lineage_id", "lineage-" + fit_id),
        "entity_id": extra.pop("entity_id", "entity-" + fit_id),
        "real": real,
        "synthetic": synthetic,
    }
    record.update(extra)
    return record


def _band(clusters, **overrides):
    arguments = {
        "space": "row_activation",
        "encoder_id": "enc-a",
        "grid": (0.0, 2.0),
        "seed": 0,
        "draws": 99,
    }
    arguments.update(overrides)
    return grouped_euclidean_bands(clusters, **arguments)


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


class GeometryDistributionControlTests(unittest.TestCase):
    def test_caller_fit_count_is_not_an_independent_unit(self):
        distances = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        for fit_count in (1, 3):
            report = dependence_aware_ecdf(
                distances, space="row_activation", row_count=4, fit_count=fit_count,
            )
            self.assertIsNone(report["independent_units"])
            self.assertEqual(report["unit_status"], "unverified")
            self.assertFalse(report["fit_count_verified"])
            self.assertEqual(report["band_status"], "unmeasured")
            self.assertIsNone(report["band"])
            self.assertEqual(report["caller_fit_count"], fit_count)
            self.assertFalse(report["hipaa"])
            self.assertFalse(report["n13b"])
        annotated = dependence_aware_ecdf(
            distances, space="row_activation", row_count=4, fit_count=5,
        )
        self.assertIsNone(annotated["independent_units"])
        self.assertEqual(annotated["caller_fit_count"], 5)
        with self.assertRaises(ValueError) as pairs:
            dependence_aware_ecdf(distances, space="row_activation", row_count=4, fit_count=6)
        self.assertEqual(str(pairs.exception), "dependent_pairs_are_not_units")
        self.assertNotIn("5.0", str(pairs.exception))

    def test_reorder_is_invariant_and_one_sided_permutation_is_not(self):
        real = np.array([[0.0, 1.0], [2.0, 0.0]])
        synthetic = np.array([[1.0, 0.0], [0.0, 3.0]])
        base = euclidean_role_distributions(real, synthetic, space="row_activation", encoder_id="enc-a")
        reordered = euclidean_role_distributions(real[::-1], synthetic, space="row_activation", encoder_id="enc-a")
        self.assertEqual(base["distances"], reordered["distances"])
        swapped = euclidean_role_distributions(
            real[:, ::-1], synthetic[:, ::-1], space="row_activation", encoder_id="enc-a",
        )
        self.assertEqual(base["distances"], swapped["distances"])
        one_sided = euclidean_role_distributions(
            real, synthetic[:, ::-1], space="row_activation", encoder_id="enc-a",
        )
        self.assertNotEqual(base["distances"]["real_synthetic"], one_sided["distances"]["real_synthetic"])

    def test_grouped_units_follow_fit_ids_and_duplicates_do_not_inflate_them(self):
        pair = [
            _cluster("fit-a", [[0.0], [0.0]], [[0.0], [0.0]]),
            _cluster("fit-b", [[0.0], [0.0]], [[2.0], [2.0]]),
        ]
        tripled = [
            _cluster("fit-a", [[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]),
            _cluster("fit-b", [[0.0], [0.0], [0.0]], [[2.0], [2.0], [2.0]]),
        ]
        short = _band(pair)
        long = _band(tripled)
        self.assertEqual(short["independent_units"], 2)
        self.assertEqual(long["independent_units"], 2)
        self.assertEqual(short["row_count"], 2)
        self.assertEqual(long["row_count"], 3)
        self.assertNotEqual(long["independent_units"], long["row_count"])
        self.assertEqual(short["fit_ids"], ("fit-a", "fit-b"))
        self.assertEqual(short["roles_result"]["real_synthetic"]["curve"], long["roles_result"]["real_synthetic"]["curve"])
        self.assertEqual(short["roles_result"], long["roles_result"])

    def test_pointwise_band_resamples_clusters_and_keeps_real_real_reference(self):
        clusters = [
            _cluster("fit-b", [[0.0], [0.0]], [[2.0], [2.0]]),
            _cluster("fit-a", [[0.0], [0.0]], [[0.0], [0.0]]),
        ]
        report = _band(clusters)
        again = _band(list(reversed(clusters)))
        self.assertEqual(report, again)
        synthetic = report["roles_result"]["real_synthetic"]
        self.assertEqual(synthetic["status"], "pointwise_band")
        self.assertEqual(synthetic["curve"], (0.5, 1.0))
        self.assertEqual(report["reference_role"], "real_real")
        self.assertEqual(report["roles_result"]["real_real"]["status"], "degenerate_clusters")
        self.assertIsNone(report["roles_result"]["real_real"]["lower"])
        self.assertIsNone(report["roles_result"]["real_real"]["upper"])
        self.assertEqual(report["roles_result"]["synthetic_synthetic"]["status"], "degenerate_clusters")
        curves = np.array([[1.0, 1.0], [0.0, 1.0]])
        picks = np.random.default_rng(0).integers(0, 2, size=(99, 2))
        resampled = curves[picks].mean(axis=1)
        lower, upper = np.quantile(resampled, (0.025, 0.975), axis=0, method="linear")
        self.assertEqual(synthetic["lower"], tuple(float(value) for value in lower))
        self.assertEqual(synthetic["upper"], tuple(float(value) for value in upper))
        self.assertTrue(all(math.isfinite(value) for value in synthetic["lower"] + synthetic["upper"] + synthetic["curve"]))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in synthetic["lower"] + synthetic["upper"] + synthetic["curve"]))
        self.assertTrue(all(left <= right for left, right in zip(synthetic["lower"], synthetic["upper"])))
        self.assertTrue(any(left < right for left, right in zip(synthetic["lower"], synthetic["upper"])))
        self.assertEqual(report["resample"], "whole_clusters_with_replacement")
        self.assertFalse(report["simultaneous"])
        self.assertEqual(report["protocol"]["format"], "dope-ecdf-band-development")
        self.assertEqual(report["protocol"]["version"], 1)
        self.assertFalse(report["protocol"]["qualifies"])
        self.assertFalse(report["protocol"]["n13b"])
        self.assertFalse(report["protocol"]["frozen_for_confirmation"])
        self.assertEqual(report["protocol"]["estimand"], ECDF_BAND_DEVELOPMENT["estimand"])
        for flag in ("qualifies_transfer", "privacy_result", "formal_dp", "hipaa", "n13b"):
            self.assertFalse(report[flag])
        self.assertFalse(any(isinstance(value, np.ndarray) for value in _leaves(report)))
        other = _band(clusters, space="action_representation")
        self.assertNotEqual(report["space"], other["space"])
        self.assertEqual(report["roles_result"], other["roles_result"])

    def test_too_few_or_degenerate_clusters_stay_unmeasured(self):
        one = _band([_cluster("fit-a", [[0.0], [1.0]], [[0.0], [1.0]])])
        self.assertEqual(one["independent_units"], 1)
        self.assertEqual(one["cluster_count"], 1)
        for role in one["roles_result"].values():
            self.assertEqual(role["status"], "insufficient_clusters")
            self.assertIsNone(role["curve"])
            self.assertIsNone(role["lower"])
            self.assertIsNone(role["upper"])
        twins = _band([
            _cluster("fit-a", [[0.0], [0.0]], [[0.0], [0.0]]),
            _cluster("fit-b", [[0.0], [0.0]], [[0.0], [0.0]]),
        ])
        for role in twins["roles_result"].values():
            self.assertEqual(role["status"], "degenerate_clusters")
            self.assertIsNone(role["lower"])
            self.assertIsNone(role["upper"])

    def test_grouped_refusals_use_static_messages(self):
        sentinel = "99999.5"
        valid = _cluster("fit-a", [[0.0], [1.0]], [[0.0], [1.0]])
        with self.assertRaises(ValueError) as query:
            _band(np.array([[0.0], [1.0]]))
        self.assertEqual(str(query.exception), "query is not a fit cluster")
        with self.assertRaises(ValueError) as rows:
            _band([np.array([[0.0], [1.0]]), valid])
        self.assertEqual(str(rows.exception), "query is not a fit cluster")
        missing = {"fit_id": sentinel, "entity_id": "entity-a", "real": [[0.0], [1.0]], "synthetic": [[0.0], [1.0]]}
        with self.assertRaises(ValueError) as incomplete:
            _band([missing, valid])
        self.assertEqual(str(incomplete.exception), "invalid cluster")
        self.assertNotIn(sentinel, str(incomplete.exception))
        with self.assertRaises(ValueError) as duplicate_fit:
            _band([
                _cluster(sentinel, [[0.0], [1.0]], [[0.0], [1.0]], entity_id="entity-a"),
                _cluster(sentinel, [[2.0], [3.0]], [[2.0], [3.0]], entity_id="entity-b"),
            ])
        self.assertEqual(str(duplicate_fit.exception), "duplicate fit id")
        self.assertNotIn(sentinel, str(duplicate_fit.exception))
        with self.assertRaises(ValueError) as duplicate_entity:
            _band([
                valid,
                _cluster("fit-b", [[0.0], [1.0]], [[2.0], [3.0]], entity_id="entity-fit-a"),
            ])
        self.assertEqual(str(duplicate_entity.exception), "duplicate entity")
        self.assertNotIn("entity-fit-a", str(duplicate_entity.exception))
        with self.assertRaises(ValueError) as encoder:
            _band([valid, _cluster("fit-b", [[0.0], [1.0]], [[2.0], [3.0]], encoder_id=sentinel)])
        self.assertEqual(str(encoder.exception), "inconsistent encoder")
        self.assertNotIn(sentinel, str(encoder.exception))
        digest = "ab" * 32
        other_digest = "z" * 64
        with self.assertRaises(ValueError) as state:
            _band([
                _cluster("fit-a", [[0.0], [1.0]], [[0.0], [1.0]], state_digest=digest),
                _cluster("fit-b", [[0.0], [1.0]], [[2.0], [3.0]], state_digest=other_digest),
            ])
        self.assertEqual(str(state.exception), "inconsistent frozen state")
        self.assertNotIn(digest, str(state.exception))
        self.assertNotIn(other_digest, str(state.exception))
        with self.assertRaises(ValueError) as cardinality:
            _band([
                valid,
                _cluster("fit-b", [[sentinel], [sentinel], [sentinel]], [[0.0], [1.0], [2.0]]),
            ])
        self.assertEqual(str(cardinality.exception), "unequal cloud cardinality")
        self.assertNotIn(sentinel, str(cardinality.exception))
        with self.assertRaises(ValueError) as inside:
            _band([_cluster("fit-a", [[0.0], [1.0]], [[sentinel], [sentinel], [sentinel]])])
        self.assertEqual(str(inside.exception), "unequal cloud cardinality")
        self.assertNotIn(sentinel, str(inside.exception))
        with self.assertRaises(ValueError) as nonfinite:
            _band([_cluster("fit-a", [[float(sentinel)], [float("nan")]], [[0.0], [1.0]])])
        self.assertEqual(str(nonfinite.exception), "invalid clouds")
        self.assertNotIn(sentinel, str(nonfinite.exception))
        with self.assertRaises(ValueError) as grid:
            _band([valid, _cluster("fit-b", [[0.0], [1.0]], [[2.0], [3.0]])], grid=(2.0, 0.0))
        self.assertEqual(str(grid.exception), "invalid ecdf")
        with self.assertRaises(ValueError) as integers:
            _band([valid, _cluster("fit-b", [[0.0], [1.0]], [[2.0], [3.0]])], grid=(0, 2))
        self.assertEqual(str(integers.exception), "invalid ecdf")
        with self.assertRaises(ValueError) as budget:
            _band([valid, _cluster("fit-b", [[0.0], [1.0]], [[2.0], [3.0]])], draws=98)
        self.assertEqual(str(budget.exception), "invalid bootstrap budget")
        declared = _band([
            _cluster("fit-a", [[0.0], [0.0]], [[0.0], [0.0]], state_digest=digest),
            _cluster("fit-b", [[0.0], [0.0]], [[2.0], [2.0]], state_digest=digest),
        ])
        self.assertEqual(declared["frozen_state"], "declared_unverified")
        self.assertNotIn(digest, [str(value) for value in _leaves(declared)])
        self.assertFalse(hasattr(geometry, "grouped_euclidean_bands"))
        self.assertFalse(hasattr(geometry, "euclidean_role_distributions"))
