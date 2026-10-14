import math
import unittest
from unittest.mock import patch

import numpy as np

import research.transfer.geometry as geometry
from research.transfer.geometry_distributions import (
    ECDF_BAND_CROSS_LINEAGE, ECDF_BAND_DEVELOPMENT, STEP5_CONTROLS, context_query_control,
    dependence_aware_ecdf, euclidean_cloud_report, euclidean_role_distributions,
    geometry_plot_hook, grouped_euclidean_bands, label_shuffle_control, tail_deletion_control,
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


def _role_oracle(real, synthetic):
    def upper(cloud):
        matrix = geometry._pairwise_norms(cloud, cloud)
        index = np.triu_indices(cloud.shape[0], k=1)
        return tuple(sorted(float(value) for value in matrix[index]))

    cross = geometry._pairwise_norms(real, synthetic)
    return {
        "real_real": upper(real),
        "real_synthetic": tuple(sorted(float(value) for value in cross.ravel())),
        "synthetic_synthetic": upper(synthetic),
    }


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
        self.assertEqual(report["roles_result"]["real_real"]["status"], "degenerate_lineages")
        self.assertIsNone(report["roles_result"]["real_real"]["lower"])
        self.assertIsNone(report["roles_result"]["real_real"]["upper"])
        self.assertEqual(report["roles_result"]["synthetic_synthetic"]["status"], "degenerate_lineages")
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
        self.assertEqual(report["resample"], "whole_lineages_with_replacement")
        self.assertFalse(report["simultaneous"])
        self.assertEqual(report["protocol"], ECDF_BAND_CROSS_LINEAGE)
        self.assertEqual(report["protocol"]["version"], 2)
        self.assertEqual(ECDF_BAND_DEVELOPMENT["version"], 1)
        self.assertNotIn("lineage_rule", ECDF_BAND_DEVELOPMENT)
        self.assertNotEqual(report["protocol"]["estimand"], ECDF_BAND_DEVELOPMENT["estimand"])
        self.assertFalse(report["conditional_fit_randomness_is_source_independence"])
        self.assertFalse(report["source_independence_verified"])
        self.assertEqual(report["unit_status"], "unverified")
        self.assertFalse(report["protocol"]["qualifies"])
        self.assertFalse(report["protocol"]["n13b"])
        self.assertFalse(report["protocol"]["frozen_for_confirmation"])
        for flag in ("qualifies_transfer", "privacy_result", "formal_dp", "hipaa", "n13b"):
            self.assertFalse(report[flag])
        self.assertFalse(any(isinstance(value, np.ndarray) for value in _leaves(report)))
        other = _band(clusters, space="action_representation")
        self.assertNotEqual(report["space"], other["space"])
        self.assertEqual(report["roles_result"], other["roles_result"])

    def test_too_few_or_degenerate_clusters_stay_unmeasured(self):
        one = _band([_cluster("fit-a", [[0.0], [1.0]], [[0.0], [1.0]])])
        self.assertEqual(one["independent_units"], 1)
        self.assertEqual(one["lineage_count"], 1)
        self.assertEqual(one["cluster_count"], 1)
        self.assertFalse(one["cross_lineage_band"])
        for role in one["roles_result"].values():
            self.assertEqual(role["status"], "insufficient_lineages")
            self.assertIsNone(role["curve"])
            self.assertIsNone(role["lower"])
            self.assertIsNone(role["upper"])
        twins = _band([
            _cluster("fit-a", [[0.0], [0.0]], [[0.0], [0.0]]),
            _cluster("fit-b", [[0.0], [0.0]], [[0.0], [0.0]]),
        ])
        for role in twins["roles_result"].values():
            self.assertEqual(role["status"], "degenerate_lineages")
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


class Step5ControlTests(unittest.TestCase):
    def test_step5_audit_keeps_ready_controls_executable(self):
        states = {item["control"]: item["state"] for item in STEP5_CONTROLS}
        self.assertEqual(list(states), [
            "duplicate", "reordered", "label_shuffled", "tail_deleted", "collapse",
            "row_count", "context", "coordinate_permutation", "pair_allocation",
            "held_out_relevance", "official_kumo_dope_measurements",
        ])
        for state in states.values():
            self.assertIn(state, {"executable", "unmeasured"})
        self.assertEqual(states["pair_allocation"], "executable")
        self.assertEqual(states["held_out_relevance"], "unmeasured")
        self.assertEqual(states["official_kumo_dope_measurements"], "unmeasured")

    def test_label_shuffle_does_not_change_coordinates_or_grant_held_out(self):
        real = np.array([[0.0], [1.0]])
        synthetic = np.array([[0.0], [1.0]])
        labels = ("keep", "drop")
        control = label_shuffle_control(
            real, synthetic, labels, space="row_activation", encoder_id="enc-a", seed=0,
        )
        plain = euclidean_role_distributions(real, synthetic, space="row_activation", encoder_id="enc-a")
        self.assertTrue(control["distances_unchanged"])
        self.assertEqual(control["distances"], plain["distances"])
        self.assertFalse(control["labels_are_coordinates"])
        self.assertFalse(control["labels_consumed"])
        self.assertFalse(control["selection_exposed"])
        self.assertFalse(control["held_out"])
        self.assertEqual(control["held_out_relevance"], "unmeasured")
        consumed = label_shuffle_control(
            real, synthetic, labels, space="row_activation", encoder_id="enc-a", seed=1,
            labels_consumed=True,
        )
        self.assertTrue(consumed["selection_exposed"])
        self.assertFalse(consumed["held_out"])
        self.assertEqual(consumed["distances"], plain["distances"])
        encoded = np.array([[0.0, 0.0], [1.0, 1.0]])
        included = euclidean_role_distributions(encoded, encoded, space="row_activation", encoder_id="enc-a")
        self.assertNotEqual(included["distances"], plain["distances"])
        sentinel = "99999.5"
        with self.assertRaises(ValueError) as mismatch:
            label_shuffle_control(
                real, synthetic, (sentinel,), space="row_activation", encoder_id="enc-a", seed=0,
            )
        self.assertEqual(str(mismatch.exception), "invalid labels")
        self.assertNotIn(sentinel, str(mismatch.exception))
        self.assertNotIn(sentinel, [str(value) for value in _leaves(control)])
        self.assertTrue(control["label_coordinate_changes_distances"])
        self.assertFalse(control["shuffle_changes_label_distances"])
        self.assertEqual(control["label_effect"], "detected")
        self.assertNotIn("keep", [str(value) for value in _leaves(control)])
        self.assertNotIn("drop", [str(value) for value in _leaves(control)])
        constant = label_shuffle_control(
            real, synthetic, ("same", "same"), space="row_activation", encoder_id="enc-a", seed=3,
        )
        self.assertEqual(constant["label_effect"], "degenerate_labels")
        self.assertFalse(constant["label_coordinate_changes_distances"])
        self.assertFalse(constant["shuffle_changes_label_distances"])
        self.assertTrue(constant["distances_unchanged"])
        self.assertEqual(constant["distances"], plain["distances"])
        self.assertFalse(constant["qualifies_transfer"])
        self.assertFalse(constant["formal_dp"])
        self.assertFalse(constant["hipaa"])
        self.assertNotIn("same", [str(value) for value in _leaves(constant)])
        rows = np.array([[0.0], [1.0], [2.0]])
        codes = np.array([0.0, 1.0, 2.0])
        coded = np.column_stack([rows, codes])
        moved = label_shuffle_control(
            rows, rows, ("a", "b", "c"), space="row_activation", encoder_id="enc-a", seed=0,
        )
        identity = label_shuffle_control(
            rows, rows, ("a", "b", "c"), space="row_activation", encoder_id="enc-a", seed=1,
        )
        shuffled_codes = codes[np.random.default_rng(0).permutation(3)]
        self.assertEqual(moved["label_distances"], _role_oracle(coded, coded))
        self.assertEqual(
            moved["shuffled_label_distances"],
            _role_oracle(np.column_stack([rows, shuffled_codes]), np.column_stack([rows, shuffled_codes])),
        )
        self.assertTrue(moved["label_coordinate_changes_distances"])
        self.assertTrue(moved["shuffle_changes_label_distances"])
        self.assertEqual(moved["label_effect"], "detected")
        self.assertTrue(identity["label_coordinate_changes_distances"])
        self.assertFalse(identity["shuffle_changes_label_distances"])
        self.assertEqual(identity["label_effect"], "detected")
        self.assertEqual(
            moved["distances"],
            euclidean_role_distributions(rows, rows, space="row_activation", encoder_id="enc-a")["distances"],
        )
        self.assertFalse(moved["labels_are_coordinates"])
        self.assertFalse(moved["qualifies_transfer"])
        self.assertFalse(moved["privacy_result"])
        self.assertFalse(moved["formal_dp"])
        self.assertFalse(moved["hipaa"])
        self.assertFalse(moved["n13b"])
        self.assertFalse(moved["held_out"])
        swapped = label_shuffle_control(
            real, synthetic, labels, space="row_activation", encoder_id="enc-a", seed=3,
        )
        self.assertTrue(swapped["label_coordinate_changes_distances"])
        self.assertFalse(swapped["shuffle_changes_label_distances"])
        self.assertNotIn("a", [str(value) for value in _leaves(moved)])

        def allocated(left, right):
            raise AssertionError("allocated")

        with patch("research.transfer.geometry_distributions._pairwise_norms", allocated):
            with self.assertRaises(ValueError) as width_budget:
                label_shuffle_control(
                    real, synthetic, labels, space="row_activation", encoder_id="enc-a",
                    seed=0, max_pairs=7,
                )
            self.assertEqual(str(width_budget.exception), "distance_budget_exceeded")

    def test_tail_deletion_changes_only_when_the_tail_is_present(self):
        changed = tail_deletion_control(
            np.array([[0.0], [0.0], [10.0]]), np.array([[1.0], [1.0], [1.0]]),
            space="row_activation", encoder_id="enc-a",
        )
        self.assertEqual(changed["real_real_before"], (0.0, 10.0, 10.0))
        self.assertEqual(changed["real_real_after"], (0.0,))
        self.assertTrue(changed["distance_values_changed"])
        self.assertEqual(changed["tail_effect"], "changed")
        self.assertTrue(changed["pair_count_changed"])
        self.assertEqual(changed["row_count_before"], 3)
        self.assertEqual(changed["row_count_after"], 2)
        self.assertFalse(changed["qualifies_transfer"])
        self.assertEqual(changed["held_out_relevance"], "unmeasured")
        unchanged = tail_deletion_control(
            np.array([[0.0], [0.0], [0.0]]), np.array([[1.0], [1.0], [1.0]]),
            space="row_activation", encoder_id="enc-a",
        )
        self.assertFalse(unchanged["distance_values_changed"])
        self.assertTrue(unchanged["pair_count_changed"])
        self.assertEqual(unchanged["tail_effect"], "unobserved")
        self.assertFalse(unchanged["qualifies_transfer"])
        with self.assertRaises(ValueError) as short:
            tail_deletion_control(
                np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]]),
                space="row_activation", encoder_id="enc-a",
            )
        self.assertEqual(str(short.exception), "invalid tail control")
        sentinel = "99999.5"
        with self.assertRaises(ValueError) as nonfinite:
            tail_deletion_control(
                np.array([[0.0], [1.0], [2.0]]),
                np.array([[float(sentinel)], [0.0], [float("nan")]]),
                space="row_activation", encoder_id="enc-a",
            )
        self.assertEqual(str(nonfinite.exception), "invalid clouds")
        self.assertNotIn(sentinel, str(nonfinite.exception))

    def test_context_rows_stay_out_of_the_query_distances(self):
        real = np.array([[0.0], [3.0]])
        synthetic = np.array([[10.0], [10.0]])
        control = context_query_control(
            ("context-row",), ("query-a", "query-b"), real, synthetic,
            space="row_activation", encoder_id="enc-a",
        )
        plain = euclidean_role_distributions(real, synthetic, space="row_activation", encoder_id="enc-a")
        self.assertEqual(control["distances"], plain["distances"])
        self.assertEqual(control["context_row_count"], 1)
        self.assertEqual(control["query_row_count"], 2)
        self.assertFalse(control["context_included_in_distances"])
        self.assertEqual(control["held_out_relevance"], "unmeasured")
        stacked = euclidean_role_distributions(
            np.array([[100.0], [0.0], [3.0]]), np.array([[10.0], [10.0], [10.0]]),
            space="row_activation", encoder_id="enc-a",
        )
        self.assertNotEqual(stacked["distances"], control["distances"])
        sentinel = "99999.5"
        with self.assertRaises(ValueError) as overlap:
            context_query_control(
                (sentinel,), (sentinel, "query-b"), real, synthetic,
                space="row_activation", encoder_id="enc-a",
            )
        self.assertEqual(str(overlap.exception), "query_context_overlap")
        self.assertNotIn(sentinel, str(overlap.exception))
        with self.assertRaises(ValueError) as pooled:
            context_query_control(
                ("context-row",), ("query-a", "query-b"), real, synthetic,
                space="row_activation", encoder_id="enc-a", include_context=True,
            )
        self.assertEqual(str(pooled.exception), "context is not a query")

    def test_pair_budget_refuses_before_allocation(self):
        real = np.array([[0.0], [3.0]])
        synthetic = np.array([[10.0], [10.0]])

        def allocated(left, right):
            raise AssertionError("allocated")

        clusters = [
            _cluster("fit-a", [[0.0], [0.0]], [[0.0], [0.0]]),
            _cluster("fit-b", [[0.0], [0.0]], [[2.0], [2.0]]),
        ]
        with patch("research.transfer.geometry_distributions._pairwise_norms", allocated):
            with self.assertRaises(ValueError) as refused:
                euclidean_role_distributions(
                    real, synthetic, space="row_activation", encoder_id="enc-a", max_pairs=3,
                )
            self.assertEqual(str(refused.exception), "distance_budget_exceeded")
            with self.assertRaises(ValueError) as invalid:
                euclidean_role_distributions(
                    real, synthetic, space="row_activation", encoder_id="enc-a", max_pairs=0,
                )
            self.assertEqual(str(invalid.exception), "invalid_distance_budget")
            with self.assertRaises(ValueError) as cap:
                euclidean_cloud_report(
                    real, synthetic, space="row_activation", encoder_id="enc-a", max_pairs=10_000_001,
                )
            self.assertEqual(str(cap.exception), "invalid_distance_budget")
            with self.assertRaises(ValueError) as grouped:
                _band(clusters, max_pairs=3)
            self.assertEqual(str(grouped.exception), "distance_budget_exceeded")
            wide = np.array([[0.0, 1.0], [3.0, 1.0]])
            wide_synthetic = np.array([[10.0, 1.0], [10.0, 4.0]])
            with self.assertRaises(ValueError) as wide_refused:
                euclidean_role_distributions(
                    wide, wide_synthetic, space="row_activation", encoder_id="enc-a", max_pairs=7,
                )
            self.assertEqual(str(wide_refused.exception), "distance_budget_exceeded")
            wide_clusters = [
                _cluster("fit-a", [[0.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]], lineage_id="L1"),
                _cluster("fit-b", [[0.0, 1.0], [2.0, 1.0]], [[3.0, 0.0], [1.0, 0.0]], lineage_id="L2"),
            ]
            with self.assertRaises(ValueError) as wide_grouped:
                _band(wide_clusters, max_pairs=7)
            self.assertEqual(str(wide_grouped.exception), "distance_budget_exceeded")
        allowed = euclidean_role_distributions(
            real, synthetic, space="row_activation", encoder_id="enc-a", max_pairs=4,
        )
        self.assertEqual(allowed["allocated_pairs"], 4)
        self.assertEqual(allowed["width"], 1)
        self.assertEqual(allowed["broadcast_elements"], 4)
        self.assertEqual(allowed["max_pairs"], 4)
        self.assertEqual(allowed["distances"]["real_real"], (3.0,))
        self.assertEqual(allowed["distances"]["real_synthetic"], (7.0, 7.0, 10.0, 10.0))
        self.assertEqual(allowed["distances"], _role_oracle(real, synthetic))
        self.assertIsNone(allowed["independent_units"])
        self.assertEqual(allowed["held_out_relevance"], "unmeasured")
        wide = np.array([[0.0, 1.0], [3.0, 1.0]])
        wide_synthetic = np.array([[10.0, 1.0], [10.0, 4.0]])
        allowed_wide = euclidean_role_distributions(
            wide, wide_synthetic, space="row_activation", encoder_id="enc-a", max_pairs=8,
        )
        self.assertEqual(allowed_wide["allocated_pairs"], 4)
        self.assertEqual(allowed_wide["width"], 2)
        self.assertEqual(allowed_wide["broadcast_elements"], 8)
        self.assertEqual(allowed_wide["distances"], _role_oracle(wide, wide_synthetic))
        self.assertFalse(allowed_wide["qualifies_transfer"])

    def test_plot_hook_keeps_real_measurements_unmeasured(self):
        real = np.array([[0.0], [3.0]])
        synthetic = np.array([[10.0], [10.0]])
        cloud = euclidean_cloud_report(real, synthetic, space="row_activation", encoder_id="enc-a", fit_count=1)
        hook = geometry_plot_hook(cloud)
        self.assertEqual(hook["format"], "dope-geometry-plot-hook-development")
        self.assertEqual(hook["version"], 1)
        self.assertEqual(hook["figure"], "unmeasured")
        self.assertEqual(hook["kumo_measurements"], "unmeasured")
        self.assertEqual(hook["dope_measurements"], "unmeasured")
        self.assertEqual(hook["held_out_relevance"], "unmeasured")
        self.assertFalse(hook["official_asset"])
        self.assertFalse(hook["qualifies_transfer"])
        self.assertFalse(hook["formal_dp"])
        self.assertFalse(hook["hipaa"])
        self.assertNotIn(10.0, list(_leaves(hook)))
        self.assertNotIn(7.0, list(_leaves(hook)))
        sentinel = "99999.5"
        with self.assertRaises(ValueError) as rejected:
            geometry_plot_hook({"estimand": sentinel, "space": "row_activation", "distances": (float(sentinel),)})
        self.assertEqual(str(rejected.exception), "invalid geometry report")
        self.assertNotIn(sentinel, str(rejected.exception))
        for name in (
            "label_shuffle_control", "tail_deletion_control", "context_query_control", "geometry_plot_hook",
        ):
            self.assertFalse(hasattr(geometry, name))

    def test_one_lineage_many_fits_is_unavailable_and_lineages_share_weight(self):
        sentinel = "99999.5"
        repeated = [
            _cluster(
                fit, [[0.0], [float(index)]], [[0.0], [0.0]],
                lineage_id=sentinel, entity_id="entity-" + fit,
            )
            for index, fit in enumerate(("fit-a", "fit-b", "fit-c"))
        ]
        one = _band(repeated)
        self.assertEqual(one["independent_units"], 1)
        self.assertEqual(one["fit_count"], 3)
        self.assertNotEqual(one["independent_units"], one["fit_count"])
        self.assertFalse(one["cross_lineage_band"])
        self.assertFalse(one["conditional_fit_randomness_is_source_independence"])
        for role in one["roles_result"].values():
            self.assertEqual(role["status"], "insufficient_lineages")
            self.assertIsNone(role["curve"])
            self.assertIsNone(role["lower"])
            self.assertIsNone(role["upper"])
        self.assertNotIn(sentinel, "".join(role["status"] for role in one["roles_result"].values()))
        distinct = [
            _cluster(fit, [[0.0], [0.0]], [[float(index)], [float(index)]])
            for index, fit in enumerate(("fit-a", "fit-b", "fit-c"))
        ]
        separate = _band(distinct)
        self.assertEqual(separate["independent_units"], 3)
        self.assertEqual(separate["fit_count"], 3)
        self.assertEqual(separate["lineage_ids"], ("lineage-fit-a", "lineage-fit-b", "lineage-fit-c"))
        self.assertEqual(len(set(separate["lineage_ids"])), separate["independent_units"])
        nested = [
            _cluster("fit-b", [[0.0], [0.0]], [[0.0], [0.0]], lineage_id="lineage-a", entity_id="entity-b"),
            _cluster("fit-c", [[0.0], [0.0]], [[2.0], [2.0]], lineage_id="lineage-b", entity_id="entity-c"),
            _cluster("fit-a", [[0.0], [0.0]], [[0.0], [0.0]], lineage_id="lineage-a", entity_id="entity-a"),
        ]
        report = _band(nested)
        again = _band(list(reversed(nested)))
        self.assertEqual(report, again)
        synthetic = report["roles_result"]["real_synthetic"]
        self.assertEqual(synthetic["status"], "pointwise_band")
        self.assertEqual(synthetic["curve"], (0.5, 1.0))
        self.assertNotEqual(synthetic["curve"], (2.0 / 3.0, 1.0))
        self.assertEqual(report["independent_units"], 2)
        self.assertEqual(report["fit_count"], 3)
        self.assertTrue(report["cross_lineage_band"])
        self.assertEqual(report["lineages"], (
            {"lineage_id": "lineage-a", "fit_ids": ("fit-a", "fit-b"), "entity_ids": ("entity-a", "entity-b")},
            {"lineage_id": "lineage-b", "fit_ids": ("fit-c",), "entity_ids": ("entity-c",)},
        ))
        curves = np.array([[1.0, 1.0], [0.0, 1.0]])
        picks = np.random.default_rng(0).integers(0, 2, size=(99, 2))
        resampled = curves[picks].mean(axis=1)
        lower, upper = np.quantile(resampled, (0.025, 0.975), axis=0, method="linear")
        self.assertEqual(synthetic["lower"], tuple(float(value) for value in lower))
        self.assertEqual(synthetic["upper"], tuple(float(value) for value in upper))
        hook = geometry_plot_hook(report)
        self.assertEqual(hook["estimand"], "equal_weight_mean_of_lineage_ecdfs")
        self.assertEqual(hook["figure"], "unmeasured")
        self.assertFalse(report["qualifies_transfer"])
