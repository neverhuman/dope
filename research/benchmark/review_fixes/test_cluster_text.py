"""The family-cluster sentence matches the clustering and the bootstrap in stats.py."""

import unittest

import numpy as np

from research.benchmark.review_fixes import stats
from research.benchmark.review_fixes.cluster_text import describe_clusters


def _cluster_draw(values, clusters, rng, within=True):
    """Clusters with replacement, then (within=True) lineages inside each drawn cluster."""
    groups = {}
    for value, cluster in zip(values, clusters):
        groups.setdefault(cluster, []).append(value)
    members = [np.asarray(group, dtype=float) for group in groups.values()]
    samples = []
    for _draw in range(stats.DRAWS):
        pooled = []
        for index in rng.integers(0, len(members), size=len(members)):
            group = members[int(index)]
            pooled.extend(group[rng.integers(0, len(group), size=len(group))] if within else group)
        samples.append(float(np.median(pooled)))
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _one_stage(values, rng):
    array = np.asarray(values, dtype=float)
    samples = [np.median(array[rng.integers(0, array.size, size=array.size)]) for _draw in range(stats.DRAWS)]
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


class ClusterTextTests(unittest.TestCase):
    def test_sentence_names_every_simulated_family_and_both_stages(self):
        text = describe_clusters()
        self.assertEqual(stats.SIMULATED, ("feynman", "strogatz", "fri", "bng"))
        for family in stats.SIMULATED:
            self.assertIn(family, text)
        self.assertIn("(feynman, strogatz, fri, and bng)", text)
        self.assertIn("four simulated name families", text)
        self.assertIn("two-stage", text)
        self.assertIn("singleton cluster", text)
        self.assertEqual(text.count("with replacement"), 2)
        self.assertNotIn("resample lineages.", text)

    def test_sentence_follows_the_tuple(self):
        self.assertIn("one simulated name family (alpha) form", describe_clusters(("alpha",)))
        self.assertIn("(alpha and beta)", describe_clusters(("alpha", "beta")))
        self.assertIn("(a\\_b, c, and d)", describe_clusters(("a_b", "c", "d")))
        for bad in ((), ("a", "a"), ("",), (None,)):
            with self.assertRaises(ValueError):
                describe_clusters(bad)

    def test_cluster_of_matches_the_sentence(self):
        for family in stats.SIMULATED:
            self.assertEqual(stats.cluster_of(f"{family}_case", "id-" + family), family)
        self.assertEqual(stats.cluster_of("cpu_small", "id-1"), "lineage:id-1")
        self.assertNotEqual(stats.cluster_of("cpu_small", "id-1"), stats.cluster_of("cpu_small", "id-2"))

    def test_hierarchical_interval_is_the_stated_two_stage_draw(self):
        names = [f"feynman_{index}" for index in range(5)] + [f"strogatz_{index}" for index in range(4)]
        names += ["cpu_small", "cpu_act", "houses", "wine", "abalone", "boston"]
        values = [0.10, 0.31, 0.52, 0.73, 0.94, 0.22, 0.43, 0.64, 0.85, 0.17, 0.36, 0.58, 0.77, 0.96, 0.47]
        clusters = [stats.cluster_of(name, f"id{index}") for index, name in enumerate(names)]
        self.assertEqual(len(set(clusters)), 8)
        label = "cluster-text-fixture"
        summary = stats.median_ci(values, label, clusters)
        printed = [summary["lo"], summary["hi"]]
        self.assertEqual(printed, _cluster_draw(values, clusters, stats.rng_for(label + "|hier")))
        # The fixture separates the stated procedure from its two simpler readings.
        self.assertNotEqual(printed, _cluster_draw(values, clusters, stats.rng_for(label + "|hier"), within=False))
        self.assertNotEqual(printed, _one_stage(values, stats.rng_for(label + "|hier")))


if __name__ == "__main__":
    unittest.main()
