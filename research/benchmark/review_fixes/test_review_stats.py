"""Checks for the pre-declared review-fix statistics. No ledger is required."""

import unittest
import hashlib
import json
from pathlib import Path

import numpy as np

from unittest.mock import patch

from research.benchmark.review_fixes import stats as STATS
from research.benchmark.review_fixes.stats import (
    cluster_of,
    delta_method_se,
    family_of,
    median_ci,
    tost_mean,
)


class FamilyTests(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(family_of("feynman_I_12_1"), "feynman")
        self.assertEqual(family_of("strogatz_vdp2"), "strogatz")
        self.assertEqual(family_of("588_fri_c4_1000_100"), "fri")
        self.assertEqual(family_of("bng_pbc"), "bng")
        self.assertEqual(family_of("215_2dplanes"), "other")

    def test_other_lineages_are_not_one_cluster(self):
        self.assertNotEqual(cluster_of("215_2dplanes", "aa"), cluster_of("adult", "bb"))
        self.assertEqual(cluster_of("feynman_a", "aa"), cluster_of("feynman_b", "bb"))


class TostTests(unittest.TestCase):
    def test_exact_zero_is_equivalent_at_the_declared_margin(self):
        result = tost_mean(np.zeros(40))
        self.assertTrue(result["equivalent"])
        self.assertLess(result["p"], 0.05)

    def test_a_shift_of_one_tenth_is_not_equivalent(self):
        result = tost_mean(np.full(40, 0.1))
        self.assertFalse(result["equivalent"])

    def test_margin_is_the_predeclared_two_hundredths(self):
        self.assertEqual(tost_mean([0.0, 0.0])["margin"], 0.02)


class MedianTests(unittest.TestCase):
    def test_cluster_ids_must_match_values(self):
        with self.assertRaises(ValueError):
            median_ci([1.0, 2.0, 1000.0], "length", ["a", "b"])

    def test_tape_draws_equal_successive_integer_calls_and_final_state(self):
        # Bounds near 2**31 make Lemire reject about half of the words, exercising the redraw path.
        plan = np.random.default_rng(7)
        for seed in range(40):
            expected, generator = np.random.default_rng(seed), np.random.default_rng(seed)
            if seed % 2:  # Start with the high half of an output buffered.
                expected.integers(0, 5, size=1)
                generator.integers(0, 5, size=1)
            tape = STATS._WordTape(generator)
            position = 0
            for _ in range(int(plan.integers(0, 60))):
                bound = int(plan.choice([2, 3, 7, 50, 1000, 2**31 + 1, 2**32 - 5]))
                for _ in range(int(plan.integers(1, 9))):
                    value, position = STATS._below(tape, position, bound)
                    self.assertEqual(value, int(expected.integers(0, bound, size=1)[0]))
            tape.commit(position)
            self.assertEqual(tape.rng.bit_generator.state, expected.bit_generator.state)

    def test_hierarchical_bootstrap_matches_the_per_call_reference(self):
        data = np.random.default_rng(11)
        with patch.object(STATS, "DRAWS", 300):
            for case in range(16):
                n_clusters = [1, 2, 9, 60][case % 4]
                cluster_ids = [f"c{data.integers(0, n_clusters)}" for _ in range(int(data.integers(2, 200)))]
                if case % 3 == 0:  # Mostly singletons, as in the lineage clusters.
                    cluster_ids = [f"s{i}" for i in range(len(cluster_ids) - 3)] + ["pair"] * 3
                array = data.normal(size=len(cluster_ids)).round(int(data.integers(1, 4)))
                if case % 5 == 0:  # Signed zeros take the one-draw-at-a-time path.
                    array[: len(array) // 2] = -0.0
                if len(set(cluster_ids)) == len(cluster_ids):
                    continue
                with self.subTest(case=case):
                    fast, slow = STATS.rng_for(f"case{case}"), STATS.rng_for(f"case{case}")
                    self.assertEqual(STATS._hier_median_ci(array, cluster_ids, fast),
                                     _per_call_reference(array, cluster_ids, slow))
                    self.assertEqual(fast.bit_generator.state, slow.bit_generator.state)

    def test_rejected_words_replay_exactly_inside_a_block(self):
        # A zero word is rejected for every bound that is not a power of two, so zeros planted
        # among real words force Lemire redraws for cluster and member draws alike.
        words = np.random.default_rng(5).integers(0, 1 << 32, size=400_000, dtype=np.uint64)
        words[np.random.default_rng(6).choice(words.size, size=900, replace=False)] = 0
        data = np.random.default_rng(12)
        for case in range(6):
            cluster_ids = [f"s{i}" for i in range(40)] + ["a"] * 9 + ["b"] * 3 + [f"c{case % 3}"] * 5
            array = data.normal(size=len(cluster_ids)).round(2)
            groups = {}
            for value, cluster in zip(array, cluster_ids):
                groups.setdefault(cluster, []).append(float(value))
            members = [np.asarray(groups[key], dtype=float) for key in groups]
            sizes = np.asarray([len(group) for group in members], dtype=np.int64)
            with self.subTest(case=case):
                listed = _WordList(words, start=case * 5000)
                got, used = STATS._hier_samples(np.concatenate(members), sizes, listed, 400)
                reference = _WordList(words, start=case * 5000)
                expected = _per_call_samples(members, reference, 400)
                np.testing.assert_array_equal(got, expected)
                self.assertEqual(used, reference.position - case * 5000)


class _WordList:
    """A fixed word sequence used as a tape, or as rng.integers for bounds below 2**32."""

    def __init__(self, words, start=0):
        self.words, self.position, self.start = words, start, start

    def get(self, stop):
        return self.words[self.start:]

    def integers(self, low, high, size):
        out = []
        for _ in range(size):
            if high == 1:
                out.append(0)
                continue
            threshold = ((1 << 32) - high) % high
            while True:
                value = int(self.words[self.position]) * high
                self.position += 1
                if value & 0xFFFFFFFF >= threshold:
                    break
            out.append(value >> 32)
        return np.asarray(out, dtype=np.int64)


def _per_call_samples(members, rng, draws):
    """The historical per-value bootstrap loop, kept verbatim as the reference."""
    n_clusters = len(members)
    buffer = np.empty(n_clusters * max(len(group) for group in members), dtype=float)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        cursor = 0
        for index in rng.integers(0, n_clusters, size=n_clusters):
            group = members[int(index)]
            if len(group) == 1:
                buffer[cursor] = group[0]
                cursor += 1
                continue
            take = rng.integers(0, len(group), size=len(group))
            buffer[cursor:cursor + len(group)] = group[take]
            cursor += len(group)
        samples[draw] = float(np.median(buffer[:cursor]))
    return samples


def _per_call_reference(array, cluster_ids, rng):
    groups = {}
    for value, cluster in zip(array, cluster_ids):
        groups.setdefault(str(cluster), []).append(float(value))
    samples = _per_call_samples([np.asarray(groups[key], dtype=float) for key in groups], rng, STATS.DRAWS)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"median": float(np.median(array)), "lo": float(lo), "hi": float(hi)}


class DeltaTests(unittest.TestCase):
    def test_se_blows_up_as_the_gap_shrinks(self):
        wide = delta_method_se([1.0, 1.2, 0.8], null_loss=2.0, trtr_loss=0.0)
        tight = delta_method_se([1.0, 1.2, 0.8], null_loss=1.01, trtr_loss=1.0)
        self.assertLess(wide, tight)

    def test_zero_gap_is_undefined(self):
        self.assertIsNone(delta_method_se([1.0, 1.1], 1.0, 1.0))


class TableTests(unittest.TestCase):
    def test_column_spec_must_match_the_header_and_every_row(self):
        from research.benchmark.review_fixes.review_tex import longtable

        with self.assertRaises(ValueError):
            longtable("Caption. No Holm family.", "A & B", ["1 \\\\"], "ll")
        with self.assertRaises(ValueError):
            longtable("Caption without the family.", "A & B", ["1 & 2 \\\\"], "lr")
        body = longtable("Caption. No Holm family.", "A & B", ["1 & 2 \\\\"], "lr")
        self.assertIn("\\begin{longtable}{lr}\n\\caption{Caption. No Holm family.}", body)


class HistoricalBootstrapConformance(unittest.TestCase):
    def test_singleton_fast_path_preserves_frozen_intervals_and_rng_state(self):
        from research.benchmark.review_fixes import stats
        path = Path(__file__).with_name('stats-hier-conformance.json')
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         'f548379b386b9151e8854d351ac5d4aefec3bb9899b2fd956b66de66ae06da9e')
        fixture = json.loads(raw)
        self.assertEqual(stats.DRAWS, fixture['draws'])
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                rng = stats.rng_for('hier-conformance|' + case['name'])
                result = stats._hier_median_ci(np.asarray(case['values']), case['clusters'], rng)
                self.assertEqual(result, case['result'])
                state = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(hashlib.sha256(state).hexdigest(), case['rng_state_sha256'])

    def test_numpy_singleton_draw_consumes_no_random_state(self):
        from research.benchmark.review_fixes import stats
        rng = stats.rng_for('singleton-state-conformance')
        before = json.dumps(rng.bit_generator.state, sort_keys=True)
        self.assertTrue(np.array_equal(rng.integers(0, 1, size=37), np.zeros(37)))
        self.assertEqual(json.dumps(rng.bit_generator.state, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
