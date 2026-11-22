"""Episode contract and collation of the batched S3 core (plan M3, stage C2).

The episode dict must feed the existing losses unchanged, and ``collate`` / ``uncollate`` must
be exact inverses with masks that add up to the true sizes.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from research.levels.s3_batch_episode import TASK_IDS, BatchError, bucket_key, collate, out_bucket, uncollate
from research.levels.s3_counter_sampler import micro_batches
from research.levels.s3_dynamic_augment import AugmentConfig, sample_dynamic_episode
from research.levels.s3_fast_pipeline import build_unique_episode
from research.levels.s3_honest_counters import HonestCounters, split_hash, view_hash
from research.levels.s3_view_builder import RetryEpisode, ViewConfig, build_episode, build_slot
from research.levels.tests.s3_view_fixtures import TASKS, make_corpus, make_sampler

SEED = 77


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def built(corpus):
    """12 steps x 32 slots at the default sampler config, with the honest counters attached."""
    sampler = make_sampler(corpus, batch_size=32)
    counters = HonestCounters(sampler.index.n_tables, sampler.index.n_families)
    out, failed = [], 0
    for step in range(12):
        plan = sampler.plan_step(SEED, step)
        counters.observe_plan(plan)
        for spec in plan:
            episode, used, why = build_slot(corpus, sampler, SEED, spec, counters=counters)
            failed += len(why)
            out.append((used, episode))
    return sampler, counters, out, failed


class _LegacyStore:
    """The slice of ``RamCorpusStore`` that ``build_unique_episode`` touches."""

    def __init__(self, corpus):
        self.corpus = corpus

    def get(self, idx):
        t = self.corpus.table(idx)
        return {"X": t["X"], "y": t["y"], "dup_groups": t["dup"], "task": t["task"], "family_stem": t["family_id"],
                "n_classes": t["n_classes"] or 1, "dataset_hash": t["dataset_hash"]}


def test_episode_has_the_keys_and_dtypes_of_the_legacy_builder(corpus, built):
    store = _LegacyStore(corpus)
    by_task = {}
    for _, ep in built[2]:
        by_task.setdefault(ep["task"], ep)
    assert set(by_task) == set(TASK_IDS)
    seen = set()
    for slot in range(60):
        old = build_unique_episode(store, 2 + 4 * (slot % 18), global_seed=3, step=20000, rank=0, slot=slot)
        new = by_task[old["task"]]
        seen.add(old["task"])
        assert set(old) <= set(new) and set(old["aug_stats"]) <= set(new["aug_stats"])
        for key, value in old.items():
            assert type(new[key]) is type(value), key
            if isinstance(value, np.ndarray):
                assert new[key].dtype == value.dtype and new[key].ndim == value.ndim, key
    assert seen == set(TASK_IDS)
    for spec, ep in built[2]:
        assert ep["n_classes"] == {"regression": 1, "binary": 2}.get(ep["task"], ep["n_classes"])
        assert ep["y_fit"].shape == ((spec.n_fit,) if ep["task"] != "multilabel" else (spec.n_fit, ep["n_classes"]))
        assert ep["family_stem"] == spec.family_id and ep["aug_stats"]["n_cols"] == spec.p_total
        assert ep["distractor_mask"].dtype == bool and ep["distractor_mask"].sum() == spec.n_noise


def test_signatures_are_the_counter_hashes_and_counters_see_every_view(built):
    sampler, counters, episodes, failed = built
    for spec, ep in episodes:
        assert ep["view_sig"] == view_hash(spec, ep["cols"], ep["fit_rows"], ep["qry_rows"], ep["target_col"])
        assert int(ep["split_sig"], 16) == split_hash(spec, ep["fit_rows"], ep["qry_rows"]) and len(ep["split_sig"]) == 16
    summary = counters.summary()
    assert summary["built"] == len(episodes) == summary["planned"] and summary["failed"] == failed
    assert len({ep["view_sig"] for _, ep in episodes}) == len(episodes)
    assert abs(summary["distinct_views"] - len(episodes)) <= 0.03 * len(episodes)
    assert summary["distinct_tables"] == len({ep["table_id"] for _, ep in episodes if ep["table_id"] >= 0})


def test_prior_tables_and_multilabel_views(corpus, built):
    sampler, _, episodes, _ = built
    priors = [(s, e) for s, e in episodes if e["kind"] == "scm_prior"]
    assert Counter(e["task"] for _, e in priors).keys() >= {"regression", "binary", "multiclass"}
    for spec, ep in priors:
        assert ep["table_id"] == -1 and ep["dataset_hash"] == f"scm_{spec.view_seed:016x}" and ep["target_col"] == -1
        assert ep["X_fit"].shape == (spec.n_fit, spec.p_total) and ep["cols"].size == spec.p_view
        again = build_episode(corpus, spec)
        assert np.array_equal(ep["X_qry"], again["X_qry"]) and np.array_equal(ep["y_fit"], again["y_fit"])
        if ep["task"] == "multiclass":
            assert 2 <= ep["n_classes"] <= spec.n_classes and set(np.unique(ep["y_qry"])) <= set(np.unique(ep["y_fit"]))
        if ep["task"] == "regression":  # standardised on FIT, then clipped at 6 (heavy tails shrink the spread)
            assert abs(float(ep["y_fit"].mean())) < 0.2 and 0.2 < float(ep["y_fit"].std()) <= 1.01
    labelled = [(s, e) for s, e in episodes if e["task"] == "multilabel"]
    assert len(labelled) >= 10
    for spec, ep in labelled:
        y = np.concatenate([ep["y_fit"], ep["y_qry"]])
        assert y.shape == (spec.n_fit + spec.n_qry, spec.n_classes) and set(np.unique(y)) <= {0.0, 1.0}
        ones = ep["y_fit"].sum(axis=0)
        assert (ones >= 1).all() and (ones <= spec.n_fit - 1).all()  # no label is constant in FIT


def test_robust_v2_episode_has_no_spikes_in_fit_constant_columns(corpus):
    sampler = make_sampler(corpus, batch_size=32)
    sparse = {i for i in range(len(TASKS)) if i % 4 == 3}  # a constant column and a 99 %-zero column
    plain = dict(jitter_prob=0.0, warp_prob=0.0, miss_prob=0.0)
    flagged = spiked = 0
    for step in range(30):
        for spec in sampler.plan_step(SEED, step):
            if spec.kind != "real" or spec.table_id not in sparse:
                continue
            v2 = build_episode(corpus, spec, ViewConfig(scaler="robust_v2", **plain))
            nc2 = build_episode(corpus, spec, ViewConfig(scaler="nc2_compat", **plain))
            if isinstance(v2, RetryEpisode):
                continue
            const = v2["constant_mask"]
            assert np.array_equal(const, nc2["constant_mask"]) and np.array_equal(v2["col_source"], nc2["col_source"])
            assert const.sum() >= 1 and not v2["X_fit"][:, const].any() and not v2["X_qry"][:, const].any()
            flagged += int(const.sum())
            spiked += int((np.abs(nc2["X_qry"][:, const]) == 6.0).any())
    assert flagged >= 40 and spiked >= 5  # the legacy scaler does spike on these columns


def test_collate_then_uncollate_is_the_identity_and_masks_sum_to_true_sizes(built):
    episodes = [ep for _, ep in built[2]]
    for start in range(0, 96, 8):
        group = episodes[start:start + 8]
        batch = collate(group)
        b = len(group)
        assert len(batch) == b and batch.X_fit.shape[2] % 8 == 0 and batch.Y_fit.shape[2] in (1, 8, 32)
        for name, arr in batch.arrays().items():
            assert arr.shape[0] == b and arr.dtype in (np.float32, bool, np.int8, np.int32, np.uint64), name
        assert batch.row_fit.sum(axis=1).tolist() == [ep["X_fit"].shape[0] for ep in group] == batch.n_fit.tolist()
        assert batch.row_qry.sum(axis=1).tolist() == [ep["X_qry"].shape[0] for ep in group] == batch.n_qry.tolist()
        assert batch.col.sum(axis=1).tolist() == [ep["X_fit"].shape[1] for ep in group] == batch.p_true.tolist()
        assert batch.out_mask.sum(axis=1).tolist() == batch.n_out.tolist()
        assert batch.dmask.sum(axis=1).tolist() == [int(ep["distractor_mask"].sum()) for ep in group]
        assert not batch.X_fit[~batch.row_fit].any() and not batch.X_qry[..., ~batch.col.any(axis=0)].any()
        assert not batch.Y_fit[~batch.row_fit].any() and not (batch.Y_qry * ~batch.out_mask[:, None, :]).any()
        assert not (batch.dmask & ~batch.col).any() and batch.view_sig.tolist() == [ep["view_sig"] for ep in group]
        assert batch.split_sig.tolist() == [int(ep["split_sig"], 16) for ep in group]
        for i, ep in enumerate(group):  # the one target layout
            y, t = batch.Y_fit[i, :batch.n_fit[i]], ep["task"]
            if t == "multiclass":
                assert batch.n_out[i] == ep["n_classes"] and (y.sum(axis=1) == 1.0).all()
                assert np.array_equal(y.argmax(axis=1), ep["y_fit"].astype(np.int64))
            elif t == "multilabel":
                assert np.array_equal(y[:, :ep["n_classes"]], ep["y_fit"]) and batch.n_out[i] == ep["n_classes"]
            else:
                assert batch.n_out[i] == 1 and np.array_equal(y[:, 0], ep["y_fit"])
        back = uncollate(batch)
        for ep, twin in zip(group, back):
            assert set(ep) == set(twin)
            for key, value in ep.items():
                if isinstance(value, np.ndarray):
                    assert twin[key].dtype == value.dtype and np.array_equal(twin[key], value), key
                else:
                    assert twin[key] == value and type(twin[key]) is type(value), key
    wide = collate(episodes[:4], n_fit=4096, n_qry=1024, p=80, c=32)
    assert wide.X_fit.shape == (4, 4096, 80) and wide.Y_qry.shape == (4, 1024, 32)
    assert all(np.array_equal(a["X_qry"], b["X_qry"]) for a, b in zip(uncollate(wide), episodes[:4]))
    with pytest.raises(BatchError, match="pad_too_small"):
        collate(episodes[:4], p=4)
    with pytest.raises(BatchError, match="empty_batch"):
        collate([])


def test_bucket_key_matches_the_plan_and_exact_buckets_need_no_row_padding(built):
    sampler, _, episodes, _ = built
    for spec, ep in episodes:
        key = bucket_key(ep)
        assert key[1:] == spec.bucket_key[1:] and key[0] <= spec.bucket_key[0]  # a target may need fewer columns
        assert key[0] == out_bucket(1 if ep["task"] in ("regression", "binary") else ep["n_classes"])
    by_slot = {(spec.step, spec.slot): ep for spec, ep in episodes}
    plan = [spec for spec, _ in episodes if spec.step == 0]
    for group in micro_batches(plan, row_slack=0.25):
        batch = collate([by_slot[(s.step, s.slot)] for s in group])
        assert batch.X_fit.shape[1:] == (max(s.n_fit for s in group), group[0].bucket_key[1])
    for group in micro_batches(plan):  # exact buckets: every row of the batch is real
        batch = collate([by_slot[(s.step, s.slot)] for s in group])
        assert batch.row_fit.all() and batch.row_qry.all()


def test_legacy_episodes_collate_too(corpus):
    table = _LegacyStore(corpus).get(2)
    old = [sample_dynamic_episode(table, global_seed=1, step=s, rank=0, cfg=AugmentConfig()) for s in range(6)]
    batch = collate(old)  # no miss_*, no view_sig, 16-hex split_sig: still one batch
    assert not batch.miss_fit.any() and not batch.view_sig.any() and batch.split_sig.all()
    for ep, twin in zip(old, uncollate(batch)):
        assert all(np.array_equal(twin[k], ep[k]) for k in ("X_fit", "y_fit", "X_qry", "y_qry", "distractor_mask"))
        assert twin["task"] == ep["task"] and twin["split_sig"] == ep["split_sig"]


def test_existing_losses_consume_the_episode(built):
    torch = pytest.importorskip("torch")
    from research.levels.v6_landmarks import _irls_downstream_eval, extract_rich_landmarks

    torch.set_num_threads(1)
    done = set()
    for spec, ep in built[2]:
        if ep["task"] in done or spec.n_fit > 320 or spec.p_total > 24:
            continue
        done.add(ep["task"])
        x_fit, y_fit = torch.as_tensor(ep["X_fit"]), torch.as_tensor(ep["y_fit"])
        x_qry, y_qry = torch.as_tensor(ep["X_qry"]), torch.as_tensor(ep["y_qry"])
        marks, priors, knots = extract_rich_landmarks(x_fit, y_fit, torch.as_tensor(ep["miss_fit"]), ep["task"], ep["n_classes"])
        assert marks.shape[0] == spec.p_total and bool(torch.isfinite(marks).all()) and bool(torch.isfinite(knots).all())
        loss, null, importance = _irls_downstream_eval(x_fit, y_fit, x_qry, y_qry, ep["task"], ep["n_classes"])
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(null)) and importance.shape == (spec.p_total,)
        assert float(loss) < float(null) + 0.5  # the real FIT half is at least a sane learner on its QUERY half
    assert done == set(TASK_IDS)
