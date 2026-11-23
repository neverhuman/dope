"""The V8 trainer runs on factory episodes: real views with fresh splits, FIT-only native tensors."""
from __future__ import annotations

import json

import numpy as np
import pytest

from research.levels import s3_episode_factory as ef
from research.levels.s3_v8_feed import FactoryFeed
from research.levels.s3_view_builder import ViewConfig
from research.levels.tests.s3_view_fixtures import NARROW

torch = pytest.importorskip("torch")
from research.levels import s3_v8_trainer as tr  # noqa: E402  (needs torch)

SEED = 20261123
FIXTURES = "research.levels.tests.s3_view_fixtures"


def _spec(batch: int):
    return ef.FactorySpec(seed=SEED, corpus=f"{FIXTURES}:make_corpus", sampler=f"{FIXTURES}:make_sampler",
                          sampler_kwargs=dict(NARROW, multilabel_view_prob=0.0, batch_size=batch),
                          view=ViewConfig(emit="raw"), stage="research.levels.s3_episode_factory:native_v8")


def test_feed_serves_deterministic_episodes_for_any_producer_count():
    with FactoryFeed(_spec(6), 6, 2, start_step=1, max_loss_fit=64, max_loss_qry=32) as a:
        first = [a(step, b) for step in (1, 2) for b in range(6)]
        st = a.stats()
    with FactoryFeed(_spec(6), 6, 4, start_step=1, max_loss_fit=64, max_loss_qry=32) as b:
        again = [b(step, i) for step in (1, 2) for i in range(6)]
    for x, y in zip(first, again):
        assert x["packet"] == y["packet"] and x["split_sig"] == y["split_sig"]
        assert np.array_equal(x["X_fit"], y["X_fit"])
    assert all(ep["X_fit"].shape[0] <= 64 and ep["X_qry"].shape[0] <= 32 for ep in first)
    assert st["served"] == 12 and st["substituted"] == 0 and not st["failures"]
    # Twelve episodes in two steps are twelve distinct splits.
    assert len({ep["split_sig"] for ep in first}) == 12


def test_v8_trainer_steps_on_factory_episodes(tmp_path):
    args = tr.parse_args([
        "--out-dir", str(tmp_path), "--steps", "3", "--episodes-per-step", "2", "--d-model", "32",
        "--n-heads", "2", "--n-enc-layers", "1", "--n-dec-layers", "1", "--n-syn", "64", "--save-every", "3",
        "--contrastive-prob", "0", "--seed", "7",
    ])
    with FactoryFeed(_spec(2), 2, 2, start_step=1, max_loss_fit=128, max_loss_qry=64,
                     log_path=tmp_path / "feed.jsonl") as feed:
        result = tr.train_v8(args, episode_provider=feed)
        assert feed.stats()["served"] == 6
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert all(np.isfinite(r["grad_norm"]) for r in rows)
    assert result["steps_done"] == 3 and (tmp_path / "latest.pt").is_file()
