"""Episode factory (plan v3, C6): fresh random split per episode, FIT-only features, parallel producers."""
from __future__ import annotations

import hashlib
import pickle
import platform
import threading
import time
from collections import defaultdict

import numpy as np
import pytest

from research.levels import s3_episode_factory as ef
from research.levels.s3_view_builder import ViewConfig, build_slot
from research.levels.tests.s3_view_fixtures import NARROW, digest, make_corpus, make_sampler

pytestmark = pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"), reason="ring is x86-64 only")

SEED = 20261123
FIXTURES = "research.levels.tests.s3_view_fixtures"
RAW = ViewConfig(emit="raw")
NATIVE = dict(NARROW, multilabel_view_prob=0.0)  # DSE7 has no multilabel task


def _have_ffi() -> bool:
    """False only when the library is not built; a stale build (ABI mismatch) fails the tests."""
    try:
        ef.ffi_path()
        return True
    except ef.FactoryError:
        return False


needs_ffi = pytest.mark.skipif(not _have_ffi(), reason="build engine/dope-ffi first (cargo build -p dope-ffi)")


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


def _spec(stage: str = "", view=None, **sampler):
    return ef.FactorySpec(seed=SEED, corpus=f"{FIXTURES}:make_corpus", sampler=f"{FIXTURES}:make_sampler",
                          sampler_kwargs=sampler or dict(NARROW), view=view, stage=stage)


def _payload_digest(payload) -> str:
    ep = payload["episode"]
    if "packet" in ep:
        h = hashlib.blake2b(ep["packet"], digest_size=16)
        h.update(np.ascontiguousarray(ep["floats"]).tobytes())
        h.update(repr((ep["split_sig"], ep["view_sig"], ep["table_id"])).encode())
        return h.hexdigest()
    return digest(ep)


def _run(spec, steps, producers, batch=24, capacity=8 << 20):
    with ef.EpisodeFactory(spec, batch, producers, capacity, ahead_steps=2, end_step=steps) as fac:
        out = []
        for step in range(steps):
            for item in fac.batch(step, timeout=120):
                assert not isinstance(item, ef.EpisodeFailed), str(item)
                out.append(item)
        stats = fac.stats()
    return out, stats


# -- ring -----------------------------------------------------------------------------------
def test_ring_handoff_order_and_errors():
    ring = ef.EpisodeRing.create(n_slots=4, capacity=64, unit0=8, slots_per_step=2, n_producers=2)
    try:
        other = ef.EpisodeRing.attach(ring.name)
        assert (other.n_slots, other.unit0, other.slots_per_step) == (4, 8, 2)
        for u in range(8, 12):
            assert ring.put(u, pickle.dumps({"u": u}), 0, 1, u % 2)
        # unit 12 reuses slot 0; its producer blocks until unit 8 is consumed
        done = threading.Event()
        threading.Thread(target=lambda: (ring.put(12, pickle.dumps({"u": 12}), 0, 1, 0), done.set()),
                         daemon=True).start()
        time.sleep(0.05)
        assert not done.is_set()
        assert other.get(4, 1)["u"] == 9  # out of order inside the window is fine
        assert other.get(4, 0)["u"] == 8
        assert done.wait(5)
        with pytest.raises(ef.FactoryError, match="unit_already_consumed"):
            other.get_raw(8, timeout=1)
        with pytest.raises(ef.FactoryError, match="ring_capacity"):
            ring.put(13, b"x" * 65, 0, 1, 1)
        assert ring.put(13, pickle.dumps({"error": "build:ViewError:slot_never_built"}), ef.FLAG_ERROR, 1, 1)
        with pytest.raises(ef.EpisodeFailed, match="slot_never_built"):
            other.get(6, 1)
        with pytest.raises(ef.FactoryError, match="ring_timeout"):
            other.get_raw(14, timeout=0.05)
        ring.stop()
        with pytest.raises(ef.FactoryError, match="ring_stopped"):
            other.get_raw(14)
        assert ring.put(14, b"", 0, 1, 0) is False
        other.close()
    finally:
        ring.close()
    assert not (ef.SHM_DIR / ring.name).exists()


def test_ring_refuses_bad_shapes():
    with pytest.raises(ef.FactoryError, match="ring_slots_not_multiple_of_producers"):
        ef.EpisodeRing.create(n_slots=5, capacity=64, unit0=0, slots_per_step=1, n_producers=2)
    with pytest.raises(ef.FactoryError, match="ring_name"):
        ef.EpisodeRing.attach("../etc/passwd")


# -- factory --------------------------------------------------------------------------------
def test_payloads_are_independent_of_producer_count(corpus):
    one, _ = _run(_spec(), steps=3, producers=1)
    three, stats = _run(_spec(), steps=3, producers=3)
    assert [_payload_digest(p) for p in one] == [_payload_digest(p) for p in three]
    assert stats["built"] >= 72 and stats["producers"] == 3
    # Same bytes as building in this process: the factory adds nothing and caches nothing.
    sampler = make_sampler(corpus, **NARROW)
    for p in three[:24]:
        spec = sampler.plan_step(SEED, p["step"])[p["slot"]]
        episode, _, _ = build_slot(corpus, sampler, SEED, spec)
        assert digest(episode) == _payload_digest(p)


def test_every_reuse_of_a_table_gets_a_fresh_split(corpus):
    sampler = make_sampler(corpus, **NARROW)
    splits = defaultdict(set)
    seen = 0
    for step in range(12):
        for spec in sampler.plan_step(SEED, step):
            episode, used, _ = build_slot(corpus, sampler, SEED, spec)
            if used.kind != "real":
                continue
            key = (episode["table_id"], episode["split_sig"])
            assert key not in {(t, s) for t, ss in splits.items() for s in ss}, "split reused"
            splits[episode["table_id"]].add(episode["split_sig"])
            seen += 1
    assert seen > 0 and any(len(s) > 1 for s in splits.values()), "fixture never reused a table"


def test_raw_emit_keeps_source_values_and_every_draw(corpus):
    sampler = make_sampler(corpus, **NARROW)
    plain = dict(jitter_prob=0.0, warp_prob=0.0, miss_prob=0.0)
    for spec in sampler.plan_step(SEED, 0)[:12]:
        scaled, used, _ = build_slot(corpus, sampler, SEED, spec, ViewConfig(**plain))
        raw, used_raw, _ = build_slot(corpus, sampler, SEED, spec, ViewConfig(emit="raw", **plain))
        assert used == used_raw
        for key in ("y_fit", "y_qry", "fit_rows", "qry_rows", "col_source", "split_sig", "miss_fit"):
            assert np.array_equal(np.asarray(scaled[key]), np.asarray(raw[key])), key
        real = raw["col_source"] >= 0
        if used.kind == "real":
            src = corpus.table(raw["table_id"])["X"]
            want = src[np.ix_(raw["fit_rows"], raw["col_source"][real])].astype(np.float32)
            assert np.array_equal(raw["X_fit"][:, real], want, equal_nan=True)
        # synthetic columns are drawn from the scaled view in both modes
        assert np.array_equal(raw["X_fit"][:, ~real], scaled["X_fit"][:, ~real])
    with pytest.raises(ValueError, match="emit"):
        build_slot(corpus, sampler, SEED, sampler.plan_step(SEED, 0)[0], ViewConfig(emit="bogus"))


# -- native DSE7 stage ----------------------------------------------------------------------
@needs_ffi
def test_native_packet_uses_fit_rows_only(corpus):
    sampler = make_sampler(corpus, **NATIVE)
    checked = 0
    for spec in sampler.plan_step(SEED, 1):
        episode, _, _ = build_slot(corpus, sampler, SEED, spec, RAW)
        a = ef.native_dse7(episode)
        assert len(a["packet"]) <= 2048
        poisoned = dict(episode, X_qry=episode["X_qry"][::-1] * 7.0 + 3.0, y_qry=episode["y_qry"][::-1])
        b = ef.native_dse7(poisoned)
        assert a["packet"] == b["packet"]
        va, vb = ef.split_native(a), ef.split_native(b)
        for name in ("col_feats", "glob", "knots", "loadings", "w_base", "x_fit", "y_fit", "col_index"):
            assert np.array_equal(va[name], vb[name]), name
        p_eff = int(va["misc"][1])
        assert 0 < p_eff <= 48 and np.all(va["col_index"][:p_eff] >= 0)
        assert np.array_equal(va["col_meta"], vb["col_meta"])  # FIT-only meta
        assert va["col_meta"].shape == (48, ef.META_FIELDS) and not va["col_meta"][p_eff:].any()
        assert set(np.unique(va["col_meta"][:p_eff, 4])) <= {0.0, 1.0, 2.0, 3.0, 4.0}
        assert va["x_fit"].shape == (a["n_fit"], 48) and va["y_qry"].shape == (a["n_qry"],)
        checked += 1
    assert checked == 24


@needs_ffi
def test_native_factory_matches_in_process_stage(corpus):
    spec = _spec(stage="research.levels.s3_episode_factory:native_dse7", view=RAW, **NATIVE)
    cap = ef.native_capacity(4096, 1024)
    got, stats = _run(spec, steps=2, producers=2, capacity=cap)
    sampler = make_sampler(corpus, **NATIVE)
    for p in got:
        s = sampler.plan_step(SEED, p["step"])[p["slot"]]
        episode, _, _ = build_slot(corpus, sampler, SEED, s, RAW)
        want = ef.native_dse7(episode)
        assert want["packet"] == p["episode"]["packet"]
        assert np.array_equal(want["floats"], p["episode"]["floats"])
    assert stats["stage_share"] is not None and stats["ms_per_episode"] > 0


@needs_ffi
def test_native_stage_counts_unsupported_tasks():
    with pytest.raises(ef.StageError, match="task_unsupported"):
        ef.dse7_arrays(np.zeros((4, 2)), np.zeros(4), np.zeros((0, 2)), np.zeros(0), "multilabel")
    with pytest.raises(ef.StageError, match="ffi_fit"):
        ef.dse7_arrays(np.random.default_rng(0).normal(size=(20, 3)), np.ones(20), np.zeros((0, 3)),
                       np.zeros(0), "binary")


@needs_ffi
def test_column_meta_of_raw_fit_rows(corpus):
    sampler = make_sampler(corpus, **NATIVE)
    episode, _, _ = build_slot(corpus, sampler, SEED, sampler.plan_step(SEED, 0)[0], RAW)
    meta = ef.column_meta(episode["X_fit"])
    assert meta.shape == (episode["X_fit"].shape[1], ef.META_FIELDS)
    assert np.array_equal(meta, ef.column_meta(episode["X_fit"][::-1]))  # row order never matters
    assert np.isfinite(meta).all()
