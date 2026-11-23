"""Counter sampler and honest counters (plan M3, stage C1): replay, zero repeats, exposure, sizes, sketches."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from research.levels.s3_counter_sampler import (
    GENERIC_STEMS, STRATA, TASKS, CorpusIndex, CounterSampler, SamplerConfig, SamplerError, micro_batches,
    plan_digest, plan_step, sampler_for, shard, snap_down, view_columns,
)
from research.levels.s3_honest_counters import (
    ExposureLedger, HonestCounters, HyperLogLog, hash64, measure_plan, split_hash, step_repeats,
)
from research.levels.s3_sampler_index import key64

LANE = Path(__file__).resolve().parents[3]
STEPS, BATCH, SEED = 2000, 64, 20261122
GRID = set(range(8, 32, 8)) | set(range(32, 513, 32)) | set(range(640, 4097, 128))


def _rows(n_singletons: int = 1150, seed: int = 11) -> list[dict]:
    """Synthetic index: about 73 / 20 / 7 % tasks, families of 60 and 55 tables, generic stems, non-train rows."""
    rng = random.Random(seed)
    plan: list[tuple[str, str, str]] = []  # (stem, task, split)
    plan += [("lineage_big_a", "regression", "train")] * 60 + [("lineage_big_b", "binary", "train")] * 55
    for k in range(6):  # ten target variants of one source table, mixed tasks
        plan += [(f"openml_src{k}_target", t, "train") for t in ["regression"] * 7 + ["binary"] * 2 + ["multiclass"]]
    plan += [("rows", "regression", "train")] * 40 + [("export", "regression", "train")] * 30
    for k in range(n_singletons):
        plan.append((f"stem{k}", rng.choices(TASKS, (0.73, 0.20, 0.07))[0], "train"))
    for k in range(140):
        plan.append((f"valstem{k}" if k % 4 else "lineage_big_a", rng.choice(TASKS), "val"))
    rng.shuffle(plan)
    rows, x_off, row_off = [], 0, 0
    for t, (stem, task, split) in enumerate(plan):
        n_rows = rng.choice([25, 40, 83, 150, 284, 600, 1000, 1000, 1000, 3000, 8000, 8192])
        n_cols = rng.choice([1, 2, 3, 5, 8, 10, 16, 23, 40, 90, 217, 256])
        n_classes = {"regression": 0, "binary": 2}.get(task, rng.choice([3, 4, 5, 7, 10, 20]))
        rows.append({"table_id": t, "dataset_hash": f"{t:016x}", "canonical_sha256": f"{t:064x}", "family_stem": stem,
                     "task": task, "n_classes": n_classes, "n_rows": n_rows, "n_cols": n_cols, "split": split,
                     "x_off": x_off, "row_off": row_off})
        x_off, row_off = x_off + n_rows * n_cols, row_off + n_rows
    rows[0].update(split="train", task="multiclass", n_classes=40)  # above the class limit: never sampled
    rows[1].update(split="train", n_rows=12)  # too few rows for any split: never sampled
    return rows


@pytest.fixture(scope="module")
def arena():
    with tempfile.TemporaryDirectory(prefix="dope-c1-") as tmp:
        rows = _rows()
        with open(Path(tmp) / "index.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n" for rec in rows)
        yield Path(tmp), rows


@pytest.fixture(scope="module")
def run(arena):
    """2,000 consecutive global steps at batch 64, planned once and shared by the measuring tests."""
    sampler = sampler_for(SamplerConfig(batch_size=BATCH, arena=str(arena[0])))
    return sampler, [sampler.plan_step(SEED, step) for step in range(STEPS)]


def test_index_reads_train_tables_only_and_groups_families(arena):
    root, rows = arena
    index = CorpusIndex.load(root)
    train = [r for r in rows if r["split"] == "train" and r["n_classes"] <= 32 and r["n_rows"] >= 24]
    assert sorted(index.eligible.tolist()) == [r["table_id"] for r in train]
    assert index.table_family[0] == -1 and index.table_family[1] == -1
    generic = [r for r in train if r["family_stem"] in GENERIC_STEMS]
    assert len(generic) == 70 and len({index.family_ids[index.table_family[r["table_id"]]] for r in generic}) == 70
    big = {int(index.table_family[r["table_id"]]) for r in train if r["family_stem"] == "lineage_big_a"}
    assert len(big) == 1 and int((index.table_family == big.pop()).sum()) == 60  # val rows with that stem stay out
    # A lineage file replaces the stem rule; a train table missing from it is refused by code, not by value.
    lineage = root / "families.jsonl"
    with open(lineage, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"dataset_hash": r["dataset_hash"], "family_id": f"lf-{r['table_id'] // 4}"}) + "\n"
                      for r in rows)
    grouped = CorpusIndex.load(root, lineage)
    assert grouped.n_families < index.n_families and max(np.bincount(grouped.table_family[grouped.eligible])) <= 4
    with open(lineage, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"dataset_hash": rows[5]["dataset_hash"], "family_id": "lf-0"}) + "\n")
    with pytest.raises(SamplerError, match="families_missing_table"):
        CorpusIndex.load(root, lineage)
    lineage.unlink()


def test_replay_is_identical_across_objects_order_processes_and_worker_counts(arena, run):
    sampler, plans = run
    cfg = sampler.cfg
    digest = plan_digest(spec for step in range(40) for spec in plans[step])
    fresh = CounterSampler(CorpusIndex.load(arena[0]), cfg)  # no shared memo
    assert plan_digest(s for step in range(40) for s in fresh.plan_step(SEED, step)) == digest
    cold = CounterSampler(CorpusIndex.load(arena[0]), cfg)  # random access: a late step first
    for step in (STEPS - 1, 1234, 0, 777):
        assert cold.plan_step(SEED, step) == plans[step] == plan_step(SEED, step, cfg)
    assert plan_digest(plan_step(SEED + 1, 0, cfg)) != plan_digest(plans[0])
    code = (
        "import sys\n"
        "from research.levels.s3_counter_sampler import SamplerConfig, plan_digest, plan_step\n"
        "from research.levels import s3_honest_counters\n"
        f"cfg = SamplerConfig(batch_size={BATCH}, arena={str(arena[0])!r})\n"
        f"print(plan_digest(s for step in range(40) for s in plan_step({SEED}, step, cfg)), 'torch' in sys.modules)\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "4242"}
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=LANE, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == [digest, "False"]  # same stream in a fresh interpreter; torch never imported
    for step in (0, 1, 999):
        for n_workers in (1, 2, 7):
            shards = [shard(plans[step], w, n_workers) for w in range(n_workers)]
            assert sorted((s for part in shards for s in part), key=lambda s: s.slot) == plans[step]
            loads = [sum(s.cells for s in part) for part in shards]
            assert max(loads) - min(loads) <= max(s.cells for s in plans[step])
    with pytest.raises(SamplerError):
        shard(plans[0], 2, 2)


def test_no_table_or_family_repeats_within_any_of_2000_steps(run):
    sampler, plans = run
    counters = HonestCounters(sampler.index.n_tables, sampler.index.n_families)
    for step, specs in enumerate(plans):
        assert len(specs) == BATCH and [s.slot for s in specs] == list(range(BATCH)) and {s.step for s in specs} == {step}
        tables = [s.table_id for s in specs if s.table_id >= 0]
        assert len(set(tables)) == len(tables) and len({s.family_id for s in specs}) == BATCH
        assert counters.observe_plan(specs) == (0, 0)
        for s in specs:
            if s.table_id >= 0:
                assert sampler.index.task[s.table_id] >= 0 and s.family_ix == sampler.index.table_family[s.table_id]
                assert s.kind == ("target_swap" if s.stratum == "target_swap" else "real")  # no voided slot here
    assert counters.count["repeated_tables"] == 0 and counters.count["repeated_families"] == 0
    assert counters.count["planned"] == STEPS * BATCH and counters.count["steps_planned"] == STEPS
    doubled = plans[0][:-1] + [plans[0][0]]  # the detector does fire on a real repeat
    assert step_repeats(doubled) == (1, 1)


def test_every_family_of_every_stratum_is_visited_exactly_once_per_epoch(run):
    sampler, plans = run
    for s in sampler.strata:
        assert s.quota > 0 and s.size >= 2 * s.quota
        seq = [spec.family_ix for specs in plans for spec in specs[s.slot0: s.slot0 + s.quota]]
        epochs = len(seq) // s.size
        assert epochs >= 4
        for e in range(epochs):
            assert sorted(seq[e * s.size: (e + 1) * s.size]) == s.fams.tolist(), (s.name, e)
        assert np.array_equal(sampler.stream(SEED, s.name, 0, epochs * s.size), np.asarray(seq[: epochs * s.size]))
    # Inside a family the table rotates by epoch: 60 tables are all seen in 60 visits, none twice.
    big = max(range(sampler.index.n_families), key=lambda f: int((sampler.index.table_family == f).sum()))
    visits = [s.table_id for specs in plans for s in specs if s.kind == "real" and s.family_ix == big]
    assert len(visits) >= 60 and len(set(visits[:60])) == 60 and visits[60] == visits[0]


def test_exposure_cap_holds_and_binds(arena, run):
    sampler, plans = run
    ledger = ExposureLedger(sampler.index.n_tables, sampler.index.n_families)
    for step, specs in enumerate(plans):
        ledger.commit(step, specs)
    exposure = ledger.exposure(sampler.family_home)
    assert 1.0 < exposure["multiclass_to_regression"] <= 2.0 and 1.0 < exposure["binary_to_regression"] <= 2.0
    assert exposure["multiclass_to_regression_all"] <= 2.0 and exposure["binary_to_regression_all"] <= 2.0
    reg, binary, multi, swap = sampler.strata
    for s in (binary, multi, swap):  # the integer form of the cap, exact
        assert s.quota * reg.size <= 2 * reg.quota * s.size
    assert (reg.quota, swap.quota) == (29, 8)  # largest remainder of 0.45 / 0.24 / 0.09 / 0.12 / 0.10 at 64
    assert multi.quota == 2 * 29 * multi.size // reg.size < 6  # ... gives multiclass 6; the cap takes one back
    assert sampler.prior_tasks.count("multiclass") == 6 - multi.quota  # ... and refills with multiclass priors
    assert sampler.prior_tasks.count("binary") == 15 - binary.quota and sampler.prior_tasks.count(None) == 6
    assert sum(s.quota for s in sampler.strata) + len(sampler.prior_tasks) == BATCH
    loose = CounterSampler(sampler.index, SamplerConfig(batch_size=BATCH, exposure_cap=50.0))
    assert [s.quota for s in loose.strata] == [29, 15, 6, 8] and loose.prior_tasks == [None] * 6
    with pytest.raises(SamplerError, match="ledger_step_order"):
        ledger.commit(STEPS - 1, plans[-1])  # a step commits once
    assert sum(ledger.histogram("table").values()) == sampler.index.n_tables
    assert int(ledger.table_visits.sum()) + ledger.prior_visits == STEPS * BATCH


def test_size_law_is_eval_matched_capped_and_on_the_bucket_grid(run):
    sampler, plans = run
    index, cfg = sampler.index, sampler.cfg
    able, prior_fit = [], []
    for spec in (s for specs in plans[:600] for s in specs):
        assert spec.n_fit in GRID and spec.n_qry in GRID and spec.n_fit >= 16 and spec.n_qry >= 8
        assert cfg.split_ratio_lo <= spec.split_ratio <= cfg.split_ratio_hi or spec.kind == "scm_prior"
        assert spec.view_seed == key64("view", SEED, spec.step, spec.slot, 0) and spec.attempt == 0
        assert spec.p_total <= cfg.max_total_cols and 0 <= spec.n_noise <= 6 and 0 <= spec.n_derived <= 4
        if spec.kind == "scm_prior":
            prior_fit.append(spec.n_fit)
            assert spec.table_id == -1 and spec.family_id.startswith("scm:") and 4 <= spec.p_view <= 35
            able.append(spec.n_fit >= 256)
            continue
        rows, cols = int(index.n_rows[spec.table_id]), int(index.n_cols[spec.table_id])
        fit_rows = min(max(int(spec.split_ratio * rows), 16), rows - 8)
        assert spec.n_fit <= fit_rows and spec.n_qry <= rows - fit_rows  # never more than a half can supply
        assert spec.p_view == min(cols - (spec.kind == "target_swap"), 64) and (spec.kind != "target_swap" or cols >= 4)
        if spec.kind == "real":
            table_task = TASKS[int(index.task[spec.table_id])]
            assert spec.task in (table_task, "multilabel") and (spec.task != "multilabel" or table_task == "regression")
            if spec.task != "multilabel":
                assert spec.n_classes == int(index.n_classes[spec.table_id])
        if snap_down(fit_rows) >= 256:
            able.append(spec.n_fit >= 256)
    assert len(able) > 10000 and np.mean(able) >= 0.40  # the law gives 0.80 when the table has the rows
    assert 0.74 <= np.mean(able) <= 0.86
    prior_fit = np.asarray(prior_fit)
    # Mixture shares on uncapped (prior) tables: 0.20 large; 0.40 + 0.40 * (one octave of six, plus snapping) mid.
    assert abs(np.mean(prior_fit >= 2048) - 0.203) < 0.03 and abs(np.mean((prior_fit >= 256) & (prior_fit <= 512)) - 0.485) < 0.03
    wide = next(s for specs in plans for s in specs if s.kind == "real" and index.n_cols[s.table_id] > 64)
    cols = view_columns(wide.view_seed, int(index.n_cols[wide.table_id]), wide.p_view)
    assert cols.size == 64 and np.all(np.diff(cols) > 0) and cols[-1] < index.n_cols[wide.table_id]
    assert np.array_equal(cols, view_columns(wide.view_seed, int(index.n_cols[wide.table_id]), wide.p_view))
    assert np.array_equal(view_columns(1, 5, 64), np.arange(5))


def test_micro_batches_partition_a_step_into_buckets_within_the_cell_budget(run):
    _, plans = run
    for specs in plans[:50]:
        exact = micro_batches(specs)
        assert sorted((s for mb in exact for s in mb), key=lambda s: s.slot) == specs
        assert all(len({s.bucket_key for s in mb}) == 1 for mb in exact)
        assert all(len(mb) == 1 or len(mb) * mb[0].cells <= 524288 for mb in exact)
        loose = micro_batches(specs, row_slack=0.25)
        assert sorted((s for mb in loose for s in mb), key=lambda s: s.slot) == specs and len(loose) <= len(exact)
        for mb in loose:
            padded = len(mb) * (3 * max(s.n_fit for s in mb) + max(s.n_qry for s in mb)) * mb[0].bucket_key[1]
            assert len({s.bucket_key[:2] for s in mb}) == 1
            assert len(mb) == 1 or (padded <= 524288 and padded <= 1.25 * sum(s.cells for s in mb))
    same = [plans[0][3]] * 100  # one bucket, many episodes: split by the budget alone
    assert [len(mb) for mb in micro_batches(same, cell_budget=10 * same[0].cells)] == [10] * 10
    assert plans[0][3].c_out in (1, 8, 32) and plans[0][3].bucket_key[1] % 8 == 0


def test_retry_keeps_the_table_shrinks_the_supply_and_ends_in_a_prior_table(run):
    sampler, plans = run
    for spec in plans[17][::3]:
        chain = [spec]
        for _ in range(5):
            chain.append(sampler.retry(SEED, chain[-1]))
        assert chain[1:] == [sampler.retry(SEED, s) for s in chain[:-1]]  # pure
        assert [s.attempt for s in chain] == list(range(6)) and len({s.view_seed for s in chain}) == 6
        assert {(s.step, s.slot, s.stratum) for s in chain} == {(spec.step, spec.slot, spec.stratum)}
        assert all(s.view_seed == key64("view", SEED, s.step, s.slot, s.attempt) for s in chain)
        if spec.kind == "scm_prior":
            assert all(s.kind == "scm_prior" for s in chain)
            continue
        rows = int(sampler.index.n_rows[spec.table_id])
        for s in chain[1:4]:
            assert (s.kind, s.table_id, s.family_id) == (spec.kind, spec.table_id, spec.family_id)
            shrink = s.attempt - 1  # retry 1 redraws at full supply; retries 2 and 3 ask for a half and a quarter
            assert s.n_fit <= max(16, int(0.78 * rows) >> shrink) and s.n_qry <= max(8, (rows - 16) >> shrink)
        assert all(s.kind == "scm_prior" and s.table_id == -1 for s in chain[4:])
        hint = "regression" if chain[3].task == "multilabel" else chain[3].task  # the last request is kept
        assert chain[4].task == hint or (hint, chain[4].task) == ("regression", "multilabel")


def test_tiny_and_skewed_corpora_never_repeat_even_when_slots_are_voided():
    rng, voided = random.Random(5), 0
    for trial in range(30):
        n = rng.choice([8, 16, 40, 120])
        rows = [{"table_id": t, "dataset_hash": f"h{t}", "family_stem": f"f{rng.randrange(max(2, n // rng.choice([1, 2, 5])))}",
                 "task": rng.choices(TASKS, (0.73, 0.2, 0.07))[0], "n_classes": 3, "n_rows": rng.choice([25, 90, 1000]),
                 "n_cols": rng.randint(1, 30), "split": "train"} for t in range(n)]
        quotas = tuple(rng.random() + 0.05 for _ in STRATA)
        sampler = CounterSampler(CorpusIndex(rows), SamplerConfig(batch_size=rng.choice([1, 3, 8, 32]), quotas=quotas))
        plans = [sampler.plan_step(trial, step) for step in range(80)]
        for specs in plans:
            assert len(specs) == sampler.cfg.batch_size and step_repeats(specs) == (0, 0)
            voided += sum(s.stratum != "scm_prior" and s.kind == "scm_prior" for s in specs)
        for s in sampler.strata[:3]:  # real strata keep exact coverage whatever the corpus
            seq = [spec.family_ix for specs in plans for spec in specs[s.slot0: s.slot0 + s.quota]]
            for e in range(len(seq) // s.size if s.quota else 0):
                assert sorted(seq[e * s.size: (e + 1) * s.size]) == s.fams.tolist()
    assert voided > 0  # the degenerate branch (a family present in every step) was exercised
    with pytest.raises(SamplerError):
        CounterSampler(CorpusIndex(rows), SamplerConfig(min_fit_rows=20))
    with pytest.raises(SamplerError):
        CorpusIndex([dict(rows[0], split="val")])


def test_hyperloglog_is_within_two_and_a_half_percent_and_merges_like_a_union():
    for n in (100_000, 1_000_000):
        sketch = HyperLogLog(14)
        sketch.add_many(hash64(np.arange(n, dtype=np.uint64)))
        assert abs(sketch.estimate() / n - 1.0) <= 0.025, (n, sketch.estimate())
    a, b, union = HyperLogLog(), HyperLogLog(), HyperLogLog()
    left, right = hash64(np.arange(0, 60_000)), hash64(np.arange(40_000, 130_000))
    a.add_many(left)
    b.add_many(right)
    union.add_many(np.concatenate([left, right]))
    before = a.estimate()
    a.add_many(left)  # adding the same items again changes nothing
    assert a.estimate() == before
    a.merge(b)
    assert np.array_equal(a.reg, union.reg) and abs(a.estimate() / 130_000 - 1.0) <= 0.025
    one_by_one = HyperLogLog()
    for h in left[:3000].tolist():
        one_by_one.add(h)
    vector = HyperLogLog()
    vector.add_many(left[:3000])
    assert np.array_equal(one_by_one.reg, vector.reg) and abs(vector.estimate() / 3000 - 1.0) <= 0.025
    assert HyperLogLog().estimate() == 0.0
    with pytest.raises(SamplerError):
        HyperLogLog(9)


def _build(counters: HonestCounters, sampler: CounterSampler, specs) -> None:
    """Stand-in for the view builder: rows and columns are a function of the spec alone."""
    for spec in specs:
        rows = int(sampler.index.n_rows[spec.table_id]) if spec.table_id >= 0 else spec.n_fit + spec.n_qry
        order = np.random.default_rng(spec.view_seed).permutation(rows)
        cols = view_columns(spec.view_seed, spec.p_view + 3, spec.p_view)
        counters.observe_built(spec, cols, order[: spec.n_fit], order[spec.n_fit: spec.n_fit + spec.n_qry])


def test_counters_measure_distinct_views_merge_and_round_trip(run):
    sampler, plans = run
    index, steps = sampler.index, 30
    new = lambda: HonestCounters(index.n_tables, index.n_families)  # noqa: E731
    rank0, rank1, other, single = new(), new(), new(), new()
    for step in range(steps):
        single.observe_plan(plans[step])
        _build(single, sampler, plans[step])
        single.commit(step, plans[step])
        _build(rank0, sampler, plans[step])
        _build(rank1, sampler, plans[step])  # a second rank with the same seed builds the same views
        _build(other, sampler, sampler.plan_step(SEED + 1, step))
    built = steps * BATCH
    base = rank0.summary()
    assert abs(base["distinct_views"] / built - 1.0) <= 0.025 and abs(base["distinct_splits"] / built - 1.0) <= 0.025
    real = [s for step in range(steps) for s in plans[step] if s.table_id >= 0]
    assert base["distinct_tables"] == len({s.table_id for s in real}) < len(real)  # tables do recur across steps
    assert base["distinct_families"] == len({s.family_ix for s in real})
    rank0.merge(rank1)
    same = rank0.summary()
    assert same["built"] == 2 * built  # the naive counter doubles ...
    for name in ("distinct_views", "distinct_splits", "distinct_tables", "distinct_families"):
        assert same[name] == base[name]  # ... the measured distinct counts do not move
    rank0.merge(other)
    assert rank0.summary()["distinct_views"] > 1.9 * base["distinct_views"]
    spec = plans[0][0]
    assert split_hash(spec, [3, 1, 2], [9, 8]) == split_hash(spec, [1, 2, 3], [8, 9]) != split_hash(spec, [1, 2], [3, 8, 9])
    # Workers that build shards and a leader that plans and commits add up to the single-process counters.
    leader, workers = new(), [new() for _ in range(3)]
    for step in range(steps):
        leader.observe_plan(plans[step])
        for w, counters in enumerate(workers):
            _build(counters, sampler, shard(plans[step], w, 3))
        leader.commit(step, plans[step])
    for counters in workers:
        leader.merge(counters)
    buffer = io.BytesIO()
    np.savez(buffer, **leader.to_arrays())
    buffer.seek(0)
    with np.load(buffer) as stored:
        restored = HonestCounters.from_arrays({name: stored[name] for name in stored.files})
    for got in (leader, restored):
        assert got.summary() == single.summary() and got.count == single.count
        for name, value in single.to_arrays().items():
            assert np.array_equal(got.to_arrays()[name], value), name
    assert restored.ledger.last_step == steps - 1 and restored.summary()["committed"] == built
    with pytest.raises(SamplerError):
        restored.commit(steps - 1, plans[steps - 1])
    restored.commit(steps, plans[steps])
    restored.observe_failed(plans[steps][0])
    assert restored.summary()["committed_steps"] == steps + 1 and restored.count["failed"] == 1


def test_measured_plan_report_on_the_synthetic_arena(run):
    sampler, _ = run
    report = measure_plan(sampler, SEED, 200)
    assert report["repeated_tables"] == 0 and report["repeated_families"] == 0 and report["voided_slots"] == 0
    assert report["n_fit_share_ge_256_where_table_allows"] >= 0.40 and report["plan_microseconds_per_episode"] < 1000.0
    assert report["tables_touched"] <= report["eligible_tables"] and abs(sum(report["kind_share"].values()) - 1.0) < 1e-3

