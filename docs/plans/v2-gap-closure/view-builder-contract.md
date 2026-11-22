# Vectorised view builder and episode batch — interface contract (M3, stage C2)

Code (numpy only; importing these modules never imports torch):

| Module | Holds |
|---|---|
| `research/levels/s3_view_corpus.py` | `ViewCorpus` protocol, `ArenaCorpus` (memory map of the flat arena, formats.md §1), `MemoryCorpus`, `write_flat_arena` |
| `research/levels/s3_view_rows.py` | row hash, group split, prior-preserving allocation, `plan_rows`, `signatures` |
| `research/levels/s3_view_scale.py` | FIT order statistics, lattice test, raw augmentation, `scale_stacked` / `scale_fit_only` |
| `research/levels/s3_view_sources.py` | `swap_target`, `scm_prior_table` |
| `research/levels/s3_view_builder.py` | `ViewConfig`, `build_episode`, `build_slot`, `RetryEpisode` |
| `research/levels/s3_batch_episode.py` | `bucket_key`, `collate`, `uncollate`, `EpisodeBatch` |
| `research/levels/s3_view_bench.py` | single-core benchmark and view audit (`python3 -m research.levels.s3_view_bench`) |

Tests: `research/levels/tests/test_s3_view_scale.py`, `test_s3_view_builder.py`,
`test_s3_batch_episode.py` (fixtures in `s3_view_fixtures.py`).

## 1. Calls

```python
corpus  = ArenaCorpus("/home/ubuntu/dope-data/flat/s3-v250")       # read-only, shared pages
sampler = sampler_for(SamplerConfig(batch_size=64, arena=...))      # stage C1
for spec in shard(sampler.plan_step(seed, step), worker, n_workers):
    episode, used_spec, failed_reasons = build_slot(corpus, sampler, seed, spec, cfg, counters)
batch = collate([episodes of one micro-batch])                      # numpy; torch.from_numpy in the trainer
```

`build_episode(corpus, spec, cfg)` is a pure function of its arguments. It returns the episode
dict or `RetryEpisode(reason, spec)`; `build_slot` runs `sampler.retry` until a view is built
and reports failed attempts and the built view to an optional `HonestCounters`.

Row and column choices are orders of splitmix64 keys (no numpy RNG), so they are the same under
any numpy version. Values (jitter, warps, missingness, synthetic columns, prior tables, class
and column permutations) come from `numpy.random.Generator(PCG64(view_seed))`: record the numpy
version in the run identity, as the full-state checkpoint already plans to.

## 2. Order of operations

1. **Source.** Stored table; target-swap view (`swap_target`: a column of the requested kind is
   preferred, another kind is accepted only if its output width fits the planned bucket,
   otherwise `no_swap_column`; 2 values → binary, 3–12 → multiclass, more → regression); or a
   synthetic prior table (`scm_prior_table`, shapes and task fixed by the spec).
2. **Columns.** `view_columns(view_seed, n_candidates, p_view)` of stage C1.
3. **Rule A-1 on the view.** Rows equal on the selected columns (rounded to 1e-5) are one group.
   `dup_key="row"` adds the target to the key (see §6).
4. **Split and subsample.** Groups are ordered by the hashed key of their first row; the cut
   starts at `int(split_ratio * rows)` and moves to the nearest group boundary that leaves
   `n_fit` and `n_qry` rows. Classification: the FIT half is drawn by largest-remainder
   allocation, so every class count is the floor or ceiling of its quota (prior moves by less
   than `1 / n_fit`); spare seats go first to classes that would otherwise vanish. QUERY rows
   are a plain hashed sample of the rows whose class is present in the FIT half.
5. **Raw augmentation** (before the scaler, as in `s3_fast_pipeline`): jitter with probability
   0.30 per episode, sigma 0.2–2.5 % of the FIT IQR, continuous columns only; power and heaping
   warps (probability 0.25 per episode, 0.35 per continuous column; the affine rescale is
   dropped, the heaping step is 5–25 % of the FIT IQR); MCAR / MAR missingness (probability
   0.30). A column is a lattice when its first 256 FIT rows hold at most 64 distinct values or
   sit on a regular grid of at most 4,096 steps; lattices are left bit-identical.
6. **FIT-only scaler.** `nc2_compat` (default) equals `_fit_only_nc2_scale`; `robust_v2` zeroes
   FIT-constant columns in both halves, flags them, and uses the IQR scale whenever IQR > 0.
7. **Synthetic columns** (`spec.n_noise`, `spec.n_derived`), then one column permutation
   applied to values, missing masks and every per-column mask.
8. **Targets.** Regression: FIT mean / std, clip at 6. Binary: 0/1, flipped with probability
   0.5 (as `build_unique_episode`). Multiclass: classes of the FIT half, permuted ids,
   `n_classes = K_fit`. Multilabel: `spec.n_classes` synthetic labels from non-distractor
   columns.

## 3. Episode dict

Every key of `build_unique_episode`, same dtypes: `dataset_hash`, `family_stem`
(= `spec.family_id`), `source_kind` (`s3_real` / `s3_target_swap` / `scm_prior`), `task`,
`n_classes` (1 regression, 2 binary, K, L), `stage_id` (0), `X_fit`, `X_qry`, `miss_fit`,
`miss_qry` (float32, rows × P), `y_fit`, `y_qry` (float32; rows × L for multilabel),
`distractor_mask` (bool P, noise columns only), `split_sig` (16 hex digits), `aug_stats`.

Added: `view_sig` (int, equal to `s3_honest_counters.view_hash`; `int(split_sig, 16)` equals
`split_hash`), `kind`, `table_id`, `step`, `slot`, `attempt`, `target_col` (−1 unless swapped),
`cols`, `fit_rows`, `qry_rows` (source indices; what `HonestCounters.observe_built` takes),
`col_source` (int64 P: source column, −1 noise, −2 derived), `constant_mask` (bool P).
`aug_stats` adds `fit_pool`, `qry_pool`, `dup_rows`, `qry_dropped`, `const_cols`, `flipped`,
`jitter`, `jitter_cols`, `lattice_cols` (−1 when not probed), `warp_cols`, `miss_cols`, and for
classification `pool_counts`, `fit_counts` (by source class id).

Row counts are exactly `spec.n_fit` and `spec.n_qry`. The actual task of a target swap may
differ from the requested one, never with a wider output than `spec.c_out`.

## 4. Retry reasons (static codes)

`split_groups` (no group boundary fits the sizes, or too few usable rows), `one_class_fit`,
`query_rows` (too few QUERY rows of classes seen in FIT), `constant_target`,
`no_swap_column`. Labels are never invented. Spec/corpus mismatches raise `ViewError`.

## 5. Batch layout

`collate(episodes, n_fit=None, n_qry=None, p=None, c=None) -> EpisodeBatch`; defaults pad to
the largest sizes in the list, columns to a multiple of 8, outputs to 1 / 8 / 32.

| Array | Shape | Meaning |
|---|---|---|
| `X_fit`, `miss_fit` / `X_qry`, `miss_qry` | (B, Nf, P) / (B, Nq, P) float32 | zero padded |
| `Y_fit` / `Y_qry` | (B, Nf, C) / (B, Nq, C) float32 | regression: column 0 value; binary: column 0 in {0,1}; multiclass: one-hot K; multilabel: L columns |
| `row_fit`, `row_qry` | (B, Nf), (B, Nq) bool | real rows |
| `col`, `dmask`, `const_mask` | (B, P) bool | columns of the episode; noise columns; FIT-constant columns |
| `out_mask` | (B, C) bool | target columns in use |
| `task_id`, `n_out`, `n_classes`, `n_fit`, `n_qry`, `p_true` | (B,) | `TASK_IDS` = regression 0, binary 1, multiclass 2, multilabel 3 |
| `view_sig`, `split_sig` | (B,) uint64 | for the HyperLogLog sketches |

`meta[b]` keeps every other key; `uncollate(collate(x)) == x` key for key. Episodes of one
exact bucket need no row padding; `micro_batches(row_slack=0.25)` groups collate with masks.

## 6. Measured on `s3-v250` (seed 0, batch 64, 40 steps = 2,560 slots)

Reproduce with `python3 -B -m research.levels.s3_view_bench --steps 40 --baseline-steps 10`.
The view audit is deterministic and was identical in every run; the timings are not. Timing
is one thread, CPU time (`thread_time_ns`), best of 3 per episode, on a host at load 69–84 of
128 threads with the SMT sibling busy. Four runs are given as a range. The three runs on the
committed tree, at load 80–84, are the slow end (495–534 and 666–699 episodes / s); the fast
end is one run at load 69 taken two small edits earlier (about 5 µs per episode). A fixed
numpy kernel, a sort of 448 × 10 doubles, took 134–166 µs across the runs. The existing
builders slowed by 7 % between the quiet and the busy runs and the new builder by 25 %: it
spends its time in memory-bound numpy passes, not in the interpreter.

| View audit | Existing pipeline | This stage |
|---|---|---|
| Views with a QUERY feature row that also occurs in FIT | 4.2 % (audit) | 0 of 2,240; 16.8 % with `dup_key="row"` |
| FIT class prior against its pool | 8.4 % minority became 19 % (audit) | within one row for every class (largest drift 0.95 rows) |
| One-class FIT half | random labels invented | retry; 27 failed attempts for 2,560 slots (`split_groups` 15, `no_swap_column` 10, `one_class_fit` 1, `query_rows` 1) |
| Jitter on lattice columns | every column with more than 16 values | none; 68 % of probed columns are lattices |
| Views with a duplicate row group | not measured | 813 of 2,240 |
| Views with a FIT-constant column | 5.9 % (audit) | 6.8 % (still ±6 spikes under `nc2_compat`; zero under `robust_v2`) |

| Episodes per second, one core | Existing builders | This stage |
|---|---|---|
| Planned mix (all kinds) | — | 495–665 (mean 1.50–2.02 ms, median 0.99–1.27 ms) |
| Median-size episode (9,760 cells ± 25 %) | — | 666–868 (mean 1.15–1.50 ms, median 1.01–1.29 ms) |
| Real slots at the planned sizes | 206–225 (`sample_dynamic_episode`, 4.45–4.86 ms) | 630–861 (1.16–1.59 ms), 3.1–3.8 times faster in the same run |
| Real slots at each builder's own sizes | 219–242 (`sample_dynamic_episode`, n_fit ≤ 512), 134–147 (`build_unique_episode`, stage 3) | as above |

Cost model of the new builder: 0.51–0.58 ms fixed plus 59–89 ns per cell. Prior tables are
12.5 % of slots and 30–31 % of the time (3.7–4.9 ms each: unlimited row supply makes them
large). **The target of 1,000 episodes per second for the median-size episode is not met on
this host**: the best run reaches it by the median time (1.01 ms) and misses it by the mean
(868 / s); the runs on the committed tree gave 666–699 / s. What is left: the FIT order
statistics (`np.partition`, about 25 ns per FIT cell), the two sorts of the group split, the
sort and grid test of the lattice probe in the 47 % of episodes with jitter or warps, and
about 340 Python-level calls per episode, which is the fixed half millisecond.
