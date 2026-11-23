# Counter sampler and honest counters — interface contract (M3, stage C1)

Code: `research/levels/s3_sampler_index.py` (index view, hash primitives),
`research/levels/s3_counter_sampler.py` (plan, shard, micro-batches),
`research/levels/s3_honest_counters.py` (sketches, counters, ledger).
Test: `research/levels/tests/test_s3_counter_sampler.py`. numpy only; importing
these modules never imports torch.

## 1. What is fixed by `(seed, optimizer_step)`

`plan_step(seed, optimizer_step, cfg)` returns `cfg.batch_size` `EpisodeSpec`s
in slot order. `cfg` is a `SamplerConfig` that names its arena (`arena`,
optional `families`) or a `CounterSampler`. The plan reads `index.jsonl` (and
`families.jsonl`) only. It holds no RNG object, cursor or clock and does not
know the rank or the worker count, so resume needs no sampler state: a restart
at step `t` plans step `t`. Every random choice is a blake2b key stretched by
splitmix64; permutations are the sort order of hashed keys, not a numpy RNG.
The only floating-point library calls are `exp`/`log` in the size law, and
their result is snapped to the row grid.

## 2. Strata, quotas, exposure cap

| Stratum | Families it walks | Default quota |
|---|---|---|
| `regression`, `binary`, `multiclass` | families whose majority task it is (tie: the rarer task) | 0.45 / 0.24 / 0.09 |
| `target_swap` | every family with a table of at least 4 columns | 0.12 |
| `scm_prior` | none (each table is new) | 0.10 |

Integer slots per step: largest remainder of `quota * batch_size`, then
`Q_k <= F_k // 2`, then the exposure cap `Q_k * F_reg <= 2 * Q_reg * F_k`
(exact integer form). Slots taken back from a task become `scm_prior` slots of
that task. Slot blocks are contiguous in the order of the table, so a slot
number identifies its stratum for a fixed config.

Family = `family_id` of `families.jsonl` when given (a train table missing from
it is refused with `families_missing_table`); otherwise `family_stem`, with the
generic stems one family per table.

## 3. The walk

Slot `q` of step `t` in stratum `k` reads stream position `u = t * Q_k + q`:
epoch `u // F_k`, index `u % F_k` of that epoch's hashed permutation. Each
family is visited exactly once per epoch; the table inside a family is
`members[(epoch + h(family)) % size]`. Two local repairs, both pure:

- **Epoch boundary.** Positions of an epoch that share a step with the next
  epoch never hold a family of that epoch's head. The head depends only on its
  own epoch, so there is no dependency chain. Always succeeds (`F >= 2Q - 1`).
- **Target swap.** A swap position whose family is used by a real slot of the
  same step trades places with another position of the same epoch (the next
  one, in stream order, where both families are free).

If no trade exists (only when a real family is present in almost every step,
i.e. a corpus far smaller than the batch) the position is *voided*: the slot
keeps `stratum = "target_swap"` and becomes `kind = "scm_prior"`. Zero repeats
hold in every case; exact swap coverage holds whenever nothing is voided.

## 4. `EpisodeSpec` — what the view builder must honour

| Field | Meaning |
|---|---|
| `step`, `slot`, `attempt` | identity; `view_seed = key64("view", seed, step, slot, attempt)` |
| `kind` | `real`, `target_swap` or `scm_prior` |
| `table_id`, `family_ix`, `family_id` | arena table and its family; `-1`, `-1`, `scm:<seed hex>` for a prior table |
| `task`, `n_classes` | `real`: the table's task, or `multilabel` with `n_classes = L` labels to synthesise. `target_swap`: the *requested* target kind (`n_classes` is an upper bound for multiclass); the builder picks a matching column or falls back and reports the actual task. `scm_prior`: exact |
| `n_fit`, `n_qry` | exact row counts, on the grid (multiples of 8 below 32, of 32 up to 512, of 128 above) |
| `split_ratio` | nominal FIT share of the table's rows; `n_fit <= max(16, int(ratio * rows))`, `n_fit + n_qry <= rows` |
| `p_view` | real columns to keep: all, or `view_columns(view_seed, n_candidates, p_view)` when the table is wider than 64 |
| `n_noise`, `n_derived` | synthetic column counts; `p_total = p_view + n_noise + n_derived <= 80` |

Derived: `c_out` (1 regression/binary, 8 multilabel or K <= 8, else 32),
`bucket_key = (c_out, ceil8(p_total), n_fit, n_qry)`, `cells = (3 * n_fit +
n_qry) * ceil8(p_total)` (`n_syn = 2 * n_fit`).

Size law: `n_fit` 40 % U[256, 512], 40 % log-uniform [32, 2048], 20 %
U[2048, 4096]; `n_qry` U[128, 1024]; both capped by what the half can supply.
A prior table is synthesised at exactly `n_fit + n_qry` rows.

If a view cannot be built (too few rows after regrouping duplicates, one class
in FIT, no swappable column) the builder calls `sampler.retry(seed, spec)`:
same table and a new seed, and from the second retry on half the row supply
per attempt; after `max_attempts`
(4) the slot becomes a prior table of the same task (for a target swap: of
the last requested task). Labels are never invented.

## 5. Workers and micro-batches

`shard(specs, worker, n_workers)` deals episodes by padded cells (largest
first, least-loaded worker); loads differ by at most one episode and the union
over workers is the plan for every `n_workers`.

`micro_batches(specs, cell_budget=524288, row_slack=0.0)` groups by exact
bucket key and fills while `B * cells <= cell_budget`. Measured on s3-v250 the
exact key yields 1.02 episodes per micro-batch at batch 64, 1.15 at 448 and
1.30 at 1,024, because tables differ in rows and columns. `row_slack = 0.25`
merges neighbouring row sizes of one `(c_out, ceil8(P))` while padded cells
stay within 1.25 of true cells: 1.6 and 2.7 episodes per micro-batch at 64 and
448, with 7–10 % padded cells. Collate needs row masks to use it.

## 6. Counters

| Call | Who | When |
|---|---|---|
| `observe_plan(specs)` | leader | once per global step, whole plan; returns (repeated tables, repeated families) |
| `observe_built(spec, cols, fit_rows, qry_rows, target_col)` | builder | per built episode, with the rows and columns actually used |
| `observe_failed(spec)` | builder | per failed attempt |
| `commit(step, specs)` | leader | after the optimizer step; a step commits once, in increasing order |
| `merge(other)` | leader | distinct sets take the union, counts and visits add |
| `to_arrays()` / `from_arrays()` | checkpoint | nine plain numpy arrays |

Distinct views and splits are HyperLogLog sketches (p = 14, 16 KiB, standard
error 0.8 %) over content hashes, so two ranks building the same views do not
raise them while `built` doubles. Hash sequential ids with `hash64` before
sketching: a weak hash of consecutive integers tripled the error in testing.

## 7. Measured on `s3-v250` (stem families, 2,000 steps, batch 64, seed 0)

The arena is not in the repository, so these claims are an evidence command,
not a unit test: `python3 -m research.levels.s3_honest_counters
/home/ubuntu/dope-data/flat/s3-v250` re-measures 300 steps and fails if a
repeat, the exposure cap, the `n_fit >= 256` share or the planning time
misses (2026-11-23: 0 repeats, exposure 1.79 / 1.95, share 0.585, 18 µs).

| Quantity | Before (audit) | Now |
|---|---|---|
| Steps with a repeated table / family | about 21 % | 0 / 0 |
| Train tables / families touched | not measured | 4,145 of 4,145 / 3,743 of 3,743 |
| Exposure, multiclass : regression (audit per table, now per family) | 6.3–11.4 | 1.79 (1.67 counting target-swap visits) |
| Exposure, binary : regression (audit per table, now per family) | 3.0–4.4 | 1.95 (1.82 counting target-swap visits) |
| `n_fit` median | 82 | 288 |
| Share `n_fit >= 256` | 4.6 % at >= 384 | 58 % (80 % where the table has the rows); 39 % at >= 384 |
| Planning time | — | 18 µs per episode; 7.5 ms cold start at any step |

Slots at batch 64: regression 29, binary 14, multiclass 5, target swap 8,
prior 8 (the cap moved one binary and one multiclass slot to priors).
