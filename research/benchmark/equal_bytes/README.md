# Equal-byte study: a DOPE program against the real rows

Evidence status: development validation, selection-exposed. Nothing here opens
the sealed test split, and no result is a qualified TDL headline.

## Question

For a byte budget `B`, which representation of a table lets learners that never
see the table learn the most from it: a compiled DOPE program of at most `B`
bytes, or the largest set of real rows that fits in `B` bytes under the best of
several lossless codecs? The answer traces the transfer-per-byte frontier of
both representations, which is what TDL measures.

## Design

| Item | Choice |
| --- | --- |
| Tables | Historical mixed/unknown-origin remainder of the public PMLB panel (`ops/ci/paper-inputs/pmlb-sources.json`), after the original name-based exclusions and three content duplicates: 26 tables. The mask retains simulated pollen; it is not an origin classifier. `--all` adds the simulated lineages (97 unique). |
| Split | Shuffle with seed 1729; test = 30% of rows, capped at 2,000; fit = up to 2,000 of the rest. Min-max scaling with fit-row statistics. |
| Learners | The v2 evaluator's: OLS, one-hidden-layer MLP (32 units, early stopping), CatBoost (100 iterations, depth 6), seed 1729. |
| DOPE arm | Ladder L1 to L5 of packed programs with the native trainer, three fit seeds (11, 23, 37), `4n` synthetic rows. L4 matches the published headline shape and is also compiled with the historical codec to record the before/after bytes. |
| Real-row arm | For each budget (the 128 B to 64 KiB grid plus each DOPE rung's own median bytes, so equal bytes are exact), the largest uniform subset of fit rows whose 8- or 16-bit codes, row-sorted and column-major, fit under the best of LZMA2-extreme, bzip2-9 and raw deflate-9; three subset seeds. The whole table scores as the reference. |
| Matched training | The MLP has an epoch budget (`max_iter=200`), so its optimizer steps scale with the rows it sees. In the real-row arm and in the real reference it keeps its better test loss of training on the rows as given or resampled with replacement to the DOPE arm's synthetic size (4n). Linear regression and CatBoost (fixed rounds) are trained as given. This choice favors the real rows; `summary.json` also reports the protocol without it (`as_given`) and without the MLP (`without_mlp`). |
| Charge | Both arms pay 8 bytes per column for the fit-row scaling. The historical charge (`projection.json`) is recorded beside it. |
| Aggregation | Retention per learner against the matched reference, clipped to [0, 1]; median over seeds in both arms; worst informative learner per table; frontier = best configuration within the budget; real-row bytes to match a level interpolated in log bytes between bracketing budgets; percentile bootstrap over tables (10,000 draws, seed 1729). |
| Scaling | Four large tables; nested fit views of 250 to 16,000 rows against a fixed 2,000-row test set; program bytes against whole-table bytes. |
| Copula ablation | The headline rung with the exact Gaussian copula (G4) on the 26 historical-cohort tables, paired with L4 by table. |

## Source-origin correction

The [pinned PMLB metadata](../../../docs/evidence/pmlb-pollen-origin-20261021.json)
explicitly describes `529_pollen` as simulated. Its absence from the old name mask
did not make it measured real-world data. The current labels are corrected without
changing cohort membership, raw results or predeclarations. Other members are not
automatically classified as real. `historical_remainder_entries` reproduces that
mask; its old `real_world_entries` alias exists only for compatibility. Neither is
a rights or origin gate for the new measured-source pilot.

## Protocol history

The committed records follow protocol `equal-bytes-v3` (the `protocol` field).
An independent review of the first version found that the real-row arm was
handicapped. Its MLP's epoch budget starved small subsets of optimizer steps,
and the same budget undertrained the real reference. Budgets matched only on
the power-of-two grid, and the bytes real rows needed were rounded up to the
grid. Version 2 fixed those. Version 3 also early-stops the resampled MLP on
held-out distinct rows, so its validation never sees a duplicate of a training
row. The `as_given` sensitivity in `summary.json` shows the real rows trained as
given.

The re-runs reused each table's DOPE rows and the real-row losses measured as
given (`run.py --reuse`); only the step-matched MLP fits and the exact-budget
cells were recomputed. The DOPE rows therefore predate the `is_packed` field
(`null`). For the headline rung, packing is confirmed by comparing each fit's
`artifact_bytes` with its `historical_codec_bytes`: packed is 1.6 to 2.0 times
smaller on every fit. A fresh run (below) records `is_packed` from the artifact
header.

## Reproduce

```bash
cargo build --release --bin dope-kernel
python3 -B -m research.benchmark.equal_bytes.prepare --sources target/pmlb --out target/eb
python3 -B -m research.benchmark.equal_bytes.run --binary target/release/dope-kernel --out target/eb-results target/eb/*
python3 -B -m research.benchmark.equal_bytes.summarize target/eb-results --out research/benchmark/equal_bytes/results/summary.json
python3 -B -m research.benchmark.equal_bytes.scaling --binary target/release/dope-kernel --sources target/pmlb \
  --out research/benchmark/equal_bytes/results/scaling 537_houses 225_puma8NH 218_house_8L 227_cpu_small
# Exact Gaussian copula ablation (rung G4) on the 26 historical-cohort tables
python3 -B -m research.benchmark.equal_bytes.run --binary target/release/dope-kernel --arms dope --rungs G4 \
  --out research/benchmark/equal_bytes/results/ablations/gaussian-copula target/eb/*
# Full panel: the 71 simulated lineages on a two-rung ladder, summarized with the 26 historical-cohort records
python3 -B -m research.benchmark.equal_bytes.prepare --sources target/pmlb --out target/eb-all --all
python3 -B -m research.benchmark.equal_bytes.run --binary target/release/dope-kernel --arms dope,data --rungs L2,L4 \
  --out research/benchmark/equal_bytes/results/panel $(comm -23 <(ls target/eb-all) <(ls target/eb) | sed 's|^|target/eb-all/|')
python3 -B -m research.benchmark.equal_bytes.summarize research/benchmark/equal_bytes/results/lineages \
  research/benchmark/equal_bytes/results/panel --rungs L2,L4 --out research/benchmark/equal_bytes/results/panel-summary.json
python3 docs/whitepaper/scripts/equal_bytes_render.py --write-readme  # figures, paper macros and tables, HISTORY.md block
python3 docs/whitepaper/scripts/equal_bytes_render.py --check  # text outputs must match the committed files
```

On two CPU cores the 26-table study takes about 40 minutes, the scaling study
about 20 minutes and the 71 simulated panel lineages about 2.5 hours. Results
were produced with the binary built from this branch; per-table records are in
`results/lineages/` (26 historical-cohort tables, full ladder) and `results/panel/`
(71 simulated lineages, rungs L2 and L4), the scaling records in
`results/scaling/`, and the summaries in `results/summary.json` and
`results/panel-summary.json`.

## Files

| Path | Role |
| --- | --- |
| `prepare.py` | fetch (SHA-pinned) and split the lineages |
| `run.py` | DOPE ladder, CPU default and real-row arms for each lineage |
| `summarize.py` | KPIs: equal-byte gap, data-equivalence ratio, TDL wins, bootstrap intervals, two-part proxy |
| `scaling.py` | program and data bytes against the number of fit rows |
| `results/` | committed outputs that the renderer reads |
| `test_equal_bytes.py` | unit tests for the codec arm, frontier and TDL logic |
