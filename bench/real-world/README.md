# Real-world benchmark

How much predictive structure does each method keep per byte, on public tables, for
several model families? This directory holds the complete, resumable protocol.

## Protocol

- **Tables.** The confirmation panel is sixteen public tables (UCI and PMLB): seven
  classification (adult, bank, credit, german, car, mushroom, covertype) and nine regression
  (california, abalone, wine, bike, concrete, airfoil, energy, student, news), mixed
  numeric and categorical, with missing values where the source has them.
  [`datasets.py`](datasets.py) downloads each archive, pins its digest in
  [`pins.json`](pins.json), and writes an 80/20 train/test split (stratified for
  classification) with at most 20,000 training and 10,000 test rows. A separate
  development panel of PMLB regression tables was used while building the engine. The
  confirmation panel is not sealed: an earlier engine was scored on adult, airfoil and
  abalone before the forest and self-audit were added, so those three tables are
  selection-exposed (the paper's Limitations section says so too). The compact artifact
  container (one brotli stream for schema and codes) and the regrowth of trees into the
  compressed slack were added after a first full run of this panel, whose DOPE cells are kept
  in `results/superseded/`; they change how a program is stored and how many trees fit a
  budget, not how anything is fitted, and every DOPE cell was re-run with them.
- **Methods.** DOPE at byte budgets 1–64 KiB; the repository's research kernel
  (`dope-kernel fit-table` and `sample-table` at their defaults; it accepts regression and
  binary targets, so it refuses car and fits covertype's class codes as a regression, and both
  count as measured); *real rows* — the largest random subset of
  training rows whose CSV fits the same budget after `xz -9e` (the information-compression
  control), plus all rows; SDV Gaussian copula, ARF, and (on the tables with at most 3,400
  training rows) CTGAN and TVAE at author defaults; real rows resampled (a row-count
  control). Ablations: DOPE with independent features, one regime, the network without trees, a
  linear target without trees, a fixed byte split without the self-audit, and four times as
  many synthetic rows.
- **Limits.** Every fit-and-sample call runs in a child process with a 30-minute wall-clock
  limit (`GEN_TIMEOUT` in `run.py`); a call over the limit is a failed cell (`Timeout`), and
  so are the remaining seeds of that method on that table.
- **Bytes.** DOPE: the safetensors file as written (its program is one brotli stream). The
  kernel: its artifact file as written. Real rows: the xz stream. Generators: the xz-compressed pickle of the fitted object, which favours
  them.
- **Auditors.** Three families trained only on the method's rows, scored on the held-out
  real test rows: ridge/logistic regression, CatBoost (500 trees, depth 6), and an MLP
  (two layers of 128; Adam with early stopping, or L-BFGS below 2,000 rows). One-hot
  encoding and scaling are fitted on each auditor's own training table.
- **Score.** Retention `R = (L_null − L_syn)/(L_null − L_real)` with MSE in original
  target units or log-loss; a family counts when the real data improves on the null by
  at least one percent. *Robust retention* is the worst counted family. Synthetic tables
  have as many rows as the real training split unless stated.
- **Statistics.** Per table, the median over seeds (three at the headline budget, one
  elsewhere). Across tables, the median with a 95% bootstrap interval (10,000 draws over
  tables); paired DOPE-minus-method differences as Hodges–Lehmann estimates with
  bootstrap intervals and win/tie/loss counts. TDL at τ is the smallest tested artifact
  whose robust retention reaches τ.

## Run

```bash
cargo build --release --manifest-path engine/Cargo.toml
pip install pandas scikit-learn catboost sdv arfpy xlrd openpyxl matplotlib
python3 bench/real-world/datasets.py --panel confirm --out target/bench-data
python3 bench/real-world/run.py --data target/bench-data --panel confirm --out bench/real-world/results/confirm
python3 bench/real-world/run.py --data target/bench-data --panel confirm --out bench/real-world/results/confirm \
    --methods ctgan,tvae --datasets abalone,airfoil,car,concrete,energy,german,student
cargo build --release --bin dope-kernel                                         # the research kernel
python3 bench/real-world/run.py --data target/bench-data --panel confirm --out bench/real-world/results/confirm \
    --methods kernel
python3 bench/real-world/report.py --results bench/real-world/results/confirm --data target/bench-data/confirm
python3 bench/real-world/render.py --results bench/real-world/results/confirm   # figures, README block, paper macros
python3 tools/claims/dope1.py --write                                           # splice into README.md
```

`run.py` appends one JSON line per cell to `records.jsonl` and skips cells already
present, so an interrupted run resumes where it stopped. Failed fits are recorded with
their error class and stay in every denominator as robust retention 0 (a learner given no
rows can only predict the null).

## Provenance

The DOPE cells in `results/confirm/` were produced by a release build of `engine/` (binary
SHA-256 `e582bbeff5b9a663ff8468a0e0d64163e439c7b5d2e246737f659f3fdf3edb7d`), the kernel cells by a
release build of `dope-kernel` at commit `36cf658a` (binary SHA-256
`66e3639ede4f391bc6fdb28be880e19544ccd4f6bfd92adfb9fe297d073a2107`). ARF on bike was re-run after fixing a harness bug (arfpy reserves a
column named `cnt`, the bike target; the wrapper now passes positional column names). Runs were
split into resumable chunks on a two-core VM; `report.py` keeps the first record of any cell.

## Files

| File | Contents |
| --- | --- |
| `results/confirm/records.jsonl` | every cell: bytes, timings, per-family losses and retention, DOPE fit report |
| `results/confirm/reference.jsonl` | real-trained and null losses per table, seed and family |
| `results/confirm/summary.json` | medians, intervals, contrasts, TDL per table |
| `results/superseded/engine-rc1-records.jsonl` | the first full run of every DOPE cell, before the compact container (schema and tensor table stored as plain safetensors header JSON); superseded, disclosed, not used in any summary |
| `results/selection-exposure/engine-v2-records.jsonl` | the earlier engine's records on adult, airfoil and abalone, seen before the forest and self-audit were designed (disclosed, not used in any summary) |
| `results/throughput.json` | in-memory sampling throughput (`throughput.py`) |
| `results/readme-block.md`, `figures/*.png` | generated README surfaces (hash-listed in `results/render-manifest.json`) |
| `../../paper/generated/dope1-*` | generated paper macros and figures |
