# DOPE fleet dashboard

Read-only monitoring for DOPE training runs. Nothing here has product, promotion
or certification authority.

## Run it

```bash
# Campaign aggregator (Rust): rebuilds the macro file every 60 s.
cargo build --manifest-path engine/Cargo.toml --locked -p dope-arena --bin dope_fleet_macro
engine/target/debug/dope_fleet_macro \
  --root "$DOPE_DATA_DIR/fleet-mirror" \
  --checkpoint-kpi "$DOPE_DATA_DIR/dashboard/checkpoint_kpi.jsonl" \
  --corpus-index "$DOPE_DATA_DIR/flat/s3-v250/index.jsonl" \
  --out "$DOPE_DATA_DIR/dashboard/fleet_macro.json" --watch 60

# Live per-node probes plus the campaign view (defaults to --hosts localhost).
python3 -m tools.dope_dashboard.server --port 8768 \
  --hosts localhost \
  --macro-json "$DOPE_DATA_DIR/dashboard/fleet_macro.json"
```

`--root` is repeatable. A run is any directory holding `train.jsonl`, up to four
levels below a root; `<host>/runs/<run>/` names the host. The aggregator only
reads; run directories, the guard directory and the index are never written.

## Two views

- **Whole campaign · all nodes** (`/api/macro`, `dope.fleet-macro/v1`): every
  mirrored run on one clock in 5-minute buckets. It shows:
  - fleet tables per second (now, last hour, peak), stacked by host;
  - tables trained, cumulative;
  - active runs and hosts;
  - median training loss relative to each run's own start;
  - checkpoint files saved;
  - scored checkpoints against the untrained step-0 model;
  - topology (which runs share weights);
  - data uniqueness;
  - a per-run roll-up.

  When the fleet reads 0 tables/s, the ASR guard levels and its last event are
  shown beside the number.
- **Live nodes** (`/ws`, `/api/snapshot`): SSH probes of each host's newest runs.
  Its "In-run retention" is the trainer's own DEV evaluation, not the release
  metric (`dope_kpi` TRG).

## How the numbers are defined

- **Tables per second** are tables completed: increments of each trainer's
  cumulative counter (`episodes_seen`, or `unique_splits_seen` in older runs),
  summed per bucket and divided by the
  bucket width. Runs without the counter fall back to the logged rate times the
  elapsed time, with each gap capped at 600 s.
  - "Now" covers the last 300 s.
  - The trainer-reported rate is shown separately. It is higher, because each
    trainer restarts its rate window after evaluation pauses.
- **Episodes, not unique splits.** The trainer counter counts training
  episodes; it is not deduplicated. Runs that do not write `exposure.jsonl` do
  not log which dataset an episode used, so for them distinct datasets and
  distinct splits read "not logged".
  - Real-table episodes are estimated from `n_synth_prior` where it is logged,
    and from the V7 sampler's 20% synthetic share otherwise. Each run is
    labelled with which basis was used.
  - Uses per training table = real-table episodes ÷ training tables in the
    corpus index.
- **Exact uniqueness.** A run that writes `exposure.jsonl` gets exact counts:
  distinct datasets, distinct splits, split repeats and within-step dataset
  duplicates, per run and fleet-wide. Each line is one optimizer step:
  `{"step": int, "time": float, "datasets": [dataset_hash, ...], "splits": [split_hash, ...]}`.
- **Weight sharing.** Each run is its own model.
  - A run with `world_size > 1` is a data-parallel model: its ranks average
    gradients, and only rank 0 logs.
  - A run whose `sync_mode` (in `config.json` or `train.jsonl`) is anything
    other than `independent_ws1` or `ddp_gloo` exchanges weights with other
    runs, for example `diloco_outer_momentum`. Runs are grouped by the
    `sync_dir` (or `sync_group`) their `config.json` records, and each group
    shows its highest `sync_round`. If the directory is not recorded, all runs
    with the same mode are shown as one group.
  - A run whose `config.json` names a source checkpoint (`init_from`,
    `resume_from`, `warm_start`, ...) is listed as warm-started.
  - Anything else shares nothing with other runs.
- **Learning.** Each run's loss is divided by the median of its first rows: at
  most 20 rows, and at most half the run. The chart shows the median and the
  middle half over the runs active in each bucket.
- **Checkpoint quality** comes from `--checkpoint-kpi`, one scored checkpoint
  per line: `{"run", "step", "variant", "metric", "value", "panel"}`.
  - Each checkpoint is placed at the time of its run's logged row at that step.
  - The chart marks 0 (as good as the untrained step-0 model) and the noise
    floor `--kpi-noise` (default 0.01).
  - Checkpoint file times more than an hour outside their run's logged span are
    counted but not placed on the clock.
