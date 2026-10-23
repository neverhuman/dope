# Development probes: portable levels and landmarks

**Status: development evidence, selection-exposed.** Nine public PMLB tables, chosen while the
designs were being written, one engine build, small held-out files. These probes show
direction and failure modes, not effect sizes. They are outside every summary in
[`../README.md`](../README.md) and support no benchmark, tier, certification or privacy claim.

They are cited by the [multilevel plan](../../../docs/plans/multilevel-implementation.md)
(E0, ML-00) and the [landmark plan](../../../docs/plans/landmark-signatures.md) (§2.2, §5.5a,
Appendix C, LM-00).

## Contents

| File | Contents |
| --- | --- |
| `inputs.json` | PMLB commit, split parameters, and per-table SHA-256 of the downloaded `.tsv.gz`, the decompressed `.tsv` and the `train.csv`/`test.csv` that `dope-probes split` writes |
| `results/levels-probe.jsonl` | Levels probe receipts: 2/4/16/64 KiB independent fits plus one fit at the summed low+med+high bytes, per table; engine release binary of `65b1d2a` |
| `results/landmark-probe.jsonl` | Landmark probe receipts for the same 45 programs: T1 per-landmark transfer, CV signatures, external T2 retention over three sampling seeds |
| `results/landmark-rawlin.jsonl` | Post-hoc linear landmark in unclipped mean/sd scaling (the §5.5a representation check) |
| `results/levels-probe-rust-rerun.jsonl` | The levels probe rerun by the Rust driver with the same engine binary on the Rust split (see below) |
| `results/auditor-repeat.jsonl` | Repeated `dope eval` of two fixed programs on fixed files (engine `65b1d2a`), showing the CPU MLP auditor's run-to-run spread |
| `results/levels-replay.md`, `results/landmark-replay.md`, `results/landmark-diagnostics.md`, `results/levels-rerun-compare.md` | Replay output; CI checks that the receipts still produce exactly these files |

The tool is the Rust crate [`engine/dope-probes`](../../../engine/dope-probes). It is a separate
binary in the engine workspace and is not part of the shipped `dope` executable.

## Replay (no data needed)

```sh
cargo build --manifest-path engine/Cargo.toml --release --locked -p dope-probes
P=engine/target/release/dope-probes
$P replay-levels                     # levels tables, carry policy, chain saving, pooled samples
$P replay-landmarks                  # landmark plan §2.2 tables A–C and per-program detail
$P landmark-diff --table 574_house_16H --from med --to high   # §2.2 D
$P rawlin                            # §5.5a
$P compare-levels bench/real-world/probes/results/levels-probe.jsonl \
  bench/real-world/probes/results/levels-probe-rust-rerun.jsonl
cargo test --manifest-path engine/Cargo.toml --locked -p dope-probes
```

## Rebuild the inputs and rerun the levels probe

```sh
mkdir -p target/probes/raw
base=https://media.githubusercontent.com/media/EpistasisLab/pmlb/7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68/datasets
for t in 197_cpu_act 218_house_8L 225_puma8NH 344_mv 574_house_16H adult car_evaluation churn mushroom; do
  curl -fsSL -o "target/probes/raw/$t.tsv.gz" "$base/$t/$t.tsv.gz"
  gzip -dkf "target/probes/raw/$t.tsv.gz"
done
$P split --raw target/probes/raw --out target/probes/data     # verifies every pinned digest

# 65b1d2a is the head of PR #308 (branch codex/budgets-tiers-elo), not a main commit.
git fetch origin 65b1d2a66fe49a1a077b0bf4c3aba3fb1108ce09
git worktree add target/probes/engine-65b1d2a 65b1d2a66fe49a1a077b0bf4c3aba3fb1108ce09
cargo build --manifest-path target/probes/engine-65b1d2a/engine/Cargo.toml --release --locked
$P levels --dope target/probes/engine-65b1d2a/engine/target/release/dope \
  --data target/probes/data --out target/probes/levels
$P levels-bytes --artifacts target/probes/levels       # program sizes vs the committed receipts
$P compare-levels bench/real-world/probes/results/levels-probe.jsonl target/probes/levels/levels-probe.jsonl
$P replay-levels target/probes/levels/levels-probe.jsonl
```

Everything is written under `target/`; raw data stays out of Git.
Parser failures retain file/line/column coordinates, and failed engine calls
retain their exit status. Their diagnostics omit source strings, declared target
names, artifact metadata and engine stderr, which may contain private input text.
Repair hints still identify the relevant command and documentation.

## What reproduces, and what does not

* **Inputs.** `dope-probes split` ports NumPy's `default_rng(20261022).permutation(n)` bit for
  bit (tested against NumPy) and selects the same rows in the same order as the probes' files.
  It copies source fields verbatim. The probes' files went through pandas, so the four
  classification tables are byte-identical, while regression values differ by at most 8e-13
  relative (pandas' default float parser).
* **Program bytes.** Saved artifacts reproduce the receipts' program, inflated and metadata
  sizes exactly. The marginal chain size `|brotli(A‖B)| − |brotli(A)|` is an estimate that depends
  on the encoder: the receipts used the C reference encoder (Python `brotli` 1.2.1), this tool uses
  the engine's Rust encoder. Standalone sizes differ by at most 0.84% per artifact and the median
  low+med+high chain saving is 36.9% with either encoder. It is not a framed conditional codec.
* **Spearman ties.** The replay uses average ranks. The prototype broke ties in NumPy's
  platform-dependent `argsort` order; six cells of the landmark plan's table B moved by 0.01 and
  the plan now shows the replayed values.
* **Landmark learners.** The landmark prototype used scikit-learn trees, kNN and a small MLP
  plus NumPy ridge bases. Those learners cannot be reproduced bit for bit in Rust, so the
  prototype is not ported; its receipts are kept as recorded. The native Rust panel
  (`dope landmarks compare`, see [`docs/landmarks.md`](../../../docs/landmarks.md)) replaces it,
  and rerunning the selection question on that panel is experiment E-L1.
* **Retention rerun.** See the next section.

## Rust rerun of the levels probe

`results/levels-probe-rust-rerun.jsonl` is the levels probe rerun by `dope-probes levels` with
the same engine binary (`65b1d2a`, release build) on the inputs from `dope-probes split`.
`dope-probes compare-levels` writes `results/levels-rerun-compare.md`:

* **Fits reproduce when the inputs do.** 25 of 36 level programs are byte-identical,
  including all 16 on the four classification tables, whose inputs are byte-identical. The other
  11 are on regression tables, where the inputs differ from the probe's pandas-written files in
  the last digits; those tiny differences change some fits (`218_house_8L` and `344_mv`
  entirely).
* **External retention is not deterministic.** On identical programs, robust retention still
  moved by up to 0.10 (`574_house_16H` med: 0.955, then 0.852). `dope eval`'s MLP auditor is
  not seeded on CPU: `engine/dope-core/src/eval/nn.rs` calls `dev.set_seed(..).ok()`, and
  candle-core 0.9's CPU device rejects `set_seed`, so weight initialization comes from the
  thread RNG. `results/auditor-repeat.jsonl` records repeated evaluations of one fixed program on
  fixed files with 1, 2 (default) and 4 threads. The linear and GBDT auditors agree to the last
  digit, while the MLP auditor ranges 0.944–0.977 on `197_cpu_act` low (seven runs) and
  0.888–0.909 on `574_house_16H` med (three runs; the original receipt's evaluation of the same
  program and files gave 1.069). The multiclass linear auditor uses the same path, and so does
  the fit-time self-audit's linear family on multiclass tables; `car_evaluation`'s self-audit
  values were nevertheless identical across three runs.
* **What survives the rerun.** Pooled low/med/high samples beat the 64 KiB program on 8/9 tables
  and the best single level on 5/9, and the median chain saving is 37% (14–58%), as before.
  Medians of the independent levels moved from 0.960 / 0.943 / 0.859 / 0.874 to
  0.950 / 0.868 / 0.859 / 0.914. The carry-forward policy (margin 0.005) had no drop above 0.02
  in the original run and two in the rerun. The policy chooses by the deterministic self-audit,
  so the change comes from the programs that changed and from the noisy external evaluation.

The retention cells of both probes, including the landmark probe's external T2 retention, carry
this auditor noise. Until the CPU MLP auditor is seeded, treat per-cell differences below about
0.1 on the regression tables as unresolved, and rerun E0 and E-L1 after that fix.
