# Reproducibility levels

This page separates four reproducibility levels. Each level has its own input,
command, output hash and result, or a precise blocker. A pass at one level does
not imply a pass at another.

The four levels are not interchangeable:

- Aggregate replay (L1) checks that published numbers match committed ledgers.
  It never shows that per-query results, sampler output or compiler output can
  be regenerated.
- A four-level reproduction claim is not supported by aggregate-only replay.
  Papers and READMEs must not use "four-level reproduction" unless L2, L3 and
  L4 each have a recorded pass.

Machine-readable status lives in
`target/e06-reproducibility/reproducibility-levels-v1.json`. That file is build
output for this lane and is regenerated from the commands below.

## Status summary

| Level | What is replayed | Status | Evidence |
| --- | --- | --- | --- |
| L1 | Aggregate tables and figures from committed immutable ledgers | **Pass** (positive) and **negative control detected** | Section L1 |
| L2 | Per-query metrics inside the authorized private boundary | **Blocked** | Section L2 |
| L3 | Isolated sampler replay from a full recipient bundle | **Partial**: native custody metadata verified; sampler replay not run | Section L3 |
| L4 | Compiler replay from admitted fit/select input, recipe and toolchain | **Blocked** | Section L4 |

## Common pins

Every level records these pins in its result. A missing pin is a failure, not
a warning.

- **Source:** git commit of the checkout (`git rev-parse HEAD`) and a clean
  `git status` for tracked files.
- **Model and budget:** the model, seed and compute budget named in the
  campaign lock. Not applicable to L1, which replays stored numbers only.
- **Runtime:** Rust toolchain `1.88.0` from `rust-toolchain.toml`, Python 3
  with `-B` (no bytecode writes), and the CPU and OS recorded by the runner.
- **Hardware tolerance:** L1 outputs are exact text and must match byte for
  byte. L2 and L3 numeric outputs may differ within the tolerance declared in
  their lock. A value outside that tolerance is a failure.

## L1: aggregate replay from immutable ledgers

**Question.** Do the numbers printed in the paper and generated tables match
the committed, hashed ledgers?

**Command.**

```bash
python3 -B research/benchmark/verify_paper_numbers.py
```

**Inputs (sha256).**

| File | sha256 |
| --- | --- |
| `research/benchmark/results/s3-matched-forest-confirmation-validation.json` | `97a25be902954cad16c5b5802d0e0ba468dea104221efa2465434c94e6774a1f` |
| `research/benchmark/results/arf-native-closure-watch-v1.receipt.json` | `7e8eeb1ba94ca747367e98d5a016d2ed3ed4c40c5a98ca76d7a5fe27ce9c173c` |
| `research/benchmark/results/beyondarena-s3-inventory.json` | `504166764631dbdf5e02d116334af5cc587731fcf3efe130d62240e6dd512d64` |
| `research/benchmark/results/s3-data.lock.json` | `857b61324d0a5b7dbbb98e0c147fcc79c5bb8616e7ca61afd6785469c17f038f` |
| `research/benchmark/methods.lock.json` | `899c355cb55769710ed369fea6b082589b31858860fb74929bf4190de28aca33` |
| `docs/whitepaper/generated/loss-curves.json` | `0124915497c98e560ef99c1e5ecf7ad876f9174439ff9c763cf1cf906b4aebe1` |
| `docs/whitepaper/generated/replay-cost.json` | `faf77e7ce1ed1d22127701bf68bbfe2f8d183f89cbd3475dd80094888c36f086` |
| `docs/whitepaper/generated/provenance.json` | `dea64ea61eaacac0e287298bba3ba76aa7900819aa00feb3a5d66cdcb4fb8a17` |
| `research/benchmark/verify_paper_numbers.py` (verifier) | `137f0973b0d17d1a220766cfc383821baefe3d9c381133807fac623db8c2be5e` |

**Positive result.** Exit status 0. Output:
`paper numbers match the committed receipts and ledgers`. Log retained in the
lane at `.agent/e06-l1-verify.log` (ignored by git).

**Negative control.** A scratch copy of the forest confirmation ledger was
changed by adding 1 to `charged_artifact_bytes` on the six
`native_selected`, size-4 Forest-Flow rows. The committed file was not
modified. The verifier, pointed at the copy through its `FOREST_LEDGER`
module constant, reported:

```text
fit-trace.json disagrees with the committed receipts and ledgers
ForestByteLo: manuscript '115{,}202{,}253', evidence '115{,}202{,}254'
ForestByteHi: manuscript '1{,}085{,}155{,}062', evidence '1{,}085{,}155{,}063'
main returned 1
```

So the check fails on a one-byte change in a traced ledger value.

**Limit.** L1 shows that printed numbers agree with stored ledgers. It does not
recompute those ledgers from queries or from samplers.

## L2: private per-query recomputation

**Question.** Can per-query metrics be recomputed from the private query rows
inside the authorized boundary, without changing the protocol?

**Result: blocked.**

- The sealed official test split stays closed. Decision D-08 in
  `docs/decisions.md` has not been made, and `docs/testing.md` states that
  expensive or holdout-touching reruns need a prior receipt.
- No per-query private rows, access grant or executor eligibility record is
  present in this lane. Without them no command can be pinned.
- Executor eligibility is a governance decision. It is not established here,
  and the executor must not be the compiler author. Output exposure and custody
  are recorded separately from the replay result.

**Unblocks when:** a named, authorized executor, a private boundary receipt, and
the per-query input hashes are recorded. The command and output hash are then
pinned in the level's record.

## L3: isolated sampler replay from a recipient bundle

**Question.** Can an isolated executor regenerate samples from the full
recipient bundle?

**Result: partial.** Native custody metadata verifies. Sampler replay has not
been run.

**Command (metadata custody replay, run in this lane).**

```bash
python3 -B -c "from research.benchmark import density_custody_replay as m; print(m.replay_native_files())"
```

It reads the scratch inventory under `/mnt/fast-scratch/dope-benchmark/`. Its
input declarations:

| File | sha256 |
| --- | --- |
| `/mnt/fast-scratch/dope-benchmark/density-s3-population-native-custody-v1/receipt-lock-v1.json` | `902c68d8ffd4ef7a8ed93539947ddb7496d0c14438f7a4a0c52631d51834f6f5` |
| `/mnt/fast-scratch/dope-benchmark/density-s3-population-native-custody-v1/reconciliation-v1.json` | `efe8f169dbf8e1b7a68b44406ced32d910a2809f09682539facd24c86ac67187` |

**Result (verified fields, as returned).**

- `workers_verified`: 100
- `default_native_cells_verified`: 600
- `distinct_artifacts_verified`: 502
- `native_trial_artifacts_verified`: 1100
- `native_refs_hashed`: 9667

**Not verified, and therefore not claimed:**

- `models_deserialized`: false. Models were not loaded.
- `rows_parsed`: false. No rows were read.
- `declared_runtime_or_import_closure_certified`: false.
- `publication_admitted`: false.
- `official_tests_opened`: false.

**Blocker.** The sampler must run in an isolated process from the recipient
bundle, with its runtime and import closure certified. Those steps are not
done. The command above is therefore not an L3 pass.

## L4: compiler replay from admitted fit/select input

**Question.** Can the compiler regenerate its fitted artifact from the admitted
fit/select input, the exact recipe and the pinned toolchain?

**Result: blocked.**

- Toolchain: Rust `1.88.0` is pinned in `rust-toolchain.toml`. This is the only
  pin verified here.
- Inputs: no admitted fit/select input receipt or exact recipe for a compiler
  run is present in this lane. A fit run is neural, GPU-scale work and is not
  started by this lane.
- Executor: the independent executor requirement is unmet, as in L2.

**Unblocks when:** the admitted fit/select input hash, the recipe file hash, the
toolchain pin and the compute budget are recorded, and an independent executor
runs the compile. Its output hash is then compared with the committed artifact
under the hardware tolerance.

## Rules for claims

- Say "L1 aggregate replay passed" only when the L1 command has exited 0 on
  the cited inputs.
- Do not say "reproduced at four levels" or "independently reproduced" from L1
  alone.
- A metadata or custody replay is not sampler replay, and sampler replay is
  not compiler replay.
- Do not call any level a formal privacy, DP or de-identification result. These
  are reproducibility checks only.

## Files this page depends on

- `research/benchmark/verify_paper_numbers.py`
- `research/benchmark/density_custody_replay.py`
- `research/benchmark/density_native_replay.py`
- `rust-toolchain.toml`
- `docs/testing.md`, `docs/claims-policy.md`, `docs/benchmark.md`
