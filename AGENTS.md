# DOPE

DOPE compiles a table with a declared target into a small generative program.
The file contains fitted summaries and schema without any complete fitting records.
`dope sample` draws synthetic rows from the file.

To consume the installed engine, follow [Using DOPE](#using-dope).
[Changing this repository](#changing-this-repository) applies to everyone who edits this source tree.

## Using DOPE

### Install

Follow the [README quick start](README.md#quick-start) to install the signed
Linux x86-64 engine on glibc 2.39 or newer. Its executable is
`$HOME/.local/opt/dope/3.0.0/bin/dope`. The published `dope-engine-v3.0.0`
release ships the analytic engine without SingleEncV8 weights.

For downloaded packages and source builds, see
[Signed engine releases](docs/signed-engine-release.md). The
[own-data walkthrough](docs/getting-started.md#fit-a-table) covers fitting,
sampling and held-out evaluation.

### Fit and sample

Use a headered CSV and name the target. Follow
[the target-last contract](docs/data-contract.md): the target comes from that
declaration, it stays out of the features, and duplicate physical rows stay in
the same split.

```bash
dope="$HOME/.local/opt/dope/3.0.0/bin/dope"
"$dope" fit train.csv --target y --format auto-v8 --budget 2048 -o model.dope
wc -c < model.dope
"$dope" sample model.dope -n 1000 -o synthetic.csv
```

`--format 1` is the CLI default. `--format auto-v8` is the v3.0.0 release encoder.
It fits Format 1 and analytic DSE8 as separate candidates and writes one file
of at most 2048 bytes. When both fit, a duplicate-grouped two-fold check on the
fitting rows can select DSE8.

The command list for this binary is [engine/README.md](engine/README.md).
[docs/cli.md](docs/cli.md) documents a different program, the historical
`dope-kernel` facade (`compile-raw`, `fit-table`, and the alpha installer).

### What you may say about the results

The numbers on the landing page come from the committed v3.0.0 scoreboard.
Read them from [README.md](README.md) and the
[scoreboard receipts](bench/real-world/results/scoreboard/v3.0.0/README.md).
The manuscript is [docs/whitepaper-v3/dope-v3.pdf](docs/whitepaper-v3/dope-v3.pdf).
Earlier manuscripts live under `docs/whitepaper/`, `docs/whitepaper-v2/`, and `paper/`.

When you summarize a result, keep these limits:

- The published tag is an uncertified engine release. The certification gates are in [docs/release.md](docs/release.md).
- The report is a fresh-split DEV-SELECT estimate. No confirmation, holdout, or sealed panel was opened.
- On that report look, the paired difference versus ARF includes zero. DOPE is PROMOTE against TVAE and HOLD against ARF, the Gaussian copula, and CTGAN. The paper reports each development split set independently.
- The file contains no complete fitting records. The privacy measures are empirical diagnostics, with the measured scope in the paper.
- Cross-model feature-driver preservation is not yet measured.
- Synthetic rows from this release remain easy for a boosted-tree classifier to distinguish from real rows.

Treat a `.dope` file and any synthetic CSV as derived data. Schema and fitted summaries can carry information about the input.

## Changing this repository

### Current language policy — 2026-11-03

The latest user instruction is **Python ONLY for deep-learning TRAINING; everything else in DOPE must be Rust**. This supersedes historical V1, advanced-data, benchmark and research-Python exceptions as end-state permission. Python may own actual neural model/loss/autograd/optimizer steps and necessary training-coupled batch/RNG/checkpoint state. Standalone preparation, inference, classical evaluator fitting, validation/metrics/Elo, reports/qualification, CLI/orchestration, CI/checkers/examples and new nontraining tests are Rust. A mixed module or use of a neural model is not an exemption.

**Current status: NOT_COMPLIANT.** The a84 snapshot has 890 tracked Python files. Inventory/classification and seven source-only Rust packets are migration work, not a waiver or completion. Preserve all old source, readers, fixtures, models, results, receipts, custody and test intent until reviewed native replacement coverage exists; no deletion/renaming or guard/exclusion weakening. Do not add new nontraining Python or hide it in a training helper. Read [the migration queue and acceptance gates](docs/plans/next-version/dope/language-policy/README.md) before assigning work. Native/full CI/PR admission and scientific/release gates remain separate.

Follow the host worktree and build rules supplied by the user. Use one writer per
checkout. This repository's release branch is developed in
`.agent/worktrees/integration`; keep generated evidence under `target/` and local
tools under the ignored `.agent/` directory.

- [Architecture](docs/architecture.md) identifies Rust product code, the V1
  Python compatibility boundary, and immutable contract inputs.
- [Testing](docs/testing.md) maps changes to deterministic commands, budgets,
  and repair evidence. Run `just fast` while editing and `just check` before a
  commit.
- [Release](docs/release.md) lists the certification and publication gates.
- [Audit](docs/audit.md) explains the pinned Jankurai evidence and exceptions.
- `agent/owner-map.json` and `agent/test-map.json` route files to owners and
  proof commands. Update them when adding a new source area.

Preserve historical DPK3.2 and v1 receipt decoding. New tier claims require
measured certification; an empirical privacy result is never formal DP or HIPAA
de-identification. Errors and public sidecars must not echo source values,
headers, or extrema.

All new canonical training and evaluation tables must follow
[the mandatory target-last contract](docs/data-contract.md). Resolve the target
from its explicit dataset declaration, normalize external CSVs once with the shared
intake, and reject downstream schema/header drift. Never infer the target merely
from the last source column. Keep query targets out of features and context
selection, and preserve original sources and physical row identities.
