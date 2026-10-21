# Repository map

DOPE is a Rust compiler and sampler with a large, deliberately frozen body of
research evidence around it. About 95% of the repository's bytes are hash-pinned
benchmark ledgers that the paper build and CI verify by exact path. This page says
what each directory is for, which parts a newcomer needs, and what can and cannot be
moved.

## If you only have ten minutes

| You want to | Read or run |
| --- | --- |
| see what DOPE does on a real table | `python3 examples/quickstart/quickstart.py --dataset adult` |
| understand the idea | the top of the [README](../README.md), then [math.md](math.md) |
| change the compiler | `rust/compiler/`, `rust/model/`, `rust/codec/`, `rust/sample/`; run `just fast` |
| change a published number | never by hand: regenerate it (see "Generated, do not edit" below) |

## Top-level directories

| Directory | Class | What it holds |
| --- | --- | --- |
| `rust/` | product | the `dope-kernel` library and binaries (`dope`, `dope-kernel`, `dope-bench`, `mfs-v3-holdout`): compiler, codec, sampler, metrics, certification, release |
| `examples/` | product examples | `quickstart/` (encoder, evaluation, selection, ablation), `healthcare-sim/`, `public-regression/`, `readme-regression/`; all run in CI |
| `crates/` | product | `dope-service`, built as a binary of the root crate |
| `contracts/`, `schemas/`, `production/` | product contracts | the TDL contract, JSON schemas and goldens, historical KPI contracts read by `build.rs` |
| `tests/` | product tests | Rust integration tests, Python V1 tests, fixtures and cross-language goldens |
| `src/` | compatibility | the Python V1 package (`dope_kernel`), a bounded exception that expires in 2027 |
| `paper/` | publication (active) | the TDL manuscript `main.tex`, its supplement, and `generated/` inputs |
| `docs/` | documentation | product notes, the math, the frozen v2 whitepaper (`docs/whitepaper/`), governance plans, archive |
| `research/benchmark/` | research and evidence | comparator adapters, publishers, and `results/` (frozen ledgers) |
| `research/transfer/` | research, partly compiled | TDL research code; some JSON is compiled into Rust with `include_str!` |
| `tools/claims/` | governance | the claims firewall: renders `STATUS.md` and `paper/generated/`, lints public text |
| `agent/` | governance | Jankurai audit configuration, owner and test maps, boundary checks |
| `ops/` | CI infrastructure | CI scripts, the merge guard, the native-control sampler, vendored `tch` |
| `scripts/` | release tooling | installers, notices, legal pins, the V1 tuner launcher |
| `validation/` | validation tooling | pinned Python requirements used by every CI job; external-learner checks |
| `evidence/` | governance | `status.json` (source of `STATUS.md`) and the tree-scrub record |
| `apps/`, `packages/` | product stubs | a static studio page and a TypeScript type generator |
| `bench/` | shim | re-exports of `research/transfer/tdl` (candidate for folding in) |

## Three paper trees

| Tree | Status | Built by |
| --- | --- | --- |
| `paper/` | **active** manuscript: TDL, the transfer theory, the compression frontier | `bash paper/build.sh` (writes `target/paper/`) |
| `docs/whitepaper/` | **frozen** historical v2 validation paper and supplement; PDFs committed | `just paper` and `just paper-check`; CI rebuilds byte for byte |
| `docs/whitepaper/vnext/` | falsifiable draft with `\TBD` placeholders | linted by `vnext_paper.py`, never compiled |

## Generated, do not edit

| Output | Generator |
| --- | --- |
| README `COMPRESSION` block, `paper/generated/compression-*` | `python3 -B docs/whitepaper/scripts/compression_frontier.py` |
| README `QUICKSTART` block, `paper/generated/quickstart-macros.tex` | `python3 -B docs/whitepaper/scripts/quickstart_block.py --write-readme` |
| README `TDL RESULTS` and `PINNED DEMOS` blocks, `paper/generated/` hero and macros, `STATUS.md` | `python3 tools/claims/render.py` |
| README `BENCHMARK` block, `docs/readme/*.png` | `just paper` (`readme_benchmark.py --write-readme`, `v2_figures.py`) |
| everything under `docs/whitepaper/generated/` and the whitepaper PDFs | `just paper` |

## Do not move

These paths are pinned by exact location (frozen-evidence manifest, paper build graph,
Rust `include_str!`, CI scripts or tests): `research/benchmark/results/**`,
`docs/whitepaper/**`, `research/benchmark/review_fixes/**`, `research/transfer/**`,
`paper/generated/**`, `docs/readme/**`, `ops/**`, `contracts/`, `schemas/`,
`production/`, `tests/fixtures/`, `tests/golden/`, and the root files `README.md`,
`STATUS.md`, `CITATION.cff`, `AGENTS.md`, `Justfile`, `LICENSE`, `Cargo.*`,
`build.rs`, `pyproject.toml`, `rust-toolchain.toml`.

## Cleanup plan

Done in the change that added this page:

- `PY_TO_RUST_PLAN.md` and `PYTHON_EXCEPTIONS.md` moved from the root to
  [`docs/governance/`](governance/); `launch_tuner_tui.sh` moved to `scripts/`.
- `.gitattributes` marks generated ledgers, vendored code and archives for GitHub's
  diff and language views (no byte normalization).
- `docs/README.md` now distinguishes the active and frozen papers.

Safe next steps (no governance change needed):

1. Archive internal process notes (`docs/TODO-2026-10-15.md`, `docs/review-inputs/`,
   `docs/whitepaper/COMPREHENSION_v*.md`, `REVIEW_GAPS.md`, `DECISIONS_PENDING.md`,
   `STYLE_REFERENCE.md`) into `docs/archive/process/`, fixing links and map keys.
2. Fold `bench/metrics` into `research/transfer/tdl` and delete the stale
   `bench/metrics/golden-parity.just`.
3. Move the twenty benchmark notes that nothing references into
   `research/benchmark/notes/`.
4. Add a short README to each top-level directory that lacks one.
5. Point the `Cargo.toml` native-control metadata at the existing
   `research/transfer/native_control_prepare.py`.

Needs an owner decision (it changes pinned evidence):

- Move the 235 MB of frozen ledgers to release assets or LFS
  ([archive/large-artifacts.md](archive/large-artifacts.md)), which requires a new
  frozen-evidence baseline and new paper-input locks.
- Drop the anonymous PDF builds once the venue process no longer needs them.
- Document a product-only sparse checkout that omits the ledgers.
