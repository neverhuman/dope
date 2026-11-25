# Repository map

DOPE ships a standalone, pure-Rust generative table compiler (`engine/`) alongside its
receipt-bound v3.0.0 white paper (`docs/whitepaper-v3/`), offline PyTorch neural training
code (`research/levels/`), and a deliberately frozen historical evidence archive.

About 90% of the repository's bytes are hash-pinned historical benchmark ledgers that CI
and the paper builds verify by exact path (`ops/ci/frozen-evidence.json`). This map
explains what each top-level directory owns, where newcomers should start, and what is
pinned in place.

## If you only have ten minutes

| You want to | Start here |
| --- | --- |
| scan the 2-page executive or technical summary | [CEO Brief (PDF)](briefs/dope-v3-ceo-brief.pdf) · [CTO Brief (PDF)](briefs/dope-v3-cto-brief.pdf) |
| install DOPE and run a real dataset end-to-end | [README Quick start](../README.md#quick-start) · [v3 real-data quickstart](../examples/v3-quickstart/README.md) |
| read or submit the v3.0.0 white paper | [dope-v3.pdf](whitepaper-v3/dope-v3.pdf) · [arXiv upload bundle](whitepaper-v3/ARXIV_UPLOAD.md) |
| train `SingleEncV8` or add Evidence Bank features | [Training guide](training.md) · [Roadmap & Co-Authorship](../ROADMAP.md) |
| edit the pure-Rust v3 product engine | `engine/dope-core/`, `engine/dope-cli/`; run `just engine` |
| change a published number | never by hand — regenerate from receipts (see *Generated outputs* below) |

## Top-level directories

### Product & active v3.0.0 surfaces

| Directory | Class | What it holds |
| --- | --- | --- |
| `engine/` | **product (v3.0.0)** | Standalone pure-Rust workspace (`dope-core`, `dope-cli`, `dope-sdk`, `dope-norm`, `dope-arena`, `dope-ffi`): FIT-only Evidence Bank, `DSE8` codec, Format 1, `auto-v8` selector, `dope` CLI, and `dope_v8_emit` / `dope_kpi` binaries |
| `contracts/`, `schemas/` | product contracts | Frozen `evidence-layout-v1.json`, `dope-kpi.json`, `dope-fidelity-v3-candidate.json`, TDL contracts, and JSON schemas |
| `examples/` | examples | **`v3-quickstart/`** (real-dataset v3 `dope` lifecycle), plus historical `dope-kernel` demos (`quickstart/`, `public-regression/`, `healthcare-sim/`, `readme-regression/`, `adult/`) |
| `docs/` | documentation | Guides (`getting-started.md`, `training.md`, `signed-engine-release.md`), [CEO/CTO briefs](briefs/), active v3 white paper (`whitepaper-v3/`), archived manuscripts (`whitepaper-v2/`, `whitepaper/`), governance (`governance/`), and archived waves/reviews (`archive/`) |
| `scripts/` | release tooling | Verified Sigstore installer (`install.sh`), release verifier (`verify-engine-release.sh`), third-party notices, and local CI wrapper |
| `bench/real-world/results/scoreboard/v3.0.0/` | v3.0.0 evidence | Committed `dope_kpi` receipts, third-party baselines, and addendum CSVs backing the README scoreboard and v3 white paper |

### Offline neural training & research

| Directory | Class | What it holds |
| --- | --- | --- |
| `research/levels/` | deep-learning training | Offline PyTorch training (`s3_v8_trainer.py`, `s3_v8_feed.py`), synthetic priors, `SingleEncV8` architecture, Rust-vs-Python parity suite (`tests/test_v8_rust_parity.py`), and `s3_v8_eval.py` |
| `models/single-enc-v7/` | archived v2 weights | Historical v2.0.0 `SingleEncV7` SafeTensors checkpoint (kept for `--format auto|v7` backward compatibility; v3.0.0 ships weight-free) |

### Legacy compatibility, frozen evidence & infrastructure

| Directory | Class | What it holds |
| --- | --- | --- |
| `rust/`, `crates/dope-service/`, `production/` | legacy kernel | Historical `dope-kernel` crate (DPK3.2 codec, MFS/TDL certification harness, embedded KPI contracts read by `build.rs`) |
| `deploy/` | standalone runtime | Locked Candle Rust inference workspace for historical TVAE deployment conformance |
| `src/dope_kernel/` | Python V1 compat | Bounded V1 compatibility package (dated exception; no product routes) |
| `research/benchmark/`, `research/transfer/`, `bench/` | frozen evidence | Historical comparator adapters, TDL research modules, and ~235 MB of hash-pinned validation ledgers |
| `paper/` | archived v1 paper | Historical v1.0.0 manuscript (`dope1.tex`) and generated receipts |
| `tests/`, `validation/` | test suites | Root Rust integration tests, cross-language goldens, V1 Python tests, and pinned external-auditor constraints |
| `ops/`, `agent/`, `tools/` | CI & governance | CI shard scripts, pure-Rust `ops/merge_guard` (source caps, frozen evidence, repo hygiene), Jankurai maps/policies, claims firewall (`tools/claims/`), and read-only training dashboard (`tools/dope_dashboard/`) |
| `apps/studio/`, `packages/protocol/` | experimental stubs | Static studio page and generated TypeScript schema types (non-release, kept for test compatibility) |

## Four manuscript trees

| Tree | Status | Built by |
| --- | --- | --- |
| `docs/whitepaper-v3/` | **active (v3.0.0)** — Transfer Score vs. ARF & SDV, Evidence Bank, arXiv bundle (`arxiv/`, `dope-v3-arxiv.zip`) | `just paper-v3` and `just arxiv-v3` |
| `docs/whitepaper-v2/` | archived v2.0.0 manuscript (`dope-v2.pdf`) | archived unchanged |
| `docs/whitepaper/` | frozen historical v2 validation paper & supplement (CI rebuilds byte-for-byte) | `just paper` / `just paper-check` |
| `paper/` | archived v1.0.0 manuscript (`dope1.tex`) | `bash paper/build.sh` |

## Fast sparse checkout for product & feature contributors

If you only want to work on the Rust engine, Evidence Bank features, `SingleEncV8` training,
or docs without downloading the ~235 MB historical `research/benchmark/results/` archive:

```bash
git clone --filter=blob:none --no-checkout https://github.com/neverhuman/dope.git
cd dope
git sparse-checkout set engine contracts schemas examples/v3-quickstart docs/training.md docs/briefs docs/whitepaper-v3 research/levels
git checkout main
```

## Do not move

These paths are pinned by exact location (`ops/ci/frozen-evidence.json`, paper build graphs,
Rust `include_str!`, CI scripts, or tests): `research/benchmark/results/**`, `docs/whitepaper/**`,
`research/benchmark/review_fixes/**`, `research/transfer/**`, `paper/generated/**`,
`docs/readme/**`, `ops/**`, `contracts/`, `schemas/`, `production/`, `tests/fixtures/`,
`tests/golden/`, `bench/real-world/results/scoreboard/**`, and root manifest files (`README.md`,
`ROADMAP.md`, `STATUS.md`, `CITATION.cff`, `AGENTS.md`, `Justfile`, `LICENSE`, `Cargo.*`,
`build.rs`, `pyproject.toml`, `rust-toolchain.toml`).
