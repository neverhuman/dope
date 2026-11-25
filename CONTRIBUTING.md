# Contributing to DOPE

Thank you for your interest in contributing to DOPE! Whether you are adding new data-specific
features to the FIT-only Evidence Bank, improving `SingleEncV8` neural training, strengthening
the pure-Rust `engine/` CLI, or expanding benchmarks and documentation, we welcome your PRs.

See [ROADMAP.md](ROADMAP.md) for high-leverage research ideas, open feature proposals, and our
**Co-Authorship Program** for contributors whose merged work ships in future versions (`v3.1+` / `v4.0`).

## Quick Contributor Setup

Install Rust from `rust-toolchain.toml` (`1.88.0`), Python 3.10–3.12, and Just (`just`).

- **Fast sparse checkout** (skips the ~235 MB historical benchmark archive if you only work on `engine/`, `research/levels/`, or docs):
  ```bash
  git clone --filter=blob:none --no-checkout https://github.com/neverhuman/dope.git
  cd dope
  git sparse-checkout set engine contracts schemas examples/v3-quickstart docs/training.md docs/briefs docs/whitepaper-v3 research/levels
  git checkout main
  ```
- **Pure-Rust engine workflow:** Run `just engine` (`cargo fmt`, `cargo clippy -D warnings`, and full `engine/` tests) and `just hygiene` (public-surface leak, map, and link checks).
- **Evidence Bank & `SingleEncV8` feature workflow:** See [docs/training.md](docs/training.md) and [ROADMAP.md](ROADMAP.md#3-fast-contributor-loop-idea---feature---quick-regression-test) for the 12-fixture Rust-vs-Python parity suite and quick synthetic-prior regression loop.
- **Full repository checks:** Run `just setup`, then `just fast` while editing and `just check` before committing. Complete commands and budgets are in [docs/testing.md](docs/testing.md).

## Language & Hygiene Rules

- **Rust-only nontraining policy (`AGENTS.md`):** Python is permitted **only for deep-learning training** (neural model, loss, autograd, optimizer, and training-coupled batch/RNG/checkpoint state). All standalone data preparation, inference, CLI tools, metrics, checkers, and nontraining tests must be Rust.
- **Source line cap:** Governed `.rs` and non-exempt `.py` files must stay strictly under 500 lines (`ops/merge_guard/source_limits.rs`).
- **Keep private infrastructure out of git:** Build output and evidence belong in `target/`. Never commit local datasets, credentials, checkpoints, operator paths, or compute host names.
- **Ownership & test routing:** When adding a new source area, register it in `agent/owner-map.json` and `agent/test-map.json` and verify with `just hygiene`.

## Pull Requests & CI

Use a branch and a pull request against `main`. Describe the user-visible change, its evidence,
and any limits. Small, logical commits make review and replay easier. Maintainers integrate
changes only after required checks pass and an independent review approves the exact commit.

Fork pull requests automatically run GitHub-hosted CPU checks (`public-smoke`, `public-hygiene`,
and path-triggered `v8-feature-regression` / `arxiv-bundle`) without secrets or write tokens.
Private runners accept same-repository branches only; maintainers may port an accepted fork PR
to a repository branch for the full self-hosted gates.

## Scientific Changes & Co-Authorship

Scientific changes need deterministic fixtures and a declared protocol. A fixture or local
regression run is not confirmation benchmark evidence. Preserve historical decoding,
`ops/ci/frozen-evidence.json`, and unsuccessful runs. Verify public prose against the claims
firewall with `python3 tools/claims/check.py`. Never open a sealed test or describe an
empirical privacy result as formal differential privacy or HIPAA de-identification.

Contributors whose merged work delivers a substantive, validated improvement to a future
version of DOPE (for example, a promoted Evidence Bank feature family, a trained neural model
improvement, a formal DP mechanism, new downstream auditor benchmarks, or a major engine
upgrade) will be **credited as co-authors** on the corresponding technical report / arXiv white
paper, [CITATION.cff](CITATION.cff), and release notes (see [ROADMAP.md](ROADMAP.md) and
[GOVERNANCE.md](GOVERNANCE.md)).

Read [GOVERNANCE.md](GOVERNANCE.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
[SECURITY.md](SECURITY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before contributing. Contributions are made under the repository's MIT license;
third-party datasets, optional adapters, and model weights retain their own terms.
