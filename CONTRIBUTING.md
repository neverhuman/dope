# Contributing to DOPE

Use a branch and a pull request against `main`. Describe the user-visible change,
its evidence, and any limits. Small commits with a task or issue ID make review
and replay easier. Maintainers integrate changes only after the required checks
and an independent review of the exact commit.

Install Rust from `rust-toolchain.toml`, Python 3.10 or later, and Just. Run
`just setup`, then `just fast` while editing and `just check` before committing.
The complete CPU, paper, security and audit commands and budgets are in
[docs/testing.md](docs/testing.md). Build output and evidence belong in the
checkout's reused `target/` directory. Do not commit local datasets, credentials,
checkpoints, operator paths, or compute host names.

Fork pull requests run the `public-smoke` job on GitHub-hosted CPU runners.
Private runners accept same-repository changes only. Fork contributors do not
need private infrastructure or sealed data; maintainers may port an accepted
change to a repository branch for the complete required checks. Never register
runners or expose secrets to a fork workflow.

Scientific changes need deterministic fixtures and a declared protocol. A fixture
is not benchmark evidence. Preserve historical decoding and unsuccessful runs.
Keep result numbers in `paper/generated/headline-results.json`; regenerate with
`python3 tools/claims/render.py`, then run `python3 tools/claims/check.py`.
Do not promote a registry candidate, open a sealed test, or describe a privacy
result as a formal guarantee without the corresponding reviewed evidence.

Read [GOVERNANCE.md](GOVERNANCE.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
[SECURITY.md](SECURITY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before contributing. Contributions are made under the repository's MIT license;
third-party datasets, optional adapters and model weights retain their own terms.
