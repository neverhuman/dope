# Dope repository guide

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
