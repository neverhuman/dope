# Offline generator benchmark (research only)

This directory holds the offline research harness behind DOPE's validation evidence:
generator adapters, method and source locks, sealed-split handling, the v2 review panel,
committed result publications and their tests. It is research tooling. It grants no
production release, no privacy status and no final-test admission.

- **What the evidence says and how to read it:** [docs/benchmark.md](../../docs/benchmark.md).
- **What may be claimed:** [docs/claims-policy.md](../../docs/claims-policy.md).
- **The metric that replaces the historical scores:** [docs/metric.md](../../docs/metric.md).

## Status

- The v2 validation panel (`review_fixes/`) is the active evidence, labelled
  `selection_exposed_validation`.
- The public-core and matched S3 comparisons are not admitted, and the official test
  partitions are unopened.
- MFS-v2, MFS-v3 and MFS-v4 scores are historical and null.
- The TDL benchmark is specified and not run; its design is in docs/benchmark.md §4.
- Kumo Tabular, Mitra-v2, TabICLv2, TabPFN, LimiX, TabDPT and TabFM are planned,
  unqualified auditors.
- A missing, failed or timed-out cell stays in its denominator and is never a DOPE win.

## Layout

| Path | Contents |
| --- | --- |
| `PROTOCOL.md` | generator benchmark protocol v1: questions, endpoints, frozen thresholds |
| `RESULTS_STATUS.md` | campaign status ledger and remaining admission work |
| `*_SOURCE_AUDIT.md`, `*.audit.json` | per-method source, licence and runtime audits |
| `methods.lock.json`, `method-locks/` | method identities and lock state |
| `review_fixes/` | the v2 review panel: pre-declarations (`predeclare_v2.json`), cell binders, statistics, sealed-once handling |
| `results/` | committed, hash-pinned result publications, each with its own README and replay command |
| `fixtures/`, `pilot_sources/`, `patches/` | test fixtures, pinned pilot sources and recorded upstream patches |
| `tests/` | unit and replay tests for the harness and the publications |
| `fetch_pmlb.py` | the public PMLB data path; no credentials needed |
| `mfs_v3_*.py`, `representation.py` | historical MFS scorers, kept so receipts verify (legacy split in T-006) |

## Run

```bash
python3 -m research.benchmark.fetch_pmlb --list                     # list public PMLB tables
python3 -m research.benchmark.fetch_pmlb DATASET_ID OUTPUT_DIRECTORY # fetch one table
python3 -m unittest discover -s research/benchmark/tests -v          # harness and publication tests
```

Each publication under `results/` records its own replay or `--check` command. The routes
are in `agent/test-map.json`. Campaign execution needs separately prepared inputs and
private compute, and is outside this repository's public path.

## History

The dated operations log that used to fill this file covered fit counts, partial cohorts,
capacity grants, custody notes and the full command catalog. It is preserved, with
private host names removed, in
[docs/archive/benchmark-operations.md](../../docs/archive/benchmark-operations.md).
