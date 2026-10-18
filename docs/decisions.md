# Decisions

This page is the active record: decisions in force and decisions awaiting the owner. The
dated history, including the vNext contract boundary and the N02 narrative decision, is in
[archive/decisions-history.md](archive/decisions-history.md). The 2026-10-15 rebuild plan is
[TODO-2026-10-15.md](TODO-2026-10-15.md), drawn from the five specifications in
[review-inputs/2026-10-15/](review-inputs/2026-10-15/INDEX.md).

## Pending owner decisions (D-01 to D-09)

Each item is **pending Jepson**. This PR uses the working default and freezes nothing.

| ID | Decision | Options in the specs | Working default in this PR |
| --- | --- | --- | --- |
| D-01 | Public metric name | QTDL (S1), TPDL (S2), LPDL (S3), TDL (S4), TransferBits (S5) | `TDL` (Transfer Description Length), held in one constant `METRIC_NAME` |
| D-02 | Headline operating point | T90/D80 (S1, S4, S5) or 90/90 (S3) | T90/D80, with 80/80, 90/90 and 95/95 in the threshold table; `protocol_frozen=false` |
| D-03 | Fixed-budget companion | CLP@10KiB (S3) or none | `transfer@{1,4,10,64} KiB` as a secondary view, never the headline |
| D-04 | Historical tag before file moves | e.g. `mfs-history-2026-10-15` at `353120f` | tag created locally only; pushed only after approval |
| D-05 | Repository split | separate product, benchmark and paper repositories, or one repository | one repository with `bench/` and `paper/` directories (S2 §12.2) |
| D-06 | Public exposure and history | private repository plus a history scrub (needs a force-push) | scrub the tracked tree only; no history rewrite |
| D-07 | Foundation-model licenses and GPU qualification budget | — | registry entries at `specified`; no qualification runs |
| D-08 | When to open the sealed test (CM-09) | — | not opened; contract field only |
| D-09 | Public binary name | retire `dope-kernel-v1`, add `dope` | add the `dope` alias; keep the old names as legacy shims |

## In force

Narrative freeze (2026-10-15):

- **One public truth.** The v2 panel is the active evidence. MFS-v3 and MFS-v4 are
  historical, and their scores are null. TDL is specified and unmeasured. No new public
  metric version appears before the TDL contract (T-020) lands.
- **One metric.** TDL replaces the weighted master score as the public metric. It is the
  smallest tested dataset-specific payload that passes the transfer and driver constraints
  and every hard gate; see [metric](metric.md). Distribution, structure, coverage,
  representation distance and encoded-value counts are diagnostics.
- **Bytes are the rate axis.** Payload bytes are primary and sampling-closure bytes are
  reported beside them. `encoded_numbers` never ranks anything.
- **Auditors come from a registry.** Each one moves through the typed lock states
  `specified` to `qualified`. Kumo Tabular, Mitra-v2, TabICLv2, TabPFN, LimiX, TabDPT and
  TabFM stay unqualified until they are admitted.
- **Claims firewall.** README and paper numbers come from
  `paper/generated/headline-results.json`, and every row carries an evidence label; see
  [claims policy](claims-policy.md).
- **Paper.** `docs/whitepaper/dope-mfs.tex` is frozen as the historical v2 validation
  paper. The TDL manuscript is the new `paper/` package.

Carried forward from earlier records (details in the archive):

- `production/behavior-contract-v1.json` is a draft research contract. It is not embedded
  and activates no production tier. Unknown margins, rights, pins, budgets, multiplicity
  or authority keep `protocol_frozen=false` and `confirmation_allowed=false`.
- Losses are computed in original target units after inverting fit-only transforms.
- Historical contracts, scorers, receipt ledgers and DPK3.2/3.3 fixtures are never
  rewritten to apply a new endpoint retroactively. Historical receipts decode under their
  original contracts.
- Execution, transfer, empirical privacy, formal mechanism, legal route and release are
  separate states. Empirical privacy never establishes formal DP or HIPAA
  de-identification.
- The historical paper's title is `DOPE: Generative Compression of Predictive Structure`
  (N02), and its whole panel is selection-exposed.

## Design note: ordinary CSV input with explicit categorical modes (T-092)

Status: follow-on, not implemented in this PR (S5 §9.1–9.2, S2 §14.3).

**Today.** L3 `compile`, `convert` and `certify` read headerless numeric CSV with the
target in the last column and every value already in `[0, 1]`. Categorical tokens are
mapped into `(0, 1)` by a 65,536-bucket HMAC projection. The Rust and benchmark maps
require a caller-supplied salt. The Python helper `standardize_numeric_table_01` has a
fixed public default salt, so its output is deterministic pseudonymization, not secrecy:
a small vocabulary can be recovered by guessing.

**Proposal.** The public CLI accepts an ordinary headered CSV or Parquet table and a
named target:

```text
dope compile data.csv --target price --categorical-mode usability --budget 10KiB -o model.dope
```

- Hashing is never implicit. The user picks a categorical mode, and the artifact
  manifest records it.
- **Usability mode** keeps the original category labels. Dictionaries and strings are
  charged to `payload_bytes`. The cleartext scan expects the declared schema tokens, and
  no privacy statement is made.
- **Pseudonymous release mode** writes stable pseudonymous identifiers and no raw
  category strings to the artifact. The reversal mapping is a separate file outside the
  sampling closure, and its size and handling are reported.
- Numeric scaling is fitted on `D_fit` only and stored in the artifact as charged state.
- Errors and public sidecars never echo source values, headers or extrema.
- The strict headerless numeric path stays as the internal canonical form, so existing
  receipts still verify.

**Acceptance for the follow-on.**

- A headered mixed-type CSV compiles without manual preprocessing.
- Usability-mode bytes include the dictionary.
- Pseudonymous-mode artifacts contain no input category string.
- The mode appears in `dope inspect`.
- An unknown mode fails closed without echoing data.

## D19 governance hold and blocker unblock matrix (2026-10-18)

**D19 — internal governance hold.** Sequence gate for the pre-campaign S1 privacy calibration and the M1 development comparison. It holds the empirical campaign order E03 -> E04 -> E05 -> E06. It does not release the P1 GPU freeze and does not authorize sealed-label access. Decision owner: `super_captain`. This text was supplied by captain_dope dispatch `gb_dope_prs_1030` and is not yet recorded as a super_captain decision.

Open conflicts: `FINAL_HANDOFF_POST_MERGE_402954ED.md` names the D19 owner "Human Owner"; `progress.json` uses "D05/D19 statistical freeze" for the Grok lane. Both need resolving.

Authority rule: `super_captain` decides internal governance and GPU-schedule holds (D10, D18, D19). Jepson-reserved applies only when a new paid spend, external license or legal commitment is needed.

| Task | Exact unblock action | Decision authority | Runnable on CPU now, without releasing the GPU hold |
|---|---|---|---|
| **W01** Kumo/TabICL official runtimes | Provision the vendor runtime and pinned adapter weights/credentials offline; run the vendor parity suite on live adapter outputs. Check terms and hashes first; accept no new terms or spend implicitly. | Jepson-reserved (new license terms, credentials, possible spend) | Offline ABI tests only (`research.transfer.tests.test_worker_abi`, 9 tests). Live parity needs the runtime and weights. |
| **E03** smallest observational/adverse pilot | Allocate the external multi-host benchmark window after the P1 GPU freeze lifts; mount the sealed dataset; obtain source/model rights and protected owner admission. | `super_captain` for the GPU schedule hold; Jepson-reserved for rights, human resources, new spend | No full-matrix run is admitted on CPU. Admission comes first. |
| **E04** precommit confirmatory design | Complete the D19 precommit admission review; ingest signed external baseline artifacts into the comparator. Depends on E03. | `super_captain` for the D19 review; Jepson-reserved if a baseline artifact needs a license | Drafting the precommit only: candidate artifacts, roles, thresholds, seeds, roster. No label access. |
| **E05** sealed confirmation | After E03 and E04 complete, obtain the D19 unlock token; run the sealed protocol through the sole authorized separate-principal custodian. | `super_captain` for the unlock token; custody stays under the D18 archival fence | Nothing. Sealed data is not opened before unlock. |
| **E06** four-level reproducibility | Re-run L1–L4 ratification once E03, E04, E05 artifacts exist. L2 and L4 are blocked; L3 is partial. | `super_captain` for sequencing; Jepson-reserved for independent owner resources and private access | L1 replay of aggregate tables and figures from the approved immutable ledger (already passing locally). |
| **E07** first observational CPU loop | Launch the post-release observation window and collect full-duration telemetry receipts from the external feed. | `super_captain` for the window schedule; Jepson-reserved if the feed needs a new external commitment | Freeze the reduced cpu-smoke panel and partition; run the deterministic adverse-mechanism fixture on public data. The production canary window is not runnable now. |

Machine-readable form: `DECISIONS.json` (proposed; see the packet's `DECISIONS_D19_PROPOSED.json`), because the ledger file is read-only and listed in `SHA256SUMS.json`.

## CPU preflight tests for the blocked tasks (W01, E03–E07)

Each blocked task has a deterministic CPU-only preflight class. It runs on CPU with no GPU, no sealed-label access and no external runtime. It checks that the admission path executes and fails closed when the external input is absent. A preflight pass does not unblock its task.

| Task | Preflight class | Command |
|---|---|---|
| W01 | `TestW01CpuPreflight` | `PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_workplan_acceptance.TestW01CpuPreflight -v` |
| E03 | `TestE03CpuPreflight` | `PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_workplan_acceptance.TestE03CpuPreflight -v` |
| E04 | `TestE04CpuPreflight` | `PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_workplan_acceptance.TestE04CpuPreflight -v` |
| E05 | `TestE05CpuPreflight` | `PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_workplan_acceptance.TestE05CpuPreflight -v` |
| E06 | `TestE06CpuPreflight` | `PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_workplan_acceptance.TestE06CpuPreflight -v` |
| E07 | `TestE07CpuPreflight` | `PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_workplan_acceptance.TestE07CpuPreflight -v` |

Status: the classes exist on `origin/feat/durable-workplan-acceptance-tests` (commit `d560468`, from the gemini lane). They are **not on this branch**, and they were **not run from this branch**. Run them after that branch lands, or cherry-pick it. Until then, the matrix above is the only record of the preflight commands.

## CPU preflight receipt digests (item 6)

Six CPU-only preflight receipts were generated for the blocked tasks. Their directory is named `preflight_31966ba2`, but **the commands ran on `d953f9b2caa800a51fe88e44d380046136d5da33`**, which is on `origin/feat/w01-e03-e07-cpu-readiness-harness`. The preflight test classes are **not on `origin/main` (`31966ba2`)**, so these receipts do not attest to main. Each receipt records `exit_code: 0` and its log shows a non-zero test count with `OK`.

| Task | Receipt | SHA-256 | Tests run (log) |
|---|---|---|---|
| W01 | `verify/preflight_31966ba2/W01_preflight.json` | `911b882aabdec6c231d6a376b6c9e4034cd9807ee25a31d7bce5da48cab1ad05` | 42 |
| E03 | `verify/preflight_31966ba2/E03_preflight.json` | `6a5455509f071633141595940911387e89b58720a8c7769b2e127d5314ba06a4` | 52 |
| E04 | `verify/preflight_31966ba2/E04_preflight.json` | `0dec7a86606cff39362f98b2a880f874b5d0782fefb87d4bd0cf46a2d2a7fa97` | 10 |
| E05 | `verify/preflight_31966ba2/E05_preflight.json` | `04eb8fafd6d75f2a500cc7142117fba1edfb9f9953352ccb621c2957c702d59d` | 48 |
| E06 | `verify/preflight_31966ba2/E06_preflight.json` | `2eb8d3fc8c7af3f5b1f28e2d5b960a80b80ed5e05e8b5f324c5d8a761665d0d0` | 37 |
| E07 | `verify/preflight_31966ba2/E07_preflight.json` | `779a1bcf30e09c61836c209ea13e59f5d1e05f7f96d92ea18adc3f112450122d` | 9 |

**Not yet linked into `tasks.json`.** Linking waits until PR #275 merges. The plan: run the six preflight commands on the new `origin/main`, link those receipts to W01 and E03–E07, and re-pin the 59 done tasks to the same SHA. Those tasks stay `blocked`: the GPU and campaign holds (D10, D18, D19, D05) are still active, so a preflight pass does not unblock any of them.
