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
| **E07** first observational CPU loop | Run the early public Bikeshare observational CPU loop with DOPE, Gaussian-copula, and compressed-row-subset controls at two byte caps and N/4N sample sizes, retaining real/null/synthetic and driver receipts. E05 is the separate sealed confirmation; post-release monitoring follows release. | `super_captain` for the window schedule; Jepson-reserved if the feed needs a new external commitment | Functional smoke is done and observed but unqualified: transfer values are negative, and this does not establish modern-panel, TDL, privacy, or confirmatory qualification. |

Machine-readable form: `DECISIONS.json` (proposed; see the packet's `DECISIONS_D19_PROPOSED.json`), because the ledger file is read-only and listed in `SHA256SUMS.json`.

### SC-DEC-097 E07 CPU smoke admission (2026-10-18)

`SC-DEC-097` authorized local CPU-only execution of the bounded `bikeshare-observational` CPU smoke loop on `xbabe2`. Its 21-test preflight receipt is retained as fixture evidence; the empirical comparator run is separately recorded under SC-DEC-106. The Rust frozen-evidence parity binary in `ops/merge_guard/Cargo.toml` is named `check-frozen-evidence`.

### SC-DEC-101 and SC-DEC-106 E07 observational smoke and Q01.FIG-3A/3B separation (2026-10-18)

SC-DEC-101 first held E07 pending actual controls. After the admitted SC-DEC-106 run, the
workplan records **60 / 65** tasks done and **5 / 65** blocked: `W01`, `E03`, `E04`, `E05`,
and `E06`. The persistent run receipt covers DOPE, Gaussian-copula, and compressed-row-subset
arms at 16,384 and 32,768 stored-byte caps and N/4N sample sizes. Its raw transfer values are
negative; E07 is complete as a functional observational smoke, while modern-panel, TDL, privacy,
and confirmatory claims remain unqualified. E05 remains the separate sealed confirmation gate.

The exact run receipt and execution provenance are in the admitted worktree target at
`target/bikeshare-observational/sc-dec-106-observation-1/receipt.json` and
`target/bikeshare-observational/sc-dec-106-observation-1/execution-provenance.json`.

`Q01.FIG-3A` is the completed deterministic figure regeneration/regression gate recorded in
`tasks.json`. `Q01.FIG-3B` is the future joint prediction-and-driver chart in `WORKPLAN.md` and
remains blocked; it is not counted as done.

### SC-DEC-125, SC-DEC-127, and SC-DEC-128 Finish-65 execution fence, D19 reconciliation, and W01 open-license admission (2026-10-19)

Under `SC-DEC-125`, `SC-DEC-127`, and `SC-DEC-128`, the historical D19 prose above is preserved as an historical record and reconciled against the canonical 65-task specification (`tasks.json` revision 102) and the adopted execution plan ([plans/finish65-major-release.md](plans/finish65-major-release.md), `sha256: 505311c62c6c28dbe0f4960d9536e739aa511eba91e4bf5f00542314cb46afe4`):

1. **Authoritative 5-task mapping and dependency reconciliation.** Older D19 narratives mapped `E03`/`E04` to earlier S1 privacy calibration and M1 comparison names, and the older task DAG listed only `W03` as an `E03` dependency. In the successor interpretation in force for the Finish-65 execution queue (`F65-00` through `F65-10`):
   - `W01` (`Qualify actual Kumo and TabICL official runtimes`) is an effective prerequisite for `E03` (`Run smallest actual observational and adverse-mechanism pilot`), because a scientifically qualified pilot requires qualified required learners in addition to `W03` engineering registry/scheduler completion.
   - `E03` is the smallest actual observational and adverse-mechanism development pilot (1 observational task, 1 adverse-mechanism task, DOPE/Gaussian-copula/compressed-row controls, 2 budgets, full trace and observed cost). The earlier `E07` (`Run the first observational CPU loop before the modern campaign`) public Bikeshare CPU loop remains completed functional smoke only (`observed_unqualified`, negative transfer) and does not substitute for `E03` or `E05`.
   - `E04` (`Precommit one confirmatory design after development selection`) requires a complete, recursively resolved confirmatory design and independent reviewer + separate-principal custodian signatures before any sealed access (`research/transfer/precommit_admission.py` and `research/transfer/protocols/confirmatory-precommit-v2.json`), distinct from the blocked v1 skeleton validator (`research/transfer/precommit.py::validate_blocked_precommit`).
   - `E05` (`Execute admitted sealed confirmation without tuning`) means 100% terminal accounting of the declared sealed batch under separate-principal custody (`rust/evaluation.rs::serve_frozen_batch`) with durable access/debit history and honest outcomes.
   - `E06` (`Obtain independent reproducibility at four distinct levels`) requires actual independent non-author replay receipts across `L1`, `L2`, `L3`, and `L4` (`docs/reproducibility.md` and `research/transfer/reproduce.py`).
2. **Exact contract locator.** The tracked v2 candidate contract path is `contracts/tdl-contract-v2-candidate.json` (hyphenated); dotted references (`contracts/tdl-contract-v2.candidate.json`) in older narratives are superseded. Both `protocol_frozen=false` and `confirmation_allowed=false` remain in force until all scientific, calibration, rights, and custody inputs qualify.
3. **W01 model identity correction and SC-DEC-128 open-license admission.**
   - Native `M001` (`001a8af11928e5fc03541682c30022b448488f37`) locks and adapters (`research/transfer/locks/kumo-tabular.json`, `research/transfer/adapters/auditors/kumo_tabular.py`) bind **NVIDIA `structured-data-models` / `sdm.models.KumoTabular` at `d50589692dbbbad4852eb1d9627f3b40c2673b63`** with HF checkpoint **`nvidia/Kumo-Tabular` at `3520742f7f3b937a2751266cea22d7fb6686882c`** (`OpenMDW-1.1`; `large/classifier.pt` `854,850,034` bytes `sha256: cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613`, `large/regressor.pt` `862,909,259` bytes `sha256: 745e0e5d97e4256d8a24cb61c2eecbd1128da6c83938c196b7589ea6f62e7421`), **not** the commercial relational `kumoai` / `KumoRFM` API. Paid API credentials (`KUMO_API_KEY`) and commercial spend are not dependencies of `W01`.
   - `research/transfer/locks/other-auditors.json` and `research/transfer/adapters/auditors/tabicl.py` bind **`soda-inria/tabicl` at `10719e21fd72b8945d4a4f9ae077e350ba8650d9` (`tabicl==2.0.3`)** with HF checkpoint **`jingang/TabICL` at `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`** (`BSD-3-Clause`; `tabicl-classifier-v2-20260212.ckpt` `110,368,038` bytes `sha256: bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`, `tabicl-regressor-v2-20260212.ckpt` `114,324,594` bytes `sha256: 0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`).
   - In `SC-DEC-128` (`2026-10-19T03:01:32Z`), `super_captain` approved open-license installation and exact pinned checkpoint verification/fetching for both `TabICL` (`BSD-3-Clause`) and `NVIDIA Kumo-Tabular` (`OpenMDW-1.1`) in an isolated DOPE certification environment (`needs_jepson = []`, zero paid spend). Installation and cache verification authorize `F65-01` execution only and do not mark `W01` done until native `Q0–Q9` regression and binary qualification receipts and independent review pass.
