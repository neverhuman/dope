# Claims policy

This file says what the public surface may claim today. The public surface is the root
README, `paper/`, `docs/`, release notes, figure captions and the citation file. The
banned phrases in section 3 are machine-checked by the claims linter (`tools/claims/`,
T-010 and T-071). This file and `docs/archive/` quote them on purpose and are exempt.

Sources: S5 §18, S2 §21 (stop-ship conditions 1–17), S3 §66, S4 §15.8 and §29 in
[`docs/review-inputs/2026-10-15/`](review-inputs/2026-10-15/INDEX.md).

## 1. Evidence status labels

Every displayed result carries exactly one label. The generator writes them to
`evidence/status.json` and `STATUS.md` (T-009). A renderer refuses to present any label
other than `sealed_test` as a final headline.

| Label | Meaning | Allowed wording |
| --- | --- | --- |
| `sealed_test` | One pre-registered run on the official test split, opened once through a frozen release manifest. | "sealed test", with the manifest digest |
| `validation` | Measured on validation rows by a protocol frozen before the outcome was seen; no selection used these rows. | "validation estimate" |
| `selection_exposed_validation` | Measured on validation rows that also informed a profile, auditor or method choice. | "selection-exposed validation estimate" |
| `historical` | Produced under a retired contract (MFS-v2/v3/v4) or a superseded paper; kept so receipts still verify. | "historical", with the contract name |
| `specified_unrun` | A protocol, metric or panel that is written down but has produced no measurement. | "specified, not measured" |
| `unqualified` | A model or method named in a contract or registry that has not reached `qualified` in the lock state machine. | "planned, unqualified" |
| `failed` | Attempted and terminally failed, timed out, refused or over budget. It stays in every denominator. | "failed", with the disposition |

Current assignments (2026-10-15):

- The v2 panel (100 PMLB regression tables, CatBoost, linear and MLP auditors) is the
  active evidence. Its headline row is `selection_exposed_validation`: every lineage and
  all three auditors informed the profile choice.
- MFS-v2, MFS-v3 and MFS-v4 release scores are `historical`. They are null for every
  method.
- TDL (the working metric name, decision D-01) is `specified_unrun` until the metric
  core, the qualified auditor panel and the benchmark campaign produce a measurement.
- Kumo Tabular, Mitra-v2, TabICLv2, TabPFN, LimiX, TabDPT and TabFM are `unqualified`.
  Their contract or registry entries are plans, not evaluations.
- A baseline with no completed cell is `failed` or absent. It is never a loss for that
  baseline or a win for DOPE.

## 2. Claims that must not be made yet

| Do not claim | Why | Say instead |
| --- | --- | --- |
| DOPE beats TabSyn, or any method, overall | The paired interval against the strongest full-panel comparator contains zero; the comparison is CatBoost-centered and selection-exposed. | "the paired interval contains zero" |
| DOPE is validated across Kumo, Mitra or TabICLv2 | These auditors are unqualified. | "planned, unqualified" |
| MFS-v4 is an established master score | It was never measured; it is historical. | "historical draft contract" |
| The artifact is private, anonymous, de-identified, HIPAA-safe or differentially private | Empirical attacks are partial, and measured copy, membership and attribute gates fail on some cells. No formal mechanism is implemented. | "empirical attack results on the stated cells" |
| 1.9 KiB is a standalone or cold deployment size | 1.9 KiB is the median dataset-specific state. Sampling also needs the runtime closure. | "1.9 KiB of dataset-specific state; the cold closure is reported separately" |
| DOPE preserves an arbitrary table | The artifact is conditioned on one declared target workload. | "target-conditioned transfer" |
| A validation or selection-exposed result is an official-test result | The official test is sealed. | the label from section 1 |
| A missing or failed baseline cell is a loss for that baseline | Missing evidence is never a DOPE win. | "failed" or "not run" |
| Categorical hashing is secrecy | The default salt is public, so it is pseudonymization only. | "deterministic pseudonymization" |
| Representation closeness proves semantic equivalence | One encoder's embedding is a diagnostic. | "representation diagnostic" |
| Encoded-number count is description length | The count changes with packing; exact bytes are the rate. | "exact charged bytes" |
| A size is the smallest possible | Only a grid of budgets was tested. | "smallest tested budget" |
| A non-significant difference shows equivalence | Equivalence needs its own pre-declared rule. | "cannot be separated from" |
| Feature drivers agree because raw SHAP magnitudes agree across models | Attribution scales differ by model family. | per-auditor grouped permutation importance, once measured |
| The learned structure is reusable or general | There is no held-out learner-family or source-family evidence yet. | "on the declared auditors and lineages" |

Stop-ship (S2 §21): do not publish a new transfer or superiority claim while any of these
holds. The README presents v4 as current. A headline model is unqualified. Held-out
learners were seen by the compiler. The source-family confirmation set informed profile
design. Closure omits a checkpoint or sidecar. `encoded_numbers` affects a ranking.
Incomplete comparator cohorts appear as losses. A figure mixes cohorts without a
prominent disclosure. Failures leave a denominator. A best seed is chosen on final
outcomes. The first-page figure is CatBoost-only. The benchmark cannot be regenerated
from committed machine-readable results.

## 3. Banned phrases

The linter matches case-insensitively in README prose outside generated blocks, in
`paper/*.tex` outside `paper/archive/`, and in `CITATION.cff`.

| Pattern | Scope | Replacement |
| --- | --- | --- |
| `privacy-preserving`, `privacy preserving` | everywhere | "empirical attack results", or no privacy statement |
| `differentially private`, `HIPAA`, `de-identified`, `anonymous` (of an artifact) | everywhere | as above |
| `minimum` (of bytes, size, length or TDL) | everywhere | "smallest tested" |
| `beats TabSyn`, `outperforms` (without a sealed matched result) | everywhere | "the paired interval contains zero" |
| `validated across Kumo` (or Mitra, TabICL, TabPFN) | everywhere | "planned, unqualified" |
| `1.9 KiB cold closure`, `1.9 KB standalone` | everywhere | "1.9 KiB dataset-specific state" |
| `knowledge` | without held-out learner- and source-family evidence | "predictive structure" |
| `transferable knowledge`, `intelligen…`, `compression is intelligence` | everywhere | "predictive structure" |
| `state of the art`, `SOTA` | without a complete matched panel | none |
| `equivalent to`, `guarantee` | everywhere | "cannot be separated from", "is designed to" |
| `kilobyte-scale`, `kilobyte generator` | title, abstract, README, citation | exact byte counts |
| `MDL` (for the compactness score) | active docs and paper | "historical compactness score" |

## 4. Wording rules

- Put an evidence label on every result block (section 1).
- Say "state bytes" or "dataset-specific bytes", and report the sampling closure beside
  them. Never call an artifact standalone unless the closure is included.
- Say "the paired interval contains zero", not "ties".
- When no tested budget qualifies, report `not_reached_in_tested_range` with the largest
  tested budget. Do not print zero, the smallest failed artifact, or an extrapolation.
- Never describe a null metric as if it were measured.
- Resampled real rows are a row-count diagnostic, not a generator.
- A number in the README or paper comes from `paper/generated/headline-results.json`
  through a generated block, or from a pin that a checker verifies. It is never typed by hand.

## 5. Claim ladder

| Rung | Evidence | Example of allowed wording |
| --- | --- | --- |
| Development | validation or benchmark-design evidence | "on the selection-exposed validation panel" |
| Matched benchmark | paired, frozen protocol, not the final test | "on the frozen matched validation benchmark" |
| Sealed final | one pre-registered sealed run | "on the sealed test" |
| Release-certified | sealed final plus artifact, copy, privacy and replay gates | "release-certified under contract …" |

The project is on the first rung. The language in S2 §2.5 applies: before held-out
confirmation, "generative compression of predictive structure"; after held-out learner and
source-family confirmation, "transfer-preserving generative compression".
