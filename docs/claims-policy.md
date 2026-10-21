# Claims policy

This file says what the public surface may claim today. The public surface is the root
README, `paper/`, `docs/`, release notes, figure captions and the citation file. The
banned phrases in section 3 are machine-checked by the claims linter (`tools/claims/`,
T-010 and T-071). This file and `docs/archive/` quote them on purpose and are exempt.

Sources: S5 §18, S2 §21 (stop-ship conditions 1–17), S3 §66, S4 §15.8 and §29 in
[`docs/internal/review-inputs/2026-10-15/`](internal/review-inputs/2026-10-15/INDEX.md).

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
active `paper/**/*.tex` outside `paper/archive/`, and in `CITATION.cff`.
Known generated TeX is exempt from prose-number lint only through its checker:
the claims renderer authenticates its outputs, and the contract-macro generator
must reproduce `paper/generated/contract-macros.tex` byte for byte. Other files in
`paper/generated/` are still linted. Direct negative disclosures such as
"not HIPAA de-identification" are allowed; a later affirmative claim still fails.

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
- Toolchain versions (for example TeX Live, `latexmk` and BibTeX in `paper/README.md`) are environment facts, not results. They need no generated source, but must match the environment that builds the paper.

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

## 6. Reviewed design prose (U6/U7)

The following spelled-out design counts were reviewed on 2026-10-16 against the
tree shared by `59500ea` and receiving main `04b6f53`. They describe designs and
historical protocol settings. They establish no measured TDL result. A change to
the wording, scope or cited source requires a new review; this register does not
exempt result numbers from the generated-claims gate.

| Surface and reviewed wording | Source and scope |
| --- | --- |
| README: "one of seven families" | [`candidate_dependence.rs`](../rust/compiler/candidate_dependence.rs), match arms zero through six, describes the default compile search. |
| README: "at most twelve screened inputs" | [`part_01.rs`](../rust/compiler/target_fitting/part_01.rs), `fit_compact_neural_target_native`, caps the default screen at twelve; [`research_training.rs`](../rust/compiler/target_fitting/research_training.rs) uses that default when no research profile is selected. Explicit `features24_*` research profiles allow twenty-four, so the README wording must be scoped to the default. |
| README: "up to four screened features" | [`noise_fitting.rs`](../rust/compiler/target_fitting/noise_fitting.rs), `fit_noise`, passes support four to the bounded fitter. This describes the default path; refined profiles can set another bound. |
| README toy: "four columns" | [`INSPECT.txt`](../examples/readme-regression/INSPECT.txt) reports three positional features and a last-column target; the bundled [`train.csv`](../examples/readme-regression/train.csv) has four fields per row. This count belongs to the bundled toy. |
| README: "Two byte scopes" | [`tdl-contract-v1.json`](../contracts/tdl-contract-v1.json), `byte_scopes.primary` and `secondary`: payload and sampling closure. Optional cold-bundle and amortized scopes are additional disclosures. |
| README: "two real-trained models"; paper driver section: "two real-trained instances" | The contract's `driver_protocol.ceiling_normalization` is a real/real agreement denominator; [`drivers.rs`](../rust/metrics/drivers.rs) consumes that ceiling. These are the reference pair in the design, not a count of completed certification fits. |
| Paper protocol: "Three tracks" | [`benchmark.md`](benchmark.md), Tracks: author-default, equal-compute and rate-constrained. These are planned protocol tracks, not three completed campaigns. |
| Paper validation evidence: "all three auditors" | [`v2_panel.py`](../research/benchmark/review_fixes/v2_panel.py), `AUDITORS`: CatBoost, linear and MLP. The statement concerns historical selection exposure, not the planned foundation-model panel. |
| Paper historical card: "four times the fit rows" | The same historical reducer selects size four (the `HEADLINE, 4, catboost` cells). This is the v2 validation panel's sample-size setting; the TDL contract's primary sample size is one times the real fit rows. |

The README family/input/noise counts above require a default-search qualifier.
README.md states it directly below the table (L3, 2026-10-16): the counts describe
the default compile search, and historical research and refined profiles can differ.
Paper and README prose remain under L3 ownership. Demo license, DOI and positional-feature counts (U5)
are checked against the native pin's `source_data` and `INSPECT.txt` by the public
claims registry, including missing or contradictory metadata.

## 7. Generated claim ledger and public firewall (Q00)

`PYTHONPATH=. python3 -B -m tools.claims.render --ledger` emits a deterministic
`dope-claim-ledger-v1` JSON document, validated by
[`claim-v1.schema.json`](../schemas/claim-v1.schema.json). The inputs are the pinned
historical results, `evidence/status.json`, and the five capability rows in
[`vnext/CLAIMS.md`](vnext/CLAIMS.md). Historical entries stay historical; the five
future capabilities stay specified and unmeasured. This additional output does
not change `paper/generated/*` or `STATUS.md`.

Each entry declares its class (`measured`, `historical`, `specified`, or
`illustrative`), capability, protocol, receipt digests, allowed scope and prohibited
extrapolations. A measured entry requires a frozen protocol, nonempty SHA-256
receipt references and no known pretraining overlap. Schema validation checks
structure; it does not verify receipt contents or admit a campaign. The existing
receipt graph and publication gates remain necessary.

CR-13 / S1-R074 rejects affirmative equivalence, parity, on-par, matches or ties
conclusions from “no significant difference”, “non-significant difference”,
“not statistically significant” or `p > 0.05`. A supported equivalence procedure
requires verified two-sided bounds, a positive finite equivalence margin and
receipt references. A direct negative disclaimer is allowed; a later affirmative
conclusion in the same paragraph still fails. Unregistered public prose cannot
self-authorize an equivalence exception.

CR-10 / D10 requires an explicit overlap disclosure for each Kumo or TabICL
checkpoint (also Mitra and TabPFN). `known_none`, `overlapping` and `unknown`
are distinct states; the latter two require a visible caveat. A declaration must
bind its exact checkpoint digest. An unrun legacy registry stub is treated as
unknown and unqualified without rewriting historical locks. Qualification or a
contamination-review digest cannot silently turn unknown overlap into a clean
source-disjoint result. Contradictory contamination-free statements fail even
when a caveat is present.

CR-04 refuses unbacked HIPAA, anonymous, formal-DP and safe-share language,
private host/bucket paths, and raw source column/value echoes. New diagnostics
report only the violated rule and trusted surface label, without the offending
text, source headers or extrema. The historical firewall still runs, with its
input-quoting failures replaced by a safe diagnostic at the public command.

`PYTHONPATH=. python3 -B -m tools.claims.check` validates the generated ledger
and public surfaces. `--claim-ledger PATH` additionally checks an external ledger.
The renderer invokes the same new surface checks from the existing paper gate.
All generated entries currently use `unassessed_local`. The claim schema's
release labels are summaries, not an ordered promotion ladder: empirical screening,
formal DP and legal authorization remain separate S00 evidence axes. Utility,
an equivalence result or schema validity grants none of those authorities.
