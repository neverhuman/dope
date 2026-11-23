# DOPE against published generators

## Verified public validation replay

The native scoreboard replay used the retained `lite-prelim-20261122` and
`lite-v2rc-20261122` ledgers. Their schedule and real-reference files match
byte for byte. The DOPE 2.0.0 arm matches the rows hash in the committed
scoreboard. The two SDV arms ran the upstream SDV 2.2.0 implementations at
their defaults and counted the bytes of each saved fitted synthesizer.
The archive reports an uncapped baseline run with numeric feature columns
and a categorical target for classification. [SDV upstream](https://github.com/sdv-dev/SDV)

| Generator | Median saved artifact | TRG at 4× | TRG at 1× | Failed / informative cells at 4× |
| --- | ---: | ---: | ---: | ---: |
| DOPE 2.0.0 small | 1,462 B | **−0.232** | **−0.349** | 5 / 125 |
| SDV GaussianCopula | 105,848 B | −0.686 | −0.934 | 0 / 125 |
| SDV TVAE | 404,722 B | −1.070 | −1.183 | 2 / 125 |

The paired 4× difference favors DOPE by **+0.454** TRG [0.342, 0.553]
against GaussianCopula and **+0.838** [0.662, 1.042] against TVAE.
The 10,000-draw family bootstrap included 94 regression, 16 binary and
7 multiclass families in both comparisons. The candidate capped each complete
artifact at 2,048 B. The median baseline artifacts were about 72× and 277×
larger, respectively.

| Generator | TRG with failure floor | No clipping, v2 failed-entry value | Median unclipped retention, linear / GBDT / held-out MLP |
| --- | ---: | ---: | ---: |
| DOPE 2.0.0 small | −0.283 | −0.261 | 0.970 / 0.976 / 0.990 |
| SDV GaussianCopula | −0.686 | −0.756 | 0.743 / 0.588 / 0.412 |
| SDV TVAE | −1.117 | −2.614 | 0.312 / 0.364 / 0.172 |

The retention medians describe successful informative entries. TRG and
the sensitivity diagnostics retain failed entries. The native comparator
receipt reads SDV minus DOPE. The differences above reverse that orientation.
The SDV CTGAN ledger has a three-cell smoke run. This replay covers
selection-exposed validation and provides a paired transfer result on these
two published methods. Joint T90/D80 qualification, the wider vendor roster,
and independent confirmation remain open.

| Replay input or output | SHA-256 |
| --- | --- |
| Source commit | `7c1f4598f6578137eb800850334943a7fa85dd15` (Git identity) |
| Exact-source native reducer | `506fdfaed69d4b0ec493d732b93758636db7cac5aaf75cb2008b4e385845c601` |
| Schedule | `5d292ea592ca122e94072595a86f3baf4d56928f1789759e5aa51748ba96143f` |
| Real reference | `c3dfe97c72ffef7e2f68f4c86383abc84d0d8f05a9b747c018662d148356fdae` |
| Native primary receipt | `598548698e68879425ef094ef0081dfd6356263d80585150cdb943db511cbd1c` |
| Native compare receipt | `f07f2647402e7a026cedd3a7ea3848658f34273c3996f40e3a22820c49389e98` |
| Unclipped and failure-floor diagnostic | `f95e1f394724f31b558d5031413efed51e912ad573e722bc69e8a4b43b07feca` |

The independent audit reconstructed the point estimates and denominators from
2,022 ledger entries per arm, including both row multipliers. It checked exact
schedule, reference and score-ledger hashes against the native receipts.
The audit also matched the committed historical scoreboard and its older
control count. That version included a held-out-only inapplicable control
cell in the reported count of 88. The current reducer reports 87 in the TRG
denominator accounting. Both reproduce the same TRG estimates.
The receipts and diagnostic are under `target/v8-verification/public-lite-replay/`.

The native audit after #399 independently reproduced the archived denominator and both paired differences. It also read the retained fitted files and checked their complete lengths and digests against the emission logs. DOPE had 310 artifacts with a maximum of 2,048 B. GaussianCopula and TVAE each had 328 artifacts, including one fitted file per arm whose real-reference cell was unscorable. Their maxima were 2,089,369 B and 5,152,455 B. The paired replay and physical inventory are under `target/v3-verification/public-lite-replay/`. The comparison audit SHA-256 is `3bc1fc359829461e197008e883ec094d50a80f77361a8aaf218cdbe396b5ec4c`.

The historical 16-table study also ran upstream SDV and ARF with a CatBoost
auditor. It reported a transfer-only threshold on its budget grid. The
1,000-table historical Elo used the repository's copula, diffusion, and ARF
prototypes. [Historical benchmark protocol](../../../../bench/real-world/README.md) ·
[Method authenticity registry](../../../../bench/methods-registry.json) ·
[Committed scoreboard](../../../../bench/real-world/results/scoreboard/v2.0.0/primary.json)

TabSyn is a meaningful challenger. In a separate historical study, the paired
difference in worst-auditor retention was +0.014 for DOPE with interval
[−0.028, +0.114] across 86 lineages. Median artifacts were 1,895 B and
42.3 MB. This supports a byte-efficiency comparison on that cohort.
[Historical TabSyn comparison](../../../../HISTORY.md)

A fresh local fixture smoke also fit DOPE and upstream SDV GaussianCopula
on the same 256 synthetic FIT rows, producing artifacts of 562 B and
53,147 B. Both sampled exactly 256 and 1,024 rows after the input moved
away from its fit path. This exercises the fit/sample handoff.

## Next public DEV roster

The staged wide DEV schedule has 108 cells across 22 physical families.
It reuses the 27-cell DEV cohort with four data seeds. Repeated cells help
measure variance within families. Power for new claims still depends on the
number and task mix of informative families.

| Staged input | SHA-256 |
| --- | --- |
| `schedule-wide-dev.jsonl` | `97439e8a978bfb3502c96b19fccc0302063f8090877a39512b1d576b85e25658` |
| `realref-wide-dev.jsonl` | `bb9ec3bed13721f3ee946d8c2d9b99c460ef91d266b62cd660147b0e8a3cb721` |
| `families.jsonl` | `3fa7f21fe4b6eceab4148fdf51bc298cde5f1733841296e2880fc3f473c0718b` |
| `baseline-v250.panel.json` | `d5bd12b0096238a11391e77fa9538123215e7cb54b23ff990f65ad5091bfbf84` |

The 4× slice has 46 informative cells across 3 binary, 4 multiclass and 4 regression task-family units. Use its repeated seeds to assess within-family variance. Expand confirmation with new admissible families and simulate paired power at the promotion effect and task guards. The target is at least 80% power, with at least 90 eligible sealed families where the protocol requires them.

| Arm | Available source and next admission evidence |
| --- | --- |
| Shipped DOPE, released V7, V7 step zero, Format 2 | Preserve distinct arm IDs. Use the baseline inventory in the [verification handoff](README.md) and bind each exact emitter and checkpoint to the common schedule. |
| DSE8 analytic and H2 challenger | DSE8 needs the pinned native emitter and full package conformance. The selected H2 soup needs its complete SHA-256 and its own DEV run identity. |
| SDV GaussianCopula and TVAE | SDV 2.2.0 defaults and retained native-budget artifacts are available for the historical replay. Bind the same recipes to the new declaration. |
| CTGAN | The author-library audit pins CTGAN 0.12.1 at `826da23f8f9385ad15fd206ecad691e04cb0ccdc`. The SDV wrapper smoke remains a distinct recipe. Choose and pin the route before the common fit/sample pilot. [Author-library audit](../../../../research/benchmark/SDV_METHOD_AUDIT.md) |
| ARF | The retained `arf_official` adapter uses upstream `arfpy`. Bind its version, fit/sample contract and complete saved state for this panel. The original authors' R package is a distinct implementation with `adversarial_rf`, `forde` and `forge`. [DOPE method registry](../../../../bench/methods-registry.json) · [ARF author package](https://bips-hb.github.io/arf/) |
| TabSyn | The source audit pins `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`. Reuse the available adapter where its task, fit and sample contracts qualify, and bind environment and full state to this panel. [TabSyn source audit](../../../../research/benchmark/TABSYN_SOURCE_AUDIT.md) |
| TabDDPM | The retained pilot is regression-only at `b476257dd460b778ba09eb97f7a51d6490fa17f8`. Carry unsupported tasks in the ledger until an authentic task-aware adapter qualifies. [TabDDPM source lock](../../../../research/benchmark/tabddpm-source.lock.json) |
| TabDiff | The audited author source is `5ecdb3356261aea72716cc9a779f31d7ad083bf4`. Its runtime and fit/sample contract are pending. [TabDiff source audit](../../../../research/benchmark/TABDIFF_SOURCE_AUDIT.md) |
| Independent CatBoost | The existing recipe uses 500 trees, depth 6, a declared seed and two threads. The existing runtime lock pins 1.2.10. Bind the CatBoost readout and task loss to the common auditor declaration before evaluating. [Recipe](../../../../bench/real-world/auditors.py) · [Runtime lock](../../../../research/benchmark/sdv-runtime.lock.json) |


- **DOPE.** Shipped 2.0.0, V7 analytic step zero and Format 2 on the same schedule.
  Admit any V7 challenger by full checkpoint and provenance. Test V8 analytic
  from the emitted packet after its native route lands.
- **Published generators.** Preserve the pinned SDV baselines and add the
  official CTGAN, ARF, TabSyn, TabDDPM and TabDiff implementations as their
  pinned builds and isolated fit/sample adapters qualify. Record the exact
  release or commit, code and lock hashes, license, bounded recipe, and
  fit/sample seeds. The SDV wrapper distribution uses the Business Source
  License. The standalone Copulas and CTGAN projects and the diffusion
  repositories have their own terms. [SDV](https://github.com/sdv-dev/SDV) ·
  [Copulas](https://github.com/sdv-dev/Copulas) ·
  [CTGAN and TVAE](https://github.com/sdv-dev/CTGAN) ·
  [TabSyn](https://github.com/amazon-science/tabsyn) ·
  [TabDDPM](https://github.com/yandex-research/tab-ddpm) ·
  [TabDiff](https://github.com/MinkaiXu/TabDiff)
- **Evaluation.** Retain native v2 TRG at 4× and 1×. Add an independently
  pinned CatBoost readout and an admitted foundation auditor if its runtime,
  task support and exposure record satisfy the panel contract. Preserve
  unclipped losses, per-auditor and per-task intervals, failure-floor results,
  structural and privacy coverage, and D80 against the real twin.
  [CatBoost upstream](https://github.com/catboost/catboost)
- **Bytes and reliability.** Measure each method's native uncapped point and
  any valid byte-bounded configurations at 512, 1,024, 2,048 and 4,096 B.
  Use matched rows, FIT groups, held-out groups, seeds, auditor recipes and
  resource limits. Record every attempt, including timeout, unsupported
  and over-budget results. Mark infrastructure interruptions explicitly.
  Charge preprocessing, vocabulary and model state needed by the recipient.
- **Admission.** Freeze the DEV family/cell roster and limits before dispatch.
  Require verified artifacts and full denominator reconciliation before
  advancing to the independent confirmation. Publish joint T90/D80 only at
  a measured passing frontier point under the frozen protocol.
