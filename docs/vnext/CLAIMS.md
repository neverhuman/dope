# Claim scope ledger

All new scientific claims are unmeasured. The behavior contract is draft,
confirmation is unavailable, and release remains restricted local. Each future
claim must bind its protocol, cohort, artifact inventory, original-unit losses,
verified receipts, analysis source, and uncertainty procedure.

| Claim ID / capability | Falsifiable claim to test | Required evidence |
| --- | --- | --- |
| `transfer-numeric-v1` / `task_replacement` | Under a declared exact byte budget, synthetic training or labeled context stays within application-justified loss margins across declared source and learner families. | Frozen original-unit paired real-query losses, all required cells including failures, held-out sources/learners, power and simultaneous uncertainty. |
| `reusable-table-v1` / `reusable_table` | One artifact supports hidden alternate targets and declared analytical queries. | Separate joint-support, missingness, tail and alternate-target/query confirmation. |
| `augmentation-v1` / `augmentation` | Adding synthetic rows improves the declared real-data training recipe. | Separate mixing/compute experiment and matched controls. |
| `relational-v1` / `relational_replacement` | Linked generated tables preserve declared entity/time task behavior. | Separate topology, cardinality, join, leakage and entity-level privacy tests. |
| `pretraining-v1` / `foundation_pretraining` | A lawful generator curriculum improves source-disjoint real evaluation. | Explicit corpus/pretraining grants and equal-architecture/update/compute experiments. |

Teaching/context samples require a declared weighting, selection and access
policy; they do not establish population frequency preservation. Task replacement
does not qualify a reusable table. Flat-table evidence does not qualify relational
use. TSTR evidence does not establish pretraining benefit.

No SOTA, equivalence, causal, general preservation, formal-DP or HIPAA claim is
authorized by these interfaces or fixtures. Negative, harmful, incomplete and
inconclusive outcomes remain in the scheduled denominator. A non-significant
comparison does not establish equivalence. Unknown rights or privacy/legal routes
block sharing irrespective of measured utility.

## Historical paper claim limits (N02, 2026-10-14)

The title `DOPE: Generative Compression of Predictive Structure` covers only
these claims:

- validation-split retention under CatBoost, linear and MLP auditors on the
  100 PMLB regression lineages;
- charged artifact bytes;
- the stated controls.

Every lineage and all three auditors are exposed to the historical profile
selection (see the N02 entry in [the decision history](../archive/decisions-history.md)). That paper therefore makes no
held-out learner, held-out source-family, transferable-knowledge, intelligence,
minimum-size, formal-DP or HIPAA claim. Only the pre-declared sealed test run
can give a selection-free estimate for that profile. The claims in the table
above remain unmeasured and are unaffected by this narrative decision.

## Generated machine-readable ledger (Q00)

The five capability rows above are the declaration input for
`PYTHONPATH=. python3 -B -m tools.claims.render --ledger`. The command combines
them with `evidence/status.json` and pinned historical results and prints a
deterministic `dope-claim-ledger-v1` JSON document. Missing or duplicate capability
rows, inconsistent status labels and historical reclassification fail the gate.
It does not rewrite existing paper artifacts or `STATUS.md`.

Every entry includes `claim_id`, `claim_class`, `capability_scope`, `protocol_id`,
`receipt_sha256s`, `release_profile_state`, `pretraining_corpus_disclosure`,
`equivalence_procedure`, `allowed_scope` and `prohibited_extrapolations`.
[`claim-v1.schema.json`](../../schemas/claim-v1.schema.json) defines the closed
Draft 2020-12 interface. The generated ledger is checked by
`PYTHONPATH=. python3 -B -m tools.claims.check`; an externally supplied ledger can
also be checked with `--claim-ledger PATH`.

Measured claims require frozen protocols and receipt digests. Known pretraining
overlap prevents a measured claim; unknown overlap remains an explicit caveat.
Kumo and TabICL disclosures bind individual checkpoint digests. Neither a
non-significant test nor an interval containing zero supplies a verified
two-sided equivalence procedure or its positive finite margin.

All current release summaries are `unassessed_local`. S00's independent utility,
empirical privacy, formal-DP and legal release axes remain authoritative; the
summary enum does not combine them into a certification. The ledger validates
references and wording, not receipt authenticity or a protected campaign.
