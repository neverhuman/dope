# Normalize-once and downstream-value successor

Status: engineering contract accepted for the Next version implementation batch. Scientific qualification and serving adoption remain unperformed. The machine-readable decisions live in `docs/plans/dope-jope-contract-decisions.json`.

## Decision

Keep released MLC-v3 preparation and DML3/CSC1 readers unchanged as Lane A. Add Lane B using the independent `dope-norm` crate and `nc1-minmax-onehot-v1`. An input is never relabeled compliant merely because its values lie near zero and one. No shim is automatic.

Fit numeric maps and nominal vocabularies on explicitly declared physical FIT rows only. Keep constant numeric zero channels; add missing indicators only from FIT evidence. Nominals are grouped one-hot with unknown handling, without in-sample target encoding. Apply-time numeric values and real regression targets remain unclipped. Features, target, task and class semantics are explicit. Initial matrices use f64 and reject expansions beyond 256 columns.

The owner keeps names, extrema, vocabulary and real matrix/row payloads private. Public structural schemas contain group, kind and shape information only. Public errors use safe fixed codes. Source, FIT membership, transform, query view, weights, recipe and evaluator identities remain distinct.

Preserve both existing metric protocols and C02 simultaneous inference. Prospective evaluation freezes the receiver on real FIT, then uses one coordinate map for synthetic and real-test views. `S_select` measures a choice made using synthetic validation; `S_transfer` measures relative learner ranking after transfer. Ineligible all-tied comparisons are explicit null states. Empirical privacy checks never establish formal DP, de-identification or a payload-derived memorization bound.

Packets are independently hard-capped at 2, 16 and 64 KiB. Keep 4 KiB and historical readers until a separate measured retirement decision. Prefix nesting, epsilon, macro sample size and row-count policy are development decisions with a fresh, preregistered confirmation gate. They are not universal constants chosen by this ADR.

## Delivery and validation

Newest user direction consolidates this batch into one DOPE PR and existing JopeDime PR #2071. Incremental focused checks precede commits; full CI stabilization follows feature integration. Do not merge, release or change live serving state. Preserve foreign canonical edits and rapid CI-repair commits. A single integrator owns shared registration and lockfiles.

The implementation ledger separates source, fixture, exact-head CI, measured efficacy and serving states. D11 produces qualification readiness, not a trained model or admission to a scientific campaign. Missing provenance, outcomes and cost remain null.
