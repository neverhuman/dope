# Audit finding review

The clean pinned v1.7.1 scan is the PR gate. `agent/baselines/main.repo-score.json`
preserves all 83 original v1.6.11 findings, while
`target/jankurai/repair-log.json` records a source-specific disposition and
rerun command for each original fingerprint. A changed fingerprint alone does
not prove that the underlying behavior was repaired.

## Future-hostile language rule

Terms from the original lexical scan were checked against their code context.
Destructive launcher behavior was removed. The V1 score probe, prior model
coefficients, early stopping counters, symbolic baseline edges, mean
predictions, and atomic staging paths now have precise names. The checked
ledger lease state and explicit unsupported model identities describe real
states; the frozen `legacy_*` strings remain wire IDs for historical receipts.
Tests cover the behavior and the clean full scan reports no wording finding.
No wording exception can bypass byte, utility, structure, privacy, or release
gates.
