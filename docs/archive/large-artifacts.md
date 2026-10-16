# Large retained result artifacts

T-084 records the tracked-tree inventory. No data is removed or moved by this PR.
The byte counts and SHA256 digests below describe the retained frozen result
files, which preserve their exact `b8340ec` byte identities (enforced by
`ops/ci/check-frozen-evidence.py`) and are documented alongside the active-tree
scrub in [the scrub inventory](../../evidence/tree-scrub.json).

| Tracked result | Bytes | SHA256 |
|---|---:|---|
| `research/benchmark/results/review-fixes-v2/scalar-cells.jsonl` | 24121982 | `ea4168621dfaadfb5735d5df74d384e57f46f8e16b8053402b663a6960a94745` |
| `research/benchmark/results/density-matched-population-validation.json` | 21794615 | `5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a` |
| `research/benchmark/results/arf-tabsyn-followon-validation-v1/panel.json` | 21453198 | `069eea5d428fe76e5fcb1c100845d067d3add4cfb2df2c53d08e1451fa29324a` |
| `research/benchmark/results/retained-classical-validation/physical-metrics.jsonl` | 21314459 | `07c815bd2d6f9c6ee6e39bd71d19ec606ff63be5f327e7a521360427128e6e5c` |

Proposed follow-on: publish a versioned release asset with the exact bytes,
SHA256, evidence status, license/rights review and source-commit binding; retain
a small tracked manifest and a verified offline-cache fetch command. Alternatively,
adopt Git LFS for future large ledgers after confirming public download quotas
and CI availability. Existing history stays intact. Neither proposal authorizes
history rewriting, deleting retained results, or changing scientific values.

Before moving a file, update every input lock, paper reducer and verification
command against the same frozen bytes, prove clean offline reproduction from
the downloaded asset, and keep all failed/incomplete rows. Record the owner
decision and release URL. No artifact storage migration is claimed complete here.
