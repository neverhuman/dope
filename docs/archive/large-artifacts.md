# Large retained result artifacts

T-084 records the tracked-tree inventory. No data is removed or moved by this PR.
The byte counts and SHA256 digests below describe the public tree after the
T-080 metadata scrub; original byte identities are retained in git history and
[the scrub inventory](../../evidence/tree-scrub.json).

| Tracked result | Bytes | SHA256 |
|---|---:|---|
| `research/benchmark/results/review-fixes-v2/scalar-cells.jsonl` | 24121982 | `ea4168621dfaadfb5735d5df74d384e57f46f8e16b8053402b663a6960a94745` |
| `research/benchmark/results/density-matched-population-validation.json` | 21794636 | `8e59083c5e4262e6cdcdb93ac6192ffabaf572704b2c0e8055a541838d7af177` |
| `research/benchmark/results/arf-tabsyn-followon-validation-v1/panel.json` | 21478208 | `e797f3ef0c1907d51caddb06c7669683afaf96274a2af5a885ab5dd7abdaf13e` |
| `research/benchmark/results/retained-classical-validation/physical-metrics.jsonl` | 21410189 | `835f4d456a909662104546c2644f6269169ab3aae4f571dbde089db3adddecb4` |

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
