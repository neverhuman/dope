# Read-only merge guard

`agent/merge_guard.py` inspects GitHub using GET requests through the existing
`gh` identity. It does not approve, enqueue, merge, change rules, switch accounts,
retry jobs or qualify a release. The installed `gh-role` remains the sanctioned
credential and eligibility helper. Its participation exclusion applies here too:
the eligible reviewer is `neverhumanbot`, distinct from `jepsontaylor`, the PR
opener and every associated commit author and committer. Unknown commit
provenance blocks readiness.

Run the synthetic adversarial fixtures without GitHub access:

```bash
rtk proxy python3 -B -m unittest discover -s agent/tests -p test_merge_guard.py -v
```

Capture actual controls without inventing an approval or touching a real PR:

```bash
rtk proxy python3 -B agent/merge_guard.py --controls-only --base FULL_CURRENT_MAIN_SHA
```

Preserve stdout under `target/` or the dispatched lane's ignored evidence
directory. `control_unavailable` means protection or rule visibility/enforcement
is absent or insufficient. Inspect the full receipt, including both base/rule
observations and API outcomes. Repository admin permission does not authorize a
rule change. A local lock is not server race protection.

## Review receipt and preflight

The independent review's frozen JSON receipt has these fields:

```json
{
  "version": 1,
  "repo": "neverhuman/dope",
  "pr": 123,
  "inspected_source_sha": "FULL_INSPECTED_SOURCE_SHA",
  "source_sha": "SAME_FULL_SOURCE_SHA",
  "base_sha": "FULL_CURRENT_MAIN_SHA",
  "source_tree_sha": "FULL_INSPECTED_TREE_SHA",
  "reviewer": "neverhumanbot",
  "review_id": 123456,
  "receipt_locator": "target/review/exact-source.json",
  "check_run_ids": [111, 112, 113, 114],
  "body_sha256": "SHA256_OF_EXACT_GITHUB_REVIEW_BODY"
}
```

The review must be `APPROVED` through the sanctioned independent role with
explicit API `commit_id` equal to the inspected source. Its first body line is
exactly `Reviewed source SOURCE; receipt LOCATOR; checks IDS.` where IDS is the
sorted comma-separated list of all workflow run IDs, without spaces. Additional
prose may follow; another full source SHA in the body refuses readiness. The
body hash, API identity, receipt, inspected source and tree must agree.
Hexadecimal identifiers of 7–40 characters in additional prose must identify the
inspected source or its prefix, regardless of letter case; decimal run IDs are
allowed. This conservative body contract rejects ambiguous shortened references.
Review
must follow the completed checks it cites. Later comments do not revoke an
approval; a dismissed or superseding non-comment review and outstanding changes
requested do block readiness.

```bash
rtk proxy python3 -B agent/merge_guard.py --pr PR_NUMBER \
  --source FULL_SOURCE_SHA --base FULL_CURRENT_MAIN_SHA \
  --review-receipt target/review/exact-source.json
```

The expected source workflow/event matrix is CI and Full Jankurai audit for
both `push` and `pull_request`; CI must expose `cpu`, `paper-clean`, `security`,
and the audit must expose `jankurai-audit`. All matching runs and jobs must be
completed success, including extra queued runs with no jobs. PR runs bind the
exact PR, source, base and branch. Canceled, skipped, missing, failed, truncated,
malformed and unadmitted retried evidence refuses readiness. The source workflow
definitions must retain this job inventory and block-form event declarations;
changes to the declared inventory require a reviewed guard update. The scheduled
GPU workflow is outside this CPU merge packet and remains separately unqualified.

Preflight also requires the inspected PR to be the sole open PR in the complete
repository listing, with base `main` and source/base repositories equal to the
inspected repository. A second open PR blocks readiness even if it targets a
different base. The helper uses GET with `gh api --paginate --slurp` to follow
all linked pages, then validates page completeness, entry identity and duplicate
PR numbers. Missing, malformed, bounded-out or failed listings cannot qualify
readiness. Both snapshots retain the normalized open listing; creation, closure
or source/base changes during inspection invalidate the receipt. An open PR
awaiting actual merge must still be the sole open PR.

This is an observed no-stacking preflight, not a server-side maximum-PR rule.
Another writer can create a PR after the final read. The coordinator retains the
single main-based publication slot and must re-inspect immediately before a
separately authorized operation. N28b has the first publication slot; N28a stays
local and independent until N28b is merged or closed, then refreshes onto actual
current main. Local serialization never replaces strict current-base server
controls or the atomic source-head predicate. Pagination follows the installed
[GitHub CLI API contract](https://cli.github.com/manual/gh_api).

Server controls must enforce the complete declared check set against the current
base, with all four checks explicitly bound to GitHub Actions app `15368`.
Missing, any-app or wrong-app bindings block qualification even when the names
match. Classic protection uses `checks[].app_id`; rulesets use
`required_status_checks[].integration_id`.
Server controls also require at least one approval, stale approval dismissal and last-push approval.
Classic protection must apply to administrators, disallow force pushes/deletion,
and have no review bypass allowances. Active rulesets must expose explicitly
empty bypass actors; omitted visibility is unknown. Extra unmodeled required
checks block readiness. A server queue needs equivalent enforced check/review
rules plus source workflows that actually handle `merge_group`. The current
workflows do not declare that event. This helper does not add server controls.

Reviews, workflow runs/jobs, PR source/base and controls are reread before
`ready_for_operator`. GitHub's mergeability must be `clean`; a queue does not
excuse `blocked`, which may reflect an unmet additional server requirement.
This state is an advisory observation, and the receipt
always records `merge_authorized: false`. Re-run inspection immediately before
any separately authorized operator action. The operator still needs the atomic
head predicate and enforced server strict-base or queue policy. JSON receipts
are not signatures or a substitute for trusted independent review.

## Actual merge verification

```bash
rtk proxy python3 -B agent/merge_guard.py --pr PR_NUMBER \
  --source FULL_SOURCE_SHA --base ORIGINAL_FULL_BASE_SHA \
  --verify-merge target/review/premerge-guard.json
```

The helper re-evaluates the complete premerge evidence rather than a saved green
boolean, and fetches the current review list twice. Dismissal or a superseding
review blocks verification; a change between those reads invalidates the receipt.
An open PR remains `awaiting_merge`, including after a CLI success or
queue acceptance. Only real `merged_at`, the actual merge commit, the receiving
main ref, an identical source tree, the exact squash parent/base and complete
receiving-main push runs can produce `merged_verified`. The premerge observation
must precede the actual merge. Main advancing beyond that commit or rules
changing refuses closure. Neither a retrospective review nor later passing CI
can repair historical premerge ordering.

This bounded version has no authenticated merge-group lineage adapter. Queue
closure therefore returns `queue_evidence_unavailable`; a supplied JSON group
cannot qualify it. Rebase/merge-commit lineage also remains outside its admitted
squash verification. These limitations block qualification rather than accepting
optimistic receipts. `awaiting_merge`, `control_unavailable` and other failures
return exit 1; inspection readiness or verified squash closure returns exit 0.

The PR233 incident and canceled duplicate source paper job remain preserved.
The guard's fixture passes do not establish N28a live controls, N28b's repaired
35-minute paper budget, science/privacy/native qualification or release authority.
