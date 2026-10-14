"""Synthetic API fixtures only: no approvals, merges, network or protected data."""

import base64
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import merge_guard as guard


SOURCE, BASE, TREE, MERGE = (letter * 40 for letter in "abcd")
PINS = {"repo": "neverhuman/dope", "pr": 7, "source_sha": SOURCE,
        "base_sha": BASE, "base_branch": "main"}


def controls():
    return {"protection": {
        "required_status_checks": {"strict": True, "contexts": sorted(guard.CHECKS), "checks": []},
        "required_pull_request_reviews": {"required_approving_review_count": 1,
            "dismiss_stale_reviews": True, "require_last_push_approval": True,
            "bypass_pull_request_allowances": {"users": [], "teams": [], "apps": []}},
        "enforce_admins": {"enabled": True}, "allow_force_pushes": {"enabled": False},
        "allow_deletions": {"enabled": False}}, "rules": [], "rulesets": {}}


def rule_controls(queue=False):
    checks = {"type": "required_status_checks", "ruleset_id": 11, "parameters": {
        "required_status_checks": [{"context": name} for name in sorted(guard.CHECKS)],
        "strict_required_status_checks_policy": not queue}}
    review = {"type": "pull_request", "ruleset_id": 11, "parameters": {
        "required_approving_review_count": 1, "dismiss_stale_reviews_on_push": True,
        "require_last_push_approval": True}}
    rules = [checks, review]
    if queue:
        rules.append({"type": "merge_queue", "ruleset_id": 11, "parameters": {}})
    return {"protection": None, "rules": rules, "rulesets": {
        "11": {"id": 11, "enforcement": "active", "bypass_actors": []}}}


def runs(source=SOURCE, events=("push", "pull_request"), branch="codex/fixture"):
    result = []
    for path, jobs in guard.WORKFLOWS.items():
        for event in events:
            identifier = len(result) + 101
            result.append({"id": identifier, "path": path, "event": event, "head_sha": source,
                "head_branch": branch, "run_attempt": 1, "status": "completed", "conclusion": "success",
                "updated_at": "2026-10-14T02:00:00Z",
                "pull_requests": [{"number": 7, "head": {"sha": source}, "base": {"sha": BASE}}],
                "jobs": [{"id": identifier * 10 + j, "name": name, "head_sha": source,
                          "status": "completed", "conclusion": "success"} for j, name in enumerate(jobs)]})
    return result


def fixture(queue=False):
    ctrl = rule_controls(True) if queue else controls()
    pull = {"number": 7, "state": "open", "draft": False, "merged": False, "merged_at": None,
            "merge_commit_sha": None, "mergeable_state": "clean",
            "head": {"sha": SOURCE, "ref": "codex/fixture"},
            "base": {"sha": BASE, "ref": "main", "repo": {"full_name": "neverhuman/dope"}},
            "user": {"login": guard.OPERATOR}, "commits": 1}
    snapshot = {"pull": pull, "base_sha": BASE, "controls": ctrl}
    defs = {path: "name: fixture\non:\n  push:\n  pull_request:\n"
            + ("  merge_group:\n" if queue else "")
            + "\njobs:\n" + "".join(f"  {job}:\n    runs-on: fixture\n" for job in jobs)
            for path, jobs in guard.WORKFLOWS.items()}
    receipt = {"version": 1, "repo": PINS["repo"], "pr": PINS["pr"],
               "inspected_source_sha": SOURCE, "source_sha": SOURCE, "base_sha": BASE,
               "source_tree_sha": TREE, "reviewer": guard.REVIEWER, "review_id": 901,
               "receipt_locator": "target/review/fixture.json", "check_run_ids": [101, 102, 103, 104]}
    body = f"Reviewed source {SOURCE}; receipt {receipt['receipt_locator']}; checks 101,102,103,104."
    receipt["body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    evidence = {"before": snapshot, "after": deepcopy(snapshot), "operator": {"login": guard.OPERATOR},
        "source_commit": {"sha": SOURCE, "tree": {"sha": TREE}},
        "reviewer_permission": {"user": {"login": guard.REVIEWER}, "permission": "write"},
        "commits": [{"sha": SOURCE, "author": {"login": guard.OPERATOR}, "committer": {"login": guard.OPERATOR}}],
        "reviews": [{"id": 901, "state": "APPROVED", "user": {"login": guard.REVIEWER},
                     "commit_id": SOURCE, "body": body, "submitted_at": "2026-10-14T02:01:00Z"}],
        "workflow_definitions": defs, "runs": runs(), "inspection_completed_at": "2026-10-14T02:02:00Z"}
    return evidence, receipt


def prior_receipt(queue=False):
    evidence, receipt = fixture(queue)
    return {"version": 1, "pins": deepcopy(PINS), "review_receipt": receipt, "evidence": evidence,
            "captured_at": "2026-10-14T02:02:00Z", "state": "ready_for_operator"}


def merge_fixture(prior, merged=True):
    before = deepcopy(prior["evidence"]["after"])
    if merged:
        before["pull"].update(state="closed", merged=True, merged_at="2026-10-14T02:03:00Z", merge_commit_sha=MERGE)
        before["base_sha"] = MERGE
    return {"before": before, "after": deepcopy(before),
            "merge_commit": {"sha": MERGE, "tree": {"sha": TREE}, "parents": [{"sha": BASE}]},
            "receiving_runs": runs(MERGE, ("push",), "main")}


class FakeAPI:
    """Exercise the collector against REST-shaped fixtures, not a ready boolean."""
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = []
        self.snapshots = 0
        self.fail_endpoint = None
        self.mutate_after = None

    def get(self, endpoint, optional=False):
        self.calls.append({"endpoint": endpoint, "method": "GET"})
        if endpoint == self.fail_endpoint:
            raise guard.Refused("api_unavailable")
        e = self.evidence
        root = "repos/" + PINS["repo"]
        if endpoint == f"{root}/pulls/7":
            self.snapshots += 1
            if self.snapshots == 2 and self.mutate_after:
                self.mutate_after(e)
            return deepcopy(e["before" if self.snapshots < 2 else "after"]["pull"])
        snap = e["before" if self.snapshots < 2 else "after"]
        ctrl = snap["controls"]
        if endpoint == "user": return deepcopy(e["operator"])
        if endpoint == f"{root}/git/ref/heads/main": return {"object": {"sha": snap["base_sha"]}}
        if endpoint == f"{root}/branches/main/protection": return deepcopy(ctrl["protection"])
        if endpoint.startswith(f"{root}/rulesets/"):
            identifier = endpoint.split("/rulesets/")[1].split("?")[0]
            return deepcopy(ctrl["rulesets"][identifier])
        if endpoint == f"{root}/git/commits/{SOURCE}": return deepcopy(e["source_commit"])
        if endpoint == f"{root}/git/commits/{MERGE}": return deepcopy(e["merge_commit"])
        if endpoint == f"{root}/collaborators/{guard.REVIEWER}/permission": return deepcopy(e["reviewer_permission"])
        if endpoint.startswith(f"{root}/contents/"):
            path = endpoint.split("/contents/")[1].split("?")[0]
            return {"encoding": "base64", "content": base64.b64encode(e["workflow_definitions"][path].encode()).decode()}
        raise AssertionError("Unexpected GET: " + endpoint)

    def listing(self, endpoint, key=None):
        self.calls.append({"endpoint": endpoint, "method": "GET"})
        if endpoint == self.fail_endpoint:
            raise guard.Refused("api_unavailable")
        e = self.evidence
        if "/rules/branches/" in endpoint:
            return deepcopy(e["before" if self.snapshots < 2 else "after"]["controls"]["rules"])
        if endpoint.endswith("/commits"): return deepcopy(e["commits"])
        if endpoint.endswith("/reviews"): return deepcopy(e["reviews"])
        if "/actions/runs?" in endpoint:
            selected = e.get("receiving_runs", []) if MERGE in endpoint else e["runs"]
            return [{k: deepcopy(v) for k, v in r.items() if k != "jobs"} for r in selected]
        if endpoint.endswith("/jobs?filter=all"):
            identifier = int(endpoint.split("/actions/runs/")[1].split("/")[0])
            selected = e["receiving_runs"] if "receiving_runs" in e else e["runs"]
            return deepcopy(next(r["jobs"] for r in selected if r["id"] == identifier))
        raise AssertionError("Unexpected listing: " + endpoint)


class PreflightTests(unittest.TestCase):
    def rejected(self, evidence, receipt, reason):
        with self.assertRaises(guard.Refused) as caught:
            guard.evaluate_preflight(evidence, PINS, receipt)
        self.assertEqual(caught.exception.reason, reason)

    def test_qualified_classic_and_ruleset_controls(self):
        evidence, receipt = fixture()
        self.assertEqual(guard.evaluate_preflight(evidence, PINS, receipt)["base_policy"], "strict")
        evidence["before"]["controls"] = rule_controls()
        evidence["after"] = deepcopy(evidence["before"])
        self.assertEqual(guard.evaluate_preflight(evidence, PINS, receipt)["base_policy"], "strict")

    def test_head_base_review_or_rule_move_during_inspection(self):
        for field in ("head", "base"):
            with self.subTest(field=field):
                e, r = fixture()
                e["after"]["pull"][field]["sha"] = "e" * 40
                self.rejected(e, r, "inspection_changed")
        e, r = fixture()
        e["after"]["controls"]["protection"]["enforce_admins"]["enabled"] = False
        self.rejected(e, r, "inspection_changed")

    def test_wrong_pinned_head_base_repository_and_closed_or_draft_pr(self):
        mutations = [
            (lambda p: p["head"].update(sha="e" * 40), "head_moved"),
            (lambda p: p["base"].update(sha="e" * 40), "base_moved"),
            (lambda p: p["base"]["repo"].update(full_name="other/repo"), "base_moved"),
            (lambda p: p.update(state="closed"), "pr_not_open"),
            (lambda p: p.update(draft=True), "pr_not_open"),
            (lambda p: p.update(mergeable_state="unknown"), "mergeability_unavailable")]
        for change, reason in mutations:
            with self.subTest(reason=reason):
                e, r = fixture()
                change(e["before"]["pull"])
                e["after"] = deepcopy(e["before"])
                self.rejected(e, r, reason)

    def test_each_review_source_identity_must_match(self):
        for field in ("inspected_source_sha", "source_sha", "base_sha", "source_tree_sha"):
            with self.subTest(field=field):
                e, r = fixture()
                r[field] = "e" * 40
                self.rejected(e, r, "receipt_source_mismatch")
        e, r = fixture()
        e["reviews"][0]["commit_id"] = "e" * 40
        self.rejected(e, r, "approval_stale")

    def test_wrong_reviewer_or_operator_or_reviewer_authored_or_committed(self):
        mutations = [lambda e, r: r.update(reviewer=guard.OPERATOR),
            lambda e, r: e["reviews"][0]["user"].update(login=guard.OPERATOR),
            lambda e, r: e["operator"].update(login=guard.REVIEWER),
            lambda e, r: e["commits"][0]["author"].update(login=guard.REVIEWER),
            lambda e, r: e["commits"][0]["committer"].update(login=guard.REVIEWER),
            lambda e, r: e["reviewer_permission"].update(permission="read")]
        for change in mutations:
            with self.subTest(change=change):
                e, r = fixture()
                change(e, r)
                self.rejected(e, r, "reviewer_ineligible")

    def test_unknown_or_truncated_commit_provenance(self):
        for field in ("author", "committer"):
            e, r = fixture()
            e["commits"][0][field] = None
            self.rejected(e, r, "commit_provenance_unavailable")
        e, r = fixture()
        e["commits"] = []
        self.rejected(e, r, "commit_provenance_unavailable")

    def test_old_body_at_new_api_head_or_changed_body_hash(self):
        for body in ("Reviewed source " + "e" * 40 + "; receipt old; checks 1.", "changed"):
            e, r = fixture()
            e["reviews"][0]["body"] = body
            r["body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
            self.rejected(e, r, "approval_body_mismatch")
        e, r = fixture()
        r["body_sha256"] = "0" * 64
        self.rejected(e, r, "approval_body_mismatch")
        e, r = fixture()
        e["reviews"][0]["body"] += "\nAlso reviewed source " + BASE
        r["body_sha256"] = hashlib.sha256(e["reviews"][0]["body"].encode()).hexdigest()
        self.rejected(e, r, "approval_body_mismatch")

    def test_missing_dismissed_commented_or_later_revoked_approval(self):
        e, r = fixture()
        e["reviews"] = []
        self.rejected(e, r, "review_missing")
        for state in ("DISMISSED", "COMMENTED", "CHANGES_REQUESTED"):
            e, r = fixture()
            e["reviews"][0]["state"] = state
            self.rejected(e, r, "approval_stale")
        e, r = fixture()
        later = deepcopy(e["reviews"][0])
        later.update(id=902, state="CHANGES_REQUESTED", submitted_at="2026-10-14T02:02:00Z")
        e["reviews"].append(later)
        self.rejected(e, r, "approval_stale")

    def test_later_comment_preserves_approval_but_other_changes_requested_blocks(self):
        e, r = fixture()
        later = deepcopy(e["reviews"][0])
        later.update(id=902, state="COMMENTED", submitted_at="2026-10-14T02:02:00Z")
        e["reviews"].append(later)
        self.assertEqual(guard.evaluate_preflight(e, PINS, r)["base_policy"], "strict")
        later["state"] = "CHANGES_REQUESTED"
        later["user"] = {"login": "other-independent-reviewer"}
        self.rejected(e, r, "changes_requested")

    def test_missing_workflow_even_when_remaining_rollup_is_green(self):
        for index in range(4):
            e, r = fixture()
            e["runs"].pop(index)
            self.rejected(e, r, "missing_workflow")

    def test_failed_canceled_skipped_pending_and_queued_no_job_runs(self):
        for status, conclusion in (("completed", "failure"), ("completed", "cancelled"),
                                  ("completed", "skipped"), ("in_progress", None), ("queued", None)):
            e, r = fixture()
            e["runs"][0].update(status=status, conclusion=conclusion, jobs=[])
            self.rejected(e, r, "checks_not_successful")
        e, r = fixture()
        extra = deepcopy(e["runs"][0])
        extra.update(id=105, status="queued", conclusion=None, jobs=[])
        e["runs"].append(extra)
        self.rejected(e, r, "checks_not_successful")

    def test_workflow_success_cannot_hide_missing_canceled_or_wrong_head_job(self):
        for index in range(3):
            e, r = fixture()
            e["runs"][0]["jobs"].pop(index)
            self.rejected(e, r, "missing_or_duplicate_jobs")
        e, r = fixture()
        e["runs"][0]["jobs"][0]["conclusion"] = "cancelled"
        self.rejected(e, r, "checks_not_successful")
        e, r = fixture()
        e["runs"][0]["jobs"][0]["head_sha"] = BASE
        self.rejected(e, r, "check_source_mismatch")

    def test_wrong_check_source_base_pr_branch_or_unadmitted_retry(self):
        mutations = [(lambda e: e["runs"][0].update(head_sha=BASE), "check_source_mismatch"),
            (lambda e: e["runs"][1]["pull_requests"][0].update(number=8), "check_pr_mismatch"),
            (lambda e: e["runs"][1]["pull_requests"][0]["base"].update(sha=SOURCE), "check_pr_mismatch"),
            (lambda e: e["runs"][0].update(head_branch="another-branch"), "check_branch_mismatch"),
            (lambda e: e["runs"][0].update(run_attempt=2), "retry_requires_separate_admission")]
        for change, reason in mutations:
            e, r = fixture()
            change(e)
            self.rejected(e, r, reason)

    def test_approval_must_cite_all_exact_runs_after_checks_complete(self):
        e, r = fixture()
        r["check_run_ids"] = [101]
        self.rejected(e, r, "approval_checks_stale")
        e, r = fixture()
        e["runs"][0]["updated_at"] = "2026-10-14T02:02:00Z"
        self.rejected(e, r, "approval_checks_stale")

    def test_unprotected_and_each_missing_server_control(self):
        e, r = fixture()
        e["before"]["controls"]["protection"] = None
        e["after"] = deepcopy(e["before"])
        self.rejected(e, r, "control_unavailable")
        changes = [lambda p: p["required_status_checks"].update(strict=False),
            lambda p: p["required_status_checks"].update(contexts=["cpu"]),
            lambda p: p["required_pull_request_reviews"].update(dismiss_stale_reviews=False),
            lambda p: p["required_pull_request_reviews"].update(require_last_push_approval=False),
            lambda p: p["required_pull_request_reviews"].update(required_approving_review_count=0),
            lambda p: p["required_pull_request_reviews"].update(bypass_pull_request_allowances={"users": ["author"]}),
            lambda p: p["enforce_admins"].update(enabled=False),
            lambda p: p["allow_force_pushes"].update(enabled=True)]
        for change in changes:
            e, r = fixture()
            change(e["before"]["controls"]["protection"])
            e["after"] = deepcopy(e["before"])
            self.rejected(e, r, "control_unavailable")

    def test_unrelated_rules_cannot_supply_classic_bypass_enforcement(self):
        e, r = fixture()
        c = e["before"]["controls"]
        c["protection"]["enforce_admins"]["enabled"] = False
        c["rules"] = [{"type": "deletion", "ruleset_id": 11}]
        c["rulesets"] = rule_controls()["rulesets"]
        e["after"] = deepcopy(e["before"])
        self.rejected(e, r, "control_unavailable")

    def test_ruleset_bypass_unknown_disabled_or_present_and_extra_required_check(self):
        for change in (lambda d: d.pop("bypass_actors"), lambda d: d.update(enforcement="evaluate"),
                       lambda d: d.update(bypass_actors=[{"actor_type": "RepositoryRole", "actor_id": 5}])):
            e, r = fixture()
            e["before"]["controls"] = rule_controls()
            change(e["before"]["controls"]["rulesets"]["11"])
            e["after"] = deepcopy(e["before"])
            self.rejected(e, r, "control_unavailable")
        e, r = fixture()
        e["before"]["controls"] = rule_controls()
        e["before"]["controls"]["rules"][0]["parameters"]["required_status_checks"].append({"context": "unknown"})
        e["after"] = deepcopy(e["before"])
        self.rejected(e, r, "control_unavailable")

    def test_queue_requires_source_workflow_merge_group_support(self):
        e, r = fixture(queue=True)
        self.assertEqual(guard.evaluate_preflight(e, PINS, r)["base_policy"], "queue")
        path = next(iter(guard.WORKFLOWS))
        e["workflow_definitions"][path] = e["workflow_definitions"][path].replace("  merge_group:\n", "")
        self.rejected(e, r, "control_unavailable")

    def test_queue_does_not_excuse_other_blocked_server_requirements(self):
        e, r = fixture(queue=True)
        e["before"]["controls"]["rules"][1]["parameters"]["required_approving_review_count"] = 2
        e["before"]["pull"]["mergeable_state"] = "blocked"
        e["after"] = deepcopy(e["before"])
        self.rejected(e, r, "mergeability_unavailable")

    def test_changed_workflow_job_inventory_fails_closed(self):
        e, r = fixture()
        path = next(iter(guard.WORKFLOWS))
        e["workflow_definitions"][path] += "  new-required-job:\n    runs-on: fixture\n"
        self.rejected(e, r, "workflow_definition_invalid")


class CollectorTests(unittest.TestCase):
    def test_real_collector_shapes_return_read_only_receipt(self):
        e, r = fixture()
        api = FakeAPI(e)
        result = guard.inspect(api, PINS, r)
        self.assertEqual(result["state"], "ready_for_operator", result)
        self.assertFalse(result["merge_authorized"])
        self.assertTrue(all(call["method"] == "GET" for call in api.calls))
        self.assertEqual(result["review_receipt_sha256"], guard.digest(r))

    def test_api_failure_or_unavailable_rule_details_never_ready(self):
        for endpoint, reason in (("user", "api_unavailable"),
                ("repos/neverhuman/dope/branches/main/protection", "control_unavailable"),
                ("repos/neverhuman/dope/rules/branches/main", "control_unavailable"),
                ("repos/neverhuman/dope/pulls/7/reviews", "api_unavailable")):
            e, r = fixture()
            api = FakeAPI(e)
            api.fail_endpoint = endpoint
            result = guard.inspect(api, PINS, r)
            self.assertEqual(result["reason"], reason)
            self.assertNotEqual(result["state"], "ready_for_operator")

    def test_collector_observes_head_and_base_race(self):
        for change in (lambda e: e["after"]["pull"]["head"].update(sha=BASE),
                       lambda e: e["after"].update(base_sha=SOURCE)):
            e, r = fixture()
            api = FakeAPI(e)
            api.mutate_after = change
            self.assertEqual(guard.inspect(api, PINS, r)["reason"], "inspection_changed")

    def test_queued_workflow_with_no_jobs_is_fetched_and_blocks(self):
        e, r = fixture()
        e["runs"][0].update(status="queued", conclusion=None, jobs=[])
        api = FakeAPI(e)
        result = guard.inspect(api, PINS, r)
        self.assertEqual(result["reason"], "checks_not_successful")
        self.assertTrue(any("/runs/101/jobs" in call["endpoint"] for call in api.calls))

    def test_missing_or_malformed_receipt_never_becomes_ready(self):
        for receipt in ({}, None, {"ready": True}):
            e, _ = fixture()
            self.assertEqual(guard.inspect(FakeAPI(e), PINS, receipt)["state"], "blocked")

    def test_live_controls_capture_includes_absence_and_exact_base(self):
        e, _ = fixture()
        e["before"]["controls"]["protection"] = None
        result = guard.inspect_controls(FakeAPI(e), PINS["repo"], BASE)
        self.assertEqual(result["state"], "control_unavailable")
        self.assertEqual(result["before_base"], BASE)
        self.assertIsNone(result["before_controls"]["protection"])
        self.assertFalse(result["merge_authorized"])

    def test_transport_uses_only_get_and_never_retries_failed_api(self):
        api = guard.GitHub()
        with patch.object(guard.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout='{"login":"fixture"}')) as run:
            self.assertEqual(api.get("user"), {"login": "fixture"})
        self.assertEqual(run.call_args.args[0], ["rtk", "proxy", "gh", "api", "--method", "GET", "user"])
        with patch.object(guard.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout='{"status":"403"}')) as run:
            with self.assertRaises(guard.Refused): api.get("user")
            self.assertEqual(run.call_count, 1)

    def test_transport_404_and_malformed_response(self):
        api = guard.GitHub()
        with patch.object(guard.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout='{"status":"404"}')):
            self.assertIsNone(api.get("protection", optional=True))
            with self.assertRaises(guard.Refused): api.get("reviews")
        with patch.object(guard.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="not-json")):
            with self.assertRaises(guard.Refused): api.get("user")

    def test_pagination_keeps_second_page_and_refuses_truncated_or_changing_totals(self):
        api = guard.GitHub()
        with patch.object(api, "get", side_effect=[{"total_count": 101, "workflow_runs": [1] * 100},
                                                 {"total_count": 101, "workflow_runs": [2]}]):
            self.assertEqual(len(api.listing("runs", "workflow_runs")), 101)
        for response in ({"total_count": 1000, "workflow_runs": []}, {"total_count": 4, "workflow_runs": [1]}):
            with patch.object(api, "get", return_value=response):
                with self.assertRaises(guard.Refused): api.listing("runs", "workflow_runs")
        with patch.object(api, "get", return_value={"workflow_runs": []}):
            with self.assertRaises(guard.Refused): api.listing("runs", "workflow_runs")


class MergeReceiptTests(unittest.TestCase):
    def test_successful_cli_or_queue_acceptance_remains_awaiting_merge(self):
        for queue in (False, True):
            prior = prior_receipt(queue)
            evidence = merge_fixture(prior, merged=False)
            evidence["cli_exit"] = 0
            evidence["queued"] = True
            self.assertEqual(guard.evaluate_merge(evidence, prior), "awaiting_merge")

    def test_real_merge_commit_tree_base_and_receiving_checks_qualify(self):
        prior = prior_receipt()
        self.assertEqual(guard.evaluate_merge(merge_fixture(prior), prior), "merged_verified")

    def test_full_read_only_verification_collects_merge_and_receiving_jobs(self):
        prior = prior_receipt()
        e = merge_fixture(prior)
        e["operator"] = {"login": guard.OPERATOR}
        api = FakeAPI(e)
        result = guard.inspect(api, PINS, prior=prior)
        self.assertEqual(result["state"], "merged_verified", result)
        self.assertFalse(result["merge_authorized"])
        self.assertTrue(any(call["endpoint"].endswith(MERGE) for call in api.calls))

    def test_full_read_only_queued_verification_does_not_fetch_fake_merge(self):
        prior = prior_receipt(True)
        api = FakeAPI(merge_fixture(prior, merged=False))
        result = guard.inspect(api, PINS, prior=prior)
        self.assertEqual(result["state"], "awaiting_merge", result)
        self.assertFalse(any("/git/commits/" in call["endpoint"] for call in api.calls))

    def test_queue_closure_without_authenticated_group_lineage_blocks(self):
        prior = prior_receipt(True)
        with self.assertRaises(guard.Refused) as caught:
            guard.evaluate_merge(merge_fixture(prior), prior)
        self.assertEqual(caught.exception.reason, "queue_evidence_unavailable")

    def test_wrong_merge_tree_base_receiving_commit_or_missing_checks(self):
        changes = [(lambda e: e["merge_commit"]["tree"].update(sha=BASE), "receiving_tree_mismatch"),
            (lambda e: e["merge_commit"]["parents"][0].update(sha=SOURCE), "receiving_base_mismatch"),
            (lambda e: e["merge_commit"].update(sha=BASE), "receiving_commit_mismatch"),
            (lambda e: e["receiving_runs"].clear(), "missing_workflow"),
            (lambda e: e["before"]["pull"].update(merged_at=None), "merge_not_verified")]
        for change, reason in changes:
            with self.subTest(reason=reason):
                prior = prior_receipt()
                e = merge_fixture(prior)
                change(e)
                e["after"] = deepcopy(e["before"])
                with self.assertRaises(guard.Refused) as caught: guard.evaluate_merge(e, prior)
                self.assertEqual(caught.exception.reason, reason)

    def test_receipt_after_merge_or_forged_ready_scalar_never_qualifies(self):
        prior = prior_receipt()
        prior["captured_at"] = "2026-10-14T02:04:00Z"
        with self.assertRaises(guard.Refused) as caught: guard.evaluate_merge(merge_fixture(prior), prior)
        self.assertEqual(caught.exception.reason, "premerge_receipt_not_premerge")
        prior = prior_receipt()
        prior["state"] = "ready_for_operator"
        prior["evidence"]["runs"] = []
        with self.assertRaises(guard.Refused): guard.evaluate_merge(merge_fixture(prior), prior)


if __name__ == "__main__":
    unittest.main()
