#!/usr/bin/env python3
"""Read-only, fail-closed DOPE merge inspection. Never approves, enqueues or merges."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote


WORKFLOWS = {
    ".github/workflows/ci.yml": ("cpu", "paper-clean", "security"),
    ".github/workflows/jankurai.yml": ("jankurai-audit",),
}
CHECKS = {job for jobs in WORKFLOWS.values() for job in jobs}
REVIEWER = "neverhumanbot"
OPERATOR = "jepsontaylor"
SHA = re.compile(r"[0-9a-f]{40}\Z")


class Refused(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def require(condition, reason):
    if not condition:
        raise Refused(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "invalid_timestamp")
    return result


class GitHub:
    """Only GET requests; bounded pagination and per-request time, without retries."""

    def __init__(self, executable="gh"):
        self.executable = executable
        self.calls = []

    def get(self, endpoint, optional=False):
        command = ["rtk", "proxy", self.executable, "api", "--method", "GET", endpoint]
        call = {"endpoint": endpoint, "method": "GET", "state": "started"}
        self.calls.append(call)
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=45)
            data = json.loads(response.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            call["state"] = "api_unavailable"
            raise Refused("api_unavailable") from None
        call.update(state="completed", exit_code=response.returncode, sha256=digest(data))
        if response.returncode:
            if optional and isinstance(data, dict) and str(data.get("status")) == "404":
                return None
            raise Refused("api_unavailable")
        return data

    def listing(self, endpoint, key=None):
        items = []
        total = None
        for page in range(1, 11):
            separator = "&" if "?" in endpoint else "?"
            data = self.get(f"{endpoint}{separator}per_page=100&page={page}")
            if key:
                require(isinstance(data, dict) and isinstance(data.get(key), list), "api_shape_invalid")
                require("total_count" in data, "api_shape_invalid")
                if "total_count" in data:
                    require(isinstance(data["total_count"], int), "api_shape_invalid")
                    require(total is None or total == data["total_count"], "api_listing_changed")
                    total = data["total_count"]
                    require(total < 1000, "api_listing_truncated")
                batch = data[key]
            else:
                require(isinstance(data, list), "api_shape_invalid")
                batch = data
            items.extend(batch)
            if len(batch) < 100:
                require(total is None or len(items) == total, "api_listing_truncated")
                return items
        raise Refused("api_listing_truncated")


def capture_controls(api, root, branch):
    try:
        protection = api.get(f"{root}/branches/{quote(branch, safe='')}/protection", optional=True)
        rules = api.listing(f"{root}/rules/branches/{quote(branch, safe='')}")
        details = {}
        for rule in rules:
            identifier = rule["ruleset_id"]
            require(isinstance(identifier, int) and identifier > 0, "control_unavailable")
            if str(identifier) not in details:
                details[str(identifier)] = api.get(f"{root}/rulesets/{identifier}?includes_parents=true")
        return {"protection": protection, "rules": rules, "rulesets": details}
    except (Refused, KeyError, TypeError):
        raise Refused("control_unavailable") from None


def qualify_controls(controls):
    """Require visible, enforced reviews/checks and an explicitly empty bypass policy."""
    protection = controls["protection"]
    rules = controls["rules"]
    queue = any(rule["type"] == "merge_queue" for rule in rules)
    classic_qualified = False
    strict = reviews = False
    if protection:
        checks = protection.get("required_status_checks") or {}
        contexts = set(checks.get("contexts", [])) | {c["context"] for c in checks.get("checks", [])}
        require(contexts <= CHECKS, "control_unavailable")
        review = protection.get("required_pull_request_reviews") or {}
        strict = checks.get("strict") is True and contexts == CHECKS
        reviews = (review.get("required_approving_review_count", 0) >= 1
                   and review.get("dismiss_stale_reviews") is True
                   and review.get("require_last_push_approval") is True)
        allowances = review.get("bypass_pull_request_allowances", {})
        bypass = (protection.get("enforce_admins", {}).get("enabled") is True
                  and isinstance(allowances, dict) and not any(allowances.values())
                  and protection.get("allow_force_pushes", {}).get("enabled") is False
                  and protection.get("allow_deletions", {}).get("enabled") is False)
        classic_qualified = strict and reviews and bypass
    rule_qualified = False
    if rules:
        # Bypass information can be omitted for readers lacking rule-write visibility.
        for rule in rules:
            detail = controls["rulesets"].get(str(rule["ruleset_id"]), {})
            require(detail.get("id") == rule["ruleset_id"]
                    and detail.get("enforcement") == "active"
                    and detail.get("bypass_actors") == [], "control_unavailable")
        rule_checks = []
        rule_reviews = False
        for rule in rules:
            parameters = rule.get("parameters", {})
            if rule["type"] == "required_status_checks":
                contexts = {c["context"] for c in parameters["required_status_checks"]}
                rule_checks.append((contexts, parameters.get("strict_required_status_checks_policy") is True))
            if rule["type"] == "pull_request":
                rule_reviews |= (parameters.get("required_approving_review_count", 0) >= 1
                                 and parameters.get("dismiss_stale_reviews_on_push") is True
                                 and parameters.get("require_last_push_approval") is True)
        require(all(contexts <= CHECKS for contexts, _ in rule_checks), "control_unavailable")
        all_contexts = set().union(*(contexts for contexts, _ in rule_checks))
        rule_strict = all_contexts == CHECKS and all(up_to_date for _, up_to_date in rule_checks)
        queue = queue and all_contexts == CHECKS
        rule_qualified = rule_reviews and (rule_strict or queue)
    require(classic_qualified or rule_qualified, "control_unavailable")
    # An enforced queue also needs an enforced review/check policy in rulesets.
    require(not queue or rule_qualified, "control_unavailable")
    return {"base_policy": "queue" if queue else "strict", "required_checks": sorted(CHECKS),
            "independent_review_required": True, "stale_approval_dismissal": True,
            "last_push_approval_required": True, "bypass_policy": "none"}


def snapshot(api, root, pr, branch):
    pull = api.get(f"{root}/pulls/{pr}")
    # Keep the relevant API identity/state, rather than unrelated PR text.
    pull = {key: pull[key] for key in (
        "number", "state", "draft", "head", "base", "user", "commits", "merged",
        "merged_at", "merge_commit_sha", "mergeable_state")}
    base = api.get(f"{root}/git/ref/heads/{quote(branch, safe='')}")["object"]["sha"]
    controls = capture_controls(api, root, branch)
    return {"pull": pull, "base_sha": base, "controls": controls}


def workflow_definitions(api, root, source):
    definitions = {}
    for path in WORKFLOWS:
        item = api.get(f"{root}/contents/{path}?ref={source}")
        require(item.get("encoding") == "base64", "workflow_definition_invalid")
        definitions[path] = base64.b64decode(item["content"], validate=False).decode()
    return definitions


def workflow_block(text, name):
    # This repository uses block-form YAML. Unknown forms fail closed, without a
    # new YAML dependency or a permissive fallback to the visible green jobs.
    match = re.search(rf"^{name}:\s*\n((?:(?:[ \t].*|)\n)*)", text, re.M)
    require(match is not None, "workflow_definition_invalid")
    return match.group(1)


def validate_definitions(definitions, queue):
    require(set(definitions) == set(WORKFLOWS), "workflow_definition_invalid")
    for path, jobs in WORKFLOWS.items():
        text = definitions[path]
        events = set(re.findall(r"^  ([a-z_]+):", workflow_block(text, "on"), re.M))
        expected = {"push", "pull_request"} | ({"merge_group"} if queue else set())
        require(expected <= events, "control_unavailable" if queue else "workflow_definition_invalid")
        names = set(re.findall(r"^  ([a-z_-]+):", workflow_block(text, "jobs"), re.M))
        require(names == set(jobs), "workflow_definition_invalid")


def collect_runs(api, root, source):
    runs = api.listing(f"{root}/actions/runs?head_sha={source}", "workflow_runs")
    collected = []
    for run in runs:
        if run.get("path", "").split("@")[0] in WORKFLOWS:
            run = dict(run)
            # Inspect queued workflow runs even when the jobs endpoint is empty.
            run["jobs"] = api.listing(f"{root}/actions/runs/{run['id']}/jobs?filter=all", "jobs")
            collected.append(run)
    return collected


def validate_runs(runs, source, events, pr=None, branch=None, base=None):
    seen = set()
    identifiers = []
    for run in runs:
        path = run["path"].split("@")[0]
        require(path in WORKFLOWS and run["head_sha"] == source, "check_source_mismatch")
        require(run["event"] in events, "unexpected_workflow_event")
        if run["event"] == "pull_request":
            require(any(p["number"] == pr and p["head"]["sha"] == source
                        and p["base"]["sha"] == base for p in run["pull_requests"]), "check_pr_mismatch")
        if branch:
            require(run["head_branch"] == branch, "check_branch_mismatch")
        require(run["status"] == "completed" and run["conclusion"] == "success", "checks_not_successful")
        require(run.get("run_attempt", 1) == 1, "retry_requires_separate_admission")
        jobs = run["jobs"]
        require({job["name"] for job in jobs} == set(WORKFLOWS[path])
                and len(jobs) == len(WORKFLOWS[path]), "missing_or_duplicate_jobs")
        for job in jobs:
            require(job["head_sha"] == source, "check_source_mismatch")
            require(job["status"] == "completed" and job["conclusion"] == "success", "checks_not_successful")
        seen.add((path, run["event"]))
        identifiers.append(run["id"])
    require(seen == {(path, event) for path in WORKFLOWS for event in events}, "missing_workflow")
    require(len(set(identifiers)) == len(identifiers), "duplicate_workflow_run")
    return sorted(identifiers)


def validate_review(evidence, pins, receipt, runs):
    source = pins["source_sha"]
    require(receipt["version"] == 1 and receipt["repo"] == pins["repo"] and receipt["pr"] == pins["pr"],
            "receipt_identity_mismatch")
    require(receipt["inspected_source_sha"] == receipt["source_sha"] == source
            and receipt["base_sha"] == pins["base_sha"]
            and receipt["source_tree_sha"] == evidence["source_commit"]["tree"]["sha"], "receipt_source_mismatch")
    require(SHA.fullmatch(receipt["source_tree_sha"]), "receipt_source_mismatch")
    require(evidence["operator"]["login"] == OPERATOR and receipt["reviewer"] == REVIEWER,
            "reviewer_ineligible")
    pull = evidence["before"]["pull"]
    excluded = {pull["user"]["login"], OPERATOR}
    commits = evidence["commits"]
    require(len(commits) == pull["commits"] and 0 < len(commits) < 250, "commit_provenance_unavailable")
    for commit in commits:
        for field in ("author", "committer"):
            identity = commit.get(field)
            require(isinstance(identity, dict) and bool(identity.get("login")), "commit_provenance_unavailable")
            excluded.add(identity["login"])
    require(REVIEWER not in excluded, "reviewer_ineligible")
    permission = evidence["reviewer_permission"]
    require(permission["user"]["login"] == REVIEWER and permission["permission"] in ("write", "maintain", "admin"),
            "reviewer_ineligible")
    reviews = evidence["reviews"]
    matches = [r for r in reviews if r["id"] == receipt["review_id"]]
    require(len(matches) == 1, "review_missing")
    review = matches[0]
    require(review["user"]["login"] == REVIEWER, "reviewer_ineligible")
    require(review["state"] == "APPROVED" and review["commit_id"] == source, "approval_stale")
    require(bool(receipt["receipt_locator"]) and "\n" not in receipt["receipt_locator"], "receipt_identity_mismatch")
    expected = f"Reviewed source {source}; receipt {receipt['receipt_locator']}; checks {','.join(map(str, runs))}."
    require(review["body"].splitlines()[0] == expected
            and set(re.findall(r"\b[0-9a-f]{40}\b", review["body"])) == {source}
            and hashlib.sha256(review["body"].encode()).hexdigest() == receipt["body_sha256"], "approval_body_mismatch")
    require(receipt["check_run_ids"] == runs, "approval_checks_stale")
    require(all(timestamp(run["updated_at"]) <= timestamp(review["submitted_at"])
                for run in evidence["runs"]), "approval_checks_stale")
    # COMMENTED does not revoke an approval. A later non-comment review does.
    later = [r for r in reviews if r["user"]["login"] == REVIEWER and r["state"] != "COMMENTED"
             and timestamp(r["submitted_at"]) > timestamp(review["submitted_at"])]
    require(not later, "approval_stale")
    require(timestamp(review["submitted_at"]) <= timestamp(evidence["inspection_completed_at"]), "approval_stale")
    latest = {}
    for item in sorted(reviews, key=lambda r: timestamp(r["submitted_at"])):
        if item["state"] != "COMMENTED":
            latest[item["user"]["login"]] = item["state"]
    require("CHANGES_REQUESTED" not in latest.values(), "changes_requested")
    return review


def evaluate_preflight(evidence, pins, receipt):
    before, after = evidence["before"], evidence["after"]
    require(before == after, "inspection_changed")
    pull = before["pull"]
    require(pull["number"] == pins["pr"] and pull["head"]["sha"] == pins["source_sha"], "head_moved")
    require(pull["base"]["sha"] == before["base_sha"] == pins["base_sha"]
            and pull["base"]["ref"] == pins["base_branch"]
            and pull["base"]["repo"]["full_name"] == pins["repo"], "base_moved")
    require(pull["state"] == "open" and pull["merged"] is False and not pull["draft"], "pr_not_open")
    policy = qualify_controls(before["controls"])
    validate_definitions(evidence["workflow_definitions"], policy["base_policy"] == "queue")
    require(evidence["source_commit"]["sha"] == pins["source_sha"], "head_moved")
    runs = validate_runs(evidence["runs"], pins["source_sha"], {"push", "pull_request"}, pins["pr"],
                         branch=pull["head"]["ref"], base=pins["base_sha"])
    validate_review(evidence, pins, receipt, runs)
    # A queue alone cannot explain "blocked": it could also mean an unmet
    # approval, conversation or other server requirement. Unknown stays blocked.
    require(pull["mergeable_state"] == "clean", "mergeability_unavailable")
    return policy


def collect_preflight(api, pins):
    root = "repos/" + pins["repo"]
    before = snapshot(api, root, pins["pr"], pins["base_branch"])
    evidence = {"before": before, "operator": api.get("user"),
                "source_commit": api.get(f"{root}/git/commits/{pins['source_sha']}"),
                "reviewer_permission": api.get(f"{root}/collaborators/{REVIEWER}/permission"),
                "commits": api.listing(f"{root}/pulls/{pins['pr']}/commits"),
                "reviews": api.listing(f"{root}/pulls/{pins['pr']}/reviews"),
                "workflow_definitions": workflow_definitions(api, root, pins["source_sha"]),
                "runs": collect_runs(api, root, pins["source_sha"])}
    # Re-read approvals/runs as well as the head/base/rules: dismissals and new
    # no-job workflow runs during the inspection must invalidate the receipt.
    require(evidence["reviews"] == api.listing(f"{root}/pulls/{pins['pr']}/reviews"), "inspection_changed")
    require(evidence["runs"] == collect_runs(api, root, pins["source_sha"]), "inspection_changed")
    evidence["after"] = snapshot(api, root, pins["pr"], pins["base_branch"])
    evidence["inspection_completed_at"] = datetime.now(timezone.utc).isoformat()
    return evidence


def evaluate_merge(evidence, prior):
    # Re-evaluate the complete premerge observation, never trust a serialized
    # ready=true flag. This is still an advisory receipt, not server authority.
    pins, receipt = prior["pins"], prior["review_receipt"]
    policy = evaluate_preflight(prior["evidence"], pins, receipt)
    before, after = evidence["before"], evidence["after"]
    require(before == after, "inspection_changed")
    pull = before["pull"]
    require(pull["number"] == pins["pr"] and pull["head"]["sha"] == pins["source_sha"], "head_moved")
    require(pull["base"]["ref"] == pins["base_branch"]
            and pull["base"]["repo"]["full_name"] == pins["repo"], "base_moved")
    require(before["controls"] == prior["evidence"]["after"]["controls"], "controls_changed")
    if pull["state"] == "open" and pull["merged"] is False and pull["merged_at"] is None:
        require(before["base_sha"] == pins["base_sha"] and pull["base"]["sha"] == pins["base_sha"], "base_moved")
        return "awaiting_merge"
    require(pull["state"] == "closed" and pull["merged"] is True and bool(pull["merged_at"]), "merge_not_verified")
    require(timestamp(prior["captured_at"]) < timestamp(pull["merged_at"]), "premerge_receipt_not_premerge")
    require(timestamp(prior["evidence"]["inspection_completed_at"]) <= timestamp(prior["captured_at"]),
            "premerge_receipt_not_premerge")
    commit = evidence["merge_commit"]
    require(bool(pull["merge_commit_sha"]) and commit["sha"] == pull["merge_commit_sha"]
            and before["base_sha"] == commit["sha"], "receiving_commit_mismatch")
    require(commit["tree"]["sha"] == receipt["source_tree_sha"], "receiving_tree_mismatch")
    # This wave admits squash receipts. Unknown/rebase/queue lineage stays blocked.
    require([parent["sha"] for parent in commit["parents"]] == [pins["base_sha"]], "receiving_base_mismatch")
    validate_runs(evidence["receiving_runs"], commit["sha"], {"push"}, branch=pins["base_branch"])
    if policy["base_policy"] == "queue":
        # This bounded worker has no authenticated queue-group lineage adapter.
        # A caller-supplied JSON group or CLI exit cannot establish that lineage.
        raise Refused("queue_evidence_unavailable")
    return "merged_verified"


def validate_pins(pins):
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", pins["repo"]) is not None, "invalid_pins")
    require(type(pins["pr"]) is int and pins["pr"] > 0 and pins["base_branch"] == "main", "invalid_pins")
    require(SHA.fullmatch(pins["source_sha"]) and SHA.fullmatch(pins["base_sha"]), "invalid_pins")


def inspect_controls(api, repo, base):
    result = {"version": 1, "read_only": True, "merge_authorized": False, "state": "blocked",
              "captured_at": datetime.now(timezone.utc).isoformat(), "repo": repo, "base_sha": base}
    try:
        validate_pins({"repo": repo, "pr": 1, "base_branch": "main", "source_sha": base, "base_sha": base})
        root = "repos/" + repo
        result["operator"] = api.get("user")
        result["before_base"] = api.get(f"{root}/git/ref/heads/main")["object"]["sha"]
        result["before_controls"] = capture_controls(api, root, "main")
        result["after_controls"] = capture_controls(api, root, "main")
        result["after_base"] = api.get(f"{root}/git/ref/heads/main")["object"]["sha"]
        require(result["before_base"] == result["after_base"] == base, "base_moved")
        require(result["before_controls"] == result["after_controls"], "controls_changed")
        result["policy"] = qualify_controls(result["after_controls"])
        result["state"] = "server_policy_observed"
    except Refused as error:
        result["reason"] = error.reason
        if error.reason == "control_unavailable":
            result["state"] = "control_unavailable"
    except (KeyError, TypeError, ValueError, AttributeError):
        result["reason"] = "invalid_or_incomplete_evidence"
    result["api_calls"] = api.calls
    return result


def inspect(api, pins, receipt=None, prior=None):
    result = {"version": 1, "read_only": True, "captured_at": datetime.now(timezone.utc).isoformat(),
              "pins": pins, "state": "blocked", "merge_authorized": False}
    try:
        validate_pins(pins)
        if prior is None:
            result["review_receipt"] = receipt
            result["review_receipt_sha256"] = digest(receipt)
            result["evidence"] = collect_preflight(api, pins)
            result["captured_at"] = result["evidence"]["inspection_completed_at"]
            result["policy"] = evaluate_preflight(result["evidence"], pins, receipt)
            result["state"] = "ready_for_operator"
        else:
            require(prior["pins"] == pins, "receipt_identity_mismatch")
            evaluate_preflight(prior["evidence"], pins, prior["review_receipt"])
            root = "repos/" + pins["repo"]
            before = snapshot(api, root, pins["pr"], pins["base_branch"])
            result["prior_receipt_sha256"] = digest(prior)
            result["evidence"] = {"before": before}
            if before["pull"]["merged"]:
                merge = before["pull"]["merge_commit_sha"]
                require(isinstance(merge, str) and SHA.fullmatch(merge), "merge_not_verified")
                result["evidence"]["merge_commit"] = api.get(f"{root}/git/commits/{merge}")
                result["evidence"]["receiving_runs"] = collect_runs(api, root, merge)
            result["evidence"]["after"] = snapshot(api, root, pins["pr"], pins["base_branch"])
            result["state"] = evaluate_merge(result["evidence"], prior)
    except Refused as error:
        result["reason"] = error.reason
        if error.reason == "control_unavailable":
            result["state"] = "control_unavailable"
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        result["reason"] = "invalid_or_incomplete_evidence"
    result["api_calls"] = api.calls
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="neverhuman/dope")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--source")
    parser.add_argument("--base", required=True)
    parser.add_argument("--gh", default="gh", help="Existing gh executable; credentials are never switched")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--review-receipt", type=Path)
    inputs.add_argument("--verify-merge", type=Path, help="Prior full premerge receipt")
    inputs.add_argument("--controls-only", action="store_true", help="Capture live server rules without a review fixture")
    args = parser.parse_args()
    if args.controls_only:
        result = inspect_controls(GitHub(args.gh), args.repo, args.base)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["state"] == "server_policy_observed" else 1
    if args.pr is None or args.source is None:
        parser.error("--pr and --source are required for review/merge inspection")
    pins = {"repo": args.repo, "pr": args.pr, "source_sha": args.source,
            "base_sha": args.base, "base_branch": "main"}
    try:
        receipt = json.loads((args.review_receipt or args.verify_merge).read_text())
    except (OSError, ValueError):
        print(json.dumps({"state": "blocked", "reason": "receipt_unavailable", "merge_authorized": False}))
        return 1
    result = inspect(GitHub(args.gh), pins, receipt=receipt if args.review_receipt else None,
                     prior=receipt if args.verify_merge else None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] in ("ready_for_operator", "merged_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
