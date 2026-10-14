//! Pure source, open-PR, preflight, and actual squash-merge proof.
use super::{Refused, Result, array, controls, require, review, runs, text, timestamp, valid_sha};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub fn pull_identity(pull: &Value) -> Result<Value> {
    let number = super::integer(&pull["number"])?;
    require(
        number > 0 && pull["draft"].is_boolean(),
        "open_pr_listing_unavailable",
    )?;
    let mut identity = json!({"number":number,"state":pull["state"],"draft":pull["draft"]});
    for field in ["head", "base"] {
        let r = &pull[field];
        require(
            valid_sha(&r["sha"])
                && r["ref"].as_str().is_some_and(|s| !s.is_empty())
                && r["repo"]["full_name"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty()),
            "open_pr_listing_unavailable",
        )?;
        identity[field] =
            json!({"sha":r["sha"],"ref":r["ref"],"repo":{"full_name":r["repo"]["full_name"]}});
    }
    Ok(identity)
}
pub fn normalize_open_pulls(pulls: &Value) -> Result<Value> {
    let parsed = (|| {
        let mut values = array(pulls)?
            .iter()
            .map(pull_identity)
            .collect::<Result<Vec<_>>>()?;
        require(
            values.iter().all(|p| p["state"] == "open"),
            "open_pr_listing_unavailable",
        )?;
        let ids: BTreeSet<_> = values
            .iter()
            .map(|p| super::integer(&p["number"]))
            .collect::<Result<_>>()?;
        require(ids.len() == values.len(), "open_pr_listing_changed")?;
        values.sort_by_key(|p| p["number"].as_u64());
        Ok(Value::Array(values))
    })();
    parsed.map_err(|e: Refused| {
        if e.0 == "invalid_or_incomplete_evidence" {
            Refused("open_pr_listing_unavailable")
        } else {
            e
        }
    })
}
pub fn validate_no_stacking(snapshot: &Value, pins: &Value) -> Result<()> {
    let pulls = array(&snapshot["open_prs"])?;
    require(pulls.len() <= 1, "multiple_open_prs")?;
    require(
        pulls.len() == 1 && pulls[0]["number"] == pins["pr"],
        "inspected_pr_not_open",
    )?;
    require(
        pulls[0] == pull_identity(&snapshot["pull"])?,
        "open_pr_identity_mismatch",
    )
}
pub fn evaluate_preflight(e: &Value, pins: &Value, receipt: &Value) -> Result<controls::Policy> {
    let before = &e["before"];
    let after = &e["after"];
    require(before == after, "inspection_changed")?;
    let pull = &before["pull"];
    require(
        pull["number"] == pins["pr"] && pull["head"]["sha"] == pins["source_sha"],
        "head_moved",
    )?;
    require(
        pull["head"]["repo"]["full_name"] == pins["repo"],
        "source_repo_mismatch",
    )?;
    require(
        pull["base"]["sha"] == pins["base_sha"]
            && before["base_sha"] == pins["base_sha"]
            && pull["base"]["ref"] == pins["base_branch"]
            && pull["base"]["repo"]["full_name"] == pins["repo"],
        "base_moved",
    )?;
    require(
        pull["state"] == "open" && pull["merged"] == false && pull["draft"] == false,
        "pr_not_open",
    )?;
    validate_no_stacking(before, pins)?;
    let policy = controls::qualify(&before["controls"])?;
    runs::validate_definitions(&e["workflow_definitions"], policy.queue())?;
    require(
        e["source_commit"]["sha"] == pins["source_sha"],
        "head_moved",
    )?;
    let ids = runs::validate_runs(
        &e["runs"],
        &pins["source_sha"],
        &["push", "pull_request"],
        Some(&pins["pr"]),
        Some(&pull["head"]["ref"]),
        Some(&pins["base_sha"]),
    )?;
    review::validate(e, pins, receipt, &ids)?;
    require(
        pull["mergeable_state"] == "clean",
        "mergeability_unavailable",
    )?;
    Ok(policy)
}
pub fn evaluate_merge(e: &Value, prior: &Value) -> Result<&'static str> {
    let pins = &prior["pins"];
    let receipt = &prior["review_receipt"];
    let policy = evaluate_preflight(&prior["evidence"], pins, receipt)?;
    let before = &e["before"];
    let pull = &before["pull"];
    require(before == &e["after"], "inspection_changed")?;
    require(
        pull["number"] == pins["pr"] && pull["head"]["sha"] == pins["source_sha"],
        "head_moved",
    )?;
    require(
        pull["head"]["repo"]["full_name"] == pins["repo"],
        "source_repo_mismatch",
    )?;
    require(
        pull["base"]["ref"] == pins["base_branch"]
            && pull["base"]["repo"]["full_name"] == pins["repo"],
        "base_moved",
    )?;
    require(
        before["controls"] == prior["evidence"]["after"]["controls"],
        "controls_changed",
    )?;
    require(e["reviews"] == e["reviews_after"], "inspection_changed")?;
    let mut current = prior["evidence"].clone();
    require(current.is_object(), "invalid_or_incomplete_evidence")?;
    current["before"] = before.clone();
    current["reviews"] = e["reviews"].clone();
    current["inspection_completed_at"] = e["inspection_completed_at"].clone();
    let ids = array(&receipt["check_run_ids"])?
        .iter()
        .map(super::integer)
        .collect::<Result<Vec<_>>>()?;
    review::validate(&current, pins, receipt, &ids)?;
    if pull["state"] == "open" && pull["merged"] == false && pull["merged_at"].is_null() {
        require(
            before["base_sha"] == pins["base_sha"] && pull["base"]["sha"] == pins["base_sha"],
            "base_moved",
        )?;
        validate_no_stacking(before, pins)?;
        return Ok("awaiting_merge");
    }
    require(
        pull["state"] == "closed"
            && pull["merged"] == true
            && pull["merged_at"].as_str().is_some_and(|s| !s.is_empty()),
        "merge_not_verified",
    )?;
    require(
        timestamp::parse(text(&prior["captured_at"])?)?
            < timestamp::parse(text(&pull["merged_at"])?)?,
        "premerge_receipt_not_premerge",
    )?;
    require(
        timestamp::parse(text(&prior["evidence"]["inspection_completed_at"])?)?
            <= timestamp::parse(text(&prior["captured_at"])?)?,
        "premerge_receipt_not_premerge",
    )?;
    let commit = &e["merge_commit"];
    require(
        valid_sha(&pull["merge_commit_sha"])
            && commit["sha"] == pull["merge_commit_sha"]
            && before["base_sha"] == commit["sha"],
        "receiving_commit_mismatch",
    )?;
    require(
        commit["tree"]["sha"] == receipt["source_tree_sha"],
        "receiving_tree_mismatch",
    )?;
    let parents = array(&commit["parents"])?;
    require(
        parents.len() == 1 && parents[0]["sha"] == pins["base_sha"],
        "receiving_base_mismatch",
    )?;
    runs::validate_runs(
        &e["receiving_runs"],
        &commit["sha"],
        &["push"],
        None,
        Some(&pins["base_branch"]),
        None,
    )?;
    require(!policy.queue(), "queue_evidence_unavailable")?;
    Ok("merged_verified")
}
