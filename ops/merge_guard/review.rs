//! Source-bound independent approval integrity and revocation.
use super::{OPERATOR, REVIEWER, Result, array, integer, require, text, timestamp, valid_sha};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

fn instant(v: &Value) -> Result<i64> {
    timestamp::parse(text(v)?)
}
fn references(body: &str, case_insensitive: bool) -> Vec<&str> {
    body.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|part| {
            (7..=40).contains(&part.len())
                && part.bytes().all(|c| {
                    c.is_ascii_digit()
                        || (b'a'..=b'f').contains(&c)
                        || (case_insensitive && (b'A'..=b'F').contains(&c))
                })
        })
        .collect()
}
pub fn validate(evidence: &Value, pins: &Value, receipt: &Value, runs: &[u64]) -> Result<()> {
    require(
        receipt.is_object()
            && [
                "version",
                "repo",
                "pr",
                "inspected_source_sha",
                "source_sha",
                "base_sha",
                "source_tree_sha",
                "reviewer",
                "review_id",
                "receipt_locator",
                "check_run_ids",
                "body_sha256",
            ]
            .iter()
            .all(|key| receipt.get(*key).is_some()),
        "invalid_or_incomplete_evidence",
    )?;
    let source = text(&pins["source_sha"])?;
    require(
        receipt["version"].as_u64() == Some(1)
            && receipt["repo"] == pins["repo"]
            && receipt["pr"] == pins["pr"],
        "receipt_identity_mismatch",
    )?;
    require(
        receipt["inspected_source_sha"] == pins["source_sha"]
            && receipt["source_sha"] == pins["source_sha"]
            && receipt["base_sha"] == pins["base_sha"]
            && receipt["source_tree_sha"] == evidence["source_commit"]["tree"]["sha"]
            && valid_sha(&receipt["source_tree_sha"]),
        "receipt_source_mismatch",
    )?;
    require(
        evidence["operator"]["login"] == OPERATOR && receipt["reviewer"] == REVIEWER,
        "reviewer_ineligible",
    )?;
    let pull = &evidence["before"]["pull"];
    let mut excluded = BTreeSet::from([
        text(&pull["user"]["login"])?.to_owned(),
        OPERATOR.to_owned(),
    ]);
    let commits = array(&evidence["commits"])?;
    require(
        integer(&pull["commits"])? == commits.len() as u64
            && !commits.is_empty()
            && commits.len() < 250,
        "commit_provenance_unavailable",
    )?;
    for commit in commits {
        for field in ["author", "committer"] {
            let login = commit[field]["login"].as_str();
            require(
                login.is_some_and(|s| !s.is_empty()),
                "commit_provenance_unavailable",
            )?;
            if let Some(login) = login {
                excluded.insert(login.to_owned());
            }
        }
    }
    require(!excluded.contains(REVIEWER), "reviewer_ineligible")?;
    let permission = &evidence["reviewer_permission"];
    require(
        permission["user"]["login"] == REVIEWER
            && permission["permission"]
                .as_str()
                .is_some_and(|s| ["write", "maintain", "admin"].contains(&s)),
        "reviewer_ineligible",
    )?;
    let reviews = array(&evidence["reviews"])?;
    let matches: Vec<_> = reviews
        .iter()
        .filter(|r| r["id"] == receipt["review_id"])
        .collect();
    require(matches.len() == 1, "review_missing")?;
    let review = matches[0];
    require(review["user"]["login"] == REVIEWER, "reviewer_ineligible")?;
    require(
        review["state"] == "APPROVED" && review["commit_id"] == pins["source_sha"],
        "approval_stale",
    )?;
    let locator = text(&receipt["receipt_locator"])?;
    require(
        !locator.is_empty() && !locator.contains('\n'),
        "receipt_identity_mismatch",
    )?;
    let ids = runs
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let expected = format!("Reviewed source {source}; receipt {locator}; checks {ids}.");
    let body = text(&review["body"])?;
    let full: BTreeSet<_> = references(body, false)
        .into_iter()
        .filter(|r| r.len() == 40)
        .collect();
    require(
        body.lines().next() == Some(expected.as_str())
            && full == BTreeSet::from([source])
            && references(body, true).iter().all(|r| {
                r.bytes().all(|c| c.is_ascii_digit()) || source.starts_with(&r.to_ascii_lowercase())
            })
            && receipt["body_sha256"] == super::json::hash_bytes(body.as_bytes()),
        "approval_body_mismatch",
    )?;
    require(
        receipt["check_run_ids"] == json!(runs),
        "approval_checks_stale",
    )?;
    let approved = instant(&review["submitted_at"])?;
    for run in array(&evidence["runs"])? {
        require(
            instant(&run["updated_at"])? <= approved,
            "approval_checks_stale",
        )?;
    }
    for r in reviews {
        require(
            !(r["user"]["login"] == REVIEWER
                && r["state"] != "COMMENTED"
                && instant(&r["submitted_at"])? > approved),
            "approval_stale",
        )?;
    }
    require(
        approved <= instant(&evidence["inspection_completed_at"])?,
        "approval_stale",
    )?;
    let mut sorted: Vec<_> = reviews
        .iter()
        .map(|r| Ok((instant(&r["submitted_at"])?, r)))
        .collect::<Result<_>>()?;
    sorted.sort_by_key(|(t, _)| *t);
    let mut latest = BTreeMap::new();
    for (_, r) in sorted {
        if r["state"] != "COMMENTED" {
            latest.insert(text(&r["user"]["login"])?, text(&r["state"])?);
        }
    }
    require(
        !latest.values().any(|v| *v == "CHANGES_REQUESTED"),
        "changes_requested",
    )
}
