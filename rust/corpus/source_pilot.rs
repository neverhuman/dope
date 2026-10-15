//! Exact development source-pilot v2 consumer. A caller-trusted digest binds
//! metadata without verifying custody keys, rights, observed receipts or admission.

use crate::error::{DopeError, Result};
use crate::evidence::{canonical_json, parse_json};
use serde_json::{Number, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub const MAX_PILOT_DOCUMENT_BYTES: usize = 4 * 1024 * 1024;
const MAX_JOBS: usize = 4096;
const MAX_INTEGER: u64 = (1 << 53) - 1;
const STAGES: [&str; 11] = [
    "fit",
    "sample",
    "learner",
    "selection",
    "probe",
    "pfi",
    "geometry",
    "calibration",
    "real_reference",
    "ensemble",
    "cache_hit",
];
const PARTITIONS: [&str; 3] = [
    "development_fit",
    "development_selection",
    "development_query",
];
const REASONS: [&str; 9] = [
    "owner_or_identity_unavailable",
    "source_identity_conflict",
    "lineage_permission_or_exposure_conflict",
    "historical_or_non_development_source",
    "producer_role_contamination",
    "rights_unknown",
    "rights_denied",
    "exposure_unavailable_or_contaminated",
    "lineage_partition_conflict",
];
const PLAN_FIELDS: [&str; 20] = [
    "confirmation_allowed",
    "cost_basis",
    "estimated_seconds",
    "execution_authorized",
    "format",
    "group_token_scheme",
    "jobs",
    "observed_seconds",
    "owner_manifest_commitment",
    "planned_attempts",
    "planned_independent_sources",
    "registry_content_commitment",
    "scientific_admission",
    "scope",
    "seed",
    "signed_admission",
    "source_groups",
    "total_seconds_cap",
    "unavailable_sources",
    "version",
];
type JobKey = (String, String, u64);

fn error(code: &'static str) -> DopeError {
    DopeError::Data(code.into())
}
fn require(condition: bool, code: &'static str) -> Result<()> {
    if condition { Ok(()) } else { Err(error(code)) }
}
fn fields(value: &Value, names: &[&str], code: &'static str) -> Result<()> {
    let object = value.as_object().ok_or_else(|| error(code))?;
    require(
        object.len() == names.len() && names.iter().all(|name| object.contains_key(*name)),
        code,
    )
}
fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn integer(value: &Value, maximum: u64, code: &'static str) -> Result<u64> {
    value
        .as_u64()
        .filter(|v| *v <= maximum)
        .ok_or_else(|| error(code))
}
fn amount<'a>(value: &'a Value, code: &'static str) -> Result<&'a Number> {
    value
        .as_number()
        .filter(|v| v.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0))
        .ok_or_else(|| error(code))
}
fn add_amount(total: &mut Number, value: &Number, code: &'static str) -> Result<()> {
    *total = if !total.is_f64() && !value.is_f64() {
        total
            .as_u64()
            .and_then(|a| value.as_u64().and_then(|b| a.checked_add(b)))
            .filter(|sum| *sum <= MAX_INTEGER)
            .map(Number::from)
            .ok_or_else(|| error(code))?
    } else {
        Number::from_f64(total.as_f64().unwrap() + value.as_f64().unwrap())
            .ok_or_else(|| error(code))?
    };
    Ok(())
}
fn document(bytes: &[u8], code: &'static str) -> Result<Value> {
    require(
        bytes.len() <= MAX_PILOT_DOCUMENT_BYTES,
        "pilot_document_bytes_exceeded",
    )?;
    parse_json(bytes).map_err(|_| error(code))
}
fn digest(value: &Value) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(canonical_json(value).map_err(|_| error("pilot_plan_invalid"))?)
    ))
}
fn partition(seed: &Value, token: &str) -> Result<&'static str> {
    let bytes = canonical_json(&json!([seed, token])).map_err(|_| error("pilot_plan_invalid"))?;
    let index = Sha256::digest(bytes)
        .iter()
        .fold(0usize, |n, byte| (n * 256 + usize::from(*byte)) % 3);
    Ok(PARTITIONS[index])
}
fn job_key(value: &Value, code: &'static str) -> Result<JobKey> {
    let group = value["group"]
        .as_str()
        .filter(|v| is_digest(v))
        .ok_or_else(|| error(code))?;
    let stage = value["stage"]
        .as_str()
        .filter(|v| STAGES.contains(v))
        .ok_or_else(|| error(code))?;
    let attempt = integer(&value["attempt"], 128, code)?;
    require(attempt > 0, code)?;
    Ok((group.into(), stage.into(), attempt))
}

fn checked_plan(plan: &Value) -> Result<BTreeSet<JobKey>> {
    let invalid = "pilot_plan_invalid";
    fields(plan, &PLAN_FIELDS, invalid)?;
    require(
        plan["format"] == "dope-development-source-pilot-v2"
            && integer(&plan["version"], 2, invalid)? == 2
            && plan["scope"] == "development_only"
            && plan["cost_basis"] == "estimate_not_observation"
            && plan["group_token_scheme"] == "hmac-sha256-custody-key-v1"
            && plan["observed_seconds"].is_null(),
        invalid,
    )?;
    for flag in [
        "execution_authorized",
        "scientific_admission",
        "confirmation_allowed",
        "signed_admission",
    ] {
        require(plan[flag].as_bool() == Some(false), invalid)?;
    }
    integer(&plan["seed"], MAX_INTEGER, "pilot_seed_invalid")?;
    for commitment in ["owner_manifest_commitment", "registry_content_commitment"] {
        require(plan[commitment].as_str().is_some_and(is_digest), invalid)?;
    }
    require(
        amount(&plan["total_seconds_cap"], invalid)?
            .as_f64()
            .unwrap()
            > 0.0,
        invalid,
    )?;
    let groups = plan["source_groups"]
        .as_array()
        .ok_or_else(|| error(invalid))?;
    let jobs = plan["jobs"].as_array().ok_or_else(|| error(invalid))?;
    require(
        !groups.is_empty() && groups.len() <= MAX_JOBS && jobs.len() <= MAX_JOBS,
        "pilot_job_budget_exceeded",
    )?;
    let mut planned = BTreeSet::new();
    let mut unavailable = 0u64;
    let mut views = 0u64;
    for group in groups {
        fields(
            group,
            &["availability", "group", "partition", "reasons", "views"],
            invalid,
        )?;
        let count = integer(&group["views"], MAX_JOBS as u64, invalid)?;
        require(count > 0, invalid)?;
        views += count;
        require(views <= MAX_JOBS as u64, invalid)?;
        let reasons = group["reasons"].as_array().ok_or_else(|| error(invalid))?;
        require(
            reasons.len() <= REASONS.len()
                && reasons
                    .iter()
                    .all(|v| v.as_str().is_some_and(|s| REASONS.contains(&s))),
            invalid,
        )?;
        require(
            reasons.windows(2).all(|v| v[0].as_str() < v[1].as_str()),
            invalid,
        )?;
        match group["availability"].as_str() {
            Some("planned") => {
                let token = group["group"]
                    .as_str()
                    .filter(|v| is_digest(v))
                    .ok_or_else(|| error(invalid))?;
                require(
                    reasons.is_empty()
                        && group["partition"] == partition(&plan["seed"], token)?
                        && planned.insert(token.to_owned()),
                    invalid,
                )?;
            }
            Some("unavailable") => {
                require(
                    !reasons.is_empty()
                        && group["partition"].is_null()
                        && (group["group"].is_null()
                            || group["group"].as_str().is_some_and(is_digest)),
                    invalid,
                )?;
                unavailable += 1;
            }
            _ => return Err(error(invalid)),
        }
    }
    require(
        integer(
            &plan["planned_independent_sources"],
            MAX_JOBS as u64,
            invalid,
        )? == planned.len() as u64
            && integer(&plan["unavailable_sources"], MAX_JOBS as u64, invalid)? == unavailable
            && integer(&plan["planned_attempts"], MAX_JOBS as u64, invalid)? == jobs.len() as u64,
        invalid,
    )?;
    let mut keys = BTreeSet::new();
    let mut estimate = Number::from(0);
    for job in jobs {
        fields(
            job,
            &["attempt", "estimated_seconds", "group", "stage"],
            invalid,
        )?;
        let key = job_key(job, invalid)?;
        require(planned.contains(&key.0) && keys.insert(key), invalid)?;
        add_amount(
            &mut estimate,
            amount(&job["estimated_seconds"], invalid)?,
            invalid,
        )?;
    }
    require(
        canonical_json(&Value::Number(estimate.clone()))?
            == canonical_json(&plan["estimated_seconds"])?
            && estimate.as_f64().unwrap() <= plan["total_seconds_cap"].as_f64().unwrap(),
        invalid,
    )?;
    Ok(keys)
}

/// Retain missing/unavailable observations in the original attempt denominator.
/// The trusted expected digest is a body binding, never a signed admission.
pub fn reconcile_source_pilot_v2(
    plan_bytes: &[u8],
    attempts_bytes: &[u8],
    expected_plan_digest: &str,
) -> Result<Value> {
    require(is_digest(expected_plan_digest), "pilot_plan_digest_invalid")?;
    let plan = document(plan_bytes, "pilot_plan_invalid")?;
    let planned = checked_plan(&plan)?;
    let published = digest(&plan)?;
    let difference = published
        .bytes()
        .zip(expected_plan_digest.bytes())
        .fold(0u8, |a, (x, y)| a | (x ^ y));
    require(difference == 0, "pilot_plan_digest_mismatch")?;
    let attempts = document(attempts_bytes, "pilot_attempts_invalid")?;
    let rows = attempts
        .as_array()
        .ok_or_else(|| error("pilot_attempts_invalid"))?;
    require(rows.len() <= MAX_JOBS, "pilot_attempts_invalid")?;
    let mut counts = BTreeMap::from([
        ("success", 0usize),
        ("failed", 0),
        ("timeout", 0),
        ("unavailable", 0),
        ("not_attempted", 0),
    ]);
    let mut observed = BTreeMap::new();
    let mut seconds = Number::from(0);
    for row in rows {
        fields(
            row,
            &["attempt", "group", "seconds", "stage", "status"],
            "pilot_attempt_invalid",
        )?;
        let key = job_key(row, "pilot_attempt_invalid")?;
        let status = row["status"]
            .as_str()
            .filter(|v| ["success", "failed", "timeout", "unavailable"].contains(v))
            .ok_or_else(|| error("pilot_attempt_invalid"))?;
        require(
            planned.contains(&key) && observed.insert(key, row.clone()).is_none(),
            "pilot_attempt_invalid",
        )?;
        add_amount(
            &mut seconds,
            amount(&row["seconds"], "pilot_attempt_invalid")?,
            "pilot_attempt_cost_invalid",
        )?;
        *counts.get_mut(status).expect("validated status") += 1;
    }
    counts.insert("not_attempted", planned.len() - observed.len());
    let failures = observed
        .values()
        .filter(|v| v["status"] == "failed" || v["status"] == "timeout")
        .cloned()
        .collect::<Vec<_>>();
    Ok(json!({
        "budget_exceeded": seconds.as_f64().unwrap() > plan["total_seconds_cap"].as_f64().unwrap(),
        "confirmation_allowed": false, "execution_authorized": false,
        "execution_error_numerator": counts["failed"] + counts["timeout"],
        "failure_denominator": planned.len(), "failures": failures,
        "format": "dope-development-source-pilot-summary-v2",
        "non_success_numerator": planned.len() - counts["success"],
        "observed_attempts": observed.len(), "observed_seconds": seconds,
        "original_seconds_cap": plan["total_seconds_cap"], "plan_digest": published,
        "planned_attempts": planned.len(), "planned_independent_sources": plan["planned_independent_sources"],
        "scientific_admission": false, "scope": "development_only", "signed_admission": false,
        "status_counts": counts, "unavailable_sources": plan["unavailable_sources"], "version": 2
    }))
}

fn read_document(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| error("pilot_document_read_failed"))?;
    let metadata = file
        .metadata()
        .map_err(|_| error("pilot_document_read_failed"))?;
    require(
        metadata.is_file() && metadata.len() <= MAX_PILOT_DOCUMENT_BYTES as u64,
        "pilot_document_bytes_exceeded",
    )?;
    let mut bytes = Vec::new();
    file.take(MAX_PILOT_DOCUMENT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| error("pilot_document_read_failed"))?;
    require(
        bytes.len() <= MAX_PILOT_DOCUMENT_BYTES,
        "pilot_document_bytes_exceeded",
    )?;
    Ok(bytes)
}

/// File inspection only. This path never starts a pilot, vendor or network task.
pub fn inspect_source_pilot_v2(
    plan: &Path,
    attempts: Option<&Path>,
    expected_plan_digest: &str,
) -> Result<Value> {
    let plan = read_document(plan)?;
    let attempts = attempts
        .map(read_document)
        .transpose()?
        .unwrap_or_else(|| b"[]".to_vec());
    let summary = reconcile_source_pilot_v2(&plan, &attempts, expected_plan_digest)?;
    Ok(
        json!({"format":"dope-native-development-pilot-inspection-v2", "version":2,
        "canonical_wire":"dope-finite-json-v1", "binding":"caller_trusted_expected_plan_digest_not_signature",
        "custody_commitments":"keyed_opaque_unverified_without_custody_key",
        "observations":"caller_metadata_not_cost_receipts", "execution_authorized":false,
        "scientific_admission":false, "confirmation_allowed":false, "signed_admission":false,
        "summary":summary}),
    )
}

#[cfg(test)]
#[path = "source_pilot_tests.rs"]
mod tests;
