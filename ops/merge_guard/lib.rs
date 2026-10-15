//! GET-only operations inspection. Every result is advisory.

pub mod collect;
pub mod controls;
pub mod evidence;
pub mod github;
pub mod inspection;
pub mod json;
pub mod review;
pub mod runs;
pub mod timestamp;

use serde_json::Value;
use std::collections::BTreeSet;

pub const CHECKS: [&str; 4] = ["cpu", "jankurai-audit", "paper-clean", "security"];
pub const APP_ID: u64 = 15368;
pub const REVIEWER: &str = "neverhumanbot";
pub const OPERATOR: &str = "jepsontaylor";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refused(pub &'static str);
pub type Result<T> = std::result::Result<T, Refused>;

pub fn require(condition: bool, reason: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Refused(reason))
    }
}
pub fn text(value: &Value) -> Result<&str> {
    value
        .as_str()
        .ok_or(Refused("invalid_or_incomplete_evidence"))
}
pub fn array(value: &Value) -> Result<&Vec<Value>> {
    value
        .as_array()
        .ok_or(Refused("invalid_or_incomplete_evidence"))
}
pub fn integer(value: &Value) -> Result<u64> {
    value
        .as_u64()
        .ok_or(Refused("invalid_or_incomplete_evidence"))
}
pub fn strings(value: &Value) -> Result<BTreeSet<String>> {
    array(value)?
        .iter()
        .map(|v| Ok(text(v)?.to_owned()))
        .collect()
}
pub fn check_set() -> BTreeSet<String> {
    CHECKS.iter().map(|s| (*s).to_owned()).collect()
}
pub fn valid_sha(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 40
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
pub fn validate_pins(pins: &Value) -> Result<()> {
    let repo = text(&pins["repo"])?;
    let parts: Vec<_> = repo.split('/').collect();
    require(
        parts.len() == 2
            && parts.iter().all(|p| {
                !p.is_empty()
                    && p.bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
            })
            && integer(&pins["pr"])? > 0
            && pins["base_branch"] == "main"
            && valid_sha(&pins["source_sha"])
            && valid_sha(&pins["base_sha"]),
        "invalid_pins",
    )
}

#[cfg(test)]
mod tests;
