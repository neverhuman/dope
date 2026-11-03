//! Exact command-marker coverage verifier, kept separate from receipt logic.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

fn sha(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn refused<T>(code: &'static str) -> Result<T, &'static str> {
    Err(code)
}
fn pinned_stage_matches(name: &str, bytes: &[u8]) -> bool {
    super::inventory::PINNED_STAGE_SHA256
        .iter()
        .find(|(path, _)| *path == name)
        .is_some_and(|(_, expected)| {
            let actual: String = Sha256::digest(bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            actual.as_str() == *expected
        })
}
fn normalize(block: &[&str]) -> String {
    let kept: Vec<&str> = block
        .iter()
        .copied()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let indent = kept
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    kept.iter()
        .map(|l| {
            if l.trim().is_empty() {
                ""
            } else {
                &l[indent.min(l.len())..]
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
fn marked(
    source: &str,
) -> Result<BTreeMap<String, (String, String, String, String)>, &'static str> {
    let lines: Vec<_> = source.lines().collect();
    let mut rows = BTreeMap::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some(rest) = line.trim().strip_prefix("# ci-command:") {
            let mut parts = rest.split_ascii_whitespace();
            let id = parts.next().ok_or("command_marker_invalid")?;
            let hash = parts.next().ok_or("command_marker_invalid")?;
            if parts.next().is_some()
                || hash.len() != 64
                || !hash.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return refused("command_marker_invalid");
            }
            let command = lines.get(i + 1).ok_or("command_marker_invalid")?;
            if command.trim().is_empty() || command.trim_start().starts_with('#') {
                return refused("command_marker_invalid");
            }
            let actual = sha(&normalize(&[command]));
            let expected = hash.to_owned();
            if expected != actual {
                return refused("command_content_mismatch");
            }
            let words = super::shell::shell_words(command.trim())?;
            let kind = words.get(2).ok_or("command_marker_invalid")?.clone();
            if words.first().map(String::as_str) != Some("ci_run") {
                return refused("command_marker_invalid");
            }
            if rows
                .insert(
                    id.to_owned(),
                    (kind, expected, actual, command.trim().to_owned()),
                )
                .is_some()
            {
                return refused("command_duplicate");
            }
            i += 1;
        }
        i += 1;
    }
    Ok(rows)
}
fn compare(
    expected: &BTreeMap<String, (String, String)>,
    actual: &BTreeMap<String, (String, String)>,
) -> Result<(), &'static str> {
    if expected.len() != actual.len() {
        return refused("command_coverage_mismatch");
    }
    for (id, (shard, hash)) in expected {
        let (actual_shard, actual_hash) = actual.get(id).ok_or("command_coverage_mismatch")?;
        if actual_shard != shard || actual_hash != hash {
            return refused("command_content_mismatch");
        }
    }
    Ok(())
}
// An inline run key is YAML syntax in this pinned source, not shell argv.
fn source_invocation(path: &str, block: &str) -> String {
    if path == ".github/workflows/ci.yml" {
        block.strip_prefix("run: ").unwrap_or(block).to_owned()
    } else {
        block.to_owned()
    }
}

fn same_invocation(
    origin: &str,
    marked_line: &str,
    allow_ci_shard_bin: bool,
) -> Result<bool, &'static str> {
    let mut actual = super::shell::shell_words(marked_line)?;
    if actual.first().map(String::as_str) != Some("ci_run") || actual.len() < 4 {
        return Ok(false);
    }
    actual.drain(0..3);
    if actual.first().map(String::as_str) == Some("env") {
        actual.remove(0);
    }
    // Only the two merge-guard checks may add exactly one helper target.
    let helper_targets = actual
        .windows(2)
        .filter(|pair| pair[0] == "--bin" && pair[1] == "ci-shard")
        .count();
    if helper_targets != if allow_ci_shard_bin { 1 } else { 0 } {
        return Ok(false);
    }
    let mut i = 0;
    while i + 1 < actual.len() {
        if actual[i] == "--bin" && actual[i + 1] == "ci-shard" {
            actual.drain(i..i + 2);
            break;
        }
        i += 1;
    }
    if actual.first().map(String::as_str) == Some("bash")
        && actual.get(1).map(String::as_str) == Some("-c")
    {
        return Ok(actual.get(2).is_some_and(|s| s.trim() == origin.trim()));
    }
    Ok(actual == super::shell::shell_words(origin)?)
}

fn inventory_matches(commands: &[Value]) -> Result<(), &'static str> {
    if commands.len() != super::inventory::EXPECTED_COMMANDS.len() {
        return refused("coverage_inventory_mismatch");
    }
    for (row, (expected_id, expected_path, expected_start, expected_end)) in commands
        .iter()
        .zip(super::inventory::EXPECTED_COMMANDS.iter())
    {
        let id = row
            .get("id")
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        let shard = row
            .get("shard")
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        let path = row
            .get("origin_path")
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        let start = row
            .get("origin_start")
            .and_then(Value::as_u64)
            .ok_or("coverage_manifest_invalid")? as usize;
        let end = row
            .get("origin_end")
            .and_then(Value::as_u64)
            .ok_or("coverage_manifest_invalid")? as usize;
        let expected_shard = expected_id
            .split_once(':')
            .map(|(name, _)| name)
            .ok_or("coverage_inventory_invalid")?;
        if id != *expected_id
            || shard != expected_shard
            || path != *expected_path
            || start != *expected_start
            || end != *expected_end
        {
            return refused("coverage_inventory_mismatch");
        }
    }
    Ok(())
}

/// Manifest command rows carry the hash of the normalized original command block.
/// Preimage source hashes pin the two source files from which the inventory was reviewed.
pub fn verify(
    manifest: &Value,
    preimages: &Path,
    stages: &Path,
    workflow: &str,
) -> Result<(), &'static str> {
    const PINNED: &[(&str, &str)] = &[
        (
            ".github/workflows/ci.yml",
            "efff84032e0170266dea7f3ab854238cc6c9c033e7b7ef6fd85fe91d9dc62263",
        ),
        (
            "ops/ci/cpu.sh",
            "5c6462184c66b7d7c74825e36321627b52d1acf8dab11818acd32b98c3fb55c5",
        ),
    ];
    let source_hashes = manifest
        .get("source_sha256")
        .and_then(Value::as_object)
        .ok_or("coverage_manifest_invalid")?;
    if source_hashes.len() != PINNED.len() {
        return refused("coverage_manifest_invalid");
    }
    let mut source_text = BTreeMap::new();
    for (name, pinned) in PINNED {
        let want = source_hashes
            .get(*name)
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        if want != *pinned {
            return refused("coverage_manifest_invalid");
        }
        let bytes = super::read_bounded(
            &preimages.join(*name),
            1024 * 1024,
            "coverage_preimage_too_large",
            "coverage_preimage_missing",
        )?;
        let got: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if got != want {
            return refused("coverage_preimage_mismatch");
        }
        source_text.insert(
            (*name).to_owned(),
            String::from_utf8(bytes).map_err(|_| "coverage_preimage_invalid")?,
        );
    }
    let commands = manifest
        .get("commands")
        .and_then(Value::as_array)
        .ok_or("coverage_manifest_invalid")?;
    // The manifest is data, not authority: pin the complete command inventory in Rust.
    // This prevents deleting/reassigning an entry and then regenerating a self-consistent map.
    inventory_matches(commands)?;
    let mut expected = BTreeMap::new();
    let mut origins = BTreeMap::new();
    for row in commands {
        let id = row
            .get("id")
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        let shard = row
            .get("shard")
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        let hash = row
            .get("body_sha256")
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        let origin = row
            .get("origin_path")
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        let start = row
            .get("origin_start")
            .and_then(Value::as_u64)
            .ok_or("coverage_manifest_invalid")? as usize;
        let end = row
            .get("origin_end")
            .and_then(Value::as_u64)
            .ok_or("coverage_manifest_invalid")? as usize;
        let origin_hash = row
            .get("origin_sha256")
            .and_then(Value::as_str)
            .ok_or("coverage_manifest_invalid")?;
        let original = source_text.get(origin).ok_or("coverage_manifest_invalid")?;
        let lines: Vec<_> = original.lines().collect();
        if start == 0
            || end < start
            || end > lines.len()
            || sha(&normalize(&lines[start - 1..end])) != origin_hash
        {
            return refused("coverage_origin_mismatch");
        }
        origins.insert(
            id.to_owned(),
            source_invocation(origin, &normalize(&lines[start - 1..end])),
        );
        if hash.len() != 64
            || expected
                .insert(id.to_owned(), (shard.to_owned(), hash.to_owned()))
                .is_some()
        {
            return refused("coverage_manifest_invalid");
        }
    }
    if expected.len() != super::inventory::EXPECTED_COMMANDS.len() {
        return refused("coverage_manifest_invalid");
    }
    let paths = manifest
        .get("stage_paths")
        .and_then(Value::as_object)
        .ok_or("coverage_manifest_invalid")?;
    if paths.len() != SHARDS_FOR_COVERAGE.len() {
        return refused("coverage_manifest_invalid");
    }
    // Pin the stage control flow, not just marker comments. This protects guard/capture/warm order.
    for (name, _) in super::inventory::PINNED_STAGE_SHA256 {
        let bytes = super::read_bounded(
            &stages.join(*name),
            2 * 1024 * 1024,
            "coverage_stage_too_large",
            "coverage_stage_missing",
        )?;
        if !pinned_stage_matches(name, &bytes) {
            return refused("coverage_stage_changed");
        }
    }
    let current_cpu = super::read_bounded(
        &stages.join("cpu.sh"),
        1024 * 1024,
        "cpu_entry_too_large",
        "cpu_entry_missing",
    )?;
    let current_cpu_sha: String = Sha256::digest(&current_cpu)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if current_cpu_sha != PINNED[1].1 {
        return refused("cpu_entry_changed");
    }
    let mut actual_all = BTreeMap::new();
    let mut actual_lines = BTreeMap::new();
    for (shard, file) in paths {
        let file = file.as_str().ok_or("coverage_manifest_invalid")?;
        if !SHARDS_FOR_COVERAGE.contains(&shard.as_str()) || file != format!("{shard}.sh") {
            return refused("coverage_manifest_invalid");
        }
        let bytes = super::read_bounded(
            &stages.join(file),
            2 * 1024 * 1024,
            "coverage_stage_too_large",
            "coverage_stage_missing",
        )?;
        let source = String::from_utf8(bytes).map_err(|_| "coverage_stage_invalid_utf8")?;
        let marker_order: Vec<String> = source
            .lines()
            .filter_map(|line| {
                line.trim()
                    .strip_prefix("# ci-command:")
                    .and_then(|rest| rest.split_ascii_whitespace().next())
                    .map(str::to_owned)
            })
            .collect();
        let wanted_order: Vec<String> = super::inventory::RUNTIME_COMMANDS
            .iter()
            .filter(|(owner, id, _)| {
                *owner == shard.as_str()
                    && !matches!(
                        *id,
                        "frozen-evidence" | "ci-shard-build" | "ci-shard-coverage"
                    )
            })
            .map(|(_, id, _)| format!("{shard}:{id}"))
            .collect();
        if marker_order != wanted_order {
            return refused("command_order_mismatch");
        }
        let marked_rows = marked(&source)?;
        for (owner, id, kind) in
            super::inventory::RUNTIME_COMMANDS
                .iter()
                .filter(|(owner, id, _)| {
                    *owner == shard.as_str()
                        && !matches!(
                            *id,
                            "frozen-evidence" | "ci-shard-build" | "ci-shard-coverage"
                        )
                })
        {
            let full_id = format!("{owner}:{id}");
            let (actual_kind, _, body_hash, command) = marked_rows
                .get(&full_id)
                .ok_or("command_coverage_mismatch")?;
            if actual_kind.as_str() != *kind {
                return refused("command_kind_mismatch");
            }
            actual_lines.insert(full_id.clone(), command.clone());
            actual_all.insert(full_id, (shard.clone(), body_hash.clone()));
        }
    }
    let actual_source: BTreeMap<_, _> = actual_all
        .iter()
        .filter(|(id, _)| expected.contains_key(*id))
        .map(|(id, value)| (id.clone(), value.clone()))
        .collect();
    compare(&expected, &actual_source)?;
    for id in expected.keys() {
        let helper_target_allowed =
            id == "cpu-native:merge-guard-clippy" || id == "cpu-native:merge-guard-test";
        if !same_invocation(
            origins.get(id).ok_or("coverage_origin_mismatch")?,
            actual_lines.get(id).ok_or("command_coverage_mismatch")?,
            helper_target_allowed,
        )? {
            return refused("command_semantics_mismatch");
        }
    }
    for shard in SHARDS_FOR_COVERAGE {
        let job = format!("  {shard}:");
        let command = format!("bash ops/ci/{shard}.sh");
        if workflow.match_indices(&job).count() != 1
            || workflow.match_indices(&command).count() != 1
        {
            return refused("workflow_coverage_mismatch");
        }
    }
    super::workflow::verify_portable(workflow)?;
    let tests = manifest
        .get("test_ids")
        .and_then(Value::as_object)
        .ok_or("coverage_manifest_invalid")?;
    if tests.len() != SHARDS_FOR_COVERAGE.len() {
        return refused("coverage_manifest_invalid");
    }
    for shard in SHARDS_FOR_COVERAGE {
        let actual_ids = tests
            .get(*shard)
            .and_then(Value::as_array)
            .ok_or("coverage_manifest_invalid")?;
        let actual_ids: Vec<_> = actual_ids
            .iter()
            .map(|v| v.as_str().ok_or("coverage_manifest_invalid"))
            .collect::<Result<_, _>>()?;
        let expected_ids: Vec<_> = super::inventory::RUNTIME_COMMANDS
            .iter()
            .filter(|(owner, _, kind)| *owner == *shard && super::is_test_kind(kind))
            .map(|(_, id, _)| *id)
            .collect();
        if actual_ids != expected_ids {
            return refused("test_command_coverage_mismatch");
        }
    }
    Ok(())
}

const SHARDS_FOR_COVERAGE: &[&str] = &["cpu-native", "cpu-compat", "cpu-transfer-paper"];

#[cfg(test)]
#[path = "ci_shard_coverage_tests.rs"]
mod tests;
