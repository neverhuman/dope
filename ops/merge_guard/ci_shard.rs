//! Rust-only receipt parser/aggregator for the CPU worker matrix.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, env, fs, path::Path};
#[path = "ci_shard_budget.rs"]
mod budget;
#[path = "ci_shard_coverage.rs"]
mod coverage;
#[path = "ci_shard_inventory.rs"]
mod inventory;
#[path = "ci_shard_shell.rs"]
mod shell;
#[path = "ci_shard_summary.rs"]
mod summaries;
#[path = "ci_shard_workflow.rs"]
mod workflow;
use summaries::summary;

const SHARDS: &[&str] = &["cpu-native", "cpu-compat", "cpu-transfer-paper"];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Command {
    id: String,
    kind: String,
    passed: u64,
    failed: u64,
    exit_code: u8,
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn fail<T>(code: &'static str) -> Result<T, &'static str> {
    Err(code)
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str, &'static str> {
    v.get(key).and_then(Value::as_str).ok_or("receipt_invalid")
}
fn number(v: &Value, key: &str) -> Result<u64, &'static str> {
    v.get(key).and_then(Value::as_u64).ok_or("receipt_invalid")
}
fn read_bounded(
    path: &Path,
    limit: u64,
    too_large: &'static str,
    read_error: &'static str,
) -> Result<Vec<u8>, &'static str> {
    if fs::metadata(path).map_err(|_| read_error)?.len() > limit {
        return fail(too_large);
    }
    let bytes = fs::read(path).map_err(|_| read_error)?;
    if bytes.len() as u64 > limit {
        return fail(too_large);
    }
    Ok(bytes)
}
fn parse_receipt(bytes: &[u8]) -> Result<Value, &'static str> {
    if bytes.len() > 1024 * 1024 {
        return fail("receipt_too_large");
    }
    let value: Value = serde_json::from_slice(bytes).map_err(|_| "receipt_invalid")?;
    if !value.is_object() {
        return fail("receipt_invalid");
    }
    Ok(value)
}
fn git_sha40(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn is_test_kind(kind: &str) -> bool {
    matches!(kind, "cargo-test" | "unittest" | "pytest" | "readiness")
}
/// Each block is emitted by run-ci-command.sh with an explicit end status.
/// Grammar: @@CI-BEGIN ID KIND@@ ... @@CI-END ID EXIT@@
fn parse_log(log: &str) -> Result<Vec<Command>, &'static str> {
    if log.len() > 32 * 1024 * 1024 {
        return fail("log_too_large");
    }
    let mut commands = Vec::new();
    let mut active: Option<(String, String, Vec<&str>)> = None;
    for line in log.lines() {
        if let Some(body) = line
            .strip_prefix("@@CI-BEGIN ")
            .and_then(|s| s.strip_suffix("@@"))
        {
            if active.is_some() {
                return fail("log_marker_invalid");
            }
            let mut parts = body.split_ascii_whitespace();
            let id = parts.next().ok_or("log_marker_invalid")?;
            let kind = parts.next().ok_or("log_marker_invalid")?;
            if parts.next().is_some() || id.is_empty() {
                return fail("log_marker_invalid");
            }
            active = Some((id.to_owned(), kind.to_owned(), Vec::new()));
        } else if let Some(body) = line
            .strip_prefix("@@CI-END ")
            .and_then(|s| s.strip_suffix("@@"))
        {
            let mut parts = body.split_ascii_whitespace();
            let id = parts.next().ok_or("log_marker_invalid")?;
            let exit = parts.next().ok_or("log_marker_invalid")?;
            if parts.next().is_some() {
                return fail("log_marker_invalid");
            }
            let (open, kind, lines) = active.take().ok_or("log_marker_invalid")?;
            let exit_code: u8 = exit.parse().map_err(|_| "log_marker_invalid")?;
            if open != id {
                return fail("log_marker_invalid");
            }
            let (passed, failed) = if is_test_kind(&kind) {
                match summary(&kind, &lines) {
                    Ok(counts) => counts,
                    Err(_) if exit_code != 0 => (0, 0),
                    Err(e) => return fail(e),
                }
            } else if kind == "check" {
                (0, 0)
            } else {
                return fail("command_kind_invalid");
            };
            commands.push(Command {
                id: open,
                kind,
                passed,
                failed,
                exit_code,
            });
        } else if let Some((_, _, lines)) = active.as_mut() {
            lines.push(line);
        }
    }
    if active.is_some() || commands.is_empty() {
        return fail("log_marker_invalid");
    }
    let mut ids = BTreeSet::new();
    if commands.iter().any(|c| !ids.insert(c.id.as_str())) {
        return fail("command_duplicate");
    }
    Ok(commands)
}
fn runtime_for(shard: &str) -> Vec<(&'static str, &'static str)> {
    inventory::RUNTIME_COMMANDS
        .iter()
        .filter(|(owner, _, _)| *owner == shard)
        .map(|(_, id, kind)| (*id, *kind))
        .collect()
}
fn make_receipt(
    shard: &str,
    head: &str,
    run: u64,
    attempt: u64,
    status: &str,
    log: &[u8],
) -> Result<Value, &'static str> {
    if !SHARDS.contains(&shard) || !git_sha40(head) || run == 0 || attempt == 0 {
        return fail("identity_invalid");
    }
    if !matches!(status, "success" | "failed") {
        return fail("receipt_invalid");
    }
    let parsed = parse_log(std::str::from_utf8(log).map_err(|_| "log_invalid_utf8")?)?;
    let expected = runtime_for(shard);
    if status == "success" {
        if parsed.len() != expected.len() {
            return fail("command_coverage_mismatch");
        }
        for (actual, (id, kind)) in parsed.iter().zip(expected.iter()) {
            if actual.id.as_str() != *id || actual.kind.as_str() != *kind {
                return fail("command_order_or_kind_mismatch");
            }
            if actual.exit_code != 0 {
                return fail("command_failed");
            }
            if actual.failed != 0 {
                return fail("test_command_failed");
            }
            if is_test_kind(kind) && actual.passed.saturating_add(actual.failed) == 0 {
                return fail("zero_test_command");
            }
        }
    } else {
        if parsed.is_empty() || parsed.len() > expected.len() {
            return fail("command_coverage_mismatch");
        }
        for (actual, (id, kind)) in parsed.iter().zip(expected.iter()) {
            if actual.id.as_str() != *id || actual.kind.as_str() != *kind {
                return fail("command_order_or_kind_mismatch");
            }
        }
        if parsed[..parsed.len() - 1]
            .iter()
            .any(|command| command.exit_code != 0)
            || parsed.last().is_none_or(|command| command.exit_code == 0)
        {
            return fail("failed_receipt_status_mismatch");
        }
        for command in &parsed[..parsed.len() - 1] {
            if is_test_kind(&command.kind) && command.passed.saturating_add(command.failed) == 0 {
                return fail("zero_test_command");
            }
        }
    }
    let tests: u64 = parsed
        .iter()
        .filter(|c| is_test_kind(&c.kind))
        .fold(0u64, |sum, c| {
            sum.saturating_add(c.passed.saturating_add(c.failed))
        });
    Ok(json!({
        "schema": 2, "shard": shard, "head_sha": head, "run_id": run,
        "run_attempt": attempt, "status": status, "tests_executed": tests,
        "log_sha256": sha(log),
        "commands": parsed.iter().map(|c| json!({
            "id": c.id, "kind": c.kind, "tests_passed": c.passed,
            "tests_failed": c.failed, "exit_code": c.exit_code
        })).collect::<Vec<_>>()
    }))
}
fn validate_receipt(
    r: &Value,
    shard: &str,
    head: &str,
    run: u64,
    attempt: u64,
    log: &[u8],
) -> Result<(), &'static str> {
    if number(r, "schema")? != 2
        || text(r, "shard")? != shard
        || text(r, "head_sha")? != head
        || number(r, "run_id")? != run
        || number(r, "run_attempt")? != attempt
        || text(r, "status")? != "success"
        || text(r, "log_sha256")? != sha(log)
    {
        return fail("receipt_identity_mismatch");
    }
    if &make_receipt(shard, head, run, attempt, "success", log)? != r {
        return fail("receipt_content_mismatch");
    }
    Ok(())
}
fn aggregate(
    needs: &Value,
    rows: &[(String, Value, Vec<u8>)],
    head: &str,
    run: u64,
    attempt: u64,
) -> Result<(), &'static str> {
    if !git_sha40(head) || run == 0 || attempt == 0 {
        return fail("aggregate_identity_invalid");
    }
    let needs = needs.as_object().ok_or("needs_invalid")?;
    if needs.len() != SHARDS.len()
        || SHARDS
            .iter()
            .any(|s| needs.get(*s).and_then(Value::as_str) != Some("success"))
    {
        return fail("worker_not_success");
    }
    if rows.len() != SHARDS.len() {
        return fail("receipt_set_invalid");
    }
    let mut seen = BTreeSet::new();
    for (shard, r, log) in rows {
        if !SHARDS.contains(&shard.as_str()) || !seen.insert(shard.as_str()) {
            return fail("receipt_set_invalid");
        }
        validate_receipt(r, shard, head, run, attempt, log)?;
    }
    if seen.len() != SHARDS.len() {
        return fail("receipt_set_invalid");
    }
    Ok(())
}
fn validate_filenames(names: &[String]) -> Result<(), &'static str> {
    let mut expected = BTreeSet::new();
    for shard in SHARDS {
        expected.insert(format!("{shard}.json"));
        expected.insert(format!("{shard}.log"));
    }
    let mut seen = BTreeSet::new();
    for name in names {
        if !expected.contains(name) || !seen.insert(name.as_str()) {
            return fail("receipt_set_invalid");
        }
    }
    if seen.len() != expected.len() {
        return fail("receipt_set_invalid");
    }
    Ok(())
}
fn main_result(args: &[String]) -> Result<(), String> {
    match args.get(1).map(String::as_str) {
        Some("budgets") if args.len() == 3 => {
            budget::verify(Path::new(&args[2])).map_err(str::to_owned)?;
            println!("Cost, time, admission, and kill-switch budget proof passed");
            Ok(())
        }
        Some("emit") if args.len() == 9 => {
            let log = read_bounded(Path::new(&args[7]), 32 * 1024 * 1024, "log_too_large", "log_read_failed")?;
            let run = args[4].parse().map_err(|_| "run_id_invalid")?;
            let attempt = args[5].parse().map_err(|_| "run_attempt_invalid")?;
            let r = make_receipt(&args[2], &args[3], run, attempt, &args[6], &log).map_err(str::to_owned)?;
            fs::write(&args[8], serde_json::to_vec(&r).map_err(|_| "receipt_encode_failed")?).map_err(|_| "receipt_write_failed")
        }
        Some("aggregate") if args.len() == 7 => {
            let needs_bytes = read_bounded(Path::new(&args[2]), 1024 * 1024, "needs_too_large", "needs_read_failed")?;
            let needs: Value = serde_json::from_slice(&needs_bytes).map_err(|_| "needs_invalid")?;
            let run = args[5].parse().map_err(|_| "run_id_invalid")?;
            let attempt = args[6].parse().map_err(|_| "run_attempt_invalid")?;
            let dir = Path::new(&args[3]);
            let names: Vec<String> = fs::read_dir(dir).map_err(|_| "receipt_dir_missing")?
                .map(|entry| entry.map_err(|_| "receipt_dir_invalid")
                    .and_then(|e| e.file_name().into_string().map_err(|_| "receipt_dir_invalid")))
                .collect::<Result<_, _>>()?;
            validate_filenames(&names).map_err(str::to_owned)?;
            let mut rows = Vec::new();
            for shard in SHARDS {
                let receipt_bytes = read_bounded(&dir.join(format!("{shard}.json")), 1024 * 1024, "receipt_too_large", "receipt_missing")?;
                let r = parse_receipt(&receipt_bytes).map_err(str::to_owned)?;
                let log = read_bounded(&dir.join(format!("{shard}.log")), 32 * 1024 * 1024, "log_too_large", "log_missing")?;
                rows.push(((*shard).to_owned(), r, log));
            }
            aggregate(&needs, &rows, &args[4], run, attempt).map_err(str::to_owned)
        }
        Some("coverage") if args.len() == 6 => {
            let manifest_bytes = read_bounded(Path::new(&args[2]), 4 * 1024 * 1024, "coverage_manifest_too_large", "coverage_manifest_missing")?;
            let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(|_| "coverage_manifest_invalid")?;
            let workflow_bytes = read_bounded(Path::new(&args[5]), 4 * 1024 * 1024, "workflow_too_large", "workflow_missing")?;
            let workflow = String::from_utf8(workflow_bytes).map_err(|_| "workflow_invalid_utf8")?;
            coverage::verify(&manifest, Path::new(&args[3]), Path::new(&args[4]), &workflow).map_err(str::to_owned)
        }
        _ => Err("usage: emit <shard> <head> <run> <attempt> <status> <log> <receipt> | aggregate <needs.json> <dir> <head> <run> <attempt> | coverage <manifest.json> <preimages> <stages> <workflow> | budgets <root>".into())
    }
}
fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(code) = main_result(&args) {
        eprintln!("ci-shard refused: {code}");
        std::process::exit(1);
    }
}
#[cfg(test)]
#[path = "ci_shard_tests.rs"]
mod tests;
