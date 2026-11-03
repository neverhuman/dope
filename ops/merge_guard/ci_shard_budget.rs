//! Exact graph quotas and preserved source/admission kill switches; no native execution.
use std::{collections::BTreeMap, fs, path::Path};
#[path = "ci_shard_budget_toml.rs"]
mod profile;
use profile::Scalar;
const CI: &[(&str, &str, u64)] = &[
    ("cpu-native", "cpu_native_minutes", 90),
    ("cpu-compat", "cpu_compat_minutes", 120),
    ("cpu-transfer-paper", "cpu_transfer_paper_minutes", 120),
    ("ci-shard-portable-arm64", "cpu_arm_portable_minutes", 30),
    ("cpu", "cpu_aggregate_minutes", 20),
    ("paper-clean", "paper_clean_minutes", 60),
    ("security", "security_minutes", 60),
];
const NUMBERS: &[(&str, u64)] = &[
    ("version", 1),
    ("cpu_pr_minutes", 120),
    ("cpu_native_minutes", 90),
    ("cpu_compat_minutes", 120),
    ("cpu_transfer_paper_minutes", 120),
    ("cpu_arm_portable_minutes", 30),
    ("cpu_aggregate_minutes", 20),
    ("paper_clean_minutes", 60),
    ("security_minutes", 60),
    ("audit_minutes", 60),
    ("scheduled_gpu_minutes", 120),
    ("single_neural_fit_seconds", 600),
    ("gpu_memory_bytes", 17179869184),
    ("fitness_sample_rows", 128),
    ("fitness_sample_dimensions", 32),
];
fn require(ok: bool, reason: &'static str) -> Result<(), &'static str> {
    if ok { Ok(()) } else { Err(reason) }
}
fn budget_profile(source: &str) -> Result<BTreeMap<String, Scalar>, &'static str> {
    let fields = profile::parse(source)?;
    require(fields.len() == NUMBERS.len() + 5, "budget_profile_keys")?;
    for (key, expected) in NUMBERS {
        require(
            fields.get(*key) == Some(&Scalar::Integer(*expected)),
            "budget_profile_value",
        )?;
    }
    for (key, expected) in [
        ("retry_rule", "repair-required"),
        (
            "kill_switch",
            "cancel GitHub job or interrupt ops/ci/gpu.sh",
        ),
    ] {
        require(
            fields.get(key) == Some(&Scalar::Text(expected.to_owned())),
            "budget_profile_value",
        )?;
    }
    for key in [
        "require_sealed_corpus",
        "require_candidate_artifact",
        "require_pinned_cuda",
    ] {
        require(
            fields.get(&format!("admission.{key}")) == Some(&Scalar::Boolean(true)),
            "budget_admission_required",
        )?;
    }
    Ok(fields)
}
fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().enumerate().all(|(i, b)| {
            b == b'_' || b.is_ascii_alphabetic() || (i > 0 && (b.is_ascii_digit() || b == b'-'))
        })
}
fn jobs(source: &str) -> Result<BTreeMap<String, u64>, &'static str> {
    let mut active: Option<String> = None;
    let mut timeouts: BTreeMap<String, u64> = BTreeMap::new();
    let mut job_headers = std::collections::BTreeSet::new();
    let mut in_jobs = false;
    let mut headers = 0;
    for line in source.lines() {
        if line == "jobs:" {
            headers += 1;
            in_jobs = true;
            continue;
        }
        if line
            .trim_start()
            .strip_prefix("- ")
            .unwrap_or(line.trim_start())
            .starts_with("timeout-minutes")
        {
            require(
                in_jobs && line.starts_with("    timeout-minutes:"),
                "budget_timeout_placement",
            )?;
            let job = active.as_ref().ok_or("budget_timeout_placement")?;
            let raw = line
                .strip_prefix("    timeout-minutes:")
                .ok_or("budget_timeout_placement")?
                .trim();
            require(
                !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()),
                "budget_timeout_invalid",
            )?;
            let value = raw.parse::<u64>().map_err(|_| "budget_timeout_invalid")?;
            require(
                value > 0 && timeouts.insert(job.clone(), value).is_none(),
                "budget_timeout_invalid",
            )?;
        }
        if !in_jobs {
            continue;
        }
        if !line.is_empty() && !line.starts_with(' ') && !line.starts_with('#') {
            return Err("budget_workflow_unsupported");
        }
        if line.starts_with("  ") && !line.starts_with("   ") && !line.trim_start().starts_with('#')
        {
            let name = line
                .strip_prefix("  ")
                .and_then(|l| l.strip_suffix(':'))
                .ok_or("budget_job_header")?;
            require(identifier(name), "budget_job_header")?;
            require(job_headers.insert(name.to_owned()), "budget_job_duplicate")?;
            active = Some(name.to_owned());
        }
    }
    require(
        headers == 1 && !job_headers.is_empty(),
        "budget_workflow_invalid",
    )?;
    require(
        timeouts.len() == job_headers.len(),
        "budget_timeout_missing",
    )?;
    Ok(timeouts)
}
fn quotas(source: &str, expected: &[(&str, u64)]) -> Result<(), &'static str> {
    let actual = jobs(source)?;
    let wanted: BTreeMap<String, u64> = expected
        .iter()
        .map(|(job, minutes)| ((*job).to_owned(), *minutes))
        .collect();
    require(actual == wanted, "budget_workflow_quota")
}
fn manual_gpu(source: &str) -> Result<(), &'static str> {
    let mut in_events = false;
    let mut found = 0;
    let mut headers = 0;
    for line in source.lines() {
        if line.starts_with("on:") {
            require(line == "on:", "budget_gpu_manual_only")?;
            headers += 1;
            in_events = true;
            continue;
        }
        if !in_events || line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if !line.starts_with(' ') {
            in_events = false;
            continue;
        }
        if line == "  workflow_dispatch:" {
            found += 1;
        } else {
            return Err("budget_gpu_manual_only");
        }
    }
    require(headers == 1 && found == 1, "budget_gpu_manual_only")
}
pub(super) struct Sources<'a> {
    ci: &'a str,
    audit: &'a str,
    gpu_workflow: &'a str,
    neural: &'a str,
    train: &'a str,
    fitness: &'a str,
    gpu_script: &'a str,
}
fn verify_sources(profile: &str, source: &Sources<'_>) -> Result<(), &'static str> {
    let fields = budget_profile(profile)?;
    let expected: Vec<_> = CI
        .iter()
        .map(|(job, key, _)| match fields.get(*key) {
            Some(Scalar::Integer(minutes)) => Ok((*job, *minutes)),
            _ => Err("budget_profile_value"),
        })
        .collect::<Result<_, _>>()?;
    quotas(source.ci, &expected)?;
    quotas(source.audit, &[("jankurai-audit", 60)])?;
    quotas(source.gpu_workflow, &[("gpu", 120)])?;
    manual_gpu(source.gpu_workflow)?;
    require(
        source.neural.contains("Duration::from_secs(600)"),
        "budget_neural_seconds",
    )?;
    require(
        source
            .train
            .contains("GPU_MEMORY_LIMIT_BYTES: u128 = 16 * 1024 * 1024 * 1024"),
        "budget_gpu_memory",
    )?;
    require(
        source.gpu_script.contains("\"${SEALED_CORPUS_DIR:-}\"")
            && source.gpu_script.contains("\"${SEALED_ARTIFACT:-}\""),
        "budget_sealed_guards",
    )?;
    require(
        source.fitness.contains("const MAX_ROWS: usize = 128")
            && source.fitness.contains("const MAX_DIMENSIONS: usize = 32"),
        "budget_fitness_bounds",
    )
}
fn read(path: &Path) -> Result<String, &'static str> {
    let metadata = fs::symlink_metadata(path).map_err(|_| "budget_source_missing")?;
    require(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "budget_source_path",
    )?;
    let bytes = super::read_bounded(
        path,
        2 * 1024 * 1024,
        "budget_source_too_large",
        "budget_source_unreadable",
    )?;
    String::from_utf8(bytes).map_err(|_| "budget_source_utf8")
}
fn walk(
    path: &Path,
    depth: usize,
    remaining: &mut usize,
    text: &mut String,
) -> Result<(), &'static str> {
    require(depth <= 32, "budget_source_depth")?;
    let metadata = fs::symlink_metadata(path).map_err(|_| "budget_source_missing")?;
    require(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "budget_source_path",
    )?;
    let mut entries = fs::read_dir(path)
        .map_err(|_| "budget_source_unreadable")?
        .take(4097)
        .map(|e| e.map(|e| e.path()).map_err(|_| "budget_source_unreadable"))
        .collect::<Result<Vec<_>, _>>()?;
    require(entries.len() <= 4096, "budget_source_entries")?;
    entries.sort();
    for entry in entries {
        require(*remaining > 0, "budget_source_entries")?;
        *remaining -= 1;
        let metadata = fs::symlink_metadata(&entry).map_err(|_| "budget_source_unreadable")?;
        require(!metadata.file_type().is_symlink(), "budget_source_path")?;
        if metadata.is_dir() {
            walk(&entry, depth + 1, remaining, text)?;
        } else {
            require(metadata.is_file(), "budget_source_path")?;
            if entry.extension().and_then(|s| s.to_str()) == Some("rs") {
                text.push('\n');
                text.push_str(&read(&entry)?);
                require(text.len() <= 8 * 1024 * 1024, "budget_source_too_large")?;
            }
        }
    }
    Ok(())
}
fn tree(root: &Path, name: &str) -> Result<String, &'static str> {
    let path = root.join(name);
    let mut text = read(&path)?;
    let mut remaining = 4096;
    walk(&path.with_extension(""), 0, &mut remaining, &mut text)?;
    Ok(text)
}
pub(super) fn verify(root: &Path) -> Result<(), &'static str> {
    let policy = read(&root.join("agent/cost-budget.toml"))?;
    let ci = read(&root.join(".github/workflows/ci.yml"))?;
    let audit = read(&root.join(".github/workflows/jankurai.yml"))?;
    let gpu = read(&root.join(".github/workflows/gpu.yml"))?;
    let neural = tree(root, "rust/compiler/neural_candidates.rs")?;
    let train = tree(root, "rust/neural_train.rs")?;
    let fitness = read(&root.join("rust/fitness_metrics.rs"))?;
    let gpu_script = read(&root.join("ops/ci/gpu.sh"))?;
    verify_sources(
        &policy,
        &Sources {
            ci: &ci,
            audit: &audit,
            gpu_workflow: &gpu,
            neural: &neural,
            train: &train,
            fitness: &fitness,
            gpu_script: &gpu_script,
        },
    )
}
#[cfg(test)]
#[path = "ci_shard_budget_tests.rs"]
mod tests;
