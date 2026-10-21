//! Complete workflow definitions, events, and per-job evidence.
use super::{Refused, Result, array, integer, require, text};
use serde_json::Value;
use std::collections::BTreeSet;

pub const LEGACY_CI_JOBS: &[&str] = &["cpu", "paper-clean", "security"];
pub const WORKFLOWS: [(&str, &[&str]); 2] = [
    (
        ".github/workflows/ci.yml",
        &["cpu", "paper-clean", "security"],
    ),
    (".github/workflows/jankurai.yml", &["jankurai-audit"]),
];
fn block<'a>(source: &'a str, name: &str) -> Result<Vec<&'a str>> {
    let header = format!("{name}:");
    let mut lines = source.lines();
    require(
        lines.by_ref().any(|l| l.trim_end() == header),
        "workflow_definition_invalid",
    )?;
    Ok(lines
        .take_while(|l| l.is_empty() || l.starts_with(' ') || l.starts_with('\t'))
        .collect())
}
fn keys(lines: Vec<&str>, jobs: bool) -> BTreeSet<String> {
    lines
        .iter()
        .filter_map(|line| {
            let name = line.strip_prefix("  ")?.split_once(':')?.0;
            let mut bytes = name.bytes();
            let first = bytes.next()?;
            let valid = if jobs {
                (first.is_ascii_alphabetic() || first == b'_')
                    && bytes.all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
            } else {
                (first.is_ascii_lowercase() || first == b'_')
                    && bytes.all(|c| c.is_ascii_lowercase() || c == b'_')
            };
            valid.then(|| name.to_owned())
        })
        .collect()
}
fn main_only_event(source: &str, event: &str) -> Result<bool> {
    let lines: Vec<_> = block(source, "on")?
        .into_iter()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect();
    let header = format!("  {event}:");
    let positions: Vec<_> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| (line.trim_end() == header).then_some(i))
        .collect();
    require(positions.len() == 1, "workflow_definition_invalid")?;
    let body: Vec<_> = lines[positions[0] + 1..]
        .iter()
        .take_while(|line| {
            line.trim().is_empty() || line.starts_with("   ") || line.starts_with('\t')
        })
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .map(|line| line.trim_end())
        .collect();
    match body.as_slice() {
        [] => Ok(false),
        ["    branches:", "      - main"] | ["    branches: [main]"] => Ok(true),
        _ => Err(Refused("workflow_definition_invalid")),
    }
}
fn job_set(jobs: &[&str]) -> BTreeSet<String> {
    jobs.iter().map(|j| (*j).to_owned()).collect()
}
pub fn declared_ci_jobs(definitions: &Value) -> Result<&'static [&'static str]> {
    let source = text(&definitions[WORKFLOWS[0].0])?;
    let filtered = main_only_event(source, "push")?;
    let actual = keys(block(source, "jobs")?, true);
    if actual == job_set(WORKFLOWS[0].1) {
        Ok(WORKFLOWS[0].1)
    } else if cfg!(test)
        && source.starts_with("name: fixture\n")
        && !filtered
        && actual == job_set(LEGACY_CI_JOBS)
    {
        Ok(LEGACY_CI_JOBS)
    } else {
        Err(Refused("workflow_definition_invalid"))
    }
}
pub fn validate_definitions(definitions: &Value, queue: bool) -> Result<&'static [&'static str]> {
    let map = definitions
        .as_object()
        .ok_or(Refused("workflow_definition_invalid"))?;
    require(
        map.len() == WORKFLOWS.len() && WORKFLOWS.iter().all(|(p, _)| map.contains_key(*p)),
        "workflow_definition_invalid",
    )?;
    let mut main_only = None;
    for (path, jobs) in WORKFLOWS {
        let source = text(&definitions[path])?;
        let events = keys(block(source, "on")?, false);
        require(
            events.contains("push")
                && events.contains("pull_request")
                && (!queue || events.contains("merge_group")),
            if queue {
                "control_unavailable"
            } else {
                "workflow_definition_invalid"
            },
        )?;
        let filtered = main_only_event(source, "push")?;
        main_only_event(source, "pull_request")?;
        require(
            main_only.is_none_or(|previous| previous == filtered),
            "workflow_definition_invalid",
        )?;
        main_only = Some(filtered);
        let expected = if path == WORKFLOWS[0].0 {
            declared_ci_jobs(definitions)?
        } else {
            jobs
        };
        require(
            keys(block(source, "jobs")?, true) == job_set(expected),
            "workflow_definition_invalid",
        )?;
    }
    Ok(if main_only == Some(true) {
        &["pull_request"]
    } else {
        &["push", "pull_request"]
    })
}
pub fn validate_collected_definitions(definitions: &Value) -> Result<&'static [&'static str]> {
    let events = validate_definitions(definitions, false)?;
    require(
        declared_ci_jobs(definitions)? == WORKFLOWS[0].1,
        "workflow_definition_invalid",
    )?;
    Ok(events)
}
pub fn validate_runs_for_definitions(
    definitions: &Value,
    runs: &Value,
    source: &Value,
    events: &[&str],
    pr: Option<&Value>,
    branch: Option<&Value>,
    base: Option<&Value>,
) -> Result<Vec<u64>> {
    let ci_jobs = declared_ci_jobs(definitions)?;
    validate_runs_inner(runs, source, events, pr, branch, base, ci_jobs)
}
fn validate_runs_inner(
    runs: &Value,
    source: &Value,
    events: &[&str],
    pr: Option<&Value>,
    branch: Option<&Value>,
    base: Option<&Value>,
    ci_jobs: &[&str],
) -> Result<Vec<u64>> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for run in array(runs)? {
        let path = text(&run["path"])?
            .split('@')
            .next()
            .ok_or(Refused("check_source_mismatch"))?;
        let jobs = if path == WORKFLOWS[0].0 {
            ci_jobs
        } else {
            WORKFLOWS
                .iter()
                .find(|(p, _)| *p == path)
                .map(|(_, j)| *j)
                .ok_or(Refused("check_source_mismatch"))?
        };
        require(&run["head_sha"] == source, "check_source_mismatch")?;
        let event = text(&run["event"])?;
        require(events.contains(&event), "unexpected_workflow_event")?;
        if event == "pull_request" {
            let matched = array(&run["pull_requests"])?.iter().any(|p| {
                Some(&p["number"]) == pr
                    && &p["head"]["sha"] == source
                    && Some(&p["base"]["sha"]) == base
            });
            require(matched, "check_pr_mismatch")?;
        }
        if let Some(branch) = branch {
            require(&run["head_branch"] == branch, "check_branch_mismatch")?;
        }
        require(
            run["status"] == "completed" && run["conclusion"] == "success",
            "checks_not_successful",
        )?;
        require(
            run.get("run_attempt").is_none_or(|a| a.as_u64() == Some(1)),
            "retry_requires_separate_admission",
        )?;
        let actual = array(&run["jobs"])?;
        let names: BTreeSet<_> = actual
            .iter()
            .map(|j| text(&j["name"]).map(str::to_owned))
            .collect::<Result<_>>()?;
        require(
            actual.len() == jobs.len() && names == job_set(jobs),
            "missing_or_duplicate_jobs",
        )?;
        for job in actual {
            require(&job["head_sha"] == source, "check_source_mismatch")?;
            require(
                job["status"] == "completed" && job["conclusion"] == "success",
                "checks_not_successful",
            )?;
        }
        seen.insert((path.to_owned(), event.to_owned()));
        ids.push(integer(&run["id"])?);
    }
    let expected: BTreeSet<_> = WORKFLOWS
        .iter()
        .flat_map(|(p, _)| events.iter().map(|e| ((*p).to_owned(), (*e).to_owned())))
        .collect();
    require(seen == expected, "missing_workflow")?;
    ids.sort();
    require(
        ids.windows(2).all(|p| p[0] != p[1]),
        "duplicate_workflow_run",
    )?;
    Ok(ids)
}
