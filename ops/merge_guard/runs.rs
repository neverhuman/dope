//! Complete workflow definitions, events, and per-job evidence.
use super::{Refused, Result, array, integer, require, text};
use serde_json::Value;
use std::collections::BTreeSet;

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
pub fn validate_definitions(definitions: &Value, queue: bool) -> Result<()> {
    let map = definitions
        .as_object()
        .ok_or(Refused("workflow_definition_invalid"))?;
    require(
        map.len() == WORKFLOWS.len() && WORKFLOWS.iter().all(|(p, _)| map.contains_key(*p)),
        "workflow_definition_invalid",
    )?;
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
        require(
            keys(block(source, "jobs")?, true) == jobs.iter().map(|j| (*j).to_owned()).collect(),
            "workflow_definition_invalid",
        )?;
    }
    Ok(())
}
pub fn validate_runs(
    runs: &Value,
    source: &Value,
    events: &[&str],
    pr: Option<&Value>,
    branch: Option<&Value>,
    base: Option<&Value>,
) -> Result<Vec<u64>> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for run in array(runs)? {
        let path = text(&run["path"])?
            .split('@')
            .next()
            .ok_or(Refused("check_source_mismatch"))?;
        let jobs = WORKFLOWS
            .iter()
            .find(|(p, _)| *p == path)
            .map(|(_, j)| *j)
            .ok_or(Refused("check_source_mismatch"))?;
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
            actual.len() == jobs.len() && names == jobs.iter().map(|j| (*j).to_owned()).collect(),
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
