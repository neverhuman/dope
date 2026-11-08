//! Strict public synthetic corpus admission; no fixture generation or rebaselining.
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use super::super::vector;
use super::{
    COMPONENTS, GOLDEN_TOLERANCE, MetricError, MetricResult, drivers, require, strict_parse_json,
    values_close,
};

const RANK_CASES: &str = "drivers/informative-v1/rank-cases.json";

fn directory(path: &Path) -> MetricResult<()> {
    let metadata =
        fs::symlink_metadata(path).map_err(|_| MetricError::new("unreadable_golden_directory"))?;
    require(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "invalid_golden_path",
    )
}

fn rank_cases(path: &Path) -> MetricResult<()> {
    let text = fs::read_to_string(path).map_err(|_| MetricError::new("unreadable_golden"))?;
    let value = strict_parse_json(&text)?;
    let cases = value
        .as_array()
        .ok_or(MetricError::new("invalid_rank_fixture"))?;
    require(cases.len() == 7, "invalid_rank_fixture")?;
    let mut names = BTreeSet::new();
    for case in cases {
        let object = case
            .as_object()
            .ok_or(MetricError::new("invalid_rank_fixture"))?;
        let name = case["name"]
            .as_str()
            .ok_or(MetricError::new("invalid_rank_fixture"))?;
        require(
            !name.is_empty() && names.insert(name) && object.len() == 4,
            "invalid_rank_fixture",
        )?;
        let real = vector(&case["real"], true)?;
        let synthetic = vector(&case["synthetic"], true)?;
        let actual = drivers::rank_agreement_informative(&real, &synthetic)?;
        require(
            values_close(&actual, &case["expected"], GOLDEN_TOLERANCE),
            "rank_golden_mismatch",
        )?;
    }
    Ok(())
}

fn walk(
    root: &Path,
    path: &Path,
    depth: usize,
    out: &mut Vec<PathBuf>,
    rank_seen: &mut bool,
) -> MetricResult<()> {
    require(depth <= 32, "golden_directory_depth_limit")?;
    let entries =
        fs::read_dir(path).map_err(|_| MetricError::new("unreadable_golden_directory"))?;
    for entry in entries {
        let entry = entry.map_err(|_| MetricError::new("unreadable_golden_entry"))?;
        let path = entry.path();
        let metadata =
            fs::symlink_metadata(&path).map_err(|_| MetricError::new("unreadable_golden_entry"))?;
        require(!metadata.file_type().is_symlink(), "invalid_golden_path")?;
        if metadata.is_dir() {
            walk(root, &path, depth + 1, out, rank_seen)?;
        } else {
            require(metadata.is_file(), "invalid_golden_path")?;
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if path.strip_prefix(root).ok() == Some(Path::new(RANK_CASES)) {
                    rank_cases(&path)?;
                    *rank_seen = true;
                } else {
                    out.push(path);
                }
            }
        }
    }
    Ok(())
}

/// Discover every standard case recursively, refusing missing/unreadable paths.
/// The seven-case informative-rank array has its own schema and is evaluated here,
/// separately from the standard corpus count. No other nested JSON is excluded.
pub fn discover_cases(root: &Path) -> MetricResult<Vec<PathBuf>> {
    directory(root)?;
    for component in COMPONENTS {
        directory(&root.join(component))?;
    }
    directory(&root.join("drivers/v2"))?;
    directory(&root.join("drivers/informative-v1"))?;
    let mut out = Vec::new();
    let mut rank_seen = false;
    for component in COMPONENTS {
        walk(root, &root.join(component), 0, &mut out, &mut rank_seen)?;
    }
    require(rank_seen, "missing_rank_fixture")?;
    require(!out.is_empty(), "empty_golden_corpus")?;
    out.sort();
    Ok(out)
}
