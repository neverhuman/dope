//! Private, exact-fit CSV-bound cross-validation inputs. These never become code sidecars.
use super::manifest::{bad, check, Schema};
use crate::{landmark::LandmarkOptions, levels::digest, Result};
use serde_json::Value;
use std::path::Path;

pub const RECIPE: &str = "fit-cv-gain23-mask-squash-v1";

fn read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path)?;
    check(
        meta.is_file() && meta.len() <= limit,
        "CV landmark file envelope",
    )?;
    Ok(std::fs::read(path)?)
}

/// Caller supplies a producer-manifest digest. The producer reads only the declared
/// fit partition; native code checks exact CSV/report/options/schema bindings again.
pub fn load(directory: &Path, expected: &str, csv: &Path, schema: &Schema) -> Result<[f32; 46]> {
    let raw = read(&directory.join("manifest.json"), 262144)?;
    check(
        digest(&raw) == expected,
        "CV landmark manifest digest mismatch",
    )?;
    let manifest: Value =
        serde_json::from_slice(&raw).map_err(|_| bad("invalid CV landmark manifest"))?;
    let scope = &manifest["scope"];
    let options: LandmarkOptions = serde_json::from_value(manifest["options"].clone())
        .map_err(|_| bad("invalid CV landmark options"))?;
    let expected_options = LandmarkOptions {
        work_budget: options.work_budget,
        ..LandmarkOptions::default()
    };
    let kinds: Vec<_> = schema.columns.iter().map(|c| c.kind.as_str()).collect();
    check(
        manifest.as_object().is_some_and(|v| v.len() == 7)
            && manifest["format"] == "dope-fit-cv-landmark-input-v1"
            && manifest["feature_recipe"] == RECIPE
            && options == expected_options
            && matches!(options.work_budget, 2_000_000_000 | 20_000_000_000)
            && manifest["engine_sha256"].as_str().is_some_and(|v| {
                v.len() == 64
                    && v.bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            })
            && scope["allowed_roles"] == serde_json::json!(["fit"])
            && scope["split_policy"] == "iid"
            && scope["training_input_sha256"] == digest(&read(csv, 200000000)?)
            && scope["column_kinds"] == serde_json::json!(kinds)
            && scope["task"].as_str() == schema.task.as_deref()
            && schema.columns.iter().enumerate().all(|(i, c)| {
                c.slot
                    == if i + 1 == schema.columns.len() {
                        0
                    } else {
                        (i + 1) as u32
                    }
            }),
        "CV landmark input scope mismatch",
    )?;
    let raw = read(&directory.join("report.json"), 1048576)?;
    check(
        manifest["report_sha256"] == digest(&raw),
        "CV landmark report digest mismatch",
    )?;
    let report: Value =
        serde_json::from_slice(&raw).map_err(|_| bad("invalid CV landmark report"))?;
    let cells = report["cells"]
        .as_array()
        .ok_or_else(|| bad("missing CV landmark cells"))?;
    let rows = scope["rows"]
        .as_u64()
        .ok_or_else(|| bad("missing CV landmark row count"))?;
    check(
        rows >= 20
            && cells.len() == 23
            && report["panel"] == "landmarks-native-v1"
            && report["kind"] == "real"
            && report["privacy"] == "private"
            && report["options"] == manifest["options"]
            && report["rows"] == rows
            && report["task"] == scope["task"],
        "CV landmark report binding mismatch",
    )?;
    let mut features = [0f32; 46];
    for (i, cell) in cells.iter().enumerate() {
        check(cell["id"] == i + 1, "CV landmark coordinate order mismatch")?;
        if cell["state"] == "complete" && !cell["gain"].is_null() {
            let gain = cell["gain"]
                .as_f64()
                .ok_or_else(|| bad("invalid CV landmark gain"))? as f32;
            check(gain.is_finite(), "nonfinite CV landmark gain")?;
            check(
                cell["query_rows"] == rows
                    && cell["train_rows"].as_array().is_some_and(|v| {
                        v.len() == 5
                            && v.iter()
                                .all(|n| n.as_u64().is_some_and(|n| n > 0 && n < rows))
                    }),
                "CV landmark out-of-fold evidence required",
            )?;
            features[i] = gain / (1. + gain.abs());
            features[23 + i] = 1.;
        }
    }
    validate(&features)?;
    Ok(features)
}

pub(crate) fn validate(features: &[f32; 46]) -> Result<()> {
    check(
        features.iter().all(|v| v.is_finite())
            && features[..23].iter().all(|v| v.abs() <= 1.)
            && features[23..].iter().all(|v| *v == 0. || *v == 1.)
            && features[23..].contains(&1.)
            && (0..23).all(|i| features[23 + i] == 1. || features[i] == 0.),
        "invalid or empty CV landmark features",
    )
}
