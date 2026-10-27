//! Private, exact-fit CSV-bound cross-validation inputs. These never become code sidecars.
use super::manifest::{bad, check, Schema};
use crate::{landmark::LandmarkOptions, levels::digest, Result};
use serde_json::Value;
use std::path::Path;

pub const RECIPE: &str = "fit-cv-gain23-mask-squash-v1";
pub const OBSERVED_RECIPE: &str = "fit-cv-gain23-observed-mask-squash-v2";

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
    load_for_recipe(directory, expected, csv, schema, RECIPE)
}

/// The model graph, rather than a caller-selected mask, binds the feature recipe.
pub fn load_for_recipe(
    directory: &Path,
    expected: &str,
    csv: &Path,
    schema: &Schema,
    recipe: &str,
) -> Result<[f32; 46]> {
    check(
        matches!(recipe, RECIPE | OBSERVED_RECIPE),
        "unknown CV landmark recipe",
    )?;
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
            && manifest["format"]
                == if recipe == RECIPE {
                    "dope-fit-cv-landmark-input-v1"
                } else {
                    "dope-fit-cv-landmark-input-v2"
                }
            && manifest["feature_recipe"] == recipe
            && options == expected_options
            && matches!(
                options.work_budget,
                2_000_000_000 | 20_000_000_000 | 2_000_000_000_000
            )
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
        if (cell["state"] == "complete"
            || (recipe == OBSERVED_RECIPE && cell["state"] == "uninformative"))
            && !cell["gain"].is_null()
        {
            let gain = cell["gain"]
                .as_f64()
                .ok_or_else(|| bad("invalid CV landmark gain"))? as f32;
            check(gain.is_finite(), "nonfinite CV landmark gain")?;
            check(
                cell["query_rows"] == expected_query_rows(i + 1, rows, &options)
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

/// Optional landmark feature validation (allows an empty or partial mask;
/// requires finite values in `[-1, 1]`, `{0, 1}` mask entries, and `+0.0` on
/// unmasked slots).
pub fn validate_optional(features: &[f32; 46]) -> Result<()> {
    check(
        features.iter().all(|v| v.is_finite())
            && features[..23].iter().all(|v| v.abs() <= 1.)
            && features[23..].iter().all(|v| *v == 0. || *v == 1.)
            && (0..23).all(|i| features[23 + i] == 1. || features[i].to_bits() == 0),
        "invalid optional CV landmark features",
    )
}

fn expected_query_rows(coordinate: usize, rows: u64, options: &LandmarkOptions) -> u64 {
    let folds = options.folds as u64;
    let cap = if coordinate <= 15 || coordinate == 22 {
        options.ridge_rows as u64
    } else {
        options.small_rows.div_ceil(options.folds) as u64
    };
    (0..folds)
        .map(|fold| (rows / folds + u64::from(fold < rows % folds)).min(cap))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iid_query_counts_bind_the_native_per_fold_caps() {
        let options = LandmarkOptions::default();
        for id in 1..=23 {
            assert_eq!(expected_query_rows(id, 100, &options), 100);
        }
        assert_eq!(expected_query_rows(1, 50_000, &options), 40_960);
        assert_eq!(expected_query_rows(22, 50_000, &options), 40_960);
        assert_eq!(expected_query_rows(16, 5_001, &options), 4_100);
        assert_eq!(expected_query_rows(23, 5_001, &options), 4_100);
        assert_eq!(expected_query_rows(23, 4_097, &options), 4_097);
    }
}
