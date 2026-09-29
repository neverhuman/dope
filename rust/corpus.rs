use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::Table;
use crate::error::{DopeError, Result, io_error};
use crate::model::Task;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub dataset_id: String,
    pub root_id: String,
    pub path: PathBuf,
    pub task: Task,
    pub origin: String,
    pub shape_stratum: String,
    pub difficulty_stratum: String,
    pub rows: usize,
    pub test_rows: usize,
    pub cols: usize,
    pub content_fingerprint: String,
    pub statistical_fingerprint: String,
    pub near_fingerprint: String,
    pub group_id: String,
    pub duplicate_of: Option<String>,
    pub near_duplicate_of: Option<String>,
    pub split: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExclusionRecord {
    pub root_id: String,
    pub path: PathBuf,
    pub reason: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusInventory {
    pub format: String,
    pub version: u8,
    pub roots: Vec<PathBuf>,
    pub datasets: Vec<DatasetRecord>,
    pub exclusions: Vec<ExclusionRecord>,
    pub discovered_paths: usize,
    pub eligible_paths: usize,
    pub checksum: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitManifest {
    pub format: String,
    pub version: u8,
    pub seed: u64,
    pub policy: String,
    pub inventory_checksum: String,
    pub sealed_test: bool,
    pub datasets: Vec<DatasetRecord>,
    pub checksum: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationAssignment {
    pub dataset_id: String,
    pub lineage_group_id: String,
    pub partition: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationSubmanifest {
    pub format: String,
    pub version: u8,
    pub seed: u64,
    pub source_manifest_checksum: String,
    pub selection_numerator: usize,
    pub selection_denominator: usize,
    pub validation_cert_sealed: bool,
    pub validation_select_lineage_groups: usize,
    pub validation_cert_lineage_groups: usize,
    pub assignments: Vec<ValidationAssignment>,
    pub checksum: String,
}

fn checksum_without_field<T>(value: &T, clear: impl FnOnce(&mut T)) -> Result<String>
where
    T: Clone + Serialize,
{
    let mut clone = value.clone();
    clear(&mut clone);
    Ok(blake3::hash(&serde_json::to_vec(&clone)?)
        .to_hex()
        .to_string())
}

fn inventory_checksum(inventory: &CorpusInventory) -> Result<String> {
    checksum_without_field(inventory, |value| value.checksum.clear())
}

fn manifest_checksum(manifest: &SplitManifest) -> Result<String> {
    checksum_without_field(manifest, |value| value.checksum.clear())
}

fn validation_submanifest_checksum(manifest: &ValidationSubmanifest) -> Result<String> {
    checksum_without_field(manifest, |value| value.checksum.clear())
}

pub fn validate_validation_submanifest(manifest: &ValidationSubmanifest) -> Result<()> {
    let select = manifest
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == "validation-select")
        .map(|assignment| &assignment.lineage_group_id)
        .collect::<BTreeSet<_>>();
    let cert = manifest
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == "validation-cert")
        .map(|assignment| &assignment.lineage_group_id)
        .collect::<BTreeSet<_>>();
    let disjoint = select.is_disjoint(&cert);
    let frozen_counts_valid = select.len() + cert.len() != FROZEN_VALIDATION_LINEAGES
        || (select.len() == FROZEN_VALIDATION_SELECT_LINEAGES
            && cert.len() == FROZEN_VALIDATION_CERT_LINEAGES);
    if manifest.format != "dope-validation-submanifest"
        || manifest.version != 1
        || manifest.selection_numerator != 3
        || manifest.selection_denominator != 5
        || !manifest.validation_cert_sealed
        || manifest.validation_select_lineage_groups != select.len()
        || manifest.validation_cert_lineage_groups != cert.len()
        || !manifest.assignments.iter().all(|assignment| {
            matches!(
                assignment.partition.as_str(),
                "validation-select" | "validation-cert"
            ) && !assignment.dataset_id.is_empty()
                && !assignment.lineage_group_id.is_empty()
        })
        || !disjoint
        || !frozen_counts_valid
        || validation_submanifest_checksum(manifest)? != manifest.checksum
    {
        return Err(DopeError::Data(
            "invalid validation-select/validation-cert submanifest".into(),
        ));
    }
    Ok(())
}

pub const FROZEN_VALIDATION_LINEAGES: usize = 19_712;
pub const FROZEN_VALIDATION_SELECT_LINEAGES: usize = 11_892;
pub const FROZEN_VALIDATION_CERT_LINEAGES: usize = 7_820;

/// Splits validation lineages without retaining paths or opening dataset rows.
/// The frozen corpus uses the predeclared 11,892/7,820 allocation. Other
/// manifests retain deterministic 60/40 behavior for fixtures and tools.
pub fn split_validation_manifest(
    manifest: &SplitManifest,
    seed: u64,
) -> Result<ValidationSubmanifest> {
    validate_split_manifest(manifest)?;
    let mut lineage_strata = BTreeMap::<String, BTreeSet<String>>::new();
    for record in manifest
        .datasets
        .iter()
        .filter(|record| record.split == "validation")
    {
        let stratum = format!(
            "{}:{}:{}",
            record.task.as_str(),
            record.shape_stratum,
            record.difficulty_stratum
        );
        lineage_strata
            .entry(record.group_id.clone())
            .or_default()
            .insert(stratum);
    }
    let lineage_count = lineage_strata.len();
    let select_target = if lineage_count == FROZEN_VALIDATION_LINEAGES {
        FROZEN_VALIDATION_SELECT_LINEAGES
    } else {
        (lineage_count * 3 + 2) / 5
    };
    let mut by_stratum = BTreeMap::<String, Vec<String>>::new();
    for (lineage, strata) in lineage_strata {
        by_stratum
            .entry(strata.into_iter().collect::<Vec<_>>().join("|"))
            .or_default()
            .push(lineage);
    }
    let mut quotas = BTreeMap::new();
    let mut assigned = 0usize;
    let mut remainders = Vec::new();
    for (stratum, lineages) in &by_stratum {
        let numerator = lineages.len() * select_target;
        let base = numerator / lineage_count.max(1);
        quotas.insert(stratum.clone(), base);
        assigned += base;
        remainders.push((numerator % lineage_count.max(1), stratum.clone()));
    }
    remainders.sort_by(|left, right| right.cmp(left));
    for (_, stratum) in remainders.into_iter().take(select_target - assigned) {
        *quotas.get_mut(&stratum).expect("known validation stratum") += 1;
    }
    let mut lineage_partition = BTreeMap::new();
    for (stratum, mut lineages) in by_stratum {
        lineages.sort_by_key(|lineage| (split_hash(seed, &stratum, lineage), lineage.clone()));
        let quota = quotas[&stratum];
        for (index, lineage) in lineages.into_iter().enumerate() {
            lineage_partition.insert(
                lineage,
                if index < quota {
                    "validation-select"
                } else {
                    "validation-cert"
                },
            );
        }
    }
    let mut assignments = manifest
        .datasets
        .iter()
        .filter(|record| record.split == "validation")
        .map(|record| ValidationAssignment {
            dataset_id: record.dataset_id.clone(),
            lineage_group_id: record.group_id.clone(),
            partition: lineage_partition[&record.group_id].into(),
        })
        .collect::<Vec<_>>();
    assignments.sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    let mut result = ValidationSubmanifest {
        format: "dope-validation-submanifest".into(),
        version: 1,
        seed,
        source_manifest_checksum: manifest.checksum.clone(),
        selection_numerator: 3,
        selection_denominator: 5,
        validation_cert_sealed: true,
        validation_select_lineage_groups: select_target,
        validation_cert_lineage_groups: lineage_count - select_target,
        assignments,
        checksum: String::new(),
    };
    result.checksum = validation_submanifest_checksum(&result)?;
    Ok(result)
}

fn exclusion_reason(name: &str) -> Option<&'static str> {
    let normalized = name.to_ascii_lowercase().replace(['-', ' '], "_");
    let tokens = [
        ("multilabel", "multi_label"),
        ("multi_label", "multi_label"),
        ("quarantine", "quarantine"),
        ("cache", "cache"),
        ("audit", "audit"),
        ("failed", "failed"),
        ("failure", "failed"),
        ("test_fixture", "test_fixture"),
        ("fixture", "test_fixture"),
        ("malformed", "malformed"),
        ("unsupported", "unsupported"),
        ("incomplete", "incomplete"),
    ];
    tokens
        .into_iter()
        .find_map(|(needle, reason)| normalized.contains(needle).then_some(reason))
}

fn canonical_table_bytes(table: &Table) -> Vec<u8> {
    let mut feature_order = Vec::<([u8; 32], usize)>::with_capacity(table.features);
    for (column_index, column) in table.columns.iter().enumerate() {
        let mut sorted: Vec<u32> = column.iter().map(|value| value.to_bits()).collect();
        sorted.sort_unstable();
        let mut hasher = blake3::Hasher::new();
        for value in sorted {
            hasher.update(&value.to_le_bytes());
        }
        feature_order.push((*hasher.finalize().as_bytes(), column_index));
    }
    feature_order.sort_unstable();
    let mut rows: Vec<Vec<u32>> = (0..table.rows)
        .map(|row| {
            let mut values: Vec<u32> = feature_order
                .iter()
                .map(|(_, column)| table.columns[*column][row].to_bits())
                .collect();
            values.push(table.target[row].to_bits());
            values
        })
        .collect();
    rows.sort_unstable();
    let mut bytes = Vec::with_capacity(table.rows * (table.features + 1) * 4);
    for row in rows {
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn quantile(sorted: &[f32], index: usize, denominator: usize) -> f32 {
    sorted
        .get(index * sorted.len().saturating_sub(1) / denominator.max(1))
        .copied()
        .unwrap_or(0.0)
}

fn correlation(left: &[f32], right: &[f32]) -> f32 {
    let pairs: Vec<(f64, f64)> = left
        .iter()
        .zip(right)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (f64::from(*x), f64::from(*y)))
        .collect();
    if pairs.len() < 3 {
        return 0.0;
    }
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / pairs.len() as f64;
    let numerator = pairs
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let xx = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>();
    let yy = pairs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>();
    if xx * yy <= 1e-18 {
        0.0
    } else {
        (numerator / (xx * yy).sqrt()) as f32
    }
}

fn statistical_fingerprints(train: &Table, test: Option<&Table>) -> (String, String) {
    let mut exact_sketches = Vec::<Vec<u8>>::new();
    let mut lsh_sketches = Vec::<Vec<u8>>::new();
    for table in std::iter::once(train).chain(test) {
        for column in table.columns.iter().chain(std::iter::once(&table.target)) {
            let mut sorted: Vec<f32> = column
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect();
            sorted.sort_by(f32::total_cmp);
            let mut exact = Vec::with_capacity(17 * 4);
            let mut lsh = Vec::with_capacity(9);
            for index in 0..17 {
                exact.extend_from_slice(&quantile(&sorted, index, 16).to_bits().to_le_bytes());
            }
            for index in 0..9 {
                lsh.push((quantile(&sorted, index, 8).clamp(0.0, 1.0) * 32.0).round() as u8);
            }
            exact_sketches.push(exact);
            lsh_sketches.push(lsh);
        }
    }
    exact_sketches.sort();
    lsh_sketches.sort();
    let mut exact = Vec::new();
    for sketch in exact_sketches {
        exact.extend_from_slice(&sketch);
    }
    let mut lsh = Vec::new();
    lsh.extend_from_slice(&(train.rows.next_power_of_two() as u64).to_le_bytes());
    lsh.extend_from_slice(&(train.features as u64).to_le_bytes());
    for sketch in lsh_sketches {
        lsh.extend_from_slice(&sketch);
    }
    let mut correlations: Vec<i8> = train
        .columns
        .iter()
        .map(|column| (correlation(column, &train.target).clamp(-1.0, 1.0) * 16.0).round() as i8)
        .collect();
    correlations.sort_unstable();
    lsh.extend(correlations.into_iter().map(|value| value as u8));
    (
        blake3::hash(&exact).to_hex().to_string(),
        blake3::hash(&lsh).to_hex().to_string(),
    )
}

fn inferred_task(train: &Table, test: Option<&Table>) -> Task {
    if std::iter::once(train)
        .chain(test)
        .flat_map(|table| &table.target)
        .all(|value| *value <= 1e-6 || *value >= 1.0 - 1e-6)
    {
        Task::Binary
    } else {
        Task::Regression
    }
}

fn scrub_seed_fields(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(entries) => serde_json::Value::Object(
            entries
                .iter()
                .filter(|(key, _)| {
                    let key = key.to_ascii_lowercase();
                    !key.contains("seed")
                        && key != "fold"
                        && key != "run"
                        && key != "run_id"
                        && key != "random_state"
                })
                .map(|(key, value)| (key.clone(), scrub_seed_fields(value)))
                .collect(),
        ),
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.iter().map(scrub_seed_fields).collect())
        }
        _ => value.clone(),
    }
}

fn metadata_group(path: &Path, fallback: &str) -> String {
    let metadata = fs::read(path.join("meta.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    if let Some(value) = metadata {
        let keys = [
            "source",
            "real_source",
            "source_path",
            "source_file",
            "source_real_data",
            "generator",
            "generator_name",
            "version",
            "generator_version",
            "recipe_family",
            "recipe",
            "generator_recipe",
            "lineage",
            "lineage_fingerprint",
        ];
        let selected: BTreeMap<_, _> = keys
            .into_iter()
            .filter_map(|key| value.get(key).map(|entry| (key, scrub_seed_fields(entry))))
            .collect();
        if !selected.is_empty() {
            return blake3::hash(&serde_json::to_vec(&selected).expect("JSON values serialize"))
                .to_hex()
                .to_string();
        }
    }
    blake3::hash(fallback.as_bytes()).to_hex().to_string()
}

fn origin(root: &Path, path: &Path) -> String {
    let name = format!("{} {}", root.display(), path.display()).to_ascii_lowercase();
    if name.contains("remote_super") {
        "mirror".into()
    } else if name.contains("synthetic") {
        "synthetic".into()
    } else {
        "real".into()
    }
}

fn shape_stratum(rows: usize, features: usize) -> String {
    let row_bucket = rows.max(1).ilog2().min(31);
    let feature_bucket = features.max(1).ilog2().min(15);
    format!("r{row_bucket}-p{feature_bucket}")
}

fn difficulty_stratum(table: &Table) -> String {
    let signal = table
        .columns
        .iter()
        .map(|column| correlation(column, &table.target).abs())
        .fold(0.0f32, f32::max);
    let missing = table
        .columns
        .iter()
        .flatten()
        .filter(|value| value.is_nan())
        .count() as f64
        / (table.rows * table.features).max(1) as f64;
    let signal_bucket = if signal < 0.1 {
        "hard"
    } else if signal < 0.4 {
        "medium"
    } else {
        "easy"
    };
    let missing_bucket = if missing >= 0.1 { "missing" } else { "dense" };
    format!("{signal_bucket}-{missing_bucket}")
}

fn root_id(index: usize, root: &Path) -> String {
    let name = root
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("root");
    format!("{index}:{name}")
}

fn dataset_candidates(root: &Path) -> Result<Vec<PathBuf>> {
    if root.join("train.csv").is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut candidates = Vec::new();
    for entry in fs::read_dir(root).map_err(|error| io_error(root, error))? {
        let entry = entry.map_err(|error| io_error(root, error))?;
        candidates.push(entry.path());
    }
    candidates.sort();
    Ok(candidates)
}

include!("corpus/inventory_and_splits.rs");
