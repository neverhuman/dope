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

pub fn inventory_corpus(roots: &[PathBuf]) -> Result<CorpusInventory> {
    if roots.is_empty() {
        return Err(DopeError::Data(
            "at least one corpus root is required".into(),
        ));
    }
    let mut canonical_roots = Vec::with_capacity(roots.len());
    let mut datasets = Vec::new();
    let mut exclusions = Vec::new();
    let mut exact_seen = BTreeMap::<String, (String, String)>::new();
    let mut near_seen = BTreeMap::<String, (String, String)>::new();

    for (root_index, requested_root) in roots.iter().enumerate() {
        let root = requested_root
            .canonicalize()
            .map_err(|error| io_error(requested_root, error))?;
        let root_id = root_id(root_index, &root);
        canonical_roots.push(root.clone());
        for path in dataset_candidates(&root)? {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            let exclude = |reason: &str, detail: String| ExclusionRecord {
                root_id: root_id.clone(),
                path: path.clone(),
                reason: reason.into(),
                detail,
            };
            if name.starts_with('.') {
                exclusions.push(exclude("hidden", "hidden corpus entry".into()));
                continue;
            }
            if !path.is_dir() {
                exclusions.push(exclude(
                    "unsupported",
                    "corpus entry is not a directory".into(),
                ));
                continue;
            }
            if let Some(reason) = exclusion_reason(name) {
                exclusions.push(exclude(
                    reason,
                    "directory name matched exclusion policy".into(),
                ));
                continue;
            }
            let train_path = path.join("train.csv");
            if !train_path.is_file() {
                exclusions.push(exclude("incomplete", "missing train.csv".into()));
                continue;
            }
            let test_path = path.join("test.csv");
            if !test_path.is_file() {
                exclusions.push(exclude("incomplete", "missing test.csv".into()));
                continue;
            }
            let raw_train = match Table::read_csv(&train_path, Task::Regression) {
                Ok(table) => table,
                Err(error) => {
                    exclusions.push(exclude("malformed", error.to_string()));
                    continue;
                }
            };
            let raw_test = if test_path.is_file() {
                match Table::read_csv(&test_path, Task::Regression) {
                    Ok(table) if table.features == raw_train.features => Some(table),
                    Ok(_) => {
                        exclusions.push(exclude(
                            "malformed",
                            "train.csv and test.csv feature widths differ".into(),
                        ));
                        continue;
                    }
                    Err(error) => {
                        exclusions.push(exclude("malformed", error.to_string()));
                        continue;
                    }
                }
            } else {
                None
            };
            let task = inferred_task(&raw_train, raw_test.as_ref());
            let train = if task == Task::Binary {
                Table::read_csv(&train_path, task)?
            } else {
                raw_train
            };
            let test = if task == Task::Binary {
                raw_test
                    .as_ref()
                    .map(|_| Table::read_csv(&test_path, task))
                    .transpose()?
            } else {
                raw_test
            };
            let mut content_hasher = blake3::Hasher::new();
            content_hasher.update(b"train\0");
            content_hasher.update(&canonical_table_bytes(&train));
            if let Some(test) = &test {
                content_hasher.update(b"test\0");
                content_hasher.update(&canonical_table_bytes(test));
            }
            let content_fingerprint = content_hasher.finalize().to_hex().to_string();
            let (statistical_fingerprint, near_fingerprint) =
                statistical_fingerprints(&train, test.as_ref());
            let relative = path.strip_prefix(&root).unwrap_or(&path);
            let relative = if relative.as_os_str().is_empty() {
                ".".into()
            } else {
                relative.to_string_lossy().into_owned()
            };
            let dataset_id = if roots.len() == 1 {
                relative
            } else {
                format!("{root_id}/{relative}")
            };
            let mut group_id = metadata_group(&path, &content_fingerprint);
            let duplicate_of =
                exact_seen
                    .get(&content_fingerprint)
                    .map(|(dataset_id, prior_group)| {
                        group_id.clone_from(prior_group);
                        dataset_id.clone()
                    });
            let near_duplicate_of = if duplicate_of.is_none() {
                near_seen
                    .get(&near_fingerprint)
                    .map(|(dataset_id, prior_group)| {
                        group_id.clone_from(prior_group);
                        dataset_id.clone()
                    })
            } else {
                None
            };
            exact_seen
                .entry(content_fingerprint.clone())
                .or_insert_with(|| (dataset_id.clone(), group_id.clone()));
            near_seen
                .entry(near_fingerprint.clone())
                .or_insert_with(|| (dataset_id.clone(), group_id.clone()));
            datasets.push(DatasetRecord {
                dataset_id,
                root_id: root_id.clone(),
                path: path.clone(),
                task,
                origin: origin(&root, &path),
                shape_stratum: shape_stratum(train.rows, train.features),
                difficulty_stratum: difficulty_stratum(&train),
                rows: train.rows,
                test_rows: test.as_ref().map_or(0, |table| table.rows),
                cols: train.features + 1,
                content_fingerprint,
                statistical_fingerprint,
                near_fingerprint,
                group_id,
                duplicate_of,
                near_duplicate_of,
                split: String::new(),
            });
        }
    }
    datasets.sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    exclusions.sort_by(|left, right| left.path.cmp(&right.path));
    if datasets.is_empty() {
        return Err(DopeError::Data(
            "no eligible train.csv datasets found".into(),
        ));
    }
    let mut inventory = CorpusInventory {
        format: "dope-corpus-inventory".into(),
        version: 2,
        roots: canonical_roots,
        discovered_paths: datasets.len() + exclusions.len(),
        eligible_paths: datasets.len(),
        datasets,
        exclusions,
        checksum: String::new(),
    };
    inventory.checksum = inventory_checksum(&inventory)?;
    Ok(inventory)
}

pub fn load_inventory(path: &Path) -> Result<CorpusInventory> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    let inventory: CorpusInventory = serde_json::from_slice(&bytes)?;
    if inventory.format != "dope-corpus-inventory"
        || inventory.version != 2
        || inventory.checksum != inventory_checksum(&inventory)?
    {
        return Err(DopeError::Data(
            "inventory checksum or version invalid".into(),
        ));
    }
    Ok(inventory)
}

pub fn refresh_inventory_lineages(inventory: &CorpusInventory) -> Result<CorpusInventory> {
    if inventory.format != "dope-corpus-inventory"
        || inventory.version != 2
        || inventory.checksum != inventory_checksum(inventory)?
    {
        return Err(DopeError::Data(
            "inventory checksum or version invalid".into(),
        ));
    }
    let mut refreshed = inventory.clone();
    refreshed
        .datasets
        .sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    let mut exact_seen = BTreeMap::<String, (String, String)>::new();
    let mut near_seen = BTreeMap::<String, (String, String)>::new();
    for record in &mut refreshed.datasets {
        let mut group_id = metadata_group(&record.path, &record.content_fingerprint);
        record.duplicate_of =
            exact_seen
                .get(&record.content_fingerprint)
                .map(|(dataset_id, prior_group)| {
                    group_id.clone_from(prior_group);
                    dataset_id.clone()
                });
        record.near_duplicate_of = if record.duplicate_of.is_none() {
            near_seen
                .get(&record.near_fingerprint)
                .map(|(dataset_id, prior_group)| {
                    group_id.clone_from(prior_group);
                    dataset_id.clone()
                })
        } else {
            None
        };
        exact_seen
            .entry(record.content_fingerprint.clone())
            .or_insert_with(|| (record.dataset_id.clone(), group_id.clone()));
        near_seen
            .entry(record.near_fingerprint.clone())
            .or_insert_with(|| (record.dataset_id.clone(), group_id.clone()));
        record.group_id = group_id;
        record.split.clear();
    }
    refreshed.checksum.clear();
    refreshed.checksum = inventory_checksum(&refreshed)?;
    Ok(refreshed)
}

fn split_hash(seed: u64, stratum: &str, group: &str) -> u64 {
    u64::from_le_bytes(
        blake3::hash(format!("{seed}:{stratum}:{group}").as_bytes()).as_bytes()[..8]
            .try_into()
            .expect("eight-byte hash prefix"),
    )
}

pub fn split_inventory(inventory: &CorpusInventory, seed: u64) -> Result<SplitManifest> {
    if inventory.format != "dope-corpus-inventory"
        || inventory.version != 2
        || inventory.checksum != inventory_checksum(inventory)?
    {
        return Err(DopeError::Data(
            "inventory checksum or version invalid".into(),
        ));
    }
    let mut group_strata = BTreeMap::<String, String>::new();
    let mut group_weights = BTreeMap::<String, usize>::new();
    for record in &inventory.datasets {
        *group_weights.entry(record.group_id.clone()).or_default() += 1;
        group_strata
            .entry(record.group_id.clone())
            .or_insert_with(|| {
                format!(
                    "{}:{}:{}:{}",
                    record.task.as_str(),
                    record.origin,
                    record.shape_stratum,
                    record.difficulty_stratum
                )
            });
    }
    let mut assignments = BTreeMap::<String, String>::new();
    let mut strata = BTreeMap::<String, Vec<String>>::new();
    for (group, stratum) in &group_strata {
        strata
            .entry(stratum.clone())
            .or_default()
            .push(group.clone());
    }
    const SPLITS: [&str; 3] = ["train", "validation", "test"];
    const FRACTIONS: [f64; 3] = [0.70, 0.15, 0.15];
    for (stratum, mut groups) in strata {
        groups.sort_by(|left, right| {
            group_weights[right]
                .cmp(&group_weights[left])
                .then_with(|| {
                    split_hash(seed, &stratum, left).cmp(&split_hash(seed, &stratum, right))
                })
                .then_with(|| left.cmp(right))
        });
        let total = groups
            .iter()
            .map(|group| group_weights[group])
            .sum::<usize>() as f64;
        let targets = FRACTIONS.map(|fraction| fraction * total);
        let mut assigned = [0.0f64; 3];
        for group in groups {
            let weight = group_weights[&group] as f64;
            let rotation = (split_hash(seed ^ 0x9e37_79b9, &stratum, &group) % 3) as usize;
            let split_index = (0..3)
                .min_by(|left, right| {
                    let cost = |candidate: usize| {
                        (0..3)
                            .map(|index| {
                                let next =
                                    assigned[index] + if index == candidate { weight } else { 0.0 };
                                (next - targets[index]).powi(2)
                            })
                            .sum::<f64>()
                    };
                    cost(*left).total_cmp(&cost(*right)).then_with(|| {
                        ((*left + 3 - rotation) % 3).cmp(&((*right + 3 - rotation) % 3))
                    })
                })
                .expect("three split candidates");
            assigned[split_index] += weight;
            assignments.insert(group, SPLITS[split_index].into());
        }
    }
    if assignments.len() == 1 {
        let group = assignments.keys().next().expect("one assignment").clone();
        assignments.insert(group, "train".into());
    } else if assignments.len() == 2 {
        let mut groups: Vec<_> = group_strata.keys().cloned().collect();
        groups.sort_by_key(|group| split_hash(seed, &group_strata[group], group));
        assignments.insert(groups[0].clone(), "train".into());
        assignments.insert(groups[1].clone(), "validation".into());
    } else if assignments.len() >= 3 {
        for required in ["train", "validation", "test"] {
            if !assignments.values().any(|split| split == required) {
                let candidate = group_strata
                    .iter()
                    .filter(|(group, _)| {
                        assignments
                            .values()
                            .filter(|split| *split == &assignments[*group])
                            .count()
                            > 1
                    })
                    .min_by_key(|(group, stratum)| split_hash(seed ^ 0xa5a5, stratum, group))
                    .map(|(group, _)| group.clone())
                    .ok_or_else(|| DopeError::Data("unable to form non-empty splits".into()))?;
                assignments.insert(candidate, required.into());
            }
        }
    }
    let mut datasets = inventory.datasets.clone();
    for record in &mut datasets {
        record.split.clone_from(&assignments[&record.group_id]);
    }
    let mut manifest = SplitManifest {
        format: "dope-corpus-split".into(),
        version: 2,
        seed,
        policy: "exact-and-quantile-rank-lsh-deduplicated-lineage-weighted-stratified-70-15-15"
            .into(),
        inventory_checksum: inventory.checksum.clone(),
        sealed_test: true,
        datasets,
        checksum: String::new(),
    };
    manifest.checksum = manifest_checksum(&manifest)?;
    validate_split_manifest(&manifest)?;
    Ok(manifest)
}

pub fn build_split_manifest_multi(
    roots: &[PathBuf],
    seed: u64,
) -> Result<(CorpusInventory, SplitManifest)> {
    let inventory = inventory_corpus(roots)?;
    let manifest = split_inventory(&inventory, seed)?;
    Ok((inventory, manifest))
}

pub fn build_split_manifest(root: &Path, seed: u64) -> Result<SplitManifest> {
    let (_, manifest) = build_split_manifest_multi(&[root.to_path_buf()], seed)?;
    Ok(manifest)
}

pub fn validate_split_manifest(manifest: &SplitManifest) -> Result<()> {
    if manifest.format != "dope-corpus-split"
        || manifest.version != 2
        || manifest.checksum != manifest_checksum(manifest)?
    {
        return Err(DopeError::Data(
            "split manifest checksum or version invalid".into(),
        ));
    }
    let mut groups = BTreeMap::new();
    let mut content = BTreeMap::new();
    let mut near = BTreeMap::new();
    let valid_splits: BTreeSet<&str> = ["train", "validation", "test"].into_iter().collect();
    for record in &manifest.datasets {
        if !valid_splits.contains(record.split.as_str()) {
            return Err(DopeError::Data("unknown split assignment".into()));
        }
        if groups
            .insert(record.group_id.clone(), record.split.clone())
            .is_some_and(|prior| prior != record.split)
        {
            return Err(DopeError::Data("lineage leakage across splits".into()));
        }
        if content
            .insert(record.content_fingerprint.clone(), record.split.clone())
            .is_some_and(|prior| prior != record.split)
        {
            return Err(DopeError::Data("duplicate leakage across splits".into()));
        }
        if near
            .insert(record.near_fingerprint.clone(), record.split.clone())
            .is_some_and(|prior| prior != record.split)
        {
            return Err(DopeError::Data(
                "near-duplicate leakage across splits".into(),
            ));
        }
    }
    Ok(())
}

pub fn write_inventory(inventory: &CorpusInventory, path: &Path) -> Result<()> {
    if inventory.checksum != inventory_checksum(inventory)? {
        return Err(DopeError::Data("inventory checksum invalid".into()));
    }
    fs::write(path, serde_json::to_vec_pretty(inventory)?).map_err(|error| io_error(path, error))
}

pub fn write_exclusions(inventory: &CorpusInventory, path: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    for exclusion in &inventory.exclusions {
        serde_json::to_writer(&mut bytes, exclusion)?;
        bytes.push(b'\n');
    }
    fs::write(path, bytes).map_err(|error| io_error(path, error))
}

pub fn write_split_manifest(manifest: &SplitManifest, path: &Path) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(manifest)?).map_err(|error| io_error(path, error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn multi_root_inventory_accounts_for_every_immediate_child() {
        let root = std::env::temp_dir().join(format!("dope-inventory-{}", std::process::id()));
        let first = root.join("first/good");
        let duplicate = root.join("second/mirror");
        let excluded = root.join("first/cache-copy");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&duplicate).unwrap();
        fs::create_dir_all(&excluded).unwrap();
        for path in [&first, &duplicate] {
            let mut train = fs::File::create(path.join("train.csv")).unwrap();
            writeln!(train, "0,1,0").unwrap();
            writeln!(train, "1,0,1").unwrap();
            let mut test = fs::File::create(path.join("test.csv")).unwrap();
            writeln!(test, "1,1,1").unwrap();
        }
        let roots = vec![root.join("first"), root.join("second")];
        let inventory = inventory_corpus(&roots).unwrap();
        assert_eq!(inventory.discovered_paths, 3);
        assert_eq!(inventory.datasets.len(), 2);
        assert_eq!(inventory.exclusions.len(), 1);
        assert!(inventory.datasets[1].duplicate_of.is_some());
        let manifest = split_inventory(&inventory, 1729).unwrap();
        validate_split_manifest(&manifest).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn weighted_split_hits_requested_ratio_for_independent_groups() {
        let template = DatasetRecord {
            dataset_id: String::new(),
            root_id: "0:test".into(),
            path: PathBuf::from("/tmp/test"),
            task: Task::Regression,
            origin: "synthetic".into(),
            shape_stratum: "r9-p4".into(),
            difficulty_stratum: "medium-dense".into(),
            rows: 512,
            test_rows: 256,
            cols: 17,
            content_fingerprint: String::new(),
            statistical_fingerprint: String::new(),
            near_fingerprint: String::new(),
            group_id: String::new(),
            duplicate_of: None,
            near_duplicate_of: None,
            split: String::new(),
        };
        let datasets = (0..100)
            .map(|index| DatasetRecord {
                dataset_id: format!("dataset-{index:03}"),
                content_fingerprint: format!("content-{index:03}"),
                statistical_fingerprint: format!("statistics-{index:03}"),
                near_fingerprint: format!("near-{index:03}"),
                group_id: format!("group-{index:03}"),
                ..template.clone()
            })
            .collect();
        let mut inventory = CorpusInventory {
            format: "dope-corpus-inventory".into(),
            version: 2,
            roots: vec![PathBuf::from("/tmp")],
            datasets,
            exclusions: Vec::new(),
            discovered_paths: 100,
            eligible_paths: 100,
            checksum: String::new(),
        };
        inventory.checksum = inventory_checksum(&inventory).unwrap();
        let manifest = split_inventory(&inventory, 1729).unwrap();
        let mut counts = BTreeMap::<String, usize>::new();
        for record in manifest.datasets {
            *counts.entry(record.split).or_default() += 1;
        }
        assert_eq!(counts["train"], 70);
        assert_eq!(counts["validation"], 15);
        assert_eq!(counts["test"], 15);
    }
}
