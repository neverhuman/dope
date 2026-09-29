use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::embedding::{
    DatasetActionEmbedding, DatasetEmbedder, EmbeddingRouterQualification, TargetSelector,
};
use crate::error::{DopeError, Result, io_error};
use crate::model::Task;
use crate::production::{canonical_json, hashes, write_canonical};

#[derive(Clone, Debug)]
pub struct RegressionEmbeddingBatchOptions<'a> {
    pub dataset_root: &'a Path,
    pub output_dir: &'a Path,
    pub router_bundle: &'a Path,
    pub router_evidence: &'a Path,
    pub method: &'a str,
    pub expected_dimension: usize,
    pub shard_index: usize,
    pub shard_count: usize,
    pub jobs: usize,
    pub determinism_checks: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct RegressionEmbeddingRecord {
    pub dataset_id: String,
    pub global_index: usize,
    pub status: String,
    pub header_source: Option<String>,
    pub rows: Option<usize>,
    pub features: Option<usize>,
    pub json_output: Option<PathBuf>,
    pub vector_output: Option<PathBuf>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct RegressionEmbeddingBatchManifest {
    pub format: String,
    pub version: u8,
    pub method: String,
    pub dataset_root: PathBuf,
    pub output_dir: PathBuf,
    pub shard_index: usize,
    pub shard_count: usize,
    pub discovered: usize,
    pub dataset_ids_sha256: String,
    pub assigned: usize,
    pub success: usize,
    pub schema_mismatch: usize,
    pub failed: usize,
    pub determinism_checked: usize,
    pub router: EmbeddingRouterQualification,
    pub records: Vec<RegressionEmbeddingRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct RegressionEmbeddingVerification {
    pub format: String,
    pub version: u8,
    pub method: String,
    pub dimension: usize,
    pub datasets: usize,
    pub zero_feature_datasets: usize,
    pub schema_sha256: String,
    pub embedding_eligible: bool,
    pub promotion_qualified: bool,
    pub promotion_failed_gates: Vec<String>,
    pub router_bundle_sha256: String,
    pub router_evidence_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct RegressionCorpusInventory {
    pub format: String,
    pub version: u8,
    pub dataset_root: PathBuf,
    pub discovered: usize,
    pub output_headers: usize,
    pub feature_column_fallbacks: usize,
    pub schema_mismatches: usize,
    pub dataset_ids_sha256: String,
}

#[derive(Clone, Debug)]
struct DatasetInput {
    id: String,
    global_index: usize,
    root: PathBuf,
    metadata: Value,
}

fn valid_dataset_id(id: &str) -> bool {
    id.len() == 16 && id.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_method(method: &str) -> bool {
    !method.is_empty()
        && method
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn discover(dataset_root: &Path) -> Result<Vec<DatasetInput>> {
    let entries = fs::read_dir(dataset_root).map_err(|error| io_error(dataset_root, error))?;
    let mut roots = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| io_error(dataset_root, error))?;
        if !entry
            .file_type()
            .map_err(|error| io_error(entry.path(), error))?
            .is_dir()
        {
            continue;
        }
        let id = match entry.file_name().into_string() {
            Ok(id) if valid_dataset_id(&id) && !id.starts_with("synth_") => id,
            _ => continue,
        };
        let root = entry.path();
        let metadata_path = root.join("meta.json");
        let train_path = root.join("train.csv");
        if !metadata_path.is_file() || !train_path.is_file() {
            continue;
        }
        let metadata_source =
            fs::read(&metadata_path).map_err(|error| io_error(metadata_path.clone(), error))?;
        let metadata: Value = serde_json::from_slice(&metadata_source)?;
        if metadata.get("task_type").and_then(Value::as_str) != Some("regression") {
            continue;
        }
        roots.push((id, root, metadata));
    }
    roots.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(roots
        .into_iter()
        .enumerate()
        .map(|(global_index, (id, root, metadata))| DatasetInput {
            id,
            global_index,
            root,
            metadata,
        })
        .collect())
}

fn string_array(value: Option<&Value>) -> Option<Vec<String>> {
    value?
        .as_array()?
        .iter()
        .map(|value| value.as_str().map(str::to_string))
        .collect()
}

fn first_row_width(path: &Path) -> Result<usize> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(file);
    match reader.records().next() {
        Some(record) => Ok(record?.len()),
        None => Err(DopeError::Data("train.csv has no non-empty rows".into())),
    }
}

fn dataset_headers(dataset: &DatasetInput) -> Result<(String, Vec<String>, String)> {
    let target = dataset
        .metadata
        .get("target_column")
        .and_then(Value::as_str)
        .filter(|target| !target.is_empty())
        .ok_or_else(|| DopeError::Data("meta.json has no target_column".into()))?
        .to_string();
    let width = first_row_width(&dataset.root.join("train.csv"))?;
    if let Some(headers) = string_array(dataset.metadata.get("output_headers"))
        && headers.len() == width
    {
        return Ok((target, headers, "output_headers".into()));
    }
    if dataset
        .metadata
        .get("target_is_final_column")
        .and_then(Value::as_bool)
        == Some(true)
        && let Some(mut headers) = string_array(dataset.metadata.get("feature_columns"))
        && headers.len() + 1 >= width
    {
        let source = if headers.len() + 1 == width {
            "feature_columns+target"
        } else {
            "truncated_feature_columns+target"
        };
        headers.truncate(width - 1);
        headers.push(target.clone());
        return Ok((target, headers, source.into()));
    }
    Err(DopeError::Data(format!(
        "no metadata header has the train.csv width {width}"
    )))
}

pub fn inspect_regression_corpus(dataset_root: &Path) -> Result<RegressionCorpusInventory> {
    let discovered = discover(dataset_root)?;
    let mut output_headers = 0;
    let mut feature_column_fallbacks = 0;
    let mut schema_mismatches = 0;
    for dataset in &discovered {
        match dataset_headers(dataset).map(|(_, _, source)| source) {
            Ok(source) if source == "output_headers" => output_headers += 1,
            Ok(_) => feature_column_fallbacks += 1,
            Err(_) => schema_mismatches += 1,
        }
    }
    let dataset_ids = discovered
        .iter()
        .map(|dataset| dataset.id.as_str())
        .collect::<Vec<_>>();
    Ok(RegressionCorpusInventory {
        format: "dope-regression-corpus-inventory".into(),
        version: 1,
        dataset_root: dataset_root.into(),
        discovered: discovered.len(),
        output_headers,
        feature_column_fallbacks,
        schema_mismatches,
        dataset_ids_sha256: hashes(&canonical_json(&dataset_ids)?).sha256,
    })
}

fn prepare_headered_csv(
    dataset: &DatasetInput,
    temporary_dir: &Path,
) -> Result<(PathBuf, String, String)> {
    let (target, headers, header_source) = dataset_headers(dataset)?;
    let path = temporary_dir.join(format!("{}.csv", dataset.id));
    let mut output = File::create(&path).map_err(|error| io_error(&path, error))?;
    {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(&mut output);
        writer.write_record(headers)?;
        writer.flush().map_err(|error| io_error(&path, error))?;
    }
    let train_path = dataset.root.join("train.csv");
    let mut input = File::open(&train_path).map_err(|error| io_error(&train_path, error))?;
    io::copy(&mut input, &mut output).map_err(|error| io_error(&path, error))?;
    output.flush().map_err(|error| io_error(&path, error))?;
    Ok((path, target, header_source))
}

pub fn verify_embedding_vector(path: &Path, dimension: usize) -> Result<()> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(file);
    let records = reader
        .records()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if records.len() != 2 {
        return Err(DopeError::Data(format!(
            "embedding vector must have two CSV records, found {}",
            records.len()
        )));
    }
    for (index, record) in records.iter().enumerate() {
        if record.len() != dimension {
            return Err(DopeError::Data(format!(
                "embedding vector record {index} has {} fields, expected {dimension}",
                record.len()
            )));
        }
    }
    Ok(())
}

pub fn verify_regression_embeddings(
    output_dir: &Path,
    method: &str,
    qualification: &EmbeddingRouterQualification,
    expected_count: usize,
) -> Result<RegressionEmbeddingVerification> {
    if !valid_method(method) {
        return Err(DopeError::Data("invalid embedding method".into()));
    }
    let suffix = format!("_{method}_{}.vs", qualification.dimension);
    let json_suffix = format!("_{method}_{}.json", qualification.dimension);
    let mut vectors = Vec::new();
    let mut json_ids = BTreeSet::new();
    for entry in fs::read_dir(output_dir).map_err(|error| io_error(output_dir, error))? {
        let entry = entry.map_err(|error| io_error(output_dir, error))?;
        if !entry
            .file_type()
            .map_err(|error| io_error(entry.path(), error))?
            .is_file()
        {
            continue;
        }
        let filename = match entry.file_name().into_string() {
            Ok(filename) => filename,
            Err(_) => continue,
        };
        let Some(dataset_id) = filename.strip_suffix(&suffix) else {
            if let Some(dataset_id) = filename.strip_suffix(&json_suffix)
                && valid_dataset_id(dataset_id)
            {
                json_ids.insert(dataset_id.to_string());
            }
            continue;
        };
        if valid_dataset_id(dataset_id) {
            vectors.push((dataset_id.to_string(), entry.path()));
        }
    }
    vectors.sort_by(|left, right| left.0.cmp(&right.0));
    if vectors.len() != expected_count {
        return Err(DopeError::Data(format!(
            "found {} embedding vectors, expected {expected_count}",
            vectors.len()
        )));
    }
    let vector_ids = vectors
        .iter()
        .map(|(dataset_id, _)| dataset_id.clone())
        .collect::<BTreeSet<_>>();
    if json_ids != vector_ids {
        return Err(DopeError::Data(
            "embedding JSON and vector dataset IDs do not match".into(),
        ));
    }
    let mut schemas = BTreeSet::new();
    let mut zero_feature_datasets = 0;
    for (dataset_id, vector_path) in &vectors {
        verify_embedding_vector(vector_path, qualification.dimension)?;
        let json_path = output_dir.join(format!(
            "{dataset_id}_{method}_{}.json",
            qualification.dimension
        ));
        let source = fs::read(&json_path).map_err(|error| io_error(&json_path, error))?;
        let embedding: DatasetActionEmbedding = serde_json::from_slice(&source)?;
        if embedding.dimension != qualification.dimension
            || embedding.vector.len() != qualification.dimension
            || embedding.component_names.len() != qualification.dimension
            || !matches!(embedding.task, Task::Regression)
            || embedding.features != embedding.normalization.features.len()
            || embedding.router_bundle_sha256 != qualification.router_bundle_sha256
            || embedding.router_evidence_sha256 != qualification.router_evidence_sha256
        {
            return Err(DopeError::Data(format!(
                "embedding metadata does not match the qualified router for {dataset_id}"
            )));
        }
        let file = File::open(vector_path).map_err(|error| io_error(vector_path, error))?;
        let records = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(false)
            .from_reader(file)
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let names_match = records[0]
            .iter()
            .eq(embedding.component_names.iter().map(String::as_str));
        let values = records[1]
            .iter()
            .map(|value| {
                value.parse::<f32>().map_err(|_| {
                    DopeError::Data(format!(
                        "embedding vector contains a non-f32 value for {dataset_id}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if !names_match || values != embedding.vector {
            return Err(DopeError::Data(format!(
                "embedding JSON and vector contents differ for {dataset_id}"
            )));
        }
        zero_feature_datasets += usize::from(embedding.features == 0);
        schemas.insert(embedding.schema_sha256);
    }
    if schemas.len() != 1 {
        return Err(DopeError::Data(format!(
            "successful embeddings have {} distinct schemas",
            schemas.len()
        )));
    }
    Ok(RegressionEmbeddingVerification {
        format: "dope-regression-embedding-verification".into(),
        version: 2,
        method: method.into(),
        dimension: qualification.dimension,
        datasets: vectors.len(),
        zero_feature_datasets,
        schema_sha256: schemas.into_iter().next().unwrap_or_default(),
        embedding_eligible: qualification.embedding_eligible,
        promotion_qualified: qualification.promotion_qualified,
        promotion_failed_gates: qualification.promotion_failed_gates.clone(),
        router_bundle_sha256: qualification.router_bundle_sha256.clone(),
        router_evidence_sha256: qualification.router_evidence_sha256.clone(),
    })
}

fn process_dataset(
    dataset: &DatasetInput,
    embedder: &DatasetEmbedder,
    temporary_dir: &Path,
    output_dir: &Path,
    method: &str,
) -> RegressionEmbeddingRecord {
    let mut record = RegressionEmbeddingRecord {
        dataset_id: dataset.id.clone(),
        global_index: dataset.global_index,
        status: "failed".into(),
        header_source: None,
        rows: None,
        features: None,
        json_output: None,
        vector_output: None,
        error: None,
    };
    let (headered_csv, target, header_source) = match prepare_headered_csv(dataset, temporary_dir) {
        Ok(prepared) => prepared,
        Err(error) => {
            record.status = "schema_mismatch".into();
            record.error = Some(error.to_string());
            return record;
        }
    };
    record.header_source = Some(header_source);
    let dimension = embedder.qualification().dimension;
    let stem = format!("{}_{}_{}", dataset.id, method, dimension);
    let json_output = output_dir.join(format!("{stem}.json"));
    let vector_output = output_dir.join(format!("{stem}.vs"));
    let partial_suffix = format!("partial-{}", std::process::id());
    let json_partial = output_dir.join(format!(".{stem}.json.{partial_suffix}"));
    let vector_partial = output_dir.join(format!(".{stem}.vs.{partial_suffix}"));
    let result = (|| -> Result<()> {
        let embedding = embedder.embed_csv(
            &headered_csv,
            TargetSelector::Name(&target),
            Task::Regression,
        )?;
        if embedding.dimension != dimension
            || embedding.component_names.len() != dimension
            || embedding.vector.len() != dimension
        {
            return Err(DopeError::Data(
                "embedding payload does not match its qualified dimension".into(),
            ));
        }
        record.rows = Some(embedding.rows);
        record.features = Some(embedding.features);
        write_canonical(&json_partial, &embedding)?;
        crate::embedding::write_embedding_csv(&vector_partial, &embedding)?;
        verify_embedding_vector(&vector_partial, dimension)?;
        fs::rename(&json_partial, &json_output).map_err(|error| io_error(&json_output, error))?;
        fs::rename(&vector_partial, &vector_output)
            .map_err(|error| io_error(&vector_output, error))?;
        Ok(())
    })();
    let _ = fs::remove_file(&headered_csv);
    let _ = fs::remove_file(&json_partial);
    let _ = fs::remove_file(&vector_partial);
    match result {
        Ok(()) => {
            record.status = "success".into();
            record.json_output = Some(json_output);
            record.vector_output = Some(vector_output);
        }
        Err(error) => record.error = Some(error.to_string()),
    }
    record
}

fn verify_determinism(
    dataset: &DatasetInput,
    record: &RegressionEmbeddingRecord,
    embedder: &DatasetEmbedder,
    temporary_dir: &Path,
) -> Result<()> {
    let (headered_csv, target, _) = prepare_headered_csv(dataset, temporary_dir)?;
    let vector_repeat = temporary_dir.join(format!("{}.repeat.vs", dataset.id));
    let result = (|| -> Result<()> {
        let repeated = embedder.embed_csv(
            &headered_csv,
            TargetSelector::Name(&target),
            Task::Regression,
        )?;
        let json_output = record
            .json_output
            .as_deref()
            .ok_or_else(|| DopeError::Data("successful record has no JSON output".into()))?;
        if fs::read(json_output).map_err(|error| io_error(json_output, error))?
            != canonical_json(&repeated)?
        {
            return Err(DopeError::Data(
                "repeated JSON embedding is not byte-stable".into(),
            ));
        }
        crate::embedding::write_embedding_csv(&vector_repeat, &repeated)?;
        let vector_output = record
            .vector_output
            .as_deref()
            .ok_or_else(|| DopeError::Data("successful record has no vector output".into()))?;
        if fs::read(vector_output).map_err(|error| io_error(vector_output, error))?
            != fs::read(&vector_repeat).map_err(|error| io_error(&vector_repeat, error))?
        {
            return Err(DopeError::Data(
                "repeated vector embedding is not byte-stable".into(),
            ));
        }
        Ok(())
    })();
    let _ = fs::remove_file(headered_csv);
    let _ = fs::remove_file(vector_repeat);
    result
}

pub fn embed_regression_corpus(
    options: &RegressionEmbeddingBatchOptions<'_>,
) -> Result<RegressionEmbeddingBatchManifest> {
    if options.shard_count == 0 || options.shard_index >= options.shard_count {
        return Err(DopeError::Data("invalid shard index or count".into()));
    }
    if options.jobs == 0 {
        return Err(DopeError::Data(
            "batch worker count must be positive".into(),
        ));
    }
    if !valid_method(options.method) {
        return Err(DopeError::Data(
            "method must contain only lowercase ASCII letters, digits, and hyphens".into(),
        ));
    }

    // Qualification deliberately precedes corpus discovery and all output writes.
    let embedder = DatasetEmbedder::load(options.router_bundle, options.router_evidence)?;
    let router = embedder.qualification().clone();
    if router.dimension != options.expected_dimension {
        return Err(DopeError::Data(format!(
            "qualified router dimension is {}, expected {}",
            router.dimension, options.expected_dimension
        )));
    }
    let discovered = discover(options.dataset_root)?;
    let assigned = discovered
        .iter()
        .filter(|dataset| dataset.global_index % options.shard_count == options.shard_index)
        .cloned()
        .collect::<Vec<_>>();
    let dataset_ids_sha256 = hashes(&canonical_json(
        &discovered
            .iter()
            .map(|dataset| dataset.id.as_str())
            .collect::<Vec<_>>(),
    )?)
    .sha256;
    fs::create_dir_all(options.output_dir).map_err(|error| io_error(options.output_dir, error))?;
    let temporary_dir = options
        .output_dir
        .join(format!(".tmp-shard-{}", options.shard_index));
    fs::create_dir_all(&temporary_dir).map_err(|error| io_error(&temporary_dir, error))?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.jobs)
        .build()
        .map_err(|error| DopeError::Data(format!("cannot build batch worker pool: {error}")))?;
    let mut records = pool.install(|| {
        assigned
            .par_iter()
            .map(|dataset| {
                process_dataset(
                    dataset,
                    &embedder,
                    &temporary_dir,
                    options.output_dir,
                    options.method,
                )
            })
            .collect::<Vec<_>>()
    });
    records.sort_by_key(|record| record.global_index);

    let successful_ids = records
        .iter()
        .filter(|record| record.status == "success")
        .take(options.determinism_checks)
        .map(|record| record.dataset_id.clone())
        .collect::<Vec<_>>();
    let mut determinism_checked = 0;
    for id in successful_ids {
        let dataset = assigned
            .iter()
            .find(|dataset| dataset.id == id)
            .expect("successful dataset remains assigned");
        let record = records
            .iter_mut()
            .find(|record| record.dataset_id == id)
            .expect("successful record remains present");
        match verify_determinism(dataset, record, &embedder, &temporary_dir) {
            Ok(()) => determinism_checked += 1,
            Err(error) => {
                record.status = "failed".into();
                record.error = Some(error.to_string());
            }
        }
    }
    let _ = fs::remove_dir(&temporary_dir);

    let success = records
        .iter()
        .filter(|record| record.status == "success")
        .count();
    let schema_mismatch = records
        .iter()
        .filter(|record| record.status == "schema_mismatch")
        .count();
    let failed = records.len() - success - schema_mismatch;
    Ok(RegressionEmbeddingBatchManifest {
        format: "dope-regression-embedding-batch".into(),
        version: 2,
        method: options.method.into(),
        dataset_root: options.dataset_root.into(),
        output_dir: options.output_dir.into(),
        shard_index: options.shard_index,
        shard_count: options.shard_count,
        discovered: discovered.len(),
        dataset_ids_sha256,
        assigned: assigned.len(),
        success,
        schema_mismatch,
        failed,
        determinism_checked,
        router,
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_and_method_names_are_restricted() {
        assert!(valid_dataset_id("0123456789abcdef"));
        assert!(valid_dataset_id("0123456789ABCDEF"));
        assert!(!valid_dataset_id("synth_0123456789"));
        assert!(valid_method("attention-small"));
        assert!(!valid_method("attention/small"));
    }
}
