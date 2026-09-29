use std::collections::BTreeSet;
use std::fs::{self, File};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::campaign::RouterEvidence;
use crate::data::Table;
use crate::error::{DopeError, Result, io_error};
use crate::model::Task;
use crate::production::{canonical_json, hashes};
use crate::router::{
    DATASET_SKETCH_VERSION, DatasetSketch, QuantizedRouter, ROUTER_ACTION_NAMES, RouterBundle,
    RouterPrediction,
};

pub const ACTION_EMBEDDING_VERSION: u8 = 1;
pub const ACTION_EMBEDDING_DIMENSION: usize = 4_168;
const ACTION_EMBEDDING_CANDIDATES: usize = 24;
const ACTION_EMBEDDING_SKETCH_WIDTH: usize = 856;
const ACTION_EMBEDDING_HIDDEN_WIDTH: usize = 128;

#[derive(Clone, Copy, Debug)]
pub enum TargetSelector<'a> {
    Name(&'a str),
    Index(usize),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NormalizationExtrema {
    pub column: String,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetNormalization {
    pub features: Vec<NormalizationExtrema>,
    pub target: NormalizationExtrema,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MetadataMode {
    #[default]
    Public,
    RestrictedResearch,
}

impl DatasetNormalization {
    fn public_positions(features: usize) -> Self {
        Self {
            features: (0..features)
                .map(|index| NormalizationExtrema {
                    column: format!("feature_{index}"),
                    minimum: None,
                    maximum: None,
                })
                .collect(),
            target: NormalizationExtrema {
                column: "target".into(),
                minimum: None,
                maximum: None,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoadedNumericDataset {
    pub table: Table,
    pub normalization: DatasetNormalization,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ComponentRange {
    pub offset: usize,
    pub length: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CandidateComponentOffsets {
    pub candidate_id: String,
    pub hidden_state: ComponentRange,
    pub standardized_actions: ComponentRange,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct EmbeddingComponentOffsets {
    pub standardized_sketch: ComponentRange,
    pub candidates: Vec<CandidateComponentOffsets>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetActionEmbedding {
    pub format: String,
    pub version: u8,
    pub dimension: usize,
    pub schema_sha256: String,
    pub router_bundle_sha256: String,
    pub router_evidence_sha256: String,
    pub task: Task,
    pub rows: usize,
    pub features: usize,
    pub normalization: DatasetNormalization,
    #[serde(default)]
    pub metadata_mode: MetadataMode,
    pub candidate_order: Vec<String>,
    pub components: EmbeddingComponentOffsets,
    pub component_names: Vec<String>,
    pub vector: Vec<f32>,
    pub predictions: Vec<RouterPrediction>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct EmbeddingRouterQualification {
    pub format: String,
    pub version: u8,
    pub embedding_eligible: bool,
    pub promotion_qualified: bool,
    pub promotion_failed_gates: Vec<String>,
    pub candidates: usize,
    pub sketch_width: usize,
    pub hidden_width: usize,
    pub action_width: usize,
    pub dimension: usize,
    pub router_bundle_sha256: String,
    pub router_evidence_sha256: String,
}

pub struct DatasetEmbedder {
    router: QuantizedRouter,
    qualification: EmbeddingRouterQualification,
}

fn parse_field(field: &str) -> Result<Option<f64>> {
    let trimmed = field.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("nan") {
        return Ok(None);
    }
    let value = trimmed
        .parse::<f64>()
        .map_err(|_| DopeError::Data("invalid numeric CSV field".into()))?;
    if !value.is_finite() {
        return Err(DopeError::Data(
            "infinite CSV values are unsupported".into(),
        ));
    }
    Ok(Some(value))
}

fn extrema(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    values.fold(None, |result, value| {
        Some(match result {
            Some((minimum, maximum)) => (minimum.min(value), maximum.max(value)),
            None => (value, value),
        })
    })
}

fn normalize(value: f64, minimum: f64, maximum: f64) -> f32 {
    if minimum == maximum {
        0.0
    } else {
        let difference = maximum - minimum;
        let normalized = if difference.is_finite() {
            (value - minimum) / difference
        } else {
            (value / 2.0 - minimum / 2.0) / (maximum / 2.0 - minimum / 2.0)
        };
        normalized.clamp(0.0, 1.0) as f32
    }
}

/// Loads one supervised, header-bearing numeric CSV and applies training-table-only
/// min-max normalization. Missing feature values remain NaN in the returned table.
pub fn load_numeric_csv(
    path: &Path,
    selector: TargetSelector<'_>,
    task: Task,
) -> Result<LoadedNumericDataset> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(file);
    let headers = reader.headers()?.clone();
    if headers.is_empty() || (task != Task::Regression && headers.len() < 2) {
        return Err(DopeError::Data(
            "CSV needs a header and one target; non-regression tasks also need a feature".into(),
        ));
    }
    let mut distinct_headers = BTreeSet::new();
    if headers
        .iter()
        .any(|header| !distinct_headers.insert(header))
    {
        return Err(DopeError::Data(
            "duplicate CSV headers are unsupported".into(),
        ));
    }
    let target_index = match selector {
        TargetSelector::Name(name) => headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| DopeError::Data("target column is absent".into()))?,
        TargetSelector::Index(index) if index < headers.len() => index,
        TargetSelector::Index(index) => {
            return Err(DopeError::Data(format!(
                "target index {index} is outside the CSV header"
            )));
        }
    };
    let feature_indices = (0..headers.len())
        .filter(|&index| index != target_index)
        .collect::<Vec<_>>();
    let mut raw_features = vec![Vec::<Option<f64>>::new(); feature_indices.len()];
    let mut raw_target = Vec::<f64>::new();
    for record in reader.records() {
        let record = record?;
        if record.len() != headers.len() {
            return Err(DopeError::Data("ragged CSV rows are unsupported".into()));
        }
        for (feature, &column) in feature_indices.iter().enumerate() {
            raw_features[feature].push(parse_field(&record[column])?);
        }
        let target = parse_field(&record[target_index])?
            .ok_or_else(|| DopeError::Data("targets must not be blank or NaN".into()))?;
        raw_target.push(target);
    }
    if raw_target.is_empty() {
        return Err(DopeError::Data("CSV dataset is empty".into()));
    }

    let mut normalized_features = Vec::with_capacity(raw_features.len());
    let mut feature_extrema = Vec::with_capacity(raw_features.len());
    for (&column, values) in feature_indices.iter().zip(raw_features) {
        let range = extrema(values.iter().flatten().copied());
        normalized_features.push(
            values
                .into_iter()
                .map(|value| match (value, range) {
                    (Some(value), Some((minimum, maximum))) => normalize(value, minimum, maximum),
                    _ => f32::NAN,
                })
                .collect(),
        );
        feature_extrema.push(NormalizationExtrema {
            column: headers[column].to_string(),
            minimum: range.map(|range| range.0),
            maximum: range.map(|range| range.1),
        });
    }
    let target_range = extrema(raw_target.iter().copied()).expect("nonempty target");
    let normalized_target: Vec<f32> = match task {
        Task::Regression => raw_target
            .iter()
            .map(|&value| normalize(value, target_range.0, target_range.1))
            .collect(),
        Task::Binary => {
            let mut labels = raw_target.clone();
            labels.sort_by(f64::total_cmp);
            labels.dedup_by(|left, right| *left == *right);
            if labels.len() != 2 {
                return Err(DopeError::Data(format!(
                    "binary targets require exactly two distinct labels, found {}",
                    labels.len()
                )));
            }
            raw_target
                .iter()
                .map(|value| if *value == labels[0] { 0.0 } else { 1.0 })
                .collect()
        }
    };
    let rows = normalized_target.len();
    Ok(LoadedNumericDataset {
        table: Table {
            rows,
            features: normalized_features.len(),
            columns: normalized_features,
            target: normalized_target,
        },
        normalization: DatasetNormalization {
            features: feature_extrema,
            target: NormalizationExtrema {
                column: headers[target_index].to_string(),
                minimum: Some(target_range.0),
                maximum: Some(target_range.1),
            },
        },
    })
}

fn validate_evidence_integrity(evidence: &RouterEvidence) -> Result<()> {
    if evidence.format != "dope-router-evidence" || evidence.version != 1 {
        return Err(DopeError::Data("invalid router evidence format".into()));
    }
    let numeric_evidence = [
        evidence.top_four_oracle_recall,
        evidence.maximum_profile_regret_upper,
        evidence.paired_hypervolume_improvement,
        evidence.paired_hypervolume_ci_lower,
        evidence.student_max_abs_difference,
        evidence.top_choice_agreement,
        evidence.inference_p95_ms,
        evidence.sketch_p95_ms,
        evidence.student_mean_regret,
        evidence.random_mean_regret,
        evidence.best_fixed_mean_regret,
        evidence.ga2m_mean_regret,
    ];
    if !numeric_evidence.into_iter().all(f64::is_finite) {
        return Err(DopeError::Data(
            "router evidence contains non-finite measurements".into(),
        ));
    }
    Ok(())
}

fn load_qualified_router(
    router_bundle_path: &Path,
    router_evidence_path: &Path,
) -> Result<(RouterBundle, EmbeddingRouterQualification)> {
    let bundle_source =
        fs::read(router_bundle_path).map_err(|error| io_error(router_bundle_path, error))?;
    let bundle: RouterBundle = serde_json::from_slice(&bundle_source)?;
    bundle.validate()?;
    if !bundle.trained {
        return Err(DopeError::Data(
            "untrained router cannot produce an action embedding".into(),
        ));
    }
    let evidence_source =
        fs::read(router_evidence_path).map_err(|error| io_error(router_evidence_path, error))?;
    let evidence: RouterEvidence = serde_json::from_slice(&evidence_source)?;
    validate_evidence_integrity(&evidence)?;
    if bundle.training_evidence_sha256.as_deref()
        != Some(evidence.training_evidence_sha256.as_str())
    {
        return Err(DopeError::Data(
            "router bundle and training evidence hashes differ".into(),
        ));
    }
    let canonical_bundle_bytes = serde_json::to_vec(&bundle)?.len();
    if evidence.bundle_bytes != canonical_bundle_bytes as u64 {
        return Err(DopeError::Data(format!(
            "router evidence records {} bundle bytes but canonical bundle has {}",
            evidence.bundle_bytes, canonical_bundle_bytes
        )));
    }
    let candidates = bundle.candidate_ids.len();
    let sketch_width = bundle.sketch_mean.len();
    let hidden_width = bundle.layers[bundle.layers.len() - 2].output;
    let dimension = candidates
        .checked_mul(hidden_width + ROUTER_ACTION_NAMES.len())
        .and_then(|candidate_width| sketch_width.checked_add(candidate_width))
        .ok_or_else(|| DopeError::Data("router embedding dimension overflowed".into()))?;
    if candidates != ACTION_EMBEDDING_CANDIDATES
        || sketch_width != ACTION_EMBEDDING_SKETCH_WIDTH
        || hidden_width != ACTION_EMBEDDING_HIDDEN_WIDTH
        || dimension != ACTION_EMBEDDING_DIMENSION
    {
        return Err(DopeError::Data(format!(
            "router embedding architecture derives {dimension} dimensions, expected {ACTION_EMBEDDING_DIMENSION} from {ACTION_EMBEDDING_SKETCH_WIDTH} + {ACTION_EMBEDDING_CANDIDATES} x ({ACTION_EMBEDDING_HIDDEN_WIDTH} + {})",
            ROUTER_ACTION_NAMES.len()
        )));
    }
    let promotion_failed_gates = evidence.failed_gates();
    let qualification = EmbeddingRouterQualification {
        format: "dope-embedding-router-qualification".into(),
        version: 2,
        embedding_eligible: true,
        promotion_qualified: promotion_failed_gates.is_empty(),
        promotion_failed_gates,
        candidates,
        sketch_width,
        hidden_width,
        action_width: ROUTER_ACTION_NAMES.len(),
        dimension,
        router_bundle_sha256: hashes(&bundle_source).sha256,
        router_evidence_sha256: hashes(&evidence_source).sha256,
    };
    Ok((bundle, qualification))
}

pub fn qualify_embedding_router(
    router_bundle_path: &Path,
    router_evidence_path: &Path,
) -> Result<EmbeddingRouterQualification> {
    load_qualified_router(router_bundle_path, router_evidence_path)
        .map(|(_, qualification)| qualification)
}