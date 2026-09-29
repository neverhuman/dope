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
        .map_err(|_| DopeError::Data(format!("invalid numeric CSV field {field:?}")))?;
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
            .ok_or_else(|| DopeError::Data(format!("target column {name:?} is absent")))?,
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

impl DatasetEmbedder {
    pub fn load(router_bundle_path: &Path, router_evidence_path: &Path) -> Result<Self> {
        let (bundle, qualification) =
            load_qualified_router(router_bundle_path, router_evidence_path)?;
        Ok(Self {
            router: QuantizedRouter::new(bundle)?,
            qualification,
        })
    }

    pub fn qualification(&self) -> &EmbeddingRouterQualification {
        &self.qualification
    }

    pub fn embed_csv(
        &self,
        csv_path: &Path,
        selector: TargetSelector<'_>,
        task: Task,
    ) -> Result<DatasetActionEmbedding> {
        let loaded = load_numeric_csv(csv_path, selector, task)?;
        let sketch = DatasetSketch::from_train(&loaded.table, task);
        if sketch.version != DATASET_SKETCH_VERSION {
            return Err(DopeError::Data("unsupported dataset sketch version".into()));
        }
        let router_embedding = self.router.action_embedding(&sketch)?;

        let mut vector = router_embedding.standardized_sketch;
        let mut component_names = (0..vector.len())
            .map(|index| format!("sketch.standardized.{index:03}"))
            .collect::<Vec<_>>();
        let sketch_range = ComponentRange {
            offset: 0,
            length: vector.len(),
        };
        let mut candidate_order = Vec::with_capacity(router_embedding.candidates.len());
        let mut candidate_offsets = Vec::with_capacity(router_embedding.candidates.len());
        let mut predictions = Vec::with_capacity(router_embedding.candidates.len());
        for candidate in router_embedding.candidates {
            candidate_order.push(candidate.candidate_id.clone());
            let hidden_state = ComponentRange {
                offset: vector.len(),
                length: candidate.hidden_state.len(),
            };
            component_names.extend((0..candidate.hidden_state.len()).map(|index| {
                format!(
                    "candidate.{}.hidden_state.{index:03}",
                    candidate.candidate_id
                )
            }));
            vector.extend(candidate.hidden_state);
            let standardized_actions = ComponentRange {
                offset: vector.len(),
                length: candidate.standardized_actions.len(),
            };
            component_names.extend(
                ROUTER_ACTION_NAMES
                    .iter()
                    .map(|name| format!("candidate.{}.action.{name}", candidate.candidate_id)),
            );
            vector.extend(candidate.standardized_actions);
            candidate_offsets.push(CandidateComponentOffsets {
                candidate_id: candidate.candidate_id,
                hidden_state,
                standardized_actions,
            });
            predictions.push(candidate.prediction);
        }
        let dimension = vector.len();
        if dimension != self.qualification.dimension {
            return Err(DopeError::Data(format!(
                "router produced dimension {dimension}, expected {}",
                self.qualification.dimension
            )));
        }
        let schema_sha256 = hashes(&canonical_json(&serde_json::json!({
            "format": "dope-dataset-action-embedding-schema",
            "version": ACTION_EMBEDDING_VERSION,
            "component_names": component_names,
        }))?)
        .sha256;
        Ok(DatasetActionEmbedding {
            format: "dope-dataset-action-embedding".into(),
            version: ACTION_EMBEDDING_VERSION,
            dimension,
            schema_sha256,
            router_bundle_sha256: self.qualification.router_bundle_sha256.clone(),
            router_evidence_sha256: self.qualification.router_evidence_sha256.clone(),
            task,
            rows: loaded.table.rows,
            features: loaded.table.features,
            normalization: loaded.normalization,
            candidate_order,
            components: EmbeddingComponentOffsets {
                standardized_sketch: sketch_range,
                candidates: candidate_offsets,
            },
            component_names,
            vector,
            predictions,
        })
    }
}

pub fn embed_dataset(
    csv_path: &Path,
    selector: TargetSelector<'_>,
    task: Task,
    router_bundle_path: &Path,
    router_evidence_path: &Path,
) -> Result<DatasetActionEmbedding> {
    DatasetEmbedder::load(router_bundle_path, router_evidence_path)?
        .embed_csv(csv_path, selector, task)
}

pub fn write_embedding_csv(path: &Path, embedding: &DatasetActionEmbedding) -> Result<()> {
    let file = File::create(path).map_err(|error| io_error(path, error))?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);
    writer.write_record(&embedding.component_names)?;
    let values = embedding
        .vector
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    writer.write_record(values)?;
    writer.flush().map_err(|error| io_error(path, error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::production::write_canonical;
    use crate::router::{QuantizedLayer, ROUTER_OUTPUTS};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "dope-embedding-{name}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_bundle(hidden: usize) -> RouterBundle {
        let sketch = 856;
        let embedding = 1;
        let first_hidden = 3;
        let candidate_ids = (0..ACTION_EMBEDDING_CANDIDATES)
            .map(|index| match index {
                0 => "independent_quantile".into(),
                1 => "chow_liu".into(),
                _ => format!("candidate-{index:02}"),
            })
            .collect::<Vec<_>>();
        RouterBundle {
            format: "dope-distilled-router".into(),
            version: 1,
            sketch_mean: vec![0.0; sketch],
            sketch_std: vec![1.0; sketch],
            input_scale: 0.01,
            candidate_embeddings: candidate_ids
                .iter()
                .enumerate()
                .map(|(index, _)| vec![index as f32 / ACTION_EMBEDDING_CANDIDATES as f32])
                .collect(),
            candidate_ids,
            layers: vec![
                QuantizedLayer {
                    input: sketch + embedding,
                    output: first_hidden,
                    weights: vec![1; (sketch + embedding) * first_hidden],
                    biases: vec![0; first_hidden],
                    multiplier: 0.01,
                },
                QuantizedLayer {
                    input: first_hidden,
                    output: hidden,
                    weights: vec![1; first_hidden * hidden],
                    biases: vec![1; hidden],
                    multiplier: 0.5,
                },
                QuantizedLayer {
                    input: hidden,
                    output: ROUTER_OUTPUTS,
                    weights: vec![1; hidden * ROUTER_OUTPUTS],
                    biases: vec![0; ROUTER_OUTPUTS],
                    multiplier: 0.5,
                },
            ],
            output_scale: vec![0.1; ROUTER_OUTPUTS],
            output_bias: vec![0.2; ROUTER_OUTPUTS],
            ga2m_router: None,
            trained: true,
            training_evidence_sha256: Some("a".repeat(64)),
        }
    }

    fn passing_evidence(bundle: &RouterBundle) -> RouterEvidence {
        RouterEvidence {
            format: "dope-router-evidence".into(),
            version: 1,
            top_four_oracle_recall: 1.0,
            maximum_profile_regret_upper: 0.01,
            beats_random: true,
            beats_best_fixed: true,
            beats_ga2m: true,
            paired_hypervolume_improvement: 0.1,
            paired_hypervolume_ci_lower: 0.01,
            student_max_abs_difference: 0.0001,
            top_choice_agreement: 1.0,
            bundle_bytes: serde_json::to_vec(bundle).unwrap().len() as u64,
            inference_p95_ms: 1.0,
            sketch_p95_ms: 10.0,
            training_evidence_sha256: "a".repeat(64),
            best_fixed_candidate_id: "independent_quantile".into(),
            student_mean_regret: 0.01,
            random_mean_regret: 0.02,
            best_fixed_mean_regret: 0.02,
            ga2m_mean_regret: 0.02,
        }
    }

    fn write_router(
        root: &Path,
        hidden: usize,
    ) -> (std::path::PathBuf, std::path::PathBuf, RouterBundle) {
        fs::create_dir_all(root).unwrap();
        let bundle = test_bundle(hidden);
        let evidence = passing_evidence(&bundle);
        let bundle_path = root.join("router.bundle.json");
        let evidence_path = root.join("router-evidence.json");
        write_canonical(&bundle_path, &bundle).unwrap();
        write_canonical(&evidence_path, &evidence).unwrap();
        (bundle_path, evidence_path, bundle)
    }

    #[test]
    fn loader_normalizes_features_targets_and_binary_labels() {
        let root = temp_dir("loader");
        let csv = root.join("train.csv");
        fs::write(
            &csv,
            "constant,missing,value,outcome\n7,,1,20\n7,NaN,3,10\n7,,5,20\n",
        )
        .unwrap();
        let binary = load_numeric_csv(&csv, TargetSelector::Name("outcome"), Task::Binary).unwrap();
        assert_eq!(binary.table.target, vec![1.0, 0.0, 1.0]);
        assert_eq!(binary.table.columns[0], vec![0.0; 3]);
        assert!(binary.table.columns[1].iter().all(|value| value.is_nan()));
        assert_eq!(binary.table.columns[2], vec![0.0, 0.5, 1.0]);
        assert_eq!(binary.normalization.features[1].minimum, None);

        let regression =
            load_numeric_csv(&csv, TargetSelector::Index(3), Task::Regression).unwrap();
        assert_eq!(regression.table.target, vec![1.0, 0.0, 1.0]);
        assert_eq!(regression.normalization.target.minimum, Some(10.0));
        assert_eq!(regression.normalization.target.maximum, Some(20.0));

        let extreme = root.join("extreme.csv");
        fs::write(&extreme, "value,outcome\n0,-1e308\n1,0\n2,1e308\n").unwrap();
        let extreme =
            load_numeric_csv(&extreme, TargetSelector::Index(1), Task::Regression).unwrap();
        assert_eq!(extreme.table.target, vec![0.0, 0.5, 1.0]);

        let target_only = root.join("target-only.csv");
        fs::write(&target_only, "outcome\n10\n20\n15\n").unwrap();
        let target_only =
            load_numeric_csv(&target_only, TargetSelector::Index(0), Task::Regression).unwrap();
        assert_eq!(target_only.table.rows, 3);
        assert_eq!(target_only.table.features, 0);
        assert!(target_only.table.columns.is_empty());
        assert!(target_only.normalization.features.is_empty());
        assert_eq!(target_only.table.target, vec![0.0, 1.0, 0.5]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn embedding_is_invariant_to_rows_columns_and_affine_rescaling() {
        let root = temp_dir("invariance");
        let base = root.join("base.csv");
        let transformed = root.join("transformed.csv");
        fs::write(&base, "a,b,y\n0,10,-1\n1,20,2\n2,NaN,-1\n3,40,2\n").unwrap();
        fs::write(&transformed, "b,y,a\n116,9,11\nNaN,4,9\n56,9,7\n26,4,5\n").unwrap();
        let (bundle, evidence, _) = write_router(&root.join("router"), 128);
        let original = embed_dataset(
            &base,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle,
            &evidence,
        )
        .unwrap();
        let changed = embed_dataset(
            &transformed,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle,
            &evidence,
        )
        .unwrap();
        assert_eq!(original.vector, changed.vector);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_csv_inputs_fail_closed() {
        let root = temp_dir("malformed");
        let cases = [
            ("empty", ""),
            ("header-only", "a,y\n"),
            ("duplicate", "a,a,y\n1,2,0\n"),
            ("ragged", "a,b,y\n1,2,0\n1,0\n"),
            ("invalid", "a,y\nnope,0\n"),
            ("infinite", "a,y\ninf,0\n"),
            ("missing-target", "a,y\n1,NaN\n"),
            ("one-binary-label", "a,y\n1,4\n2,4\n"),
            ("three-binary-labels", "a,y\n1,4\n2,5\n3,6\n"),
        ];
        for (name, contents) in cases {
            let path = root.join(format!("{name}.csv"));
            fs::write(&path, contents).unwrap();
            assert!(
                load_numeric_csv(&path, TargetSelector::Name("y"), Task::Binary).is_err(),
                "{name} unexpectedly loaded"
            );
        }
        let valid = root.join("valid.csv");
        fs::write(&valid, "a,y\n1,0\n2,1\n").unwrap();
        assert!(load_numeric_csv(&valid, TargetSelector::Name("absent"), Task::Binary).is_err());
        assert!(load_numeric_csv(&valid, TargetSelector::Index(2), Task::Binary).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn qualified_embedding_has_expected_offsets_and_prediction_parity() {
        let root = temp_dir("shape");
        let csv = root.join("train.csv");
        fs::write(&csv, "a,b,y\n1,4,10\n2,8,20\n3,12,15\n").unwrap();
        let hidden = 128;
        let (bundle_path, evidence_path, bundle) =
            write_router(&root.join(hidden.to_string()), hidden);
        let embedding = embed_dataset(
            &csv,
            TargetSelector::Index(2),
            Task::Regression,
            &bundle_path,
            &evidence_path,
        )
        .unwrap();
        assert_eq!(
            embedding.dimension,
            856 + ACTION_EMBEDDING_CANDIDATES * (hidden + ROUTER_OUTPUTS)
        );
        assert_eq!(embedding.component_names.len(), embedding.dimension);
        assert_eq!(embedding.components.standardized_sketch.offset, 0);
        assert_eq!(embedding.components.standardized_sketch.length, 856);
        assert_eq!(embedding.components.candidates[0].hidden_state.offset, 856);
        assert_eq!(
            embedding.components.candidates[0].hidden_state.length,
            hidden
        );

        let loaded = load_numeric_csv(&csv, TargetSelector::Index(2), Task::Regression).unwrap();
        let sketch = DatasetSketch::from_train(&loaded.table, Task::Regression);
        let predictions = QuantizedRouter::new(bundle)
            .unwrap()
            .predict(&sketch)
            .unwrap();
        assert_eq!(embedding.predictions, predictions);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn output_is_bit_reproducible_and_csv_matches_json_vector() {
        let root = temp_dir("output");
        let csv = root.join("train.csv");
        fs::write(&csv, "a,y\n1,0\n2,1\n").unwrap();
        let (bundle, evidence, _) = write_router(&root.join("router"), 128);
        let first = embed_dataset(
            &csv,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle,
            &evidence,
        )
        .unwrap();
        let second = embed_dataset(
            &csv,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle,
            &evidence,
        )
        .unwrap();
        assert_eq!(
            canonical_json(&first).unwrap(),
            canonical_json(&second).unwrap()
        );
        let vector_path = root.join("embedding.csv");
        write_embedding_csv(&vector_path, &first).unwrap();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(vector_path)
            .unwrap();
        assert_eq!(
            reader.headers().unwrap().iter().collect::<Vec<_>>(),
            first
                .component_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        let row = reader.records().next().unwrap().unwrap();
        let parsed = row
            .iter()
            .map(|value| value.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(parsed, first.vector);
        assert!(reader.records().next().is_none());
        let json_vector = serde_json::to_string(&first.vector).unwrap();
        let csv_vector = fs::read_to_string(root.join("embedding.csv"))
            .unwrap()
            .lines()
            .nth(1)
            .unwrap()
            .to_string();
        assert_eq!(csv_vector, json_vector[1..json_vector.len() - 1]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn promotion_failures_are_admitted_and_integrity_failures_are_rejected() {
        let root = temp_dir("qualification");
        let csv = root.join("train.csv");
        fs::write(&csv, "a,y\n1,0\n2,1\n").unwrap();
        let bundle_path = root.join("bundle.json");
        let evidence_path = root.join("evidence.json");

        let bundle = test_bundle(128);
        write_canonical(&bundle_path, &bundle).unwrap();
        let mut evidence = passing_evidence(&bundle);
        evidence.beats_random = false;
        write_canonical(&evidence_path, &evidence).unwrap();
        let qualification = qualify_embedding_router(&bundle_path, &evidence_path).unwrap();
        assert!(qualification.embedding_eligible);
        assert!(!qualification.promotion_qualified);
        assert_eq!(qualification.promotion_failed_gates, vec!["beats_random"]);
        assert_eq!(qualification.version, 2);
        assert_eq!(qualification.dimension, ACTION_EMBEDDING_DIMENSION);
        embed_dataset(
            &csv,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle_path,
            &evidence_path,
        )
        .unwrap();

        let mut evidence = passing_evidence(&bundle);
        evidence.bundle_bytes += 1;
        write_canonical(&evidence_path, &evidence).unwrap();
        assert!(
            embed_dataset(
                &csv,
                TargetSelector::Name("y"),
                Task::Binary,
                &bundle_path,
                &evidence_path,
            )
            .is_err()
        );

        let mut evidence = passing_evidence(&bundle);
        evidence.training_evidence_sha256 = "b".repeat(64);
        write_canonical(&evidence_path, &evidence).unwrap();
        assert!(
            embed_dataset(
                &csv,
                TargetSelector::Name("y"),
                Task::Binary,
                &bundle_path,
                &evidence_path,
            )
            .is_err()
        );

        let wrong_dimension = test_bundle(127);
        write_canonical(&bundle_path, &wrong_dimension).unwrap();
        write_canonical(&evidence_path, &passing_evidence(&wrong_dimension)).unwrap();
        assert!(
            qualify_embedding_router(&bundle_path, &evidence_path)
                .unwrap_err()
                .to_string()
                .contains("expected 4168")
        );

        let mut corrupted = bundle.clone();
        corrupted.layers[0].weights.pop();
        write_canonical(&bundle_path, &corrupted).unwrap();
        write_canonical(&evidence_path, &passing_evidence(&corrupted)).unwrap();
        assert!(
            embed_dataset(
                &csv,
                TargetSelector::Name("y"),
                Task::Binary,
                &bundle_path,
                &evidence_path,
            )
            .is_err()
        );

        let mut untrained = bundle;
        untrained.trained = false;
        untrained.training_evidence_sha256 = None;
        write_canonical(&bundle_path, &untrained).unwrap();
        write_canonical(&evidence_path, &passing_evidence(&untrained)).unwrap();
        assert!(
            embed_dataset(
                &csv,
                TargetSelector::Name("y"),
                Task::Binary,
                &bundle_path,
                &evidence_path,
            )
            .unwrap_err()
            .to_string()
            .contains("untrained router")
        );

        let bundle = test_bundle(128);
        write_canonical(&bundle_path, &bundle).unwrap();
        fs::write(&evidence_path, b"{malformed").unwrap();
        assert!(qualify_embedding_router(&bundle_path, &evidence_path).is_err());

        let finite_evidence = canonical_json(&passing_evidence(&bundle)).unwrap();
        let non_finite_evidence = String::from_utf8(finite_evidence).unwrap().replace(
            "\"top_four_oracle_recall\":1.0",
            "\"top_four_oracle_recall\":1e999",
        );
        fs::write(&evidence_path, non_finite_evidence).unwrap();
        assert!(qualify_embedding_router(&bundle_path, &evidence_path).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
