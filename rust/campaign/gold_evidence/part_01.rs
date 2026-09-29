#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BaselineFailure {
    pub dataset_id: String,
    pub candidate_id: String,
    pub auditor_id: String,
    #[serde(default)]
    pub size_multiplier: Option<usize>,
    #[serde(default)]
    pub generation_seed: Option<u64>,
    #[serde(default)]
    pub auditor_seed: Option<u64>,
    pub error: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BaselineSummary {
    pub format: String,
    pub version: u8,
    pub cohort_records: usize,
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub failures: Vec<BaselineFailure>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct GoldRecordBlock {
    pub(crate) format: String,
    pub(crate) version: u8,
    pub(crate) dataset_id: String,
    matrix_identity: String,
    #[serde(default)]
    routing: Option<GoldRoutingDecision>,
    pub(crate) evidence: Vec<JobEvidence>,
    pub(crate) failures: Vec<BaselineFailure>,
    #[serde(default = "default_neural_target_weight")]
    pub(crate) neural_target_weight: f64,
    #[serde(default)]
    pub(crate) neural_structural_penalty: f64,
}

const GOLD_RECORD_BLOCK_VERSION: u8 = 2;
const GOLD_BLOCK_RECEIPT_VERSION: u8 = 2;

fn default_neural_target_weight() -> f64 {
    2.0
}

struct GoldMatrixIdentity<'a> {
    phase: &'a str,
    dataset_id: &'a str,
    candidate_ids: &'a [String],
    auditor_ids: &'a [String],
    size_multipliers: &'a [usize],
    generation_seeds: &'a [u64],
    auditor_seeds: &'a [u64],
    primary_only: bool,
    neural_target_weight: f64,
    neural_structural_penalty: f64,
    routing_sha256: Option<&'a str>,
}

fn gold_record_matrix_identity(identity: &GoldMatrixIdentity<'_>) -> Result<String> {
    let mut value = serde_json::json!({
        "format": "dope-gold-record-block-key",
        "version": GOLD_RECORD_BLOCK_VERSION,
        "phase": identity.phase,
        "dataset": identity.dataset_id,
        "candidates": identity.candidate_ids,
        "auditors": identity.auditor_ids,
        "size_multipliers": identity.size_multipliers,
        "generation_seeds": identity.generation_seeds,
        "auditor_seeds": identity.auditor_seeds,
        "primary_only": identity.primary_only,
        "neural_target_weight": identity.neural_target_weight,
        "neural_structural_penalty": identity.neural_structural_penalty
    });
    if let Some(routing_sha256) = identity.routing_sha256 {
        value
            .as_object_mut()
            .expect("gold matrix identity is an object")
            .insert(
                "routing_sha256".into(),
                Value::String(routing_sha256.into()),
            );
    }
    Ok(hashes(&canonical_json(&value)?).blake3)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct GoldRoutingDecision {
    format: String,
    version: u8,
    router_bundle_sha256: String,
    ranked_candidates: Vec<String>,
    top_four_candidates: Vec<String>,
    selected_candidates: Vec<String>,
    prediction_sha256: String,
    sketch_ms: f64,
    inference_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GoldBlockReceipt {
    pub format: String,
    pub version: u8,
    pub dataset_id: String,
    pub matrix_identity: String,
    pub phase: String,
    pub worker: String,
    pub source_sha256: String,
    pub evaluator_binary_sha256: String,
    #[serde(default)]
    pub router_bundle_sha256: Option<String>,
    pub cohort: ContentHashes,
    pub block: ContentHashes,
    pub evidence_cells: usize,
    pub failed_cells: usize,
    pub signed_unix_seconds: u64,
    pub signature: Option<String>,
}

impl GoldBlockReceipt {
    fn signing_bytes(&self) -> Result<Vec<u8>> {
        let mut unsigned = self.clone();
        unsigned.signature = None;
        canonical_json(&unsigned)
    }

    fn sign(&mut self, key: &[u8; 32]) -> Result<()> {
        self.signature = Some(format!(
            "blake3-keyed-v1:{}",
            blake3::keyed_hash(key, &self.signing_bytes()?).to_hex()
        ));
        Ok(())
    }

    pub fn verify(&self, key: &[u8; 32]) -> Result<()> {
        let expected = format!(
            "blake3-keyed-v1:{}",
            blake3::keyed_hash(key, &self.signing_bytes()?).to_hex()
        );
        if self.format != "dope-gold-block-receipt"
            || self.version != GOLD_BLOCK_RECEIPT_VERSION
            || !matches!(
                self.phase.as_str(),
                "training_gold" | "validation_select" | "validation_cert"
            )
            || !matches!(self.worker.as_str(), "xbabe1" | "xbabe2" | "xbabe3")
            || !is_lower_hex(&self.source_sha256, &[64])
            || !is_lower_hex(&self.evaluator_binary_sha256, &[64])
            || self
                .router_bundle_sha256
                .as_ref()
                .is_some_and(|value| !is_lower_hex(value, &[64]))
            || !is_lower_hex(&self.matrix_identity, &[64])
            || self.signature.as_deref() != Some(&expected)
        {
            return Err(DopeError::Data("invalid gold block receipt".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GoldReceiptSummary {
    pub format: String,
    pub version: u8,
    pub worker: String,
    pub blocks: usize,
    pub evidence_cells: usize,
    pub failed_cells: usize,
    pub cohort: ContentHashes,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MeasuredCandidateSummary {
    pub candidate_id: String,
    pub implementation_hash: String,
    pub cells: usize,
    pub mean_bounded_utility: f64,
    pub runtime_p95_ms: u64,
    pub peak_memory_p95_bytes: u64,
    pub artifact_p95_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MeasuredEvidenceSummary {
    pub format: String,
    pub version: u8,
    pub cells: usize,
    pub lineage_groups: usize,
    pub lineage_profile_outcomes: usize,
    pub candidates: usize,
    pub auditors: usize,
    pub tasks: Vec<String>,
    pub candidate_frontier: Vec<MeasuredCandidateSummary>,
    pub calibration_degradation_max: Option<f64>,
    pub rare_tail_retention_min: Option<f64>,
    pub subgroup_retention_min: Option<f64>,
    pub nominal_coverage_min: Option<f64>,
    pub nominal_coverage_max: Option<f64>,
    pub driver_agreement_min: Option<f64>,
    pub joint_fidelity_min: Option<f64>,
    pub query_error_max: Option<f64>,
    pub type_i_error_max: Option<f64>,
    pub membership_auc_max: Option<f64>,
    pub feature_importance_complete: bool,
    pub feature_importance_applicable: bool,
    pub feature_importance_spearman_min: Option<f64>,
    pub feature_importance_top_k_jaccard_min: Option<f64>,
    pub attribute_advantage_max: Option<f64>,
    pub exact_copies: usize,
    pub near_copies: usize,
}

fn aggregate_importance<'a>(
    cells: impl IntoIterator<Item = &'a JobEvidence>,
) -> (bool, bool, Option<f64>, Option<f64>) {
    let mut count = 0usize;
    let mut complete = true;
    let mut applicable = false;
    let mut spearman = None::<f64>;
    let mut jaccard = None::<f64>;
    let mut missing_rank = false;
    for cell in cells {
        count += 1;
        complete &= cell.version == JOB_EVIDENCE_VERSION
            && cell.feature_importance_feature_count == cell.features
            && cell.feature_importance_real_shares.len() == cell.features
            && cell.feature_importance_synthetic_shares.len() == cell.features;
        if cell.feature_importance_informative_count >= 3 {
            applicable = true;
            if let Some(value) = cell.feature_importance_spearman {
                spearman = Some(spearman.map_or(value, |minimum| minimum.min(value)));
            } else {
                missing_rank = true;
            }
            let value = cell.feature_importance_top_k_agreement;
            jaccard = Some(jaccard.map_or(value, |minimum| minimum.min(value)));
        }
    }
    (
        count > 0 && complete,
        applicable,
        if missing_rank { None } else { spearman },
        jaccard,
    )
}

fn bounded_router_utility(evidence: &JobEvidence) -> f64 {
    let improvement = evidence.null_loss - evidence.trtr_loss;
    if improvement >= 0.01 * evidence.null_loss.abs() && improvement.abs() > f64::EPSILON {
        ((evidence.null_loss - evidence.tstr_loss) / improvement).clamp(0.0, 1.0)
    } else {
        f64::from(evidence.tstr_loss <= evidence.trtr_loss + 0.01 * evidence.null_loss)
    }
}

fn for_each_evidence(
    paths: &[PathBuf],
    callback: &mut impl FnMut(JobEvidence) -> Result<()>,
) -> Result<()> {
    let mut expanded = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut entries = fs::read_dir(path)
                .map_err(|error| io_error(path, error))?
                .map(|entry| {
                    entry
                        .map(|entry| entry.path())
                        .map_err(|error| io_error(path, error))
                })
                .collect::<Result<Vec<_>>>()?;
            entries.retain(|entry| {
                entry
                    .extension()
                    .is_some_and(|extension| extension == "json")
            });
            entries.sort();
            expanded.extend(entries);
        } else {
            expanded.push(path.clone());
        }
    }
    for path in expanded {
        let file = File::open(&path).map_err(|error| io_error(&path, error))?;
        let mut lines = std::io::BufReader::new(file).lines();
        let Some(first_line) = lines.next() else {
            continue;
        };
        let first_line = first_line.map_err(|error| io_error(&path, error))?;
        if first_line.trim().is_empty() {
            continue;
        }
        let first_value: Value = serde_json::from_str(&first_line)?;
        if first_value.get("format").and_then(Value::as_str) == Some("dope-gold-record-block") {
            let block: GoldRecordBlock = serde_json::from_value(first_value)?;
            if block.version != GOLD_RECORD_BLOCK_VERSION || lines.next().is_some() {
                return Err(DopeError::Data(format!(
                    "invalid evidence record block at {}",
                    path.display()
                )));
            }
            for cell in block.evidence {
                callback(cell)?;
            }
        } else {
            callback(serde_json::from_value(first_value)?)?;
            for line in lines {
                let line = line.map_err(|error| io_error(&path, error))?;
                if !line.trim().is_empty() {
                    callback(serde_json::from_str(&line)?)?;
                }
            }
        }
    }
    Ok(())
}