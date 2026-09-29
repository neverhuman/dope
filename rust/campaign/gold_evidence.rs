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

pub fn summarize_evidence(metric_paths: &[PathBuf], out: &Path) -> Result<MeasuredEvidenceSummary> {
    if metric_paths.is_empty() {
        return Err(DopeError::Data("evidence metrics inputs are empty".into()));
    }
    #[derive(Default)]
    struct CandidateAggregate {
        cells: usize,
        bounded_utility_sum: f64,
        runtimes: BTreeMap<u64, usize>,
        memories: BTreeMap<u64, usize>,
        artifact_bytes: BTreeMap<u64, usize>,
    }
    let histogram_percentile = |histogram: &BTreeMap<u64, usize>, cells: usize| {
        if cells == 0 {
            return 0;
        }
        let index = ((cells - 1) as f64 * 0.95).ceil() as usize;
        let mut cumulative = 0usize;
        for (&value, &count) in histogram {
            cumulative += count;
            if cumulative > index {
                return value;
            }
        }
        0
    };
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let mut full_replicates = BTreeMap::<(String, String, String, String, bool), u128>::new();
    let mut other_identities = BTreeSet::<String>::new();
    let mut candidates = BTreeMap::<String, CandidateAggregate>::new();
    let mut lineage_groups = BTreeSet::<String>::new();
    let mut lineage_profile_outcomes = BTreeSet::<(String, String)>::new();
    let mut auditors = BTreeSet::<String>::new();
    let mut tasks = BTreeSet::<String>::new();
    let mut cells = 0usize;
    let mut calibration_degradation_max = None;
    let mut rare_tail_retention_min = None;
    let mut subgroup_retention_min = None;
    let mut nominal_coverage_min = None;
    let mut nominal_coverage_max = None;
    let mut driver_agreement_min = None;
    let mut joint_fidelity_min = None;
    let mut query_error_max = None;
    let mut type_i_error_max = None;
    let mut membership_auc_max = None;
    let mut attribute_advantage_max = None;
    let mut importance_complete = true;
    let mut importance_applicable = false;
    let mut importance_spearman_min = None::<f64>;
    let mut importance_jaccard_min = None::<f64>;
    let mut importance_missing_rank = false;
    let mut exact_copies = 0usize;
    let mut near_copies = 0usize;
    let update_min = |slot: &mut Option<f64>, value: Option<f64>| {
        if let Some(value) = value {
            *slot = Some(slot.map_or(value, |existing| existing.min(value)));
        }
    };
    let update_max = |slot: &mut Option<f64>, value: Option<f64>| {
        if let Some(value) = value {
            *slot = Some(slot.map_or(value, |existing| existing.max(value)));
        }
    };
    for_each_evidence(metric_paths, &mut |cell| {
        if let Ok(slot) = router_replicate_slot(&cell, &auditor_ids) {
            let seen = full_replicates
                .entry((
                    cell.phase.clone(),
                    cell.lineage_group_id.clone(),
                    cell.structural_profile.clone(),
                    cell.candidate_id.clone(),
                    cell.routed,
                ))
                .or_default();
            let bit = 1u128 << slot;
            if *seen & bit != 0 {
                return Err(DopeError::Data(
                    "evidence metrics contain duplicate cell identities".into(),
                ));
            }
            *seen |= bit;
        } else {
            let identity = hashes(&canonical_json(&serde_json::json!({
                "phase": cell.phase,
                "lineage": cell.lineage_group_id,
                "profile": cell.structural_profile,
                "candidate": cell.candidate_id,
                "auditor": cell.auditor_id,
                "size_multiplier": cell.size_multiplier,
                "generation_seed": cell.generation_seed,
                "auditor_seed": cell.auditor_seed,
                "routed": cell.routed,
            }))?)
            .blake3;
            if !other_identities.insert(identity) {
                return Err(DopeError::Data(
                    "evidence metrics contain duplicate cell identities".into(),
                ));
            }
        }
        let aggregate = candidates.entry(cell.candidate_id.clone()).or_default();
        aggregate.cells += 1;
        aggregate.bounded_utility_sum += bounded_router_utility(&cell);
        *aggregate.runtimes.entry(cell.runtime_ms).or_default() += 1;
        *aggregate
            .memories
            .entry(cell.peak_memory_bytes)
            .or_default() += 1;
        *aggregate
            .artifact_bytes
            .entry(cell.artifact_bytes)
            .or_default() += 1;
        lineage_groups.insert(cell.lineage_group_id.clone());
        lineage_profile_outcomes.insert((
            cell.lineage_group_id.clone(),
            cell.structural_profile.clone(),
        ));
        auditors.insert(cell.auditor_id.clone());
        tasks.insert(cell.task.clone());
        update_max(
            &mut calibration_degradation_max,
            cell.calibration_degradation,
        );
        update_min(
            &mut rare_tail_retention_min,
            cell.rare_class_or_tail_retention,
        );
        update_min(
            &mut subgroup_retention_min,
            cell.supported_subgroup_retention,
        );
        update_min(&mut nominal_coverage_min, cell.nominal_95_coverage);
        update_max(&mut nominal_coverage_max, cell.nominal_95_coverage);
        update_min(&mut driver_agreement_min, cell.driver_agreement);
        update_min(&mut joint_fidelity_min, cell.joint_fidelity);
        update_max(&mut query_error_max, cell.query_p95_normalized_error);
        update_max(&mut type_i_error_max, cell.type_i_error);
        update_max(&mut membership_auc_max, cell.membership_auc);
        importance_complete &= cell.version == JOB_EVIDENCE_VERSION
            && cell.feature_importance_feature_count == cell.features
            && cell.feature_importance_real_shares.len() == cell.features
            && cell.feature_importance_synthetic_shares.len() == cell.features;
        if cell.feature_importance_informative_count >= 2 {
            importance_applicable = true;
            if let Some(value) = cell.feature_importance_spearman {
                update_min(&mut importance_spearman_min, Some(value));
            } else {
                importance_missing_rank = true;
            }
            update_min(
                &mut importance_jaccard_min,
                Some(cell.feature_importance_top_k_agreement),
            );
        }
        update_max(
            &mut attribute_advantage_max,
            cell.attribute_inference_advantage,
        );
        exact_copies += cell.exact_copies;
        near_copies += cell.near_copies;
        cells += 1;
        Ok(())
    })?;
    let mut candidate_frontier = candidates
        .into_iter()
        .map(|(candidate_id, aggregate)| MeasuredCandidateSummary {
            implementation_hash: empirical_backends()
                .into_iter()
                .find(|backend| backend.id == candidate_id)
                .map(|backend| backend.implementation_hash)
                .unwrap_or_default(),
            candidate_id,
            cells: aggregate.cells,
            mean_bounded_utility: aggregate.bounded_utility_sum / aggregate.cells.max(1) as f64,
            runtime_p95_ms: histogram_percentile(&aggregate.runtimes, aggregate.cells),
            peak_memory_p95_bytes: histogram_percentile(&aggregate.memories, aggregate.cells),
            artifact_p95_bytes: histogram_percentile(&aggregate.artifact_bytes, aggregate.cells),
        })
        .collect::<Vec<_>>();
    candidate_frontier.sort_by(|left, right| {
        right
            .mean_bounded_utility
            .total_cmp(&left.mean_bounded_utility)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let summary = MeasuredEvidenceSummary {
        format: "dope-measured-evidence-summary".into(),
        version: 2,
        cells,
        lineage_groups: lineage_groups.len(),
        lineage_profile_outcomes: lineage_profile_outcomes.len(),
        candidates: candidate_frontier.len(),
        auditors: auditors.len(),
        tasks: tasks.into_iter().collect(),
        candidate_frontier,
        calibration_degradation_max,
        rare_tail_retention_min,
        subgroup_retention_min,
        nominal_coverage_min,
        nominal_coverage_max,
        driver_agreement_min,
        joint_fidelity_min,
        query_error_max,
        type_i_error_max,
        membership_auc_max,
        feature_importance_complete: cells > 0 && importance_complete,
        feature_importance_applicable: importance_applicable,
        feature_importance_spearman_min: if importance_missing_rank {
            None
        } else {
            importance_spearman_min
        },
        feature_importance_top_k_jaccard_min: importance_jaccard_min,
        attribute_advantage_max,
        exact_copies,
        near_copies,
    };
    write_canonical(out, &summary)?;
    Ok(summary)
}
