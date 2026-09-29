
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CandidateFrontierEntry {
    pub candidate_id: String,
    #[serde(default)]
    pub implementation_hash: String,
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub timed_out: usize,
    pub mean_retention: Option<f64>,
    pub runtime_p95_ms: Option<u64>,
    pub peak_memory_p95_bytes: Option<u64>,
    pub artifact_bytes_p95: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MetricsBundle {
    pub format: String,
    pub version: u8,
    pub status: crate::ledger::LedgerStatus,
    pub kpi: KpiSummary,
    pub candidate_frontier: Vec<CandidateFrontierEntry>,
    pub router: Option<RouterEvidence>,
    pub failed_gates: Vec<String>,
    pub validation_cert_open_count: usize,
    pub sealed_test_open_count: usize,
    pub receipts_reconciled: usize,
    #[serde(default)]
    pub certification: Option<CertificationDetail>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProfileRegretDetail {
    pub outcomes: usize,
    pub mean_regret: f64,
    pub upper_95: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CertificationDetail {
    pub eligible_lineages: usize,
    pub evaluated_lineages: usize,
    pub missing_lineages: usize,
    pub timeout_lineages: usize,
    pub exhaustive_cells: usize,
    pub routed_cells: usize,
    pub shadow_cells: usize,
    pub failed_cells: usize,
    pub top_four_oracle_recall: Option<f64>,
    pub mean_candidate_regret: Option<f64>,
    pub random_candidate_regret: Option<f64>,
    pub best_fixed_candidate_id: String,
    pub best_fixed_candidate_regret: Option<f64>,
    pub beats_random_candidate: bool,
    pub beats_best_fixed_candidate: bool,
    pub maximum_profile_regret_upper: Option<f64>,
    pub regret_by_profile: BTreeMap<String, ProfileRegretDetail>,
    pub work_avoided_fraction: Option<f64>,
    pub latency_reduction_fraction: Option<f64>,
    pub router_bundle_bytes: u64,
    pub router_inference_p95_ms: Option<f64>,
    pub sketch_p95_ms: Option<f64>,
    pub paired_hypervolume_improvement: Option<f64>,
    pub paired_hypervolume_ci_lower: Option<f64>,
    pub candidate_availability: BTreeMap<String, usize>,
    pub auditor_availability: BTreeMap<String, usize>,
    pub candidate_implementation_hashes: BTreeMap<String, String>,
    pub auditor_implementation_hashes: BTreeMap<String, String>,
    pub receipts_expected: usize,
    pub receipts_reconciled: usize,
}

fn percentile(values: &mut [u64], quantile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[((values.len() - 1) as f64 * quantile).ceil() as usize])
}

fn option_max(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values.flatten().reduce(f64::max)
}

fn option_min(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values.flatten().reduce(f64::min)
}