
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReleaseDecision {
    pub format: String,
    pub version: u8,
    pub release_identity: Option<String>,
    pub measured_ptf_v1: Option<f64>,
    pub failed_gates: Vec<String>,
    pub sealed_test_open_count: usize,
}

pub fn failed_release_decision(
    state_dir: &Path,
    metrics: &MetricsBundle,
) -> Result<ReleaseDecision> {
    let state = load_state(state_dir)?;
    let decision = ReleaseDecision {
        format: "dope-release-decision".into(),
        version: 1,
        release_identity: None,
        measured_ptf_v1: metrics.kpi.ptf_v1,
        failed_gates: metrics.failed_gates.clone(),
        sealed_test_open_count: state.sealed_test_open_count,
    };
    Ok(decision)
}
