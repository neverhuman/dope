//! Private evidence types and source-free design admission.
use super::*;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossEvidence {
    pub query_ids: Vec<String>,
    pub cluster_ids: Vec<String>,
    pub real: Vec<f64>,
    pub synthetic: Vec<f64>,
    pub null: Vec<f64>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceMargin {
    pub relative: f64,
    pub absolute_scaled: f64,
    pub absolute_raw: Option<f64>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRun {
    pub family_id: String,
    pub task_id: String,
    pub method_id: String,
    pub fit_id: String,
    pub sample_id: String,
    pub learner_seed_id: String,
    pub task_axis_id: String,
    pub pairs: Option<LossEvidence>,
    pub scale: f64,
    pub margin: InferenceMargin,
    pub status: String,
    pub global_seed_namespace: Option<String>,
    pub global_seed_id: Option<String>,
}

impl EvaluationRun {
    pub fn key(&self) -> [String; 6] {
        [
            &self.family_id,
            &self.task_id,
            &self.method_id,
            &self.fit_id,
            &self.sample_id,
            &self.learner_seed_id,
        ]
        .map(Clone::clone)
    }

    pub(super) fn values(&self, endpoint: &str) -> MetricResult<Vec<f64>> {
        let p = self
            .pairs
            .as_ref()
            .ok_or(MetricError::new("missing_loss_evidence"))?;
        let n = p.real.len();
        require(
            n > 0
                && [
                    p.synthetic.len(),
                    p.null.len(),
                    p.query_ids.len(),
                    p.cluster_ids.len(),
                ]
                .iter()
                .all(|v| *v == n),
            "unpaired_queries",
        )?;
        require(
            p.query_ids.iter().collect::<BTreeSet<_>>().len() == n,
            "unpaired_queries",
        )?;
        require(
            p.query_ids
                .iter()
                .chain(&p.cluster_ids)
                .all(|s| !s.is_empty()),
            "unpaired_queries",
        )?;
        require(
            p.real
                .iter()
                .chain(&p.synthetic)
                .chain(&p.null)
                .all(|v| v.is_finite() && *v >= 0.0),
            "invalid_loss_evidence",
        )?;
        let values: Vec<_> = p
            .real
            .iter()
            .zip(&p.synthetic)
            .map(|(r, s)| match endpoint {
                "gap" => s - r,
                "normalized_gap" => (s - r) / self.scale,
                _ => s - r - (self.margin.relative * r + self.margin.absolute_scaled * self.scale),
            })
            .collect();
        require(
            values.iter().all(|v| v.is_finite()),
            "invalid_loss_evidence",
        )?;
        Ok(values)
    }
}

#[derive(Clone, Deserialize)]
pub struct CrossedDesign {
    pub runs: Vec<EvaluationRun>,
    pub required_methods: Vec<String>,
    pub scheduled_keys: Vec<[String; 6]>,
    pub cluster_unit: String,
}

impl CrossedDesign {
    pub fn validate(&self) -> MetricResult<()> {
        require(self.cluster_unit == "source_family", "invalid_cluster_unit")?;
        let methods: BTreeSet<_> = self.required_methods.iter().collect();
        require(
            !self.runs.is_empty()
                && !methods.is_empty()
                && methods.len() == self.required_methods.len()
                && methods.iter().all(|m| !m.is_empty()),
            "invalid_method_schedule",
        )?;
        let keys: BTreeSet<_> = self.runs.iter().map(EvaluationRun::key).collect();
        require(keys.len() == self.runs.len(), "duplicate_run")?;
        require(
            self.scheduled_keys.len() == keys.len()
                && self.scheduled_keys.iter().cloned().collect::<BTreeSet<_>>() == keys,
            "incomplete_schedule",
        )?;
        let (mut tasks, mut policies, mut fits, mut seeds, mut task_axes) = (
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        for r in &self.runs {
            require(
                r.key().iter().all(|s| !s.is_empty())
                    && !r.task_axis_id.is_empty()
                    && methods.contains(&r.method_id),
                "invalid_run_identity",
            )?;
            require(
                r.scale.is_finite()
                    && r.scale > 0.0
                    && [r.margin.relative, r.margin.absolute_scaled]
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0)
                    && r.margin
                        .absolute_raw
                        .is_none_or(|v| v.is_finite() && v >= 0.0),
                "invalid_run_policy",
            )?;
            require(
                matches!(
                    r.status.as_str(),
                    "measured" | "technical_failure" | "unsupported" | "unresolved"
                ) && (r.status == "measured") == r.pairs.is_some(),
                "invalid_run_status",
            )?;
            require(
                r.global_seed_namespace.is_some() == r.global_seed_id.is_some()
                    && r.global_seed_namespace
                        .iter()
                        .chain(&r.global_seed_id)
                        .all(|s| !s.is_empty()),
                "invalid_seed_namespace",
            )?;
            let lineage = (&r.family_id, &r.task_axis_id);
            require(
                task_axes
                    .insert(lineage, &r.task_id)
                    .is_none_or(|v| v == &r.task_id),
                "inconsistent_task_axis",
            )?;
            require(
                tasks
                    .insert(&r.task_id, lineage)
                    .is_none_or(|v| v == lineage),
                "inconsistent_task_lineage",
            )?;
            let policy = (
                r.scale,
                r.margin.relative,
                r.margin.absolute_scaled,
                r.margin.absolute_raw,
                &r.global_seed_namespace,
            );
            require(
                policies
                    .insert((&r.task_id, &r.method_id), policy)
                    .is_none_or(|v| v == policy),
                "inconsistent_run_policy",
            )?;
            let coupling = (&r.global_seed_namespace, &r.global_seed_id);
            require(
                fits.insert((&r.task_id, &r.method_id, &r.fit_id), coupling)
                    .is_none_or(|v| v == coupling),
                "inconsistent_fit_seed",
            )?;
            if r.global_seed_namespace.is_some() {
                require(
                    seeds
                        .insert((&r.task_id, &r.method_id, coupling), &r.fit_id)
                        .is_none_or(|v| v == &r.fit_id),
                    "duplicate_fit_seed",
                )?;
            }
        }
        Ok(())
    }
}

pub fn task_substream(master_seed: u64, task: &str, component: &str) -> MetricResult<u64> {
    require(
        !task.is_empty() && !component.is_empty(),
        "invalid_stream_identity",
    )?;
    let identity = serde_json::to_vec(&json!([
        "transfer-statistics-v1",
        master_seed,
        task,
        component
    ]))
    .map_err(|_| MetricError::new("invalid_stream_identity"))?;
    let digest = Sha256::digest(identity);
    Ok(u64::from_le_bytes(digest[..8].try_into().unwrap()))
}
