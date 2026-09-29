
#[cfg(feature = "gpu-training")]
pub fn train_distilled_router(
    labels: &[RouterLabel],
    candidate_ids: &[String],
    training_evidence_sha256: &str,
    validation_lineage_ids: Option<&BTreeSet<String>>,
) -> Result<(RouterBundle, RouterTrainingReport)> {
    use tch::nn::{Module, OptimizerConfig};

    if !tch::Cuda::is_available() {
        return Err(DopeError::Unsupported(
            "router teacher training requires CUDA libtorch".into(),
        ));
    }
    if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":4096:8") {
        return Err(DopeError::Unsupported(
            "router teacher requires CUBLAS_WORKSPACE_CONFIG=:4096:8".into(),
        ));
    }
    if labels.is_empty() || candidate_ids.is_empty() {
        return Err(DopeError::Data("router training labels are empty".into()));
    }
    let candidate_index = candidate_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let sketch_width = labels[0].sketch.values.len();
    if labels.iter().any(|label| {
        label.sketch.version != DATASET_SKETCH_VERSION
            || label.sketch.values.len() != sketch_width
            || !candidate_index.contains_key(label.candidate_id.as_str())
    }) {
        return Err(DopeError::Data(
            "router labels have inconsistent sketches or candidates".into(),
        ));
    }
    let mut by_outcome = BTreeMap::<(&str, &str), Vec<usize>>::new();
    for (index, label) in labels.iter().enumerate() {
        by_outcome
            .entry((
                label.lineage_group_id.as_str(),
                label.structural_profile.as_str(),
            ))
            .or_default()
            .push(index);
    }
    if by_outcome.values().any(|indices| {
        indices.len() != candidate_ids.len()
            || indices
                .iter()
                .map(|&index| labels[index].candidate_id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                != candidate_ids.len()
    }) {
        return Err(DopeError::Data(
            "every router lineage/profile outcome must have every frozen candidate label".into(),
        ));
    }
    let all_lineages = labels
        .iter()
        .map(|label| label.lineage_group_id.as_str())
        .collect::<BTreeSet<_>>();
    let validation_lineages = if let Some(explicit) = validation_lineage_ids {
        let selected = all_lineages
            .iter()
            .filter(|lineage| explicit.contains(**lineage))
            .copied()
            .collect::<BTreeSet<_>>();
        if selected.len() != explicit.len()
            || selected.is_empty()
            || selected.len() == all_lineages.len()
        {
            return Err(DopeError::Data(
                "explicit router validation lineages are absent, empty, or consume training".into(),
            ));
        }
        selected
    } else {
        let selected = all_lineages
            .iter()
            .filter(|lineage| blake3::hash(lineage.as_bytes()).as_bytes()[0] % 5 == 0)
            .copied()
            .collect::<BTreeSet<_>>();
        if selected.is_empty() {
            BTreeSet::from([*all_lineages.iter().next().expect("nonempty lineages")])
        } else {
            selected
        }
    };
    let train_indices = labels
        .iter()
        .enumerate()
        .filter(|(_, label)| !validation_lineages.contains(label.lineage_group_id.as_str()))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if train_indices.is_empty() {
        return Err(DopeError::Data(
            "router training split contains no training lineages".into(),
        ));
    }
    let mut sketch_mean = vec![0.0f32; sketch_width];
    for &index in &train_indices {
        for (slot, value) in sketch_mean.iter_mut().zip(&labels[index].sketch.values) {
            *slot += *value / train_indices.len() as f32;
        }
    }
    let mut sketch_std = vec![0.0f32; sketch_width];
    for &index in &train_indices {
        for ((slot, value), mean) in sketch_std
            .iter_mut()
            .zip(&labels[index].sketch.values)
            .zip(&sketch_mean)
        {
            *slot += (*value - *mean).powi(2) / train_indices.len() as f32;
        }
    }
    sketch_std
        .iter_mut()
        .for_each(|value| *value = value.sqrt().max(1e-5));
    let ga2m_router = train_ga2m_router(
        labels,
        candidate_ids,
        &candidate_index,
        &train_indices,
        &sketch_mean,
        &sketch_std,
    );
    let mut ga2m_scores = vec![0.0f32; labels.len()];
    for indices in by_outcome.values() {
        let predictions = ga2m_router
            .predict(&labels[indices[0]].sketch)?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        for &index in indices {
            ga2m_scores[index] = predictions[labels[index].candidate_id.as_str()];
        }
    }

    let oracle = by_outcome
        .iter()
        .map(|(&outcome, indices)| {
            let best = indices
                .iter()
                .map(|&index| labels[index].retention)
                .reduce(f32::max)
                .unwrap_or(0.0);
            (outcome, best)
        })
        .collect::<BTreeMap<_, _>>();
    let targets = labels
        .iter()
        .zip(&ga2m_scores)
        .map(|(label, ga2m_score)| {
            // Retention is deliberately unclamped in KPI aggregation. The
            // teacher uses a robust bounded target so a single near-zero TRTR
            // denominator cannot dominate every candidate ranking update.
            let retention = label.retention.clamp(-2.0, 4.0);
            let retention_residual = retention - ga2m_score;
            let regret = (oracle[&(
                label.lineage_group_id.as_str(),
                label.structural_profile.as_str(),
            )] - label.retention)
                .max(0.0)
                .clamp(0.0, 4.0);
            [
                retention_residual,
                retention_residual,
                f32::from(label.failed || label.retention < 0.0),
                label.runtime_ms,
                label.runtime_ms,
                label.peak_memory_bytes,
                label.artifact_bytes,
                f32::from(label.failed),
                regret,
                regret,
            ]
        })
        .collect::<Vec<_>>();
    let mut target_mean = [0.0f32; ROUTER_OUTPUTS];
    for &index in &train_indices {
        for (slot, value) in target_mean.iter_mut().zip(targets[index]) {
            *slot += value / train_indices.len() as f32;
        }
    }
    let mut target_std = [0.0f32; ROUTER_OUTPUTS];
    for &index in &train_indices {
        for ((slot, value), mean) in target_std.iter_mut().zip(targets[index]).zip(target_mean) {
            *slot += (value - mean).powi(2) / train_indices.len() as f32;
        }
    }
    target_std
        .iter_mut()
        .for_each(|value| *value = value.sqrt().max(1e-5));
    let flat_sketches = labels
        .iter()
        .flat_map(|label| {
            label
                .sketch
                .values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    ((*value - sketch_mean[index]) / sketch_std[index]).clamp(-8.0, 8.0)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let flat_targets = targets
        .iter()
        .flat_map(|target| {
            (0..ROUTER_OUTPUTS)
                .map(|index| (target[index] - target_mean[index]) / target_std[index])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let candidate_indices = labels
        .iter()
        .map(|label| candidate_index[label.candidate_id.as_str()] as i64)
        .collect::<Vec<_>>();
    crate::libtorch::with_seeded_libtorch(1729, include!("student_training/teacher_student.rs"))
}
