#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ControlContrastiveLift {
    pub retention: f64,
    pub predictor_only_retention: f64,
    pub independent_marginals_retention: f64,
    pub real_bootstrap_4n_retention: Option<f64>,
    pub lift_over_predictor_only: f64,
    pub lift_over_independent_marginals: f64,
    pub lift_vs_real_bootstrap_4n: Option<f64>,
    pub positive_predictor_lift: bool,
    pub positive_dependence_lift: bool,
    pub contrastive_utility_score: f64,
}

pub fn control_contrastive_lift(
    retention: f64,
    predictor_only_retention: f64,
    independent_marginals_retention: f64,
    real_bootstrap_4n_retention: Option<f64>,
) -> Result<ControlContrastiveLift, MetricError> {
    let retention = finite(retention)?;
    let predictor_only = finite(predictor_only_retention)?;
    let independent = finite(independent_marginals_retention)?;
    let bootstrap_4n = match real_bootstrap_4n_retention {
        Some(value) => Some(finite(value)?),
        None => None,
    };
    let lift_over_predictor_only = finite(retention - predictor_only)?;
    let lift_over_independent_marginals = finite(retention - independent)?;
    let lift_vs_real_bootstrap_4n = match bootstrap_4n {
        Some(value) => Some(finite(retention - value)?),
        None => None,
    };
    let base = retention.clamp(0.0, 1.0);
    let predictor_bonus = (lift_over_predictor_only.max(0.0) / 0.20).clamp(0.0, 1.0);
    let dependence_bonus = (lift_over_independent_marginals.max(0.0) / 0.25).clamp(0.0, 1.0);
    let contrastive_utility_score =
        finite((0.65 * base + 0.20 * predictor_bonus + 0.15 * dependence_bonus).clamp(0.0, 1.0))?;
    Ok(ControlContrastiveLift {
        retention,
        predictor_only_retention: predictor_only,
        independent_marginals_retention: independent,
        real_bootstrap_4n_retention: bootstrap_4n,
        lift_over_predictor_only,
        lift_over_independent_marginals,
        lift_vs_real_bootstrap_4n,
        positive_predictor_lift: lift_over_predictor_only > 0.0,
        positive_dependence_lift: lift_over_independent_marginals > 0.0,
        contrastive_utility_score,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelDriverAttribution {
    pub k: usize,
    pub ndcg_at_k: f64,
    pub top_k_jaccard: f64,
    pub top_k_overlap: f64,
    pub l1_attribution_agreement: f64,
    pub sign_agreement: f64,
    pub composite_driver_fidelity: f64,
}

pub fn driver_attribution_agreement(
    real_importances: &[f64],
    synthetic_importances: &[f64],
    real_signs: Option<&[i8]>,
    synthetic_signs: Option<&[i8]>,
) -> Result<ModelDriverAttribution, MetricError> {
    let features = real_importances.len();
    if features == 0 || features != synthetic_importances.len() {
        return Err(MetricError::InvalidEvidence);
    }
    for &value in real_importances.iter().chain(synthetic_importances) {
        if finite(value)? < 0.0 {
            return Err(MetricError::InvalidEvidence);
        }
    }
    let real_total = finite(real_importances.iter().sum())?;
    let syn_total = finite(synthetic_importances.iter().sum())?;
    let active_real_features = real_importances.iter().filter(|&&v| v > 1e-12).count();
    if real_total <= 1e-12 || active_real_features == 0 {
        return Err(MetricError::InvalidEvidence);
    }
    let sign_agreement = match (real_signs, synthetic_signs) {
        (None, None) => return Err(MetricError::InvalidEvidence),
        (Some(real_s), Some(syn_s))
            if real_s.len() >= features && real_s.len() == syn_s.len() =>
        {
            if real_s
                .iter()
                .chain(syn_s)
                .any(|&sign| !(-1..=1).contains(&sign))
            {
                return Err(MetricError::InvalidEvidence);
            }
            if real_importances
                .iter()
                .zip(real_s.iter())
                .any(|(&imp, &sign)| (imp > 1e-12 && sign == 0) || (imp <= 1e-12 && sign != 0))
                || synthetic_importances
                    .iter()
                    .zip(syn_s.iter())
                    .any(|(&imp, &sign)| imp > 1e-12 && sign == 0)
            {
                return Err(MetricError::InvalidEvidence);
            }
            let active: Vec<(usize, i8, i8)> = real_s
                .iter()
                .copied()
                .zip(syn_s.iter().copied())
                .enumerate()
                .filter_map(|(idx, (r, s))| (r != 0).then_some((idx, r, s)))
                .collect();
            if active.is_empty() {
                return Err(MetricError::InvalidEvidence);
            }
            let matches = active
                .iter()
                .filter(|&&(idx, r, s)| {
                    r == s && (idx >= features || synthetic_importances[idx] > 1e-12)
                })
                .count();
            matches as f64 / active.len() as f64
        }
        _ => return Err(MetricError::InvalidEvidence),
    };
    let k = ((features as f64 * 0.10).ceil() as usize)
        .clamp(5, 20)
        .min(features)
        .min(active_real_features)
        .max(1);
    let (ndcg_at_k, top_k_jaccard, top_k_overlap, l1_attribution_agreement) = if syn_total <= 1e-12
    {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let mut real_order: Vec<usize> = (0..features).collect();
        real_order.sort_by(|&a, &b| {
            real_importances[b]
                .total_cmp(&real_importances[a])
                .then_with(|| a.cmp(&b))
        });
        let mut syn_order: Vec<usize> = (0..features).collect();
        syn_order.sort_by(|&a, &b| {
            synthetic_importances[b]
                .total_cmp(&synthetic_importances[a])
                .then_with(|| a.cmp(&b))
        });
        let real_top: std::collections::BTreeSet<usize> = real_order
            .iter()
            .copied()
            .filter(|&idx| real_importances[idx] > 1e-12)
            .take(k)
            .collect();
        let syn_top: std::collections::BTreeSet<usize> = syn_order
            .iter()
            .copied()
            .filter(|&idx| synthetic_importances[idx] > 1e-12)
            .take(k)
            .collect();
        let intersection = real_top.intersection(&syn_top).count();
        let union = real_top.union(&syn_top).count().max(1);
        let overlap = intersection as f64 / k as f64;
        let jaccard = intersection as f64 / union as f64;
        let idcg: f64 = finite(
            real_order
                .iter()
                .take(k)
                .enumerate()
                .map(|(rank, &idx)| real_importances[idx] / ((rank + 2) as f64).log2())
                .sum(),
        )?;
        let dcg: f64 = finite(
            syn_order
                .iter()
                .take(k)
                .enumerate()
                .filter(|(_, idx)| synthetic_importances[**idx] > 1e-12)
                .map(|(rank, &idx)| real_importances[idx] / ((rank + 2) as f64).log2())
                .sum(),
        )?;
        let ndcg = if idcg <= 1e-12 {
            0.0
        } else {
            finite(dcg / idcg)?.clamp(0.0, 1.0)
        };
        let tv_distance = finite(
            0.5 * (0..features)
                .map(|idx| {
                    (real_importances[idx] / real_total - synthetic_importances[idx] / syn_total)
                        .abs()
                })
                .sum::<f64>(),
        )?;
        let l1_agreement = finite(1.0 - tv_distance)?.clamp(0.0, 1.0);
        (ndcg, jaccard, overlap, l1_agreement)
    };
    let composite_driver_fidelity = finite(
        (0.35 * ndcg_at_k
            + 0.25 * top_k_jaccard
            + 0.25 * l1_attribution_agreement
            + 0.15 * sign_agreement)
            .clamp(0.0, 1.0),
    )?;
    Ok(ModelDriverAttribution {
        k,
        ndcg_at_k,
        top_k_jaccard,
        top_k_overlap,
        l1_attribution_agreement,
        sign_agreement,
        composite_driver_fidelity,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KnowledgeCompressionMetrics {
    pub artifact_bytes: u64,
    pub encoded_numbers: u64,
    pub real_rows: u64,
    pub real_columns: u64,
    pub byte_limit: u64,
    pub bits_per_original_cell: f64,
    pub numbers_per_original_cell: f64,
    pub bpc_score: f64,
    pub parameter_sparsity_score: f64,
    pub log_byte_score: f64,
    pub knowledge_compression_score: f64,
    pub utility_per_kilobyte: f64,
    pub utility_per_encoded_number: f64,
}

pub fn knowledge_compression_metrics(
    artifact_bytes: u64,
    encoded_numbers: u64,
    real_rows: u64,
    real_columns: u64,
    byte_limit: u64,
    retained_utility: f64,
) -> Result<KnowledgeCompressionMetrics, MetricError> {
    if artifact_bytes == 0
        || encoded_numbers == 0
        || real_rows == 0
        || real_columns == 0
        || byte_limit < 2
    {
        return Err(MetricError::InvalidEvidence);
    }
    let retained_utility = finite(retained_utility)?;
    let cells = (real_rows as f64) * (real_columns as f64);
    let bits_per_original_cell = finite(8.0 * (artifact_bytes as f64) / cells)?;
    let numbers_per_original_cell = finite((encoded_numbers as f64) / cells)?;
    let bpc_score = finite((1.0 / (1.0 + bits_per_original_cell)).clamp(0.0, 1.0))?;
    let parameter_sparsity_score = finite(
        (1.0 / (1.0 + (encoded_numbers as f64) / ((real_columns as f64) * 8.0))).clamp(0.0, 1.0),
    )?;
    let log_byte_score = if artifact_bytes <= byte_limit {
        finite((1.0 - (artifact_bytes as f64).ln() / (byte_limit as f64).ln()).clamp(0.0, 1.0))?
    } else {
        0.0
    };
    let knowledge_compression_score = if retained_utility > 0.0 && artifact_bytes <= byte_limit {
        finite(
            (0.40 * bpc_score + 0.35 * parameter_sparsity_score + 0.25 * log_byte_score)
                .clamp(0.0, 1.0),
        )?
    } else {
        0.0
    };
    let positive_utility = retained_utility.max(0.0);
    let utility_per_kilobyte = finite(positive_utility / ((artifact_bytes as f64) / 1024.0))?;
    let utility_per_encoded_number = finite(positive_utility / (encoded_numbers as f64))?;
    Ok(KnowledgeCompressionMetrics {
        artifact_bytes,
        encoded_numbers,
        real_rows,
        real_columns,
        byte_limit,
        bits_per_original_cell,
        numbers_per_original_cell,
        bpc_score,
        parameter_sparsity_score,
        log_byte_score,
        knowledge_compression_score,
        utility_per_kilobyte,
        utility_per_encoded_number,
    })
}
