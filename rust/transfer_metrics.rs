//! Original-unit transfer diagnostics, independent of historical certification.
//!
//! Deliberately unregistered until the integrator adds `pub mod transfer_metrics;`.
//! Query losses are evaluator-private. Neither a score nor these diagnostics
//! authorize release. Bytes remain an independent feasibility constraint.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetricError {
    InvalidEvidence,
    Nonfinite,
    DegenerateScale,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossSpace {
    OriginalMse,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainingScale {
    Positive(f64),
    Degenerate,
}

fn finite(value: f64) -> Result<f64, MetricError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MetricError::Nonfinite)
    }
}

fn mean(values: &[f64]) -> Result<f64, MetricError> {
    if values.is_empty() {
        return Err(MetricError::InvalidEvidence);
    }
    finite(values.iter().map(|v| v / values.len() as f64).sum())
}

pub fn training_variance(
    fit_targets: &[f64],
    minimum_variance: f64,
) -> Result<TrainingScale, MetricError> {
    if finite(minimum_variance)? < 0.0 {
        return Err(MetricError::InvalidEvidence);
    }
    for value in fit_targets {
        finite(*value)?;
    }
    let center = mean(fit_targets)?;
    let variance = mean(
        &fit_targets
            .iter()
            .map(|value| (value - center).powi(2))
            .collect::<Vec<_>>(),
    )?;
    Ok(if variance <= minimum_variance {
        TrainingScale::Degenerate
    } else {
        TrainingScale::Positive(variance)
    })
}

#[derive(Clone)]
pub struct PairedLosses {
    real: Vec<f64>,
    synthetic: Vec<f64>,
    null: Vec<f64>,
    pub loss_space: LossSpace,
}

impl std::fmt::Debug for PairedLosses {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairedLosses")
            .field("loss_space", &self.loss_space)
            .finish_non_exhaustive()
    }
}

impl PairedLosses {
    /// Arrays must already have exactly aligned private query/entity identities.
    pub fn new(real: Vec<f64>, synthetic: Vec<f64>, null: Vec<f64>) -> Result<Self, MetricError> {
        if real.is_empty() || real.len() != synthetic.len() || real.len() != null.len() {
            return Err(MetricError::InvalidEvidence);
        }
        for value in real.iter().chain(&synthetic).chain(&null) {
            if finite(*value)? < 0.0 {
                return Err(MetricError::InvalidEvidence);
            }
        }
        Ok(Self {
            real,
            synthetic,
            null,
            loss_space: LossSpace::OriginalMse,
        })
    }

    pub fn regression(
        raw_targets: &[f64],
        real_predictions: &[f64],
        synthetic_predictions: &[f64],
        null_predictions: &[f64],
    ) -> Result<Self, MetricError> {
        let losses = |predictions: &[f64]| -> Result<Vec<f64>, MetricError> {
            if predictions.len() != raw_targets.len() {
                return Err(MetricError::InvalidEvidence);
            }
            raw_targets
                .iter()
                .zip(predictions)
                .map(|(y, p)| finite((finite(*y)? - finite(*p)?).powi(2)))
                .collect()
        };
        Self::new(
            losses(real_predictions)?,
            losses(synthetic_predictions)?,
            losses(null_predictions)?,
        )
    }

    pub fn means(&self) -> Result<(f64, f64, f64), MetricError> {
        Ok((mean(&self.real)?, mean(&self.synthetic)?, mean(&self.null)?))
    }

    /// Paired Z retains the estimated real reference inside each query pair.
    pub fn query_z(&self, scale: TrainingScale, margin: Margin) -> Result<Vec<f64>, MetricError> {
        self.real
            .iter()
            .zip(&self.synthetic)
            .map(|(r, s)| finite(s - r - margin.tolerance(*r, scale)?))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GuardedRatio {
    Finite(f64),
    BothZero,
    Unbounded,
}

pub fn loss_ratio(real: f64, synthetic: f64) -> Result<GuardedRatio, MetricError> {
    if finite(real)? < 0.0 || finite(synthetic)? < 0.0 {
        return Err(MetricError::InvalidEvidence);
    }
    Ok(if real == 0.0 {
        if synthetic == 0.0 {
            GuardedRatio::BothZero
        } else {
            GuardedRatio::Unbounded
        }
    } else if (synthetic / real).is_finite() {
        GuardedRatio::Finite(synthetic / real)
    } else {
        GuardedRatio::Unbounded
    })
}

#[derive(Clone, Copy, Debug)]
pub struct LossSummary {
    pub real_loss: f64,
    pub synthetic_loss: f64,
    pub null_loss: f64,
    pub gap: f64,
    pub normalized_gap: Option<f64>,
    pub null_normalized_gap: Option<f64>,
    pub retention_numerator: f64,
    pub retention_denominator: f64,
    pub retention: Option<f64>,
    pub ratio: GuardedRatio,
}

pub fn loss_summary(
    pairs: &PairedLosses,
    scale: TrainingScale,
) -> Result<LossSummary, MetricError> {
    let (real, synthetic, null) = pairs.means()?;
    let gap = finite(synthetic - real)?;
    let numerator = null - synthetic;
    let denominator = null - real;
    let normalized_gap = match scale {
        TrainingScale::Degenerate => None,
        TrainingScale::Positive(value) if finite(value)? > 0.0 => Some(finite(gap / value)?),
        TrainingScale::Positive(_) => return Err(MetricError::InvalidEvidence),
    };
    Ok(LossSummary {
        real_loss: real,
        synthetic_loss: synthetic,
        null_loss: null,
        gap,
        normalized_gap,
        null_normalized_gap: if null > 0.0 {
            Some(finite(gap / null)?)
        } else {
            None
        },
        retention_numerator: numerator,
        retention_denominator: denominator,
        retention: if denominator > 0.0 && denominator >= 0.01 * null.abs() {
            Some(finite(numerator / denominator)?)
        } else {
            None
        },
        ratio: loss_ratio(real, synthetic)?,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Margin {
    pub relative: f64,
    pub absolute_scaled: f64,
    pub absolute_raw: Option<f64>,
}

impl Margin {
    pub fn tolerance(self, real: f64, scale: TrainingScale) -> Result<f64, MetricError> {
        if finite(real)? < 0.0
            || finite(self.relative)? < 0.0
            || finite(self.absolute_scaled)? < 0.0
        {
            return Err(MetricError::InvalidEvidence);
        }
        if let Some(raw) = self.absolute_raw
            && finite(raw)? < 0.0
        {
            return Err(MetricError::InvalidEvidence);
        }
        let tolerance = match scale {
            TrainingScale::Positive(value) if finite(value)? > 0.0 => {
                finite(self.relative * real + self.absolute_scaled * value)?
            }
            TrainingScale::Positive(_) => return Err(MetricError::InvalidEvidence),
            TrainingScale::Degenerate => {
                finite(self.absolute_raw.ok_or(MetricError::DegenerateScale)?)?
            }
        };
        if tolerance < 0.0 {
            return Err(MetricError::InvalidEvidence);
        }
        Ok(tolerance)
    }
}

pub fn weighted_cvar(values: &[f64], weights: &[f64], alpha: f64) -> Result<f64, MetricError> {
    if values.is_empty() || values.len() != weights.len() || !(0.0..1.0).contains(&finite(alpha)?) {
        return Err(MetricError::InvalidEvidence);
    }
    let mut rows = Vec::new();
    for (value, weight) in values.iter().zip(weights) {
        if finite(*weight)? < 0.0 {
            return Err(MetricError::InvalidEvidence);
        }
        rows.push((finite(*value)?, *weight));
    }
    let total = finite(weights.iter().sum())?;
    if total <= 0.0 {
        return Err(MetricError::InvalidEvidence);
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut remaining = 1.0 - alpha;
    let mut result = 0.0;
    for (value, weight) in rows {
        let take = (weight / total).min(remaining);
        result += value * take;
        remaining -= take;
        if remaining <= 0.0 {
            break;
        }
    }
    finite(result / (1.0 - alpha))
}

include!("transfer_metrics/mfs_v4.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_python_rust_golden_corpus() {
        let corpus: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/metric-golden-v1.json")).unwrap();
        for case in corpus["cases"].as_array().unwrap() {
            let pairs = PairedLosses::new(
                vec![case["real"].as_f64().unwrap()],
                vec![case["synthetic"].as_f64().unwrap()],
                vec![case["null"].as_f64().unwrap()],
            )
            .unwrap();
            let scale = case["scale"]
                .as_f64()
                .map_or(TrainingScale::Degenerate, TrainingScale::Positive);
            let summary = loss_summary(&pairs, scale).unwrap();
            assert_eq!(summary.gap, case["gap"].as_f64().unwrap());
            assert_eq!(summary.normalized_gap, case["normalized_gap"].as_f64());
            assert_eq!(summary.retention, case["retention"].as_f64());
            let ratio = match case["ratio_status"].as_str().unwrap() {
                "finite" => GuardedRatio::Finite(case["ratio"].as_f64().unwrap()),
                "both_zero" => GuardedRatio::BothZero,
                "unbounded" => GuardedRatio::Unbounded,
                _ => panic!("unknown golden ratio disposition"),
            };
            assert_eq!(summary.ratio, ratio);
        }
        if let Some(mfs_v4) = corpus.get("mfs_v4") {
            let lift_case = &mfs_v4["control_contrastive_lift"];
            let lift = control_contrastive_lift(
                lift_case["retention"].as_f64().unwrap(),
                lift_case["predictor_only_retention"].as_f64().unwrap(),
                lift_case["independent_marginals_retention"]
                    .as_f64()
                    .unwrap(),
                lift_case["real_bootstrap_4n_retention"].as_f64(),
            )
            .unwrap();
            assert!(
                (lift.lift_over_predictor_only
                    - lift_case["lift_over_predictor_only"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
            assert!(
                (lift.lift_over_independent_marginals
                    - lift_case["lift_over_independent_marginals"]
                        .as_f64()
                        .unwrap())
                .abs()
                    < 1e-12
            );
            assert!(
                (lift.contrastive_utility_score
                    - lift_case["contrastive_utility_score"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );

            let drv_case = &mfs_v4["driver_attribution"];
            let real_imp: Vec<f64> = drv_case["real_importances"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let syn_imp: Vec<f64> = drv_case["synthetic_importances"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let real_signs: Vec<i8> = drv_case["real_signs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i8)
                .collect();
            let syn_signs: Vec<i8> = drv_case["synthetic_signs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i8)
                .collect();
            let drv = driver_attribution_agreement(
                &real_imp,
                &syn_imp,
                Some(&real_signs),
                Some(&syn_signs),
            )
            .unwrap();
            assert!((drv.ndcg_at_k - drv_case["ndcg_at_k"].as_f64().unwrap()).abs() < 1e-12);
            assert!(
                (drv.top_k_jaccard - drv_case["top_k_jaccard"].as_f64().unwrap()).abs() < 1e-12
            );
            assert!(
                (drv.l1_attribution_agreement
                    - drv_case["l1_attribution_agreement"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
            assert!(
                (drv.sign_agreement - drv_case["sign_agreement"].as_f64().unwrap()).abs() < 1e-12
            );
            assert!(
                (drv.composite_driver_fidelity
                    - drv_case["composite_driver_fidelity"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );

            let comp_case = &mfs_v4["knowledge_compression"];
            let comp = knowledge_compression_metrics(
                comp_case["artifact_bytes"].as_u64().unwrap(),
                comp_case["encoded_numbers"].as_u64().unwrap(),
                comp_case["real_rows"].as_u64().unwrap(),
                comp_case["real_columns"].as_u64().unwrap(),
                comp_case["byte_limit"].as_u64().unwrap(),
                comp_case["retained_utility"].as_f64().unwrap(),
            )
            .unwrap();
            assert!(
                (comp.bits_per_original_cell
                    - comp_case["bits_per_original_cell"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
            assert!((comp.bpc_score - comp_case["bpc_score"].as_f64().unwrap()).abs() < 1e-12);
            assert!(
                (comp.parameter_sparsity_score
                    - comp_case["parameter_sparsity_score"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
            assert!(
                (comp.log_byte_score - comp_case["log_byte_score"].as_f64().unwrap()).abs() < 1e-12
            );
            assert!(
                (comp.knowledge_compression_score
                    - comp_case["knowledge_compression_score"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn retention_does_not_hide_elevenfold_mse() {
        let pairs = PairedLosses::new(vec![1.0], vec![11.0], vec![101.0]).unwrap();
        let summary = loss_summary(&pairs, TrainingScale::Positive(100.0)).unwrap();
        assert_eq!(summary.retention, Some(0.9));
        assert_eq!(summary.ratio, GuardedRatio::Finite(11.0));
        assert_eq!(summary.normalized_gap, Some(0.1));
    }

    #[test]
    fn raw_targets_and_affine_units() {
        let pairs = PairedLosses::regression(
            &[0.0, 1.0, 1000.0],
            &[1.0, 1.0, 1000.0],
            &[0.0, 1.0, 900.0],
            &[0.0; 3],
        )
        .unwrap();
        assert!(pairs.means().unwrap().1 > 1000.0 * pairs.means().unwrap().0);
        let scale = training_variance(&[0.0, 1.0, 4.0], 0.0).unwrap();
        let changed = training_variance(&[30.0, 37.0, 58.0], 0.0).unwrap();
        if let (TrainingScale::Positive(a), TrainingScale::Positive(b)) = (scale, changed) {
            assert!((49.0 * a - b).abs() < 1e-10);
        } else {
            panic!("positive scales expected");
        }
    }

    #[test]
    fn zero_degenerate_invalid_and_unclipped_retention() {
        assert_eq!(loss_ratio(0.0, 1.0).unwrap(), GuardedRatio::Unbounded);
        assert_eq!(loss_ratio(0.0, 0.0).unwrap(), GuardedRatio::BothZero);
        assert_eq!(
            training_variance(&[5.0, 5.0], 0.0).unwrap(),
            TrainingScale::Degenerate
        );
        for value in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(PairedLosses::new(vec![value], vec![1.0], vec![2.0]).is_err());
        }
        for (syn, retained) in [(120.0, -0.19), (0.0, 1.01)] {
            let pairs = PairedLosses::new(vec![1.0], vec![syn], vec![101.0]).unwrap();
            assert_eq!(
                loss_summary(&pairs, TrainingScale::Degenerate)
                    .unwrap()
                    .retention,
                Some(retained)
            );
        }
        let margin = Margin {
            relative: 0.1,
            absolute_scaled: 0.0,
            absolute_raw: None,
        };
        assert_eq!(
            margin.tolerance(1.0, TrainingScale::Degenerate),
            Err(MetricError::DegenerateScale)
        );
    }

    #[test]
    fn fractional_tail_mass() {
        let tail = weighted_cvar(&[0.0, 10.0, 20.0], &[0.85, 0.1, 0.05], 0.9).unwrap();
        assert!((tail - 15.0).abs() < 1e-10);
        assert!(weighted_cvar(&[0.0], &[0.0], 0.9).is_err());
    }

    #[test]
    fn control_contrastive_and_knowledge_compression_monotone_invariants() {
        let strong = control_contrastive_lift(0.95, 0.70, 0.60, Some(0.92)).unwrap();
        let predictor_clone = control_contrastive_lift(0.70, 0.70, 0.60, Some(0.92)).unwrap();
        assert!(strong.positive_predictor_lift);
        assert!(strong.positive_dependence_lift);
        assert!(!predictor_clone.positive_predictor_lift);
        assert!(strong.contrastive_utility_score > predictor_clone.contrastive_utility_score);

        let compact = knowledge_compression_metrics(1_536, 48, 1_000, 16, 10_240, 0.95).unwrap();
        let bloated = knowledge_compression_metrics(8_192, 512, 1_000, 16, 10_240, 0.95).unwrap();
        assert!(compact.knowledge_compression_score > bloated.knowledge_compression_score);
        assert!(compact.utility_per_kilobyte > bloated.utility_per_kilobyte);
        assert!(compact.utility_per_encoded_number > bloated.utility_per_encoded_number);

        let zero_utility =
            knowledge_compression_metrics(1_536, 48, 1_000, 16, 10_240, 0.0).unwrap();
        assert_eq!(zero_utility.knowledge_compression_score, 0.0);
        let over_budget =
            knowledge_compression_metrics(12_288, 48, 1_000, 16, 10_240, 0.95).unwrap();
        assert_eq!(over_budget.knowledge_compression_score, 0.0);

        assert!(
            driver_attribution_agreement(
                &[0.0, 0.0, 0.0, 0.0],
                &[0.0, 0.0, 0.0, 0.0],
                Some(&[1, -1, 1, -1]),
                Some(&[1, -1, 1, -1]),
            )
            .is_err()
        );
        assert!(
            driver_attribution_agreement(
                &[0.6, 0.4, 0.0, 0.0],
                &[0.6, 0.4, 0.0, 0.0],
                Some(&[0, 1, 0, 0]),
                Some(&[0, 1, 0, 0]),
            )
            .is_err()
        );
        let disjoint = driver_attribution_agreement(
            &[0.6, 0.4, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.5, 0.5, 0.0],
            Some(&[1, -1, 0, 0, 0]),
            Some(&[0, 0, 1, -1, 0]),
        )
        .unwrap();
        assert_eq!(disjoint.k, 2);
        assert_eq!(disjoint.top_k_jaccard, 0.0);
        assert_eq!(disjoint.top_k_overlap, 0.0);
        assert_eq!(disjoint.ndcg_at_k, 0.0);
    }
}
