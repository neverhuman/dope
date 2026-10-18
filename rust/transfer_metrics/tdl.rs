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

    /// Inverse the fit-only affine target transform before squared error.
    pub fn regression_transformed(
        raw_targets: &[f64],
        real: &[f64],
        synthetic: &[f64],
        null: &[f64],
        offset: f64,
        scale: f64,
    ) -> Result<Self, MetricError> {
        finite(offset)?;
        if finite(scale)? <= 0.0 {
            return Err(MetricError::InvalidEvidence);
        }
        let inverse = |predictions: &[f64]| -> Result<Vec<f64>, MetricError> {
            predictions
                .iter()
                .map(|p| finite(finite(*p)? * scale + offset))
                .collect()
        };
        Self::regression(
            raw_targets,
            &inverse(real)?,
            &inverse(synthetic)?,
            &inverse(null)?,
        )
    }

    /// Labels index the same frozen class order in every prediction arm.
    /// Check normalization before clipping each class and renormalizing.
    pub fn classification(
        targets: &[usize],
        real: &[Vec<f64>],
        synthetic: &[Vec<f64>],
        null: &[Vec<f64>],
        classes: usize,
    ) -> Result<Self, MetricError> {
        if classes < 2 || targets.iter().any(|y| *y >= classes) {
            return Err(MetricError::InvalidEvidence);
        }
        let losses = |predictions: &[Vec<f64>]| -> Result<Vec<f64>, MetricError> {
            if predictions.len() != targets.len() {
                return Err(MetricError::InvalidEvidence);
            }
            targets
                .iter()
                .zip(predictions)
                .map(|(target, row)| {
                    if row.len() != classes {
                        return Err(MetricError::InvalidEvidence);
                    }
                    for p in row {
                        if !(0.0..=1.0).contains(&finite(*p)?) {
                            return Err(MetricError::InvalidEvidence);
                        }
                    }
                    if (finite(row.iter().sum())? - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
                        return Err(MetricError::InvalidEvidence);
                    }
                    let clipped: Vec<f64> = row
                        .iter()
                        .map(|p| p.clamp(PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP))
                        .collect();
                    finite(-(clipped[*target] / finite(clipped.iter().sum())?).ln())
                })
                .collect()
        };
        let mut pairs = Self::new(losses(real)?, losses(synthetic)?, losses(null)?)?;
        pairs.loss_space = LossSpace::DiscreteLogLoss;
        Ok(pairs)
    }

    pub fn query_harm_margin(&self, margin: TdlMargin) -> Result<Vec<f64>, MetricError> {
        self.synthetic
            .iter()
            .zip(&self.real)
            .zip(&self.null)
            .map(|((s, r), n)| query_harm(*s, *r, *n, margin.tau))
            .collect()
    }

    pub fn query_benefit_margin(&self, margin: TdlMargin) -> Result<Vec<f64>, MetricError> {
        Ok(self
            .query_harm_margin(margin)?
            .into_iter()
            .map(|z| -z)
            .collect())
    }

    pub(crate) fn loss_vectors(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.real, &self.synthetic, &self.null)
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

pub const PROBABILITY_CLIP: f64 = 1e-15;
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-12;

/// TDL retention threshold; independent of historical generic `Margin`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TdlMargin {
    pub tau: f64,
}

impl Default for TdlMargin {
    fn default() -> Self {
        Self { tau: 0.90 }
    }
}

/// Positive harm violates the retention constraint. A point estimate cannot certify it.
pub fn query_harm(ls: f64, lr: f64, l0: f64, tau: f64) -> Result<f64, MetricError> {
    if finite(ls)? < 0.0
        || finite(lr)? < 0.0
        || finite(l0)? < 0.0
        || !(0.0..=1.0).contains(&finite(tau)?)
    {
        return Err(MetricError::InvalidEvidence);
    }
    // Group the convex reference to retain exact zero at the hand-calculated T90 boundary.
    finite(ls - finite(tau * lr + (1.0 - tau) * l0)?)
}

pub fn query_benefit(ls: f64, lr: f64, l0: f64, tau: f64) -> Result<f64, MetricError> {
    Ok(-query_harm(ls, lr, l0, tau)?)
}

#[cfg(test)]
mod tests_tdl {
    use super::*;

    #[test]
    fn t90_boundary_orientation_and_historical_margin() {
        assert_eq!(TdlMargin::default().tau, 0.9);
        for (ls, z) in [(46.0, 0.0), (45.0, -1.0), (47.0, 1.0)] {
            assert_eq!(query_harm(ls, 40.0, 100.0, 0.9).unwrap(), z);
            assert_eq!(query_benefit(ls, 40.0, 100.0, 0.9).unwrap(), -z);
        }
        let pairs = PairedLosses::new(vec![40.0], vec![46.0], vec![100.0]).unwrap();
        assert_eq!(
            pairs.query_harm_margin(TdlMargin::default()).unwrap(),
            vec![0.0]
        );
        assert_eq!(
            pairs.query_benefit_margin(TdlMargin::default()).unwrap(),
            vec![0.0]
        );
        let historical = Margin {
            relative: 0.1,
            absolute_scaled: 0.0,
            absolute_raw: None,
        };
        assert_eq!(
            pairs
                .query_z(TrainingScale::Positive(1.0), historical)
                .unwrap(),
            vec![2.0]
        );
        for invalid in [f64::NAN, f64::INFINITY, -1.0, 1.1] {
            assert!(query_harm(1.0, 1.0, 1.0, invalid).is_err());
        }
        for invalid in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(query_harm(invalid, 1.0, 1.0, 0.9).is_err());
        }
    }

    #[test]
    fn inverse_transform_precedes_loss_in_every_arm() {
        let pairs = PairedLosses::regression_transformed(
            &[10.0, 14.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            10.0,
            4.0,
        )
        .unwrap();
        assert_eq!(pairs.means().unwrap(), (0.0, 8.0, 8.0));
        assert_eq!(
            pairs.query_harm_margin(TdlMargin::default()).unwrap(),
            vec![16.0, -1.5999999999999996]
        );
        assert!(
            PairedLosses::regression_transformed(&[0.0], &[1.0], &[1.0], &[1.0], 0.0, 0.0).is_err()
        );
    }

    #[test]
    fn proper_discrete_loss_checks_every_class_and_clips_endpoints() {
        for classes in [2, 3] {
            let real = vec![vec![1.0 / classes as f64; classes]];
            let mut exact = vec![vec![0.0; classes]];
            exact[0][0] = 1.0;
            let pairs = PairedLosses::classification(&[0], &real, &exact, &real, classes).unwrap();
            assert_eq!(pairs.loss_space, LossSpace::DiscreteLogLoss);
            assert!((pairs.means().unwrap().0 - (classes as f64).ln()).abs() < 1e-12);
            assert!(pairs.means().unwrap().1.is_finite());
        }
        for row in [
            vec![0.7, 0.7],
            vec![-0.1, 1.1],
            vec![f64::NAN, 0.5],
            vec![f64::INFINITY, 0.0],
            vec![1.0],
        ] {
            assert!(
                PairedLosses::classification(&[0], &[row], &[vec![0.5, 0.5]], &[vec![0.5, 0.5]], 2)
                    .is_err()
            );
        }
        assert!(
            PairedLosses::classification(
                &[2],
                &[vec![0.5, 0.5]],
                &[vec![0.5, 0.5]],
                &[vec![0.5, 0.5]],
                2
            )
            .is_err()
        );
    }
}
