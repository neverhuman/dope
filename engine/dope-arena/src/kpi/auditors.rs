//! The native auditor roster and the per-row losses of the scoreboard.
//!
//! One entry point fits a named auditor on a training table and predicts the
//! cell's T rows. At a row ratio of 1 it returns the same bits as
//! `dope_core::eval::fit_predict_auditor` (tested below). It stays a thin
//! wrapper over the public fitters because [`Recipe`] also scales the
//! count-based knobs with the training table, which that hook cannot do.
//! Inputs are already in the cell's A-only unit space (`celldata::ScaleCard`).
//!
//! The tree auditor keeps exact duplicate rows on one side of its
//! early-stopping holdout, like dope-core's own auditors. The default of
//! `gbdt::Params` does not (it keeps `landmarks-native-v1` bit-identical), so
//! the recipe sets `group_duplicates` explicitly.
//!
//! Losses: squared error in A-standardised target units for regression;
//! log-loss with probabilities clipped at `1e-12` and renormalised for
//! classification. The null predictor is the A mean (0 in unit space) or the
//! A class prior.

use dope_arena::{ArenaError, ArenaResult};
use dope_core::eval::{gbdt, linear, nn, Matrix};
use dope_core::target::Head;

pub const LINEAR: &str = "linear-seeded-v1";
pub const GBDT: &str = "gbdt";
pub const MLP: &str = "mlp-seeded-v1";
pub const ROSTER: [&str; 3] = [LINEAR, GBDT, MLP];
pub const PROB_FLOOR: f64 = 1e-12;

/// Widest view any auditor is asked to fit (the neural recipe's own limit).
pub const MAX_COLS: usize = 1024;
/// Limits of `nn::fit_predict_seeded_cpu`, checked up front so a cell is
/// unsupported for every arm alike instead of failing for the larger tables.
const NN_MAX_ROWS: usize = 50_000;
const NN_MAX_VALUES: usize = 2_000_000;
const NN_MAX_OUTPUTS: usize = 64;
const NN_MAX_BATCH: usize = 4096;

pub const REASON_SHAPE: &str = "auditor_shape_limit";
pub const REASON_FAILED: &str = "auditor_failed";

pub fn is_known(auditor: &str) -> bool {
    ROSTER.contains(&auditor)
}

fn uses_nn(auditor: &str, head: Head) -> bool {
    auditor == MLP || (auditor == LINEAR && matches!(head, Head::Multiclass(_)))
}

/// Whether `auditor` can fit `train_rows` and score `test_rows` at this width.
pub fn supports(
    auditor: &str,
    head: Head,
    cols: usize,
    train_rows: usize,
    test_rows: usize,
) -> bool {
    if cols == 0 || cols > MAX_COLS || train_rows == 0 || test_rows == 0 {
        return false;
    }
    if !uses_nn(auditor, head) {
        return true;
    }
    let fits = |rows: usize| rows <= NN_MAX_ROWS && rows * cols <= NN_MAX_VALUES;
    fits(train_rows) && fits(test_rows) && head.outputs() <= NN_MAX_OUTPUTS
}

/// How a recipe scales with the size of its training table.
///
/// `row_ratio` is `training rows / rows of the cell's A table` (at least 1).
/// Tree leaf size, leaf shrinkage and the neural batch are multiplied by it,
/// so a table that repeats every row `k` times yields the model of the
/// unrepeated table: the count-based knobs are per unit of fit-table
/// information, not per physical row.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    pub seed: u64,
    pub row_ratio: f64,
}

impl Recipe {
    pub fn new(seed: u64, train_rows: usize, reference_rows: usize) -> Self {
        let ratio = train_rows as f64 / reference_rows.max(1) as f64;
        Self {
            seed,
            row_ratio: if ratio.is_finite() {
                ratio.max(1.0)
            } else {
                1.0
            },
        }
    }

    fn gbdt(&self) -> gbdt::Params {
        let base = gbdt::Params::default();
        gbdt::Params {
            min_leaf: (base.min_leaf as f64 * self.row_ratio).round() as usize,
            lambda: base.lambda * self.row_ratio,
            seed: self.seed,
            group_duplicates: true,
            ..base
        }
    }

    fn nn(&self, base: nn::Params) -> nn::Params {
        let batch = (base.batch as f64 * self.row_ratio).round() as usize;
        nn::Params {
            batch: batch.clamp(1, NN_MAX_BATCH),
            ..base
        }
    }
}

/// Fit `auditor` on `(x, y)` and predict `test`: values (regression),
/// P(class 1) (binary) or row-major class probabilities (multiclass).
pub fn fit_predict(
    auditor: &str,
    head: Head,
    x: &Matrix,
    y: &[f64],
    test: &Matrix,
    recipe: Recipe,
) -> ArenaResult<Vec<f64>> {
    let failed = || ArenaError::Contract(REASON_FAILED.into());
    if x.rows != y.len() || x.cols != test.cols || x.rows == 0 {
        return Err(failed());
    }
    let pred = match (auditor, head) {
        (LINEAR, Head::Multiclass(_)) => nn::fit_predict_seeded_cpu(
            x,
            y,
            head,
            test,
            &recipe.nn(nn::Params::softmax_linear(recipe.seed)),
        )
        .map_err(|_| failed())?,
        (LINEAR, _) => linear::fit_predict(x, y, head, test),
        (GBDT, _) => gbdt::fit_predict(x, y, head, test, &recipe.gbdt()),
        (MLP, _) => {
            nn::fit_predict_seeded_cpu(x, y, head, test, &recipe.nn(nn::Params::mlp(recipe.seed)))
                .map_err(|_| failed())?
        }
        _ => return Err(ArenaError::Contract("unknown auditor".into())),
    };
    if pred.len() != test.rows * head.outputs() || pred.iter().any(|v| !v.is_finite()) {
        return Err(failed());
    }
    Ok(pred)
}

/// Predictions of the null model for `rows` query rows, from A targets in unit space.
pub fn null_predictions(head: Head, y_a: &[f64], rows: usize) -> Vec<f64> {
    let n = y_a.len().max(1) as f64;
    match head {
        Head::Regression | Head::Binary => vec![y_a.iter().sum::<f64>() / n; rows],
        Head::Multiclass(k) => {
            let mut prior = vec![0.0; k];
            for &class in y_a {
                prior[(class as usize).min(k - 1)] += 1.0;
            }
            prior.iter_mut().for_each(|count| *count /= n);
            (0..rows).flat_map(|_| prior.iter().copied()).collect()
        }
    }
}

/// Per-row losses and the predictions they were computed from (probabilities
/// clipped at `PROB_FLOOR` and renormalised; regression values unchanged).
pub fn losses(head: Head, pred: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = head.outputs();
    let mut loss = Vec::with_capacity(y.len());
    let mut used = Vec::with_capacity(pred.len());
    for (row, &target) in y.iter().enumerate() {
        match head {
            Head::Regression => {
                loss.push((pred[row] - target) * (pred[row] - target));
                used.push(pred[row]);
            }
            Head::Binary => {
                let one = pred[row].max(PROB_FLOOR);
                let zero = (1.0 - pred[row]).max(PROB_FLOOR);
                let p_true = if target >= 0.5 { one } else { zero } / (one + zero);
                loss.push(0.0 - p_true.ln());
                used.push(one / (one + zero));
            }
            Head::Multiclass(_) => {
                let probs = &pred[row * k..(row + 1) * k];
                let total: f64 = probs.iter().map(|p| p.max(PROB_FLOOR)).sum();
                let class = (target as usize).min(k - 1);
                loss.push(0.0 - (probs[class].max(PROB_FLOOR) / total).ln());
                used.extend(probs.iter().map(|p| p.max(PROB_FLOOR) / total));
            }
        }
    }
    (loss, used)
}

#[cfg(test)]
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::seeds::KpiRng;

    /// `n` rows of a noisy signal on three features, in unit-like scale.
    fn table(head: Head, n: usize, seed: u64) -> (Matrix, Vec<f64>) {
        let mut rng = KpiRng::new(seed);
        let (mut data, mut y) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let x: Vec<f64> = (0..3).map(|_| rng.normal()).collect();
            let score = 1.2 * x[0] - 0.8 * x[1] + 0.5 * x[0] * x[2];
            y.push(match head {
                Head::Regression => score + 0.3 * rng.normal(),
                Head::Binary => f64::from(score + 0.7 * rng.normal() > 0.0),
                Head::Multiclass(_) => {
                    let noisy = score + 0.6 * rng.normal();
                    f64::from(noisy > -0.7) + f64::from(noisy > 0.7)
                }
            });
            data.extend(x);
        }
        (
            Matrix {
                rows: n,
                cols: 3,
                data,
            },
            y,
        )
    }

    fn repeat(x: &Matrix, y: &[f64], times: usize) -> (Matrix, Vec<f64>) {
        let mut data = Vec::new();
        let mut targets = Vec::new();
        for _ in 0..times {
            data.extend_from_slice(&x.data);
            targets.extend_from_slice(y);
        }
        (
            Matrix {
                rows: x.rows * times,
                cols: x.cols,
                data,
            },
            targets,
        )
    }

    fn mean_loss(head: Head, pred: &[f64], y: &[f64]) -> f64 {
        mean(&losses(head, pred, y).0)
    }

    #[test]
    fn losses_clip_and_renormalise() {
        let (loss, used) = losses(Head::Regression, &[1.0, -2.0], &[3.0, -2.0]);
        assert_eq!((loss, used), (vec![4.0, 0.0], vec![1.0, -2.0]));

        let (loss, used) = losses(Head::Binary, &[0.25, 0.0, 1.0], &[1.0, 1.0, 1.0]);
        assert!((loss[0] - 4.0f64.ln()).abs() < 1e-12);
        assert!((loss[1] - (1e12f64).ln()).abs() < 1e-6, "{}", loss[1]);
        assert!(loss[2] >= 0.0 && loss[2] < 2e-12);
        assert!(used[1] > 0.0 && used[1] < 2e-12 && used[2] <= 1.0);

        let pred = [0.5, 0.5, 0.0, 0.2, 0.2, 0.2];
        let (loss, used) = losses(Head::Multiclass(3), &pred, &[2.0, 1.0]);
        assert!((loss[0] - (1e12f64).ln()).abs() < 1e-6);
        assert!((loss[1] - 3.0f64.ln()).abs() < 1e-12, "renormalised to 1/3");
        assert!((used[..3].iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(loss.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn null_model_is_the_fit_mean_or_class_prior() {
        assert_eq!(
            null_predictions(Head::Regression, &[1.0, 3.0], 2),
            [2.0, 2.0]
        );
        assert_eq!(
            null_predictions(Head::Binary, &[1.0, 0.0, 0.0, 0.0], 1),
            [0.25]
        );
        let prior = null_predictions(Head::Multiclass(3), &[0.0, 2.0, 2.0, 2.0], 2);
        assert_eq!(prior, [0.25, 0.0, 0.75, 0.25, 0.0, 0.75]);
        // A class absent from the fit rows is charged the probability floor.
        let (loss, _) = losses(Head::Multiclass(3), &prior, &[1.0, 2.0]);
        assert!(loss[0] > 27.0 && (loss[1] - (4.0f64 / 3.0).ln()).abs() < 1e-9);
    }

    #[test]
    fn support_limits_match_the_neural_contract() {
        assert!(supports(GBDT, Head::Regression, 1024, 500_000, 9));
        assert!(!supports(GBDT, Head::Regression, 1025, 10, 9));
        assert!(supports(MLP, Head::Binary, 400, 4096, 2000));
        assert!(!supports(MLP, Head::Binary, 600, 4096, 2000));
        assert!(supports(LINEAR, Head::Binary, 600, 4096, 2000));
        assert!(!supports(LINEAR, Head::Multiclass(3), 600, 4096, 2000));
        assert!(!supports(MLP, Head::Multiclass(65), 4, 100, 100));
        assert!(!supports(MLP, Head::Regression, 4, 0, 100));
        assert!(is_known(MLP) && !is_known("mlp"));
    }

    #[test]
    fn every_auditor_learns_and_ignores_repeated_training_rows() {
        for head in [Head::Regression, Head::Binary, Head::Multiclass(3)] {
            let (x, y) = table(head, 400, 11);
            let (xt, yt) = table(head, 600, 12);
            let null = mean_loss(head, &null_predictions(head, &y, xt.rows), &yt);
            let (x4, y4) = repeat(&x, &y, 4);
            for auditor in ROSTER {
                let once = Recipe::new(7, x.rows, x.rows);
                let pred = fit_predict(auditor, head, &x, &y, &xt, once).unwrap();
                let loss = mean_loss(head, &pred, &yt);
                assert!(
                    loss < 0.8 * null,
                    "{auditor} {head:?}: {loss} vs null {null}"
                );
                // Four copies of every row: the same model up to optimiser noise.
                let scaled = Recipe::new(7, x4.rows, x.rows);
                assert_eq!(scaled.row_ratio, 4.0);
                let pred4 = fit_predict(auditor, head, &x4, &y4, &xt, scaled).unwrap();
                let loss4 = mean_loss(head, &pred4, &yt);
                let drift = (loss4 - loss).abs() / (null - loss);
                assert!(
                    drift < 0.05,
                    "{auditor} {head:?}: repeated rows moved retention by {drift}"
                );
            }
        }
    }

    #[test]
    fn bad_inputs_are_errors_not_panics() {
        let (x, y) = table(Head::Regression, 40, 1);
        let (xt, _) = table(Head::Regression, 10, 2);
        let recipe = Recipe::new(0, 40, 40);
        let err = fit_predict("mlp", Head::Regression, &x, &y, &xt, recipe).unwrap_err();
        assert!(err.to_string().contains("unknown auditor"));
        let err = fit_predict(GBDT, Head::Regression, &x, &y[..39], &xt, recipe).unwrap_err();
        assert!(err.to_string().contains(REASON_FAILED));
        let mut wild = x.clone();
        wild.data[0] = f64::NAN;
        let err = fit_predict(MLP, Head::Regression, &wild, &y, &xt, recipe).unwrap_err();
        assert!(err.to_string().contains(REASON_FAILED));
        assert_eq!(Recipe::new(0, 10, 40).row_ratio, 1.0);
        assert_eq!(Recipe::new(0, 10, 0).row_ratio, 10.0);
    }

    #[test]
    fn tree_recipe_groups_duplicate_rows_at_every_ratio() {
        // `gbdt::Params::default()` leaves this off; the scoreboard must not.
        for (train_rows, reference_rows) in [(400, 400), (1600, 400), (10, 40), (10, 0)] {
            let params = Recipe::new(3, train_rows, reference_rows).gbdt();
            assert!(params.group_duplicates, "{train_rows}/{reference_rows}");
        }
    }

    #[test]
    fn at_row_ratio_one_every_auditor_is_the_dope_core_hook() {
        // Each match arm of `fit_predict` once, on the cheapest head it takes.
        let cases = [
            (LINEAR, Head::Regression),
            (LINEAR, Head::Multiclass(3)),
            (GBDT, Head::Regression),
            (MLP, Head::Regression),
        ];
        for (auditor, head) in cases {
            // 200 rows, the fewest for which the tree auditor holds rows out
            // for early stopping, each twice: a tree recipe that let a copy
            // reach both sides of the holdout would fit another model.
            let (x, y) = table(head, 100, 21);
            let (x, y) = repeat(&x, &y, 2);
            let (xt, _) = table(head, 20, 22);
            let recipe = Recipe::new(5, x.rows, x.rows);
            let local = fit_predict(auditor, head, &x, &y, &xt, recipe).unwrap();
            let hook = dope_core::eval::fit_predict_auditor(auditor, head, &x, &y, &xt, 5).unwrap();
            assert_eq!(local, hook, "{auditor} {head:?}");
        }
    }
}
