//! Plain downstream evaluators for the end-to-end tests, written without the
//! code under test: `Ridge(alpha = 1)` and `LogisticRegression(C = 1)` on
//! inputs standardised by the training rows, the test losses, the retention
//! ratio and feature-correlation gaps.

use dope_core::stats::solve_spd;

/// Row-major table: `p` feature columns and one target per row.
#[derive(Clone, Debug)]
pub struct Table {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub p: usize,
}

impl Table {
    pub fn rows(&self) -> usize {
        self.y.len()
    }
    pub fn column(&self, j: usize) -> Vec<f64> {
        self.x.iter().skip(j).step_by(self.p).copied().collect()
    }
    /// Rows `from..to`.
    pub fn slice(&self, from: usize, to: usize) -> Self {
        Self {
            x: self.x[from * self.p..to * self.p].to_vec(),
            y: self.y[from..to].to_vec(),
            p: self.p,
        }
    }
    /// The same rows with the feature columns in the order `order`.
    pub fn permuted(&self, order: &[usize]) -> Self {
        let x = self
            .x
            .chunks(self.p)
            .flat_map(|row| order.iter().map(|&j| row[j]))
            .collect();
        Self {
            x,
            y: self.y.clone(),
            p: self.p,
        }
    }
}

/// Per-column mean and SD of the training rows (SD 1 for a constant column).
pub struct Standardiser {
    mean: Vec<f64>,
    sd: Vec<f64>,
}

impl Standardiser {
    pub fn fit(train: &Table) -> Self {
        let n = train.rows() as f64;
        let (mut mean, mut sd) = (vec![0.0; train.p], vec![0.0; train.p]);
        for j in 0..train.p {
            let col = train.column(j);
            mean[j] = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|v| (v - mean[j]).powi(2)).sum::<f64>() / n;
            sd[j] = if var > 1e-18 { var.sqrt() } else { 1.0 };
        }
        Self { mean, sd }
    }
    /// Design matrix with a leading column of ones, row-major `n x (p + 1)`.
    pub fn design(&self, table: &Table) -> Vec<f64> {
        let mut out = Vec::with_capacity(table.rows() * (table.p + 1));
        for row in table.x.chunks(table.p) {
            out.push(1.0);
            for (j, v) in row.iter().enumerate() {
                out.push((v - self.mean[j]) / self.sd[j]);
            }
        }
        out
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Test mean squared error of `Ridge(alpha = 1)` trained on `train`.
pub fn ridge_test_mse(train: &Table, test: &Table) -> f64 {
    let scale = Standardiser::fit(train);
    let (a, d) = (scale.design(train), train.p + 1);
    let (mut gram, mut rhs) = (vec![0.0; d * d], vec![0.0; d]);
    for (row, &y) in a.chunks(d).zip(&train.y) {
        for i in 0..d {
            rhs[i] += row[i] * y;
            for j in 0..d {
                gram[i * d + j] += row[i] * row[j];
            }
        }
    }
    for j in 1..d {
        gram[j * d + j] += 1.0;
    }
    let theta = solve_spd(&gram, &rhs, d);
    let b = scale.design(test);
    let sse: f64 = b
        .chunks(d)
        .zip(&test.y)
        .map(|(row, y)| (dot(row, &theta) - y).powi(2))
        .sum();
    sse / test.rows() as f64
}

/// Class probabilities of `classes` logit rows (last axis), stable.
fn probabilities(logit: &[f64]) -> Vec<f64> {
    let top = logit.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let weight: Vec<f64> = logit.iter().map(|v| (v - top).exp()).collect();
    let total: f64 = weight.iter().sum();
    weight.into_iter().map(|w| w / total).collect()
}

/// Symmetric multinomial logistic regression with penalty `|w|^2 / 2` on the
/// non-intercept weights (scikit-learn's `LogisticRegression(C = 1)` for three
/// or more classes), by Newton steps on the full `classes * d` system.
fn softmax_fit(a: &[f64], y: &[usize], d: usize, classes: usize) -> Vec<f64> {
    let n = y.len();
    let size = classes * d;
    let mut theta = vec![0.0; size];
    let objective = |theta: &[f64]| -> f64 {
        let mut total = 0.0;
        for (row, &label) in a.chunks(d).zip(y) {
            let logit: Vec<f64> = (0..classes)
                .map(|k| dot(row, &theta[k * d..(k + 1) * d]))
                .collect();
            total += probabilities(&logit)[label].max(1e-300).ln();
        }
        let penalty: f64 = (0..size)
            .filter(|i| i % d != 0)
            .map(|i| theta[i] * theta[i])
            .sum();
        total - 0.5 * penalty
    };
    let mut best = objective(&theta);
    for _ in 0..30 {
        let (mut grad, mut hess) = (vec![0.0; size], vec![0.0; size * size]);
        for i in 0..n {
            let row = &a[i * d..(i + 1) * d];
            let logit: Vec<f64> = (0..classes)
                .map(|k| dot(row, &theta[k * d..(k + 1) * d]))
                .collect();
            let prob = probabilities(&logit);
            for k in 0..classes {
                let err = (y[i] == k) as u8 as f64 - prob[k];
                for u in 0..d {
                    grad[k * d + u] += err * row[u];
                }
                for l in 0..classes {
                    let w = if k == l {
                        prob[k] * (1.0 - prob[k])
                    } else {
                        -prob[k] * prob[l]
                    };
                    for u in 0..d {
                        let wu = w * row[u];
                        let line = &mut hess[(k * d + u) * size + l * d..][..d];
                        line.iter_mut().zip(row).for_each(|(h, v)| *h += wu * v);
                    }
                }
            }
        }
        for i in 0..size {
            if i % d != 0 {
                grad[i] -= theta[i];
                hess[i * size + i] += 1.0;
            } else {
                // The common shift of all intercepts is free: pin it lightly.
                hess[i * size + i] += 1e-6;
            }
        }
        let step = solve_spd(&hess, &grad, size);
        let mut scale = 1.0;
        let mut moved = false;
        for _ in 0..20 {
            let trial: Vec<f64> = theta
                .iter()
                .zip(&step)
                .map(|(t, s)| t + scale * s)
                .collect();
            let value = objective(&trial);
            if value >= best {
                moved = step.iter().any(|s| (scale * s).abs() > 1e-10);
                theta = trial;
                best = value;
                break;
            }
            scale *= 0.5;
        }
        if !moved {
            break;
        }
    }
    theta
}

/// Test log-loss of the logistic model trained on `train` (labels are class
/// indices in `0..classes`). Two classes use the binary parametrisation with
/// penalty `|w|^2 / 2` on one weight vector, as scikit-learn does.
pub fn logistic_test_loss(train: &Table, test: &Table, classes: usize) -> f64 {
    let scale = Standardiser::fit(train);
    let d = train.p + 1;
    let label = |t: &Table| -> Vec<usize> { t.y.iter().map(|&v| v as usize).collect() };
    let b = scale.design(test);
    let theta = if classes == 2 {
        // Binary: a symmetric two-class fit with penalty |w0|^2/2 + |w1|^2/2 has
        // w1 - w0 = v with |v|^2/4, so the plain binary fit is done directly.
        binary_fit(&scale.design(train), &label(train), d)
    } else {
        softmax_fit(&scale.design(train), &label(train), d, classes)
    };
    let mut loss = 0.0;
    for (row, &y) in b.chunks(d).zip(&label(test)) {
        let prob = if classes == 2 {
            let p1 = 1.0 / (1.0 + (-dot(row, &theta)).exp());
            vec![1.0 - p1, p1]
        } else {
            let logit: Vec<f64> = (0..classes)
                .map(|k| dot(row, &theta[k * d..(k + 1) * d]))
                .collect();
            probabilities(&logit)
        };
        loss -= prob[y].clamp(1e-5, 1.0).ln();
    }
    loss / test.rows() as f64
}

fn binary_fit(a: &[f64], y: &[usize], d: usize) -> Vec<f64> {
    let mut theta = vec![0.0; d];
    let objective = |theta: &[f64]| -> f64 {
        let mut total = 0.0;
        for (row, &label) in a.chunks(d).zip(y) {
            let e = dot(row, theta);
            total += label as f64 * e - (e.max(0.0) + (-e.abs()).exp().ln_1p());
        }
        total - 0.5 * theta[1..].iter().map(|t| t * t).sum::<f64>()
    };
    let mut best = objective(&theta);
    for _ in 0..30 {
        let (mut grad, mut hess) = (vec![0.0; d], vec![0.0; d * d]);
        for (row, &label) in a.chunks(d).zip(y) {
            let prob = 1.0 / (1.0 + (-dot(row, &theta)).exp());
            let w = (prob * (1.0 - prob)).max(1e-12);
            for u in 0..d {
                grad[u] += (label as f64 - prob) * row[u];
                for v in 0..d {
                    hess[u * d + v] += w * row[u] * row[v];
                }
            }
        }
        for j in 1..d {
            grad[j] -= theta[j];
            hess[j * d + j] += 1.0;
        }
        let step = solve_spd(&hess, &grad, d);
        let mut scale = 1.0;
        let mut moved = false;
        for _ in 0..20 {
            let trial: Vec<f64> = theta
                .iter()
                .zip(&step)
                .map(|(t, s)| t + scale * s)
                .collect();
            let value = objective(&trial);
            if value >= best {
                moved = step.iter().any(|s| (scale * s).abs() > 1e-10);
                theta = trial;
                best = value;
                break;
            }
            scale *= 0.5;
        }
        if !moved {
            break;
        }
    }
    theta
}

/// Loss of the best constant prediction learned from `train`, on `test`.
pub fn null_loss(train: &Table, test: &Table, classes: usize) -> f64 {
    let n = train.rows() as f64;
    if classes == 0 {
        let mean = train.y.iter().sum::<f64>() / n;
        return test.y.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / test.rows() as f64;
    }
    let mut prior = vec![0.5; classes];
    train.y.iter().for_each(|&y| prior[y as usize] += 1.0);
    let total: f64 = prior.iter().sum();
    let loss: f64 = test
        .y
        .iter()
        .map(|&y| -(prior[y as usize] / total).ln())
        .sum();
    loss / test.rows() as f64
}

/// Test loss of the plain model of the task (`classes` 0 means regression).
pub fn test_loss(train: &Table, test: &Table, classes: usize) -> f64 {
    if classes == 0 {
        ridge_test_mse(train, test)
    } else {
        logistic_test_loss(train, test, classes)
    }
}

/// Retention `R = (L0 - L_S) / (L0 - L_A)`.
pub fn retention(null: f64, real: f64, synthetic: f64) -> f64 {
    (null - synthetic) / (null - real)
}

pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa * sbb > 0.0 {
        sab / (saa * sbb).sqrt()
    } else {
        0.0
    }
}

/// Largest absolute difference between the pairwise feature correlations of two tables.
pub fn max_corr_gap(a: &Table, b: &Table) -> f64 {
    let (ca, cb): (Vec<_>, Vec<_>) = (0..a.p).map(|j| (a.column(j), b.column(j))).unzip();
    let mut worst = 0.0f64;
    for j in 0..a.p {
        for k in j + 1..a.p {
            worst = worst.max((pearson(&ca[j], &ca[k]) - pearson(&cb[j], &cb[k])).abs());
        }
    }
    worst
}
