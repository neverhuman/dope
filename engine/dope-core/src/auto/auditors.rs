//! Quick auditors of the `--format auto` internal check: an L2 linear model (ridge, logistic
//! or softmax) on standardized features and a small histogram GBDT (`eval::gbdt`).
//! Everything is sequential or reduces deterministically, so results do not depend on the
//! number of threads.

use crate::eval::{gbdt, linear, Matrix};
use crate::schema::{CategoricalSpec, Kind, ModelData, Schema};
use crate::stats::cholesky;
use crate::target::Head;

/// Auditor families; the check scores the minimum retention over the usable ones.
pub(crate) const AUDITORS: [&str; 2] = ["linear", "gbdt"];
/// Standardized numerics are clipped here so a stray synthetic tail cannot dominate.
const Z_CLIP: f64 = 8.0;
const SOFTMAX_ITERS: usize = 100;

enum Column {
    Numeric { mean: f64, inv_sd: f64 },
    Categorical(CategoricalSpec),
    Skip,
}

/// Design matrix of a model-space table: numerics standardized with the real training
/// part's mean and deviation (missing = mean), categoricals one-hot.
pub(crate) struct Featurizer {
    columns: Vec<Column>,
    width: usize,
}

impl Featurizer {
    pub(crate) fn fit(schema: &Schema, real: &ModelData) -> Self {
        let mut width = 0;
        let columns = schema
            .features
            .iter()
            .zip(&real.x)
            .map(|(f, x)| match &f.kind {
                Kind::Numeric(_) => {
                    let v: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
                    let n = v.len().max(1) as f64;
                    let mean = v.iter().sum::<f64>() / n;
                    let sd = (v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n).sqrt();
                    width += 1;
                    Column::Numeric {
                        mean,
                        inv_sd: if sd > 1e-12 { 1.0 / sd } else { 0.0 },
                    }
                }
                Kind::Categorical(spec) if spec.levels.len() >= 2 => {
                    width += spec.levels.len();
                    Column::Categorical(spec.clone())
                }
                Kind::Categorical(_) => Column::Skip,
            })
            .collect();
        Self { columns, width }
    }

    /// Rows with an observed target, thinned by a fixed stride to at most `cap`.
    pub(crate) fn matrix(&self, data: &ModelData, cap: usize) -> (Matrix, Vec<f64>) {
        let rows: Vec<usize> = (0..data.rows())
            .filter(|&r| data.y[r].is_finite())
            .collect();
        let stride = rows.len().div_ceil(cap.max(1)).max(1);
        let rows: Vec<usize> = rows.into_iter().step_by(stride).collect();
        let w = self.width;
        let mut m = vec![0.0; rows.len() * w];
        for (out, &r) in m.chunks_mut(w.max(1)).zip(&rows) {
            let mut at = 0;
            for (col, x) in self.columns.iter().zip(&data.x) {
                let v = x[r];
                match col {
                    Column::Numeric { mean, inv_sd } => {
                        if v.is_finite() {
                            out[at] = ((v - mean) * inv_sd).clamp(-Z_CLIP, Z_CLIP);
                        }
                        at += 1;
                    }
                    Column::Categorical(spec) => {
                        if v.is_finite() {
                            out[at + spec.level_from_code(v)] = 1.0;
                        }
                        at += spec.levels.len();
                    }
                    Column::Skip => {}
                }
            }
        }
        let y = rows.iter().map(|&r| data.y[r]).collect();
        (
            Matrix {
                rows: rows.len(),
                cols: w,
                data: m,
            },
            y,
        )
    }
}

/// Predictions of the best constant: the training mean, or smoothed class frequencies.
pub(crate) fn null_predictions(head: Head, y: &[f64], n: usize) -> Vec<f64> {
    match head {
        Head::Regression | Head::Binary => {
            vec![y.iter().sum::<f64>() / y.len().max(1) as f64; n]
        }
        Head::Multiclass(k) => {
            let p = class_priors(y, k);
            (0..n).flat_map(|_| p.iter().copied()).collect()
        }
    }
}

fn class_priors(y: &[f64], k: usize) -> Vec<f64> {
    let mut c = vec![0.5; k];
    y.iter().for_each(|t| c[(*t as usize).min(k - 1)] += 1.0);
    let s: f64 = c.iter().sum();
    c.into_iter().map(|v| v / s).collect()
}

/// Fit auditor `name` on `(x, y)` and predict `test`: values, P(class 1) or row-major
/// class probabilities, as `eval::prediction_loss` expects.
pub(crate) fn fit_predict(
    name: &str,
    head: Head,
    x: &Matrix,
    y: &[f64],
    test: &Matrix,
    seed: u64,
) -> Vec<f64> {
    match (name, head) {
        ("linear", Head::Multiclass(k)) => softmax(x, y, k, test),
        ("linear", _) => linear::fit_predict(x, y, head, test),
        _ => {
            let params = gbdt::Params {
                rounds: 150,
                learning_rate: 0.1,
                max_depth: 4,
                min_leaf: 10,
                lambda: 1.0,
                bins: 32,
                patience: 20,
                seed,
                group_duplicates: true,
            };
            gbdt::fit_predict(x, y, head, test, &params)
        }
    }
}

/// Design value `j` of row `r` with a leading intercept column.
fn at(m: &Matrix, r: usize, j: usize) -> f64 {
    if j == 0 {
        1.0
    } else {
        m.at(r, j - 1)
    }
}

/// L2 multinomial logistic regression by bounded-Hessian MM steps (Böhning): the Hessian is
/// at most `½ XᵀX` per class, so one factorization serves every iteration and the
/// penalized log-loss never increases.
fn softmax(x: &Matrix, y: &[f64], k: usize, test: &Matrix) -> Vec<f64> {
    let (n, p) = (x.rows, x.cols + 1);
    let mut g = vec![0.0; p * p];
    for r in 0..n {
        for i in 0..p {
            let a = at(x, r, i);
            if a != 0.0 {
                for j in i..p {
                    g[i * p + j] += a * at(x, r, j);
                }
            }
        }
    }
    let avg = (1..p).map(|i| g[i * p + i]).sum::<f64>() / (p - 1).max(1) as f64;
    let lam = 1e-3 * avg.max(1e-12) + 1e-9;
    let mut b = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            b[i * p + j] = 0.5 * g[i * p + j];
            b[j * p + i] = b[i * p + j];
        }
        b[i * p + i] += if i == 0 { 1e-6 * n.max(1) as f64 } else { lam };
    }
    let mut w = vec![0.0; p * k];
    for (c, prior) in class_priors(y, k).into_iter().enumerate() {
        w[c] = prior.ln();
    }
    let Some(l) = cholesky(&b, p) else {
        return predict_softmax(&w, k, test);
    };
    let mut grad = vec![0.0; p * k];
    let mut prob = vec![0.0; k];
    for _ in 0..SOFTMAX_ITERS {
        grad.iter_mut().for_each(|v| *v = 0.0);
        for r in 0..n {
            softmax_row(&w, k, x, r, &mut prob);
            prob[(y[r] as usize).min(k - 1)] -= 1.0;
            for j in 0..p {
                let a = at(x, r, j);
                if a != 0.0 {
                    for c in 0..k {
                        grad[j * k + c] += a * prob[c];
                    }
                }
            }
        }
        for j in 1..p {
            for c in 0..k {
                grad[j * k + c] += lam * w[j * k + c];
            }
        }
        let mut largest = 0.0f64;
        for c in 0..k {
            let rhs: Vec<f64> = (0..p).map(|j| grad[j * k + c]).collect();
            for (j, s) in chol_solve(&l, &rhs, p).into_iter().enumerate() {
                w[j * k + c] -= s;
                largest = largest.max(s.abs());
            }
        }
        if !largest.is_finite() || largest < 1e-7 {
            break;
        }
    }
    predict_softmax(&w, k, test)
}

fn softmax_row(w: &[f64], k: usize, m: &Matrix, r: usize, out: &mut [f64]) {
    for (c, o) in out.iter_mut().enumerate() {
        *o = (0..m.cols + 1).map(|j| at(m, r, j) * w[j * k + c]).sum();
    }
    let max = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0;
    for o in out.iter_mut() {
        *o = (*o - max).exp();
        s += *o;
    }
    out.iter_mut().for_each(|o| *o /= s);
}

fn predict_softmax(w: &[f64], k: usize, test: &Matrix) -> Vec<f64> {
    let mut out = vec![0.0; test.rows * k];
    for (r, row) in out.chunks_mut(k).enumerate() {
        softmax_row(w, k, test, r, row);
    }
    out
}

/// Solve `L Lᵀ x = b` for a lower Cholesky factor.
fn chol_solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i * n + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::prediction_loss;

    #[test]
    fn softmax_learns_separable_classes_better_than_the_prior() {
        let (n, k) = (300, 3);
        let mut data = Vec::new();
        let mut y = Vec::new();
        for r in 0..n {
            let c = r % k;
            let jitter = ((r * 7919) % 100) as f64 / 100.0 - 0.5;
            data.extend([
                c as f64 * 2.0 + jitter,
                (c == 1) as u8 as f64 + 0.3 * jitter,
            ]);
            y.push(c as f64);
        }
        let x = Matrix {
            rows: n,
            cols: 2,
            data,
        };
        let pred = softmax(&x, &y, k, &x);
        assert_eq!(pred.len(), n * k);
        assert!(pred
            .chunks(k)
            .all(|p| (p.iter().sum::<f64>() - 1.0).abs() < 1e-9));
        let head = Head::Multiclass(k);
        let fitted = prediction_loss(head, &pred, &y);
        let null = prediction_loss(head, &null_predictions(head, &y, n), &y);
        assert!(fitted < 0.5 * null, "softmax {fitted} vs prior {null}");
        assert_eq!(pred, softmax(&x, &y, k, &x), "deterministic");
    }
}
