//! Penalised least-squares and logistic solvers of the target fit.
//!
//! Every solver maximises a penalised log-likelihood over the coefficients of
//! a [`Basis`] (a design matrix with one ridge weight per coefficient) on top
//! of a fixed per-row offset. Step counts are constants, the line search is a
//! fixed halving schedule and every sum runs in row order, so identical inputs
//! give identical coefficients.
//!
//! - [`ridge`]: one solve of `(A'A + D) c = A'y`.
//! - [`logistic`]: Newton steps (iteratively reweighted least squares); the
//!   Hessian changes with the fitted probabilities, so it is factorised again
//!   at every step.
//! - [`softmax`]: symmetric multinomial model with one coefficient vector per
//!   class. Each step solves one system per class with the weights
//!   `2 p (1 - p)`: `diag(p) - p p'` is dominated by `2 diag(p (1 - p))`, so
//!   the per-class blocks bound the full Hessian and no `classes x dim` system
//!   is ever formed. Steps are centred over classes (the optimum is).

use dope_core::stats::solve_spd;

/// Newton steps of [`logistic`].
pub const NEWTON_STEPS: usize = 12;
/// Block steps of [`softmax`].
pub const SOFTMAX_STEPS: usize = 40;
/// Step halvings tried before a step is abandoned.
pub const HALVINGS: usize = 8;
/// A solver stops early once no coefficient moves by this much.
pub const STEP_TOLERANCE: f64 = 1e-9;
const WEIGHT_FLOOR: f64 = 1e-10;
/// Marks a row that takes no part in a [`softmax`] fit.
pub const SKIP_ROW: usize = usize::MAX;
/// Marks a cell of a [`OneHot`] basis without a coefficient.
pub const NO_COEF: u32 = u32::MAX;

pub trait Basis {
    fn rows(&self) -> usize;
    fn dim(&self) -> usize;
    /// Ridge weight of every coefficient (0 leaves it free).
    fn penalty(&self) -> &[f64];
    /// `A' diag(w) A` (`w = 1` without weights), row-major `dim x dim`.
    fn gram(&self, w: Option<&[f64]>) -> Vec<f64>;
    /// `A' r`.
    fn project(&self, r: &[f64]) -> Vec<f64>;
    /// `A c`.
    fn apply(&self, c: &[f64]) -> Vec<f64>;
}

/// Row-major `n x d` design.
pub struct Dense {
    pub a: Vec<f64>,
    pub n: usize,
    pub d: usize,
    pub penalty: Vec<f64>,
}

impl Basis for Dense {
    fn rows(&self) -> usize {
        self.n
    }
    fn dim(&self) -> usize {
        self.d
    }
    fn penalty(&self) -> &[f64] {
        &self.penalty
    }
    fn gram(&self, w: Option<&[f64]>) -> Vec<f64> {
        let d = self.d;
        let mut g = vec![0.0; d * d];
        for (i, row) in self.a.chunks(d).enumerate() {
            let wi = w.map_or(1.0, |w| w[i]);
            for a in 0..d {
                let va = wi * row[a];
                if va != 0.0 {
                    let line = &mut g[a * d + a..(a + 1) * d];
                    line.iter_mut()
                        .zip(&row[a..])
                        .for_each(|(g, b)| *g += va * b);
                }
            }
        }
        for a in 0..d {
            for b in 0..a {
                g[a * d + b] = g[b * d + a];
            }
        }
        g
    }
    fn project(&self, r: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.d];
        for (row, &ri) in self.a.chunks(self.d).zip(r) {
            out.iter_mut().zip(row).for_each(|(o, a)| *o += a * ri);
        }
        out
    }
    fn apply(&self, c: &[f64]) -> Vec<f64> {
        self.a
            .chunks(self.d)
            .map(|row| row.iter().zip(c).map(|(a, c)| a * c).sum())
            .collect()
    }
}

/// Indicator design: every row has at most one active coefficient per group.
pub struct OneHot {
    /// Row-major `n x groups` coefficient ids ([`NO_COEF`] for none).
    pub id: Vec<u32>,
    pub n: usize,
    pub groups: usize,
    pub d: usize,
    pub penalty: Vec<f64>,
}

impl OneHot {
    fn active(row: &[u32]) -> impl Iterator<Item = usize> + '_ {
        row.iter().filter(|&&c| c != NO_COEF).map(|&c| c as usize)
    }
}

impl Basis for OneHot {
    fn rows(&self) -> usize {
        self.n
    }
    fn dim(&self) -> usize {
        self.d
    }
    fn penalty(&self) -> &[f64] {
        &self.penalty
    }
    fn gram(&self, w: Option<&[f64]>) -> Vec<f64> {
        let d = self.d;
        let mut g = vec![0.0; d * d];
        for (i, row) in self.id.chunks(self.groups.max(1)).enumerate() {
            let wi = w.map_or(1.0, |w| w[i]);
            for a in Self::active(row) {
                for b in Self::active(row) {
                    g[a * d + b] += wi;
                }
            }
        }
        g
    }
    fn project(&self, r: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.d];
        for (row, &ri) in self.id.chunks(self.groups.max(1)).zip(r) {
            Self::active(row).for_each(|a| out[a] += ri);
        }
        out
    }
    fn apply(&self, c: &[f64]) -> Vec<f64> {
        self.id
            .chunks(self.groups.max(1))
            .map(|row| Self::active(row).map(|a| c[a]).sum())
            .collect()
    }
}

fn solve(mut gram: Vec<f64>, penalty: &[f64], rhs: &[f64]) -> Vec<f64> {
    let d = rhs.len();
    for (j, pen) in penalty.iter().enumerate() {
        gram[j * d + j] += pen;
    }
    solve_spd(&gram, rhs, d)
}

fn penalty_term(penalty: &[f64], c: &[f64]) -> f64 {
    0.5 * penalty.iter().zip(c).map(|(p, c)| p * c * c).sum::<f64>()
}

/// `argmin |y - A c|^2 + c' D c`.
pub fn ridge(basis: &dyn Basis, y: &[f64]) -> Vec<f64> {
    if basis.dim() == 0 {
        return Vec::new();
    }
    solve(basis.gram(None), basis.penalty(), &basis.project(y))
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Tries `c + s * delta` for `s = 1, 1/2, ..`; returns the accepted step
/// length, or `None` if no trial reaches the current objective.
fn line_search(objective: impl Fn(f64) -> f64, current: f64) -> Option<(f64, f64)> {
    let mut s = 1.0;
    for _ in 0..=HALVINGS {
        let value = objective(s);
        if value.is_finite() && value >= current {
            return Some((s, value));
        }
        s *= 0.5;
    }
    None
}

/// Binary logistic coefficients: maximises
/// `sum_i y_i e_i - log(1 + exp(e_i)) - c' D c / 2`, `e = offset + A c`,
/// from `init`. `y` holds 0 or 1.
pub fn logistic(basis: &dyn Basis, offset: &[f64], y: &[f64], init: &[f64]) -> Vec<f64> {
    let d = basis.dim();
    if d == 0 {
        return Vec::new();
    }
    let value = |eta: &[f64], c: &[f64]| -> f64 {
        let fit: f64 = eta.iter().zip(y).map(|(e, y)| y * e - softplus(*e)).sum();
        fit - penalty_term(basis.penalty(), c)
    };
    let shifted = |base: &[f64], step: &[f64], s: f64| -> Vec<f64> {
        base.iter().zip(step).map(|(b, d)| b + s * d).collect()
    };
    let mut c = init.to_vec();
    let mut eta = shifted(offset, &basis.apply(&c), 1.0);
    let mut best = value(&eta, &c);
    for _ in 0..NEWTON_STEPS {
        let prob: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();
        let weight: Vec<f64> = prob
            .iter()
            .map(|p| (p * (1.0 - p)).max(WEIGHT_FLOOR))
            .collect();
        let resid: Vec<f64> = y.iter().zip(&prob).map(|(y, p)| y - p).collect();
        let mut grad = basis.project(&resid);
        for (g, (pen, c)) in grad.iter_mut().zip(basis.penalty().iter().zip(&c)) {
            *g -= pen * c;
        }
        let delta = solve(basis.gram(Some(&weight)), basis.penalty(), &grad);
        let along = basis.apply(&delta);
        let trial = |s: f64| value(&shifted(&eta, &along, s), &shifted(&c, &delta, s));
        let Some((s, reached)) = line_search(trial, best) else {
            break;
        };
        c = shifted(&c, &delta, s);
        eta = shifted(&eta, &along, s);
        best = reached;
        if delta.iter().all(|v| (s * v).abs() < STEP_TOLERANCE) {
            break;
        }
    }
    c
}

/// Class probabilities of one row of logits (stable for any finite input).
pub fn softmax_row(logit: &[f64], out: &mut [f64]) {
    let top = logit.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let mut total = 0.0;
    for (o, &l) in out.iter_mut().zip(logit) {
        *o = (l - top).exp();
        total += *o;
    }
    out.iter_mut().for_each(|o| *o /= total);
}

/// Symmetric multinomial coefficients, class-major `classes x dim`: maximises
/// `sum_i log softmax(offset_i + A c)[y_i] - sum_k c_k' D c_k / 2` from `init`.
/// `offset` is row-major `n x classes`; a row with `y = SKIP_ROW` is ignored.
pub fn softmax(
    basis: &dyn Basis,
    offset: &[f64],
    y: &[usize],
    classes: usize,
    init: &[f64],
) -> Vec<f64> {
    let (n, d) = (basis.rows(), basis.dim());
    if d == 0 || classes == 0 {
        return Vec::new();
    }
    // `A c` of every class, row-major `n x classes`.
    let linear = |c: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; n * classes];
        for k in 0..classes {
            let lin = basis.apply(&c[k * d..(k + 1) * d]);
            for (i, v) in lin.into_iter().enumerate() {
                out[i * classes + k] = v;
            }
        }
        out
    };
    let value = |eta: &[f64], c: &[f64]| -> f64 {
        let mut fit = 0.0;
        for (row, &label) in eta.chunks(classes).zip(y) {
            if label != SKIP_ROW {
                let top = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
                let lse = top + row.iter().map(|v| (v - top).exp()).sum::<f64>().ln();
                fit += row[label] - lse;
            }
        }
        let penalty: f64 = c
            .chunks(d)
            .map(|ck| penalty_term(basis.penalty(), ck))
            .sum();
        fit - penalty
    };
    let shifted = |base: &[f64], step: &[f64], s: f64| -> Vec<f64> {
        base.iter().zip(step).map(|(b, d)| b + s * d).collect()
    };
    let mut c = init.to_vec();
    let mut eta = shifted(offset, &linear(&c), 1.0);
    let mut best = value(&eta, &c);
    let (mut prob, mut weight, mut resid) = (vec![0.0; n * classes], vec![0.0; n], vec![0.0; n]);
    for _ in 0..SOFTMAX_STEPS {
        for (row, out) in eta.chunks(classes).zip(prob.chunks_mut(classes)) {
            softmax_row(row, out);
        }
        let mut delta = vec![0.0; classes * d];
        for k in 0..classes {
            for i in 0..n {
                let p = prob[i * classes + k];
                let on = y[i] != SKIP_ROW;
                weight[i] = if on {
                    (2.0 * p * (1.0 - p)).max(WEIGHT_FLOOR)
                } else {
                    0.0
                };
                resid[i] = if on {
                    (y[i] == k) as u8 as f64 - p
                } else {
                    0.0
                };
            }
            let ck = &c[k * d..(k + 1) * d];
            let mut grad = basis.project(&resid);
            for (g, (pen, c)) in grad.iter_mut().zip(basis.penalty().iter().zip(ck)) {
                *g -= pen * c;
            }
            let step = solve(basis.gram(Some(&weight)), basis.penalty(), &grad);
            delta[k * d..(k + 1) * d].copy_from_slice(&step);
        }
        // A common shift of all classes changes no probability and only costs
        // penalty, so the optimum is centred over classes; the step is kept
        // centred too, because the per-class blocks move slowly in that direction.
        for j in 0..d {
            let mean = (0..classes).map(|k| delta[k * d + j]).sum::<f64>() / classes as f64;
            (0..classes).for_each(|k| delta[k * d + j] -= mean);
        }
        let along = linear(&delta);
        let trial = |s: f64| value(&shifted(&eta, &along, s), &shifted(&c, &delta, s));
        let Some((s, reached)) = line_search(trial, best) else {
            break;
        };
        c = shifted(&c, &delta, s);
        eta = shifted(&eta, &along, s);
        best = reached;
        if delta.iter().all(|v| (s * v).abs() < STEP_TOLERANCE) {
            break;
        }
    }
    c
}
