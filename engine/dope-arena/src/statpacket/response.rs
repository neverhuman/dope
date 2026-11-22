//! The response of the FIT rows in the form the solvers take, with the
//! support floor applied: a class (or a multilabel side) carried by fewer than
//! 10 FIT rows takes no part in any fit and gets no parameters.
//!
//! Wire parametrisation of the outputs: regression 1 output; binary 1 output,
//! the logit of class 1; multiclass `K - 1` outputs, output `c - 1` being the
//! logit of class `c` against class 0; multilabel one logit per label.

use super::glm::{logistic, ridge, sigmoid, softmax, softmax_row, Basis, SKIP_ROW};
use super::packet::{Packet, Task, SUPPORT_K};
use super::StatError;

/// Target of the FIT rows.
#[derive(Clone, Copy, Debug)]
pub enum Response<'a> {
    /// Standardised regression target, one value in [-6, 6] per row.
    Regression(&'a [f64]),
    /// Binary or multiclass: one class id per row.
    Classes(&'a [u16]),
    /// Multilabel: row-major `n x K` cells, each 0 or 1.
    Multilabel(&'a [u8]),
}

/// The response in the form the solvers take.
pub enum Resp {
    Gaussian(Vec<f64>),
    /// One 0/1 vector per output; `None` if either side has fewer than 10 rows.
    Logit(Vec<Option<Vec<f64>>>),
    /// Supported classes (ascending), each row's position among them (or
    /// [`SKIP_ROW`]) and the output count `K - 1`.
    Softmax {
        classes: Vec<usize>,
        row: Vec<usize>,
        outputs: usize,
    },
}

fn supported(y: Vec<f64>) -> Option<Vec<f64>> {
    let ones = y.iter().filter(|&&v| v == 1.0).count();
    (ones >= SUPPORT_K && y.len() - ones >= SUPPORT_K).then_some(y)
}

impl Resp {
    pub fn new(packet: &Packet, response: Response, n: usize) -> Result<Self, StatError> {
        let k = packet.k as usize;
        let bad = Err(StatError::Invalid("target does not match the task"));
        match (packet.task, response) {
            (Task::Regression, Response::Regression(y)) if y.len() == n => {
                if y.iter().any(|v| !v.is_finite() || v.abs() > 6.0) {
                    return Err(StatError::Invalid(
                        "regression target is not a finite standardised value",
                    ));
                }
                Ok(Self::Gaussian(y.to_vec()))
            }
            (Task::Binary, Response::Classes(ids)) if ids.len() == n => {
                if ids.iter().any(|&c| c > 1) {
                    return bad;
                }
                let y = ids.iter().map(|&c| c as f64).collect();
                Ok(Self::Logit(vec![supported(y)]))
            }
            (Task::Multiclass, Response::Classes(ids)) if ids.len() == n => {
                if ids.iter().any(|&c| c as usize >= k) {
                    return bad;
                }
                let mut count = vec![0usize; k];
                ids.iter().for_each(|&c| count[c as usize] += 1);
                let mut classes: Vec<usize> = (0..k).filter(|&c| count[c] >= SUPPORT_K).collect();
                if classes.len() < 2 {
                    classes.clear();
                }
                let place = |c: u16| classes.iter().position(|&s| s == c as usize);
                let row = ids.iter().map(|&c| place(c).unwrap_or(SKIP_ROW)).collect();
                Ok(Self::Softmax {
                    classes,
                    row,
                    outputs: k - 1,
                })
            }
            (Task::Multilabel, Response::Multilabel(cells)) if cells.len() == n * k => {
                if cells.iter().any(|&c| c > 1) {
                    return bad;
                }
                let label = |l: usize| (0..n).map(|i| cells[i * k + l] as f64).collect();
                Ok(Self::Logit((0..k).map(|l| supported(label(l))).collect()))
            }
            _ => bad,
        }
    }

    pub fn outputs(&self) -> usize {
        match self {
            Self::Gaussian(_) => 1,
            Self::Logit(labels) => labels.len(),
            Self::Softmax { outputs, .. } => *outputs,
        }
    }

    /// Probabilities of the supported classes from one row of wire logits
    /// (class 0 has logit 0, class `c` has `eta[c - 1]`).
    pub fn class_probs(classes: &[usize], eta: &[f64], out: &mut [f64]) {
        let logit: Vec<f64> = classes
            .iter()
            .map(|&c| if c == 0 { 0.0 } else { eta[c - 1] })
            .collect();
        softmax_row(&logit, out);
    }

    /// Response residuals `y - mean(eta)`, row-major `n x outputs`; rows and
    /// outputs without support are zero. `eta` already includes the temperature.
    pub fn residuals(&self, eta: &[f64]) -> Vec<f64> {
        let o = self.outputs();
        let mut out = vec![0.0; eta.len()];
        match self {
            Self::Gaussian(y) => {
                out.iter_mut()
                    .zip(y.iter().zip(eta))
                    .for_each(|(r, (y, e))| *r = y - e);
            }
            Self::Logit(labels) => {
                for (k, y) in labels.iter().enumerate() {
                    for (i, y) in y.iter().flatten().enumerate() {
                        out[i * o + k] = y - sigmoid(eta[i * o + k]);
                    }
                }
            }
            Self::Softmax { classes, row, .. } => {
                let mut prob = vec![0.0; classes.len()];
                for (i, &at) in row.iter().enumerate().filter(|(_, &at)| at != SKIP_ROW) {
                    Self::class_probs(classes, &eta[i * o..(i + 1) * o], &mut prob);
                    for (s, &class) in classes.iter().enumerate().filter(|(_, &c)| c > 0) {
                        out[i * o + class - 1] = (s == at) as u8 as f64 - prob[s];
                    }
                }
            }
        }
        out
    }

    /// Starting intercept of every output.
    pub fn intercept_start(&self) -> Vec<f64> {
        match self {
            Self::Logit(labels) => labels
                .iter()
                .map(|y| match y {
                    Some(y) => {
                        let rate = y.iter().sum::<f64>() / y.len() as f64;
                        (rate / (1.0 - rate)).ln()
                    }
                    None => 0.0,
                })
                .collect(),
            _ => vec![0.0; self.outputs()],
        }
    }
}

/// Coefficients `outputs x dim` (wire parametrisation) of one additive block
/// fitted on top of `offset` (row-major `n x outputs`, temperature included).
/// `start` gives the starting value of coefficient 0 of every output.
pub fn fit_block(basis: &dyn Basis, resp: &Resp, offset: &[f64], start: &[f64]) -> Vec<f64> {
    let (n, d, o) = (basis.rows(), basis.dim(), resp.outputs());
    let mut out = vec![0.0; o * d];
    if d == 0 {
        return out;
    }
    match resp {
        Resp::Gaussian(y) => {
            let resid: Vec<f64> = y.iter().zip(offset).map(|(y, e)| y - e).collect();
            out = ridge(basis, &resid);
        }
        Resp::Logit(labels) => {
            for (k, y) in labels.iter().enumerate() {
                if let Some(y) = y {
                    let column: Vec<f64> = (0..n).map(|i| offset[i * o + k]).collect();
                    let mut init = vec![0.0; d];
                    init[0] = start[k];
                    out[k * d..(k + 1) * d].copy_from_slice(&logistic(basis, &column, y, &init));
                }
            }
        }
        Resp::Softmax { classes, row, .. } => {
            let c = classes.len();
            if c < 2 {
                return out;
            }
            let mut wire = vec![0.0; n * c];
            for i in 0..n {
                for (s, &class) in classes.iter().enumerate().filter(|(_, &class)| class > 0) {
                    wire[i * c + s] = offset[i * o + class - 1];
                }
            }
            let symmetric = softmax(basis, &wire, row, c, &vec![0.0; c * d]);
            // Wire outputs are logits against the lowest supported class.
            for (s, &class) in classes.iter().enumerate().skip(1) {
                for j in 0..d {
                    out[(class - 1) * d + j] = symmetric[s * d + j] - symmetric[j];
                }
            }
        }
    }
    out
}

/// Slope and curvature in `T` of the log-likelihood of `softmax(T * eta)`.
pub fn temperature_slope(resp: &Resp, eta: &[f64], t: f64) -> (f64, f64) {
    let o = resp.outputs();
    let (mut slope, mut curve) = (0.0, 0.0);
    match resp {
        Resp::Gaussian(_) => {}
        Resp::Logit(labels) => {
            for (k, y) in labels.iter().enumerate() {
                for (i, y) in y.iter().flatten().enumerate() {
                    let e = eta[i * o + k];
                    let prob = sigmoid(t * e);
                    slope += (y - prob) * e;
                    curve += prob * (1.0 - prob) * e * e;
                }
            }
        }
        Resp::Softmax { classes, row, .. } => {
            let mut prob = vec![0.0; classes.len()];
            let mut scaled = vec![0.0; o];
            for (i, &at) in row.iter().enumerate().filter(|(_, &at)| at != SKIP_ROW) {
                let e = &eta[i * o..(i + 1) * o];
                scaled.iter_mut().zip(e).for_each(|(s, e)| *s = t * e);
                Resp::class_probs(classes, &scaled, &mut prob);
                let of = |s: usize| {
                    if classes[s] == 0 {
                        0.0
                    } else {
                        e[classes[s] - 1]
                    }
                };
                let m1: f64 = (0..prob.len()).map(|s| prob[s] * of(s)).sum();
                let m2: f64 = (0..prob.len()).map(|s| prob[s] * of(s) * of(s)).sum();
                slope += of(at) - m1;
                curve += m2 - m1 * m1;
            }
        }
    }
    (slope, curve)
}
