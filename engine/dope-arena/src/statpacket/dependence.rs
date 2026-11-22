//! Dependence: tie-aware normal scores, within-class latent correlation,
//! principal-axis loadings whose ranks 4 / 8 / 12 are nested prefixes, and
//! class factor means.
//!
//! Latent-normal convention. Every column is read as a monotone function of a
//! standard normal latent. A block of tied rows covering the probability
//! interval `(a, b)` scores the conditional mean of the latent on that
//! interval, `(pdf(ppf(a)) - pdf(ppf(b))) / (b - a)`; an untied row scores
//! `ppf((rank - 0.5) / n)`. Scores of a tied column have less than unit
//! variance, which attenuates every correlation and class mean it enters by
//! `kappa_j = sd(scores_j) / sd(untied scores)`; the estimates are divided by
//! it, so the sampler, which discretises a unit-variance latent, reproduces
//! both the level masses and the observed association.

use super::packet::{MAX_RANK, SUPPORT_K};
use dope_core::stats::{norm_pdf, norm_ppf, sym_eigen};

/// Loading value = code / 127.
pub const LOADING_SCALE: f64 = 127.0;
/// Class factor mean value = code * 6 / 127.
pub const MEAN_STEP: f64 = 6.0 / 127.0;
/// Uniqueness floor: `psi_j = max(0.02, 1 - sum_k l_jk^2)`.
pub const PSI_FLOOR: f64 = 0.02;
/// Class means are shrunk so that at least this share of a column's latent variance stays within classes.
pub const WITHIN_FLOOR: f64 = 0.1;
/// Ridge of the per-factor class-mean projection.
pub const MEAN_RIDGE: f64 = 1e-3;
/// Principal-axis iteration: passes, stopping tolerance and communality cap.
pub const AXIS_PASSES: usize = 10;
pub const AXIS_TOLERANCE: f64 = 1e-3;
pub const COMMUNALITY_MAX: f64 = 1.0 - PSI_FLOOR;

/// Quantised S4/S6/S9 content: `P x 12` loadings and `classes x 12` factor means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dependence {
    pub loading: Vec<i8>,
    pub class_mean: Vec<i8>,
}

fn pdf_at(u: f64) -> f64 {
    if u <= 0.0 || u >= 1.0 {
        0.0
    } else {
        norm_pdf(norm_ppf(u))
    }
}

/// Tie-aware normal scores of one column (see the module text).
pub fn tie_scores(col: &[f64]) -> Vec<f64> {
    let n = col.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| col[a].total_cmp(&col[b]).then(a.cmp(&b)));
    let mut out = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && col[order[j + 1]] == col[order[i]] {
            j += 1;
        }
        let (lo, hi) = (i as f64 / n as f64, (j + 1) as f64 / n as f64);
        let score = if i == j {
            norm_ppf((i as f64 + 0.5) / n as f64)
        } else {
            (pdf_at(lo) - pdf_at(hi)) / (hi - lo)
        };
        for &row in &order[i..=j] {
            out[row] = score;
        }
        i = j + 1;
    }
    out
}

/// Variance of the scores of `n` untied rows.
pub fn untied_score_variance(n: usize) -> f64 {
    let sum: f64 = (0..n)
        .map(|i| norm_ppf((i as f64 + 0.5) / n as f64).powi(2))
        .sum();
    sum / n as f64
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn code_i8(v: f64) -> i8 {
    v.round().clamp(-127.0, 127.0) as i8
}

/// Standardised scores per column and the attenuation `kappa` (0 for a constant column).
fn standard_scores(x: &[f64], n: usize, p: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let reference = untied_score_variance(n);
    let mut scores = Vec::with_capacity(p);
    let mut kappa = vec![0.0; p];
    for (j, k) in kappa.iter_mut().enumerate() {
        let col: Vec<f64> = (0..n).map(|i| x[i * p + j]).collect();
        let mut s = tie_scores(&col);
        let mean = s.iter().sum::<f64>() / n as f64;
        s.iter_mut().for_each(|v| *v -= mean);
        let var = dot(&s, &s) / n as f64;
        if var > 1e-12 {
            let sd = var.sqrt();
            s.iter_mut().for_each(|v| *v /= sd);
            *k = (var / reference).sqrt().min(1.0);
        } else {
            s.fill(0.0);
        }
        scores.push(s);
    }
    (scores, kappa)
}

/// Latent class means per column in total-SD units (`classes x P`, zero for a
/// class with fewer than 10 rows), shrunk to respect [`WITHIN_FLOOR`], and the
/// class shares.
fn class_means(
    scores: &[Vec<f64>],
    kappa: &[f64],
    classes: &[u16],
    count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (n, p) = (classes.len(), scores.len());
    let mut rows = vec![0usize; count];
    for &c in classes {
        rows[c as usize] += 1;
    }
    let share: Vec<f64> = rows.iter().map(|&r| r as f64 / n as f64).collect();
    let mut mu = vec![0.0; count * p];
    for (j, s) in scores.iter().enumerate() {
        if kappa[j] <= 0.0 {
            continue;
        }
        for (&c, v) in classes.iter().zip(s) {
            mu[c as usize * p + j] += v;
        }
        let mut between = 0.0;
        for c in 0..count {
            let cell = &mut mu[c * p + j];
            *cell = if rows[c] >= SUPPORT_K {
                *cell / rows[c] as f64 / kappa[j]
            } else {
                0.0
            };
            between += share[c] * *cell * *cell;
        }
        if between > 1.0 - WITHIN_FLOOR {
            let shrink = ((1.0 - WITHIN_FLOOR) / between).sqrt();
            (0..count).for_each(|c| mu[c * p + j] *= shrink);
        }
    }
    (mu, share)
}

/// Principal axes of a correlation matrix for `rank` factors: eigenvalues
/// (descending) and eigenvectors (columns) of the matrix whose diagonal holds
/// the communalities. With `p <= rank` the diagonal stays 1 (principal
/// components reproduce the matrix). Otherwise communalities start at the
/// largest absolute correlation of each column and are replaced by the
/// `rank`-factor reproduction, capped at 0.98, until they move by less than
/// 1e-3 or 10 passes are done.
fn principal_axes(corr: &[f64], p: usize, rank: usize) -> (Vec<f64>, Vec<f64>) {
    if p <= rank {
        return sym_eigen(corr, p);
    }
    let mut reduced = corr.to_vec();
    let mut communality: Vec<f64> = (0..p)
        .map(|j| {
            (0..p)
                .filter(|&k| k != j)
                .map(|k| corr[j * p + k].abs())
                .fold(0.0, f64::max)
        })
        .collect();
    let mut axes = (Vec::new(), Vec::new());
    for _ in 0..AXIS_PASSES {
        for (j, h) in communality.iter().enumerate() {
            reduced[j * p + j] = h.min(COMMUNALITY_MAX);
        }
        axes = sym_eigen(&reduced, p);
        let mut moved = 0.0f64;
        for (j, h) in communality.iter_mut().enumerate() {
            let fitted: f64 = (0..rank)
                .map(|f| axes.0[f].max(0.0) * axes.1[j * p + f].powi(2))
                .sum();
            let fitted = fitted.min(COMMUNALITY_MAX);
            moved = moved.max((fitted - *h).abs());
            *h = fitted;
        }
        if moved < AXIS_TOLERANCE {
            break;
        }
    }
    axes
}

/// Fits loadings and class factor means from unit-space rows (`x` row-major
/// `n x p`). `classes` is `Some((class id per row, class count))` for binary
/// and multiclass tasks, `None` for a single regime.
pub fn fit_dependence(
    x: &[f64],
    n: usize,
    p: usize,
    classes: Option<(&[u16], usize)>,
) -> Dependence {
    let (scores, kappa) = standard_scores(x, n, p);
    // Latent total correlation, disattenuated and clipped; a constant column is isolated with a zero diagonal.
    let mut corr = vec![0.0; p * p];
    for j in 0..p {
        if kappa[j] <= 0.0 {
            continue;
        }
        corr[j * p + j] = 1.0;
        for k in j + 1..p {
            if kappa[k] > 0.0 {
                let c = dot(&scores[j], &scores[k]) / n as f64;
                let rho = (c / (kappa[j] * kappa[k])).clamp(-1.0, 1.0);
                corr[j * p + k] = rho;
                corr[k * p + j] = rho;
            }
        }
    }
    let regimes = classes.map_or(0, |(_, count)| count);
    let mut mu = vec![0.0; regimes * p];
    if let Some((ids, count)) = classes {
        let (means, share) = class_means(&scores, &kappa, ids, count);
        // Within-class share of each column's latent variance.
        let within: Vec<f64> = (0..p)
            .map(|j| {
                let b: f64 = (0..count)
                    .map(|c| share[c] * means[c * p + j].powi(2))
                    .sum();
                1.0 - b
            })
            .collect();
        for j in 0..p {
            for k in j + 1..p {
                let b: f64 = (0..count)
                    .map(|c| share[c] * means[c * p + j] * means[c * p + k])
                    .sum();
                let rho = ((corr[j * p + k] - b) / (within[j] * within[k]).sqrt()).clamp(-1.0, 1.0);
                corr[j * p + k] = rho;
                corr[k * p + j] = rho;
            }
        }
        // Class means in within-class SD units, the scale the sampler works in.
        for c in 0..count {
            for j in 0..p {
                mu[c * p + j] = means[c * p + j] / within[j].sqrt();
            }
        }
    }
    let rank = MAX_RANK.min(p);
    let (values, vectors) = principal_axes(&corr, p, rank);
    let mut loading = vec![0i8; p * MAX_RANK];
    for f in 0..rank {
        let sd = values[f].max(0.0).sqrt();
        // Sign convention: the entry of largest magnitude is positive (lowest column on ties).
        let mut top = 0;
        for j in 1..p {
            if vectors[j * p + f].abs() > vectors[top * p + f].abs() {
                top = j;
            }
        }
        let sign = if vectors[top * p + f] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for j in 0..p {
            loading[j * MAX_RANK + f] = code_i8(sign * sd * vectors[j * p + f] * LOADING_SCALE);
        }
    }
    // One ridge projection per factor against the quantised loadings keeps the ranks nested.
    let mut class_mean = vec![0i8; regimes * MAX_RANK];
    for c in 0..regimes {
        for f in 0..rank {
            let (mut num, mut den) = (0.0, MEAN_RIDGE);
            for j in 0..p {
                let l = loading[j * MAX_RANK + f] as f64 / LOADING_SCALE;
                num += l * mu[c * p + j];
                den += l * l;
            }
            class_mean[c * MAX_RANK + f] = code_i8(num / den / MEAN_STEP);
        }
    }
    Dependence {
        loading,
        class_mean,
    }
}
