//! Input screening and latent (normal-score) correlation.

use crate::marginal::Marginal;
use crate::stats::{chatterjee_xi, mid_ranks, pearson, spearman};
use crate::target::Head;
use rayon::prelude::*;

/// Rows (deterministic stride) with both values finite, at most `cap`.
fn paired(x: &[f64], y: &[f64], cap: usize) -> (Vec<f64>, Vec<f64>) {
    let idx: Vec<usize> = (0..x.len())
        .filter(|&i| x[i].is_finite() && y[i].is_finite())
        .collect();
    let stride = idx.len().div_ceil(cap.max(1)).max(1);
    idx.iter().step_by(stride).map(|&i| (x[i], y[i])).unzip()
}

/// Association of each feature with the target: max(|Spearman ρ|, ξ) for
/// regression and binary targets (ξ catches non-monotone drivers), and the
/// rank correlation ratio η for multiclass targets.
pub fn associations(x: &[Vec<f64>], y: &[f64], head: Head, cap: usize) -> Vec<f64> {
    x.par_iter()
        .map(|col| {
            let (a, b) = paired(col, y, cap);
            if a.len() < 5 {
                return 0.0;
            }
            match head {
                Head::Multiclass(k) => correlation_ratio(&mid_ranks(&a), &b, k),
                _ => spearman(&a, &b).abs().max(chatterjee_xi(&a, &b)).max(0.0),
            }
        })
        .collect()
}

fn correlation_ratio(x: &[f64], class: &[f64], k: usize) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let (mut sum, mut cnt) = (vec![0.0; k], vec![0.0; k]);
    for (v, c) in x.iter().zip(class) {
        let c = (*c as usize).min(k - 1);
        sum[c] += v;
        cnt[c] += 1.0;
    }
    let between: f64 = (0..k)
        .filter(|&c| cnt[c] > 0.0)
        .map(|c| cnt[c] * (sum[c] / cnt[c] - m).powi(2))
        .sum();
    let total: f64 = x.iter().map(|v| (v - m).powi(2)).sum();
    if total > 0.0 {
        (between / total).sqrt()
    } else {
        0.0
    }
}

/// Feature indices sorted by decreasing association (ties by index).
pub fn ranking(assoc: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..assoc.len()).collect();
    order.sort_by(|&a, &b| assoc[b].total_cmp(&assoc[a]).then(a.cmp(&b)));
    order
}

/// Latent correlation matrix (row-major d × d) from normal scores, with the
/// first-order polyserial/polychoric correction `ρ ≈ r / (sd_i sd_j)` for
/// discretised variables. Uses pairwise-complete rows (at most `cap`).
pub fn latent_correlation(
    marginals: &[Marginal],
    x: &[Vec<f64>],
    rows: &[usize],
    cap: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let d = marginals.len();
    let stride = rows.len().div_ceil(cap.max(1)).max(1);
    let sel: Vec<usize> = rows.iter().step_by(stride).copied().collect();
    let scored: Vec<(Vec<f64>, f64)> = marginals
        .par_iter()
        .zip(x.par_iter())
        .map(|(m, col)| {
            let sub: Vec<f64> = sel.iter().map(|&r| col[r]).collect();
            m.scores(&sub)
        })
        .collect();
    let pairs: Vec<(usize, usize)> = (0..d)
        .flat_map(|i| (i + 1..d).map(move |j| (i, j)))
        .collect();
    let values: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let (a, b) = paired(&scored[i].0, &scored[j].0, usize::MAX);
            if a.len() < 10 {
                return 0.0;
            }
            (pearson(&a, &b) / (scored[i].1 * scored[j].1)).clamp(-0.995, 0.995)
        })
        .collect();
    let mut r = vec![0.0; d * d];
    for i in 0..d {
        r[i * d + i] = 1.0;
    }
    for (&(i, j), v) in pairs.iter().zip(values) {
        r[i * d + j] = v;
        r[j * d + i] = v;
    }
    let sds = scored.iter().map(|s| s.1).collect();
    let scores = scored.into_iter().map(|s| s.0).collect();
    (r, scores, sds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_prefers_informative_features() {
        let n = 500;
        let x0: Vec<f64> = (0..n).map(|i| (i % 50) as f64 / 49.0).collect();
        let noise: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 / 100.0).collect();
        let y: Vec<f64> = x0.iter().map(|v| (v - 0.5).powi(2)).collect();
        let a = associations(&[noise, x0], &y, Head::Regression, 10_000);
        assert_eq!(ranking(&a)[0], 1, "{a:?}");
    }
}
