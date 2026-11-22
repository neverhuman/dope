//! Ridge Certificate: exact closed-form `||w_Q - w_P||_2` parameter & prediction distance.

use crate::stats::cholesky;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct RidgeCertificate {
    pub lambda: f64,
    pub weight_l2_diff: f64,
    pub pred_rmse_diff: f64,
}

fn solve_ridge(
    x_cols: &[Vec<f64>],
    y: &[f64],
    means: &[f64],
    sds: &[f64],
    lambda: f64,
) -> Vec<f64> {
    let d = x_cols.len();
    let p = d + 1;
    let n = y.len().max(1);
    let mut a = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    let mut row = vec![0.0; p];
    let inv_n = 1.0 / n as f64;
    for r in 0..n {
        for j in 0..d {
            row[j] = if x_cols[j][r].is_finite() {
                ((x_cols[j][r] - means[j]) / sds[j]).clamp(-6.0, 6.0)
            } else {
                0.0
            };
        }
        row[d] = 1.0;
        let yr = y[r];
        for i in 0..p {
            b[i] += inv_n * row[i] * yr;
            for j in 0..=i {
                a[i * p + j] += inv_n * row[i] * row[j];
            }
        }
    }
    for i in 0..p {
        for j in (i + 1)..p {
            a[i * p + j] = a[j * p + i];
        }
        a[i * p + i] += if i == d { 1e-6 } else { lambda };
    }
    let Some(l) = cholesky(&a, p) else {
        return vec![0.0; p];
    };
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * p + k] * z[k];
        }
        z[i] = s / l[i * p + i].max(1e-9);
    }
    let mut w = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for k in (i + 1)..p {
            s -= l[k * p + i] * w[k];
        }
        w[i] = s / l[i * p + i].max(1e-9);
    }
    w
}

/// Compute Ridge Certificate comparing exact L2 ridge fits on real `P` vs synthetic `Q`.
pub fn ridge_certificate(
    real_x: &[Vec<f64>],
    real_y: &[f64],
    synth_x: &[Vec<f64>],
    synth_y: &[f64],
    val_x: &[Vec<f64>],
    lambda: f64,
) -> RidgeCertificate {
    let d = real_x.len();
    let mut means = vec![0.0; d];
    let mut sds = vec![1.0; d];
    for j in 0..d {
        let mut s = 0.0;
        let mut cnt = 0usize;
        for &v in &real_x[j] {
            if v.is_finite() {
                s += v;
                cnt += 1;
            }
        }
        let m = if cnt > 0 { s / cnt as f64 } else { 0.0 };
        let var: f64 = real_x[j]
            .iter()
            .filter(|v| v.is_finite())
            .map(|&v| (v - m).powi(2))
            .sum::<f64>()
            / cnt.max(1) as f64;
        means[j] = m;
        sds[j] = var.sqrt().max(1e-6);
    }
    let wp = solve_ridge(real_x, real_y, &means, &sds, lambda);
    let wq = solve_ridge(synth_x, synth_y, &means, &sds, lambda);
    let weight_l2_diff = wp
        .iter()
        .zip(&wq)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let nv = val_x.first().map_or(0, Vec::len);
    let pred_rmse_diff = if nv == 0 {
        0.0
    } else {
        let mut ss = 0.0;
        for r in 0..nv {
            let mut dp = wq[d] - wp[d];
            for j in 0..d {
                if val_x[j][r].is_finite() {
                    let z = ((val_x[j][r] - means[j]) / sds[j]).clamp(-6.0, 6.0);
                    dp += (wq[j] - wp[j]) * z;
                }
            }
            ss += dp * dp;
        }
        (ss / nv as f64).sqrt()
    };
    RidgeCertificate {
        lambda,
        weight_l2_diff,
        pred_rmse_diff,
    }
}
