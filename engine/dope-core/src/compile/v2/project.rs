//! Owner-side entropy-balanced moment projection (`P+`) via damped Newton (`ESS >= 0.30`)
//! with full 5-gap ladder logging and zero receiver rejection loops (`T19`).

use crate::audit::LearningSignatureV1;
use crate::stats::cholesky;
use serde::Serialize;

pub const MIN_PROJECTION_ESS: f64 = 0.30;

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionFiveGapLadder {
    pub gap_raw_q: f64,
    pub gap_reweighted: f64,
    pub gap_compiled: f64,
    pub gap_decoded: f64,
    pub ess: f64,
    pub newton_steps: usize,
    pub gamma_used: f64,
}

#[derive(Clone, Debug)]
pub struct ProjectionResult {
    /// Normalized probability weights `sum w_i = 1.0`, satisfying `ESS >= 0.30`.
    pub weights: Vec<f64>,
    /// Deterministic stratified resampling indices implementing `N_eff * w_i` pseudo-counts
    /// for compiling the projected distribution directly into a stateless sampler (zero rejection!).
    pub compiled_indices: Vec<usize>,
    pub ladder: ProjectionFiveGapLadder,
}

/// Solve entropy-balanced exponential tilting `w_i \propto exp(theta^T (Phi_i - mu_P))`
/// via damped Newton with L2 ridge `gamma` and strict `ESS >= 0.30` guard (`T19`).
pub fn project_moments_damped_newton(
    sig: &LearningSignatureV1,
    synth_x: &[Vec<f64>],
    synth_y: &[f64],
    decoded_gap_override: Option<f64>,
) -> ProjectionResult {
    let n = synth_y.len().max(1);
    let m = sig.real_witness_means.len();
    let mat = sig.witness_matrix(synth_x, synth_y);
    let raw_report = sig.evaluate_gaps(synth_x, synth_y, None);
    let gap_raw_q = raw_report.max_abs_gap;

    // Center features: delta[r, k] = Phi[r, k] - mu_P[k] (clipped for numerical stability).
    let mut delta = vec![0.0; n * m];
    for r in 0..n {
        for k in 0..m {
            delta[r * m + k] = (mat[r * m + k] - sig.real_witness_means[k]).clamp(-10.0, 10.0);
        }
    }

    let mut theta = vec![0.0; m];
    let mut best_w = vec![1.0 / n as f64; n];
    let mut best_ess = 1.0;
    let mut best_gap = gap_raw_q;
    let mut gamma = 1e-2_f64;
    let mut steps_taken = 0usize;

    for step in 0..20 {
        // Compute current weights, gradient `E_w[delta] + gamma * theta`, and Hessian `Cov_w[delta] + gamma I`.
        let (w, ess) = tilted_weights(&delta, &theta, n, m);
        let mut grad = vec![0.0; m];
        for r in 0..n {
            for k in 0..m {
                grad[k] += w[r] * delta[r * m + k];
            }
        }
        let max_g = grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if ess >= MIN_PROJECTION_ESS && max_g < best_gap {
            best_gap = max_g;
            best_w = w.clone();
            best_ess = ess;
        }
        if max_g < 1e-4 {
            break;
        }

        // Build `m x m` damped Hessian `H = sum_r w_r d_r d_r^T - g g^T + gamma I`.
        let mut h = vec![0.0; m * m];
        for r in 0..n {
            let wr = w[r];
            let row = &delta[r * m..(r + 1) * m];
            for i in 0..m {
                let wi = wr * row[i];
                for j in 0..=i {
                    h[i * m + j] += wi * row[j];
                }
            }
        }
        for i in 0..m {
            for j in 0..=i {
                let val = h[i * m + j] - grad[i] * grad[j];
                h[i * m + j] = val;
                h[j * m + i] = val;
            }
            h[i * m + i] += gamma;
            grad[i] += gamma * theta[i];
        }
        let Some(dir) = solve_spd(&h, &grad, m) else {
            gamma *= 4.0;
            continue;
        };

        // Backtracking line search enforcing strict `ESS >= MIN_PROJECTION_ESS`.
        let mut accepted = false;
        let mut alpha = 1.0_f64;
        for _ in 0..10 {
            let cand_theta: Vec<f64> = theta
                .iter()
                .zip(&dir)
                .map(|(&t, &d)| (t - alpha * d).clamp(-8.0, 8.0))
                .collect();
            let (cw, cess) = tilted_weights(&delta, &cand_theta, n, m);
            if cess >= MIN_PROJECTION_ESS {
                let mut cg = vec![0.0; m];
                for r in 0..n {
                    for k in 0..m {
                        cg[k] += cw[r] * delta[r * m + k];
                    }
                }
                let cmax = cg.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                if cmax < best_gap + 1e-9 {
                    theta = cand_theta;
                    best_w = cw;
                    best_ess = cess;
                    best_gap = cmax;
                    steps_taken = step + 1;
                    accepted = true;
                    break;
                }
            }
            alpha *= 0.5;
        }
        if !accepted {
            gamma *= 4.0;
            if gamma > 1e4 {
                break;
            }
        }
    }

    // Compile projected weights into a direct deterministic stratified sample of size `n`
    // (`N_eff * w_i` pseudo-counts) -> zero receiver rejection loop!
    let compiled_indices = systematic_resample(&best_w, n);
    let comp_x: Vec<Vec<f64>> = synth_x
        .iter()
        .map(|col| compiled_indices.iter().map(|&i| col[i]).collect())
        .collect();
    let comp_y: Vec<f64> = compiled_indices.iter().map(|&i| synth_y[i]).collect();
    let gap_compiled = sig.evaluate_gaps(&comp_x, &comp_y, None).max_abs_gap;
    let gap_reweighted = sig
        .evaluate_gaps(synth_x, synth_y, Some(&best_w))
        .max_abs_gap;
    let gap_decoded = decoded_gap_override.unwrap_or(gap_compiled);

    ProjectionResult {
        weights: best_w,
        compiled_indices,
        ladder: ProjectionFiveGapLadder {
            gap_raw_q,
            gap_reweighted,
            gap_compiled,
            gap_decoded,
            ess: best_ess,
            newton_steps: steps_taken,
            gamma_used: gamma,
        },
    }
}

fn tilted_weights(delta: &[f64], theta: &[f64], n: usize, m: usize) -> (Vec<f64>, f64) {
    let mut scores = Vec::with_capacity(n);
    let mut max_s = f64::NEG_INFINITY;
    for r in 0..n {
        let mut s = 0.0;
        let row = &delta[r * m..(r + 1) * m];
        for k in 0..m {
            s += theta[k] * row[k];
        }
        s = s.clamp(-30.0, 30.0);
        max_s = max_s.max(s);
        scores.push(s);
    }
    let mut w = Vec::with_capacity(n);
    let mut sum = 0.0;
    for s in scores {
        let e = (s - max_s).exp();
        sum += e;
        w.push(e);
    }
    let inv = 1.0 / sum.max(1e-12);
    let mut sum_sq = 0.0;
    for v in &mut w {
        *v *= inv;
        sum_sq += *v * *v;
    }
    let ess = 1.0 / (n as f64 * sum_sq.max(1e-12));
    (w, ess)
}

fn solve_spd(a: &[f64], b: &[f64], m: usize) -> Option<Vec<f64>> {
    let l = cholesky(a, m)?;
    let mut z = vec![0.0; m];
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * m + k] * z[k];
        }
        z[i] = s / l[i * m + i].max(1e-12);
    }
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut s = z[i];
        for k in (i + 1)..m {
            s -= l[k * m + i] * x[k];
        }
        x[i] = s / l[i * m + i].max(1e-12);
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Deterministic zero-variance systematic resampling of `n` rows from probability weights `w`.
fn systematic_resample(w: &[f64], n: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    let mut cum = 0.0;
    let mut idx = 0usize;
    for k in 0..n {
        let u = (k as f64 + 0.5) / n as f64;
        while idx + 1 < w.len() && cum + w[idx] < u {
            cum += w[idx];
            idx += 1;
        }
        out.push(idx);
    }
    out
}
