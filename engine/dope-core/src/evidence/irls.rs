//! Regularized Newton-IRLS solvers (ridge, binary logistic, exact damped multinomial softmax)
//! shared by `dope_core::evidence` (Tier T1) and `dope_core::single_encoding::v8` (Step-0 target).

use crate::stats::solve_spd;

#[inline]
pub(crate) fn finite_f64(v: f32) -> f64 {
    if v.is_finite() {
        v as f64
    } else {
        0.0
    }
}

/// Fit regularized linear/logistic/exact-multinomial IRLS on a row subset `rows`.
pub(crate) fn fit_irls_subset(
    x_unit: &[Vec<f32>],
    y: &[f64],
    rows: &[usize],
    task_id: u8,
    k_eff: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = rows.len() as f64;
    let lam = match task_id {
        0 => (0.05 * n).max(2.0),
        1 => (0.01 * n).max(1.0),
        _ => (0.02 * n).max(1.5),
    };
    fit_irls_subset_lam(x_unit, y, rows, task_id, k_eff, lam, 6, 8.0, 1e-2, 0.01)
}

/// Build the regression Gram matrix once on `rows` and solve ridge for every `lam` in `lams`.
pub(crate) fn fit_ridge_grid_subset(
    x_unit: &[Vec<f32>],
    y: &[f64],
    rows: &[usize],
    lams: &[f64],
) -> Vec<(Vec<f64>, f64)> {
    let p = x_unit.len();
    let n = rows.len();
    let m = p + 1;
    if n < 4 || p == 0 {
        return lams.iter().map(|_| (vec![0.0; p], 0.0)).collect();
    }
    let mut gram = vec![0.0f64; m * m];
    let mut rhs = vec![0.0f64; m];
    for &r in rows {
        let yr = y[r];
        for i in 0..m {
            let xi = if i < p { finite_f64(x_unit[i][r]) } else { 1.0 };
            rhs[i] += xi * yr;
            for j in i..m {
                let xj = if j < p { finite_f64(x_unit[j][r]) } else { 1.0 };
                gram[i * m + j] += xi * xj;
            }
        }
    }
    for i in 0..m {
        for j in 0..i {
            gram[i * m + j] = gram[j * m + i];
        }
    }
    let mut out = Vec::with_capacity(lams.len());
    let mut work = vec![0.0f64; m * m];
    for &lam in lams {
        work.copy_from_slice(&gram);
        for i in 0..p {
            work[i * m + i] += lam.max(1e-6);
        }
        work[p * m + p] += 1e-4;
        let sol = solve_spd(&work, &rhs, m);
        let mut w = vec![0.0f64; p];
        for j in 0..p {
            w[j] = sol[j].clamp(-64.0, 64.0);
        }
        out.push((w, sol[p].clamp(-12.0, 12.0)));
    }
    out
}

/// Parameterized regularized Newton-IRLS solver (used by `DSE8` OOF-CV linear head and `Evidence`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_irls_subset_lam(
    x_unit: &[Vec<f32>],
    y: &[f64],
    rows: &[usize],
    task_id: u8,
    k_eff: usize,
    lam: f64,
    max_iters: usize,
    w_clip: f64,
    reg_b: f64,
    wt_floor: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let p = x_unit.len();
    let n = rows.len();
    let m = p + 1;
    let o_dim = if task_id == 2 { k_eff.max(2) } else { 1 };
    let mut w_out = vec![vec![0.0f64; p]; o_dim];
    let mut b_out = vec![0.0f64; o_dim];
    if n < 4 || p == 0 {
        return (w_out, b_out);
    }

    if task_id == 0 {
        let mut gram = vec![0.0f64; m * m];
        let mut rhs = vec![0.0f64; m];
        for &r in rows {
            let yr = y[r];
            for i in 0..m {
                let xi = if i < p { finite_f64(x_unit[i][r]) } else { 1.0 };
                rhs[i] += xi * yr;
                for j in i..m {
                    let xj = if j < p { finite_f64(x_unit[j][r]) } else { 1.0 };
                    gram[i * m + j] += xi * xj;
                }
            }
        }
        for i in 0..m {
            for j in 0..i {
                gram[i * m + j] = gram[j * m + i];
            }
            gram[i * m + i] += if i < p { lam } else { reg_b };
        }
        let sol = solve_spd(&gram, &rhs, m);
        for i in 0..p {
            w_out[0][i] = sol[i].clamp(-w_clip, w_clip);
        }
        b_out[0] = sol[p].clamp(-12.0, 12.0);
    } else if task_id == 1 {
        let p1 = (rows.iter().filter(|&&r| y[r] > 0.5).count() as f64 / n as f64).clamp(0.01, 0.99);
        let mut wb = vec![0.0f64; m];
        wb[p] = (p1 / (1.0 - p1)).ln();
        let eval_bin_obj = |v: &[f64]| -> f64 {
            let mut obj = 0.0;
            for &r in rows {
                let mut z = v[p];
                for i in 0..p {
                    z += finite_f64(x_unit[i][r]) * v[i];
                }
                let zc = z.clamp(-14.0, 14.0);
                let yr = if y[r] > 0.5 { 1.0 } else { 0.0 };
                obj += if zc >= 0.0 {
                    (1.0 + (-zc).exp()).ln() + (1.0 - yr) * zc
                } else {
                    (1.0 + zc.exp()).ln() - yr * zc
                };
            }
            for &wi in v.iter().take(p) {
                obj += 0.5 * lam * wi * wi;
            }
            obj + 0.5 * reg_b * v[p] * v[p]
        };
        let mut cur_obj = eval_bin_obj(&wb);
        let mut h = vec![0.0f64; m * m];
        let mut g = vec![0.0f64; m];
        for _ in 0..max_iters {
            h.fill(0.0);
            g.fill(0.0);
            for &r in rows {
                let mut z = wb[p];
                for i in 0..p {
                    z += finite_f64(x_unit[i][r]) * wb[i];
                }
                let pr = 1.0 / (1.0 + (-z.clamp(-14.0, 14.0)).exp());
                let wt = (pr * (1.0 - pr)).max(wt_floor);
                let diff = pr - if y[r] > 0.5 { 1.0 } else { 0.0 };
                for i in 0..m {
                    let xi = if i < p { finite_f64(x_unit[i][r]) } else { 1.0 };
                    g[i] += xi * diff;
                    for j in i..m {
                        let xj = if j < p { finite_f64(x_unit[j][r]) } else { 1.0 };
                        h[i * m + j] += xi * xj * wt;
                    }
                }
            }
            for i in 0..m {
                for j in 0..i {
                    h[i * m + j] = h[j * m + i];
                }
                let reg = if i < p { lam } else { reg_b };
                h[i * m + i] += reg;
                g[i] += reg * wb[i];
            }
            let step = solve_spd(&h, &g, m);
            let max_s = step.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            if !max_s.is_finite() || max_s < 1e-5 {
                break;
            }
            // Uniform backtracking line search (NEVER coordinatewise clamp Newton steps!).
            let mut alpha = if max_s > 4.0 { 4.0 / max_s } else { 1.0 };
            let mut accepted = false;
            for _ in 0..6 {
                let mut cand = vec![0.0f64; m];
                for i in 0..m {
                    let cap = if i < p { w_clip } else { 12.0 };
                    cand[i] = (wb[i] - alpha * step[i]).clamp(-cap, cap);
                }
                let cand_obj = eval_bin_obj(&cand);
                if cand_obj <= cur_obj + 1e-9 {
                    wb = cand;
                    cur_obj = cand_obj;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted || max_s * alpha < 1e-4 {
                break;
            }
        }
        w_out[0].copy_from_slice(&wb[..p]);
        b_out[0] = wb[p];
    } else {
        // Exact damped Multinomial Newton-IRLS with uniform step scaling.
        let mut cnts = vec![1.0f64; o_dim];
        for &r in rows {
            let c = (y[r].round() as isize).clamp(0, (o_dim - 1) as isize) as usize;
            cnts[c] += 1.0;
        }
        let tot: f64 = cnts.iter().sum();
        for c in 0..o_dim {
            b_out[c] = (cnts[c] / tot).max(1e-3).ln();
        }
        let mut h = vec![0.0f64; m * m];
        let mut g = vec![0.0f64; m];
        for _ in 0..max_iters {
            let mut probs = vec![vec![0.0f64; o_dim]; n];
            for (ri, &r) in rows.iter().enumerate() {
                let mut max_l = f64::NEG_INFINITY;
                for c in 0..o_dim {
                    let mut z = b_out[c];
                    for i in 0..p {
                        z += finite_f64(x_unit[i][r]) * w_out[c][i];
                    }
                    let zc = z.clamp(-14.0, 14.0);
                    probs[ri][c] = zc;
                    if zc > max_l {
                        max_l = zc;
                    }
                }
                let mut sum_e = 0.0;
                for c in 0..o_dim {
                    probs[ri][c] = (probs[ri][c] - max_l).exp();
                    sum_e += probs[ri][c];
                }
                let inv = sum_e.max(1e-12).recip();
                for c in 0..o_dim {
                    probs[ri][c] *= inv;
                }
            }
            let mut max_s = 0.0f64;
            for c in 0..o_dim {
                h.fill(0.0);
                g.fill(0.0);
                for (ri, &r) in rows.iter().enumerate() {
                    let pr = probs[ri][c];
                    let target = if (y[r].round() as isize) == c as isize {
                        1.0
                    } else {
                        0.0
                    };
                    let diff = pr - target;
                    let wt = (pr * (1.0 - pr)).max(wt_floor);
                    for i in 0..m {
                        let xi = if i < p { finite_f64(x_unit[i][r]) } else { 1.0 };
                        g[i] += xi * diff;
                        for j in i..m {
                            let xj = if j < p { finite_f64(x_unit[j][r]) } else { 1.0 };
                            h[i * m + j] += xi * xj * wt;
                        }
                    }
                }
                for i in 0..m {
                    for j in 0..i {
                        h[i * m + j] = h[j * m + i];
                    }
                    let reg = if i < p { lam } else { reg_b };
                    let wi = if i < p { w_out[c][i] } else { b_out[c] };
                    h[i * m + i] += reg;
                    g[i] += reg * wi;
                }
                let step = solve_spd(&h, &g, m);
                let step_max = step.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                let alpha = if step_max > 3.0 { 3.0 / step_max } else { 0.85 };
                max_s = max_s.max(step_max * alpha);
                for i in 0..p {
                    w_out[c][i] = (w_out[c][i] - alpha * step[i]).clamp(-w_clip, w_clip);
                }
                b_out[c] = (b_out[c] - alpha * step[p]).clamp(-12.0, 12.0);
            }
            for i in 0..p {
                let mu = (0..o_dim).map(|c| w_out[c][i]).sum::<f64>() / o_dim as f64;
                for c in 0..o_dim {
                    w_out[c][i] -= mu;
                }
            }
            let mb = b_out.iter().sum::<f64>() / o_dim as f64;
            for c in 0..o_dim {
                b_out[c] -= mb;
            }
            if max_s < 1e-4 {
                break;
            }
        }
    }
    (w_out, b_out)
}

/// Predict OOF values (mean for reg, prob(1) for bin, softmax probs for mc).
pub(crate) fn predict_irls_rows(
    x_unit: &[Vec<f32>],
    rows: &[usize],
    w: &[Vec<f64>],
    b: &[f64],
    task_id: u8,
) -> Vec<Vec<f64>> {
    let p = x_unit.len();
    let o_dim = w.len();
    let mut out = vec![vec![0.0f64; rows.len()]; o_dim];
    for (ri, &r) in rows.iter().enumerate() {
        if task_id == 0 {
            let mut z = b[0];
            for i in 0..p {
                z += finite_f64(x_unit[i][r]) * w[0][i];
            }
            out[0][ri] = z;
        } else if task_id == 1 {
            let mut z = b[0];
            for i in 0..p {
                z += finite_f64(x_unit[i][r]) * w[0][i];
            }
            out[0][ri] = 1.0 / (1.0 + (-z.clamp(-12.0, 12.0)).exp());
        } else {
            let mut max_l = f64::NEG_INFINITY;
            let mut lgs = vec![0.0; o_dim];
            for c in 0..o_dim {
                let mut z = b[c];
                for i in 0..p {
                    z += finite_f64(x_unit[i][r]) * w[c][i];
                }
                lgs[c] = z.clamp(-12.0, 12.0);
                if lgs[c] > max_l {
                    max_l = lgs[c];
                }
            }
            let sum_e: f64 = lgs
                .iter()
                .map(|&v| (v - max_l).exp())
                .sum::<f64>()
                .max(1e-12);
            for c in 0..o_dim {
                out[c][ri] = (lgs[c] - max_l).exp() / sum_e;
            }
        }
    }
    out
}
