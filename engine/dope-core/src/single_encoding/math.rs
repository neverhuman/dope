//! Pure-Rust numerical linear algebra, normal-score, and IEEE-754 f16 helpers for `DSE7` packets.

use crate::stats::norm_ppf;

#[inline]
pub(crate) fn gelu_exact(x: f32) -> f32 {
    let z = (x as f64) * std::f64::consts::FRAC_1_SQRT_2;
    (0.5 * (x as f64) * (1.0 + erf_f64(z))) as f32
}

#[inline]
pub(crate) fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn percentile_sorted(sorted: &[f64], u: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let pos = u.clamp(0.0, 1.0) * (n - 1) as f64;
    let i0 = (pos.floor() as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    sorted[i0] + (pos - i0 as f64) * (sorted[i1] - sorted[i0])
}

pub(crate) fn count_unique_sorted(sorted: &[f64]) -> Vec<(f64, usize)> {
    let mut out: Vec<(f64, usize)> = Vec::new();
    for &v in sorted {
        if let Some(last) = out.last_mut() {
            if (last.0 - v).abs() <= 1e-9 {
                last.1 += 1;
                continue;
            }
        }
        out.push((v, 1));
    }
    out
}

pub(crate) fn norm_cdf_f64(z: f64) -> f64 {
    0.5 * (1.0 + erf_f64(z * std::f64::consts::FRAC_1_SQRT_2))
}

/// Conditional-mean normal scores (SP1 tie disattenuation for discrete & zero-inflated columns).
pub(crate) fn normal_scores_conditional(col: &[f32], u_floor: f64) -> Vec<f64> {
    let n = col.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| col[a].total_cmp(&col[b]));
    let mut z = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (col[order[j]] - col[order[i]]).abs() <= 1e-9 {
            j += 1;
        }
        let u_lo = ((i as f64 + 0.5) / n as f64).clamp(u_floor, 1.0 - u_floor);
        let u_hi = (((j - 1) as f64 + 0.5) / n as f64).clamp(u_floor, 1.0 - u_floor);
        let score = if j - i >= 2 && (u_hi - u_lo) > 1e-4 {
            let zlo = norm_ppf(u_lo);
            let zhi = norm_ppf(u_hi);
            let pdf = |t: f64| (-0.5 * t * t).exp() / (2.0 * std::f64::consts::PI).sqrt();
            ((pdf(zlo) - pdf(zhi)) / (u_hi - u_lo)).clamp(-3.5, 3.5)
        } else {
            norm_ppf(0.5 * (u_lo + u_hi))
        };
        for &idx in &order[i..j] {
            z[idx] = score;
        }
        i = j;
    }
    z
}

/// Symmetric Jacobi eigendecomposition -> top-`rank_use` int8-quantized Gaussian copula loadings.
pub(crate) fn fit_factor_loadings_i8(z_mat: &[Vec<f64>], rank_use: usize) -> Vec<Vec<i8>> {
    let p = z_mat.len();
    let mut out = vec![vec![0i8; rank_use]; p];
    if rank_use == 0 || p < 2 || z_mat[0].len() < 8 {
        return out;
    }
    let n = z_mat[0].len();
    let means: Vec<f64> = z_mat
        .iter()
        .map(|c| c.iter().sum::<f64>() / n as f64)
        .collect();
    let mut a = vec![vec![0.0f64; p]; p];
    for i in 0..p {
        for j in i..p {
            let mut s = 0.0;
            for r in 0..n {
                s += (z_mat[i][r] - means[i]) * (z_mat[j][r] - means[j]);
            }
            let v = s / (n - 1).max(1) as f64;
            a[i][j] = v;
            a[j][i] = v;
        }
    }
    let mut v = vec![vec![0.0f64; p]; p];
    for i in 0..p {
        v[i][i] = 1.0;
    }
    for _ in 0..(p * p * 4).min(2400) {
        let (mut bi, mut bj, mut bm) = (0, 1, 0.0);
        for i in 0..p {
            for j in (i + 1)..p {
                if a[i][j].abs() > bm {
                    bm = a[i][j].abs();
                    bi = i;
                    bj = j;
                }
            }
        }
        if bm < 1e-8 {
            break;
        }
        let tau = (a[bj][bj] - a[bi][bi]) / (2.0 * a[bi][bj]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = (1.0 + t * t).sqrt().recip();
        let s = t * c;
        let (aii, ajj, aij) = (a[bi][bi], a[bj][bj], a[bi][bj]);
        a[bi][bi] = aii - t * aij;
        a[bj][bj] = ajj + t * aij;
        a[bi][bj] = 0.0;
        a[bj][bi] = 0.0;
        for k in 0..p {
            if k != bi && k != bj {
                let (aki, akj) = (a[k][bi], a[k][bj]);
                a[k][bi] = c * aki - s * akj;
                a[bi][k] = a[k][bi];
                a[k][bj] = s * aki + c * akj;
                a[bj][k] = a[k][bj];
            }
            let (vki, vkj) = (v[k][bi], v[k][bj]);
            v[k][bi] = c * vki - s * vkj;
            v[k][bj] = s * vki + c * vkj;
        }
    }
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&x, &y| a[y][y].total_cmp(&a[x][x]));
    let r_eff = rank_use.min(p);
    let mut raw = vec![vec![0.0f64; r_eff]; p];
    for (ri, &col) in order.iter().take(r_eff).enumerate() {
        let ev = a[col][col].max(0.0).sqrt();
        for i in 0..p {
            raw[i][ri] = v[i][col] * ev;
        }
    }
    for i in 0..p {
        let norm = raw[i].iter().map(|&x| x * x).sum::<f64>().sqrt();
        let sc = (norm / 0.99).max(1.0);
        for r in 0..r_eff {
            out[i][r] = ((raw[i][r] / sc) * 127.0).round().clamp(-127.0, 127.0) as i8;
        }
    }
    out
}

/// Fit analytic S3 target model (ridge / 5-step Newton-IRLS logistic / multiclass log-odds ridge).
pub(crate) fn fit_target_model_i8(
    x_sc: &[Vec<f32>],
    y_sc: &[f64],
    task_id: u8,
    k_eff: usize,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<i8>>, f32) {
    let p = x_sc.len();
    let n = y_sc.len();
    let m = p + 1;
    let o_dim = if task_id == 2 { k_eff } else { 1 };
    let mut w_scales = vec![1e-6f32; o_dim];
    let mut b_vals = vec![0.0f32; o_dim];
    let mut w_i8 = vec![vec![0i8; p]; o_dim];
    let mut resid_sd = 0.5f32;

    let mut gram = vec![vec![0.0f64; m]; m];
    for r in 0..n {
        for i in 0..m {
            let xi = if i < p { x_sc[i][r] as f64 } else { 1.0 };
            for j in i..m {
                let xj = if j < p { x_sc[j][r] as f64 } else { 1.0 };
                gram[i][j] += xi * xj;
            }
        }
    }
    for i in 0..m {
        for j in 0..i {
            gram[i][j] = gram[j][i];
        }
    }
    let lam = (0.05 * n as f64).max(4.0);
    if task_id == 0 {
        let mut a = gram.clone();
        for i in 0..p {
            a[i][i] += lam;
        }
        a[p][p] += 1e-2;
        let mut rhs = vec![0.0f64; m];
        for r in 0..n {
            for i in 0..p {
                rhs[i] += x_sc[i][r] as f64 * y_sc[r];
            }
            rhs[p] += y_sc[r];
        }
        let sol = solve_linear(&mut a, &mut rhs);
        let mut sse = 0.0;
        for r in 0..n {
            let mut pred = sol[p];
            for i in 0..p {
                pred += x_sc[i][r] as f64 * sol[i];
            }
            sse += (y_sc[r] - pred).powi(2);
        }
        resid_sd = ((sse / n.max(1) as f64).sqrt() as f32).clamp(0.05, 4.0);
        quantize_w(
            &sol[..p],
            sol[p],
            &mut w_scales[0],
            &mut b_vals[0],
            &mut w_i8[0],
        );
    } else if task_id == 1 {
        let p1 = (y_sc.iter().sum::<f64>() / n as f64).clamp(0.02, 0.98);
        let mut wb = vec![0.0f64; m];
        wb[p] = (p1 / (1.0 - p1)).ln();
        let lam_b = (0.005 * n as f64).max(1.0);
        for _ in 0..5 {
            let mut h = vec![vec![0.0f64; m]; m];
            let mut g = vec![0.0f64; m];
            for r in 0..n {
                let mut z = wb[p];
                for i in 0..p {
                    z += x_sc[i][r] as f64 * wb[i];
                }
                let pr = 1.0 / (1.0 + (-z.clamp(-8.0, 8.0)).exp());
                let wt = (pr * (1.0 - pr)).max(0.02);
                let diff = pr - y_sc[r];
                for i in 0..m {
                    let xi = if i < p { x_sc[i][r] as f64 } else { 1.0 };
                    g[i] += xi * diff;
                    for j in i..m {
                        let xj = if j < p { x_sc[j][r] as f64 } else { 1.0 };
                        h[i][j] += xi * xj * wt;
                    }
                }
            }
            for i in 0..m {
                for j in 0..i {
                    h[i][j] = h[j][i];
                }
                let reg = if i < p { lam_b } else { 1e-2 };
                h[i][i] += reg;
                g[i] += reg * wb[i];
            }
            let step = solve_linear(&mut h, &mut g);
            for i in 0..m {
                wb[i] = (wb[i] - step[i]).clamp(-8.0, 8.0);
            }
        }
        quantize_w(
            &wb[..p],
            wb[p],
            &mut w_scales[0],
            &mut b_vals[0],
            &mut w_i8[0],
        );
    } else {
        let mut cnts = vec![1.0f64; k_eff];
        for &v in y_sc {
            cnts[(v.round() as usize).min(k_eff - 1)] += 1.0;
        }
        let tot: f64 = cnts.iter().sum();
        let priors: Vec<f64> = cnts.iter().map(|&c| c / tot).collect();
        for c in 0..k_eff {
            let mut a = gram.clone();
            for i in 0..p {
                a[i][i] += lam;
            }
            a[p][p] += 1e-2;
            let mut rhs = vec![0.0f64; m];
            let lp = priors[c].max(1e-3).ln();
            for r in 0..n {
                let oh = if (y_sc[r].round() as usize).min(k_eff - 1) == c {
                    1.0
                } else {
                    0.0
                };
                let yt = (oh - priors[c]) * 3.0 + lp;
                for i in 0..p {
                    rhs[i] += x_sc[i][r] as f64 * yt;
                }
                rhs[p] += yt;
            }
            let sol = solve_linear(&mut a, &mut rhs);
            quantize_w(
                &sol[..p],
                sol[p],
                &mut w_scales[c],
                &mut b_vals[c],
                &mut w_i8[c],
            );
        }
    }
    (w_scales, b_vals, w_i8, resid_sd)
}

fn quantize_w(w: &[f64], b: f64, scale_out: &mut f32, b_out: &mut f32, out_i8: &mut [i8]) {
    let max_a = w.iter().map(|v| v.abs()).fold(0.0f64, f64::max).min(8.0);
    let sc = (max_a / 127.0) as f32 + 1e-6;
    *scale_out = sc;
    *b_out = (b.clamp(-20.0, 20.0)) as f32;
    for (i, &v) in w.iter().enumerate() {
        out_i8[i] = ((v.clamp(-8.0, 8.0) as f32) / sc)
            .round()
            .clamp(-127.0, 127.0) as i8;
    }
}

fn solve_linear(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > best {
                best = a[i][k].abs();
                piv = i;
            }
        }
        a.swap(k, piv);
        b.swap(k, piv);
        let d = a[k][k];
        if d.abs() < 1e-12 {
            continue;
        }
        for i in (k + 1)..n {
            let f = a[i][k] / d;
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() > 1e-12 {
            s / a[i][i]
        } else {
            0.0
        };
    }
    x
}

pub(crate) fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7bff;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        return f32::from_bits(sign);
    }
    if exp == 31 {
        return f32::from_bits(sign | (0xff << 23));
    }
    f32::from_bits(sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13))
}

pub(crate) fn quant_f16_roundtrip(v: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(v))
}
