//! Rust port of `research/levels/building_block_landmarks.py` (building-block
//! probes, row summaries and the multiclass/target simplex).
//!
//! Every reduction mirrors the numpy 1.26 kernel the Python reference uses so
//! the results are bit-identical where the libm functions agree (see [`np`]):
//! 1-D sums are numpy's pairwise sum, axis-0 sums of an `(n, k)` matrix are
//! sequential per column for `k >= 2` and pairwise for `k == 1`, `argsort` is
//! numpy's (unstable) introsort, quantiles use numpy's `_lerp`.
//!
//! Matrices are row-major `n × p` `f64` slices.

pub use super::building_blocks_np as np;
pub use super::building_blocks_simplex::*;

/// Output of [`compute_building_block_landmarks`] (float32 like Python).
#[derive(Debug, Clone, PartialEq)]
pub struct BuildingBlockDescriptors {
    /// `_squash(bb_gains16)`.
    pub bb_gains16: [f32; 16],
    /// `_squash(row_corrs)`.
    pub row_summary_corrs8: [f32; 8],
    /// Per-column `[asinh, atan, erf, step, hinge, fract, xi]`, `max(p, 1)` rows.
    pub col_bb_probes7: Vec<[f32; 7]>,
    /// `_squash(het_and_tail8)`.
    pub het_and_tail8: [f32; 8],
    /// Multiclass simplex / target moments.
    pub multiclass_simplex20: [f32; 20],
    /// `[bb16 | row8 | contrasts16 | het_tail8 | mc20]`.
    pub building_block_vec68: [f32; 68],
}

impl BuildingBlockDescriptors {
    /// The all-zero block (Python's `n < 6 or p == 0` branch; also `mode=none`).
    pub fn zeroed(p: usize) -> Self {
        Self {
            bb_gains16: [0.0; 16],
            row_summary_corrs8: [0.0; 8],
            col_bb_probes7: vec![[0.0; 7]; p.max(1)],
            het_and_tail8: [0.0; 8],
            multiclass_simplex20: [0.0; 20],
            building_block_vec68: [0.0; 68],
        }
    }
}

/// Zero-matrix-allocation columnwise correlation against `y_z`, bit-identical
/// (0 ULP) to `safe_corr_vec(&map_mat_col(z, p, f), n, p, y_z)`.
fn corr_cols(
    z_cols: &[Vec<f64>],
    n: usize,
    p: usize,
    y_z: &[f64],
    col_buf: &mut [f64],
    sq_buf: &mut [f64],
    f: impl Fn(f64, usize) -> f64,
) -> Vec<f64> {
    if n * p == 0 || y_z.is_empty() {
        return vec![0.0; p.max(if p == 0 { 1 } else { 0 })];
    }
    let inv_n = 1.0 / n as f64;
    let mut out = vec![0.0f64; p];
    let mut valid_count = 0usize;
    let mut first_valid_j = 0usize;
    let mut first_valid_pw = 0.0f64;
    for j in 0..p {
        let zj = &z_cols[j];
        let mu = if p == 1 {
            for i in 0..n {
                col_buf[i] = f(zj[i], j);
            }
            np::sum(&col_buf[..n]) * inv_n
        } else {
            let mut s = 0.0;
            for i in 0..n {
                let v = f(zj[i], j);
                col_buf[i] = v;
                s += v;
            }
            s * inv_n
        };
        let m2 = if p == 1 {
            for v in &mut col_buf[..n] {
                *v -= mu;
            }
            np::sum(&col_buf[..n]) * inv_n
        } else {
            let mut s2 = 0.0;
            for v in &mut col_buf[..n] {
                let c = *v - mu;
                *v = c;
                s2 += c;
            }
            s2 * inv_n
        };
        let a_s = if p == 1 {
            for i in 0..n {
                let d = col_buf[i] - m2;
                sq_buf[i] = d * d;
            }
            (np::sum(&sq_buf[..n]) * inv_n).sqrt()
        } else {
            let mut sq_s = 0.0;
            for &v in &col_buf[..n] {
                let d = v - m2;
                sq_s += d * d;
            }
            (sq_s * inv_n).sqrt()
        };
        if a_s > 1e-9 {
            let mut seq_sum = 0.0;
            if valid_count == 0 {
                for i in 0..n {
                    let pr = (col_buf[i] / a_s) * y_z[i];
                    col_buf[i] = pr;
                    seq_sum += pr;
                }
                first_valid_j = j;
                first_valid_pw = (np::sum(&col_buf[..n]) * inv_n).abs();
            } else {
                for i in 0..n {
                    seq_sum += (col_buf[i] / a_s) * y_z[i];
                }
            }
            valid_count += 1;
            out[j] = np::nan_to_zero((seq_sum * inv_n).abs());
        }
    }
    if valid_count == 1 {
        out[first_valid_j] = np::nan_to_zero(first_valid_pw);
    }
    out
}

/// `compute_building_block_landmarks(X_train_scaled, y_train_scaled, level=level)`.
///
/// `x` is row-major `n × p`, `y` has length `n`.
pub fn compute_building_block_landmarks(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    level: &str,
) -> BuildingBlockDescriptors {
    assert_eq!(x.len(), n * p, "x must be row-major n×p");
    assert_eq!(y.len(), n, "y must have n rows");
    let lvl_r = level_rank(level);
    if n < 6 || p == 0 {
        return BuildingBlockDescriptors::zeroed(p);
    }

    // Standardize columns on D_train
    let c_mean = np::mean_axis0(x, n, p);
    let c_std = np::std_axis0_with_mean(x, n, p, &c_mean);
    let mut z = vec![0.0f64; n * p];
    let mut z_cols: Vec<Vec<f64>> = vec![vec![0.0f64; n]; p];
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        for j in 0..p {
            if c_std[j] > 1e-9 {
                let v = np::clip((row[j] - c_mean[j]) / c_std[j], -8.0, 8.0);
                z[i * p + j] = v;
                z_cols[j][i] = v;
            }
        }
    }
    let z = z;

    let y_std = np::std(y);
    let y_z: Vec<f64> = if y_std > 1e-9 {
        let ym = np::mean(y);
        y.iter().map(|&v| (v - ym) / y_std).collect()
    } else {
        vec![0.0; n]
    };

    let mut col_buf = vec![0.0f64; n];
    let mut sq_buf = vec![0.0f64; n];
    let zeros_p = || vec![0.0f64; p];

    // 1. Per-column building-block probes (evaluated column-by-column in L1 cache)
    let c_asinh = corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
        np::asinh(v)
    });
    let c_atan = corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
        v.atan()
    });
    let sqrt2 = 2.0f64.sqrt();
    let c_erf = corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
        erf_as(v / sqrt2)
    });
    let c_cbrt = corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
        np::sign(v) * v.abs().powf(1.0 / 3.0)
    });
    let c_log1p = corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
        np::sign(v) * v.abs().ln_1p()
    });
    let c_tanh = if lvl_r >= 1 {
        corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
            (0.75 * v).tanh()
        })
    } else {
        zeros_p()
    };
    let c_laplace = if lvl_r >= 1 {
        corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
            (-v.abs()).exp_m1()
        })
    } else {
        zeros_p()
    };

    let need_xi = lvl_r >= 2 && y_std > 1e-9;
    let mut col_orders: Vec<Vec<usize>> = if need_xi {
        Vec::with_capacity(p)
    } else {
        Vec::new()
    };
    let mut q = [
        vec![0.0f64; p],
        vec![0.0; p],
        vec![0.0; p],
        vec![0.0; p],
        vec![0.0; p],
    ];
    for j in 0..p {
        let zj = &z_cols[j];
        let s = if need_xi {
            let ord = np::argsort(zj);
            for (k, &idx) in ord.iter().enumerate() {
                col_buf[k] = zj[idx];
            }
            col_orders.push(ord);
            &col_buf[..n]
        } else {
            let sv = np::sorted(zj);
            col_buf[..n].copy_from_slice(&sv);
            &col_buf[..n]
        };
        for (qi, &qq) in [0.10, 0.25, 0.50, 0.75, 0.90].iter().enumerate() {
            q[qi][j] = np::quantile_sorted(s, qq);
        }
    }
    let [q10, q25, q50, q75, q90] = q;
    let c_clamp = if lvl_r >= 1 {
        corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, j| {
            np::clip(v, q10[j], q90[j])
        })
    } else {
        zeros_p()
    };
    let c_step = if lvl_r >= 2 {
        corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, j| {
            if v >= q50[j] {
                1.0
            } else {
                0.0
            }
        })
    } else {
        zeros_p()
    };
    let c_hinge = if lvl_r >= 2 {
        corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, j| {
            np::maximum(0.0, v - q75[j]) - np::maximum(0.0, q25[j] - v)
        })
    } else {
        zeros_p()
    };
    let half_pi = 0.5 * std::f64::consts::PI;
    let c_sincos = if lvl_r >= 2 {
        let s = corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
            (half_pi * v).sin()
        });
        let c = corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
            (half_pi * v).cos()
        });
        s.iter().zip(&c).map(|(&a, &b)| np::maximum(a, b)).collect()
    } else {
        zeros_p()
    };
    let c_fract = if lvl_r >= 2 {
        corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| {
            v - v.floor()
        })
    } else {
        zeros_p()
    };

    // Chatterjee xi rank-jump proxy
    let mut c_xi = zeros_p();
    if need_xi {
        let order_y = np::argsort(y);
        let mut y_ranks = vec![0.0f64; n];
        for (rank, &idx) in order_y.iter().enumerate() {
            y_ranks[idx] = rank as f64;
        }
        let denom = (n + 1).max(2) as f64;
        let inv_nm1 = 1.0 / (n - 1) as f64;
        for j in 0..p {
            let order = &col_orders[j];
            for i in 0..n - 1 {
                col_buf[i] = (y_ranks[order[i + 1]] - y_ranks[order[i]]).abs();
            }
            let d_j = if p == 1 {
                np::sum(&col_buf[..n - 1]) * inv_nm1
            } else {
                let mut s = 0.0;
                for &v in &col_buf[..n - 1] {
                    s += v;
                }
                s * inv_nm1
            };
            c_xi[j] = np::clip(1.0 - (3.0 * d_j) / denom, 0.0, 1.0);
        }
    }

    // 2. Row-level summary features & their target correlations
    let row_feats = compute_row_summary_features(&z, n, p);
    let row_corrs = safe_corr_vec(&row_feats, n, 8, &y_z);
    let row_gain = np::max(&row_corrs);

    // 3. Bivariate polar & normalized-ratio probes on top driver pairs
    let (mut hypot_gain, mut atan2_gain, mut norm_diff_gain) = (0.0f64, 0.0f64, 0.0f64);
    if lvl_r >= 2 && p >= 2 && y_std > 1e-9 {
        let base_score: Vec<f64> = (0..p)
            .map(|j| np::maximum(c_asinh[j], np::maximum(c_step[j], c_xi[j])))
            .collect();
        let mut order = np::argsort(&base_score);
        order.reverse();
        let top: Vec<usize> = order.into_iter().take(p.min(6)).collect();
        let mut pairs = Vec::new();
        for a in 0..top.len() {
            for b in a + 1..top.len() {
                pairs.push((top[a], top[b]));
            }
        }
        if !pairs.is_empty() {
            hypot_gain = np::max(&corr_pairs(
                &z_cols,
                &pairs,
                n,
                &y_z,
                &mut col_buf,
                &mut sq_buf,
                f64::hypot,
            ));
            if lvl_r >= 3 {
                atan2_gain = np::max(&corr_pairs(
                    &z_cols,
                    &pairs,
                    n,
                    &y_z,
                    &mut col_buf,
                    &mut sq_buf,
                    f64::atan2,
                ));
                norm_diff_gain = np::max(&corr_pairs(
                    &z_cols,
                    &pairs,
                    n,
                    &y_z,
                    &mut col_buf,
                    &mut sq_buf,
                    |zi, zj| (zi - zj) / (1.0 + zi.abs() + zj.abs()),
                ));
            }
        }
    }

    let gate = |ok: bool, v: f64| if ok { v } else { 0.0 };
    let bb_gains16: [f64; 16] = [
        np::max(&c_asinh),
        np::max(&c_atan),
        np::max(&c_erf),
        np::max(&c_cbrt),
        np::max(&c_log1p),
        row_gain,
        gate(lvl_r >= 1, np::max(&c_tanh)),
        gate(lvl_r >= 1, np::max(&c_laplace)),
        gate(lvl_r >= 1, np::max(&c_clamp)),
        gate(lvl_r >= 2, np::max(&c_step)),
        gate(lvl_r >= 2, np::max(&c_hinge)),
        gate(lvl_r >= 2, np::max(&c_sincos)),
        gate(lvl_r >= 2, np::max(&c_fract)),
        gate(lvl_r >= 2, hypot_gain),
        gate(lvl_r >= 3, atan2_gain),
        gate(lvl_r >= 3, norm_diff_gain),
    ];

    // 4. Conditional heteroscedasticity & tail co-dependence
    let mut het_and_tail8 = [0.0f64; 8];
    if lvl_r >= 1 && y_std > 1e-9 {
        let best_j = np::argmax(&c_asinh);
        let res_abs: Vec<f64> = (0..n)
            .map(|i| (y_z[i] - c_asinh[best_j] * z_cols[best_j][i]).abs())
            .collect();
        let res_std = np::std(&res_abs);
        let het_corrs = if res_std > 1e-9 {
            let rm = np::mean(&res_abs);
            let den = res_std.max(1e-9);
            let target: Vec<f64> = res_abs.iter().map(|&r| (r - rm) / den).collect();
            corr_cols(&z_cols, n, p, &target, &mut col_buf, &mut sq_buf, |v, _| v)
        } else {
            zeros_p()
        };
        let s_het = np::sorted_desc(&het_corrs);
        het_and_tail8[0] = s_het[0];
        het_and_tail8[1] = np::mean(&s_het[..p.min(3)]);
        het_and_tail8[2] = np::mean(&het_corrs);
        het_and_tail8[3] = gini(&het_corrs);
        if lvl_r >= 2 {
            let ind = |f: &dyn Fn(f64) -> bool, v: &[f64]| -> Vec<f64> {
                v.iter().map(|&x| if f(x) { 1.0 } else { 0.0 }).collect()
            };
            let y_hi = ind(&|v| v > 1.0, &y_z);
            let y_lo = ind(&|v| v < -1.0, &y_z);
            let mh = np::mean(&y_hi);
            let ml = np::mean(&y_lo);
            let t_hi: Vec<f64> = y_hi.iter().map(|&v| v - mh).collect();
            let t_lo: Vec<f64> = y_lo.iter().map(|&v| v - ml).collect();
            let u_corrs = corr_cols(&z_cols, n, p, &t_hi, &mut col_buf, &mut sq_buf, |v, _| {
                if v > 1.0 {
                    1.0
                } else {
                    0.0
                }
            });
            let l_corrs = corr_cols(&z_cols, n, p, &t_lo, &mut col_buf, &mut sq_buf, |v, _| {
                if v < -1.0 {
                    1.0
                } else {
                    0.0
                }
            });
            het_and_tail8[4] = np::max(&u_corrs);
            het_and_tail8[5] = np::mean(&u_corrs);
            het_and_tail8[6] = np::max(&l_corrs);
            het_and_tail8[7] = np::mean(&l_corrs);
        }
    }

    // 5. Multiclass simplex, Fisher discriminant & target modality profile
    let mc_vec20 = compute_multiclass_and_target_simplex(&z, n, p, y, lvl_r);

    // 6. Contrasts among the building-block probes
    let c_lin = corr_cols(&z_cols, n, p, &y_z, &mut col_buf, &mut sq_buf, |v, _| v);
    let lin_max = np::max(&c_lin);
    let denom = np::py_max(np::py_max(np::max(&bb_gains16), lin_max), 0.02);
    let g = &bb_gains16;
    let contrasts_raw: [f64; 16] = [
        (g[0] - lin_max) / denom,
        (g[1] - lin_max) / denom,
        (g[2] - lin_max) / denom,
        (g[3] - lin_max) / denom,
        (g[5] - lin_max) / denom,
        np::mean(&row_corrs) / denom,
        gini(&row_corrs),
        gate(lvl_r >= 1, (g[7] - g[6]) / denom),
        gate(lvl_r >= 1, (g[8] - lin_max) / denom),
        gate(lvl_r >= 2, (g[9] - g[2]) / denom),
        gate(lvl_r >= 2, (g[10] - lin_max) / denom),
        gate(lvl_r >= 2, (g[11] - lin_max) / denom),
        gate(lvl_r >= 2, (g[12] - lin_max) / denom),
        gate(lvl_r >= 2, (g[13] - lin_max) / denom),
        gate(lvl_r >= 3, (g[14] - g[13]) / denom),
        gate(lvl_r >= 3, (g[15] - lin_max) / denom),
    ];
    let col_bb_probes7: Vec<[f32; 7]> = (0..p)
        .map(|j| {
            [
                c_asinh[j], c_atan[j], c_erf[j], c_step[j], c_hinge[j], c_fract[j], c_xi[j],
            ]
            .map(|v| np::nan_to_zero(v as f32 as f64) as f32)
        })
        .collect();
    pack_building_block_descriptors(
        &bb_gains16,
        &row_corrs,
        &contrasts_raw,
        &het_and_tail8,
        &mc_vec20,
        col_bb_probes7,
    )
}
