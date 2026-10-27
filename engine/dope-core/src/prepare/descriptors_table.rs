//! Train-only table descriptors (`table_vec24`, `column_desc`, `empirical_driver_weights`).

use super::descriptors::Level;
use super::np::{
    argsort, clip, col_mean, col_std_with_mean, eigvalsh_sym, finite_or_zero, finite_or_zero32,
    gini_coefficient, mean1d, np_max, pairwise_sum, quantile_sorted, ranks_double_argsort, row_sum,
    sorted, squash, std1d,
};

/// Output of `descriptors.compute_table_descriptors`.
#[derive(Clone, Debug, PartialEq)]
pub struct TableDescriptors {
    pub table_vec24: [f32; 24],
    /// `max(p, 1)` rows of 8 descriptors.
    pub column_desc: Vec<[f32; 8]>,
    pub empirical_driver_weights: Vec<f32>,
}

/// `descriptors.compute_table_descriptors(X_train_scaled, y_train_scaled, level=level)`;
/// `x` is row-major `n × p`.
pub fn compute_table_descriptors(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    level: Level,
) -> TableDescriptors {
    assert_eq!(x.len(), n * p, "x must be n*p");
    if n < 4 || p == 0 {
        let pp = p.max(1);
        return TableDescriptors {
            table_vec24: [0.0; 24],
            column_desc: vec![[0.0; 8]; pp],
            empirical_driver_weights: vec![(1.0 / pp as f64) as f32; pp],
        };
    }
    let cm = col_mean(x, n, p);
    let cs = col_std_with_mean(x, n, p, &cm);
    let valid: Vec<bool> = cs.iter().map(|&s| s > 1e-9).collect();
    let mut cols: Vec<Vec<f64>> = vec![vec![0.0f64; n]; p];
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        for j in 0..p {
            if valid[j] {
                cols[j][i] = (row[j] - cm[j]) / cs[j];
            }
        }
    }
    let n_f = n as f64;
    let col_mean_1d = |buf: &[f64]| -> f64 {
        if p == 1 {
            pairwise_sum(buf) / n_f
        } else {
            let mut s = 0.0;
            for &v in buf {
                s += v;
            }
            s / n_f
        }
    };
    let mut tmp_col = vec![0.0f64; n];
    let mut skew = vec![0.0f64; p];
    let mut kurt = vec![0.0f64; p];
    let mut tail_ratio = vec![0.0f64; p];
    let mut mode_conc = vec![0.0f64; p];
    let y_std = std1d(y);
    let mut col_orders: Vec<Vec<usize>> = if y_std > 1e-9 {
        Vec::with_capacity(p)
    } else {
        Vec::new()
    };
    for j in 0..p {
        let cj = &cols[j];
        if p == 1 {
            for i in 0..n {
                tmp_col[i] = cj[i].powf(3.0);
            }
            skew[j] = pairwise_sum(&tmp_col) / n_f;
            for i in 0..n {
                tmp_col[i] = cj[i].powf(4.0);
            }
            kurt[j] = clip(pairwise_sum(&tmp_col) / n_f - 3.0, -3.0, 50.0);
        } else {
            let mut s3 = 0.0;
            let mut s4 = 0.0;
            for &v in cj {
                s3 += v.powf(3.0);
                s4 += v.powf(4.0);
            }
            skew[j] = s3 / n_f;
            kurt[j] = clip(s4 / n_f - 3.0, -3.0, 50.0);
        }
        let s = if y_std > 1e-9 {
            let ord = argsort(cj);
            for (k, &idx) in ord.iter().enumerate() {
                tmp_col[k] = cj[idx];
            }
            col_orders.push(ord);
            &tmp_col[..]
        } else {
            let sorted_vec = sorted(cj);
            tmp_col.copy_from_slice(&sorted_vec);
            &tmp_col[..]
        };
        let q05 = quantile_sorted(s, 0.05);
        let q25 = quantile_sorted(s, 0.25);
        let q75 = quantile_sorted(s, 0.75);
        let q95 = quantile_sorted(s, 0.95);
        let iqr = (q75 - q25).max(1e-6);
        tail_ratio[j] = clip((q95 - q05) / iqr, 0.0, 20.0) / 10.0;
        let med_j = if s[n - 1].is_nan() {
            f64::NAN
        } else if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        if p == 1 {
            for i in 0..n {
                tmp_col[i] = if (cj[i] - med_j).abs() < 0.05 {
                    1.0
                } else {
                    0.0
                };
            }
            mode_conc[j] = pairwise_sum(&tmp_col) / n_f;
        } else {
            let mut mc_s = 0.0;
            for &v in cj {
                if (v - med_j).abs() < 0.05 {
                    mc_s += 1.0;
                }
            }
            mode_conc[j] = mc_s / n_f;
        }
    }

    let mut y_z = vec![0.0; n];
    let (pearson_r, spearman_r) = if y_std > 1e-9 {
        let ym = mean1d(y);
        for i in 0..n {
            y_z[i] = (y[i] - ym) / y_std;
        }
        let mut pr = vec![0.0f64; p];
        for j in 0..p {
            let cj = &cols[j];
            if p == 1 {
                for i in 0..n {
                    tmp_col[i] = cj[i] * y_z[i];
                }
                pr[j] = (pairwise_sum(&tmp_col) / n_f).abs();
            } else {
                let mut ps = 0.0;
                for i in 0..n {
                    ps += cj[i] * y_z[i];
                }
                pr[j] = (ps / n_f).abs();
            }
        }
        let yr = ranks_double_argsort(y);
        let yrm = mean1d(&yr);
        let yrs = std1d(&yr).max(1e-9);
        let yr: Vec<f64> = yr.iter().map(|v| (v - yrm) / yrs).collect();
        let mut sr = vec![0.0f64; p];
        let mut r_col = vec![0.0f64; n];
        for j in 0..p {
            for (rank, &idx) in col_orders[j].iter().enumerate() {
                r_col[idx] = rank as f64;
            }
            let xrm = col_mean_1d(&r_col);
            let xrs = if p == 1 {
                for i in 0..n {
                    let d = r_col[i] - xrm;
                    tmp_col[i] = d * d;
                }
                (pairwise_sum(&tmp_col) / n_f).sqrt().max(1e-9)
            } else {
                let mut sq_s = 0.0;
                for &rv in &r_col {
                    let d = rv - xrm;
                    sq_s += d * d;
                }
                (sq_s / n_f).sqrt().max(1e-9)
            };
            if p == 1 {
                for i in 0..n {
                    tmp_col[i] = ((r_col[i] - xrm) / xrs) * yr[i];
                }
                sr[j] = (pairwise_sum(&tmp_col) / n_f).abs();
            } else {
                let mut sp_s = 0.0;
                for i in 0..n {
                    sp_s += ((r_col[i] - xrm) / xrs) * yr[i];
                }
                sr[j] = (sp_s / n_f).abs();
            }
        }
        (pr, sr)
    } else {
        (vec![0.0; p], vec![0.0; p])
    };

    let (col_max_corr, col_mean_corr, eff_rank, spec_entropy, top1_eig, top3_eig, top5_eig);
    if p > 1 {
        let mut corr = vec![0.0f64; p * p];
        let mut a0 = 0usize;
        while a0 < p {
            let a_end = (a0 + 4).min(p);
            for a in a0..a_end {
                let ca = &cols[a];
                for b in a..a_end {
                    let cb = &cols[b];
                    let mut s = 0.0;
                    for i in 0..n {
                        s += ca[i] * cb[i];
                    }
                    let v = s / n_f;
                    corr[a * p + b] = v;
                    corr[b * p + a] = v;
                }
            }
            let mut b0 = a_end;
            while b0 < p {
                let b_end = (b0 + 4).min(p);
                if a_end - a0 == 4 && b_end - b0 == 4 {
                    let (ca0, ca1, ca2, ca3) =
                        (&cols[a0], &cols[a0 + 1], &cols[a0 + 2], &cols[a0 + 3]);
                    let (cb0, cb1, cb2, cb3) =
                        (&cols[b0], &cols[b0 + 1], &cols[b0 + 2], &cols[b0 + 3]);
                    let mut s = [[0.0f64; 4]; 4];
                    for i in 0..n {
                        let va = [ca0[i], ca1[i], ca2[i], ca3[i]];
                        let vb = [cb0[i], cb1[i], cb2[i], cb3[i]];
                        for u in 0..4 {
                            for v in 0..4 {
                                s[u][v] += va[u] * vb[v];
                            }
                        }
                    }
                    for (u, row_s) in s.iter().enumerate() {
                        for (v_idx, &s_val) in row_s.iter().enumerate() {
                            let (a, b, val) = (a0 + u, b0 + v_idx, s_val / n_f);
                            corr[a * p + b] = val;
                            corr[b * p + a] = val;
                        }
                    }
                } else {
                    for a in a0..a_end {
                        let ca = &cols[a];
                        for b in b0..b_end {
                            let cb = &cols[b];
                            let mut s = 0.0;
                            for i in 0..n {
                                s += ca[i] * cb[i];
                            }
                            let v = s / n_f;
                            corr[a * p + b] = v;
                            corr[b * p + a] = v;
                        }
                    }
                }
                b0 += 4;
            }
            a0 += 4;
        }
        for a in 0..p {
            corr[a * p + a] = 1.0;
        }
        for v in &mut corr {
            *v = finite_or_zero(*v);
        }
        let off: Vec<f64> = (0..p * p)
            .map(|k| {
                let eye = if k / p == k % p { 1.0 } else { 0.0 };
                (corr[k] - eye).abs()
            })
            .collect();
        col_max_corr = (0..p)
            .map(|a| np_max(&off[a * p..(a + 1) * p]))
            .collect::<Vec<_>>();
        col_mean_corr = row_sum(&off, p, p)
            .into_iter()
            .map(|s| s / (p - 1) as f64)
            .collect::<Vec<_>>();
        let mut ev = eigvalsh_sym(&corr, p);
        ev.reverse();
        let ev: Vec<f64> = ev.into_iter().map(|v| v.max(0.0)).collect();
        let eig_sum = pairwise_sum(&ev).max(1e-9);
        let shares: Vec<f64> = ev.iter().map(|v| v / eig_sum).collect();
        let sq: Vec<f64> = ev.iter().map(|v| v * v).collect();
        eff_rank = ((eig_sum * eig_sum) / pairwise_sum(&sq).max(1e-9)) / p as f64;
        let ent: Vec<f64> = shares
            .iter()
            .filter(|&&s| s > 1e-12)
            .map(|&s| s * s.ln())
            .collect();
        spec_entropy = -pairwise_sum(&ent) / (p as f64).ln().max(1e-9);
        top1_eig = shares[0];
        top3_eig = pairwise_sum(&shares[..p.min(3)]);
        top5_eig = pairwise_sum(&shares[..p.min(5)]);
    } else {
        col_max_corr = vec![0.0];
        col_mean_corr = vec![0.0];
        eff_rank = 1.0;
        spec_entropy = 1.0;
        top1_eig = 1.0;
        top3_eig = 1.0;
        top5_eig = 1.0;
    }

    let quad_r: Vec<f64> = if y_std > 1e-9 && level == Level::Ultra {
        let mut qr = vec![0.0f64; p];
        for j in 0..p {
            let cj = &cols[j];
            for i in 0..n {
                let v = cj[i];
                tmp_col[i] = v * v - 1.0;
            }
            let z2m = col_mean_1d(&tmp_col);
            let z2s = if p == 1 {
                let sq: Vec<f64> = tmp_col
                    .iter()
                    .map(|&v| {
                        let d = v - z2m;
                        d * d
                    })
                    .collect();
                (pairwise_sum(&sq) / n_f).sqrt().max(1e-6)
            } else {
                let mut sq_s = 0.0;
                for &v in &tmp_col {
                    let d = v - z2m;
                    sq_s += d * d;
                }
                (sq_s / n_f).sqrt().max(1e-6)
            };
            if p == 1 {
                for i in 0..n {
                    tmp_col[i] = (tmp_col[i] / z2s) * y_z[i];
                }
                qr[j] = (pairwise_sum(&tmp_col) / n_f).abs();
            } else {
                let mut pr_s = 0.0;
                for i in 0..n {
                    pr_s += (tmp_col[i] / z2s) * y_z[i];
                }
                qr[j] = (pr_s / n_f).abs();
            }
        }
        qr
    } else {
        vec![0.0; p]
    };

    let raw_driver: Vec<f64> = (0..p)
        .map(|j| 0.45 * pearson_r[j] + 0.35 * spearman_r[j] + 0.20 * quad_r[j])
        .collect();
    let rd_sum = pairwise_sum(&raw_driver);
    let driver_weights: Vec<f64> = if rd_sum > 1e-9 {
        raw_driver.iter().map(|v| v / rd_sum).collect()
    } else {
        vec![1.0 / p as f64; p]
    };
    let mut sorted_r = sorted(&pearson_r);
    sorted_r.reverse();
    let mut sorted_sp = sorted(&spearman_r);
    sorted_sp.reverse();

    let inc_l = if level.includes_larger() { 1.0 } else { 0.0 };
    let inc_f = if level.includes_fine() { 1.0 } else { 0.0 };
    let abs_skew: Vec<f64> = skew.iter().map(|v| v.abs()).collect();
    let mas = mean1d(&abs_skew);
    let xas = np_max(&abs_skew);
    let kpos: Vec<f64> = kurt.iter().map(|v| v.max(0.0)).collect();
    let mk = mean1d(&kpos);
    let sp_minus_p: Vec<f64> = (0..p).map(|j| spearman_r[j] - pearson_r[j]).collect();
    let tv: [f64; 24] = [
        eff_rank,
        spec_entropy,
        mas / (1.0 + mas),
        mean1d(&mode_conc),
        sorted_r[0],
        mean1d(&pearson_r),
        gini_coefficient(&pearson_r),
        (n.max(1) as f64).log10() / 6.0,
        inc_l * top1_eig,
        inc_l * top3_eig,
        inc_l * top5_eig,
        inc_l * np_max(&col_max_corr),
        inc_l * mean1d(&col_mean_corr),
        inc_l * mean1d(&tail_ratio),
        inc_l * sorted_sp[0],
        inc_l * np_max(&sp_minus_p),
        inc_f * (xas / (1.0 + xas)),
        inc_f * (mk / (1.0 + mk)),
        inc_f * np_max(&tail_ratio),
        inc_f * mean1d(&sorted_r[..p.min(3)]),
        inc_f * mean1d(&sorted_sp[..p.min(3)]),
        inc_f * np_max(&quad_r),
        inc_f * gini_coefficient(&raw_driver),
        inc_f * np_max(&raw_driver),
    ];
    let table_vec24 = tv.map(|v| finite_or_zero32(v as f32));
    // col_max_corr has length 1 when p == 1 (matches numpy broadcasting for p == 1).
    let column_desc: Vec<[f32; 8]> = (0..p)
        .map(|j| {
            [
                squash(skew[j]),
                squash(kurt[j]) * inc_l,
                tail_ratio[j] * inc_l,
                mode_conc[j],
                col_max_corr[j.min(col_max_corr.len() - 1)] * inc_l,
                pearson_r[j],
                spearman_r[j] * inc_l,
                quad_r[j] * inc_f,
            ]
            .map(|v| finite_or_zero32(v as f32))
        })
        .collect();
    let empirical_driver_weights: Vec<f32> = driver_weights
        .iter()
        .map(|&v| {
            let f = v as f32;
            if f.is_nan() {
                (1.0 / p as f64) as f32
            } else {
                f.clamp(f32::MIN, f32::MAX)
            }
        })
        .collect();
    TableDescriptors {
        table_vec24,
        column_desc,
        empirical_driver_weights,
    }
}

/// `descriptors._summarize_feature_spectrum`.
pub fn summarize_feature_spectrum(spec: &[[f64; 9]], mask: &[[bool; 9]]) -> [f64; 16] {
    if spec.is_empty() || !mask.iter().any(|r| r.iter().any(|&b| b)) {
        return [0.0; 16];
    }
    let s: Vec<[f64; 9]> = spec.iter().map(|r| r.map(squash)).collect();
    let pos = |v: f64| v.max(0.0);
    let rmax =
        |r: &[f64; 9], ks: &[usize]| np_max(&ks.iter().map(|&k| pos(r[k])).collect::<Vec<_>>());
    let lin_col: Vec<f64> = s.iter().map(|r| pos(r[0])).collect();
    let nonlin_macro: Vec<f64> = s.iter().map(|r| rmax(r, &[3, 4, 6])).collect();
    let nonlin_larger: Vec<f64> = s.iter().map(|r| rmax(r, &[1, 2, 5, 8])).collect();
    let xi_col: Vec<f64> = s.iter().map(|r| pos(r[7])).collect();
    let best: Vec<f64> = s
        .iter()
        .map(|r| rmax(r, &[0, 1, 2, 3, 4, 5, 6, 7, 8]))
        .collect();
    let mut sb = sorted(&best);
    sb.reverse();
    let top1 = sb[0];
    let top3 = mean1d(&sb[..sb.len().min(3)]);
    let diff =
        |a: &[f64], b: &[f64]| np_max(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>());
    [
        np_max(&lin_col),
        mean1d(&lin_col),
        gini_coefficient(&lin_col),
        np_max(&nonlin_macro),
        mean1d(&nonlin_macro),
        np_max(&nonlin_larger),
        mean1d(&nonlin_larger),
        np_max(&xi_col),
        mean1d(&xi_col),
        gini_coefficient(&xi_col),
        top1,
        top3,
        gini_coefficient(&best),
        diff(&nonlin_macro, &lin_col),
        diff(&nonlin_larger, &lin_col),
        diff(&xi_col, &lin_col),
    ]
}
