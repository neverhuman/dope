//! Column-parallel kernels for [`super::descriptors_table::compute_table_descriptors`].
//!
//! Each column (or correlation entry) runs exactly the serial arithmetic it ran before,
//! in the same element order, and results are collected in column order. The
//! descriptors are therefore bitwise identical for every Rayon thread count.

use super::np::{argsort, clip, pairwise_sum, quantile_sorted, sorted};
use rayon::prelude::*;

/// Per-column shape statistics, plus the ascending order when the target varies.
pub(super) struct ColumnShape {
    pub skew: f64,
    pub kurt: f64,
    pub tail_ratio: f64,
    pub mode_conc: f64,
    pub order: Option<Vec<usize>>,
}

/// Standardized column copies of the row-major `x`; columns that are not `valid` stay zero.
pub(super) fn standardized_columns(
    x: &[f64],
    n: usize,
    p: usize,
    cm: &[f64],
    cs: &[f64],
    valid: &[bool],
) -> Vec<Vec<f64>> {
    (0..p)
        .into_par_iter()
        .map(|j| {
            let mut c = vec![0.0f64; n];
            if valid[j] {
                for (i, v) in c.iter_mut().enumerate() {
                    *v = (x[i * p + j] - cm[j]) / cs[j];
                }
            }
            c
        })
        .collect()
}

/// Mean of one length-`n` buffer, with numpy's pairwise sum for a single column.
fn col_mean_1d(buf: &[f64], n_f: f64, p: usize) -> f64 {
    if p == 1 {
        pairwise_sum(buf) / n_f
    } else {
        let mut s = 0.0;
        for &v in buf {
            s += v;
        }
        s / n_f
    }
}

pub(super) fn column_shapes(cols: &[Vec<f64>], n: usize, p: usize, y_std: f64) -> Vec<ColumnShape> {
    cols.par_iter()
        .map(|cj| column_shape(cj, n, p, y_std))
        .collect()
}

fn column_shape(cj: &[f64], n: usize, p: usize, y_std: f64) -> ColumnShape {
    let n_f = n as f64;
    let mut tmp_col = vec![0.0f64; n];
    let (skew, kurt);
    if p == 1 {
        for i in 0..n {
            tmp_col[i] = cj[i].powf(3.0);
        }
        skew = pairwise_sum(&tmp_col) / n_f;
        for i in 0..n {
            tmp_col[i] = cj[i].powf(4.0);
        }
        kurt = clip(pairwise_sum(&tmp_col) / n_f - 3.0, -3.0, 50.0);
    } else {
        let mut s3 = 0.0;
        let mut s4 = 0.0;
        for &v in cj {
            s3 += v.powf(3.0);
            s4 += v.powf(4.0);
        }
        skew = s3 / n_f;
        kurt = clip(s4 / n_f - 3.0, -3.0, 50.0);
    }
    let order = if y_std > 1e-9 {
        let ord = argsort(cj);
        for (k, &idx) in ord.iter().enumerate() {
            tmp_col[k] = cj[idx];
        }
        Some(ord)
    } else {
        let sorted_vec = sorted(cj);
        tmp_col.copy_from_slice(&sorted_vec);
        None
    };
    let s = &tmp_col[..];
    let q05 = quantile_sorted(s, 0.05);
    let q25 = quantile_sorted(s, 0.25);
    let q75 = quantile_sorted(s, 0.75);
    let q95 = quantile_sorted(s, 0.95);
    let iqr = (q75 - q25).max(1e-6);
    let tail_ratio = clip((q95 - q05) / iqr, 0.0, 20.0) / 10.0;
    let med_j = if s[n - 1].is_nan() {
        f64::NAN
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    let mode_conc = if p == 1 {
        for i in 0..n {
            tmp_col[i] = if (cj[i] - med_j).abs() < 0.05 {
                1.0
            } else {
                0.0
            };
        }
        pairwise_sum(&tmp_col) / n_f
    } else {
        let mut mc_s = 0.0;
        for &v in cj {
            if (v - med_j).abs() < 0.05 {
                mc_s += 1.0;
            }
        }
        mc_s / n_f
    };
    ColumnShape {
        skew,
        kurt,
        tail_ratio,
        mode_conc,
        order,
    }
}

/// `|mean(col * y_z)|` per column.
pub(super) fn pearson_abs(cols: &[Vec<f64>], y_z: &[f64], n: usize, p: usize) -> Vec<f64> {
    let n_f = n as f64;
    cols.par_iter()
        .map(|cj| {
            if p == 1 {
                let tmp_col: Vec<f64> = (0..n).map(|i| cj[i] * y_z[i]).collect();
                (pairwise_sum(&tmp_col) / n_f).abs()
            } else {
                let mut ps = 0.0;
                for i in 0..n {
                    ps += cj[i] * y_z[i];
                }
                (ps / n_f).abs()
            }
        })
        .collect()
}

/// `|mean(rank_z(col) * yr)|` per column, from each column's ascending order.
pub(super) fn spearman_abs(orders: &[Vec<usize>], yr: &[f64], n: usize, p: usize) -> Vec<f64> {
    let n_f = n as f64;
    orders
        .par_iter()
        .map(|order| {
            let mut r_col = vec![0.0f64; n];
            for (rank, &idx) in order.iter().enumerate() {
                r_col[idx] = rank as f64;
            }
            let xrm = col_mean_1d(&r_col, n_f, p);
            if p == 1 {
                let sq: Vec<f64> = r_col
                    .iter()
                    .map(|&rv| {
                        let d = rv - xrm;
                        d * d
                    })
                    .collect();
                let xrs = (pairwise_sum(&sq) / n_f).sqrt().max(1e-9);
                let prod: Vec<f64> = (0..n).map(|i| ((r_col[i] - xrm) / xrs) * yr[i]).collect();
                (pairwise_sum(&prod) / n_f).abs()
            } else {
                let mut sq_s = 0.0;
                for &rv in &r_col {
                    let d = rv - xrm;
                    sq_s += d * d;
                }
                let xrs = (sq_s / n_f).sqrt().max(1e-9);
                let mut sp_s = 0.0;
                for i in 0..n {
                    sp_s += ((r_col[i] - xrm) / xrs) * yr[i];
                }
                (sp_s / n_f).abs()
            }
        })
        .collect()
}

/// `cols · colsᵀ / n` for `p > 1`, as 4-column block rows run in parallel. Every entry
/// keeps its own row-order sum, so the matrix is identical to the serial tiling.
pub(super) fn correlation_matrix(cols: &[Vec<f64>], n: usize, p: usize) -> Vec<f64> {
    let n_f = n as f64;
    let block_starts: Vec<usize> = (0..p).step_by(4).collect();
    let parts: Vec<Vec<(usize, usize, f64)>> = block_starts
        .par_iter()
        .map(|&a0| correlation_block_row(cols, n, p, n_f, a0))
        .collect();
    let mut corr = vec![0.0f64; p * p];
    for part in parts {
        for (a, b, v) in part {
            corr[a * p + b] = v;
            corr[b * p + a] = v;
        }
    }
    corr
}

fn dot_mean(ca: &[f64], cb: &[f64], n: usize, n_f: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        s += ca[i] * cb[i];
    }
    s / n_f
}

fn correlation_block_row(
    cols: &[Vec<f64>],
    n: usize,
    p: usize,
    n_f: f64,
    a0: usize,
) -> Vec<(usize, usize, f64)> {
    let a_end = (a0 + 4).min(p);
    let mut out = Vec::new();
    for a in a0..a_end {
        for b in a..a_end {
            out.push((a, b, dot_mean(&cols[a], &cols[b], n, n_f)));
        }
    }
    let mut b0 = a_end;
    while b0 < p {
        let b_end = (b0 + 4).min(p);
        if a_end - a0 == 4 && b_end - b0 == 4 {
            let (ca0, ca1, ca2, ca3) = (&cols[a0], &cols[a0 + 1], &cols[a0 + 2], &cols[a0 + 3]);
            let (cb0, cb1, cb2, cb3) = (&cols[b0], &cols[b0 + 1], &cols[b0 + 2], &cols[b0 + 3]);
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
                    out.push((a0 + u, b0 + v_idx, s_val / n_f));
                }
            }
        } else {
            for a in a0..a_end {
                for b in b0..b_end {
                    out.push((a, b, dot_mean(&cols[a], &cols[b], n, n_f)));
                }
            }
        }
        b0 += 4;
    }
    out
}
