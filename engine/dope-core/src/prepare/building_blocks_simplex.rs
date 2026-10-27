//! Row summary features, correlation helpers, and multiclass/target simplex
//! for `prepare::building_blocks`.

use super::building_blocks::np;

/// `_squash` on a scalar: non-finite → 0, else `v / (1 + |v|)`.
pub fn squash(v: f64) -> f64 {
    if !v.is_finite() {
        0.0
    } else {
        v / (1.0 + v.abs())
    }
}

/// `_squash` on an array (nan_to_num first, then `v / (1 + |v|)`).
pub fn squash_vec(v: &[f64]) -> Vec<f64> {
    v.iter()
        .map(|&x| {
            let a = np::nan_to_zero(x);
            a / (1.0 + a.abs())
        })
        .collect()
}

/// `_erf_vec` (Abramowitz–Stegun 7.1.26) on a scalar.
pub fn erf_as(x: f64) -> f64 {
    let sign = np::sign(x);
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-(ax * ax)).exp())
}

/// `_gini`.
pub fn gini(values: &[f64]) -> f64 {
    let arr: Vec<f64> = values
        .iter()
        .map(|v| v.abs())
        .filter(|v| v.is_finite())
        .collect();
    if arr.len() <= 1 || np::sum(&arr) <= 1e-12 {
        return 0.0;
    }
    let s = np::sorted(&arr);
    let n = s.len();
    let weighted: Vec<f64> = s
        .iter()
        .enumerate()
        .map(|(i, &v)| (i + 1) as f64 * v)
        .collect();
    np::clip(
        (2.0 * np::sum(&weighted)) / (n as f64 * np::sum(&s)) - (n as f64 + 1.0) / n as f64,
        0.0,
        1.0,
    )
}

/// `_safe_corr_vec`: |Pearson| of each column of `a` (`n × k`) with `y_z`.
pub fn safe_corr_vec(a: &[f64], n: usize, k: usize, y_z: &[f64]) -> Vec<f64> {
    if n * k == 0 || y_z.is_empty() {
        return vec![0.0; k.max(if k == 0 { 1 } else { 0 })];
    }
    let mu = np::mean_axis0(a, n, k);
    let mut a_c = vec![0.0f64; n * k];
    for i in 0..n {
        for j in 0..k {
            a_c[i * k + j] = a[i * k + j] - mu[j];
        }
    }
    let a_s = np::std_axis0(&a_c, n, k);
    let valid: Vec<usize> = (0..k).filter(|&j| a_s[j] > 1e-9).collect();
    let mut out = vec![0.0f64; k];
    if !valid.is_empty() {
        let kv = valid.len();
        let mut prod = vec![0.0f64; n * kv];
        for i in 0..n {
            for (c, &j) in valid.iter().enumerate() {
                prod[i * kv + c] = (a_c[i * k + j] / a_s[j]) * y_z[i];
            }
        }
        let m = np::mean_axis0(&prod, n, kv);
        for (c, &j) in valid.iter().enumerate() {
            out[j] = m[c].abs();
        }
    }
    out.into_iter().map(np::nan_to_zero).collect()
}

/// `_safe_pair_corr`: clipped Pearson correlation of two equal-length vectors.
pub fn safe_pair_corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() != a.len() {
        return 0.0;
    }
    let (sa, sb) = (np::std(a), np::std(b));
    if sa <= 1e-9 || sb <= 1e-9 {
        return 0.0;
    }
    let (ma, mb) = (np::mean(a), np::mean(b));
    let prod: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| ((x - ma) / sa) * ((y - mb) / sb))
        .collect();
    let c = np::mean(&prod);
    if c.is_finite() {
        np::clip(c, -1.0, 1.0)
    } else {
        0.0
    }
}

/// `compute_row_summary_features` (no missing mask): row-major `n × 8`
/// `[row_mean, row_std, row_l1, row_l2, row_min, row_max, row_skew, row_missing_frac]`.
pub fn compute_row_summary_features(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * 8];
    if n == 0 || p == 0 {
        return out;
    }
    let mut clean = vec![0.0f64; p];
    let mut tmp = vec![0.0f64; p];
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        let mut finite = 0usize;
        for j in 0..p {
            if row[j].is_finite() {
                finite += 1;
                clean[j] = row[j];
            } else {
                clean[j] = 0.0;
            }
        }
        let miss_frac = 1.0 - finite as f64 / p as f64;
        let r_mean = np::mean(&clean);
        for j in 0..p {
            let c = clean[j] - r_mean;
            tmp[j] = c * c;
        }
        let r_std = (np::mean(&tmp) + 1e-8).sqrt();
        for j in 0..p {
            tmp[j] = clean[j].abs();
        }
        let r_l1 = np::mean(&tmp);
        for j in 0..p {
            tmp[j] = clean[j] * clean[j];
        }
        let r_l2 = (np::mean(&tmp) + 1e-8).sqrt();
        let r_min = np::min(&clean);
        let r_max = np::max(&clean);
        for j in 0..p {
            tmp[j] = ((clean[j] - r_mean) / r_std).powf(3.0);
        }
        let r_skew = np::clip(np::mean(&tmp), -10.0, 10.0);
        out[i * 8..(i + 1) * 8]
            .copy_from_slice(&[r_mean, r_std, r_l1, r_l2, r_min, r_max, r_skew, miss_frac]);
    }
    out
}

/// `_compute_multiclass_and_target_simplex(Z, y, lvl_r)` on standardized `z`.
pub fn compute_multiclass_and_target_simplex(
    z: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    lvl_r: u8,
) -> [f64; 20] {
    let mut out = [0.0f64; 20];
    let rounded = np::sorted(&y.iter().map(|&v| np::round5(v)).collect::<Vec<_>>());
    let mut u_vals: Vec<f64> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for &v in &rounded {
        if u_vals.last().is_some_and(|&last| last == v) {
            *counts.last_mut().expect("non-empty") += 1;
        } else {
            u_vals.push(v);
            counts.push(1);
        }
    }
    let k = u_vals.len();
    if k <= 1 {
        return out;
    }
    let is_discrete = 2 <= k && k <= 32usize.min((n / 3).max(2));
    out[0] = if is_discrete {
        (k.min(64) as f64).log2() / 6.0
    } else {
        0.0
    };
    if is_discrete {
        let priors: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
        let terms: Vec<f64> = priors.iter().map(|&q| q * q.max(1e-12).ln()).collect();
        let ent = -np::sum(&terms) / (k as f64).ln().max(1e-9);
        out[1] = np::clip(ent, 0.0, 1.0);
        out[2] = np::min(&priors) / np::max(&priors).max(1e-9);
        out[3] = gini(&priors);
        if lvl_r >= 1 {
            let mut centroids = vec![0.0f64; k * p];
            for (c, &val) in u_vals.iter().enumerate() {
                let rows: Vec<usize> = (0..n).filter(|&i| (y[i] - val).abs() <= 1e-4).collect();
                if !rows.is_empty() {
                    let mut sub = Vec::with_capacity(rows.len() * p);
                    for &i in &rows {
                        sub.extend_from_slice(&z[i * p..(i + 1) * p]);
                    }
                    let m = np::mean_axis0(&sub, rows.len(), p);
                    centroids[c * p..(c + 1) * p].copy_from_slice(&m);
                }
            }
            let mut weighted = vec![0.0f64; k * p];
            for c in 0..k {
                for j in 0..p {
                    let v = centroids[c * p + j];
                    weighted[c * p + j] = priors[c] * (v * v);
                }
            }
            let between = np::sum_axis0(&weighted, k, p);
            let ratio: Vec<f64> = between
                .iter()
                .map(|&b| b / np::maximum(1.0 - b, 1e-4))
                .collect();
            let fisher = squash_vec(&ratio);
            let s_fish = np::sorted_desc(&fisher);
            let n_fish = 4.min(s_fish.len());
            out[4..(4 + n_fish)].copy_from_slice(&s_fish[..n_fish]);
            out[8] = np::mean(&fisher);
            out[9] = gini(&fisher);
            if lvl_r >= 2 && k >= 2 {
                let mut scaled = vec![0.0f64; k * p];
                for c in 0..k {
                    let sp = priors[c].sqrt();
                    for j in 0..p {
                        scaled[c * p + j] = sp * centroids[c * p + j];
                    }
                }
                let sv_sq = squash_vec(&np::singular_values(&scaled, k, p));
                let n_sv = 6.min(sv_sq.len());
                out[10..(10 + n_sv)].copy_from_slice(&sv_sq[..n_sv]);
            }
        }
    }
    let y_std = np::std(y).max(1e-9);
    let ym = np::mean(y);
    let y_z: Vec<f64> = y.iter().map(|&v| (v - ym) / y_std).collect();
    let m3: Vec<f64> = y_z.iter().map(|&v| v.powf(3.0)).collect();
    let m4: Vec<f64> = y_z.iter().map(|&v| v.powf(4.0)).collect();
    let ash: Vec<f64> = y_z
        .iter()
        .map(|&v| {
            let a = np::asinh(v);
            a * a
        })
        .collect();
    out[16] = squash(np::mean(&m3));
    out[17] = squash((np::mean(&m4) - 3.0) / 4.0);
    out[18] = squash(np::mean(&ash));
    let s = np::sorted(&y_z);
    let q10 = np::quantile_sorted(&s, 0.10);
    let q25 = np::quantile_sorted(&s, 0.25);
    let q75 = np::quantile_sorted(&s, 0.75);
    let q90 = np::quantile_sorted(&s, 0.90);
    out[19] = squash((q90 - q10) / (q75 - q25).max(1e-4) / 3.0);
    out.map(np::nan_to_zero)
}

pub(crate) fn pack_building_block_descriptors(
    bb_gains16: &[f64; 16],
    row_corrs: &[f64],
    contrasts_raw: &[f64; 16],
    het_and_tail8: &[f64; 8],
    mc_vec20: &[f64; 20],
    col_bb_probes7: Vec<[f32; 7]>,
) -> super::building_blocks::BuildingBlockDescriptors {
    use super::building_blocks::{squash_vec, to_f32, BuildingBlockDescriptors};
    let bb_contrasts16: Vec<f64> = contrasts_raw
        .iter()
        .map(|&v| np::clip(v, -1.0, 1.0))
        .collect();
    let bb_sq = squash_vec(bb_gains16);
    let row_sq = squash_vec(row_corrs);
    let het_sq = squash_vec(het_and_tail8);
    let mut vec68 = Vec::with_capacity(68);
    vec68.extend_from_slice(&bb_sq);
    vec68.extend_from_slice(&row_sq);
    vec68.extend_from_slice(&bb_contrasts16);
    vec68.extend_from_slice(&het_sq);
    vec68.extend_from_slice(mc_vec20);
    let mut building_block_vec68 = [0.0f32; 68];
    for (o, &v) in building_block_vec68.iter_mut().zip(&vec68) {
        let f = v as f32;
        *o = if f.is_finite() { f } else { 0.0 };
    }
    BuildingBlockDescriptors {
        bb_gains16: to_f32(&bb_sq),
        row_summary_corrs8: to_f32(&row_sq),
        col_bb_probes7,
        het_and_tail8: to_f32(&het_sq),
        multiclass_simplex20: to_f32(mc_vec20),
        building_block_vec68,
    }
}

#[cfg(test)]
mod tests {
    use super::np;

    #[test]
    fn argsort_ties_follow_numpy_introsort() {
        let v: Vec<f64> = (0..20).map(|i| (i % 2) as f64).collect();
        let a = np::argsort(&v);
        let mut sorted: Vec<f64> = a.iter().map(|&i| v[i]).collect();
        let check = sorted.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(sorted, check);
    }

    #[test]
    fn asinh_matches_reference_points() {
        for &x in &[0.0f64, 1e-30, 0.3, -1.7, 2.5, -40.0, 1e10] {
            let r = np::asinh(x);
            assert!((r - x.asinh()).abs() <= 4.0 * f64::EPSILON * r.abs().max(1.0));
        }
    }

    #[test]
    fn quantile_matches_numpy_linear() {
        let s = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(np::quantile_sorted(&s, 0.5), 3.0);
        assert_eq!(np::quantile_sorted(&s, 1.0), 8.0);
        assert_eq!(np::quantile_sorted(&s, 0.0), 1.0);
    }
}
