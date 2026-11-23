//! Tie-disattenuated Gaussian factor copula and populated multi-regime factor means (`Rg`)
//! for `DSE8` v1 (`dope_core::single_encoding::v8`).

use crate::evidence::primitives::{tie_aware_normal_scores, SUPPORT_FLOOR_K};
use crate::stats::{nearest_correlation, pearson, sym_eigen};

pub const V8_MAX_RANK: usize = 12;
pub const V8_MAX_REGIMES: usize = 8;
pub const V8_REGIME_RANK: usize = 6;

/// One factor-space regime (class-conditioned for cls, k-means for reg; k >= 10 enforced).
#[derive(Clone, Debug)]
pub struct RegimeV8 {
    pub weight: f32,
    pub mean_f: Vec<f32>, // length r_reg <= 6
}

/// Multi-regime factor mixture (`TAG_REGIMES`, `0` bytes if single-regime!).
#[derive(Clone, Debug, Default)]
pub struct RegimesSectionV8 {
    pub r_reg: usize,
    pub regimes: Vec<RegimeV8>,
}

/// Return bundle from `fit_copula_loadings_v8`:
/// `(loadings_i8, loadings_f32, z_tie_mat, factor_scores)`.
pub type CopulaFitOut = (
    Vec<Vec<i8>>,
    Vec<[f32; V8_MAX_RANK]>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
);

/// Fit tie-disattenuated PD-projected rank-`rank_use` factor loadings quantized to `i8`,
/// and return `(loadings_i8, loadings_f32, z_tie_mat, factor_scores)`.
pub fn fit_copula_loadings_v8(x_unit: &[Vec<f32>], rank_use: usize) -> CopulaFitOut {
    let p = x_unit.len();
    let n = x_unit.first().map(|c| c.len()).unwrap_or(0);
    let r = rank_use.min(V8_MAX_RANK).min(p);

    let mut z_tie = Vec::with_capacity(p);
    let mut kappa = Vec::with_capacity(p);
    let mut is_const = vec![false; p];
    for col in x_unit {
        let xf: Vec<f64> = col.iter().map(|&v| v as f64).collect();
        let (z, k, n_obs) = tie_aware_normal_scores(&xf);
        let z_var = z.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
        is_const[z_tie.len()] = n_obs < 4 || z_var < 1e-6;
        z_tie.push(z);
        kappa.push(k);
    }

    let mut r_raw = vec![0.0f64; p * p];
    for i in 0..p {
        r_raw[i * p + i] = 1.0;
    }
    for i in 0..p {
        if is_const[i] {
            continue;
        }
        for j in (i + 1)..p {
            if is_const[j] {
                continue;
            }
            let rz = pearson(&z_tie[i], &z_tie[j]);
            let r_lat = (rz / (kappa[i] * kappa[j]).max(0.0225)).clamp(-0.999, 0.999);
            r_raw[i * p + j] = r_lat;
            r_raw[j * p + i] = r_lat;
        }
    }
    let r_lat = if p >= 2 {
        nearest_correlation(&r_raw, p, 1e-4)
    } else {
        r_raw
    };
    let (eig_vals, eig_vecs) = if p >= 1 {
        sym_eigen(&r_lat, p)
    } else {
        (Vec::new(), Vec::new())
    };

    let mut l_f64 = vec![vec![0.0f64; r]; p];
    for k in 0..r {
        let s = eig_vals[k].max(0.0).sqrt();
        let sum_cube: f64 = (0..p).map(|i| eig_vecs[i * p + k].powi(3)).sum();
        let sign = if sum_cube < 0.0 { -1.0 } else { 1.0 };
        for i in 0..p {
            l_f64[i][k] = sign * eig_vecs[i * p + k] * s;
        }
    }
    for row in &mut l_f64 {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.999 {
            let inv = 0.999 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    let mut l_i8 = vec![vec![0i8; r]; p];
    let mut l_f32 = vec![[0.0f32; V8_MAX_RANK]; p];
    for i in 0..p {
        for k in 0..r {
            let q = (l_f64[i][k] * 127.0).round().clamp(-127.0, 127.0) as i8;
            l_i8[i][k] = q;
            l_f32[i][k] = q as f32 / 127.0;
        }
    }

    // Approximate per-row factor scores f_hat[row][k] = (L^T z)_k / max(eig_k, 1.0)
    let mut factor_scores = vec![vec![0.0f64; r]; n];
    for row in 0..n {
        for k in 0..r {
            let denom = eig_vals.get(k).copied().unwrap_or(1.0).max(0.5);
            let mut dot = 0.0;
            for i in 0..p {
                dot += l_f64[i][k] * z_tie[i][row];
            }
            factor_scores[row][k] = (dot / denom).clamp(-3.5, 3.5);
        }
    }

    (l_i8, l_f32, z_tie, factor_scores)
}

/// Fit populated multi-regime factor means (`Rg`) strictly respecting `k >= 10` support floor.
/// Returns empty section (`0` bytes!) when single-regime suffices.
pub fn fit_regimes_v8(
    factor_scores: &[Vec<f64>],
    y_sc: &[f64],
    task_id: u8,
    k_eff: usize,
    max_regimes: usize,
) -> RegimesSectionV8 {
    let n = factor_scores.len();
    let r = factor_scores.first().map(|v| v.len()).unwrap_or(0);
    let r_reg = r.min(V8_REGIME_RANK);
    let m_cap = max_regimes.min(V8_MAX_REGIMES);
    if n < 2 * SUPPORT_FLOOR_K || r_reg == 0 || m_cap < 2 {
        return RegimesSectionV8::default();
    }

    let assign: Vec<usize> = if task_id != 0 {
        // Classification: regime per FIT class (preserves exact class-conditioned X geometry!).
        y_sc.iter()
            .map(|&v| (v.round() as isize).clamp(0, (k_eff.min(m_cap) - 1) as isize) as usize)
            .collect()
    } else {
        // Regression: deterministic k-means in factor space (choose K in 2..=m_cap.min(6)).
        let k_try = (n / (4 * SUPPORT_FLOOR_K)).clamp(2, m_cap.min(6));
        kmeans_factor_assign(factor_scores, r_reg, k_try)
    };

    let n_cand = assign.iter().copied().max().unwrap_or(0) + 1;
    let mut sums = vec![vec![0.0f64; r_reg]; n_cand];
    let mut cnts = vec![0usize; n_cand];
    for (row, &g) in assign.iter().enumerate() {
        cnts[g] += 1;
        for k in 0..r_reg {
            sums[g][k] += factor_scores[row][k];
        }
    }

    // Standardize factor scores per dim so regime means live in exact N(0,1) factor coordinates.
    let mut f_sd = vec![1.0f64; r_reg];
    for k in 0..r_reg {
        let m = factor_scores.iter().map(|row| row[k]).sum::<f64>() / n as f64;
        let v = factor_scores
            .iter()
            .map(|row| (row[k] - m).powi(2))
            .sum::<f64>()
            / n as f64;
        f_sd[k] = v.sqrt().max(0.25);
    }

    // Only keep regimes meeting the strict k >= 10 privacy/support floor!
    let mut kept = Vec::new();
    let mut tot_kept = 0usize;
    for g in 0..n_cand {
        if cnts[g] >= SUPPORT_FLOOR_K {
            tot_kept += cnts[g];
            let mean_f: Vec<f32> = (0..r_reg)
                .map(|k| ((sums[g][k] / (cnts[g] as f64 * f_sd[k])).clamp(-0.85, 0.85)) as f32)
                .collect();
            kept.push((cnts[g], mean_f));
        }
    }
    if kept.len() < 2 || tot_kept == 0 {
        return RegimesSectionV8::default();
    }
    // Zero-center kept regime means (weighted by regime share) so E[z_factor] == 0 exactly!
    for k in 0..r_reg {
        let mu_bar: f64 = kept
            .iter()
            .map(|(c, mf)| (*c as f64 / tot_kept as f64) * mf[k] as f64)
            .sum();
        for (_, mf) in &mut kept {
            mf[k] = (mf[k] as f64 - mu_bar).clamp(-0.85, 0.85) as f32;
        }
    }

    // Check between-regime variance share: skip if centroids are trivially near zero!
    let mut bet_var = 0.0f64;
    let tot_var = (tot_kept * r_reg) as f64;
    for (c, mf) in &kept {
        let sq: f64 = mf.iter().map(|&v| (v as f64).powi(2)).sum();
        bet_var += (*c as f64) * sq;
    }
    let r2_reg = bet_var / tot_var.max(1e-6);
    if r2_reg < 0.05 {
        return RegimesSectionV8::default();
    }

    let regimes = kept
        .into_iter()
        .map(|(c, mean_f)| RegimeV8 {
            weight: c as f32 / tot_kept as f32,
            mean_f,
        })
        .collect();
    RegimesSectionV8 { r_reg, regimes }
}

fn kmeans_factor_assign(scores: &[Vec<f64>], r_reg: usize, k: usize) -> Vec<usize> {
    let n = scores.len();
    // Deterministic farthest-first / quantile-strided initialization on norm.
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&a, &b| {
        let na: f64 = scores[a][..r_reg].iter().map(|v| v * v).sum();
        let nb: f64 = scores[b][..r_reg].iter().map(|v| v * v).sum();
        na.total_cmp(&nb).then(a.cmp(&b))
    });
    let mut centers: Vec<Vec<f64>> = (0..k)
        .map(|c| scores[ord[(c * n + n / (2 * k)).min(n - 1)]][..r_reg].to_vec())
        .collect();
    let mut assign = vec![0usize; n];
    for _ in 0..10 {
        for (i, row) in scores.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f64::INFINITY;
            for (c, cen) in centers.iter().enumerate() {
                let d: f64 = (0..r_reg).map(|j| (row[j] - cen[j]).powi(2)).sum();
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            assign[i] = best_c;
        }
        let mut sums = vec![vec![0.0; r_reg]; k];
        let mut cnts = vec![0usize; k];
        for (i, &c) in assign.iter().enumerate() {
            cnts[c] += 1;
            for j in 0..r_reg {
                sums[c][j] += scores[i][j];
            }
        }
        for c in 0..k {
            if cnts[c] > 0 {
                for j in 0..r_reg {
                    centers[c][j] = sums[c][j] / cnts[c] as f64;
                }
            }
        }
    }
    assign
}

impl RegimesSectionV8 {
    pub fn encode_bytes(&self, out: &mut Vec<u8>) {
        if self.regimes.len() < 2 || self.r_reg == 0 {
            return;
        }
        out.push(self.regimes.len() as u8);
        out.push(self.r_reg as u8);
        for reg in &self.regimes {
            out.push((reg.weight * 255.0).round().clamp(1.0, 255.0) as u8);
            for &m in reg.mean_f.iter().take(self.r_reg) {
                // Quantize mean in [-2.5, 2.5] -> i8 [-125, 125] (step 0.02)
                out.push(((m * 50.0).round().clamp(-125.0, 125.0) as i8) as u8);
            }
        }
    }

    pub fn decode_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 2 {
            return None;
        }
        let m = bytes[0] as usize;
        let r_reg = bytes[1] as usize;
        if !(2..=V8_MAX_REGIMES).contains(&m) || r_reg == 0 || r_reg > V8_REGIME_RANK {
            return None;
        }
        let stride = 1 + r_reg;
        if bytes.len() != 2 + m * stride {
            return None;
        }
        let mut cur = 2;
        let mut regimes = Vec::with_capacity(m);
        let mut sum_w = 0.0f32;
        for _ in 0..m {
            let w = bytes[cur] as f32 / 255.0;
            cur += 1;
            let mut mean_f = Vec::with_capacity(r_reg);
            for _ in 0..r_reg {
                mean_f.push((bytes[cur] as i8) as f32 / 50.0);
                cur += 1;
            }
            sum_w += w;
            regimes.push(RegimeV8 { weight: w, mean_f });
        }
        if sum_w > 1e-6 {
            for r in &mut regimes {
                r.weight /= sum_w;
            }
        }
        Some(Self { r_reg, regimes })
    }
}
