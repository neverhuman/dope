//! Retained fold predictors for the native `K=5` cross-fitted teacher panel under Rule A-1.

use crate::forest::{boost, column_codes, predict_columns, BoostOptions, Forest, Split};
use crate::rng::Rng;
use crate::target::{log_sum_exp, Head};
use std::collections::BTreeMap;

/// Partition row indices `0..n` into `k` Rule A-1 duplicate-grouped folds so exact duplicate
/// `(x_row, y)` rows NEVER straddle two folds (`T01`).
pub fn duplicate_grouped_kfolds(x: &[Vec<f64>], y: &[f64], k: usize, seed: u64) -> Vec<Vec<usize>> {
    let n = y.len();
    let k = k.clamp(2, n.max(2));
    let mut groups: BTreeMap<Vec<u64>, Vec<usize>> = BTreeMap::new();
    for r in 0..n {
        let mut key = Vec::with_capacity(x.len() + 1);
        for col in x {
            let v = col[r];
            key.push(if v.is_nan() {
                0x7ff8_0000_0000_0000
            } else if v == 0.0 {
                0
            } else {
                v.to_bits()
            });
        }
        let yv = y[r];
        key.push(if yv.is_nan() {
            0x7ff8_0000_0000_0000
        } else if yv == 0.0 {
            0
        } else {
            yv.to_bits()
        });
        groups.entry(key).or_default().push(r);
    }
    let mut buckets: Vec<Vec<usize>> = groups.into_values().collect();
    let mut rng = Rng::new(seed ^ 0x7e4c_f01d);
    rng.shuffle(&mut buckets);
    // Stratify round-robin assignment by rounded group target so minority classes never vanish (BUG-R5).
    buckets.sort_by_key(|b| (b.first().map_or(0.0, |&r| y[r]) * 100.0).round() as i64);
    let mut folds = vec![Vec::new(); k];
    for (idx, bucket) in buckets.into_iter().enumerate() {
        let tgt = idx % k;
        folds[tgt].extend(bucket);
    }
    for f in &mut folds {
        f.sort_unstable();
    }
    folds
}

/// Convert raw logits `[n * o]` to calibrated-scale expectations / probabilities `[n * o]`.
pub fn logits_to_expectations(head: Head, logits: &[f64]) -> Vec<f64> {
    let o = head.outputs();
    match head {
        Head::Regression => logits.to_vec(),
        Head::Binary => logits.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect(),
        Head::Multiclass(k) => {
            let n = logits.len() / o;
            let mut out = vec![0.0; n * k];
            for r in 0..n {
                let row = &logits[r * k..(r + 1) * k];
                let lse = log_sum_exp(row);
                for c in 0..k {
                    out[r * k + c] = (row[c] - lse).exp();
                }
            }
            out
        }
    }
}

/// Regularized linear / multinomial logistic member retained in memory for probe scoring.
#[derive(Clone, Debug)]
pub struct LinearMember {
    head: Head,
    d: usize,
    mean: Vec<f64>,
    scale: Vec<f64>,
    weights: Vec<f64>, // [o * (d + 1)] with intercept last
}

impl LinearMember {
    pub fn fit(head: Head, x: &[Vec<f64>], y: &[f64], rows: &[usize]) -> Self {
        let d = x.len();
        let o = head.outputs();
        let n = rows.len().max(1);
        let mut mean = vec![0.0; d];
        let mut scale = vec![1.0; d];
        for j in 0..d {
            let mut s = 0.0;
            let mut cnt = 0usize;
            for &r in rows {
                if x[j][r].is_finite() {
                    s += x[j][r];
                    cnt += 1;
                }
            }
            let m = if cnt > 0 { s / cnt as f64 } else { 0.5 };
            let mut var = 0.0;
            for &r in rows {
                if x[j][r].is_finite() {
                    var += (x[j][r] - m).powi(2);
                }
            }
            mean[j] = m;
            scale[j] = (var / cnt.max(1) as f64).sqrt().max(1e-4);
        }
        let p = d + 1;
        let mut weights = vec![0.0; o * p];
        // Initialize intercepts to target mean / prior logits.
        match head {
            Head::Regression => {
                weights[d] = rows.iter().map(|&r| y[r]).sum::<f64>() / n as f64;
            }
            Head::Binary => {
                let pos = rows.iter().filter(|&&r| y[r] > 0.5).count() as f64;
                let pr = ((pos + 0.5) / (n as f64 + 1.0)).clamp(1e-4, 1.0 - 1e-4);
                weights[d] = (pr / (1.0 - pr)).ln();
            }
            Head::Multiclass(k) => {
                for c in 0..k {
                    let cnt = rows
                        .iter()
                        .filter(|&&r| (y[r].round() as usize).min(k - 1) == c)
                        .count() as f64;
                    weights[c * p + d] = ((cnt + 0.5) / (n as f64 + k as f64 * 0.5)).ln();
                }
            }
        }
        // Sequential cyclic coordinate Newton (monotone descent on correlated columns, BUG-R3).
        let lambda = 0.05 * (n as f64).sqrt().max(1.0);
        let mut phi = vec![0.0; n * p];
        for (i, &r) in rows.iter().enumerate() {
            for j in 0..d {
                phi[i * p + j] = if x[j][r].is_finite() {
                    ((x[j][r] - mean[j]) / scale[j]).clamp(-6.0, 6.0)
                } else {
                    0.0
                };
            }
            phi[i * p + d] = 1.0;
        }
        let mut pred = vec![0.0; n * o];
        for i in 0..n {
            for c in 0..o {
                pred[i * o + c] = weights[c * p + d];
            }
        }
        for _step in 0..25 {
            for j in 0..p {
                let reg_l = if j == d { 1e-4 } else { lambda };
                match head {
                    Head::Regression => {
                        let (mut g, mut h) = (reg_l * weights[j], reg_l);
                        for (i, &r) in rows.iter().enumerate() {
                            let xij = phi[i * p + j];
                            g += (pred[i] - y[r]) * xij;
                            h += xij * xij;
                        }
                        let delta = (-g / h.max(1e-6)).clamp(-2.0, 2.0);
                        let nw = (weights[j] + delta).clamp(-10.0, 10.0);
                        let actual = nw - weights[j];
                        weights[j] = nw;
                        if actual != 0.0 {
                            for i in 0..n {
                                pred[i] += actual * phi[i * p + j];
                            }
                        }
                    }
                    Head::Binary => {
                        let (mut g, mut h) = (reg_l * weights[j], reg_l);
                        for (i, &r) in rows.iter().enumerate() {
                            let xij = phi[i * p + j];
                            let pr = 1.0 / (1.0 + (-pred[i].clamp(-20.0, 20.0)).exp());
                            g += (pr - y[r].clamp(0.0, 1.0)) * xij;
                            h += (pr * (1.0 - pr)).max(1e-4) * xij * xij;
                        }
                        let delta = (-0.75 * g / h.max(1e-6)).clamp(-1.5, 1.5);
                        let nw = (weights[j] + delta).clamp(-8.0, 8.0);
                        let actual = nw - weights[j];
                        weights[j] = nw;
                        if actual != 0.0 {
                            for i in 0..n {
                                pred[i] += actual * phi[i * p + j];
                            }
                        }
                    }
                    Head::Multiclass(k) => {
                        for c in 0..k {
                            let idx_w = c * p + j;
                            let (mut g, mut h) = (reg_l * weights[idx_w], reg_l);
                            for (i, &r) in rows.iter().enumerate() {
                                let xij = phi[i * p + j];
                                let lse = log_sum_exp(&pred[i * o..(i + 1) * o]);
                                let pr = (pred[i * o + c] - lse).exp();
                                let yc = (y[r].round() as usize).min(k - 1);
                                g += (pr - if c == yc { 1.0 } else { 0.0 }) * xij;
                                h += (pr * (1.0 - pr)).max(1e-4) * xij * xij;
                            }
                            let delta = (-0.75 * g / h.max(1e-6)).clamp(-1.5, 1.5);
                            let nw = (weights[idx_w] + delta).clamp(-8.0, 8.0);
                            let actual = nw - weights[idx_w];
                            weights[idx_w] = nw;
                            if actual != 0.0 {
                                for i in 0..n {
                                    pred[i * o + c] += actual * phi[i * p + j];
                                }
                            }
                        }
                    }
                }
            }
        }
        Self {
            head,
            d,
            mean,
            scale,
            weights,
        }
    }

    pub fn predict_rows(&self, x: &[Vec<f64>], rows: &[usize]) -> Vec<f64> {
        let o = self.head.outputs();
        let p = self.d + 1;
        let mut logits = vec![0.0; rows.len() * o];
        for (i, &r) in rows.iter().enumerate() {
            for c in 0..o {
                let mut acc = self.weights[c * p + self.d];
                for j in 0..self.d {
                    if x[j][r].is_finite() {
                        let z = ((x[j][r] - self.mean[j]) / self.scale[j]).clamp(-6.0, 6.0);
                        acc += self.weights[c * p + j] * z;
                    }
                }
                logits[i * o + c] = acc;
            }
        }
        logits_to_expectations(self.head, &logits)
    }
}

/// Retained Oblivious GBDT fold member (depth 4 or depth 6).
#[derive(Clone, Debug)]
pub struct GbdtMember {
    head: Head,
    d: usize,
    impute: Vec<f64>,
    base_bias: Vec<f64>,
    forest: Forest,
}

impl GbdtMember {
    pub fn fit(
        head: Head,
        x: &[Vec<f64>],
        y: &[f64],
        rows: &[usize],
        depth: usize,
        max_trees: usize,
        seed: u64,
    ) -> Self {
        let d = x.len();
        let o = head.outputs();
        let impute: Vec<f64> = (0..d)
            .map(|j| {
                let mut s = 0.0;
                let mut cnt = 0usize;
                for &r in rows {
                    if x[j][r].is_finite() {
                        s += x[j][r];
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    s / cnt as f64
                } else {
                    0.5
                }
            })
            .collect();
        // Rule A-1 duplicate-grouped internal early-stopping split inside complement fold `C`.
        let sub_x: Vec<Vec<f64>> = (0..d)
            .map(|j| rows.iter().map(|&r| x[j][r]).collect())
            .collect();
        let sub_y: Vec<f64> = rows.iter().map(|&r| y[r]).collect();
        let inner = duplicate_grouped_kfolds(&sub_x, &sub_y, 5, seed ^ 0xab12);
        let va: Vec<usize> = inner[0].iter().map(|&i| rows[i]).collect();
        let tr: Vec<usize> = inner[1..]
            .iter()
            .flat_map(|f| f.iter().map(|&i| rows[i]))
            .collect();

        let base_bias = match head {
            Head::Regression => {
                vec![tr.iter().map(|&r| y[r]).sum::<f64>() / tr.len().max(1) as f64]
            }
            Head::Binary => {
                let pos = tr.iter().filter(|&&r| y[r] > 0.5).count() as f64;
                let pr = ((pos + 0.5) / (tr.len() as f64 + 1.0)).clamp(1e-4, 1.0 - 1e-4);
                vec![(pr / (1.0 - pr)).ln()]
            }
            Head::Multiclass(k) => (0..k)
                .map(|c| {
                    let cnt = tr
                        .iter()
                        .filter(|&&r| (y[r].round() as usize).min(k - 1) == c)
                        .count() as f64;
                    ((cnt + 0.5) / (tr.len() as f64 + k as f64 * 0.5)).ln()
                })
                .collect(),
        };
        let cf = column_codes(x, &tr, &impute);
        let cv = column_codes(x, &va, &impute);
        let yf: Vec<f64> = tr.iter().map(|&r| y[r]).collect();
        let yv: Vec<f64> = va.iter().map(|&r| y[r]).collect();
        let bf: Vec<f64> = (0..tr.len())
            .flat_map(|_| base_bias.iter().copied())
            .collect();
        let bv: Vec<f64> = (0..va.len())
            .flat_map(|_| base_bias.iter().copied())
            .collect();
        let boosted = boost(
            head,
            d,
            &Split {
                codes: &cf,
                base: &bf,
                y: &yf,
            },
            &Split {
                codes: &cv,
                base: &bv,
                y: &yv,
            },
            &BoostOptions {
                depth,
                bits: 8,
                eta: 0.15,
                lambda: 1.0,
                max_trees,
                patience: 20,
                min_rows: 10.min((tr.len() / 6).max(2)),
                sample_rows: 20_000,
                seed,
            },
        );
        let best_t = boosted
            .val_path
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(&b.0)))
            .map_or(0, |(t, _)| t);
        let _ = o;
        Self {
            head,
            d,
            impute,
            base_bias,
            forest: boosted.forest.truncate(best_t),
        }
    }

    pub fn predict_rows(&self, x: &[Vec<f64>], rows: &[usize]) -> Vec<f64> {
        let o = self.head.outputs();
        let codes = column_codes(x, rows, &self.impute);
        let mut logits = predict_columns(&self.forest, &codes, self.d, rows.len());
        for r in 0..rows.len() {
            for c in 0..o {
                logits[r * o + c] += self.base_bias[c];
            }
        }
        logits_to_expectations(self.head, &logits)
    }
}

/// Retained 16-NN distance-weighted member (capped reference <= 2048 rows for fast CPU scoring).
#[derive(Clone, Debug)]
pub struct KnnMember {
    head: Head,
    d: usize,
    k_nn: usize,
    ref_x: Vec<Vec<f64>>, // [m][d]
    ref_y: Vec<f64>,
    impute: Vec<f64>,
}

impl KnnMember {
    pub fn fit(head: Head, x: &[Vec<f64>], y: &[f64], rows: &[usize], seed: u64) -> Self {
        let d = x.len();
        let impute: Vec<f64> = (0..d)
            .map(|j| {
                let mut s = 0.0;
                let mut cnt = 0usize;
                for &r in rows {
                    if x[j][r].is_finite() {
                        s += x[j][r];
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    s / cnt as f64
                } else {
                    0.5
                }
            })
            .collect();
        let mut chosen = rows.to_vec();
        if chosen.len() > 2048 {
            let mut rng = Rng::new(seed ^ 0x4444);
            rng.shuffle(&mut chosen);
            chosen.truncate(2048);
        }
        let ref_x: Vec<Vec<f64>> = chosen
            .iter()
            .map(|&r| {
                (0..d)
                    .map(|j| {
                        if x[j][r].is_finite() {
                            x[j][r]
                        } else {
                            impute[j]
                        }
                    })
                    .collect()
            })
            .collect();
        let ref_y: Vec<f64> = chosen.iter().map(|&r| y[r]).collect();
        Self {
            head,
            d,
            k_nn: 16.min(ref_y.len().max(1)),
            ref_x,
            ref_y,
            impute,
        }
    }

    pub fn predict_rows(&self, x: &[Vec<f64>], rows: &[usize]) -> Vec<f64> {
        let o = self.head.outputs();
        let mut out = vec![0.0; rows.len() * o];
        let mut q = vec![0.0; self.d];
        let mut dists: Vec<(f64, usize)> = Vec::with_capacity(self.ref_x.len());
        for (i, &r) in rows.iter().enumerate() {
            for j in 0..self.d {
                q[j] = if x[j][r].is_finite() {
                    x[j][r]
                } else {
                    self.impute[j]
                };
            }
            dists.clear();
            for (idx, rx) in self.ref_x.iter().enumerate() {
                let d2: f64 = q.iter().zip(rx).map(|(&a, &b)| (a - b) * (a - b)).sum();
                dists.push((d2, idx));
            }
            dists.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let k = self.k_nn.min(dists.len());
            let mut sum_w = 0.0;
            for &(d2, idx) in &dists[..k] {
                let w = 1.0 / (d2.sqrt() + 0.05);
                sum_w += w;
                match self.head {
                    Head::Regression => out[i] += w * self.ref_y[idx],
                    Head::Binary => out[i] += w * self.ref_y[idx].clamp(0.0, 1.0),
                    Head::Multiclass(nc) => {
                        let c = (self.ref_y[idx].round() as usize).min(nc - 1);
                        out[i * o + c] += w;
                    }
                }
            }
            if sum_w > 0.0 {
                for c in 0..o {
                    out[i * o + c] /= sum_w;
                }
            }
        }
        out
    }
}
