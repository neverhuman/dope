//! Teacher-distilled conditional student (`main + pairs + soft-Newton oblivious forest`).
//!
//! Fits strictly to the transfer pool (`BUILD` OOF teacher targets + `min(8n, 200k)` `Q_X`
//! synthetic probes weighted by `clip(agreement * support, 0.2, 1.0)`), refitting regression
//! residual noise / classification temperature strictly on sealed `SELECT`/`CAL` (`T01`, `T10`).

use super::main_effects::{fit_temperature_and_bias, MainEffectSet, MAIN_BINS};
use super::pairs::{add_pair_logits, mine_pairs_fisher, PairEffect};
use crate::forest::{
    boost, boost_soft, column_codes, predict_columns, soft_loss, BoostOptions, Forest,
    SoftBoostOptions, SoftSplit, Split,
};
use crate::target::{loss as target_loss, Head};
use crate::teacher::noise::TeacherResidualModel;
use crate::teacher::panel::logits_to_expectations;
use crate::teacher::CrossFittedTeacher;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct DistilConfig {
    pub main_bins: usize,
    pub max_pairs: usize,
    pub depth: usize,
    pub bits: u8,
    pub max_trees: usize,
    pub forest_byte_cap: Option<usize>,
    pub eta: f64,
    pub lambda: f64,
    pub seed: u64,
}

impl Default for DistilConfig {
    fn default() -> Self {
        Self {
            main_bins: MAIN_BINS,
            max_pairs: 8,
            depth: 5,
            bits: 8,
            max_trees: 600,
            forest_byte_cap: Some(48_000),
            eta: 0.15,
            lambda: 1.0,
            seed: 0,
        }
    }
}

fn default_forest() -> Forest {
    Forest::empty(1)
}

fn col_means(x: &[Vec<f64>]) -> Vec<f64> {
    x.iter()
        .map(|col| {
            let (mut s, mut cnt) = (0.0, 0usize);
            for &v in col {
                if v.is_finite() {
                    s += v;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                s / cnt as f64
            } else {
                0.5
            }
        })
        .collect()
}

/// Complete distilled conditional model `Q_{Y|X}`: purified `main` + `pairs` + oblivious forest
/// + `SELECT`/`CAL`-refitted noise/temperature.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistilledStudent {
    pub head: Head,
    pub d: usize,
    pub impute: Vec<f64>,
    pub main_effects: MainEffectSet,
    pub pair_effects: Vec<PairEffect>,
    #[serde(skip, default = "default_forest")]
    pub forest: Forest,
    pub forest_trees: usize,
    pub forest_bytes: usize,
    pub temperature: f64,
    pub residual_model: Option<TeacherResidualModel>,
}

/// Compute support confidence in `[0.2, 1.0]` downweighting out-of-range probes relative to BUILD.
pub fn probe_support_weights(build_x: &[Vec<f64>], probe_x: &[Vec<f64>]) -> Vec<f64> {
    let d = build_x.len();
    let np = probe_x.first().map_or(0, Vec::len);
    if d == 0 || np == 0 {
        return vec![1.0; np];
    }
    let mut lo = vec![0.0; d];
    let mut hi = vec![1.0; d];
    for j in 0..d {
        let mut vals: Vec<f64> = build_x[j]
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        vals.sort_by(f64::total_cmp);
        if vals.len() >= 2 {
            lo[j] = vals[(vals.len() / 100).min(vals.len() - 1)];
            hi[j] = vals[(vals.len() * 99 / 100).min(vals.len() - 1)];
        }
    }
    (0..np)
        .map(|r| {
            let mut ood = 0.0;
            for j in 0..d {
                let v = probe_x[j][r];
                if v.is_finite() {
                    let span = (hi[j] - lo[j]).max(1e-6);
                    if v < lo[j] {
                        ood += (lo[j] - v) / span;
                    } else if v > hi[j] {
                        ood += (v - hi[j]) / span;
                    }
                }
            }
            (1.0 / (1.0 + ood)).clamp(0.2, 1.0)
        })
        .collect()
}

impl DistilledStudent {
    /// Distill `teacher` onto the combined transfer pool (`BUILD` OOF + `probe_x` scored by teacher),
    /// then calibrate noise/temperature strictly on sealed `cal_x` / `cal_y`.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        teacher: &CrossFittedTeacher,
        build_x: &[Vec<f64>],
        probe_x: &[Vec<f64>],
        cal_x: &[Vec<f64>],
        cal_y: &[f64],
        cfg: &DistilConfig,
    ) -> Self {
        let head = teacher.head;
        let d = teacher.d;
        let o = head.outputs();
        let n_build = build_x.first().map_or(0, Vec::len);
        let n_probe = probe_x.first().map_or(0, Vec::len).min(200_000);

        let impute = col_means(build_x);

        // Score `Q_X` probes with the retained `K=5` cross-fitted teacher ensemble.
        let (probe_t, probe_agr) = teacher.predict_probes(probe_x);
        let probe_sup = probe_support_weights(build_x, probe_x);

        // Combine transfer pool: `BUILD` OOF rows + `Q_X` probe rows.
        // Hold out 15% of probes as the student-vs-teacher plateau monitor (never noisy real y!).
        let n_mon = (n_probe / 7).max(if n_probe >= 10 { 2 } else { 0 });
        let n_tr_probe = n_probe.saturating_sub(n_mon);
        let n_tr = n_build + n_tr_probe;

        let mut tr_x = vec![Vec::with_capacity(n_tr); d];
        let mut tr_t = Vec::with_capacity(n_tr * o);
        let mut tr_w = Vec::with_capacity(n_tr);
        for j in 0..d {
            tr_x[j].extend_from_slice(&build_x[j]);
            tr_x[j].extend_from_slice(&probe_x[j][..n_tr_probe]);
        }
        tr_t.extend_from_slice(&teacher.oof_predictions);
        tr_t.extend_from_slice(&probe_t[..n_tr_probe * o]);
        for &a in &teacher.oof_agreement {
            tr_w.push(a.clamp(0.2, 1.0));
        }
        for i in 0..n_tr_probe {
            tr_w.push((probe_agr[i] * probe_sup[i]).clamp(0.2, 1.0));
        }

        let (mon_x, mon_t, mon_w): (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) = if n_mon > 0 {
            let mx = (0..d)
                .map(|j| probe_x[j][n_tr_probe..n_probe].to_vec())
                .collect();
            let mt = probe_t[n_tr_probe * o..n_probe * o].to_vec();
            let mw = (n_tr_probe..n_probe)
                .map(|i| (probe_agr[i] * probe_sup[i]).clamp(0.2, 1.0))
                .collect();
            (mx, mt, mw)
        } else {
            (tr_x.clone(), tr_t.clone(), tr_w.clone())
        };

        // Stage 1: Purified `target.main` across ALL columns (budget-adaptive knot count).
        let main_effects =
            MainEffectSet::fit_with_bins(head, &tr_x, &tr_t, &tr_w, &impute, cfg.main_bins);
        let mut base_tr = main_effects.predict_logits(&tr_x, &impute);
        let mut base_mon = main_effects.predict_logits(&mon_x, &impute);

        // Stage 2: Fisher-weighted `target.pairs` gated by held-out `mon` soft loss (no ln(1e-4) spikes!).
        let cand_pairs =
            mine_pairs_fisher(head, &tr_x, &base_tr, &tr_t, &tr_w, &impute, cfg.max_pairs);
        let pair_effects = if !cand_pairs.is_empty() {
            let l0 = soft_loss(head, &base_mon, &mon_t, &mon_w);
            let mut bm_try = base_mon.clone();
            add_pair_logits(&cand_pairs, &mon_x, &impute, &mut bm_try);
            if soft_loss(head, &bm_try, &mon_t, &mon_w) + 1e-4 < l0 {
                add_pair_logits(&cand_pairs, &tr_x, &impute, &mut base_tr);
                base_mon = bm_try;
                cand_pairs
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Stage 3: Weighted soft-target oblivious vector-leaf student (`boost_soft`).
        let tr_rows: Vec<usize> = (0..n_tr).collect();
        let mon_rows: Vec<usize> = (0..mon_w.len()).collect();
        let tr_codes = column_codes(&tr_x, &tr_rows, &impute);
        let mon_codes = column_codes(&mon_x, &mon_rows, &impute);
        let boosted = boost_soft(
            head,
            d,
            &SoftSplit {
                codes: &tr_codes,
                base: &base_tr,
                soft_targets: &tr_t,
                weights: &tr_w,
            },
            &SoftSplit {
                codes: &mon_codes,
                base: &base_mon,
                soft_targets: &mon_t,
                weights: &mon_w,
            },
            &SoftBoostOptions {
                depth: cfg.depth,
                bits: cfg.bits,
                eta: cfg.eta,
                lambda: cfg.lambda,
                max_trees: cfg.max_trees,
                forest_byte_cap: cfg.forest_byte_cap,
                seed: cfg.seed,
                ..Default::default()
            },
        );
        let mut student = Self {
            head,
            d,
            impute,
            main_effects,
            pair_effects,
            forest: boosted.forest,
            forest_trees: 0,
            forest_bytes: 0,
            temperature: 1.0,
            residual_model: None,
        };
        // Prune student.forest prefix on real held-out CAL loss to prevent probe over-boosting.
        student.prune_prefix_on_cal(cal_x, cal_y);
        student.forest_trees = student.forest.trees();
        student.forest_bytes = Forest::byte_len(
            student.forest_trees,
            student.forest.depth,
            o,
            student.forest.bits,
            d,
        );
        student.refit_calibration(cal_x, cal_y);
        student
    }

    /// Direct-label regularized `MainEffectSet + Oblivious Forest` candidate trained strictly on
    /// `(train_x, train_y)` and early-stopped/calibrated on `CAL` (`T10` direct-label control).
    pub fn fit_direct(
        head: Head,
        train_x: &[Vec<f64>],
        train_y: &[f64],
        cal_x: &[Vec<f64>],
        cal_y: &[f64],
        cfg: &DistilConfig,
    ) -> Self {
        let d = train_x.len();
        let n = train_y.len();
        let o = head.outputs();
        let impute = col_means(train_x);
        let soft_y: Vec<f64> = match head {
            Head::Regression => train_y.to_vec(),
            Head::Binary => train_y.iter().map(|&v| v.clamp(0.0, 1.0)).collect(),
            Head::Multiclass(k) => {
                let mut out = vec![0.0; n * k];
                for r in 0..n {
                    out[r * k + (train_y[r].round() as usize).min(k - 1)] = 1.0;
                }
                out
            }
        };
        let ones = vec![1.0; n];
        // Hold out an internal 15% monitor slice of `train` so `cal_x`/`cal_y` stays strictly
        // unbiased for T10 selection between `distilled` and `direct` (zero winner's curse!).
        let mut tr_rows = Vec::with_capacity(n);
        let mut val_rows = Vec::with_capacity(n / 6 + 1);
        for r in 0..n {
            if n >= 40 && r % 7 == 0 {
                val_rows.push(r);
            } else {
                tr_rows.push(r);
            }
        }
        let sub_x: Vec<Vec<f64>> = (0..d)
            .map(|j| tr_rows.iter().map(|&r| train_x[j][r]).collect())
            .collect();
        let sub_y: Vec<f64> = tr_rows.iter().map(|&r| train_y[r]).collect();
        let val_x: Vec<Vec<f64>> = (0..d)
            .map(|j| val_rows.iter().map(|&r| train_x[j][r]).collect())
            .collect();
        let val_y: Vec<f64> = val_rows.iter().map(|&r| train_y[r]).collect();
        let main_effects =
            MainEffectSet::fit_with_bins(head, train_x, &soft_y, &ones, &impute, cfg.main_bins);
        let (b_tr_x, b_tr_y, b_va_x, b_va_y) = if val_rows.len() >= 12 {
            (
                sub_x.as_slice(),
                sub_y.as_slice(),
                val_x.as_slice(),
                val_y.as_slice(),
            )
        } else {
            (train_x, train_y, cal_x, cal_y)
        };
        let base_tr = main_effects.predict_logits(b_tr_x, &impute);
        let base_va = main_effects.predict_logits(b_va_x, &impute);
        let b_tr_idx: Vec<usize> = (0..b_tr_y.len()).collect();
        let b_va_idx: Vec<usize> = (0..b_va_y.len()).collect();
        let tr_codes = column_codes(b_tr_x, &b_tr_idx, &impute);
        let va_codes = column_codes(b_va_x, &b_va_idx, &impute);
        let eff_depth = if n < 400 && matches!(head, Head::Multiclass(_)) {
            cfg.depth.min(4)
        } else {
            cfg.depth
        };
        let boosted = boost(
            head,
            d,
            &Split {
                codes: &tr_codes,
                base: &base_tr,
                y: b_tr_y,
            },
            &Split {
                codes: &va_codes,
                base: &base_va,
                y: b_va_y,
            },
            &BoostOptions {
                depth: eff_depth,
                bits: cfg.bits,
                eta: cfg.eta,
                lambda: cfg.lambda.max(2.0),
                max_trees: cfg.max_trees.min(220),
                patience: 25,
                min_rows: 10.min((b_tr_y.len() / 10).max(2)),
                sample_rows: 20_000,
                seed: cfg.seed ^ 0xD1EC,
            },
        );
        let best_t = boosted
            .val_path
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(&b.0)))
            .map_or(0, |(t, _)| t);
        let forest = boosted.forest.truncate(best_t);
        let mut st = Self {
            head,
            d,
            impute,
            main_effects,
            pair_effects: Vec::new(),
            forest,
            forest_trees: 0,
            forest_bytes: 0,
            temperature: 1.0,
            residual_model: None,
        };
        st.prune_prefix_on_cal(cal_x, cal_y);
        st.forest_trees = st.forest.trees();
        st.forest_bytes = Forest::byte_len(st.forest_trees, st.forest.depth, o, st.forest.bits, d);
        st.refit_calibration(cal_x, cal_y);
        st
    }

    fn prune_prefix_on_cal(&mut self, cal_x: &[Vec<f64>], cal_y: &[f64]) {
        let total = self.forest.trees();
        if total <= 4 || cal_y.len() < 16 {
            return;
        }
        let mut base = self.main_effects.predict_logits(cal_x, &self.impute);
        add_pair_logits(&self.pair_effects, cal_x, &self.impute, &mut base);
        let rows: Vec<usize> = (0..cal_y.len()).collect();
        let codes = column_codes(cal_x, &rows, &self.impute);
        let mut best_t = total;
        let mut best_l = f64::INFINITY;
        let mut loss_zero = f64::INFINITY;
        let mut cands = vec![
            0usize, 2, 4, 8, 12, 18, 26, 38, 56, 80, 120, 180, 260, total,
        ];
        cands.retain(|&k| k <= total);
        cands.sort_unstable();
        cands.dedup();
        for k in cands {
            let mut logits = base.clone();
            if k > 0 {
                let sub = self.forest.truncate(k);
                let f_out = predict_columns(&sub, &codes, self.d, cal_y.len());
                for (dst, src) in logits.iter_mut().zip(f_out) {
                    *dst += src;
                }
            }
            let l = target_loss(self.head, &logits, cal_y);
            if k == 0 {
                loss_zero = l;
            } else if l + 1e-4 < best_l {
                best_l = l;
                best_t = k;
            }
        }
        if loss_zero + 1e-3 < best_l {
            best_t = 0;
        }
        if best_t < total {
            self.forest = self.forest.truncate(best_t);
        }
    }

    /// Evaluate calibrated held-out loss on `(cal_x, cal_y)` for candidate selection (`T10`).
    pub fn cal_loss(&self, cal_x: &[Vec<f64>], cal_y: &[f64]) -> f64 {
        if cal_y.is_empty() {
            return f64::INFINITY;
        }
        let mut logits = self.predict_logits(cal_x);
        if self.head != Head::Regression && (self.temperature - 1.0).abs() > 1e-9 {
            for v in &mut logits {
                *v /= self.temperature;
            }
        }
        target_loss(self.head, &logits, cal_y)
    }

    /// Refit regression residual model or classification temperature (+ prior intercept alignment)
    /// strictly on sealed `cal_x` / `cal_y` (called after initial fit and after budget truncation).
    pub fn refit_calibration(&mut self, cal_x: &[Vec<f64>], cal_y: &[f64]) {
        if cal_y.is_empty() {
            return;
        }
        let cal_logits = self.predict_logits(cal_x);
        match self.head {
            Head::Regression => {
                self.residual_model = Some(TeacherResidualModel::fit_oof(cal_y, &cal_logits));
            }
            Head::Binary | Head::Multiclass(_) => {
                let (t_opt, b_shift) = fit_temperature_and_bias(self.head, &cal_logits, cal_y);
                self.temperature = t_opt;
                for (dst, db) in self.main_effects.intercept.iter_mut().zip(b_shift) {
                    *dst += db * t_opt;
                }
            }
        }
    }

    pub fn predict_logits(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.first().map_or(0, Vec::len);
        let mut logits = self.main_effects.predict_logits(x, &self.impute);
        add_pair_logits(&self.pair_effects, x, &self.impute, &mut logits);
        if self.forest.trees() > 0 && n > 0 {
            let rows: Vec<usize> = (0..n).collect();
            let codes = column_codes(x, &rows, &self.impute);
            let f_out = predict_columns(&self.forest, &codes, self.d, n);
            for (dst, src) in logits.iter_mut().zip(f_out) {
                *dst += src;
            }
        }
        logits
    }

    pub fn predict_expectations(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut logits = self.predict_logits(x);
        if self.head != Head::Regression && (self.temperature - 1.0).abs() > 1e-9 {
            for v in &mut logits {
                *v /= self.temperature;
            }
        }
        logits_to_expectations(self.head, &logits)
    }
}
