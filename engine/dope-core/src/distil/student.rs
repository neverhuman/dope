//! Teacher-distilled conditional student (`main + pairs + soft-Newton oblivious forest`).
//!
//! Fits strictly to the transfer pool (`BUILD` OOF teacher targets + `min(8n, 200k)` `Q_X`
//! synthetic probes weighted by `clip(agreement * support, 0.2, 1.0)`), refitting regression
//! residual noise / classification temperature strictly on sealed `SELECT`/`CAL` (`T01`, `T10`).

use super::main_effects::{targets_to_logits, MainEffectSet};
use super::pairs::{add_pair_logits, mine_pairs, PairEffect};
use crate::forest::{
    boost_soft, column_codes, predict_columns, Forest, SoftBoostOptions, SoftSplit,
};
use crate::target::{log_sum_exp, Head};
use crate::teacher::noise::TeacherResidualModel;
use crate::teacher::panel::logits_to_expectations;
use crate::teacher::CrossFittedTeacher;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct DistilConfig {
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
            max_pairs: 12,
            depth: 6,
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

        let impute: Vec<f64> = (0..d)
            .map(|j| {
                let mut s = 0.0;
                let mut cnt = 0usize;
                for &v in &build_x[j] {
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
            .collect();

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

        // Stage 1: Purified `target.main` across ALL columns.
        let main_effects = MainEffectSet::fit(head, &tr_x, &tr_t, &tr_w, &impute);
        let mut base_tr = main_effects.predict_logits(&tr_x, &impute);
        let mut base_mon = main_effects.predict_logits(&mon_x, &impute);

        // Stage 2: Purified `target.pairs` on residual teacher logits.
        let target_logits = targets_to_logits(head, &tr_t, n_tr);
        let res_tr: Vec<f64> = target_logits
            .iter()
            .zip(&base_tr)
            .map(|(&t, &b)| t - b)
            .collect();
        let pair_effects = mine_pairs(&tr_x, &res_tr, &tr_w, &impute, o, cfg.max_pairs);
        add_pair_logits(&pair_effects, &tr_x, &impute, &mut base_tr);
        add_pair_logits(&pair_effects, &mon_x, &impute, &mut base_mon);

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
        let forest_trees = boosted.forest.trees();
        let forest_bytes = Forest::byte_len(
            forest_trees,
            boosted.forest.depth,
            o,
            boosted.forest.bits,
            d,
        );

        let mut student = Self {
            head,
            d,
            impute,
            main_effects,
            pair_effects,
            forest: boosted.forest,
            forest_trees,
            forest_bytes,
            temperature: 1.0,
            residual_model: None,
        };

        // Stage 4: Refit regression noise or classification temperature strictly on `SELECT`/`CAL`.
        if !cal_y.is_empty() {
            let cal_logits = student.predict_logits(cal_x);
            match head {
                Head::Regression => {
                    student.residual_model =
                        Some(TeacherResidualModel::fit_oof(cal_y, &cal_logits));
                }
                Head::Binary | Head::Multiclass(_) => {
                    student.temperature = fit_temperature(head, &cal_logits, cal_y);
                }
            }
        }
        student
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

fn fit_temperature(head: Head, logits: &[f64], y: &[f64]) -> f64 {
    let n = y.len();
    let o = head.outputs();
    let mut best_t = 1.0;
    let mut best_nll = f64::INFINITY;
    for &t in &[0.5, 0.7, 0.85, 1.0, 1.15, 1.35, 1.7, 2.2] {
        let mut nll = 0.0;
        match head {
            Head::Binary => {
                for r in 0..n {
                    let z = logits[r] / t;
                    let ce = if z >= 0.0 {
                        z + (-z).exp().ln_1p()
                    } else {
                        z.exp().ln_1p()
                    } - y[r].clamp(0.0, 1.0) * z;
                    nll += ce;
                }
            }
            Head::Multiclass(k) => {
                for r in 0..n {
                    let row: Vec<f64> = (0..k).map(|c| logits[r * o + c] / t).collect();
                    let lse = log_sum_exp(&row);
                    let c = (y[r].round() as usize).min(k - 1);
                    nll += lse - row[c];
                }
            }
            Head::Regression => return 1.0,
        }
        if nll < best_nll {
            best_nll = nll;
            best_t = t;
        }
    }
    best_t
}
