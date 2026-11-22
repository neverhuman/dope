//! Row-local analytic sampler: features, then the target.
//!
//! Row `i` is a pure function of `(packet, seed, i)`:
//!
//! ```text
//! c   ~ Cat(pi)                          regime (binary / multiclass with rank > 0)
//! f_k ~ N(0, 1), k < rank
//! z_j = l_j . (m_c + f) + sqrt(psi_j) e_j,   psi_j = max(0.02, 1 - |l_j|^2)
//! u_j = sum_c pi_c Phi((z_j - l_j . m_c) / s_j),   s_j^2 = |l_j|^2 + psi_j
//! x_j = Q_j(ppf(u_j))                    knot curve or atom table
//! eta = b + w . x + main effects + pairs (`linpred`), on the decoded x
//! y   = clip(eta + sd * e, +-6)          regression, standardised units
//! y   ~ Cat(softmax(T * eta))            classes with a non-zero prior, sampled
//! ```
//!
//! `u_j` is the exact mixture CDF of `z_j`, so no batch statistic is needed and
//! every draw is counter based: `hash3(seed, purpose << 32 | draw, i)`. With a
//! single regime `ppf(u_j)` reduces to `z_j / s_j` and no CDF is evaluated.
//! A label is always a draw, never an arg-max.

use super::dependence::{LOADING_SCALE, MEAN_STEP, PSI_FLOOR};
use super::glm::sigmoid;
use super::linpred::{noise_sd, temperature, TargetModel};
use super::marginals::{knot_plan, ColumnSampler, UNIT_CLIP};
use super::packet::{Packet, Task, MAX_RANK};
use super::scale::ScaleCard;
use super::StatError;
use dope_core::rng::{normal, uniform};
use dope_core::stats::{norm_cdf, norm_ppf};

/// Counter streams: purpose in the high 32 bits, draw index in the low 32.
pub const STREAM_REGIME: u64 = 1 << 32;
pub const STREAM_FACTOR: u64 = 2 << 32;
pub const STREAM_NOISE: u64 = 3 << 32;
/// Target noise (draw 0), class label (draw 0) or multilabel cell (draw = label).
pub const STREAM_TARGET: u64 = 4 << 32;

/// How a class label is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelRule {
    /// `y ~ softmax(T * eta(x))`, a separate draw from the feature regime (the default).
    Sampled,
    /// The label is the regime that generated the features; without regimes
    /// (level A, regression, multilabel) this falls back to `Sampled`.
    Regime,
}

#[derive(Clone, Debug)]
pub struct Decoder {
    p: usize,
    rank: usize,
    task: Task,
    /// Regime probabilities; empty when the sampler has a single regime.
    prior: Vec<f64>,
    prior_cum: Vec<f64>,
    /// Last regime with a non-zero prior.
    last_regime: usize,
    /// `regimes x rank` factor means.
    factor_mean: Vec<f64>,
    /// `p x rank` loadings.
    loading: Vec<f64>,
    noise_sd: Vec<f64>,
    total_sd: Vec<f64>,
    /// `p x regimes` values of `l_j . m_c`.
    shift: Vec<f64>,
    columns: Vec<ColumnSampler>,
    target: TargetModel,
    /// Classes that can be drawn (binary, multiclass) or the label rate codes (multilabel).
    class_on: Vec<bool>,
    label_code: Vec<u8>,
    target_sd: f64,
    temperature: f64,
    rule: LabelRule,
}

impl Decoder {
    pub fn new(packet: &Packet) -> Result<Self, StatError> {
        packet.validate()?;
        let (p, rank) = (packet.cols(), packet.level.rank());
        let class_task = matches!(packet.task, Task::Binary | Task::Multiclass);
        let mut share = Vec::new();
        if class_task {
            let head: u32 = packet.prior.iter().map(|&c| c as u32).sum();
            share = packet.prior.iter().map(|&c| c as f64 / 255.0).collect();
            share.push((255 - head) as f64 / 255.0);
        }
        let class_on: Vec<bool> = share.iter().map(|&q| q > 0.0).collect();
        let prior = if rank > 0 { share } else { Vec::new() };
        let regimes = prior.len();
        let mut running = 0.0;
        let prior_cum = prior
            .iter()
            .map(|q| {
                running += q;
                running
            })
            .collect();
        let last_regime = prior.iter().rposition(|&q| q > 0.0).unwrap_or(0);
        let factor_mean: Vec<f64> = (0..regimes * rank)
            .map(|i| packet.class_mean[i / rank * MAX_RANK + i % rank] as f64 * MEAN_STEP)
            .collect();
        let loading: Vec<f64> = (0..p * rank)
            .map(|i| packet.loading[i / rank * MAX_RANK + i % rank] as f64 / LOADING_SCALE)
            .collect();
        let (mut noise, mut total_sd) = (vec![1.0; p], vec![1.0; p]);
        let mut shift = vec![0.0; p * regimes];
        for j in 0..p {
            let l = &loading[j * rank..(j + 1) * rank];
            let common: f64 = l.iter().map(|v| v * v).sum();
            let psi = (1.0 - common).max(PSI_FLOOR);
            noise[j] = psi.sqrt();
            total_sd[j] = (common + psi).sqrt();
            for c in 0..regimes {
                let m = &factor_mean[c * rank..(c + 1) * rank];
                shift[j * regimes + c] = l.iter().zip(m).map(|(a, b)| a * b).sum();
            }
        }
        let plan = knot_plan(packet.log2_n);
        let columns: Vec<ColumnSampler> = packet
            .columns
            .iter()
            .map(|col| ColumnSampler::new(col, packet.level, &plan))
            .collect();
        let target = TargetModel::new(packet, packet.level, &columns);
        Ok(Self {
            p,
            rank,
            task: packet.task,
            prior,
            prior_cum,
            last_regime,
            factor_mean,
            loading,
            noise_sd: noise,
            total_sd,
            shift,
            columns,
            target,
            class_on,
            label_code: if packet.task == Task::Multilabel {
                packet.prior.clone()
            } else {
                Vec::new()
            },
            target_sd: noise_sd(packet, packet.level),
            temperature: temperature(packet),
            rule: LabelRule::Sampled,
        })
    }

    /// The same decoder with another label rule (the default is [`LabelRule::Sampled`]).
    pub fn with_label_rule(mut self, rule: LabelRule) -> Self {
        self.rule = rule;
        self
    }

    pub fn cols(&self) -> usize {
        self.p
    }

    /// Target cells per row: one, or the label count for multilabel.
    pub fn targets(&self) -> usize {
        match self.task {
            Task::Multilabel => self.label_code.len(),
            _ => 1,
        }
    }

    /// Writes unit-space row `row` into `out` (length P) and returns its regime.
    pub fn row(&self, seed: u64, row: u64, out: &mut [f64]) -> usize {
        let regimes = self.prior.len();
        let mut regime = 0;
        let mut f = [0.0f64; MAX_RANK];
        for (k, slot) in f.iter_mut().enumerate().take(self.rank) {
            *slot = normal(seed, STREAM_FACTOR + k as u64, row);
        }
        if regimes > 1 {
            let u = uniform(seed, STREAM_REGIME, row);
            let c = self.prior_cum.iter().take_while(|&&q| q <= u).count();
            regime = c.min(self.last_regime);
            let mean = &self.factor_mean[regime * self.rank..(regime + 1) * self.rank];
            f.iter_mut().zip(mean).for_each(|(v, m)| *v += m);
        }
        for (j, slot) in out.iter_mut().enumerate().take(self.p) {
            if matches!(self.columns[j], ColumnSampler::Constant) {
                *slot = 0.0;
                continue;
            }
            let l = &self.loading[j * self.rank..(j + 1) * self.rank];
            let common: f64 = l.iter().zip(&f).map(|(a, b)| a * b).sum();
            let z = common + self.noise_sd[j] * normal(seed, STREAM_NOISE + j as u64, row);
            let zq = if regimes > 1 {
                let shift = &self.shift[j * regimes..(j + 1) * regimes];
                let u: f64 = self
                    .prior
                    .iter()
                    .zip(shift)
                    .map(|(q, a)| q * norm_cdf((z - a) / self.total_sd[j]))
                    .sum();
                norm_ppf(u.clamp(1e-12, 1.0 - 1e-12))
            } else {
                z / self.total_sd[j]
            };
            *slot = self.columns[j].eval(zq);
        }
        regime
    }

    /// Class drawn from `softmax(T * eta)` over the classes with a non-zero
    /// prior: the first class whose cumulative weight exceeds `u * total`.
    fn draw_class(&self, eta: &[f64], u: f64) -> usize {
        let logit = |c: usize| {
            if c == 0 {
                0.0
            } else {
                self.temperature * eta[c - 1]
            }
        };
        let on = |c: &usize| self.class_on[*c];
        let top = (0..self.class_on.len())
            .filter(on)
            .map(logit)
            .fold(f64::NEG_INFINITY, f64::max);
        let weight = |c: usize| (logit(c) - top).exp();
        let total: f64 = (0..self.class_on.len()).filter(on).map(weight).sum();
        let (mut running, mut last) = (0.0, 0);
        for c in (0..self.class_on.len()).filter(on) {
            running += weight(c);
            last = c;
            if running > u * total {
                return c;
            }
        }
        last
    }

    /// Target of a decoded row: `x` are its unit-space features, `regime` the
    /// value [`Decoder::row`] returned. Regression gives the standardised
    /// value, classification the class index, multilabel one 0/1 cell per label.
    pub fn target_row(&self, seed: u64, row: u64, x: &[f64], regime: usize, y: &mut [f64]) {
        let mut eta = [0.0f64; 255];
        let eta = &mut eta[..self.target.outputs()];
        self.target.eta(x, eta);
        match self.task {
            Task::Regression => {
                let noise = self.target_sd * normal(seed, STREAM_TARGET, row);
                y[0] = (eta[0] + noise).clamp(-UNIT_CLIP, UNIT_CLIP);
            }
            Task::Binary | Task::Multiclass => {
                let tied = self.rule == LabelRule::Regime && self.prior.len() > 1;
                y[0] = if tied {
                    regime as f64
                } else {
                    self.draw_class(eta, uniform(seed, STREAM_TARGET, row)) as f64
                };
            }
            Task::Multilabel => {
                for (l, slot) in y.iter_mut().enumerate().take(self.label_code.len()) {
                    *slot = match self.label_code[l] {
                        0 => 0.0,
                        255 => 1.0,
                        _ => {
                            let u = uniform(seed, STREAM_TARGET + l as u64, row);
                            (u < sigmoid(self.temperature * eta[l])) as u8 as f64
                        }
                    };
                }
            }
        }
    }

    fn check(&self, card: &ScaleCard) -> Result<(), StatError> {
        if card.columns.len() != self.p {
            return Err(StatError::Invalid(
                "scale card and packet disagree on the column count",
            ));
        }
        Ok(())
    }

    /// Feature rows `row0 .. row0 + n` in unit space, row-major `n x P`.
    pub fn rows_unit(&self, seed: u64, row0: u64, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * self.p);
        let mut row = vec![0.0f64; self.p];
        for i in 0..n {
            self.row(seed, row0 + i as u64, &mut row);
            out.extend(row.iter().map(|&v| v as f32));
        }
        out
    }

    /// Feature rows `row0 .. row0 + n` in the owner's raw units, row-major `n x P`.
    pub fn rows_raw(
        &self,
        card: &ScaleCard,
        seed: u64,
        row0: u64,
        n: usize,
    ) -> Result<Vec<f32>, StatError> {
        self.check(card)?;
        let mut out = Vec::with_capacity(n * self.p);
        let mut row = vec![0.0f64; self.p];
        for i in 0..n {
            self.row(seed, row0 + i as u64, &mut row);
            let raw = row.iter().zip(&card.columns);
            out.extend(raw.map(|(&v, col)| col.to_raw(v) as f32));
        }
        Ok(out)
    }

    /// Rows `row0 .. row0 + n` with the target last, row-major
    /// `n x (P + targets)`. With a scale card the features and a regression
    /// target are in raw units and a class index becomes its raw label;
    /// without one everything stays in unit space.
    pub fn table(
        &self,
        card: Option<&ScaleCard>,
        seed: u64,
        row0: u64,
        n: usize,
    ) -> Result<Vec<f32>, StatError> {
        if let Some(card) = card {
            self.check(card)?;
        }
        let width = self.p + self.targets();
        let mut out = Vec::with_capacity(n * width);
        let (mut x, mut y) = (vec![0.0f64; self.p], vec![0.0f64; self.targets()]);
        for i in 0..n {
            let at = row0 + i as u64;
            let regime = self.row(seed, at, &mut x);
            self.target_row(seed, at, &x, regime, &mut y);
            match card {
                Some(card) => {
                    let raw = x.iter().zip(&card.columns);
                    out.extend(raw.map(|(&v, col)| col.to_raw(v) as f32));
                    out.extend(y.iter().map(|&t| card.target_to_raw(t) as f32));
                }
                None => out.extend(x.iter().chain(&y).map(|&v| v as f32)),
            }
        }
        Ok(out)
    }
}
