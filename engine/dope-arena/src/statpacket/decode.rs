//! Row-local analytic sampler (features only in this stage).
//!
//! Row `i` is a pure function of `(packet, seed, i)`:
//!
//! ```text
//! c   ~ Cat(pi)                          regime (binary / multiclass with rank > 0)
//! f_k ~ N(0, 1), k < rank
//! z_j = l_j . (m_c + f) + sqrt(psi_j) e_j,   psi_j = max(0.02, 1 - |l_j|^2)
//! u_j = sum_c pi_c Phi((z_j - l_j . m_c) / s_j),   s_j^2 = |l_j|^2 + psi_j
//! x_j = Q_j(ppf(u_j))                    knot curve or atom table
//! ```
//!
//! `u_j` is the exact mixture CDF of `z_j`, so no batch statistic is needed and
//! every draw is counter based: `hash3(seed, purpose << 32 | draw, i)`. With a
//! single regime `ppf(u_j)` reduces to `z_j / s_j` and no CDF is evaluated.

use super::dependence::{LOADING_SCALE, MEAN_STEP, PSI_FLOOR};
use super::marginals::{knot_plan, ColumnSampler};
use super::packet::{Packet, MAX_RANK};
use super::scale::ScaleCard;
use super::StatError;
use dope_core::rng::{normal, uniform};
use dope_core::stats::{norm_cdf, norm_ppf};

/// Counter streams: purpose in the high 32 bits, draw index in the low 32.
pub const STREAM_REGIME: u64 = 1 << 32;
pub const STREAM_FACTOR: u64 = 2 << 32;
pub const STREAM_NOISE: u64 = 3 << 32;

#[derive(Clone, Debug)]
pub struct Decoder {
    p: usize,
    rank: usize,
    /// Regime probabilities; empty when the sampler has a single regime.
    prior: Vec<f64>,
    prior_cum: Vec<f64>,
    /// `regimes x rank` factor means.
    factor_mean: Vec<f64>,
    /// `p x rank` loadings.
    loading: Vec<f64>,
    noise_sd: Vec<f64>,
    total_sd: Vec<f64>,
    /// `p x regimes` values of `l_j . m_c`.
    shift: Vec<f64>,
    columns: Vec<ColumnSampler>,
}

impl Decoder {
    pub fn new(packet: &Packet) -> Result<Self, StatError> {
        packet.validate()?;
        let (p, rank) = (packet.cols(), packet.level.rank());
        let regimes = if rank > 0 { packet.regime_means() } else { 0 };
        let mut prior = Vec::new();
        if regimes > 0 {
            let head: u32 = packet.prior.iter().map(|&c| c as u32).sum();
            prior = packet.prior.iter().map(|&c| c as f64 / 255.0).collect();
            prior.push((255 - head) as f64 / 255.0);
        }
        let mut running = 0.0;
        let prior_cum = prior
            .iter()
            .map(|q| {
                running += q;
                running
            })
            .collect();
        let factor_mean: Vec<f64> = (0..regimes * rank)
            .map(|i| packet.class_mean[i / rank * MAX_RANK + i % rank] as f64 * MEAN_STEP)
            .collect();
        let loading: Vec<f64> = (0..p * rank)
            .map(|i| packet.loading[i / rank * MAX_RANK + i % rank] as f64 / LOADING_SCALE)
            .collect();
        let (mut noise_sd, mut total_sd) = (vec![1.0; p], vec![1.0; p]);
        let mut shift = vec![0.0; p * regimes];
        for j in 0..p {
            let l = &loading[j * rank..(j + 1) * rank];
            let common: f64 = l.iter().map(|v| v * v).sum();
            let psi = (1.0 - common).max(PSI_FLOOR);
            noise_sd[j] = psi.sqrt();
            total_sd[j] = (common + psi).sqrt();
            for c in 0..regimes {
                let m = &factor_mean[c * rank..(c + 1) * rank];
                shift[j * regimes + c] = l.iter().zip(m).map(|(a, b)| a * b).sum();
            }
        }
        let plan = knot_plan(packet.log2_n);
        let columns = packet
            .columns
            .iter()
            .map(|col| ColumnSampler::new(col, packet.level, &plan))
            .collect();
        Ok(Self {
            p,
            rank,
            prior,
            prior_cum,
            factor_mean,
            loading,
            noise_sd,
            total_sd,
            shift,
            columns,
        })
    }

    pub fn cols(&self) -> usize {
        self.p
    }

    /// Writes unit-space row `row` into `out` (length P).
    pub fn row(&self, seed: u64, row: u64, out: &mut [f64]) {
        let regimes = self.prior.len();
        let mut f = [0.0f64; MAX_RANK];
        for (k, slot) in f.iter_mut().enumerate().take(self.rank) {
            *slot = normal(seed, STREAM_FACTOR + k as u64, row);
        }
        if regimes > 1 {
            let u = uniform(seed, STREAM_REGIME, row);
            let c = self.prior_cum.iter().take_while(|&&q| q <= u).count();
            let c = c.min(regimes - 1);
            let mean = &self.factor_mean[c * self.rank..(c + 1) * self.rank];
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
    }

    /// Rows `row0 .. row0 + n` in unit space, row-major `n x P`.
    pub fn rows_unit(&self, seed: u64, row0: u64, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * self.p);
        let mut row = vec![0.0f64; self.p];
        for i in 0..n {
            self.row(seed, row0 + i as u64, &mut row);
            out.extend(row.iter().map(|&v| v as f32));
        }
        out
    }

    /// Rows `row0 .. row0 + n` in the owner's raw units, row-major `n x P`.
    pub fn rows_raw(
        &self,
        card: &ScaleCard,
        seed: u64,
        row0: u64,
        n: usize,
    ) -> Result<Vec<f32>, StatError> {
        if card.columns.len() != self.p {
            return Err(StatError::Invalid(
                "scale card and packet disagree on the column count",
            ));
        }
        let mut out = Vec::with_capacity(n * self.p);
        let mut row = vec![0.0f64; self.p];
        for i in 0..n {
            self.row(seed, row0 + i as u64, &mut row);
            out.extend(
                row.iter()
                    .zip(&card.columns)
                    .map(|(&v, col)| col.to_raw(v) as f32),
            );
        }
        Ok(out)
    }
}
