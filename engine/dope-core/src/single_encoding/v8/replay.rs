//! Same-packet decoder experiments using the served counter streams.

use super::copula_regimes::V8_MAX_RANK;
use super::decode::{apply_loading_delta, build_decode_plan_v8, decode_row};
use super::{DecodePlanV8, DecodedV8Row, Dse8Packet, SingleEncV8Weights};
use crate::error::Result;

/// Replay arms hold packet bytes and RNG counters fixed. Component ablations
/// start from the historical learned decoder and change one calculation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayArmV8 {
    Analytic,
    Learned,
    LearnedX,
    LearnedY,
    MaskInactiveRanks,
    NoKnots,
    NoFlow,
    NoLinearCorrection,
    NoGatedHead,
    NoResidualScale,
}

pub struct ReplayPlanV8 {
    plan: DecodePlanV8,
    flow: bool,
    head: bool,
}

impl ReplayPlanV8 {
    pub fn new(
        packet: &Dse8Packet,
        weights: &SingleEncV8Weights,
        arm: ReplayArmV8,
    ) -> Result<Self> {
        let learned = arm != ReplayArmV8::Analytic;
        let mut plan = build_decode_plan_v8(packet, learned.then_some(weights))?;
        let flow = learned && !matches!(arm, ReplayArmV8::LearnedY | ReplayArmV8::NoFlow);
        let head = learned && !matches!(arm, ReplayArmV8::LearnedX | ReplayArmV8::NoGatedHead);
        if learned {
            if matches!(arm, ReplayArmV8::LearnedX | ReplayArmV8::NoLinearCorrection) {
                plan.w_eff.clone_from(&packet.linear.w);
            }
            if matches!(arm, ReplayArmV8::LearnedX | ReplayArmV8::NoResidualScale) {
                plan.resid_sd_eff = packet.header.resid_sd;
            }
            if matches!(arm, ReplayArmV8::LearnedY | ReplayArmV8::NoKnots) {
                for (knots, marginal) in plan.knots_eff.iter_mut().zip(&packet.marginals) {
                    *knots = marginal.knots17;
                }
            }
            if matches!(arm, ReplayArmV8::LearnedY | ReplayArmV8::MaskInactiveRanks) {
                plan.l_eff.clone_from(&packet.loadings);
                for (column, (loading, noise)) in
                    plan.l_eff.iter_mut().zip(&mut plan.noise_sd).enumerate()
                {
                    if arm == ReplayArmV8::MaskInactiveRanks {
                        if let Some(neural) = plan.neural.as_ref().filter(|np| !np.is_identity) {
                            apply_loading_delta(
                                loading,
                                &neural.delta_l[column],
                                packet.header.rank_use as usize,
                            );
                        }
                    }
                    let square_sum: f32 = loading.iter().map(|v| v * v).sum();
                    *noise = (1.0 - square_sum).max(1e-4).sqrt();
                }
            }
        }
        Ok(Self { plan, flow, head })
    }

    /// Observe the same post-regime factors and post-flow latents used by the row conditional.
    pub fn row_at(
        &self,
        seed: u64,
        row: u64,
        observe_latents: impl FnOnce([f32; V8_MAX_RANK], Vec<f32>),
    ) -> DecodedV8Row {
        decode_row(&self.plan, seed, row, self.flow, self.head, observe_latents)
    }
}
