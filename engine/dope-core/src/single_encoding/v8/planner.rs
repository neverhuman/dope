//! Utility-per-byte greedy budget allocator for `DSE8` v1 (`512 B .. 65536 B`).
//!
//! Guarantees strict `<= budget_bytes` compliance across every budget rung:
//! 1. Fits mandatory base sections (`Scale` if `!normalized_only`, `Marg`, `Copula`, `Linear`),
//!    scaling down `(knot_level, rank_use, p_eff)` only if needed for tight budgets (`512 B`).
//! 2. Allocates remaining bytes across optional enhancement sections in priority/utility order:
//!    `Nominal -> Regimes -> Effects -> Pairs -> Trees -> Resid -> Code (nested FSQ tiers 1/2/3)`.

use super::scale_v8::V8_MAX_COLS;
use super::trees_v8::{V8_MAX_TREES, V8_TREE_WIRE_BYTES};

pub const DSE8_HEADER_BYTES: usize = 24;
pub const DSE8_CRC_BYTES: usize = 4;
pub const TLV_HDR_BYTES: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct Dse8LayoutPlan {
    pub p_eff: usize,
    pub knot_level: usize,
    pub rank_use: usize,
    pub max_effects: usize,
    pub max_pairs: usize,
    pub max_trees: usize,
    pub max_regimes: usize,
    pub allow_nominal: bool,
    pub allow_resid: bool,
    pub code_tier: u8,
}

/// Choose base layout `(p_eff, knot_level, rank_use)` that leaves healthy room for `C/Rg/E/Pr/Tr/Rs/Z`.
pub fn choose_base_layout(
    x_cols_raw: usize,
    task_id: u8,
    k_eff: usize,
    normalized_only: bool,
    budget_bytes: usize,
    want_code_tier: u8,
) -> Option<Dse8LayoutPlan> {
    let overhead = DSE8_HEADER_BYTES + DSE8_CRC_BYTES + 4 * TLV_HDR_BYTES;
    if budget_bytes <= overhead + 24 {
        return None;
    }
    let o_dim = if task_id == 2 { k_eff.max(2) } else { 1 };

    // Estimate minimum mandatory bytes for candidate (p_eff, knot_level, rank_use).
    let base_est = |p: usize, kl: usize, r: usize| -> usize {
        let scale_b = if normalized_only {
            0
        } else {
            let y_b = if task_id == 0 { 8 } else { 4 * k_eff };
            let map_b = if p < x_cols_raw { 2 * p } else { 0 };
            let drop_b = if p < x_cols_raw {
                2 * (x_cols_raw - p)
            } else {
                0
            };
            y_b + map_b + drop_b + 7 * p // conservative upper bound before bool/onehot savings
        };
        // Marginal: 6 + kl bytes typical (up to +12 B for atoms/lattice on some cols)
        let marg_b = p * (6 + kl) + (p.min(8) * 6);
        let cop_b = p * r;
        let lin_b = 2 + o_dim * (8 + p);
        overhead + scale_b + marg_b + cop_b + lin_b
    };

    let p_max = x_cols_raw.min(V8_MAX_COLS);
    // Reserve some budget for enhancement sections when budget >= 1024 B!
    let target_enh = if budget_bytes >= 2048 {
        (budget_bytes / 3).min(768)
    } else if budget_bytes >= 1024 {
        220
    } else {
        64
    };

    // First try keeping good room for enhancements; fall back to pure base fit if tight.
    for reserve in [target_enh, 0] {
        for &kl in &[9usize, 17, 5] {
            let max_r = if budget_bytes >= 4096 {
                12
            } else if budget_bytes >= 2048 {
                8
            } else if budget_bytes >= 1024 {
                6
            } else {
                4
            };
            for r in (1..=max_r.min(p_max)).rev() {
                for p in (1..=p_max).rev() {
                    if base_est(p, kl, r) + reserve <= budget_bytes {
                        let rem = budget_bytes.saturating_sub(base_est(p, kl, r));
                        let code_tier = match want_code_tier {
                            3 if rem >= 256 => 3,
                            2..=3 if rem >= 128 => 2,
                            1..=3 if rem >= 64 => 1,
                            _ => 0,
                        };
                        return Some(Dse8LayoutPlan {
                            p_eff: p,
                            knot_level: kl,
                            rank_use: r.min(p),
                            max_effects: (rem / 48).clamp(2, 24),
                            max_pairs: (rem / 48).clamp(2, 24),
                            max_trees: (rem / (V8_TREE_WIRE_BYTES + 2)).min(V8_MAX_TREES),
                            max_regimes: if rem >= 48 { 8 } else { 0 },
                            allow_nominal: rem >= 16,
                            allow_resid: task_id == 0 && rem >= 24,
                            code_tier,
                        });
                    }
                }
            }
        }
    }

    // Ultra-tight fallback (rank 0, knot_level 5)
    for p in (1..=p_max).rev() {
        if base_est(p, 5, 0) <= budget_bytes {
            return Some(Dse8LayoutPlan {
                p_eff: p,
                knot_level: 5,
                rank_use: 0,
                max_effects: 0,
                max_pairs: 0,
                max_trees: 0,
                max_regimes: 0,
                allow_nominal: false,
                allow_resid: false,
                code_tier: 0,
            });
        }
    }
    None
}
