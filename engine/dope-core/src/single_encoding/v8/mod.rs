//! Pure-Rust `DSE8` v1 Codec, Analytic Step-0, and `SingleEncV8` Row-Local Runtime.
//!
//! Public entry points locked with `dope-arena` / `dope-ffi` / `codex` verification:
//! - `fit_dse8_packet(x_raw, y_raw, opts) -> Result<Vec<u8>>`
//! - `fit_dse8_packet_with_encoder(x_raw, y_raw, opts, weights) -> Result<Vec<u8>>`
//! - `unpack_dse8_packet(bytes) -> Result<Dse8Packet>`
//! - `truncate_dse8_packet(bytes, target_budget) -> Result<Vec<u8>>`
//! - `build_decode_plan_v8(packet, weights) -> Result<DecodePlanV8>`
//! - `decode_dse8_row_at(plan, seed, row_idx) -> DecodedV8Row`
//! - `decode_dse8_rows(packet, weights, n_rows, seed) -> Result<DecodedV8Table>`
//! - `extract_jopedime_embedding(packet) -> Vec<f32>`

pub mod calibrate;
pub mod copula_regimes;
pub mod decode;
pub mod effects_v8;
pub mod fit_meta;
pub mod marginals;
pub mod model;
pub mod model_ops;
pub mod nominal;
pub mod packet;
pub mod packet_decode;
pub mod planner;
pub mod rng_v8;
pub mod scale_v8;
pub mod target_v8;
pub mod trees_v8;

pub use copula_regimes::{RegimeV8, RegimesSectionV8, V8_MAX_RANK};
pub use decode::{
    build_decode_plan_v8, decode_dse8_row_at, decode_dse8_rows, extract_jopedime_embedding,
    DecodePlanV8, DecodedV8Row, DecodedV8Table,
};
pub use fit_meta::{fit_unit_meta, fit_unit_meta_v2};
pub use marginals::{
    MarginalColV8, MARG_ATOMS, MARG_CONST, MARG_CONT, MARG_LATTICE, MARG_ZERO_INFLATED,
};
pub use model::{
    build_decoder_inputs, entmax_15, NeuralPlanV8, SingleEncV8Weights, DEC_CODE_DIM, DEC_COL_DIM,
    DEC_EDGE_DIM, DEC_GLOB_DIM,
};
pub use nominal::NominalGroupV8;
pub use packet::{
    pack_dse8_packet, truncate_dse8_packet, unpack_dse8_packet, Dse8Header, Dse8Packet, DSE8_MAGIC,
    DSE8_VERSION, FLAG_HAS_COL_MAP, FLAG_HAS_CUSTOM_CLASSES, FLAG_HAS_DROPPED_MEDS,
    FLAG_HAS_NEURAL_ENCODER, FLAG_HAS_SKEW_T, FLAG_NORMALIZED_ONLY,
};
pub use scale_v8::{Dse8ColumnMeta, ScaleCardV8, V8_MAX_CLASSES, V8_MAX_COLS};
pub use target_v8::{CodeSectionV8, Effect1DV8, LinearSectionV8, PairIntV8, ResidSectionV8};
pub use trees_v8::{OblivTreeV8, TreesSectionV8};

use crate::error::{DopeError, Result};
use crate::evidence::primitives::canonical_row_order;
use crate::evidence::{
    extract as extract_evidence, ColumnMeta as EvColMeta, EvidenceInput, EvidenceTier,
};
use crate::stats::norm_ppf;
use calibrate::{
    calibrate_intercepts_on_packet, eval_true_loss, sample_synthetic_x_probe, CdfIntervals,
};
use copula_regimes::{fit_copula_loadings_v8, fit_regimes_v8};
use nominal::fit_nominal_groups;
use planner::choose_base_layout;
use target_v8::{fit_effects_v8, fit_linear_v8, fit_pairs_v8, fit_resid_model_v8};

/// Options controlling `DSE8` v1 fitting, section ablations, and byte budget (`512..65536 B`).
#[derive(Clone, Debug)]
pub struct Dse8FitOptions<'a> {
    pub task_id: u8,
    pub budget_bytes: usize,
    pub normalized_only: bool,
    pub enable_trees: bool,
    pub enable_regimes: bool,
    pub enable_effects: bool,
    pub enable_residual_model: bool,
    /// Opt-in ablation (default `false` per 81-table OpenML evidence!).
    pub enable_skew_t: bool,
    /// Requested nested FSQ neural code tier (`0=off, 1=32B, 2=64B, 3=128B`).
    pub code_tier: u8,
    pub col_meta: Option<&'a [Dse8ColumnMeta]>,
    pub seed: u64,
}

impl Default for Dse8FitOptions<'_> {
    fn default() -> Self {
        Self {
            task_id: 0,
            budget_bytes: 2048,
            normalized_only: false,
            enable_trees: true,
            enable_regimes: true,
            enable_effects: true,
            enable_residual_model: true,
            enable_skew_t: false,
            code_tier: 0,
            col_meta: None,
            seed: 0,
        }
    }
}

/// Fit an analytic Step-0 `DSE8` v1 packet (`<= opts.budget_bytes`, strictly FIT-only, CRC32).
pub fn fit_dse8_packet(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    opts: &Dse8FitOptions<'_>,
) -> Result<Vec<u8>> {
    fit_dse8_internal(x_raw, y_raw, opts, None)
}

/// Fit a neural-compiled `DSE8` v1 packet using `SingleEncV8Weights` Owner Evidence Encoder.
pub fn fit_dse8_packet_with_encoder(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    opts: &Dse8FitOptions<'_>,
    weights: &SingleEncV8Weights,
) -> Result<Vec<u8>> {
    fit_dse8_internal(x_raw, y_raw, opts, Some(weights))
}

fn fit_dse8_internal(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    opts: &Dse8FitOptions<'_>,
    weights: Option<&SingleEncV8Weights>,
) -> Result<Vec<u8>> {
    let mut plan_budget = opts.budget_bytes;
    let mut last_err = DopeError::Budget("over_budget".into());
    for _ in 0..6 {
        match fit_dse8_attempt(x_raw, y_raw, opts, weights, plan_budget) {
            Ok(bytes) => return Ok(bytes),
            Err(DopeError::Budget(msg)) if !opts.normalized_only && plan_budget >= 192 => {
                last_err = DopeError::Budget(msg);
                plan_budget = (plan_budget * 82 / 100).max(128);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err)
}

fn fit_dse8_attempt(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    opts: &Dse8FitOptions<'_>,
    weights: Option<&SingleEncV8Weights>,
    plan_budget: usize,
) -> Result<Vec<u8>> {
    let x_cols_raw = x_raw.len();
    if x_cols_raw == 0 || x_cols_raw > u16::MAX as usize {
        return Err(DopeError::Input("dse8_invalid_col_count".into()));
    }
    if opts.task_id > 2 {
        return Err(DopeError::Input("dse8_invalid_task_id".into()));
    }
    // Explicit admission guard in normalized_only mode: refuse > V8_MAX_COLS or > V8_MAX_CLASSES instead of silently projecting!
    if opts.normalized_only {
        if x_cols_raw > V8_MAX_COLS {
            return Err(DopeError::Input(
                "dse8_normalized_col_limit_exceeded".into(),
            ));
        }
        if opts.task_id == 2 {
            let mut cnts = std::collections::BTreeMap::<i32, usize>::new();
            for &v in y_raw {
                if v.is_finite() {
                    *cnts.entry(v.round() as i32).or_default() += 1;
                }
            }
            let min_c = if y_raw.len() >= 20 { 10 } else { 1 };
            let n_supported = cnts.values().filter(|&&c| c >= min_c).count();
            if n_supported > V8_MAX_CLASSES {
                return Err(DopeError::Input(
                    "dse8_normalized_class_limit_exceeded".into(),
                ));
            }
        }
    }

    // Probe card at V8_MAX_COLS to discover exact k_eff (respecting k >= 10 class floor).
    let probe = ScaleCardV8::fit(
        x_raw,
        y_raw,
        opts.task_id,
        V8_MAX_COLS,
        opts.normalized_only,
        opts.col_meta,
    )?;
    let k_eff = match opts.task_id {
        0 => 1,
        1 => 2,
        _ => probe.classes.len(),
    };
    let want_code = if weights.is_some() {
        opts.code_tier.max(1)
    } else {
        opts.code_tier
    };
    let plan = choose_base_layout(
        x_cols_raw,
        opts.task_id,
        k_eff,
        opts.normalized_only,
        plan_budget,
        want_code,
    )
    .ok_or_else(|| DopeError::Budget("over_budget".into()))?;
    if opts.normalized_only && plan.p_eff != x_cols_raw {
        return Err(DopeError::Budget("over_budget".into()));
    }

    let card = if plan.p_eff == probe.col_indices.len() {
        probe
    } else {
        ScaleCardV8::fit(
            x_raw,
            y_raw,
            opts.task_id,
            plan.p_eff,
            opts.normalized_only,
            opts.col_meta,
        )?
    };
    let x_all = card.forward_x(x_raw);
    let (y_sc, valid_rows) = card.forward_y(y_raw);
    let n_fit = valid_rows.len();
    if n_fit < 4 {
        return Err(DopeError::Input("dse8_too_few_valid_rows".into()));
    }
    let p_eff = card.col_indices.len();
    let x_unit: Vec<Vec<f32>> = (0..p_eff)
        .map(|a| valid_rows.iter().map(|&r| x_all[a][r]).collect())
        .collect();
    let active_meta: Option<Vec<Dse8ColumnMeta>> = opts
        .col_meta
        .map(|ms| card.col_indices.iter().map(|&raw_j| ms[raw_j]).collect());

    let u_floor = crate::evidence::primitives::support_u_floor(n_fit);
    let z_max = norm_ppf(1.0 - u_floor).clamp(1.2, 2.6) as f32;

    // 1. Typed marginals & exact nominal one-hot simplex groups.
    let marginals: Vec<MarginalColV8> = (0..p_eff)
        .map(|j| {
            let mj = active_meta.as_ref().map(|m| &m[j]);
            MarginalColV8::fit(
                &x_unit[j],
                mj,
                plan.knot_level,
                z_max,
                card.is_int_active[j],
            )
        })
        .collect();
    let nominal_groups = if plan.allow_nominal {
        fit_nominal_groups(&x_unit, active_meta.as_deref())
    } else {
        Vec::new()
    };

    // 2. Tie-disattenuated copula loadings & populated multi-regime factor means.
    let (_, loadings, z_tie, factor_scores) = fit_copula_loadings_v8(&x_unit, plan.rank_use);
    let regimes = if opts.enable_regimes && plan.max_regimes >= 2 {
        fit_regimes_v8(&factor_scores, &y_sc, opts.task_id, k_eff, plan.max_regimes)
    } else {
        RegimesSectionV8::default()
    };

    // 3. OOF-CV Newton-IRLS linear head (`W`) with collinear synthetic-X stability guard,
    // decode-aligned CDF intervals, and 80/20 inner split.
    let x_syn_probe = sample_synthetic_x_probe(
        &marginals,
        &loadings,
        &nominal_groups,
        &regimes,
        plan.rank_use,
        z_max,
    );
    let (linear, mut cur_logits) = fit_linear_v8(&x_unit, &y_sc, opts.task_id, k_eff, &x_syn_probe);
    let cdf = CdfIntervals::build(&x_unit);
    let row_ord = canonical_row_order(&x_unit);
    let mut tr_rows = Vec::with_capacity(n_fit);
    let mut va_rows = Vec::with_capacity(n_fit / 4 + 1);
    for (pos, &r) in row_ord.iter().enumerate() {
        if pos % 5 == 0 {
            va_rows.push(r);
        } else {
            tr_rows.push(r);
        }
    }

    // Sequential inner-val-gated boosting: Effects (`E`) -> Pairs (`Pr`) -> Oblivious Trees (`Tr`).
    let effects = if opts.enable_effects && plan.max_effects > 0 {
        fit_effects_v8(
            &cdf,
            &mut cur_logits,
            &y_sc,
            &tr_rows,
            &va_rows,
            opts.task_id,
            linear.temp,
            plan.max_effects,
        )
    } else {
        Vec::new()
    };
    let pairs = if opts.enable_effects && plan.max_pairs > 0 {
        fit_pairs_v8(
            &cdf,
            &mut cur_logits,
            &y_sc,
            &tr_rows,
            &va_rows,
            opts.task_id,
            linear.temp,
            plan.max_pairs,
        )
    } else {
        Vec::new()
    };
    let trees = if opts.enable_trees && plan.max_trees > 0 {
        TreesSectionV8::fit(
            &cdf,
            &mut cur_logits,
            &y_sc,
            &tr_rows,
            &va_rows,
            opts.task_id,
            linear.temp,
            plan.max_trees,
        )
    } else {
        TreesSectionV8::default()
    };

    // Recompute regression resid_sd on held-out va_rows (or all rows if no nonlinear terms) + ResidSection.
    let mut final_resid_sd = linear.resid_sd;
    let mut resid_model = ResidSectionV8::default();
    if opts.task_id == 0 {
        let eval_rows: &[usize] =
            if !effects.is_empty() || !pairs.is_empty() || !trees.trees.is_empty() {
                &va_rows
            } else {
                &row_ord
            };
        let va_mse = eval_true_loss(&cur_logits, &y_sc, eval_rows, 0, 1.0);
        final_resid_sd = va_mse.sqrt().clamp(0.002, 5.0) as f32;
        if opts.enable_residual_model && plan.allow_resid {
            let rem_e: Vec<f64> = (0..n_fit).map(|r| y_sc[r] - cur_logits[0][r]).collect();
            let (rm, sd_mult) = fit_resid_model_v8(&z_tie, &rem_e, final_resid_sd);
            resid_model = rm;
            final_resid_sd = (final_resid_sd * sd_mult).clamp(0.002, 5.0);
        }
    }

    // 5. Optional opt-in Student-t inv_nu & Owner Evidence Encoder nested FSQ code (`Z`).
    let mut inv_nu = 0.0f32;
    let mut code = CodeSectionV8::default();
    let need_ev = opts.enable_skew_t || (weights.is_some() && plan.code_tier > 0);
    if need_ev {
        let ev_meta: Option<Vec<EvColMeta>> = active_meta
            .as_ref()
            .map(|ms| ms.iter().map(EvColMeta::from).collect());
        if let Ok(ev) = extract_evidence(&EvidenceInput {
            x_unit: &x_unit,
            x_raw: None,
            y: &y_sc,
            task_id: opts.task_id,
            n_classes: k_eff,
            col_meta: ev_meta.as_deref(),
            packet_probe: None,
            tier: if weights.is_some() {
                EvidenceTier::T2
            } else {
                EvidenceTier::T1
            },
            target_free: false,
            seed: opts.seed,
        }) {
            if opts.enable_skew_t {
                inv_nu = ev.glob[42].clamp(0.0, 0.45);
            }
            if let Some(w) = weights {
                if plan.code_tier > 0 {
                    if let Some(code_vec) = w.encode_evidence_code(&ev) {
                        code = CodeSectionV8::from_unit_vec(&code_vec, plan.code_tier);
                    }
                }
            }
        }
    }

    let has_col_map = !card.normalized_only
        && (p_eff != x_cols_raw || card.col_indices.iter().enumerate().any(|(a, &r)| a != r));
    let has_custom_classes = !card.normalized_only
        && opts.task_id != 0
        && card
            .classes
            .iter()
            .enumerate()
            .any(|(i, &cid)| cid != i as i32);
    let has_dropped_meds = !card.normalized_only && !card.dropped_medians.is_empty();

    let mut flags = 0u8;
    if card.normalized_only {
        flags |= FLAG_NORMALIZED_ONLY;
    }
    if has_col_map {
        flags |= FLAG_HAS_COL_MAP;
    }
    if has_custom_classes {
        flags |= FLAG_HAS_CUSTOM_CLASSES;
    }
    if has_dropped_meds {
        flags |= FLAG_HAS_DROPPED_MEDS;
    }
    if inv_nu > 1e-4 {
        flags |= FLAG_HAS_SKEW_T;
    }
    if weights.is_some() {
        flags |= FLAG_HAS_NEURAL_ENCODER;
    }

    let p1_prior = if opts.task_id == 1 {
        (y_sc.iter().filter(|&&v| v > 0.5).count() as f32 / n_fit as f32).clamp(0.01, 0.99)
    } else {
        0.0
    };
    let log2_nfit_norm = ((n_fit.max(2) as f32).log2() / 12.0).clamp(0.0, 1.0);

    let pkt = Dse8Packet {
        header: Dse8Header {
            task_id: opts.task_id,
            k_eff: k_eff as u8,
            p_eff: p_eff as u8,
            x_cols_raw: x_cols_raw as u16,
            flags,
            rank_use: plan.rank_use as u8,
            knot_level: plan.knot_level as u8,
            log2_nfit_norm,
            z_max,
            resid_sd: final_resid_sd,
            p1_prior,
            inv_nu,
            code_tier: code.tier,
        },
        card,
        marginals,
        loadings,
        linear,
        nominal_groups,
        regimes,
        effects,
        pairs,
        trees,
        resid_model,
        code,
        weights_fp: weights.map(|w| w.fingerprint()),
    };

    // Pack strictly within budget, then calibrate intercepts `b` against the EXACT
    // budget-surviving X/E/Pr/Tr sections via 512 deterministic synthetic rows!
    let packed = pack_dse8_packet(&pkt, opts.budget_bytes)?;
    let mut final_pkt = unpack_dse8_packet(&packed)?;
    calibrate_intercepts_on_packet(&mut final_pkt, &y_sc);
    pack_dse8_packet(&final_pkt, opts.budget_bytes)
}
