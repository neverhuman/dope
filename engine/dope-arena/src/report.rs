//! Swap-ladder diagnostic runner (`Real-A`, `Twin-B`, `Real-X+Synth-Y`, `Synth-1n`, `Synth-4n`)
//! and `dope-fidelity/v3` receipt reducer.

use crate::split::{slice_table, split_table_disjoint, SplitRatios};
use crate::{ArenaError, ArenaResult};
use dope_core::dvs::v3::bundle::{PredictionBundle, TaskKind};
use dope_core::dvs::v3::equiv::KappaThresholds;
use dope_core::dvs::v3::realism::{column_marginal_auc, RealismInput};
use dope_core::dvs::v3::{
    evaluate_v3, FidelityCellInput, FidelityCellRequest, FidelityV3Receipt, FidelityV3Request,
    PROTOCOL_V3,
};
use dope_core::eval::{evaluate, EvalOptions};
use dope_core::{decode, fit, output, sample, sample_given, FitOptions, RawTable, SampleOptions};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct SwapLadderArm {
    pub arm_id: String,
    pub rows: usize,
    pub mean_retention: Option<f64>,
    pub per_auditor: Vec<(String, f64, f64, f64, f64)>,
    pub max_marginal_auc: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SwapLadderReport {
    pub budget_bytes: usize,
    pub actual_bytes: usize,
    pub split_sha256: String,
    pub n_fit: usize,
    pub n_twin: usize,
    pub n_test: usize,
    pub arms: Vec<SwapLadderArm>,
}

fn samples_to_raw(program: &dope_core::Program, s: &dope_core::Samples) -> ArenaResult<RawTable> {
    let mut buf = Vec::new();
    output::write_raw(program, s, &mut buf)?;
    Ok(RawTable::from_reader(buf.as_slice())?)
}

fn max_col_marg_auc(program: &dope_core::Program, a_tbl: &RawTable, s_tbl: &RawTable) -> f64 {
    let Ok(a_mod) = program.schema.to_model(a_tbl) else {
        return 1.0;
    };
    let Ok(s_mod) = program.schema.to_model(s_tbl) else {
        return 1.0;
    };
    a_mod
        .x
        .iter()
        .zip(&s_mod.x)
        .filter_map(|(r, s)| column_marginal_auc(r, s).ok())
        .fold(0.5_f64, f64::max)
}

/// Run the native 4-arm swap ladder on a single table:
/// 1. `twin_b_real` (fresh real twin `B`, upper-bound reference)
/// 2. `real_x_synth_y_given` (real `A` features + `y ~ Q(Y|X)` via `--given`)
/// 3. `synth_matched_1n` (full synthetic draw at `1n`)
/// 4. `synth_faithful_4n` (full synthetic draw at `4n`)
pub fn run_swap_ladder(
    table: &RawTable,
    target_col: usize,
    budget_bytes: usize,
    seed: u64,
    auditors: Vec<String>,
) -> ArenaResult<SwapLadderReport> {
    let splits = split_table_disjoint(table, None, None, &SplitRatios::default(), seed)?;
    let a_tbl = slice_table(table, &splits.a_fit);
    let b_tbl = slice_table(table, &splits.b_twin);
    let t_tbl = slice_table(table, &splits.t_test);

    let fitted = fit(
        &a_tbl,
        target_col,
        &FitOptions {
            budget: Some(budget_bytes),
            seed,
            ..Default::default()
        },
    )?;
    let program = decode(&fitted.bytes)?;
    let n_fit = a_tbl.rows();

    let eval_opts = EvalOptions {
        auditors,
        factor: 1.0,
        seed,
    };

    let mut arms = Vec::new();
    // Arm 1: Fresh real twin B.
    let res_b = evaluate(&fitted.bytes, &a_tbl, &t_tbl, Some(&b_tbl), &eval_opts)?;
    arms.push(arm_from_eval(
        "twin_b_real",
        b_tbl.rows(),
        &res_b,
        max_col_marg_auc(&program, &a_tbl, &b_tbl),
    ));

    // Arm 2: Real X from A + synthetic Y from Q(Y|X) via `sample_given` (`--given`).
    let a_mod = program.schema.to_model(&a_tbl)?;
    let given_s = sample_given(&program, &a_mod.x, seed)?;
    let given_tbl = samples_to_raw(&program, &given_s)?;
    let res_given = evaluate(&fitted.bytes, &a_tbl, &t_tbl, Some(&given_tbl), &eval_opts)?;
    arms.push(arm_from_eval(
        "real_x_synth_y_given",
        given_tbl.rows(),
        &res_given,
        max_col_marg_auc(&program, &a_tbl, &given_tbl),
    ));

    // Arm 3: Full synthetic at 1n (`matched`).
    let s1 = sample(
        &program,
        &SampleOptions {
            rows: n_fit,
            seed,
            ..Default::default()
        },
    )?;
    let s1_tbl = samples_to_raw(&program, &s1)?;
    let res_1n = evaluate(&fitted.bytes, &a_tbl, &t_tbl, Some(&s1_tbl), &eval_opts)?;
    arms.push(arm_from_eval(
        "synth_matched_1n",
        s1_tbl.rows(),
        &res_1n,
        max_col_marg_auc(&program, &a_tbl, &s1_tbl),
    ));

    // Arm 4: Full synthetic at 4n (`faithful`).
    let s4 = sample(
        &program,
        &SampleOptions {
            rows: 4 * n_fit,
            seed,
            ..Default::default()
        },
    )?;
    let s4_tbl = samples_to_raw(&program, &s4)?;
    let res_4n = evaluate(&fitted.bytes, &a_tbl, &t_tbl, Some(&s4_tbl), &eval_opts)?;
    arms.push(arm_from_eval(
        "synth_faithful_4n",
        s4_tbl.rows(),
        &res_4n,
        max_col_marg_auc(&program, &a_tbl, &s4_tbl),
    ));

    Ok(SwapLadderReport {
        budget_bytes,
        actual_bytes: fitted.bytes.len(),
        split_sha256: splits.split_sha256,
        n_fit,
        n_twin: b_tbl.rows(),
        n_test: t_tbl.rows(),
        arms,
    })
}

fn arm_from_eval(
    id: &str,
    rows: usize,
    ev: &dope_core::eval::EvalResult,
    max_marg: f64,
) -> SwapLadderArm {
    let per_auditor: Vec<(String, f64, f64, f64, f64)> = ev
        .auditors
        .iter()
        .map(|a| {
            (
                a.auditor.clone(),
                a.null_loss,
                a.real_loss,
                a.synth_loss,
                a.retention,
            )
        })
        .collect();
    let finite: Vec<f64> = ev
        .auditors
        .iter()
        .map(|a| a.retention)
        .filter(|r| r.is_finite())
        .collect();
    let mean_ret = if finite.is_empty() {
        None
    } else {
        Some(finite.iter().sum::<f64>() / finite.len() as f64)
    };
    SwapLadderArm {
        arm_id: id.to_string(),
        rows,
        mean_retention: mean_ret,
        per_auditor,
        max_marginal_auc: max_marg,
    }
}

/// Reduce a list of Arena ledger cells into a canonical `FidelityV3Receipt`.
pub fn reduce_cells(
    cells: Vec<FidelityCellRequest>,
    mode: String,
    seed: u64,
    bootstrap_draws: usize,
) -> ArenaResult<FidelityV3Receipt> {
    if cells.is_empty() {
        return Err(ArenaError::Contract("cannot reduce empty cell list".into()));
    }
    let req = FidelityV3Request {
        protocol: PROTOCOL_V3.to_string(),
        mode,
        seed,
        bootstrap_draws,
        kappa_thresholds: KappaThresholds::default(),
        cells,
    };
    Ok(evaluate_v3(&req)?)
}

/// Helper to construct a synthetic smoke cell for self-checks and tests.
pub fn build_complete_cell(
    cell_id: &str,
    family_id: &str,
    auditor_id: &str,
    task: TaskKind,
    bundle: PredictionBundle,
    realism: Option<RealismInput>,
) -> FidelityCellRequest {
    let mut b = bundle;
    b.task = task;
    FidelityCellRequest {
        cell_id: cell_id.to_string(),
        family_id: family_id.to_string(),
        auditor_id: auditor_id.to_string(),
        input: FidelityCellInput::Complete {
            bundle: Box::new(b),
            driver: None,
            deploy_roster: None,
            realism: realism.map(Box::new),
            privacy: None,
        },
    }
}
