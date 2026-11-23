//! 3-way grouped acceptance guard for DOPE v3 (`Format 1` vs `DSE8 P0` analytic vs `DSE8 P1/P2` neural).
//!
//! Strictly additive alongside `auto::fit_auto` (`DPA1` V7 compatibility untouched):
//! fits all 3 candidates inside `budget` bytes on FIT rows, scores via stratified 2-fold grouped
//! 50/50 cross-fitting (`MIN_CHECK_GROUPS = 80`) with a single-fold drop veto (`<= 0.05`) and
//! `SELECTION_MARGIN = 0.02`, never leaking raw cell values, headers, or extrema.

mod folds;

use super::check::{duplicate_groups, MIN_CHECK_GROUPS};
use super::SELECTION_MARGIN;
use crate::compile::{fit as fit_f1, FitOptions};
use crate::error::{DopeError, Result};
use crate::rng::Rng;
use crate::schema::{infer, IntakeOptions, Kind, Schema, Task};
use crate::single_encoding::v8::{
    fit_dse8_packet, fit_dse8_packet_with_encoder, fit_unit_meta, unpack_dse8_packet,
    Dse8FitOptions, Dse8Packet, SingleEncV8Weights,
};
use crate::table::{is_missing, parse_number, RawTable};
use serde::Serialize;

const TRAIN_CAP: usize = 8192;
const TEST_CAP: usize = 4096;
const MIN_GAIN: f64 = 1e-9;
// The r2 recipe only trained on views up to this active width.
const VALIDATED_NEURAL_WIDTH: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum V8Choice {
    Format1,
    Dse8Analytic,
    Dse8Neural,
}

#[derive(Clone, Debug)]
pub struct AutoV8Options {
    pub budget: usize,
    pub format1: FitOptions,
    pub normalized_only: bool,
    pub enable_trees: bool,
    pub enable_regimes: bool,
    pub enable_effects: bool,
    pub enable_residual_model: bool,
    pub enable_skew_t: bool,
    pub code_tier: u8,
}

impl Default for AutoV8Options {
    fn default() -> Self {
        Self {
            budget: 2048,
            format1: FitOptions::default(),
            normalized_only: false,
            enable_trees: true,
            enable_regimes: true,
            enable_effects: true,
            enable_residual_model: true,
            enable_skew_t: false,
            code_tier: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoV8Report {
    pub choice: V8Choice,
    pub bytes: usize,
    pub budget: usize,
    pub groups: usize,
    pub score_format1: Option<f64>,
    pub score_dse8_analytic: Option<f64>,
    pub score_dse8_neural: Option<f64>,
    pub note: String,
}

pub struct AutoV8Fitted {
    pub choice: V8Choice,
    pub bytes: Vec<u8>,
    pub report: AutoV8Report,
}

/// Pure 3-way preference rule:
/// - `Dse8Neural` wins only if it beats BOTH `Format1` and `Dse8Analytic` by `> SELECTION_MARGIN`.
/// - Else `Dse8Analytic` wins only if it beats `Format1` by `> SELECTION_MARGIN`.
/// - Otherwise keep `Format1`.
pub fn select_v8_candidate(
    score_f1: Option<f64>,
    score_p0: Option<f64>,
    score_neural: Option<f64>,
) -> V8Choice {
    let Some(f1) = score_f1 else {
        return V8Choice::Format1;
    };
    let best_non_neural = match score_p0 {
        Some(p0) => f1.max(p0),
        None => f1,
    };
    if let Some(nn) = score_neural {
        if nn > best_non_neural + SELECTION_MARGIN {
            return V8Choice::Dse8Neural;
        }
    }
    if let Some(p0) = score_p0 {
        if p0 > f1 + SELECTION_MARGIN {
            return V8Choice::Dse8Analytic;
        }
    }
    V8Choice::Format1
}

fn raw_to_numeric_inputs(
    raw: &RawTable,
    target: usize,
    schema: &Schema,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, u8)> {
    let task_id: u8 = match (&schema.task, &schema.target.kind) {
        (Task::Regression, _) => 0,
        (Task::Classification, Kind::Categorical(spec)) if spec.levels.len() > 2 => 2,
        (Task::Classification, _) => 1,
    };
    let mut rows = Vec::with_capacity(raw.rows());
    let mut y = Vec::with_capacity(raw.rows());
    for (r, cell) in raw.columns[target].iter().enumerate() {
        if is_missing(cell) {
            continue;
        }
        let val = match &schema.target.kind {
            Kind::Numeric(_) => parse_number(cell),
            Kind::Categorical(spec) => spec.levels.iter().position(|l| l == cell).map(|k| k as f64),
        };
        if let Some(v) = val {
            rows.push(r);
            y.push(v);
        }
    }
    let mut x = Vec::with_capacity(schema.features.len());
    for f in &schema.features {
        let c = raw
            .headers
            .iter()
            .position(|h| *h == f.name)
            .ok_or_else(|| DopeError::Input("feature column missing".into()))?;
        let col = &raw.columns[c];
        x.push(
            rows.iter()
                .map(|&r| match &f.kind {
                    Kind::Numeric(_) => parse_number(&col[r]).unwrap_or(f64::NAN),
                    Kind::Categorical(spec) if !is_missing(&col[r]) => {
                        spec.level_of(&col[r]).map_or(f64::NAN, |k| k as f64)
                    }
                    Kind::Categorical(_) => f64::NAN,
                })
                .collect(),
        );
    }
    Ok((x, y, task_id))
}

fn decoded_v8_to_raw(schema: &Schema, x: &[Vec<f64>], y: &[f64]) -> RawTable {
    let mut columns: Vec<Vec<String>> = schema
        .features
        .iter()
        .enumerate()
        .map(|(j, f)| {
            (0..y.len())
                .map(|r| {
                    let v = x[j][r];
                    if v.is_finite() {
                        f.to_raw(v)
                    } else {
                        String::new()
                    }
                })
                .collect()
        })
        .collect();
    columns.push(y.iter().map(|&v| schema.target_to_raw(v)).collect());
    let mut headers: Vec<String> = schema.features.iter().map(|f| f.name.clone()).collect();
    headers.push(schema.target.name.clone());
    RawTable { headers, columns }
}

/// Fit Format 1, `DSE8` analytic P0, and optional `DSE8` neural P1/P2 inside `opts.budget`
/// and choose via the 3-way grouped internal acceptance guard (`SELECTION_MARGIN = 0.02`).
pub fn fit_auto_v8(
    raw: &RawTable,
    target: usize,
    opts: &AutoV8Options,
    weights: Option<&SingleEncV8Weights>,
) -> Result<AutoV8Fitted> {
    fit_auto_v8_with_payload_room(raw, target, opts, weights, None)
}

pub(super) fn fit_auto_v8_with_payload_room(
    raw: &RawTable,
    target: usize,
    opts: &AutoV8Options,
    weights: Option<&SingleEncV8Weights>,
    dpa_payload_room: Option<usize>,
) -> Result<AutoV8Fitted> {
    let intake = IntakeOptions {
        task: opts.format1.task,
        ..Default::default()
    };
    let schema = infer(raw, target, &intake)?;
    let (x_full, y_full, task_id) = raw_to_numeric_inputs(raw, target, &schema)?;
    let mut f1_opts = opts.format1.clone();
    f1_opts.budget = Some(opts.format1.budget.unwrap_or(opts.budget));
    f1_opts.task = Some(schema.task);

    let full_meta = fit_unit_meta(&x_full, &y_full, task_id, opts.normalized_only).ok();
    let dse8_opts = Dse8FitOptions {
        task_id,
        budget_bytes: opts.budget,
        normalized_only: opts.normalized_only,
        enable_trees: opts.enable_trees,
        enable_regimes: opts.enable_regimes,
        enable_effects: opts.enable_effects,
        enable_residual_model: opts.enable_residual_model,
        enable_skew_t: opts.enable_skew_t,
        code_tier: opts.code_tier,
        col_meta: full_meta.as_deref(),
        seed: opts.format1.seed,
    };

    let cand_f1 = super::guarded("format1", || fit_f1(raw, target, &f1_opts).map(|f| f.bytes));
    let cand_p0 = super::guarded("dse8_analytic", || {
        fit_dse8_packet(&x_full, &y_full, &dse8_opts)
    });
    let cand_nn = weights.and_then(|w| {
        super::guarded("dse8_neural", || {
            let bytes = fit_dse8_packet_with_encoder(&x_full, &y_full, &dse8_opts, w)?;
            admit_neural_packet(&bytes, w.fingerprint())?;
            Ok(bytes)
        })
        .ok()
    });

    let f1_ok = cand_f1.as_ref().ok();
    let p0_ok = cand_p0.as_ref().ok();
    let nn_ok = cand_nn.as_ref();
    let fold_weights = weights.filter(|_| nn_ok.is_some());

    if f1_ok.is_none() && p0_ok.is_none() {
        return Err(cand_p0.unwrap_err());
    }

    let data = schema.to_model(raw)?;
    let valid_rows: Vec<usize> = (0..data.rows())
        .filter(|&r| data.y[r].is_finite())
        .collect();
    let mut cols: Vec<usize> = schema
        .features
        .iter()
        .filter_map(|f| raw.headers.iter().position(|h| *h == f.name))
        .collect();
    cols.push(target);
    let groups = duplicate_groups(raw, &cols, &valid_rows);
    let n_groups = groups.len();

    let (choice, sf1, sp0, snn, note) = if n_groups >= MIN_CHECK_GROUPS
        && f1_ok.is_some()
        && (p0_ok.is_some() || nn_ok.is_some())
    {
        // Stratified 2-fold grouped 50/50 cross-fitting tests 100% of FIT rows (4x held-out rows
        // vs 75/25) and blocks noisy single-fold regressions > 0.05 below Format 1 on either fold.
        let strat = schema.task == Task::Classification;
        let (half_a, half_b) =
            grouped_two_fold_split(raw, target, strat, groups, opts.format1.seed);
        let run_fold = |tr: &[usize], te: &[usize], sd: u64| {
            folds::score_three_way(
                raw,
                target,
                &schema,
                tr,
                te,
                &f1_opts,
                &dse8_opts,
                fold_weights,
                dpa_payload_room,
                sd,
            )
            .unwrap_or((None, None, None))
        };
        let f0 = run_fold(&half_a, &half_b, opts.format1.seed);
        let f1_cv = run_fold(&half_b, &half_a, opts.format1.seed ^ 0x2F01_D5E8);
        let avg_opt = |a: Option<f64>, b: Option<f64>| a.zip(b).map(|(x, y)| 0.5 * (x + y));
        let (s1, s0, sn) = (
            avg_opt(f0.0, f1_cv.0),
            avg_opt(f0.1, f1_cv.1),
            avg_opt(f0.2, f1_cv.2),
        );
        let no_drop = |c0: Option<f64>, c1: Option<f64>| matches!((f0.0, f1_cv.0, c0, c1), (Some(a), Some(b), Some(x), Some(y)) if x >= a - 0.05 && y >= b - 0.05);
        let pick = select_v8_candidate(
            s1,
            s0.filter(|_| no_drop(f0.1, f1_cv.1)),
            sn.filter(|_| no_drop(f0.2, f1_cv.2)),
        );
        (pick, s1, s0, sn, "scored_3way_2fold".to_string())
    } else if f1_ok.is_some() {
        (
            V8Choice::Format1,
            None,
            None,
            None,
            format!("skipped: groups={n_groups} < {MIN_CHECK_GROUPS}"),
        )
    } else {
        (
            V8Choice::Dse8Analytic,
            None,
            None,
            None,
            "only_dse8_analytic_fit".to_string(),
        )
    };

    let (final_choice, bytes) = match (choice, f1_ok, p0_ok, nn_ok) {
        (V8Choice::Dse8Neural, _, _, Some(b)) => (V8Choice::Dse8Neural, b.clone()),
        (V8Choice::Dse8Analytic, _, Some(b), _) => (V8Choice::Dse8Analytic, b.clone()),
        (_, Some(b), _, _) => (V8Choice::Format1, b.clone()),
        (_, _, Some(b), _) => (V8Choice::Dse8Analytic, b.clone()),
        _ => return Err(DopeError::Budget("no_candidate_fit".into())),
    };

    let len = bytes.len();
    Ok(AutoV8Fitted {
        choice: final_choice,
        bytes,
        report: AutoV8Report {
            choice: final_choice,
            bytes: len,
            budget: opts.budget,
            groups: n_groups,
            score_format1: sf1,
            score_dse8_analytic: sp0,
            score_dse8_neural: snn,
            note,
        },
    })
}

fn admit_neural_packet(bytes: &[u8], fingerprint: [u8; 8]) -> Result<Dse8Packet> {
    let packet = unpack_dse8_packet(bytes)?;
    if usize::from(packet.header.p_eff) > VALIDATED_NEURAL_WIDTH
        || packet.code.tier == 0
        || packet.weights_fp != Some(fingerprint)
    {
        return Err(DopeError::Artifact(
            "auto_v8_neural_packet_outside_support".into(),
        ));
    }
    Ok(packet)
}

fn grouped_two_fold_split(
    raw: &RawTable,
    target: usize,
    stratify: bool,
    mut groups: Vec<Vec<usize>>,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut rng = Rng::new(seed ^ 0xA070_5E1E_C7ED_2F01);
    for i in (1..groups.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        groups.swap(i, j);
    }
    if stratify {
        let tcol = &raw.columns[target];
        groups.sort_by(|a, b| tcol[a[0]].cmp(&tcol[b[0]]));
    }
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (idx, g) in groups.into_iter().enumerate() {
        if idx % 2 == 0 {
            a.extend(g);
        } else {
            b.extend(g);
        }
    }
    a.sort_unstable();
    b.sort_unstable();
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::single_encoding::v8::{pack_dse8_packet, CodeSectionV8};

    #[test]
    fn verified_width_and_code_are_read_from_the_wire() {
        for width in [48, 49, 64] {
            let x: Vec<Vec<f64>> = (0..width)
                .map(|j| (0..24).map(|r| ((r + j) % 13) as f64 / 13.0).collect())
                .collect();
            let y: Vec<_> = (0..24).map(|r| r as f64 / 24.0).collect();
            let opts = Dse8FitOptions {
                normalized_only: true,
                budget_bytes: 4096,
                ..Default::default()
            };
            let mut packet = unpack_dse8_packet(&fit_dse8_packet(&x, &y, &opts).unwrap()).unwrap();
            let fingerprint = [7; 8];
            packet.weights_fp = Some(fingerprint);
            assert_eq!(usize::from(packet.header.p_eff), width);
            let code_free = pack_dse8_packet(&packet, 4096).unwrap();
            assert!(admit_neural_packet(&code_free, fingerprint).is_err());

            packet.code = CodeSectionV8::from_unit_vec(&[0.0; 64], 1);
            let coded = pack_dse8_packet(&packet, 4096).unwrap();
            assert_eq!(unpack_dse8_packet(&coded).unwrap().code.tier, 1);
            assert_eq!(
                admit_neural_packet(&coded, fingerprint).is_ok(),
                width == 48
            );
            assert!(admit_neural_packet(&coded, [8; 8]).is_err());
        }
    }
}
