//! Grouped acceptance guard for DOPE v3 (`Format 1` vs embedded `V7` vs `DSE8 P0` analytic vs
//! optional `DSE8 P1/P2` neural).
//!
//! Strictly additive alongside `auto::fit_auto` (`DPA1` V7 compatibility untouched):
//! fits available candidates inside `budget` bytes on FIT rows, scores via stratified 2-fold
//! grouped 50/50 cross-fitting (`MIN_CHECK_GROUPS = 80`) with single-fold, per-auditor, and
//! FIT-only split-FIT empirical privacy vetoes (`v8_guard`) plus `SELECTION_MARGIN = 0.02`,
//! never leaking raw cell values, headers, or extrema.

use super::auditors::{fit_predict, null_predictions, Featurizer, AUDITORS};
use super::check::{
    self, duplicate_groups, retention, CheckInput, CHECK_SYNTH_FACTOR, MIN_CHECK_GROUPS,
};
use super::prefer_v7;
use super::v7::{self, V7Decode, V7Model};
use super::v8_guard::{
    candidate_passes_v8_guards, decoded_v8_to_raw, grouped_two_fold_split, raw_to_numeric_inputs,
    split_fit_privacy, CandFoldScore,
};
use super::SELECTION_MARGIN;
use crate::compile::{fit as fit_f1, FitOptions};
use crate::error::{DopeError, Result};
use crate::eval::{prediction_loss, Matrix};
use crate::program::Program;
use crate::sample::{sample as sample_f1, SampleOptions};
use crate::schema::{infer, IntakeOptions, Schema, Task};
use crate::single_encoding::v8::{
    decode_dse8_rows, fit_dse8_packet, fit_dse8_packet_with_encoder, fit_unit_meta,
    unpack_dse8_packet, Dse8FitOptions, SingleEncV8Weights,
};
use crate::table::RawTable;
use serde::Serialize;

const TRAIN_CAP: usize = 8192;
const TEST_CAP: usize = 4096;
const MIN_GAIN: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum V8Choice {
    Format1,
    V7,
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
    pub score_v7: Option<f64>,
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

/// Fit Format 1, `DSE8` analytic P0, and optional `DSE8` neural P1/P2 inside `opts.budget`
/// and choose via the grouped internal acceptance guard (`SELECTION_MARGIN = 0.02`).
pub fn fit_auto_v8(
    raw: &RawTable,
    target: usize,
    opts: &AutoV8Options,
    weights: Option<&SingleEncV8Weights>,
) -> Result<AutoV8Fitted> {
    fit_auto_v8_with_v7(raw, target, opts, None, weights)
}

/// Fit Format 1, optional embedded `V7`, `DSE8` analytic P0, and optional `DSE8` neural inside
/// budget, upgrading over the `{Format 1, V7}` baseline only on 2-fold retention + per-auditor
/// non-regression + FIT-only split-FIT empirical privacy approval.
pub fn fit_auto_v8_with_v7(
    raw: &RawTable,
    target: usize,
    opts: &AutoV8Options,
    v7_model: Option<&V7Model>,
    weights: Option<&SingleEncV8Weights>,
) -> Result<AutoV8Fitted> {
    let intake = IntakeOptions {
        task: opts.format1.task,
        ..Default::default()
    };
    let schema = infer(raw, target, &intake)?;
    let (x_full, y_full, task_id) = raw_to_numeric_inputs(raw, target, &schema)?;
    let v7_room = opts.format1.budget.unwrap_or(opts.budget);
    let mut f1_opts = opts.format1.clone();
    f1_opts.budget = Some(v7_room);
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
    let cand_v7 = v7_model.map(|_| {
        super::guarded("v7", || {
            v7::fit(raw, target, Some(schema.task), f1_opts.anonymize, v7_room).map(|f| f.payload())
        })
    });
    let cand_p0 = super::guarded("dse8_analytic", || {
        fit_dse8_packet(&x_full, &y_full, &dse8_opts)
    });
    let cand_nn = weights.map(|w| {
        super::guarded("dse8_neural", || {
            fit_dse8_packet_with_encoder(&x_full, &y_full, &dse8_opts, w)
        })
    });

    let f1_ok = cand_f1.as_ref().ok();
    let v7_ok = cand_v7.as_ref().and_then(|r| r.as_ref().ok());
    let p0_ok = cand_p0.as_ref().ok();
    let nn_ok = cand_nn.as_ref().and_then(|r| r.as_ref().ok());

    if f1_ok.is_none() && v7_ok.is_none() && p0_ok.is_none() && nn_ok.is_none() {
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
    // Level 1: establish the certified v2.0.0 baseline (`Format 1` vs embedded `V7`) on FIT rows.
    let (base_choice, base_f1_score, base_v7_score) = match (f1_ok, v7_ok, v7_model) {
        (Some(_), Some(_), Some(m7)) => {
            let chk = check::run(
                &CheckInput {
                    raw,
                    target,
                    schema: &schema,
                    rows: &valid_rows,
                    format1: &f1_opts,
                    room: v7_room,
                    model: m7,
                },
                opts.format1.seed,
            );
            let ch = if prefer_v7(chk.score_format1, chk.score_v7) {
                V8Choice::V7
            } else {
                V8Choice::Format1
            };
            (ch, chk.score_format1, chk.score_v7)
        }
        (Some(_), _, _) => (V8Choice::Format1, None, None),
        (None, Some(_), _) => (V8Choice::V7, None, None),
        (None, None, _) => (
            if p0_ok.is_some() {
                V8Choice::Dse8Analytic
            } else {
                V8Choice::Dse8Neural
            },
            None,
            None,
        ),
    };

    let has_v8_cand = p0_ok.is_some() || nn_ok.is_some();
    let base_is_legacy = matches!(base_choice, V8Choice::Format1 | V8Choice::V7);
    let (choice, sf1, sv7, sp0, snn, note) =
        if n_groups >= MIN_CHECK_GROUPS && base_is_legacy && has_v8_cand {
            // Level 2:
            // - Regression & Binary: upgrade `base_choice -> DSE8` only on 2-fold 50/50 gain > 0.02
            //   passing all 3 FIT-only vetoes (single-fold, per-auditor, split-FIT privacy).
            // - Multiclass (`task_id == 2`): whenever `Format 1` fits in budget (`f1_ok.is_some()`),
            //   use `Format 1` as the 2-fold baseline (avoiding noisy 1-fold V7 flips on multiclass
            //   while retaining V7 fallback when `Format 1` does not fit) and upgrade to `DSE8`
            //   only when all 3 FIT-only vetoes pass.
            let multi_f1_base = task_id == 2 && f1_ok.is_some();
            let cmp_base = if multi_f1_base {
                V8Choice::Format1
            } else {
                base_choice
            };
            let strat = schema.task == Task::Classification;
            let (half_a, half_b) =
                grouped_two_fold_split(raw, target, strat, groups, opts.format1.seed);
            let run_fold = |tr: &[usize], te: &[usize], sd: u64| {
                score_upgrade(
                    raw, target, &schema, tr, te, cmp_base, &f1_opts, v7_room, v7_model,
                    &dse8_opts, weights, sd,
                )
                .unwrap_or([None; 3])
            };
            let f0 = run_fold(&half_a, &half_b, opts.format1.seed);
            let f1_cv = run_fold(&half_b, &half_a, opts.format1.seed ^ 0x2F01_D5E8);
            let avg = |i: usize| {
                f0[i]
                    .zip(f1_cv[i])
                    .map(|(x, y)| 0.5 * (x.min_score + y.min_score))
            };
            let (s_base, s0, sn) = (avg(0), avg(1), avg(2));
            let base_folds = (f0[0], f1_cv[0]);
            let p0_valid = s0.filter(|_| {
                p0_ok.is_some() && candidate_passes_v8_guards(base_folds, (f0[1], f1_cv[1]))
            });
            let nn_valid = sn.filter(|_| {
                nn_ok.is_some() && candidate_passes_v8_guards(base_folds, (f0[2], f1_cv[2]))
            });
            let pick = match select_v8_candidate(s_base, p0_valid, nn_valid) {
                V8Choice::Format1 => cmp_base,
                other => other,
            };
            let (sf1, sv7) = match cmp_base {
                V8Choice::Format1 => (s_base.or(base_f1_score), base_v7_score),
                V8Choice::V7 => (base_f1_score, s_base.or(base_v7_score)),
                _ => (base_f1_score, base_v7_score),
            };
            (pick, sf1, sv7, s0, sn, "scored_v8_2fold".to_string())
        } else {
            let note = if n_groups < MIN_CHECK_GROUPS {
                format!("skipped: groups={n_groups} < {MIN_CHECK_GROUPS}")
            } else {
                "single_candidate_fit".to_string()
            };
            (base_choice, base_f1_score, base_v7_score, None, None, note)
        };

    let (final_choice, bytes) = match (choice, f1_ok, v7_ok, p0_ok, nn_ok) {
        (V8Choice::Dse8Neural, _, _, _, Some(b)) => (V8Choice::Dse8Neural, b.clone()),
        (V8Choice::Dse8Analytic, _, _, Some(b), _) => (V8Choice::Dse8Analytic, b.clone()),
        (V8Choice::V7, _, Some(b), _, _) => (V8Choice::V7, b.clone()),
        (_, Some(b), _, _, _) => (V8Choice::Format1, b.clone()),
        (_, _, Some(b), _, _) => (V8Choice::V7, b.clone()),
        (_, _, _, Some(b), _) => (V8Choice::Dse8Analytic, b.clone()),
        (_, _, _, _, Some(b)) => (V8Choice::Dse8Neural, b.clone()),
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
            score_v7: sv7,
            score_dse8_analytic: sp0,
            score_dse8_neural: snn,
            note,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn score_upgrade(
    raw: &RawTable,
    target: usize,
    whole_schema: &Schema,
    train_idx: &[usize],
    test_idx: &[usize],
    base_choice: V8Choice,
    f1_opts: &FitOptions,
    v7_room: usize,
    v7_model: Option<&V7Model>,
    dse8_opts: &Dse8FitOptions<'_>,
    weights: Option<&SingleEncV8Weights>,
    seed: u64,
) -> Result<[Option<CandFoldScore>; 3]> {
    let train_tbl = raw.select_rows(train_idx);
    let test_tbl = raw.select_rows(test_idx);
    let intake = IntakeOptions {
        task: Some(whole_schema.task),
        ..Default::default()
    };
    let schema = infer(&train_tbl, target, &intake)?;
    let real = schema.to_model(&train_tbl)?;
    let featurizer = Featurizer::fit(&schema, &real);
    let head = Program::head_for(&schema);
    let (xr, yr) = featurizer.matrix(&real, TRAIN_CAP);
    let (xt, yt) = featurizer.matrix(&schema.to_model(&test_tbl)?, TEST_CAP);
    if yr.len() < 20 || yt.len() < 10 {
        return Ok([None; 3]);
    }
    if schema.task == Task::Classification {
        let mut cnts = std::collections::BTreeMap::<i32, usize>::new();
        for &v in &yt {
            if v.is_finite() {
                *cnts.entry(v.round() as i32).or_default() += 1;
            }
        }
        if cnts.len() < 2 || cnts.values().any(|&c| c < 5) {
            return Ok([None; 3]);
        }
    }
    let n_syn = CHECK_SYNTH_FACTOR * train_tbl.rows();
    let as_mat = |tbl: Result<RawTable>| -> Option<(Matrix, Vec<f64>)> {
        let data = schema.to_model(&tbl.ok()?).ok()?;
        let (x, y) = featurizer.matrix(&data, TRAIN_CAP);
        (!y.is_empty()).then_some((x, y))
    };

    let syn_base = match base_choice {
        V8Choice::V7 => v7_model.and_then(|m| {
            as_mat(super::guarded("v7_chk", || {
                let f = v7::fit(
                    &train_tbl,
                    target,
                    Some(whole_schema.task),
                    f1_opts.anonymize,
                    v7_room,
                )?;
                f.decoder()?
                    .table(V7Decode::Trained(m), n_syn, seed ^ 0x5637_0000)
            }))
        }),
        _ => as_mat(super::guarded("f1_chk", || {
            let fitted = fit_f1(&train_tbl, target, f1_opts)?;
            let s = sample_f1(
                &fitted.program,
                &SampleOptions {
                    rows: n_syn,
                    seed: seed ^ 0xF101,
                    ..Default::default()
                },
            )?;
            Ok(decoded_v8_to_raw(&schema, &s.x, &s.y))
        })),
    };

    let (x_tr, y_tr, tr_tid) = raw_to_numeric_inputs(&train_tbl, target, &schema)?;
    let tr_meta = fit_unit_meta(&x_tr, &y_tr, tr_tid, dse8_opts.normalized_only).ok();
    let tr_dse8_opts = Dse8FitOptions {
        task_id: tr_tid,
        col_meta: tr_meta.as_deref(),
        ..*dse8_opts
    };
    let syn_p0 = as_mat(super::guarded("p0_chk", || {
        let bytes = fit_dse8_packet(&x_tr, &y_tr, &tr_dse8_opts)?;
        let pkt = unpack_dse8_packet(&bytes)?;
        let dec = decode_dse8_rows(&pkt, None, n_syn, seed ^ 0xD5E8_0000)?;
        Ok(decoded_v8_to_raw(&schema, &dec.x, &dec.y))
    }));
    let syn_nn = weights.and_then(|w| {
        as_mat(super::guarded("nn_chk", || {
            let bytes = fit_dse8_packet_with_encoder(&x_tr, &y_tr, &tr_dse8_opts, w)?;
            let pkt = unpack_dse8_packet(&bytes)?;
            let dec = decode_dse8_rows(&pkt, Some(w), n_syn, seed ^ 0xD5E8_0001)?;
            Ok(decoded_v8_to_raw(&schema, &dec.x, &dec.y))
        }))
    });

    let syns = [&syn_base, &syn_p0, &syn_nn];
    let l0 = prediction_loss(head, &null_predictions(head, &yr, yt.len()), &yt);
    let mut mins = [f64::INFINITY; 3];
    let mut auds = [[None; 2]; 3];
    let mut any_usable = false;
    for (ai, name) in AUDITORS.iter().enumerate() {
        let s = seed ^ 0xA0D1_8888;
        let l_real = prediction_loss(head, &fit_predict(name, head, &xr, &yr, &xt, s), &yt);
        if !l_real.is_finite() || l0 - l_real <= MIN_GAIN {
            continue;
        }
        any_usable = true;
        for ci in 0..3 {
            let Some((xs, ys)) = syns[ci].as_ref() else {
                continue;
            };
            let ls = prediction_loss(head, &fit_predict(name, head, xs, ys, &xt, s), &yt);
            let r = retention(l0, l_real, ls);
            mins[ci] = mins[ci].min(r);
            auds[ci][ai] = Some(r);
        }
    }
    if !any_usable {
        return Ok([None; 3]);
    }
    let mut out = [None; 3];
    for ci in 0..3 {
        let Some((xs, ys)) = syns[ci].as_ref() else {
            continue;
        };
        let (dcr_share, tpr_1pct) = match split_fit_privacy(&xr, &yr, &xt, &yt, xs, ys) {
            Some((d, t)) => (Some(d), Some(t)),
            None => (None, None),
        };
        out[ci] = Some(CandFoldScore {
            min_score: mins[ci],
            aud: auds[ci],
            dcr_share,
            tpr_1pct,
        });
    }
    Ok(out)
}
