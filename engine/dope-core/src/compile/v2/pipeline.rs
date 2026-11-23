//! Full end-to-end Fidelity Engine 2.0 compiler pipeline (`fit_v2` -> Format 2 `.dope` packet).
//!
//! Wires `Schema` -> `CompilerRoles` (`BUILD/SELECT/CAL`) -> `ColumnMarginalV2` + `RuleSet` +
//! `CatfModel` -> `CrossFittedTeacher` (or `TeacherExchangeBundle`) -> `DistilledStudent` ->
//! `LearningSignatureV1` + `P+` damped-Newton projection + `ridge_certificate` -> Format 2 binary
//! encode + verified decode round-trip (`G-01`, `G-02`, `G-05`).

use super::backoff::apply_standardized_privacy_backoff;
use super::project::project_moments_damped_newton;
use super::roles::CompilerRoles;
use super::{build_embed_summary, compact_marginals, prune_catf_copula, prune_main_effects};
use crate::artifact::format2::TierPrefixEntry;
use crate::artifact::format2_codec::{
    decode_format2_program, encode_format2_program, Format2Program,
};
use crate::audit::{ridge_certificate, LearningSignatureV1};
use crate::budget::ModelSize;
use crate::catf::{CatfConfig, CatfModel};
use crate::distil::{DistilConfig, DistilledStudent};
use crate::error::Result;
use crate::marginal::atoms::DEFAULT_SUPPORT_K;
use crate::marginal::lattice::is_f32_heritage;
use crate::marginal::v2::tier_discrete_cap;
use crate::schema::{infer, IntakeOptions, Kind, Task};
use crate::structure::RuleMode;
use crate::table::{parse_number, RawTable};
use crate::target::Head;
use crate::teacher::exchange::TeacherExchangeBundle;
use crate::teacher::{CrossFittedTeacher, TeacherConfig};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct FitV2Options {
    pub budget: Option<usize>,
    pub size: Option<ModelSize>,
    pub seed: u64,
    pub task: Option<Task>,
    pub anonymize: bool,
    pub rule_mode: RuleMode,
    pub support_k: usize,
    pub teacher_exchange_dir: Option<PathBuf>,
}

impl Default for FitV2Options {
    fn default() -> Self {
        Self {
            budget: None,
            size: None,
            seed: 0,
            task: None,
            anonymize: false,
            rule_mode: RuleMode::Structural,
            support_k: DEFAULT_SUPPORT_K,
            teacher_exchange_dir: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FitV2Report {
    pub format_version: u16,
    pub bytes: usize,
    pub budget_bytes: usize,
    pub n_fit: usize,
    pub features: usize,
    pub task: String,
    pub rules_mined: usize,
    pub catf_trees: usize,
    pub student_trees: usize,
    pub teacher_beats_null: bool,
    pub seen_teacher_families: Vec<String>,
    pub projection_ess: f64,
    pub projection_gap_raw: f64,
    pub projection_gap_reweighted: f64,
    pub ridge_cert_relative_risk: f64,
    pub packet_sha256: String,
}

pub struct FittedV2 {
    pub bytes: Vec<u8>,
    pub program: Format2Program,
    pub report: FitV2Report,
}

fn slice_cols(cols: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    cols.iter()
        .map(|c| rows.iter().map(|&r| c[r]).collect())
        .collect()
}

fn slice_vec(v: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&r| v[r]).collect()
}

/// Compile a `RawTable` through the complete Fidelity Engine 2.0 pipeline into a verified
/// `.dope` Format 2 binary artifact.
pub fn fit_v2(table: &RawTable, target: usize, opts: &FitV2Options) -> Result<FittedV2> {
    let budget_bytes = opts
        .budget
        .or_else(|| opts.size.map(|s| s.budget()))
        .unwrap_or(65_536);
    let support_k = opts.support_k.max(DEFAULT_SUPPORT_K);
    let intake = IntakeOptions {
        task: opts.task,
        min_level_count: support_k,
        ..Default::default()
    };
    let mut schema = infer(table, target, &intake)?;
    let data = schema.to_model(table)?;
    let raw_f32 = |name: &str| -> bool {
        table
            .headers
            .iter()
            .position(|h| h == name)
            .is_some_and(|c| {
                let vals: Vec<f64> = table.columns[c]
                    .iter()
                    .filter_map(|s| parse_number(s))
                    .collect();
                is_f32_heritage(&vals)
            })
    };
    let mut col_f32_heritage: Vec<bool> = schema
        .features
        .iter()
        .map(|f| matches!(f.kind, Kind::Numeric(_)) && raw_f32(&f.name))
        .collect();
    col_f32_heritage
        .push(matches!(schema.target.kind, Kind::Numeric(_)) && raw_f32(&schema.target.name));
    if opts.anonymize {
        for (i, f) in schema.features.iter_mut().enumerate() {
            f.name = format!("f{i}");
            if let Kind::Categorical(ref mut c) = f.kind {
                c.levels = (0..c.levels.len()).map(|k| format!("c{k}")).collect();
            }
        }
        schema.target.name = "target".into();
        if let Kind::Categorical(ref mut c) = schema.target.kind {
            c.levels = (0..c.levels.len()).map(|k| format!("y{k}")).collect();
        }
    }
    let n = data.y.len();
    let d = schema.features.len();
    let head = match &schema.target.kind {
        Kind::Numeric(_) => Head::Regression,
        Kind::Categorical(c) if c.levels.len() <= 2 => Head::Binary,
        Kind::Categorical(c) => Head::Multiclass(c.levels.len()),
    };
    let is_nominal: Vec<bool> = schema
        .features
        .iter()
        .map(|f| matches!(f.kind, Kind::Categorical(_)))
        .collect();

    // 1. Disjoint BUILD / SELECT / CAL roles under Rule A-1 duplicate grouping.
    // Fit final model on BUILD and SELECT (80% of n_fit, CL-11) while keeping CAL (20%) isolated!
    let roles = CompilerRoles::split(&data.x, &data.y, opts.seed);
    let mut train_rows = Vec::with_capacity(roles.build_rows.len() + roles.select_rows.len());
    train_rows.extend_from_slice(&roles.build_rows);
    train_rows.extend_from_slice(&roles.select_rows);
    let build_x = slice_cols(&data.x, &roles.build_rows);
    let select_x = slice_cols(&data.x, &roles.select_rows);
    let train_x = slice_cols(&data.x, &train_rows);
    let train_y = slice_vec(&data.y, &train_rows);
    let cal_x = slice_cols(&data.x, &roles.cal_rows);
    let cal_y = slice_vec(&data.y, &roles.cal_rows);

    // 2. Column realism + pre-CATF structural rules + zero-byte calibrated copula + CATF.
    let catf_cap = if budget_bytes <= 4_500 {
        0
    } else if budget_bytes <= 8_192 {
        2
    } else if budget_bytes <= 32_768 {
        6
    } else {
        10
    };
    let max_catf_trees = ((budget_bytes / 4) / (d.max(1) * 140)).min(catf_cap);
    let max_knots = if budget_bytes <= 2_500 {
        6
    } else if budget_bytes <= 4_500 {
        10
    } else if budget_bytes <= 16_384 {
        16
    } else {
        24
    };
    let max_copula_preds = if budget_bytes <= 2_500 {
        4
    } else if budget_bytes <= 4_500 {
        8
    } else if budget_bytes <= 8_192 {
        16
    } else {
        usize::MAX
    };
    let catf_cfg = CatfConfig {
        max_trees_per_col: max_catf_trees,
        max_depth: 3,
        support_k,
        max_knots,
        max_discrete: tier_discrete_cap(budget_bytes),
        max_copula_preds,
        rule_mode: opts.rule_mode,
        ..Default::default()
    };
    let mut catf = CatfModel::fit(&build_x, &select_x, &is_nominal, &catf_cfg);

    // 3. Cross-fitted K=5 teacher on full BUILD and SELECT (80%, CL-11) or verified external bundle.
    let teacher = if let Some(ref dir) = opts.teacher_exchange_dir {
        let bundle = TeacherExchangeBundle::load_dir(dir)?;
        CrossFittedTeacher::from_exchange(head, d, &train_y, &bundle)?
    } else {
        let t_cfg = TeacherConfig {
            folds: 5,
            max_trees_d4: if budget_bytes <= 8_192 { 60 } else { 120 },
            max_trees_d6: if budget_bytes <= 8_192 { 80 } else { 180 },
            seed: opts.seed,
        };
        CrossFittedTeacher::fit(head, &train_x, &train_y, &t_cfg)
    };

    // 4. Synthetic probe pool Q_X + Distilled Student + direct-label T10 CAL control selector.
    let n_probes = (train_rows.len() * 8).clamp(512, 16_384);
    let probe_x = catf.sample_x(n_probes, opts.seed ^ 0xA11CE, 1);
    let max_student_trees = if budget_bytes <= 4_500 {
        48
    } else if budget_bytes <= 16_384 {
        120
    } else if budget_bytes <= 131_072 {
        360
    } else {
        800
    };
    let distil_cfg = DistilConfig {
        main_bins: if budget_bytes <= 2_500 {
            2
        } else if budget_bytes <= 4_500 {
            3
        } else if budget_bytes <= 16_384 {
            4
        } else {
            5
        },
        max_pairs: if budget_bytes <= 8_192 { 0 } else { 4 },
        depth: if budget_bytes <= 16_384 { 4 } else { 5 },
        bits: if budget_bytes <= 16_384 { 6 } else { 7 },
        max_trees: max_student_trees,
        forest_byte_cap: Some((budget_bytes / 2).max(1_024)),
        eta: 0.15,
        lambda: 1.0,
        seed: opts.seed ^ 0xD157,
    };
    let mut student =
        DistilledStudent::fit(&teacher, &train_x, &probe_x, &cal_x, &cal_y, &distil_cfg);
    if opts.teacher_exchange_dir.is_none() && cal_y.len() >= 16 {
        let direct =
            DistilledStudent::fit_direct(head, &train_x, &train_y, &cal_x, &cal_y, &distil_cfg);
        if direct.cal_loss(&cal_x, &cal_y) + 1e-3 < student.cal_loss(&cal_x, &cal_y) {
            student = direct;
        }
    }

    // 5. Standardized (X, Y) empirical privacy backoff (BUG-R10 / PK-17; non-DP / non-HIPAA).
    apply_standardized_privacy_backoff(
        &mut catf,
        &mut student,
        &train_x,
        &train_y,
        &cal_x,
        &cal_y,
        support_k,
        opts.seed,
    );

    // 6. LearningSignatureV1 + P+ damped-Newton witness diagnostic + Ridge Certificate.
    let synth_diag = catf.sample_with_student(&student, train_rows.len().max(128), opts.seed, 2);
    let sig = LearningSignatureV1::fit(&train_x, &train_y);
    let proj = project_moments_damped_newton(&sig, &synth_diag.x, &synth_diag.y, None);
    let cert = ridge_certificate(
        &train_x,
        &train_y,
        &synth_diag.x,
        &synth_diag.y,
        &cal_x,
        1e-1,
    );

    // 7. Strict exact-artifact byte-cap enforcer: compact non-forest overhead if needed so
    //    `bytes.len() <= budget_bytes` holds even at 2–4 KiB, while keeping predictive forest trees!
    let encode_cfg =
        |catf_ref: &CatfModel, st_base: &DistilledStudent, keep_trees: usize| -> Result<Vec<u8>> {
            let mut st = st_base.clone();
            st.forest = st_base.forest.truncate(keep_trees);
            st.forest_trees = st.forest.trees();
            st.forest_bytes = st.forest.to_bytes(d).len();
            if keep_trees < st_base.forest.trees() {
                st.refit_calibration(&cal_x, &cal_y);
            }
            let emb = build_embed_summary(
                n,
                budget_bytes,
                catf_ref,
                &st,
                proj.ladder.ess,
                proj.ladder.gap_raw_q,
                proj.ladder.gap_reweighted,
                cert.pred_rmse_diff,
            );
            let prog = Format2Program {
                schema: schema.clone(),
                col_f32_heritage: col_f32_heritage.clone(),
                n_fit: n as u64,
                budget_bytes: budget_bytes as u32,
                catf: catf_ref.clone(),
                student: st,
                embed_summary: emb,
            };
            let tiers = vec![TierPrefixEntry {
                tier_budget_bytes: budget_bytes as u32,
                section_prefix_count: 9,
                forest_trees_prefix: prog.student.forest_trees as u32,
            }];
            encode_format2_program(&prog, tiers)
        };
    let fit_trees =
        |catf_ref: &CatfModel, st_base: &DistilledStudent| -> Result<(usize, Vec<u8>)> {
            let max_t = st_base.forest.trees();
            let b_full = encode_cfg(catf_ref, st_base, max_t)?;
            if b_full.len() <= budget_bytes || max_t == 0 {
                return Ok((max_t, b_full));
            }
            let (mut lo, mut hi, mut best_k, mut best_b) =
                (0usize, max_t, 0usize, encode_cfg(catf_ref, st_base, 0)?);
            if best_b.len() > budget_bytes {
                return Ok((0, best_b));
            }
            while lo <= hi {
                let mid = (lo + hi) / 2;
                let b_mid = encode_cfg(catf_ref, st_base, mid)?;
                if b_mid.len() <= budget_bytes {
                    best_k = mid;
                    best_b = b_mid;
                    lo = mid + 1;
                } else if mid == 0 {
                    break;
                } else {
                    hi = mid - 1;
                }
            }
            Ok((best_k, best_b))
        };
    let (mut kept_t, mut bytes) = fit_trees(&catf, &student)?;
    // Non-destructive compaction ladder: ONLY engage when `bytes > budget_bytes` (strict cap
    // compliance!) or when non-forest overhead starved `student.forest` down to 0 trees!
    // Never wipe the Gaussian copula or marginals when the artifact already fits under budget!
    let ladder: &[(usize, usize, usize, bool, bool)] = &[
        (usize::MAX, usize::MAX, usize::MAX, true, false),
        (6, 10, 16, true, false),
        (4, 8, 12, true, false),
        (2, 6, 8, true, false),
        (1, 5, 6, true, false),
        (0, 4, 4, true, false),
        (0, 3, 2, true, true),
        (0, 2, 0, true, true),
    ];
    for &(max_preds, max_k, max_main, drop_extra, trim_extra) in ladder {
        let starving = kept_t == 0 && student.forest.trees() > 0 && max_preds >= 4;
        if bytes.len() <= budget_bytes && !starving {
            break;
        }
        let mut c_try = catf.clone();
        let mut st_try = student.clone();
        if drop_extra {
            st_try.pair_effects.clear();
        }
        prune_catf_copula(&mut c_try, max_preds, true);
        compact_marginals(&mut c_try, max_k, trim_extra && bytes.len() > budget_bytes);
        if max_main < st_try.main_effects.effects.len() {
            prune_main_effects(&mut st_try, max_main);
            st_try.refit_calibration(&cal_x, &cal_y);
        }
        let (kt, b_try) = fit_trees(&c_try, &st_try)?;
        let fixes_cap = bytes.len() > budget_bytes && b_try.len() < bytes.len();
        let fixes_starve =
            bytes.len() <= budget_bytes && b_try.len() <= budget_bytes && kept_t == 0 && kt > 0;
        if fixes_cap || fixes_starve {
            catf = c_try;
            student = st_try;
            kept_t = kt;
            bytes = b_try;
        }
    }

    // 7. Strict fail-closed budget enforcement + mandatory exact-artifact decode round-trip check!
    if (opts.budget.is_some() || opts.size.is_some()) && bytes.len() > budget_bytes {
        return Err(crate::error::DopeError::Budget(format!(
            "format2 artifact ({} B) exceeds requested budget ({} B)",
            bytes.len(),
            budget_bytes
        )));
    }
    let decoded = decode_format2_program(&bytes)?;
    let packet_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let catf_trees: usize = decoded
        .catf
        .conditionals
        .iter()
        .map(|c| c.trees.len())
        .sum();
    let report = FitV2Report {
        format_version: 2,
        bytes: bytes.len(),
        budget_bytes,
        n_fit: n,
        features: d,
        task: match schema.task {
            Task::Regression => "regression".into(),
            Task::Classification => "classification".into(),
        },
        rules_mined: decoded.catf.rules.rules.len(),
        catf_trees,
        student_trees: decoded.student.forest.trees(),
        teacher_beats_null: teacher.oof_beats_null,
        seen_teacher_families: teacher.seen_teacher_families,
        projection_ess: proj.ladder.ess,
        projection_gap_raw: proj.ladder.gap_raw_q,
        projection_gap_reweighted: proj.ladder.gap_reweighted,
        ridge_cert_relative_risk: cert.pred_rmse_diff,
        packet_sha256,
    };
    Ok(FittedV2 {
        bytes,
        program: decoded,
        report,
    })
}
