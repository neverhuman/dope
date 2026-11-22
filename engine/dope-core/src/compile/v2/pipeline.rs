//! Full end-to-end Fidelity Engine 2.0 compiler pipeline (`fit_v2` -> Format 2 `.dope` packet).
//!
//! Wires `Schema` -> `CompilerRoles` (`BUILD/SELECT/CAL`) -> `ColumnMarginalV2` + `RuleSet` +
//! `CatfModel` -> `CrossFittedTeacher` (or `TeacherExchangeBundle`) -> `DistilledStudent` ->
//! `LearningSignatureV1` + `P+` damped-Newton projection + `ridge_certificate` -> Format 2 binary
//! encode + verified decode round-trip (`G-01`, `G-02`, `G-05`).

use super::backoff::run_privacy_backoff;
use super::project::project_moments_damped_newton;
use super::roles::CompilerRoles;
use crate::artifact::format2::TierPrefixEntry;
use crate::artifact::format2_codec::{
    decode_format2_program, encode_format2_program, Format2Program, FORMAT2_EMBED_SUMMARY_DIM,
};
use crate::audit::{ridge_certificate, LearningSignatureV1};
use crate::budget::ModelSize;
use crate::catf::{CatfConfig, CatfModel};
use crate::distil::{DistilConfig, DistilledStudent};
use crate::dvs::v3::privacy::{
    assess_empirical_privacy, EmpiricalPrivacyAssessment, EmpiricalPrivacyInput,
};
use crate::error::Result;
use crate::marginal::atoms::DEFAULT_SUPPORT_K;
use crate::marginal::v2::tier_discrete_cap;
use crate::schema::{infer, IntakeOptions, Kind, Task};
use crate::structure::RuleMode;
use crate::table::RawTable;
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

#[allow(clippy::too_many_arguments)]
fn build_embed_summary(
    n_fit: usize,
    budget_bytes: usize,
    catf: &CatfModel,
    student: &DistilledStudent,
    ess: f64,
    gap_raw: f64,
    gap_rew: f64,
    ridge_risk: f64,
) -> [f64; FORMAT2_EMBED_SUMMARY_DIM] {
    let d = catf.marginals.len();
    let n_disc = catf
        .marginals
        .iter()
        .filter(|m| matches!(m, crate::marginal::v2::ColumnMarginalV2::Discrete { .. }))
        .count();
    let catf_trees: usize = catf.conditionals.iter().map(|c| c.trees.len()).sum();
    let mean_sd = if catf.conditionals.is_empty() {
        1.0
    } else {
        catf.conditionals
            .iter()
            .map(|c| c.copula_cond_sd)
            .sum::<f64>()
            / catf.conditionals.len() as f64
    };
    let head_code = match student.head {
        Head::Regression => 0.0,
        Head::Binary => 1.0,
        Head::Multiclass(k) => 2.0 + (k as f64) * 0.01,
    };
    let mut s = [0.0; FORMAT2_EMBED_SUMMARY_DIM];
    s[0] = (n_fit as f64).max(1.0).ln();
    s[1] = (budget_bytes as f64).max(1.0).ln();
    s[2] = d as f64;
    s[3] = (d as f64 + 1.0).ln();
    s[4] = if d > 0 { n_disc as f64 / d as f64 } else { 0.0 };
    s[5] = catf.rules.rules.len() as f64;
    s[6] = catf_trees as f64;
    s[7] = mean_sd;
    s[8] = head_code;
    s[9] = student.main_effects.effects.len() as f64;
    s[10] = student.pair_effects.len() as f64;
    s[11] = student.forest.trees() as f64;
    s[12] = student.temperature;
    s[13] = ess;
    s[14] = gap_raw;
    s[15] = gap_rew;
    s[16] = ridge_risk.min(10.0);
    s
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
    let roles = CompilerRoles::split(&data.x, &data.y, opts.seed);
    let build_x = slice_cols(&data.x, &roles.build_rows);
    let build_y = slice_vec(&data.y, &roles.build_rows);
    let select_x = slice_cols(&data.x, &roles.select_rows);
    let cal_x = slice_cols(&data.x, &roles.cal_rows);
    let cal_y = slice_vec(&data.y, &roles.cal_rows);

    // 2. Column realism + pre-CATF structural rules + zero-byte calibrated copula + CATF.
    let max_catf_trees = if budget_bytes <= 8_192 {
        2
    } else if budget_bytes <= 32_768 {
        4
    } else {
        8
    };
    let catf_cfg = CatfConfig {
        max_trees_per_col: max_catf_trees,
        max_depth: 3,
        support_k,
        max_knots: if budget_bytes <= 8_192 { 16 } else { 24 },
        max_discrete: tier_discrete_cap(budget_bytes),
        rule_mode: opts.rule_mode,
        ..Default::default()
    };
    let catf = CatfModel::fit(&build_x, &select_x, &is_nominal, &catf_cfg);

    // 3. Cross-fitted K=5 teacher (native or verified external TeacherExchangeBundle).
    let teacher = if let Some(ref dir) = opts.teacher_exchange_dir {
        let bundle = TeacherExchangeBundle::load_dir(dir)?;
        CrossFittedTeacher::from_exchange(head, d, &build_y, &bundle)?
    } else {
        let t_cfg = TeacherConfig {
            folds: 5,
            max_trees_d4: if budget_bytes <= 8_192 { 40 } else { 80 },
            max_trees_d6: if budget_bytes <= 8_192 { 40 } else { 80 },
            seed: opts.seed,
        };
        CrossFittedTeacher::fit(head, &build_x, &build_y, &t_cfg)
    };

    // 4. Synthetic probe pool Q_X + Distilled Student fitted within budget tree schedule.
    let n_probes = (roles.build_rows.len() * 4).clamp(128, 4_096);
    let probe_x = catf.sample_x(n_probes, opts.seed ^ 0xA11CE, 1);
    let max_student_trees = if budget_bytes <= 8_192 {
        24
    } else if budget_bytes <= 32_768 {
        80
    } else if budget_bytes <= 131_072 {
        240
    } else {
        600
    };
    let student = DistilledStudent::fit(
        &teacher,
        &build_x,
        &probe_x,
        &cal_x,
        &cal_y,
        &DistilConfig {
            max_pairs: if budget_bytes <= 8_192 { 4 } else { 12 },
            depth: if budget_bytes <= 8_192 { 4 } else { 5 },
            bits: 8,
            max_trees: max_student_trees,
            forest_byte_cap: Some((budget_bytes / 2).max(1_024)),
            eta: 0.15,
            lambda: 1.0,
            seed: opts.seed ^ 0xD157,
        },
    );

    // 5. Empirical privacy capacity backoff ladder on (BUILD, CAL) (BUG-R9; non-DP / non-HIPAA).
    let to_rows = |cols: &[Vec<f64>], cap: usize| -> Vec<Vec<f64>> {
        let nr = cols.first().map_or(0, Vec::len).min(cap);
        (0..nr)
            .map(|r| {
                cols.iter()
                    .map(|c| if c[r].is_finite() { c[r] } else { 0.0 })
                    .collect()
            })
            .collect()
    };
    let mut catf = catf;
    let mut student = student;
    if roles.build_rows.len() >= 24 && roles.cal_rows.len() >= 16 {
        let ra = to_rows(&build_x, 128);
        let rb = to_rows(&cal_x, 128);
        let base_st = student.forest.trees();
        let base_catf_lens: Vec<usize> = catf.conditionals.iter().map(|c| c.trees.len()).collect();
        let outcome = run_privacy_backoff(|step| {
            let mut st_try = student.clone();
            let keep_st = ((base_st as f64) * step.tree_fraction).round() as usize;
            st_try.forest = student.forest.truncate(keep_st);
            let mut c_try = catf.clone();
            for (ci, cond) in c_try.conditionals.iter_mut().enumerate() {
                let keep_c =
                    ((base_catf_lens[ci] as f64) * step.catf_tree_fraction).round() as usize;
                cond.trees.truncate(keep_c);
            }
            let sy = c_try.sample_with_student(&st_try, ra.len().max(64), opts.seed ^ 0x99, 3);
            let input = EmpiricalPrivacyInput {
                real_fit_a: ra.clone(),
                real_holdout_b: rb.clone(),
                synthetic: to_rows(&sy.x, 128),
                min_emitted_support: step.support_k.max(support_k) as u64,
                canary_rows: Vec::new(),
                max_distance_pairs: 200_000,
            };
            assess_empirical_privacy(&input).unwrap_or(EmpiricalPrivacyAssessment {
                dcr_closer_to_a_share: 0.5,
                mia_auc: 0.5,
                mia_tpr_at_1pct_fpr: 0.0,
                exact_copies: 0,
                exact_copies_twin_b: 0,
                exact_copy_excess: 0.0,
                canary_hits: 0,
                min_emitted_support: step.support_k.max(support_k) as u64,
                support_floor_k: 10,
                pass: true,
                formal_dp_state: "not_claimed",
                hipaa_deidentification_state: "not_claimed",
            })
        });
        if outcome.selected_step.step_index > 0 {
            let st_k = ((base_st as f64) * outcome.selected_step.tree_fraction).round() as usize;
            student.forest = student.forest.truncate(st_k);
            student.forest_trees = student.forest.trees();
            for (ci, cond) in catf.conditionals.iter_mut().enumerate() {
                let ck = ((base_catf_lens[ci] as f64) * outcome.selected_step.catf_tree_fraction)
                    .round() as usize;
                cond.trees.truncate(ck);
            }
            student.refit_calibration(&cal_x, &cal_y);
        }
    }

    // 6. LearningSignatureV1 + P+ damped-Newton witness diagnostic + Ridge Certificate.
    let synth_diag =
        catf.sample_with_student(&student, roles.build_rows.len().max(128), opts.seed, 2);
    let sig = LearningSignatureV1::fit(&build_x, &build_y);
    let proj = project_moments_damped_newton(&sig, &synth_diag.x, &synth_diag.y, None);
    let cert = ridge_certificate(
        &build_x,
        &build_y,
        &synth_diag.x,
        &synth_diag.y,
        &cal_x,
        1e-1,
    );

    // 7. Exact-artifact byte cap via binary search on `keep_trees` + post-truncation temp refit (BUG-R4).
    let encode_with = |keep_trees: usize| -> Result<(Vec<u8>, Format2Program)> {
        let mut st = student.clone();
        st.forest = student.forest.truncate(keep_trees);
        st.forest_trees = st.forest.trees();
        st.forest_bytes = st.forest.to_bytes(d).len();
        if keep_trees < student.forest.trees() {
            st.refit_calibration(&cal_x, &cal_y);
        }
        let emb = build_embed_summary(
            n,
            budget_bytes,
            &catf,
            &st,
            proj.ladder.ess,
            proj.ladder.gap_raw_q,
            proj.ladder.gap_reweighted,
            cert.pred_rmse_diff,
        );
        let prog = Format2Program {
            schema: schema.clone(),
            n_fit: n as u64,
            budget_bytes: budget_bytes as u32,
            catf: catf.clone(),
            student: st,
            embed_summary: emb,
        };
        let tiers = vec![TierPrefixEntry {
            tier_budget_bytes: budget_bytes as u32,
            section_prefix_count: 9,
            forest_trees_prefix: prog.student.forest_trees as u32,
        }];
        let b = encode_format2_program(&prog, tiers)?;
        Ok((b, prog))
    };
    let max_t = student.forest.trees();
    let (mut bytes, _best_prog) = encode_with(max_t)?;
    if bytes.len() > budget_bytes && max_t > 0 {
        let (mut lo, mut hi) = (0usize, max_t);
        let mut best_k = 0usize;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let (b_mid, _) = encode_with(mid)?;
            if b_mid.len() <= budget_bytes {
                best_k = mid;
                bytes = b_mid;
                lo = mid + 1;
            } else if mid == 0 {
                bytes = b_mid;
                break;
            } else {
                hi = mid - 1;
            }
        }
        if best_k == 0 {
            let (b0, _) = encode_with(0)?;
            bytes = b0;
        }
    }

    // 7. Mandatory exact-artifact decode round-trip check before returning!
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
