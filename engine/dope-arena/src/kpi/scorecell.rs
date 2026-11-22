//! Scoring of one cell for one arm: every scheduled (sample seed, multiplier)
//! file times every auditor becomes exactly one ledger entry.
//!
//! A file that is missing, malformed, mis-shaped, non-finite or single-class,
//! and an emit row that reports a failure, yield `Failed`, `OverBudget` or
//! `Unsupported` entries; nothing is skipped. NaN in a synthetic feature is a
//! missing cell (imputed like a real one); an infinite feature or an invalid
//! target fails the file.

use super::auditors::{self, Recipe};
use super::celldata::{target_varies, targets_valid, task_kind, CellData, Rows};
use super::cellplan::Cell;
use super::emitlog::{
    known_class, sanitize_reason, EmitKey, EmitRow, CLASS_CONTROL, EMIT_FAILED, EMIT_OK,
    EMIT_OVER_BUDGET, EMIT_UNSUPPORTED,
};
use super::io::Arena;
use super::mat::{mat_relpath, read_mat};
use super::realref::{auditor_seed, max_train_rows, RealRef, C2ST_TREES, REF_OK};
use super::roles::MIN_ROLE_ROWS;
use dope_arena::{ArenaError, ArenaResult};
use dope_core::dvs::v3::bundle::PredictionBundle;
use dope_core::dvs::v3::c2st::build_native_realism_input;
use dope_core::dvs::v3::privacy::EmpiricalPrivacyInput;
use dope_core::dvs::v3::realism::{roc_auc, RealismInput};
use dope_core::dvs::v3::FidelityCellInput;
use dope_core::eval::Matrix;
use dope_core::target::Head;
use std::collections::BTreeMap;
use std::path::Path;

/// Support floor written when the emitter declares none (see `score.rs`).
pub const UNDECLARED_SUPPORT: u64 = 10;

pub struct ScoreContext<'a> {
    pub arena: &'a Arena,
    pub emit_dir: &'a Path,
    pub arm_id: &'a str,
    pub emits: &'a BTreeMap<EmitKey, EmitRow>,
    /// Packets longer than this are over budget (no limit when absent).
    pub budget_bytes: Option<u64>,
    /// Rows of A, B and S kept in a privacy input; 0 leaves it out.
    pub privacy_rows: usize,
}

/// One ledger entry before serialisation.
pub struct Entry {
    pub auditor: String,
    pub sample_seed: u64,
    pub multiplier: u32,
    pub input: FidelityCellInput,
    pub reason: String,
    pub packet_bytes: u64,
    pub artifact_class: String,
    pub synthetic_rows: u64,
    pub l_s: Option<f64>,
    pub r_s: Option<f64>,
    pub c2st_auc: Option<f64>,
}

/// A synthetic table that passed every file check, in the cell's unit space.
struct Scored {
    x: Matrix,
    y: Vec<f64>,
    realism: Option<RealismInput>,
    c2st_auc: Option<f64>,
    privacy: Option<EmpiricalPrivacyInput>,
}

enum FileState {
    Scored(Box<Scored>),
    Terminal(FidelityCellInput, String),
}

struct FileOutcome {
    state: FileState,
    packet_bytes: u64,
    artifact_class: String,
    synthetic_rows: u64,
}

fn failed(reason: &str) -> FileState {
    FileState::Terminal(
        FidelityCellInput::Failed {
            reason: reason.to_string(),
        },
        reason.to_string(),
    )
}

/// Six decimals in unit space: equal raw values stay equal and the rows take half the bytes.
fn compact(rows: &[Vec<f64>], keep: usize) -> Vec<Vec<f64>> {
    rows.iter()
        .take(keep)
        .map(|row| row.iter().map(|v| (v * 1e6).round() / 1e6).collect())
        .collect()
}

fn privacy_input(
    unit_a: &[Vec<f64>],
    unit_b: &[Vec<f64>],
    unit_s: &[Vec<f64>],
    keep: usize,
    min_emitted_support: u64,
) -> Option<EmpiricalPrivacyInput> {
    if keep == 0 {
        return None;
    }
    let (a, b, s) = (
        compact(unit_a, keep),
        compact(unit_b, keep),
        compact(unit_s, keep),
    );
    let matched = a.len().min(b.len()) as u64;
    Some(EmpiricalPrivacyInput {
        max_distance_pairs: 4 * matched * s.len() as u64,
        real_fit_a: a,
        real_holdout_b: b,
        synthetic: s,
        min_emitted_support,
        canary_rows: Vec::new(),
    })
}

/// Checks on the values of a synthetic table; `Err` is the failure reason.
fn check_values(head: Head, rows: &Rows) -> Result<(), &'static str> {
    if rows.x.iter().any(|v| v.is_infinite()) || rows.y.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite_rows");
    }
    if !targets_valid(head, &rows.y) {
        return Err("invalid_class_label");
    }
    if head != Head::Regression && !target_varies(&rows.y) {
        return Err("single_class_target");
    }
    Ok(())
}

fn load_file(
    ctx: &ScoreContext,
    cell: &Cell,
    data: &CellData,
    seed: u64,
    multiplier: u32,
) -> FileOutcome {
    let key = (cell.cell_id.clone(), seed, multiplier);
    let Some(emit) = ctx.emits.get(&key) else {
        return FileOutcome {
            state: failed("missing_emit_row"),
            packet_bytes: 0,
            artifact_class: String::new(),
            synthetic_rows: 0,
        };
    };
    if !known_class(&emit.artifact_class) {
        // Free text from a log is never copied into a scored output.
        return FileOutcome {
            state: failed("unknown_artifact_class"),
            packet_bytes: emit.packet_bytes,
            artifact_class: String::new(),
            synthetic_rows: 0,
        };
    }
    let outcome = |state, synthetic_rows| FileOutcome {
        state,
        packet_bytes: emit.packet_bytes,
        artifact_class: emit.artifact_class.clone(),
        synthetic_rows,
    };
    let over_budget = |cap: u64| {
        FileState::Terminal(
            FidelityCellInput::OverBudget {
                actual_bytes: emit.packet_bytes,
                cap_bytes: cap,
            },
            EMIT_OVER_BUDGET.to_string(),
        )
    };
    let reason = sanitize_reason(&emit.reason);
    match emit.status.as_str() {
        EMIT_OK => {}
        EMIT_FAILED => return outcome(failed(&format!("emit:{reason}")), 0),
        EMIT_UNSUPPORTED => {
            let state =
                FileState::Terminal(FidelityCellInput::Unsupported, format!("emit:{reason}"));
            return outcome(state, 0);
        }
        EMIT_OVER_BUDGET => return outcome(over_budget(ctx.budget_bytes.unwrap_or(0)), 0),
        _ => return outcome(failed("unknown_emit_status"), 0),
    }
    if let Some(cap) = ctx.budget_bytes.filter(|cap| emit.packet_bytes > *cap) {
        return outcome(over_budget(cap), 0);
    }
    let rel = mat_relpath(ctx.arm_id, &cell.cell_id, seed, multiplier);
    if emit.file != rel {
        return outcome(failed("file_name_mismatch"), 0);
    }
    let path = ctx.emit_dir.join(&rel);
    if !path.is_file() {
        return outcome(failed("missing_file"), 0);
    }
    let Ok(mat) = read_mat(&path) else {
        return outcome(failed("malformed_file"), 0);
    };
    let rows = mat.rows as u64;
    if mat.x_cols != cell.cols.len() {
        return outcome(failed("shape_mismatch"), rows);
    }
    let scheduled = cell.a_idx.len() * multiplier as usize;
    let exact = emit.artifact_class != CLASS_CONTROL;
    if emit.rows != rows || mat.rows > max_train_rows(cell) || (exact && mat.rows != scheduled) {
        return outcome(failed("row_count_mismatch"), rows);
    }
    if mat.rows < MIN_ROLE_ROWS {
        return outcome(failed("too_few_rows"), rows);
    }
    let synthetic = Rows::from_mat(&mat);
    if let Err(reason) = check_values(data.head, &synthetic) {
        return outcome(failed(reason), rows);
    }
    // Realism compares the synthetic rows with real rows the generator never saw (B).
    let (unit_a, unit_b) = (data.card.unit_rows(&data.a), data.card.unit_rows(&data.b));
    let unit_s = data.card.unit_rows(&synthetic);
    let realism = build_native_realism_input(&unit_b, &unit_s, None, None, 0, C2ST_TREES).ok();
    let c2st_auc = realism
        .as_ref()
        .and_then(|r| roc_auc(&r.c2st_labels, &r.c2st_scores, true).ok());
    let support = emit.min_emitted_support.unwrap_or(UNDECLARED_SUPPORT);
    let scored = Scored {
        x: data.card.features(&synthetic),
        y: data.card.targets(&synthetic),
        realism,
        c2st_auc,
        privacy: privacy_input(&unit_a, &unit_b, &unit_s, ctx.privacy_rows, support),
    };
    outcome(FileState::Scored(Box::new(scored)), rows)
}

/// Fit one auditor on the synthetic table and assemble the complete input.
fn audit(
    cell: &Cell,
    data: &CellData,
    reference: &RealRef,
    scored: &Scored,
    test: &Matrix,
) -> Result<(FidelityCellInput, f64, Option<f64>), &'static str> {
    let recipe = Recipe::new(auditor_seed(cell), scored.x.rows, data.a.n);
    let raw = auditors::fit_predict(
        &reference.auditor,
        data.head,
        &scored.x,
        &scored.y,
        test,
        recipe,
    )
    .map_err(|_| "auditor_failed_on_synthetic")?;
    let (synth_losses, pred_s) = auditors::losses(data.head, &raw, &reference.y_true);
    let bundle = PredictionBundle {
        task: task_kind(data.head),
        null_losses: reference.null_losses.clone(),
        real_a_losses: reference.real_a_losses.clone(),
        real_b_losses: Some(reference.real_b_losses.clone()),
        synth_losses,
        donor_ids: data.t_donors.clone(),
        y_true: reference.y_true.clone(),
        pred_a: reference.pred_a.clone(),
        pred_b: Some(reference.pred_b.clone()),
        pred_s,
        num_outputs: reference.num_outputs,
    };
    bundle.validate().map_err(|err| err.0)?;
    let (l_s, r_s) = (bundle.mean_synth(), bundle.unclipped_retention_s());
    let input = FidelityCellInput::Complete {
        bundle: Box::new(bundle),
        driver: None,
        deploy_roster: None,
        realism: scored.realism.clone().map(Box::new),
        privacy: scored.privacy.clone().map(Box::new),
    };
    Ok((input, l_s, r_s))
}

/// Every ledger entry of one cell: sample seeds x multipliers x auditors of `references`.
pub fn score_cell(
    ctx: &ScoreContext,
    cell: &Cell,
    references: &[&RealRef],
) -> ArenaResult<Vec<Entry>> {
    let scorable = references.iter().any(|r| r.status == REF_OK);
    let data = if scorable {
        CellData::load(ctx.arena, cell)?.ok()
    } else {
        None
    };
    // A reference that scored this cell was computed on these very T rows.
    let stale = references
        .iter()
        .any(|r| r.status == REF_OK && data.as_ref().is_none_or(|d| d.t.n != r.y_true.len()));
    if stale {
        return Err(ArenaError::Contract(format!(
            "realref disagrees with the arena at cell {}",
            cell.cell_id
        )));
    }
    let test = data.as_ref().map(|d| d.card.features(&d.t));
    let mut entries = Vec::new();
    for &seed in &cell.sample_seeds {
        for &multiplier in &cell.multipliers {
            let file = data
                .as_ref()
                .map(|d| load_file(ctx, cell, d, seed, multiplier));
            for reference in references {
                let mut entry = Entry {
                    auditor: reference.auditor.clone(),
                    sample_seed: seed,
                    multiplier,
                    input: FidelityCellInput::Unsupported,
                    reason: reference.reason.clone(),
                    packet_bytes: file.as_ref().map_or(0, |f| f.packet_bytes),
                    artifact_class: file
                        .as_ref()
                        .map_or_else(String::new, |f| f.artifact_class.clone()),
                    synthetic_rows: file.as_ref().map_or(0, |f| f.synthetic_rows),
                    l_s: None,
                    r_s: None,
                    c2st_auc: None,
                };
                let (Some(data), Some(file), Some(test), true) =
                    (&data, &file, &test, reference.status == REF_OK)
                else {
                    // The cell or this auditor is unsupported on real rows alone.
                    if entry.reason.is_empty() {
                        entry.reason = "real_reference_unsupported".to_string();
                    }
                    entries.push(entry);
                    continue;
                };
                match &file.state {
                    FileState::Terminal(input, reason) => {
                        entry.input = input.clone();
                        entry.reason = reason.clone();
                    }
                    FileState::Scored(scored) => {
                        entry.c2st_auc = scored.c2st_auc;
                        match audit(cell, data, reference, scored, test) {
                            Ok((input, l_s, r_s)) => {
                                entry.input = input;
                                entry.l_s = Some(l_s);
                                entry.r_s = r_s;
                            }
                            Err(reason) => {
                                entry.input = FidelityCellInput::Failed {
                                    reason: reason.to_string(),
                                };
                                entry.reason = reason.to_string();
                            }
                        }
                    }
                }
                entries.push(entry);
            }
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(x: &[f32], y: &[f32]) -> Rows {
        Rows {
            n: y.len(),
            p: 1,
            x: x.to_vec(),
            y: y.to_vec(),
        }
    }

    #[test]
    fn value_checks_name_the_failure() {
        let ok = rows(&[1.0, f32::NAN, 3.0], &[0.0, 1.0, 1.0]);
        assert_eq!(check_values(Head::Binary, &ok), Ok(()));
        let inf = rows(&[1.0, f32::NEG_INFINITY, 3.0], &[0.0, 1.0, 1.0]);
        assert_eq!(check_values(Head::Binary, &inf), Err("nonfinite_rows"));
        let nan_y = rows(&[1.0, 2.0], &[0.5, f32::NAN]);
        assert_eq!(
            check_values(Head::Regression, &nan_y),
            Err("nonfinite_rows")
        );
        let label = rows(&[1.0, 2.0], &[0.0, 0.5]);
        assert_eq!(
            check_values(Head::Binary, &label),
            Err("invalid_class_label")
        );
        let one = rows(&[1.0, 2.0], &[2.0, 2.0]);
        assert_eq!(
            check_values(Head::Multiclass(3), &one),
            Err("single_class_target")
        );
        // A constant regression target is a poor table, not a broken file.
        assert_eq!(check_values(Head::Regression, &one), Ok(()));
    }

    #[test]
    fn privacy_input_is_capped_rounded_and_sized_for_its_budget() {
        let set = |n: usize, base: f64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|r| vec![base + r as f64 + 1.234_567_891e-3, 0.5])
                .collect()
        };
        let (a, b, s) = (set(5, 0.0), set(5, 10.0), set(20, 0.0));
        assert!(privacy_input(&a, &b, &s, 0, 10).is_none());
        let full = privacy_input(&a, &b, &s, 4096, 10).unwrap();
        assert_eq!(full.synthetic.len(), 20);
        assert_eq!(full.max_distance_pairs, 4 * 5 * 20);
        assert_eq!(full.real_fit_a[0], vec![0.001235, 0.5]);
        // Rows of A copied into S stay exact copies after rounding.
        assert_eq!(full.synthetic[3], full.real_fit_a[3]);
        let capped = privacy_input(&a, &b, &s, 3, 7).unwrap();
        assert_eq!((capped.real_fit_a.len(), capped.synthetic.len()), (3, 3));
        assert_eq!(capped.max_distance_pairs, 36);
        assert_eq!(capped.min_emitted_support, 7);
        dope_core::dvs::v3::privacy::assess_empirical_privacy(&full).unwrap();
    }
}
