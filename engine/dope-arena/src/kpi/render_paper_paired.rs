//! Table 2 of `dope_kpi render-paper-tables`: paired `v3 - method` Transfer Score differences,
//! one block per look. The report look is the first block; a second look (Addendum H part (a),
//! a different set of DEV row splits) is an optional second block with its own Holm family.
//! Looks are never pooled: each block reads only its own primary and compare receipts.

use super::compare::ComparisonResult;
use super::primary::ArmSummary;
use super::receipt::Receipt;
use super::render_paper_tex::{display_name, short_guard, tex_delta_ci, tex_escape, tex_p};
use super::rule::{Check, Outcome};
use super::rundir::contract;
use dope_arena::ArenaResult;
use std::collections::BTreeMap;

/// One look of Table 2: its comparisons in the table's method order, and their Holm p.
pub(super) struct LookBlock<'a> {
    /// Block heading; `None` for a table with a single look (no heading row).
    pub title: Option<String>,
    pub comps: Vec<&'a ComparisonResult>,
    pub holm_p: Vec<f64>,
}

pub(super) fn same_run(a: &Receipt, b: &Receipt) -> bool {
    a.contract.sha256 == b.contract.sha256
        && a.inputs.schedule_sha256 == b.inputs.schedule_sha256
        && a.inputs.schedule_panel_sha256 == b.inputs.schedule_panel_sha256
        && a.inputs.realref_sha256 == b.inputs.realref_sha256
        && a.panel == b.panel
        && a.bootstrap == b.bootstrap
}

pub(super) fn check_arm_provenance(
    primary: &Receipt,
    comp_receipt: &Receipt,
    arm_id: &str,
    index: usize,
) -> ArenaResult<()> {
    match (
        primary.inputs.rows_sha256.get(arm_id),
        comp_receipt.inputs.rows_sha256.get(arm_id),
    ) {
        (Some(a), Some(b)) if a == b => {}
        _ => {
            return contract(format!(
                "compare receipt {index} differs from primary on rows_sha256 for arm {arm_id}"
            ));
        }
    }
    if primary.inputs.guards_sha256.get(arm_id) != comp_receipt.inputs.guards_sha256.get(arm_id) {
        return contract(format!(
            "compare receipt {index} differs from primary on guards_sha256 for arm {arm_id}"
        ));
    }
    if primary.inputs.guards_summary_sha256.get(arm_id)
        != comp_receipt.inputs.guards_summary_sha256.get(arm_id)
    {
        return contract(format!(
            "compare receipt {index} differs from primary on guards_summary_sha256 for arm {arm_id}"
        ));
    }
    if let (Some(pa), Some(ca)) = (
        primary.arms.iter().find(|a| a.arm_id == arm_id),
        comp_receipt.arms.iter().find(|a| a.arm_id == arm_id),
    ) {
        if pa != ca {
            return contract(format!(
                "compare receipt {index} disagrees with primary summary for arm {arm_id}"
            ));
        }
    }
    Ok(())
}

/// The comparison `candidate` vs each of `method_ids` (in that order): exactly one per method across
/// `receipts`, each bound to `primary` by rows, guards and arm summary for both arms.
pub(super) fn look_comparisons<'a>(
    primary: &Receipt,
    receipts: &'a [Receipt],
    candidate: &str,
    method_ids: &[&str],
) -> ArenaResult<Vec<&'a ComparisonResult>> {
    let mut out = Vec::with_capacity(method_ids.len());
    for &method in method_ids {
        let hits: Vec<(usize, &Receipt, &ComparisonResult)> = receipts
            .iter()
            .enumerate()
            .flat_map(|(i, r)| r.comparisons.iter().map(move |c| (i + 1, r, c)))
            .filter(|(_, _, c)| c.id == candidate && c.baseline == method)
            .collect();
        if hits.len() != 1 {
            return contract(format!(
                "expected 1 comparison for {candidate} vs {method}, found {}",
                hits.len()
            ));
        }
        let (index, comp_receipt, comp) = hits[0];
        check_arm_provenance(primary, comp_receipt, candidate, index)?;
        check_arm_provenance(primary, comp_receipt, method, index)?;
        out.push(comp);
    }
    Ok(out)
}

pub(super) fn effective_outcome(comp: &ComparisonResult, p_holm: f64) -> ArenaResult<Outcome> {
    let Some(holm_thresh) = comp
        .decision
        .conditions
        .iter()
        .find(|c| c.name == "holm_p")
        .and_then(|c| c.threshold)
    else {
        return contract(format!(
            "comparison {} vs {} lacks a holm_p condition threshold",
            comp.id, comp.baseline
        ));
    };
    let holm_pass = p_holm < holm_thresh;
    let other_conds_pass = comp
        .decision
        .conditions
        .iter()
        .filter(|c| c.name != "holm_p")
        .all(|c| c.check == Check::Pass);
    let all_guards_pass = comp.decision.guards.iter().all(|g| g.check == Check::Pass);
    if holm_pass && other_conds_pass && all_guards_pass {
        Ok(Outcome::Promote)
    } else if comp.decision.outcome == Outcome::Reject {
        Ok(Outcome::Reject)
    } else {
        Ok(Outcome::Hold)
    }
}

fn paired_row(
    arm: &ArmSummary,
    comp: &ComparisonResult,
    p_h: f64,
    labels: &BTreeMap<String, String>,
) -> ArenaResult<String> {
    let outcome = effective_outcome(comp, p_h)?;
    let matched = comp.matched.as_ref().map_or_else(
        || "--".to_string(),
        |m| tex_delta_ci(m.delta, m.ci_low, m.ci_high),
    );
    let missed: Vec<String> = comp
        .decision
        .guards
        .iter()
        .filter_map(|g| match g.check {
            Check::Pass => None,
            Check::Fail => Some(short_guard(&g.name).to_string()),
            Check::NotEvaluated => Some(format!("{} (n/e)", short_guard(&g.name))),
        })
        .collect();
    let guards_text = if missed.is_empty() {
        "none".to_string()
    } else {
        missed.join(", ")
    };
    Ok(format!(
        "{} & {} & {} & {} & {} & \\textbf{{{}}} & {} \\\\",
        display_name(&arm.arm_id, labels, false),
        tex_delta_ci(comp.delta, comp.ci_low, comp.ci_high),
        matched,
        tex_p(comp.p_one_sided),
        tex_p(p_h),
        outcome.word(),
        guards_text
    ))
}

pub(super) fn render_paired_tex(
    methods: &[&ArmSummary],
    blocks: &[LookBlock<'_>],
    labels: &BTreeMap<String, String>,
) -> ArenaResult<String> {
    let mult = methods.first().map_or(4, |a| a.primary.multiplier);
    let matched_mult = blocks
        .first()
        .and_then(|b| b.comps.first())
        .and_then(|c| c.matched.as_ref())
        .map_or(1, |m| m.multiplier);
    let m = methods.len();
    let mut lines = vec![
        "% Generated by dope_kpi render-paper-tables -- DO NOT EDIT BY HAND.".to_string(),
        r"\begin{tabular}{@{}lrrrrll@{}}".to_string(),
        r"\toprule".to_string(),
        format!(
            r"Method & \makecell{{v3 \ensuremath{{-}} method,\\${mult}\times$ [95\% CI]}} & \makecell{{${matched_mult}\times$ [95\% CI]}} & \makecell{{one-sided\\$p$}} & \makecell{{Holm $p$\\($m={m}$)}} & outcome & guards missed \\"
        ),
        r"\midrule".to_string(),
    ];
    for (i, block) in blocks.iter().enumerate() {
        if block.comps.len() != m || block.holm_p.len() != m {
            return contract(format!(
                "look {} has {} comparisons for {m} methods",
                i + 1,
                block.comps.len()
            ));
        }
        if i > 0 {
            lines.push(r"\midrule".to_string());
        }
        if let Some(title) = &block.title {
            lines.push(format!(
                r"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{}}}}} \\",
                tex_escape(title)
            ));
        }
        for ((arm, comp), &p_h) in methods
            .iter()
            .zip(block.comps.iter())
            .zip(block.holm_p.iter())
        {
            lines.push(paired_row(arm, comp, p_h, labels)?);
        }
    }
    lines.extend([r"\bottomrule".to_string(), r"\end{tabular}".to_string()]);
    Ok(lines.join("\n") + "\n")
}
