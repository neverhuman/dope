//! `dope_kpi render-readme`: the README block `DOPE1 SCOREBOARD` from receipts.
//!
//! One table row per arm (arm id, artifact class, packet bytes median / max /
//! share within the headline budget, TRG with its 95% interval, per-task TRG
//! with family counts, unclipped median retention of the three auditors,
//! median C2ST, failed / applicable cells), one line per comparison (delta
//! with its interval, Holm p, guards, outcome), the sign-flip null when a
//! receipt holds one, and a footer with the contract SHA-256, the panel and
//! the SHA-256 of every receipt. Numbers have 3 decimals; the block is a pure
//! function of the receipt bytes. Several receipts of one run (same
//! contract, schedule, real reference and panel) render as one block.

use super::jsonl::{parse_json, read_file, sha256_hex};
use super::primary::ArmSummary;
use super::receipt::{Receipt, RECEIPT_SCHEMA};
use super::rule::{Check, Item};
use super::rundir::{contract, TASKS};
use dope_arena::ArenaResult;
use std::path::PathBuf;

pub const BLOCK: &str = "DOPE1 SCOREBOARD";

#[derive(clap::Args, Debug)]
pub struct RenderArgs {
    /// Receipt written by `primary`, `compare` or `null` (repeat for several of one run).
    #[arg(long = "receipt", required = true)]
    pub receipts: Vec<PathBuf>,
}

/// `x` with 3 decimals and no negative zero.
pub fn fixed3(x: f64) -> String {
    let text = format!("{x:.3}");
    if text == "-0.000" {
        "0.000".to_string()
    } else {
        text
    }
}

/// `x` with 3 decimals and an explicit sign (none for zero).
pub fn signed3(x: f64) -> String {
    let text = fixed3(x);
    if text == "0.000" || text.starts_with('-') {
        text
    } else {
        format!("+{text}")
    }
}

fn opt3(x: Option<f64>) -> String {
    x.map_or_else(|| "-".to_string(), fixed3)
}

fn p_value(p: f64) -> String {
    if p < 0.001 {
        "<0.001".to_string()
    } else {
        fixed3(p)
    }
}

fn bytes_cell(arm: &ArmSummary) -> String {
    let b = &arm.bytes;
    let (Some(median), Some(share)) = (b.median, b.share_within_budget) else {
        return "-".to_string();
    };
    let median = if median.fract() == 0.0 {
        format!("{median:.0}")
    } else {
        format!("{median:.1}")
    };
    format!("{median} / {} / {}", b.max, fixed3(share))
}

fn arm_row(arm: &ArmSummary) -> String {
    let p = &arm.primary;
    let trg = match (p.trg, p.ci_low, p.ci_high) {
        (Some(t), Some(low), Some(high)) => {
            format!("{} [{}, {}]", fixed3(t), fixed3(low), fixed3(high))
        }
        _ => "-".to_string(),
    };
    let tasks: Vec<String> = TASKS
        .iter()
        .map(|task| {
            p.tasks.iter().find(|t| t.task == *task).map_or_else(
                || "-".to_string(),
                |t| format!("{} ({})", fixed3(t.value), t.families),
            )
        })
        .collect();
    let retention: Vec<String> = arm
        .median_retention
        .iter()
        .map(|m| opt3(m.median))
        .collect();
    let mut cells = format!("{} / {}", p.failed_cells, p.cells);
    if p.not_applicable_cells > 0 {
        cells.push_str(&format!(" ({} n/a)", p.not_applicable_cells));
    }
    format!(
        "| `{}` | {} | {} | {} | {} | {} | {} | {} |",
        arm.arm_id,
        arm.artifact_class,
        bytes_cell(arm),
        trg,
        tasks.join(" | "),
        retention.join(" / "),
        opt3(arm.c2st_median),
        cells
    )
}

fn guard_name(name: &str) -> &str {
    match name {
        "task_ci_low" => "tasks",
        "mlp_delta" => "mlp",
        "realism_coverage_delta" => "realism",
        "privacy_coverage_delta" => "privacy",
        "exact_copy_excess" => "copies",
        other => other,
    }
}

fn guard_word(item: &Item) -> String {
    let check = match item.check {
        Check::Pass => "pass",
        Check::Fail => "FAIL",
        Check::NotEvaluated => "n/e",
    };
    format!("{} {check}", guard_name(&item.name))
}

fn comparison_row(c: &super::compare::ComparisonResult) -> String {
    let mut guards: Vec<String> = c.decision.guards.iter().map(guard_word).collect();
    if c.kind == "seed_group" {
        let signs = c
            .decision
            .conditions
            .iter()
            .find(|i| i.name == "seed_signs");
        let same = signs.and_then(|i| i.value).unwrap_or(0.0);
        guards.push(format!("seeds {same:.0}/{}", c.members.len()));
    }
    format!(
        "| `{}` vs `{}` | {} [{}, {}] | {} | {} | **{}** |",
        c.id,
        c.baseline,
        signed3(c.delta),
        signed3(c.ci_low),
        signed3(c.ci_high),
        p_value(c.p_holm),
        guards.join("; "),
        c.decision.outcome.word()
    )
}

/// The block for `receipts` (file bytes in command-line order).
pub fn render(receipts: &[Vec<u8>]) -> ArenaResult<String> {
    let mut parsed: Vec<(Receipt, String)> = Vec::new();
    for (index, bytes) in receipts.iter().enumerate() {
        let receipt: Receipt = parse_json(&format!("receipt {}", index + 1), bytes)?;
        if receipt.schema != RECEIPT_SCHEMA {
            return contract(format!("receipt {} is not {RECEIPT_SCHEMA}", index + 1));
        }
        parsed.push((receipt, sha256_hex(bytes)));
    }
    let Some((first, _)) = parsed.first() else {
        return contract("render-readme needs a receipt");
    };
    let same_run = |r: &Receipt| {
        r.contract.sha256 == first.contract.sha256
            && r.inputs.schedule_sha256 == first.inputs.schedule_sha256
            && r.inputs.realref_sha256 == first.inputs.realref_sha256
            && r.panel == first.panel
            && r.bootstrap == first.bootstrap
    };
    if !parsed.iter().all(|(r, _)| same_run(r)) {
        return contract(
            "receipts differ in contract, schedule, real reference, panel or bootstrap",
        );
    }
    let mut arms: Vec<&ArmSummary> = Vec::new();
    for arm in parsed.iter().flat_map(|(r, _)| &r.arms) {
        match arms.iter().find(|a| a.arm_id == arm.arm_id) {
            Some(known) if *known != arm => {
                return contract(format!("receipts disagree on arm {}", arm.arm_id));
            }
            Some(_) => {}
            None => arms.push(arm),
        }
    }
    let budget = arms.first().map_or(0, |a| a.bytes.budget);
    let mut out = vec![
        format!("<!-- BEGIN {BLOCK} -->"),
        format!("<!-- Generated by dope_kpi render-readme from {RECEIPT_SCHEMA} receipts; do not edit. -->"),
    ];
    if !arms.is_empty() {
        out.push(format!(
            "| Arm | Class | Bytes median / max / share <= {budget} B | TRG [95% CI] | Regression (families) | Binary (families) | Multiclass (families) | Median retention linear / gbdt / mlp | C2ST | Failed / cells |"
        ));
        out.push("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|".to_string());
        out.extend(arms.iter().map(|arm| arm_row(arm)));
    }
    let comparisons: Vec<_> = parsed.iter().flat_map(|(r, _)| &r.comparisons).collect();
    if !comparisons.is_empty() {
        out.push(String::new());
        out.push("| Comparison | Delta TRG [95% CI] | Holm p | Guards | Outcome |".to_string());
        out.push("|---|---:|---:|---|---|".to_string());
        out.extend(comparisons.iter().map(|c| comparison_row(c)));
    }
    for null in parsed.iter().filter_map(|(r, _)| r.null.as_ref()) {
        let blocked: Vec<&str> = null
            .not_evaluated_guards
            .iter()
            .map(|g| guard_name(g))
            .collect();
        let blocked = if blocked.is_empty() {
            String::new()
        } else {
            format!(" (not evaluated: {})", blocked.join(", "))
        };
        out.push(String::new());
        let rate = |r: &super::null::Rate| {
            format!(
                "{} [{}, {}]",
                fixed3(r.rate),
                fixed3(r.wilson95_low),
                fixed3(r.wilson95_high)
            )
        };
        out.push(format!(
            "Sign-flip null, `{}` vs `{}`, {} replicates (Wilson 95%): false PROMOTE {}{blocked}; with bytes, copies and unevaluated guards passing {}; conditions (1)-(3) alone {}.",
            null.candidate,
            null.baseline,
            null.replicates,
            rate(&null.false_promote),
            rate(&null.flip_guard_rule),
            rate(&null.statistical_pass)
        ));
    }
    let shas: Vec<String> = parsed.iter().map(|(_, sha)| format!("`{sha}`")).collect();
    out.push(String::new());
    out.push(format!(
        "Contract `{}` sha256 `{}`; panel `{}`; receipt sha256 {}.",
        first.contract.id,
        first.contract.sha256,
        first.panel.id,
        shas.join(", ")
    ));
    out.push(format!("<!-- END {BLOCK} -->"));
    Ok(out.join("\n"))
}

pub fn run(args: &RenderArgs) -> ArenaResult<String> {
    let receipts = args
        .receipts
        .iter()
        .map(|path| read_file(path))
        .collect::<ArenaResult<Vec<_>>>()?;
    render(&receipts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_have_three_decimals_and_no_negative_zero() {
        assert_eq!(fixed3(-0.0004), "0.000");
        assert_eq!(fixed3(-0.0306), "-0.031");
        assert_eq!(signed3(0.0726), "+0.073");
        assert_eq!(signed3(-0.0001), "0.000");
        assert_eq!(signed3(-0.25), "-0.250");
        assert_eq!(p_value(0.0004), "<0.001");
        assert_eq!(p_value(0.5), "0.500");
        assert_eq!(opt3(None), "-");
    }
}
