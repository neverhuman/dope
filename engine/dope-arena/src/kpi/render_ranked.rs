//! Ranked Transfer Score summary for `dope_kpi render-readme --ranked`.

use super::compare::ComparisonResult;
use super::primary::ArmSummary;
use super::rundir::contract;
use dope_arena::ArenaResult;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct RankedOptions<'a> {
    pub budget: u64,
    pub noise_sd: f64,
    pub labels: &'a BTreeMap<String, String>,
    pub include: &'a [String],
    pub references: &'a [String],
    pub show_size: bool,
}

/// Transfer Score (`100 * (1 + TRG)`) to one decimal with no negative zero.
pub(super) fn score1(trg: f64) -> String {
    let text = format!("{:.1}", 100.0 * (1.0 + trg));
    if text == "-0.0" {
        "0.0".to_string()
    } else {
        text
    }
}

/// Signed Transfer Score delta (`100 * delta`) to one decimal.
pub(super) fn signed_pts1(delta: f64) -> String {
    let text = format!("{:+.1}", 100.0 * delta);
    if text == "-0.0" || text == "+0.0" {
        "0.0".to_string()
    } else {
        text
    }
}

/// Grouped decimal byte count: `1,787 B`, `867.5 B`.
fn grouped_bytes(bytes: f64) -> String {
    let text = if bytes.fract() == 0.0 {
        format!("{bytes:.0}")
    } else {
        format!("{bytes:.1}")
    };
    let (whole, frac) = text.split_once('.').unwrap_or((&text, ""));
    let mut grouped = String::new();
    for (i, digit) in whole.chars().enumerate() {
        if i > 0 && (whole.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    if frac.is_empty() {
        format!("{grouped} B")
    } else {
        format!("{grouped}.{frac} B")
    }
}

/// Compact SI byte text for cross-scale generator rankings (`1,787 B`, `141 KB`, `2.94 MB`).
pub(super) fn human_bytes(bytes: f64) -> String {
    if bytes < 10_000.0 {
        grouped_bytes(bytes)
    } else if bytes < 1_000_000.0 {
        let kb = bytes / 1_000.0;
        if kb < 100.0 {
            format!("{kb:.1} KB")
        } else {
            format!("{kb:.0} KB")
        }
    } else {
        let mb = bytes / 1_000_000.0;
        if mb < 100.0 {
            format!("{mb:.2} MB")
        } else {
            format!("{mb:.1} MB")
        }
    }
}

pub(super) fn arm_labels(
    items: &[String],
    has_arm: impl Fn(&str) -> bool,
) -> ArenaResult<BTreeMap<String, String>> {
    let mut labels = BTreeMap::new();
    for item in items {
        let Some((arm, text)) = item.split_once('=') else {
            return contract("--label takes ARM=Text");
        };
        if text.trim().is_empty() {
            return contract(format!("--label {arm} has no text"));
        }
        if !has_arm(arm) {
            return contract(format!("--label names arm {arm}, which no receipt holds"));
        }
        if labels.insert(arm.to_string(), text.to_string()).is_some() {
            return contract(format!("--label names arm {arm} twice"));
        }
    }
    Ok(labels)
}

pub(super) fn validate_ranked_arms(
    include: &[String],
    references: &[String],
    arms: &[&ArmSummary],
) -> ArenaResult<()> {
    let mut seen_include = BTreeSet::new();
    for id in include {
        let Some(arm) = arms.iter().find(|a| a.arm_id == *id) else {
            return contract(format!("--include names arm {id}, which no receipt holds"));
        };
        if !seen_include.insert(id.as_str()) {
            return contract(format!("--include names arm {id} twice"));
        }
        if arm.artifact_class == "control" {
            return contract(format!(
                "--include arm {id} is a control; pass it with --reference"
            ));
        }
        if arm.primary.trg.is_none() {
            return contract(format!("--include arm {id} has no primary TRG"));
        }
    }
    let mut seen_ref = BTreeSet::new();
    for id in references {
        let Some(arm) = arms.iter().find(|a| a.arm_id == *id) else {
            return contract(format!(
                "--reference names arm {id}, which no receipt holds"
            ));
        };
        if !seen_ref.insert(id.as_str()) {
            return contract(format!("--reference names arm {id} twice"));
        }
        if seen_include.contains(id.as_str()) {
            return contract(format!(
                "arm {id} cannot be passed to both --include and --reference"
            ));
        }
        if arm.artifact_class != "control" {
            return contract(format!(
                "--reference arm {id} is not a control; pass it with --include"
            ));
        }
        if arm.primary.trg.is_none() {
            return contract(format!("--reference arm {id} has no primary TRG"));
        }
    }
    Ok(())
}

/// Oriented paired delta (`top - second`) when a comparison links the top two ranked arms.
fn oriented_top_pair(
    comparisons: &[&ComparisonResult],
    top_id: &str,
    second_id: &str,
) -> Option<(f64, f64, f64)> {
    comparisons.iter().find_map(|c| {
        if c.id == top_id && c.baseline == second_id {
            Some((c.delta, c.ci_low, c.ci_high))
        } else if c.id == second_id && c.baseline == top_id {
            Some((-c.delta, -c.ci_high, -c.ci_low))
        } else {
            None
        }
    })
}

fn above_budget_footnote(scored: &[(f64, &ArmSummary)], budget: u64) -> Option<String> {
    let above: Vec<&ArmSummary> = scored
        .iter()
        .map(|(_, a)| *a)
        .filter(|a| a.bytes.scored_above_budget > 0)
        .collect();
    if above.is_empty() {
        return None;
    }
    let all_zero_within = above.iter().all(|a| a.bytes.within_budget == 0);
    let floor = above[0].trg_at_budget;
    let same_floor = floor.is_some() && above.iter().all(|a| a.trg_at_budget == floor);
    if all_zero_within && same_floor {
        let floor_score = score1(floor.unwrap_or(-1.0));
        Some(format!(
            "† Every observed file exceeds {budget} B (0 files within {budget} B; Transfer Score shown is for those oversized files, whereas applying the {budget} B cap penalty without re-fitting scores {floor_score})."
        ))
    } else {
        Some(format!(
            "† Scored with files above the {budget} B cap (Transfer Score reflects rows as scored; see full metrics)."
        ))
    }
}

pub(super) fn ranked_summary(
    out: &mut Vec<String>,
    arms: &[&ArmSummary],
    comparisons: &[&ComparisonResult],
    opts: &RankedOptions<'_>,
) {
    let label = |id: &str| {
        opts.labels.get(id).map_or_else(
            || format!("`{id}`"),
            |text| text.replace('|', "\\|").replace(['\r', '\n'], " "),
        )
    };
    let mut scored: Vec<_> = arms
        .iter()
        .copied()
        .filter(|arm| {
            if opts.include.is_empty() {
                arm.artifact_class == "source_free"
            } else {
                opts.include.iter().any(|id| id == &arm.arm_id)
            }
        })
        .filter_map(|arm| arm.primary.trg.map(|trg| (trg, arm)))
        .collect();
    scored.sort_by(|(a, _), (b, _)| b.total_cmp(a));
    out.push(String::new());
    if let ([comparison], Some(arm)) = (comparisons, arms.first()) {
        out.push(format!(
            "**{} points at {}× sampling** for {} vs {} (paired 95% CI {} to {}).",
            signed_pts1(comparison.delta),
            arm.primary.multiplier,
            label(&comparison.id),
            label(&comparison.baseline),
            signed_pts1(comparison.ci_low),
            signed_pts1(comparison.ci_high)
        ));
        out.push(String::new());
    } else if scored.len() >= 2 {
        let (top_trg, top_arm) = scored[0];
        let (_, second_arm) = scored[1];
        if let Some((delta, lo, hi)) =
            oriented_top_pair(comparisons, &top_arm.arm_id, &second_arm.arm_id)
        {
            let span = if lo <= 0.0 && hi >= 0.0 {
                "; interval includes zero"
            } else {
                ""
            };
            let kind = if span.is_empty() {
                "gain"
            } else {
                "difference"
            };
            out.push(format!(
                "**{} ranks first at {}** ({}× sampling); paired {kind} vs {} is **{} points** (95% CI {} to {}{span}).",
                label(&top_arm.arm_id),
                score1(top_trg),
                top_arm.primary.multiplier,
                label(&second_arm.arm_id),
                signed_pts1(delta),
                signed_pts1(lo),
                signed_pts1(hi)
            ));
            out.push(String::new());
        }
    }
    if !scored.is_empty() {
        out.push("**Transfer Score = 100 × (1 + TRG).** A score of 100 matches a second real training sample. Higher is better.".to_string());
        if let Some(copy_trg) = arms
            .iter()
            .find(|a| a.arm_id == "copy_a" && a.artifact_class == "control")
            .and_then(|a| a.primary.trg)
        {
            out.push(format!(
                "On this report a copy of the real training rows scores **{}**.",
                score1(copy_trg)
            ));
        }
        let show_size =
            opts.show_size || scored.iter().any(|(_, a)| a.bytes.scored_above_budget > 0);
        out.push(String::new());
        if show_size {
            out.push("| Rank | Generator | Transfer Score ↑ | Median file |".to_string());
            out.push("|---:|---|---:|---:|".to_string());
        } else {
            out.push(format!(
                "| Rank | Encoder ({} B budget) | Transfer Score ↑ |",
                opts.budget
            ));
            out.push("|---:|---|---:|".to_string());
        }
        let mut start = 0;
        while start < scored.len() {
            let top = scored[start].0;
            let size = scored[start..]
                .iter()
                .take_while(|(trg, _)| *trg == top || top - *trg < opts.noise_sd)
                .count();
            let rank = start + 1;
            let tie = if size > 1 { "=" } else { "" };
            let color = ["🟩", "🟦"].get(start).unwrap_or(&"⬜");
            for (trg, arm) in &scored[start..start + size] {
                let above_budget = if arm.bytes.scored_above_budget > 0 {
                    " †"
                } else {
                    ""
                };
                if show_size {
                    let bytes = arm
                        .bytes
                        .median
                        .map_or_else(|| "—".to_string(), human_bytes);
                    out.push(format!(
                        "| {color} {rank}{tie} | {} | {}{above_budget} | {bytes} |",
                        label(&arm.arm_id),
                        score1(*trg)
                    ));
                } else {
                    out.push(format!(
                        "| {color} {rank}{tie} | {} | {}{above_budget} |",
                        label(&arm.arm_id),
                        score1(*trg)
                    ));
                }
            }
            start += size;
        }
        for id in opts.references {
            if let Some(arm) = arms.iter().find(|a| a.arm_id == *id) {
                if let Some(trg) = arm.primary.trg {
                    if show_size {
                        out.push(format!(
                            "| — | *{}* | *{}* | *—* |",
                            label(&arm.arm_id),
                            score1(trg)
                        ));
                    } else {
                        out.push(format!(
                            "| — | *{}* | *{}* |",
                            label(&arm.arm_id),
                            score1(trg)
                        ));
                    }
                }
            }
        }
        out.push(String::new());
        if let Some(note) = above_budget_footnote(&scored, opts.budget) {
            out.push(note);
            out.push(String::new());
        }
        if opts.noise_sd > 0.0 {
            out.push(format!(
                "A tie spans less than {} point{}.",
                100.0 * opts.noise_sd,
                if 100.0 * opts.noise_sd == 1.0 {
                    ""
                } else {
                    "s"
                }
            ));
            out.push(String::new());
        }
    }
    if let [comparison] = comparisons {
        out.push(format!(
            "**Promotion outcome is {}.**",
            comparison.decision.outcome.word()
        ));
        out.push(String::new());
    }
    out.push("<details>".to_string());
    out.push("<summary>Full benchmark metrics and guard results</summary>".to_string());
    out.push(String::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_and_size_formats_are_deterministic() {
        assert_eq!(score1(-0.121), "87.9");
        assert_eq!(score1(-1.0004), "0.0");
        assert_eq!(score1(-1.039), "-3.9");
        assert_eq!(signed_pts1(0.1078), "+10.8");
        assert_eq!(signed_pts1(-0.0004), "0.0");
        assert_eq!(human_bytes(1787.0), "1,787 B");
        assert_eq!(human_bytes(867.5), "867.5 B");
        assert_eq!(human_bytes(12_340.0), "12.3 KB");
        assert_eq!(human_bytes(140_675.5), "141 KB");
        assert_eq!(human_bytes(475_125.5), "475 KB");
        assert_eq!(human_bytes(1_053_957.5), "1.05 MB");
        assert_eq!(human_bytes(2_937_782.5), "2.94 MB");
    }
}
