//! `dope_kpi render-paper-tables`: receipt-bound LaTeX tables for `docs/whitepaper-v3/`.
//!
//! Emits two `\label`-free `booktabs` tables directly from committed receipts:
//! - Table 1 (`v3-ranking-table.tex`): DOPE v3, external generators, and
//!   real-data controls in Transfer Score (`100 * (1 + TRG)`), wrapped into
//!   panels (a) and (b) so every row stays inside the manuscript width.
//! - Table 2 (`v3-paired-table.tex`): paired `v3 - method` Transfer Score
//!   differences at 4x and 1x, one-sided p, Holm-Bonferroni p across the
//!   external methods, contract outcome, and missed guards.

use super::boot::holm;
use super::jsonl::{canonical_json, parse_json, read_file, sha256_hex, write_atomic};
use super::primary::ArmSummary;
use super::receipt::{Receipt, RECEIPT_SCHEMA};
use super::render::fixed3;
use super::render_paper_paired::{look_comparisons, render_paired_tex, same_run, LookBlock};
use super::render_paper_tex::{
    display_name, tex_bytes, tex_grouped_u64, tex_opt3, tex_score, tex_score_ci,
};
use super::render_ranked::arm_labels;
use super::rundir::{contract, TASKS};
use dope_arena::ArenaResult;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub(super) const DEFAULT_CANDIDATE: &str = "dope-v3b-auto-v8";
pub(super) const DEFAULT_SECOND_CANDIDATE: &str = "served-auto-v8";
const DEFAULT_METHODS: [&str; 4] = ["arfpy-arf", "sdv-gaussian-copula", "sdv-tvae", "sdv-ctgan"];
const DEFAULT_REFS: [&str; 4] = ["copy_a", "bootstrap_a", "indep_marginals", "label_shuffle"];
const DEFAULT_LABELS: [(&str, &str); 9] = [
    ("dope-v3b-auto-v8", "DOPE v3"),
    ("arfpy-arf", "ARF"),
    ("sdv-gaussian-copula", "SDV Gaussian copula"),
    ("sdv-tvae", "SDV TVAE"),
    ("sdv-ctgan", "SDV CTGAN"),
    ("copy_a", "Copy of fitting rows"),
    ("bootstrap_a", "Bootstrap of fitting rows"),
    ("indep_marginals", "Independent marginals"),
    ("label_shuffle", "Label shuffle"),
];

#[derive(clap::Args, Debug)]
pub struct PaperTablesArgs {
    /// Primary receipt containing DOPE v3, external methods and controls.
    #[arg(long)]
    pub primary: PathBuf,
    /// Compare receipt(s) with candidate = DOPE v3 and baseline = each external method.
    #[arg(long = "compare", required = true)]
    pub compares: Vec<PathBuf>,
    /// Output LaTeX path for Table 1 (ranking, `\label`-free).
    #[arg(long = "ranking-out")]
    pub ranking_out: PathBuf,
    /// Output LaTeX path for Table 2 (paired v3 minus method, `\label`-free).
    #[arg(long = "paired-out")]
    pub paired_out: PathBuf,
    /// Candidate arm id (default: `dope-v3b-auto-v8`).
    #[arg(long, default_value = DEFAULT_CANDIDATE)]
    pub candidate: String,
    /// External method arm ids (repeat; default: the four published generators).
    #[arg(long = "method")]
    pub methods: Vec<String>,
    /// Control arm ids (repeat; default: copy_a, bootstrap_a, indep_marginals, label_shuffle).
    #[arg(long = "reference")]
    pub references: Vec<String>,
    /// Display name override `ARM=Text` (repeat).
    #[arg(long = "label")]
    pub labels: Vec<String>,
    /// Second look's primary receipt (a different set of DEV row splits, e.g. Addendum H part
    /// (a)). Its comparisons become a second block of Table 2 with their own Holm family; Table 1
    /// stays on the report look.
    #[arg(long = "second-primary", requires_all = ["second_compares", "look_titles"])]
    pub second_primary: Option<PathBuf>,
    /// Second look's compare receipt(s): candidate = `--second-candidate`, baseline = each method.
    #[arg(long = "second-compare")]
    pub second_compares: Vec<PathBuf>,
    /// Candidate arm id in the second look's receipts.
    #[arg(long = "second-candidate", default_value = DEFAULT_SECOND_CANDIDATE)]
    pub second_candidate: String,
    /// Table 2 block headings, report look first (exactly two, with `--second-primary`).
    #[arg(long = "look-title")]
    pub look_titles: Vec<String>,
}

#[derive(Serialize)]
struct Summary {
    command: &'static str,
    candidate: String,
    methods: usize,
    references: usize,
    holm_p: BTreeMap<String, f64>,
    ranking_sha256: String,
    paired_sha256: String,
}

fn parse_receipt(label: &str, bytes: &[u8]) -> ArenaResult<Receipt> {
    let r: Receipt = parse_json(label, bytes)?;
    if r.schema != RECEIPT_SCHEMA {
        return contract(format!("{label} is not {RECEIPT_SCHEMA}"));
    }
    Ok(r)
}

fn ranking_row_a(
    arm: &ArmSummary,
    labels: &BTreeMap<String, String>,
    bold: bool,
) -> ArenaResult<String> {
    let p = &arm.primary;
    let (Some(trg), Some(lo), Some(hi)) = (p.trg, p.ci_low, p.ci_high) else {
        return contract(format!("arm {} lacks a primary TRG interval", arm.arm_id));
    };
    let mut primary = tex_score_ci(trg, lo, hi);
    if arm.bytes.scored_above_budget > 0 {
        primary.push_str(r"\textdagger{}");
    }
    let matched = arm.matched.trg.map_or_else(|| "--".to_string(), tex_score);
    let tasks: Vec<String> = TASKS
        .iter()
        .map(|task| {
            p.tasks
                .iter()
                .find(|t| t.task == *task)
                .map_or_else(|| "--".to_string(), |t| tex_score(t.value))
        })
        .collect();
    Ok(format!(
        "{} & {} & {} & {} & {} & {} & {} \\\\",
        display_name(&arm.arm_id, labels, bold),
        primary,
        matched,
        tasks[0],
        tasks[1],
        tasks[2],
        tex_bytes(arm.bytes.median)
    ))
}

fn ranking_row_b(arm: &ArmSummary, labels: &BTreeMap<String, String>, bold: bool) -> String {
    let copies = arm.exact_copy.as_ref().map_or_else(
        || tex_opt3(arm.exact_copy_excess),
        |c| format!("{} ({})", fixed3(c.mean), fixed3(c.max)),
    );
    let cells = format!("{} / {}", arm.primary.failed_cells, arm.primary.cells);
    format!(
        "{} & {} & {} & {} & {} & {} \\\\",
        display_name(&arm.arm_id, labels, bold),
        tex_opt3(arm.realism_coverage),
        tex_opt3(arm.privacy_coverage),
        copies,
        tex_opt3(arm.c2st_median),
        cells
    )
}

fn render_ranking_tex(
    cand: &ArmSummary,
    methods: &[&ArmSummary],
    refs: &[&ArmSummary],
    labels: &BTreeMap<String, String>,
) -> ArenaResult<String> {
    let mult = cand.primary.multiplier;
    let matched_mult = cand.matched.multiplier;
    let fam = |task: &str| {
        cand.primary
            .tasks
            .iter()
            .find(|t| t.task == task)
            .map_or(0, |t| t.families)
    };
    let head_a = format!(
        r"Generator & \makecell{{Transfer Score\\${mult}\times$ [95\% CI]}} & \makecell{{Transfer\\Score ${matched_mult}\times$}} & \makecell{{Regression\\({})}} & \makecell{{Binary\\({})}} & \makecell{{Multiclass\\({})}} & \makecell{{Median\\bytes}} \\",
        fam("regression"),
        fam("binary"),
        fam("multiclass")
    );
    let head_b = r"Generator & \makecell{Realism\\cov.} & \makecell{Privacy\\cov.} & \makecell{Exact-copy\\mean (max)} & C2ST & \makecell{Failed\\/ cells} \\";
    let mut lines = vec![
        "% Generated by dope_kpi render-paper-tables -- DO NOT EDIT BY HAND.".to_string(),
        r"\textbf{(a)}\par\smallskip".to_string(),
        r"\begin{tabular}{@{}lrrrrrr@{}}".to_string(),
        r"\toprule".to_string(),
        head_a,
        r"\midrule".to_string(),
    ];
    lines.push(ranking_row_a(cand, labels, true)?);
    for arm in methods {
        lines.push(ranking_row_a(arm, labels, false)?);
    }
    if !refs.is_empty() {
        lines.push(r"\midrule".to_string());
        for arm in refs {
            lines.push(ranking_row_a(arm, labels, false)?);
        }
    }
    lines.extend([
        r"\bottomrule".to_string(),
        r"\end{tabular}".to_string(),
        r"\par\medskip\textbf{(b)}\par\smallskip".to_string(),
        r"\begin{tabular}{@{}lrrrrr@{}}".to_string(),
        r"\toprule".to_string(),
        head_b.to_string(),
        r"\midrule".to_string(),
    ]);
    lines.push(ranking_row_b(cand, labels, true));
    for arm in methods {
        lines.push(ranking_row_b(arm, labels, false));
    }
    if !refs.is_empty() {
        lines.push(r"\midrule".to_string());
        for arm in refs {
            lines.push(ranking_row_b(arm, labels, false));
        }
    }
    lines.extend([r"\bottomrule".to_string(), r"\end{tabular}".to_string()]);
    let above: Vec<&ArmSummary> = methods
        .iter()
        .copied()
        .filter(|a| a.bytes.scored_above_budget > 0)
        .collect();
    if !above.is_empty() {
        let budget = cand.bytes.budget;
        let all_zero = above.iter().all(|a| a.bytes.within_budget == 0);
        let floor = above[0].trg_at_budget;
        let same_floor = floor.is_some() && above.iter().all(|a| a.trg_at_budget == floor);
        let note = if all_zero && same_floor {
            format!(
                r"\textdagger{{}} Fitted with library defaults and no byte cap. Every observed file exceeds {}~B. Applying the size penalty to those files without refitting gives {} (the failure floor).",
                tex_grouped_u64(budget),
                tex_score(floor.unwrap_or(-1.0))
            )
        } else {
            format!(
                r"\textdagger{{}} Fitted with files exceeding the {}~B budget; Transfer Score reflects the rows as scored.",
                tex_grouped_u64(budget)
            )
        };
        lines.push(r"\par\smallskip".to_string());
        lines.push(format!(r"\parbox{{\linewidth}}{{\raggedright {note}}}"));
    }
    Ok(lines.join("\n") + "\n")
}

fn parse_look(
    look: &str,
    primary_bytes: &[u8],
    compare_bytes: &[Vec<u8>],
) -> ArenaResult<(Receipt, Vec<Receipt>)> {
    let primary = parse_receipt(&format!("{look} primary receipt"), primary_bytes)?;
    let mut receipts: Vec<Receipt> = Vec::new();
    for (i, bytes) in compare_bytes.iter().enumerate() {
        let r = parse_receipt(&format!("{look} compare receipt {}", i + 1), bytes)?;
        if !same_run(&primary, &r) {
            return contract(format!(
                "{look} compare receipt {} differs from its primary in contract, schedule, real reference, panel or bootstrap",
                i + 1
            ));
        }
        receipts.push(r);
    }
    Ok((primary, receipts))
}

/// The second look's inputs: its primary receipt and compare receipts.
pub(super) type SecondLook<'a> = Option<(&'a [u8], &'a [Vec<u8>])>;

pub(super) fn build_tables(
    args: &PaperTablesArgs,
    primary_bytes: &[u8],
    compare_bytes: &[Vec<u8>],
    second: SecondLook<'_>,
) -> ArenaResult<(String, String, BTreeMap<String, f64>)> {
    let (primary, compare_receipts) = parse_look("report-look", primary_bytes, compare_bytes)?;
    let method_ids: Vec<String> = if args.methods.is_empty() {
        DEFAULT_METHODS.iter().map(|s| (*s).to_string()).collect()
    } else {
        args.methods.clone()
    };
    let ref_ids: Vec<String> = if args.references.is_empty() {
        DEFAULT_REFS.iter().map(|s| (*s).to_string()).collect()
    } else {
        args.references.clone()
    };
    let find_arm = |id: &str| -> ArenaResult<&ArmSummary> {
        match primary.arms.iter().find(|a| a.arm_id == id) {
            Some(arm) => Ok(arm),
            None => contract(format!("arm {id} is absent from primary receipt")),
        }
    };
    let cand = find_arm(&args.candidate)?;
    let mut seen = BTreeSet::new();
    let mut methods: Vec<&ArmSummary> = Vec::new();
    for id in &method_ids {
        if !seen.insert(id.as_str()) {
            return contract(format!("--method names {id} twice"));
        }
        methods.push(find_arm(id)?);
    }
    methods.sort_by(|a, b| {
        b.primary
            .trg
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&a.primary.trg.unwrap_or(f64::NEG_INFINITY))
    });
    let mut refs: Vec<&ArmSummary> = Vec::new();
    for id in &ref_ids {
        refs.push(find_arm(id)?);
    }
    let mut labels: BTreeMap<String, String> = DEFAULT_LABELS
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    labels.extend(arm_labels(&args.labels, |id| {
        primary.arms.iter().any(|a| a.arm_id == id)
    })?);
    // Table 2 keeps the report look's method order in every block.
    let order: Vec<&str> = methods.iter().map(|a| a.arm_id.as_str()).collect();
    let matched_comps = look_comparisons(&primary, &compare_receipts, &args.candidate, &order)?;
    let adjusted = holm(
        &matched_comps
            .iter()
            .map(|c| c.p_one_sided)
            .collect::<Vec<f64>>(),
    );
    let holm_map: BTreeMap<String, f64> = methods
        .iter()
        .zip(adjusted.iter())
        .map(|(arm, &p)| (arm.arm_id.clone(), p))
        .collect();
    let ranking = render_ranking_tex(cand, &methods, &refs, &labels)?;
    let second_look = match second {
        Some((p2, c2)) => Some(parse_look("second-look", p2, c2)?),
        None => None,
    };
    let mut blocks = Vec::new();
    match &second_look {
        None => {
            if !args.look_titles.is_empty() {
                return contract("--look-title is only valid with --second-primary".to_string());
            }
            blocks.push(LookBlock {
                title: None,
                comps: matched_comps,
                holm_p: adjusted,
            });
        }
        Some((primary2, receipts2)) => {
            if args.look_titles.len() != 2 {
                return contract(format!(
                    "expected 2 --look-title values, found {}",
                    args.look_titles.len()
                ));
            }
            let Some(cand2) = primary2
                .arms
                .iter()
                .find(|a| a.arm_id == args.second_candidate)
            else {
                return contract(format!(
                    "arm {} is absent from the second-look primary",
                    args.second_candidate
                ));
            };
            if cand2.primary.multiplier != cand.primary.multiplier {
                return contract(
                    "the second look's primary multiplier differs from the report look's"
                        .to_string(),
                );
            }
            let comps2 = look_comparisons(primary2, receipts2, &args.second_candidate, &order)?;
            let holm2 = holm(&comps2.iter().map(|c| c.p_one_sided).collect::<Vec<f64>>());
            blocks.push(LookBlock {
                title: Some(args.look_titles[0].clone()),
                comps: matched_comps,
                holm_p: adjusted,
            });
            blocks.push(LookBlock {
                title: Some(args.look_titles[1].clone()),
                comps: comps2,
                holm_p: holm2,
            });
        }
    }
    let paired = render_paired_tex(&methods, &blocks, &labels)?;
    Ok((ranking, paired, holm_map))
}

pub fn run(args: &PaperTablesArgs) -> ArenaResult<String> {
    let primary = read_file(&args.primary)?;
    let compares = args
        .compares
        .iter()
        .map(|p| read_file(p))
        .collect::<ArenaResult<Vec<_>>>()?;
    let second_primary = args
        .second_primary
        .as_ref()
        .map(|p| read_file(p))
        .transpose()?;
    let second_compares = args
        .second_compares
        .iter()
        .map(|p| read_file(p))
        .collect::<ArenaResult<Vec<_>>>()?;
    let second = second_primary
        .as_deref()
        .map(|p| (p, second_compares.as_slice()));
    let (ranking, paired, holm_p) = build_tables(args, &primary, &compares, second)?;
    write_atomic(&args.ranking_out, ranking.as_bytes())?;
    write_atomic(&args.paired_out, paired.as_bytes())?;
    canonical_json(&Summary {
        command: "render-paper-tables",
        candidate: args.candidate.clone(),
        methods: holm_p.len(),
        references: if args.references.is_empty() {
            DEFAULT_REFS.len()
        } else {
            args.references.len()
        },
        holm_p,
        ranking_sha256: sha256_hex(ranking.as_bytes()),
        paired_sha256: sha256_hex(paired.as_bytes()),
    })
}
