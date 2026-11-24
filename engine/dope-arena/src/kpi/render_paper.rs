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
use super::compare::ComparisonResult;
use super::jsonl::{canonical_json, parse_json, read_file, sha256_hex, write_atomic};
use super::primary::ArmSummary;
use super::receipt::{Receipt, RECEIPT_SCHEMA};
use super::render::fixed3;
use super::render_ranked::{arm_labels, human_bytes, score1, signed_pts1};
use super::rule::Check;
use super::rundir::{contract, TASKS};
use dope_arena::ArenaResult;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub(super) const DEFAULT_CANDIDATE: &str = "dope-v3b-auto-v8";
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

fn tex_escape(text: &str) -> String {
    text.replace('\\', r"\textbackslash{}")
        .replace('&', r"\&")
        .replace('%', r"\%")
        .replace('_', r"\_")
        .replace('#', r"\#")
}

fn tex_minus(text: &str) -> String {
    if let Some(rest) = text.strip_prefix('-') {
        format!(r"\ensuremath{{-}}{rest}")
    } else {
        text.to_string()
    }
}

fn tex_score(trg: f64) -> String {
    tex_minus(&score1(trg))
}

fn tex_signed_pts(delta: f64) -> String {
    tex_minus(&signed_pts1(delta))
}

fn tex_score_ci(trg: f64, lo: f64, hi: f64) -> String {
    format!("{} [{}, {}]", tex_score(trg), tex_score(lo), tex_score(hi))
}

fn tex_delta_ci(delta: f64, lo: f64, hi: f64) -> String {
    format!(
        "{} [{}, {}]",
        tex_signed_pts(delta),
        tex_signed_pts(lo),
        tex_signed_pts(hi)
    )
}

fn tex_p(p: f64) -> String {
    if p < 0.001 {
        r"\textless{}0.001".to_string()
    } else {
        fixed3(p)
    }
}

fn tex_opt3(x: Option<f64>) -> String {
    x.map_or_else(|| "--".to_string(), |v| tex_minus(&fixed3(v)))
}

fn tex_grouped_u64(n: u64) -> String {
    let raw = n.to_string();
    let mut out = String::new();
    for (i, ch) in raw.chars().enumerate() {
        if i > 0 && (raw.len() - i) % 3 == 0 {
            out.push_str("{,}");
        }
        out.push(ch);
    }
    out
}

fn tex_bytes(bytes: Option<f64>) -> String {
    bytes.map_or_else(
        || "--".to_string(),
        |b| human_bytes(b).replace(',', "{,}").replace(' ', "~"),
    )
}

fn short_guard(name: &str) -> &str {
    match name {
        "task_ci_low" => "tasks",
        "mlp_delta" => "mlp",
        "realism_coverage_delta" => "realism",
        "privacy_coverage_delta" => "privacy",
        "exact_copy_excess" => "copies",
        other => other,
    }
}

fn same_run(a: &Receipt, b: &Receipt) -> bool {
    a.contract.sha256 == b.contract.sha256
        && a.inputs.schedule_sha256 == b.inputs.schedule_sha256
        && a.inputs.realref_sha256 == b.inputs.realref_sha256
        && a.panel == b.panel
        && a.bootstrap == b.bootstrap
}

fn parse_receipt(label: &str, bytes: &[u8]) -> ArenaResult<Receipt> {
    let r: Receipt = parse_json(label, bytes)?;
    if r.schema != RECEIPT_SCHEMA {
        return contract(format!("{label} is not {RECEIPT_SCHEMA}"));
    }
    Ok(r)
}

fn display_name(arm_id: &str, labels: &BTreeMap<String, String>, bold: bool) -> String {
    let text = labels
        .get(arm_id)
        .map(|s| tex_escape(s))
        .unwrap_or_else(|| format!(r"\texttt{{{}}}", tex_escape(arm_id)));
    if bold {
        format!(r"\textbf{{{text}}}")
    } else {
        text
    }
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
                r"\textdagger{{}} Fitted with library defaults and no byte cap; no file fits within {}~B, so at the {}~B budget each scores {} (failure floor).",
                tex_grouped_u64(budget),
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

fn render_paired_tex(
    methods: &[&ArmSummary],
    comps: &[&ComparisonResult],
    holm_p: &[f64],
    labels: &BTreeMap<String, String>,
) -> ArenaResult<String> {
    let mult = methods.first().map_or(4, |a| a.primary.multiplier);
    let matched_mult = comps
        .first()
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
    for ((arm, comp), &p_h) in methods.iter().zip(comps.iter()).zip(holm_p.iter()) {
        let matched = comp.matched.as_ref().map_or_else(
            || "--".to_string(),
            |m| tex_delta_ci(m.delta, m.ci_low, m.ci_high),
        );
        let missed: Vec<&str> = comp
            .decision
            .guards
            .iter()
            .filter(|g| g.check == Check::Fail)
            .map(|g| short_guard(&g.name))
            .collect();
        let guards_text = if missed.is_empty() {
            "none".to_string()
        } else {
            missed.join(", ")
        };
        lines.push(format!(
            "{} & {} & {} & {} & {} & \\textbf{{{}}} & {} \\\\",
            display_name(&arm.arm_id, labels, false),
            tex_delta_ci(comp.delta, comp.ci_low, comp.ci_high),
            matched,
            tex_p(comp.p_one_sided),
            tex_p(p_h),
            comp.decision.outcome.word(),
            guards_text
        ));
    }
    lines.extend([r"\bottomrule".to_string(), r"\end{tabular}".to_string()]);
    Ok(lines.join("\n") + "\n")
}

pub(super) fn build_tables(
    args: &PaperTablesArgs,
    primary_bytes: &[u8],
    compare_bytes: &[Vec<u8>],
) -> ArenaResult<(String, String, BTreeMap<String, f64>)> {
    let primary = parse_receipt("primary receipt", primary_bytes)?;
    let mut all_comps: Vec<ComparisonResult> = Vec::new();
    for (i, bytes) in compare_bytes.iter().enumerate() {
        let r = parse_receipt(&format!("compare receipt {}", i + 1), bytes)?;
        if !same_run(&primary, &r) {
            return contract(format!(
                "compare receipt {} differs from primary in contract, schedule, real reference, panel or bootstrap",
                i + 1
            ));
        }
        all_comps.extend(r.comparisons);
    }
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
    let mut matched_comps: Vec<&ComparisonResult> = Vec::new();
    for arm in &methods {
        let hits: Vec<&ComparisonResult> = all_comps
            .iter()
            .filter(|c| c.id == args.candidate && c.baseline == arm.arm_id)
            .collect();
        if hits.len() != 1 {
            return contract(format!(
                "expected 1 comparison for {} vs {}, found {}",
                args.candidate,
                arm.arm_id,
                hits.len()
            ));
        }
        matched_comps.push(hits[0]);
    }
    let raw_p: Vec<f64> = matched_comps.iter().map(|c| c.p_one_sided).collect();
    let adjusted = holm(&raw_p);
    let holm_map: BTreeMap<String, f64> = methods
        .iter()
        .zip(adjusted.iter())
        .map(|(arm, &p)| (arm.arm_id.clone(), p))
        .collect();
    let ranking = render_ranking_tex(cand, &methods, &refs, &labels)?;
    let paired = render_paired_tex(&methods, &matched_comps, &adjusted, &labels)?;
    Ok((ranking, paired, holm_map))
}

pub fn run(args: &PaperTablesArgs) -> ArenaResult<String> {
    let primary = read_file(&args.primary)?;
    let compares = args
        .compares
        .iter()
        .map(|p| read_file(p))
        .collect::<ArenaResult<Vec<_>>>()?;
    let (ranking, paired, holm_p) = build_tables(args, &primary, &compares)?;
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
