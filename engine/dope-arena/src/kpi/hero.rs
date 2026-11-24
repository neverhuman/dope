//! `dope_kpi render-hero`: the README hook graphic from receipts.
//!
//! Reads `primary` or `compare` receipts of one run (same contract,
//! schedule, real reference, panel and bootstrap; an arm present in several
//! must agree) and draws them with `hero_svg`: every arm with a byte median
//! and a TRG is a point at `bytes.median` (for a third-party arm, the fitted
//! model file), every `--reference` arm a horizontal line at its TRG. An arm
//! that is neither is listed as unplotted in the stdout summary. Arms keep
//! the receipts' order, so the same receipts and flags give the same bytes.
//!
//! `--previous` defaults to the arms named `dope-X.Y.Z-...` with a release
//! below the highlight's. In `--note`, `{cells}`, `{families}` and `{panel}`
//! take the highlight's (else the first point's) statistic and the panel id.

use super::hero_model::{
    Emphasis, Hero, Point, Reference, Theme, YScale, DEFAULT_TITLE, SCORE_TITLE,
};
use super::hero_svg::render;
use super::jsonl::{canonical_json, parse_json, read_file, sha256_hex, write_atomic};
use super::primary::ArmSummary;
use super::receipt::{Receipt, RECEIPT_SCHEMA};
use super::render::arm_labels;
use super::rundir::contract;
use dope_arena::ArenaResult;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum ThemeArg {
    Light,
    Dark,
}

#[derive(clap::Args, Debug)]
pub struct HeroArgs {
    /// Receipt written by `primary` or `compare` (repeat for several of one run).
    #[arg(long = "receipt", required = true)]
    pub receipts: Vec<PathBuf>,
    /// Colour theme (one file per theme; a README picks with `<picture>`).
    #[arg(long, value_enum)]
    pub theme: ThemeArg,
    /// Output SVG.
    #[arg(long)]
    pub out: PathBuf,
    /// Direct label of an arm, `ARM=Text` (repeat; default: the arm id).
    #[arg(long = "label")]
    pub labels: Vec<String>,
    /// Arm drawn in the accent, with its score in a bold label.
    #[arg(long)]
    pub highlight: Option<String>,
    /// Arm drawn in the lighter accent (repeat; default: older `dope-X.Y.Z-*` releases).
    #[arg(long = "previous")]
    pub previous: Vec<String>,
    /// Control drawn as a horizontal reference line (repeat).
    #[arg(long = "reference")]
    pub references: Vec<String>,
    /// Point arm allowlist (repeat; arms outside move to `unplotted`).
    #[arg(long = "include")]
    pub include: Vec<String>,
    /// Plot `Transfer Score = 100 * (1 + TRG)` instead of raw TRG on the y axis.
    #[arg(long = "score-scale")]
    pub score_scale: bool,
    /// Draw a horizontal reference line at the real-twin anchor (TRG 0 / Transfer Score 100).
    #[arg(long = "real-twin")]
    pub real_twin: Option<String>,
    /// Preset for DOPE v3 vs external generators on the Transfer Score scale.
    #[arg(long)]
    pub methods: bool,
    /// Title (default: "Predictive structure kept vs. artifact size").
    #[arg(long)]
    pub title: Option<String>,
    /// Subtitle; `{cells}`, `{families}` and `{panel}` take the receipt's values.
    #[arg(long)]
    pub note: Option<String>,
}

const DEFAULT_NOTE: &str = "Panel {panel}: {families} families, {cells} cells, scored by dope_kpi. TRG 0 = matches the real B sample under both auditors.";
const SCORE_NOTE: &str = "Panel {panel}: {families} families, {cells} cells, scored by dope_kpi. Transfer Score = 100 × (1 + TRG); 100 matches a second real sample.";
const METHODS_NOTE: &str = "DEV panel ({families} families, {cells} cells, 4×). Transfer Score = 100 × (1 + TRG); 100 = real twin. External methods have no byte cap. DOPE is the only generator inside the 2 KiB budget.";
const METHODS_ARMS: [&str; 5] = [
    "dope-v3b-auto-v8",
    "arfpy-arf",
    "sdv-gaussian-copula",
    "sdv-tvae",
    "sdv-ctgan",
];
const METHODS_LABELS: [(&str, &str); 6] = [
    ("dope-v3b-auto-v8", "DOPE v3"),
    ("arfpy-arf", "ARF"),
    ("sdv-gaussian-copula", "Gaussian copula"),
    ("sdv-tvae", "TVAE"),
    ("sdv-ctgan", "CTGAN"),
    ("copy_a", "copy of the fitting rows"),
];

#[derive(Serialize)]
struct Summary {
    command: &'static str,
    theme: &'static str,
    points: usize,
    references: usize,
    unplotted: Vec<String>,
    svg_sha256: String,
}

/// `(1, 3, 0)` for `dope-1.3.0-small`.
fn release(arm_id: &str) -> Option<(u64, u64, u64)> {
    let version = arm_id.strip_prefix("dope-")?.split('-').next()?;
    let parts: Vec<u64> = version
        .split('.')
        .map(|p| p.parse().ok())
        .collect::<Option<_>>()?;
    match parts[..] {
        [major, minor, patch] => Some((major, minor, patch)),
        _ => None,
    }
}

/// The receipts' digests and their arms, merged by id in receipt order.
fn load(receipts: &[Vec<u8>]) -> ArenaResult<(Receipt, Vec<String>, Vec<ArmSummary>)> {
    let mut parsed: Vec<Receipt> = Vec::new();
    for (index, bytes) in receipts.iter().enumerate() {
        let receipt: Receipt = parse_json(&format!("receipt {}", index + 1), bytes)?;
        if receipt.schema != RECEIPT_SCHEMA {
            return contract(format!("receipt {} is not {RECEIPT_SCHEMA}", index + 1));
        }
        parsed.push(receipt);
    }
    let Some(first) = parsed.first() else {
        return contract("render-hero needs a receipt");
    };
    let same_run = |r: &Receipt| {
        r.contract.sha256 == first.contract.sha256
            && r.inputs.schedule_sha256 == first.inputs.schedule_sha256
            && r.inputs.realref_sha256 == first.inputs.realref_sha256
            && r.panel == first.panel
            && r.bootstrap == first.bootstrap
    };
    if !parsed.iter().all(same_run) {
        return contract(
            "receipts differ in contract, schedule, real reference, panel or bootstrap",
        );
    }
    let mut arms: Vec<ArmSummary> = Vec::new();
    for arm in parsed.iter().flat_map(|r| &r.arms) {
        match arms.iter().find(|a| a.arm_id == arm.arm_id) {
            Some(known) if known != arm => {
                return contract(format!("receipts disagree on arm {}", arm.arm_id));
            }
            Some(_) => {}
            None => arms.push(arm.clone()),
        }
    }
    let shas = receipts.iter().map(|bytes| sha256_hex(bytes)).collect();
    Ok((parsed.swap_remove(0), shas, arms))
}

/// Byte median and TRG with its interval, when the arm can be a point.
fn plotted(arm: &ArmSummary) -> Option<(f64, f64, f64, f64)> {
    let p = &arm.primary;
    let bytes = arm.bytes.median.filter(|b| *b > 0.0)?;
    Some((bytes, p.trg?, p.ci_low?, p.ci_high?))
}

/// Every flag names an arm of the receipts once, a highlight or earlier
/// release that can be a point and a reference with a TRG.
fn check_roles(
    highlight: Option<&str>,
    previous: &[String],
    references: &[String],
    include: &[String],
    arms: &[ArmSummary],
) -> ArenaResult<()> {
    let find = |id: &str| arms.iter().find(|a| a.arm_id == id);
    let mut seen: Vec<&str> = Vec::new();
    let roles = highlight.iter().map(|id| ("--highlight", *id));
    let roles = roles
        .chain(previous.iter().map(|id| ("--previous", id.as_str())))
        .chain(references.iter().map(|id| ("--reference", id.as_str())));
    for (flag, id) in roles {
        let Some(arm) = find(id) else {
            return contract(format!("{flag} names arm {id}, which no receipt holds"));
        };
        if seen.contains(&id) {
            return contract(format!("arm {id} is given more than one role"));
        }
        seen.push(id);
        let fits = if flag == "--reference" {
            arm.primary.trg.is_some()
        } else {
            plotted(arm).is_some()
        };
        if !fits {
            let need = if flag == "--reference" {
                "a TRG"
            } else {
                "a byte median and a TRG"
            };
            return contract(format!("{flag} arm {id} needs {need}"));
        }
    }
    let mut seen_inc: Vec<&str> = Vec::new();
    for id in include {
        let Some(arm) = find(id) else {
            return contract(format!("--include names arm {id}, which no receipt holds"));
        };
        if seen_inc.contains(&id.as_str()) {
            return contract(format!("--include names arm {id} twice"));
        }
        seen_inc.push(id.as_str());
        if references.iter().any(|r| r == id) {
            return contract(format!("arm {id} is given more than one role"));
        }
        if plotted(arm).is_none() {
            return contract(format!("--include arm {id} needs a byte median and a TRG"));
        }
    }
    if !include.is_empty() {
        if let Some(hl) = highlight {
            if !include.iter().any(|id| id == hl) {
                return contract(format!(
                    "--highlight arm {hl} must be included by --include"
                ));
            }
        }
        for prev in previous {
            if !include.iter().any(|id| id == prev) {
                return contract(format!(
                    "--previous arm {prev} must be included by --include"
                ));
            }
        }
    }
    Ok(())
}

/// The graphic's input from the receipts and flags, and the unplotted arms.
pub(super) fn hero(args: &HeroArgs, receipts: &[Vec<u8>]) -> ArenaResult<(Hero, Vec<String>)> {
    let (first, shas, arms) = load(receipts)?;
    let y_scale = if args.score_scale || args.methods {
        YScale::TransferScore
    } else {
        YScale::Trg
    };
    let include: Vec<String> = if args.methods && args.include.is_empty() {
        METHODS_ARMS.iter().map(|s| (*s).to_string()).collect()
    } else {
        args.include.clone()
    };
    let references_ids: Vec<String> = if args.methods && args.references.is_empty() {
        vec!["copy_a".to_string()]
    } else {
        args.references.clone()
    };
    let highlight_owned = args
        .highlight
        .clone()
        .or_else(|| args.methods.then(|| METHODS_ARMS[0].to_string()));
    let highlight = highlight_owned.as_deref();
    let real_twin = args
        .real_twin
        .clone()
        .or_else(|| args.methods.then(|| "real twin".to_string()));
    let mut labels: BTreeMap<String, String> = if args.methods {
        METHODS_LABELS
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    } else {
        BTreeMap::new()
    };
    labels.extend(arm_labels(&args.labels, |id| {
        arms.iter().any(|arm| arm.arm_id == id)
    })?);
    check_roles(highlight, &args.previous, &references_ids, &include, &arms)?;
    let label = |id: &str| labels.get(id).cloned().unwrap_or_else(|| id.to_string());
    let newest = highlight.and_then(release);
    let tier = |id: &str| {
        if Some(id) == highlight {
            Emphasis::Highlight
        } else if args.previous.iter().any(|p| p == id)
            || (args.previous.is_empty() && release(id).zip(newest).is_some_and(|(v, h)| v < h))
        {
            Emphasis::Previous
        } else {
            Emphasis::Other
        }
    };
    let (mut points, mut unplotted) = (Vec::new(), Vec::new());
    let mut plotted_arms: Vec<&ArmSummary> = Vec::new();
    for arm in &arms {
        if references_ids.contains(&arm.arm_id) {
            continue;
        }
        if !include.is_empty() && !include.contains(&arm.arm_id) {
            unplotted.push(arm.arm_id.clone());
            continue;
        }
        match plotted(arm) {
            Some((bytes, trg, ci_low, ci_high)) => {
                plotted_arms.push(arm);
                points.push(Point {
                    label: label(&arm.arm_id),
                    bytes,
                    trg,
                    ci_low,
                    ci_high,
                    tier: tier(&arm.arm_id),
                });
            }
            None => unplotted.push(arm.arm_id.clone()),
        }
    }
    let mut references = Vec::new();
    if let Some(twin_label) = real_twin {
        if !twin_label.trim().is_empty() {
            references.push(Reference {
                label: twin_label,
                trg: 0.0,
            });
        }
    }
    references.extend(
        references_ids
            .iter()
            .filter_map(|id| arms.iter().find(|a| a.arm_id == *id))
            .filter_map(|arm| {
                let trg = arm.primary.trg?;
                Some(Reference {
                    label: label(&arm.arm_id),
                    trg,
                })
            }),
    );
    let Some(anchor) = highlight
        .and_then(|id| plotted_arms.iter().copied().find(|a| a.arm_id == id))
        .or_else(|| plotted_arms.first().copied())
    else {
        return contract("render-hero needs an arm with a byte median and a TRG");
    };
    let budget = anchor.bytes.budget;
    if budget == 0 {
        return contract("the receipts hold no byte budget");
    }
    let default_note = if args.methods {
        METHODS_NOTE
    } else if y_scale == YScale::TransferScore {
        SCORE_NOTE
    } else {
        DEFAULT_NOTE
    };
    let note = args.note.as_deref().unwrap_or(default_note);
    let note = note
        .replace("{cells}", &anchor.primary.cells.to_string())
        .replace("{families}", &anchor.primary.families.to_string())
        .replace("{panel}", &first.panel.id);
    let digests: Vec<&str> = shas.iter().map(|s| &s[..8]).collect();
    let noun = if digests.len() == 1 {
        "receipt"
    } else {
        "receipts"
    };
    let default_title = match y_scale {
        YScale::Trg => DEFAULT_TITLE,
        YScale::TransferScore => SCORE_TITLE,
    };
    let hero = Hero {
        title: args
            .title
            .clone()
            .unwrap_or_else(|| default_title.to_string()),
        note: Some(note).filter(|n| !n.trim().is_empty()),
        points,
        references,
        budget: budget as f64,
        footer: format!(
            "contract {} \u{b7} {noun} {}",
            &first.contract.sha256[..8],
            digests.join(", ")
        ),
        y_scale,
    };
    Ok((hero, unplotted))
}

pub fn run(args: &HeroArgs) -> ArenaResult<String> {
    let receipts = args
        .receipts
        .iter()
        .map(|path| read_file(path))
        .collect::<ArenaResult<Vec<_>>>()?;
    let (hero, unplotted) = hero(args, &receipts)?;
    let theme = match args.theme {
        ThemeArg::Light => Theme::Light,
        ThemeArg::Dark => Theme::Dark,
    };
    let svg = render(&hero, theme);
    write_atomic(&args.out, svg.as_bytes())?;
    canonical_json(&Summary {
        command: "render-hero",
        theme: theme.name(),
        points: hero.points.len(),
        references: hero.references.len(),
        unplotted,
        svg_sha256: sha256_hex(svg.as_bytes()),
    })
}
