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

use super::hero_model::{Hero, Point, Reference, Theme, Tier, DEFAULT_TITLE};
use super::hero_svg::render;
use super::jsonl::{canonical_json, parse_json, read_file, sha256_hex, write_atomic};
use super::primary::ArmSummary;
use super::receipt::{Receipt, RECEIPT_SCHEMA};
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
    /// Arm drawn in the accent, with its TRG in a bold label.
    #[arg(long)]
    pub highlight: Option<String>,
    /// Arm drawn in the lighter accent (repeat; default: older `dope-X.Y.Z-*` releases).
    #[arg(long = "previous")]
    pub previous: Vec<String>,
    /// Control drawn as a horizontal reference line at its TRG (repeat).
    #[arg(long = "reference")]
    pub references: Vec<String>,
    /// Title (default: "Predictive structure kept vs. artifact size").
    #[arg(long)]
    pub title: Option<String>,
    /// Subtitle; `{cells}`, `{families}` and `{panel}` take the receipt's values.
    #[arg(long)]
    pub note: Option<String>,
}

const DEFAULT_NOTE: &str = "Panel {panel}: {families} families, {cells} cells, scored by dope_kpi. TRG 0 = matches the real B sample under both auditors.";

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

fn labels(args: &HeroArgs, arms: &[ArmSummary]) -> ArenaResult<BTreeMap<String, String>> {
    let mut labels = BTreeMap::new();
    for item in &args.labels {
        let Some((arm, text)) = item.split_once('=') else {
            return contract("--label takes ARM=Text");
        };
        if text.trim().is_empty() {
            return contract(format!("--label {arm} has no text"));
        }
        if !arms.iter().any(|a| a.arm_id == arm) {
            return contract(format!("--label names arm {arm}, which no receipt holds"));
        }
        if labels.insert(arm.to_string(), text.to_string()).is_some() {
            return contract(format!("--label names arm {arm} twice"));
        }
    }
    Ok(labels)
}

/// Every flag names an arm of the receipts once, a highlight or earlier
/// release that can be a point and a reference with a TRG.
fn check_roles(args: &HeroArgs, arms: &[ArmSummary]) -> ArenaResult<()> {
    let find = |id: &str| arms.iter().find(|a| a.arm_id == id);
    let mut seen: Vec<&str> = Vec::new();
    let roles = args.highlight.iter().map(|id| ("--highlight", id));
    let roles = roles
        .chain(args.previous.iter().map(|id| ("--previous", id)))
        .chain(args.references.iter().map(|id| ("--reference", id)));
    for (flag, id) in roles {
        let Some(arm) = find(id) else {
            return contract(format!("{flag} names arm {id}, which no receipt holds"));
        };
        if seen.contains(&id.as_str()) {
            return contract(format!("arm {id} is given more than one role"));
        }
        seen.push(id.as_str());
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
    Ok(())
}

/// The graphic's input from the receipts and flags, and the unplotted arms.
fn hero(args: &HeroArgs, receipts: &[Vec<u8>]) -> ArenaResult<(Hero, Vec<String>)> {
    let (first, shas, arms) = load(receipts)?;
    let labels = labels(args, &arms)?;
    check_roles(args, &arms)?;
    let label = |id: &str| labels.get(id).cloned().unwrap_or_else(|| id.to_string());
    let highlight = args.highlight.as_deref();
    let newest = highlight.and_then(release);
    let tier = |id: &str| {
        if Some(id) == highlight {
            Tier::Highlight
        } else if args.previous.iter().any(|p| p == id)
            || (args.previous.is_empty() && release(id).zip(newest).is_some_and(|(v, h)| v < h))
        {
            Tier::Previous
        } else {
            Tier::Other
        }
    };
    let (mut points, mut unplotted) = (Vec::new(), Vec::new());
    for arm in &arms {
        if args.references.contains(&arm.arm_id) {
            continue;
        }
        match plotted(arm) {
            Some((bytes, trg, ci_low, ci_high)) => points.push(Point {
                label: label(&arm.arm_id),
                bytes,
                trg,
                ci_low,
                ci_high,
                tier: tier(&arm.arm_id),
            }),
            None => unplotted.push(arm.arm_id.clone()),
        }
    }
    let references = args
        .references
        .iter()
        .filter_map(|id| arms.iter().find(|a| a.arm_id == *id))
        .filter_map(|arm| {
            let trg = arm.primary.trg?;
            Some(Reference {
                label: label(&arm.arm_id),
                trg,
            })
        })
        .collect();
    let Some(anchor) = highlight
        .and_then(|id| arms.iter().find(|a| a.arm_id == id))
        .or_else(|| arms.iter().find(|a| plotted(a).is_some()))
    else {
        return contract("render-hero needs an arm with a byte median and a TRG");
    };
    let budget = anchor.bytes.budget;
    if budget == 0 {
        return contract("the receipts hold no byte budget");
    }
    let note = args.note.as_deref().unwrap_or(DEFAULT_NOTE);
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
    let hero = Hero {
        title: args
            .title
            .clone()
            .unwrap_or_else(|| DEFAULT_TITLE.to_string()),
        note: Some(note).filter(|n| !n.trim().is_empty()),
        points,
        references,
        budget: budget as f64,
        footer: format!(
            "contract {} \u{b7} {noun} {}",
            &first.contract.sha256[..8],
            digests.join(", ")
        ),
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
