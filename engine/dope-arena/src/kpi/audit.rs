//! Reconstruct receipt points from admitted ledger entries without using the scoreboard reducer.

mod points;

use self::points::{factorial, paired, Bytes, Factorial, Slice};
use super::emitlog;
use super::ext_cell::{file_stem, PACKET_DIR};
use super::ext_scratch::measure_packet;
use super::jsonl::{canonical_json, parse_json, read_file, sha256_hex};
use super::panelsel::PanelKind;
use super::receipt::{Context, Receipt, RunArgs, RECEIPT_SCHEMA};
use super::rundir::{contract, Arm, STATUS_UNSUPPORTED};
use dope_arena::ArenaResult;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(clap::Args, Debug)]
pub struct AuditArgs {
    /// Directory holding the receipt's schedule, real reference and score ledgers.
    #[arg(long)]
    pub run_dir: PathBuf,
    /// Native primary or compare receipt to verify.
    #[arg(long)]
    pub receipt: PathBuf,
    /// The exact list used to partition a DEV or confirmation receipt.
    #[arg(long)]
    pub dev_tables: Option<PathBuf>,
    /// Contract to apply (default: the compiled `contracts/dope-kpi.json`).
    #[arg(long)]
    pub contract: Option<PathBuf>,
    /// Read historical v2 receipts from before the explicit auditor roster.
    #[arg(long)]
    pub legacy_v2: bool,
    /// E0 arms ordered as packet/packet, real/packet, packet/tree and real/tree.
    #[arg(long, num_args = 4, value_names = ["PP", "RP", "PT", "RT"])]
    pub factorial: Option<Vec<String>>,
    /// Retained emit-cli run root for checking measured artifacts against the score ledger.
    #[arg(long)]
    pub artifact_root: Option<PathBuf>,
}

fn check(condition: bool, scope: &'static str) -> ArenaResult<()> {
    if condition {
        Ok(())
    } else {
        contract(format!("KPI receipt audit mismatch in {scope}"))
    }
}

fn same(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
        (None, None) => true,
        _ => false,
    }
}

#[derive(Serialize)]
struct Sensitivity {
    multiplier: u32,
    cells: usize,
    score_of_record: Option<f64>,
    failed_at_floor: Option<f64>,
    unclipped: Option<f64>,
}

#[derive(Serialize)]
struct Attribution<'a> {
    packet_packet: &'a str,
    real_packet: &'a str,
    packet_tree: &'a str,
    real_tree: &'a str,
    readouts: [Factorial<'a>; 2],
}

#[derive(Serialize)]
struct Inventory {
    files: usize,
    unscorable_reference_files: usize,
    max_complete_bytes: u64,
    sha256: String,
}

fn artifact_inventory(root: &Path, arm: &Arm) -> ArenaResult<Option<Inventory>> {
    let log = emitlog::load_arm(root, &arm.id)?;
    let directory = root.join(&arm.id).join(PACKET_DIR);
    let mut files = BTreeMap::new();
    let mut unscorable = BTreeSet::new();
    for row in &arm.rows {
        let key = (row.cell_id.clone(), row.sample_seed, row.multiplier);
        let emitted = log.get(&key);
        // The scorer omits the emitted file when the real reference rules out a cell.
        let omitted = row.packet_bytes == 0
            && row.status == STATUS_UNSUPPORTED
            && row.artifact_class.is_empty()
            && !row.informative
            && row.r_b.is_none();
        check(
            emitted.map_or(0, |e| e.packet_bytes) == row.packet_bytes || omitted,
            "emitted byte count",
        )?;
        let Some(emitted) = emitted.filter(|e| e.packet_bytes > 0) else {
            continue;
        };
        let stem = file_stem(&row.cell_id);
        check(
            Path::new(&stem).file_name().and_then(|s| s.to_str()) == Some(&stem),
            "packet name",
        )?;
        if omitted {
            unscorable.insert(stem.clone());
        }
        let expected = (emitted.packet_bytes, emitted.packet_sha256.as_str());
        if let Some(previous) = files.insert(stem.clone(), expected) {
            check(previous == expected, "artifact identity across draws")?;
        } else {
            let packet = directory.join(format!("{stem}.packet"));
            let card = directory.join(format!("{stem}.card"));
            let actual = measure_packet(&packet, &card)?;
            check(
                (actual.0, actual.1.as_str()) == expected,
                "artifact bytes and digest",
            )?;
        }
    }
    let Some(max_complete_bytes) = files.values().map(|(size, _)| *size).max() else {
        return Ok(None);
    };
    Ok(Some(Inventory {
        files: files.len(),
        unscorable_reference_files: unscorable.len(),
        max_complete_bytes,
        sha256: sha256_hex(canonical_json(&files)?.as_bytes()),
    }))
}

impl From<&Slice<'_>> for Sensitivity {
    fn from(slice: &Slice<'_>) -> Self {
        Self {
            multiplier: slice.multiplier,
            cells: slice.trg.len(),
            score_of_record: slice.points.point(),
            failed_at_floor: slice.floor.point(),
            unclipped: slice.unclipped.point(),
        }
    }
}

pub fn run(args: &AuditArgs) -> ArenaResult<String> {
    let raw = read_file(&args.receipt)?;
    let receipt: Receipt = parse_json("KPI receipt", &raw)?;
    check(
        receipt.schema == RECEIPT_SCHEMA
            && matches!(receipt.command.as_str(), "primary" | "compare")
            && receipt.null.is_none()
            && (receipt.command == "primary") == receipt.comparisons.is_empty(),
        "receipt kind",
    )?;
    let schedule = &receipt.inputs.schedule_file;
    check(
        Path::new(schedule).file_name().and_then(|n| n.to_str()) == Some(schedule.as_str()),
        "schedule name",
    )?;
    let panel = match receipt.panel.id.as_str() {
        "all" => PanelKind::All,
        "dev" => PanelKind::Dev,
        "confirm" => PanelKind::Confirm,
        _ => return contract("KPI receipt audit: unknown panel"),
    };
    let ids: Vec<&str> = receipt
        .inputs
        .rows_sha256
        .keys()
        .map(String::as_str)
        .collect();
    check(
        !ids.is_empty()
            && ids.len() == receipt.arms.len()
            && ids.iter().copied().collect::<BTreeSet<_>>()
                == receipt.arms.iter().map(|a| a.arm_id.as_str()).collect(),
        "arm roster",
    )?;
    let run = RunArgs {
        run_dir: args.run_dir.clone(),
        schedule: Some(args.run_dir.join(schedule)),
        panel,
        dev_tables: args.dev_tables.clone(),
        contract: args.contract.clone(),
        seed: Some(receipt.bootstrap.seed),
    };
    let ctx = Context::open(&run, &ids)?;
    let actual = ctx.receipt(&receipt.command);
    let mut expected = receipt.inputs.clone();
    // The historical v2 writer omitted the auditor roster and counted held-out-only
    // inapplicable controls separately. Keep those receipts replayable.
    let legacy = args.legacy_v2;
    check(
        legacy == expected.realref_auditors.is_empty(),
        "auditor roster",
    )?;
    if legacy {
        expected.realref_auditors = actual.inputs.realref_auditors.clone();
    }
    check(actual.inputs == expected, "input identities")?;
    let mut contract_ref = actual.contract;
    check(
        matches!(receipt.contract.source.as_str(), "embedded" | "file"),
        "contract source",
    )?;
    contract_ref.source.clone_from(&receipt.contract.source);
    check(contract_ref == receipt.contract, "contract identity")?;
    check(actual.panel == receipt.panel, "family partition")?;
    check(
        actual.bootstrap == receipt.bootstrap,
        "bootstrap parameters",
    )?;

    let mut slices = BTreeMap::new();
    let mut arms = BTreeMap::new();
    let mut artifacts = BTreeMap::new();
    for summary in &receipt.arms {
        let arm = ctx.arm(&summary.arm_id);
        if let Some(root) = &args.artifact_root {
            if let Some(inventory) = artifact_inventory(root, arm)? {
                artifacts.insert(arm.id.as_str(), inventory);
            }
        }
        let spec = &ctx.contract.statistic;
        let primary = Slice::read(&ctx, arm, spec.primary_multiplier, legacy);
        let matched = Slice::read(&ctx, arm, spec.matched_multiplier, legacy);
        primary.verify(&summary.primary, legacy)?;
        matched.verify(&summary.matched, legacy)?;
        check(
            same(primary.mlp.point(), summary.mlp_trg),
            "held-out auditor",
        )?;
        Bytes::read(&ctx, arm)?.verify(&summary.bytes, legacy)?;
        if !legacy || summary.trg_failed_at_floor.is_some() {
            check(
                same(primary.floor.point(), summary.trg_failed_at_floor),
                "failure-floor sensitivity",
            )?;
        }
        arms.insert(
            arm.id.as_str(),
            [Sensitivity::from(&primary), Sensitivity::from(&matched)],
        );
        slices.insert(arm.id.as_str(), [primary, matched]);
    }

    let mut comparisons = BTreeMap::new();
    for comparison in &receipt.comparisons {
        let Some(base) = slices.get(comparison.baseline.as_str()) else {
            return contract("KPI receipt audit: missing comparison baseline");
        };
        let members: Option<Vec<_>> = comparison
            .members
            .iter()
            .map(|id| slices.get(id.as_str()))
            .collect();
        let Some(members) = members else {
            return contract("KPI receipt audit: missing comparison member");
        };
        check(
            !members.is_empty()
                && !comparison.members.contains(&comparison.baseline)
                && comparison.members.iter().collect::<BTreeSet<_>>().len() == members.len()
                && match comparison.kind.as_str() {
                    "candidate" => comparison.members == [comparison.id.as_str()],
                    "seed_group" => true,
                    _ => false,
                },
            "comparison roster",
        )?;
        let prime = paired(&base[0], &members.iter().map(|s| &s[0]).collect::<Vec<_>>());
        prime.verify(comparison)?;
        let matched = paired(&base[1], &members.iter().map(|s| &s[1]).collect::<Vec<_>>());
        matched.verify_matched(base[1].multiplier, comparison.matched.as_ref())?;
        let sensitivity = |slice: &Slice<'_>, p: &points::Paired<'_>| Sensitivity {
            multiplier: slice.multiplier,
            cells: p.cells,
            score_of_record: p.delta.point(),
            failed_at_floor: p.floor.point(),
            unclipped: p.unclipped.point(),
        };
        let old = comparisons.insert(
            comparison.id.as_str(),
            [
                sensitivity(&base[0], &prime),
                sensitivity(&base[1], &matched),
            ],
        );
        check(old.is_none(), "comparison identity")?;
    }

    let attribution = args
        .factorial
        .as_ref()
        .map(|ids| {
            check(
                ids.iter().collect::<BTreeSet<_>>().len() == 4,
                "factorial arm roster",
            )?;
            let selected: Option<Vec<_>> = ids.iter().map(|id| slices.get(id.as_str())).collect();
            let Some(selected) = selected else {
                return contract("KPI receipt audit: missing factorial arm");
            };
            let [pp, rp, pt, rt] = selected[..] else {
                return contract("KPI receipt audit: invalid factorial roster");
            };
            let at = |i| factorial(&pp[i], &rp[i], &pt[i], &rt[i]);
            Ok(Attribution {
                packet_packet: &ids[0],
                real_packet: &ids[1],
                packet_tree: &ids[2],
                real_tree: &ids[3],
                readouts: [at(0), at(1)],
            })
        })
        .transpose()?;

    #[derive(Serialize)]
    struct Audit<'a> {
        schema: &'static str,
        receipt_sha256: String,
        entries: BTreeMap<&'a str, usize>,
        arms: BTreeMap<&'a str, [Sensitivity; 2]>,
        comparisons: BTreeMap<&'a str, [Sensitivity; 2]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        factorial: Option<Attribution<'a>>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        artifacts: BTreeMap<&'a str, Inventory>,
    }
    canonical_json(&Audit {
        schema: "dope.kpi-audit/v1",
        receipt_sha256: sha256_hex(&raw),
        entries: ctx
            .arms
            .values()
            .map(|a| (a.id.as_str(), a.rows.len()))
            .collect(),
        arms,
        comparisons,
        factorial: attribution,
        artifacts,
    })
}
