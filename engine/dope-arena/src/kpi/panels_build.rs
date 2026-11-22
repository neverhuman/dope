//! `dope_kpi build-panels`: lineage families and the baseline panel on the current split.
//!
//! The panel holds every `val` table whose lineage family contains no `train`
//! table. Outputs, all under `--out`: `families.jsonl`, `<panel_id>.panel.json`
//! (the existing `PanelManifest`, protocol `dope-fidelity/v3`, role development,
//! committed by its canonical SHA-256) and `overlap_report.json`.

use super::families::{
    build_families, is_generic, train_families, verify_panel_families, FamilyRow, EVIDENCE_CONTENT,
    EVIDENCE_NAME, GENERIC_STEMS,
};
use super::io::{Arena, SPLIT_TRAIN, SPLIT_VAL};
use super::jsonl::{canonical_json, canonical_pretty, sha256_hex, write_atomic, write_jsonl};
use super::sketch::{
    ContentStats, ShareBasis, BUCKET_CAP, LSH_KEYS, MIN_DISTINCT, MIN_MATCHED_COLUMNS,
    NEIGHBOR_WINDOW, SKETCH_K,
};
use dope_arena::panel::{PanelDatasetEntry, PanelManifest, PanelRole};
use dope_arena::{ArenaError, ArenaResult};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const PANEL_PROTOCOL: &str = "dope-fidelity/v3";
pub const TARGET_COLUMN: &str = "__target__";
pub const OVERLAP_SCHEMA: &str = "dope.kpi-overlap/v1";

#[derive(clap::Args, Debug)]
pub struct BuildPanelsArgs {
    /// Flat corpus arena directory (`dope.flat-corpus/v1`).
    #[arg(long)]
    pub arena: PathBuf,
    /// Output directory for the families, panel manifest and overlap report.
    #[arg(long)]
    pub out: PathBuf,
    /// Identifier of the panel built from the current validation split.
    #[arg(long)]
    pub panel_id: String,
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// Recompute the SHA-256 of X.f32, y.f32 and dup.i32 before reading them.
    #[arg(long)]
    pub verify_checksums: bool,
    /// Content links need matched columns covering half of the eligible columns
    /// of the smaller table, or of each table (stricter, fewer chance links).
    #[arg(long, value_enum, default_value_t = ShareBasis::Smaller)]
    pub content_share: ShareBasis,
}

type Counts = BTreeMap<String, usize>;

/// Table and family counts, in total and per task. A family counts once under
/// every task it has among the counted tables.
#[derive(Debug, Default, Serialize)]
struct Tally {
    tables: usize,
    families: usize,
    tables_by_task: Counts,
    families_by_task: Counts,
}

impl Tally {
    fn of<'a>(rows: impl Iterator<Item = &'a FamilyRow>) -> Self {
        let mut tally = Self::default();
        let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
        let mut families: BTreeSet<&str> = BTreeSet::new();
        for row in rows {
            tally.tables += 1;
            *tally.tables_by_task.entry(row.task.clone()).or_default() += 1;
            families.insert(row.family_id.as_str());
            if seen.insert((row.task.as_str(), row.family_id.as_str())) {
                *tally.families_by_task.entry(row.task.clone()).or_default() += 1;
            }
        }
        tally.families = families.len();
        tally
    }
}

/// Validation tables left out because their family has a training table.
/// Evidence is the kind of link across the split: `name` when a training
/// table of the family shares the canonical stem, otherwise `content`.
#[derive(Debug, Serialize)]
struct Excluded {
    #[serde(flatten)]
    tally: Tally,
    tables_by_evidence: Counts,
    families_by_evidence: Counts,
    dataset_hashes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Rule {
    generic_stems: Vec<&'static str>,
    sketch_k: usize,
    eligible_distinct_values_above: usize,
    column_match: &'static str,
    min_matched_columns: usize,
    content_share: &'static str,
    table_link: &'static str,
    index_keys: usize,
    bucket_cap: usize,
    neighbor_window: usize,
}

#[derive(Debug, Serialize)]
struct ArenaIdentity {
    schema: String,
    source_manifest_sha256: String,
    index_sha256: String,
    tables: usize,
    checksums_verified: bool,
}

#[derive(Debug, Serialize)]
struct OverlapReport {
    schema: &'static str,
    panel_id: String,
    panel_sha256: String,
    arena: ArenaIdentity,
    rule: Rule,
    tables_by_split: Counts,
    tables_by_family_evidence: Counts,
    families: usize,
    validation: Tally,
    excluded: Excluded,
    panel: Tally,
    content: ContentStats,
}

#[derive(Debug, Serialize)]
struct Summary {
    command: &'static str,
    panel_id: String,
    panel_sha256: String,
    panel_file: String,
    panel_tables: usize,
    panel_families: usize,
    excluded_tables: usize,
    excluded_families: usize,
    families: usize,
    families_sha256: String,
    overlap_report_sha256: String,
}

fn rule(basis: ShareBasis) -> Rule {
    Rule {
        generic_stems: GENERIC_STEMS.to_vec(),
        sketch_k: SKETCH_K,
        eligible_distinct_values_above: MIN_DISTINCT,
        column_match: "bottom-k Jaccard estimate >= 1/2 on values rounded to 6 significant digits",
        min_matched_columns: MIN_MATCHED_COLUMNS,
        content_share: basis.name(),
        table_link: match basis {
            ShareBasis::Smaller => {
                "matched columns >= 1/2 of the eligible columns of the table with fewer"
            }
            ShareBasis::Each => "matched columns >= 1/2 of the eligible columns of each table",
        },
        index_keys: LSH_KEYS,
        bucket_cap: BUCKET_CAP,
        neighbor_window: NEIGHBOR_WINDOW,
    }
}

fn valid_panel_id(panel_id: &str) -> bool {
    !panel_id.is_empty()
        && panel_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn count_by<'a>(keys: impl Iterator<Item = &'a str>) -> Counts {
    let mut counts = Counts::new();
    for key in keys {
        *counts.entry(key.to_string()).or_default() += 1;
    }
    counts
}

fn excluded_report(families: &[FamilyRow], excluded: &[&FamilyRow]) -> Excluded {
    let mut train_stems: BTreeSet<(&str, &str)> = BTreeSet::new();
    for row in families.iter().filter(|r| r.split == SPLIT_TRAIN) {
        train_stems.insert((row.family_id.as_str(), row.canon_stem.as_str()));
    }
    let evidence = |row: &FamilyRow| {
        let named = !is_generic(&row.canon_stem)
            && train_stems.contains(&(row.family_id.as_str(), row.canon_stem.as_str()));
        if named {
            EVIDENCE_NAME
        } else {
            EVIDENCE_CONTENT
        }
    };
    let mut family_evidence: BTreeMap<&str, &str> = BTreeMap::new();
    for row in excluded {
        let entry = family_evidence
            .entry(row.family_id.as_str())
            .or_insert(EVIDENCE_CONTENT);
        if evidence(row) == EVIDENCE_NAME {
            *entry = EVIDENCE_NAME;
        }
    }
    let mut dataset_hashes: Vec<String> = excluded.iter().map(|r| r.dataset_hash.clone()).collect();
    dataset_hashes.sort_unstable();
    Excluded {
        tally: Tally::of(excluded.iter().copied()),
        tables_by_evidence: count_by(excluded.iter().map(|r| evidence(r))),
        families_by_evidence: count_by(family_evidence.values().copied()),
        dataset_hashes,
    }
}

/// The panel manifest over `kept` validation tables, committed by its canonical SHA-256.
fn panel_manifest(
    arena: &Arena,
    panel_id: &str,
    kept: &[&FamilyRow],
) -> ArenaResult<PanelManifest> {
    let mut datasets = Vec::with_capacity(kept.len());
    for row in kept {
        let table_id = arena
            .table_id_of(&row.dataset_hash)
            .ok_or_else(|| ArenaError::Contract("family row without an arena table".into()))?;
        let table = arena.table(table_id)?;
        datasets.push(PanelDatasetEntry {
            dataset_id: table.dataset_hash.clone(),
            family_id: row.family_id.clone(),
            target_column: TARGET_COLUMN.to_string(),
            task: table.task.clone(),
            group_column: None,
            time_column: None,
            source_sha256: table.canonical_sha256.clone(),
            rows: table.n_rows,
            features: table.n_cols,
        });
    }
    datasets.sort_by(|a, b| {
        a.family_id
            .cmp(&b.family_id)
            .then(a.dataset_id.cmp(&b.dataset_id))
    });
    let mut manifest = PanelManifest {
        protocol: PANEL_PROTOCOL.to_string(),
        panel_id: panel_id.to_string(),
        role: PanelRole::Development,
        datasets,
        manifest_sha256: None,
    };
    manifest.manifest_sha256 = Some(manifest.canonical_sha256()?);
    manifest.verify()?;
    Ok(manifest)
}

pub fn run(args: &BuildPanelsArgs) -> ArenaResult<String> {
    if !valid_panel_id(&args.panel_id) {
        return Err(ArenaError::Contract(
            "panel id must be non-empty and use only letters, digits, '.', '_' and '-'".into(),
        ));
    }
    let arena = Arena::open(&args.arena)?;
    if args.verify_checksums {
        arena.verify_checksums()?;
    }
    let (families, content) = build_families(&arena, args.content_share, args.threads)?;
    let with_train = train_families(&families);
    let validation: Vec<&FamilyRow> = families.iter().filter(|r| r.split == SPLIT_VAL).collect();
    let (excluded, kept): (Vec<&FamilyRow>, Vec<&FamilyRow>) = validation
        .iter()
        .copied()
        .partition(|r| with_train.contains(r.family_id.as_str()));
    if kept.is_empty() {
        return Err(ArenaError::Contract(
            "no validation table is free of training siblings; the panel would be empty".into(),
        ));
    }
    let manifest = panel_manifest(&arena, &args.panel_id, &kept)?;
    verify_panel_families(&families, &manifest)?;
    let panel_sha256 = manifest.manifest_sha256.clone().unwrap_or_default();

    let families_sha256 = write_jsonl(&args.out.join("families.jsonl"), &families)?;
    let panel_file = format!("{}.panel.json", args.panel_id);
    write_atomic(
        &args.out.join(&panel_file),
        canonical_pretty(&manifest)?.as_bytes(),
    )?;
    let report = OverlapReport {
        schema: OVERLAP_SCHEMA,
        panel_id: args.panel_id.clone(),
        panel_sha256: panel_sha256.clone(),
        arena: ArenaIdentity {
            schema: arena.meta().schema.clone(),
            source_manifest_sha256: arena.meta().source_manifest_sha256.clone(),
            index_sha256: arena.meta().index_sha256.clone(),
            tables: arena.tables().len(),
            checksums_verified: args.verify_checksums,
        },
        rule: rule(args.content_share),
        tables_by_split: count_by(families.iter().map(|r| r.split.as_str())),
        tables_by_family_evidence: count_by(families.iter().map(|r| r.evidence.as_str())),
        families: families
            .iter()
            .map(|r| r.family_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        validation: Tally::of(validation.iter().copied()),
        excluded: excluded_report(&families, &excluded),
        panel: Tally::of(kept.iter().copied()),
        content,
    };
    let report_text = canonical_pretty(&report)?;
    write_atomic(
        &args.out.join("overlap_report.json"),
        report_text.as_bytes(),
    )?;
    canonical_json(&Summary {
        command: "build-panels",
        panel_id: args.panel_id.clone(),
        panel_sha256,
        panel_file,
        panel_tables: report.panel.tables,
        panel_families: report.panel.families,
        excluded_tables: report.excluded.tally.tables,
        excluded_families: report.excluded.tally.families,
        families: report.families,
        families_sha256,
        overlap_report_sha256: sha256_hex(report_text.as_bytes()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(hash: &str, family: &str, stem: &str, split: &str, task: &str) -> FamilyRow {
        FamilyRow {
            dataset_hash: hash.into(),
            family_id: family.into(),
            canon_stem: stem.into(),
            evidence: "name".into(),
            split: split.into(),
            task: task.into(),
        }
    }

    #[test]
    fn panel_ids_are_safe_file_and_cell_name_parts() {
        for good in ["dev-v250", "sealed.2", "A_b-3"] {
            assert!(valid_panel_id(good), "{good}");
        }
        for bad in ["", "a:b", "a/b", "a b", "../x", "caf\u{e9}"] {
            assert!(!valid_panel_id(bad), "{bad}");
        }
    }

    #[test]
    fn tallies_count_families_once_per_task() {
        let rows = [
            row("a", "f1", "s", "val", "regression"),
            row("b", "f1", "s", "val", "binary"),
            row("c", "f1", "s", "val", "binary"),
            row("d", "f2", "t", "val", "binary"),
        ];
        let tally = Tally::of(rows.iter());
        assert_eq!((tally.tables, tally.families), (4, 2));
        assert_eq!(tally.tables_by_task["binary"], 3);
        assert_eq!(tally.families_by_task["binary"], 2);
        assert_eq!(tally.families_by_task["regression"], 1);
        assert_eq!(Tally::of(rows[..0].iter()).tables, 0);
    }

    #[test]
    fn exclusion_evidence_is_the_link_across_the_split() {
        let families = vec![
            row("t1", "f1", "x", "train", "regression"),
            row("v1", "f1", "x", "val", "regression"),
            row("v2", "f1", "renamed", "val", "binary"),
            row("t2", "f2", "rows", "train", "binary"),
            row("v3", "f2", "rows", "val", "binary"),
            row("v4", "f3", "clean", "val", "multiclass"),
        ];
        let excluded: Vec<&FamilyRow> = vec![&families[4], &families[2], &families[1]];
        let report = excluded_report(&families, &excluded);
        assert_eq!((report.tally.tables, report.tally.families), (3, 2));
        assert_eq!(report.tables_by_evidence["name"], 1);
        // `renamed` shares no stem with training; `rows` is generic even though it repeats.
        assert_eq!(report.tables_by_evidence["content"], 2);
        assert_eq!(report.families_by_evidence["name"], 1);
        assert_eq!(report.families_by_evidence["content"], 1);
        assert_eq!(report.dataset_hashes, ["v1", "v2", "v3"]);
        let text = canonical_json(&report).unwrap();
        assert!(
            text.contains("\"tables\":3") && text.contains("\"tables_by_task\""),
            "{text}"
        );
    }
}
