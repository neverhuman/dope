//! `dope_kpi plan`: the frozen cell schedule `dope.kpi-schedule/v1` (formats.md section 4).
//!
//! One cell per (table, view, data seed), sorted by `cell_id`. Row roles and
//! column views depend on the panel, the table and the data seed only, never
//! on a sample seed or an arm, so every arm is scored on identical cells.
//! A table too small for 16 rows in each of A, B and T still gets its cells,
//! marked `status: "unsupported"` with empty row lists, so denominators stay
//! complete (`status` extends the field list of formats.md; absent means `ok`).

use super::families::{verify_panel_families, FamilyRow};
use super::io::Arena;
use super::jsonl::{canonical_json, read_jsonl, write_jsonl};
use super::par::par_map;
use super::roles::{split_roles, view_columns, VIEW_COLUMNS};
use super::seeds::{structural_seed, PURPOSE_COLS, PURPOSE_SPLIT};
use dope_arena::panel::{PanelDatasetEntry, PanelManifest};
use dope_arena::{ArenaError, ArenaResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const VIEW_FULL: &str = "full";
pub const VIEW_P48: &str = "p48";
pub const STRATUM_HEADLINE: &str = "headline";
pub const STRATUM_SMALL_N: &str = "small_n";
pub const STATUS_OK: &str = "ok";
pub const STATUS_UNSUPPORTED: &str = "unsupported";
/// Synthetic row counts to emit, as multiples of `len(a_idx)`.
pub const MULTIPLIERS: [u32; 2] = [4, 1];
const TARGET_COLUMN: &str = "__target__";

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Tier {
    Lite,
    Full,
    Confirm,
    Sealed,
}

impl Tier {
    /// `(data seeds, sample seeds)` of the tier.
    pub fn seed_counts(self) -> (u64, u64) {
        match self {
            Tier::Lite => (1, 1),
            Tier::Full => (2, 2),
            Tier::Confirm => (3, 2),
            Tier::Sealed => (3, 3),
        }
    }

    /// The lite tier scores one table per family; the others score every table.
    pub fn one_table_per_family(self) -> bool {
        self == Tier::Lite
    }

    fn name(self) -> &'static str {
        match self {
            Tier::Lite => "lite",
            Tier::Full => "full",
            Tier::Confirm => "confirm",
            Tier::Sealed => "sealed",
        }
    }
}

#[derive(clap::Args, Debug)]
pub struct PlanArgs {
    /// Flat corpus arena directory (`dope.flat-corpus/v1`).
    #[arg(long)]
    pub arena: PathBuf,
    /// Panel manifest written by `build-panels`.
    #[arg(long)]
    pub panel: PathBuf,
    /// `families.jsonl` written by `build-panels`.
    #[arg(long)]
    pub families: PathBuf,
    #[arg(long, value_enum)]
    pub tier: Tier,
    /// Output `schedule.jsonl`.
    #[arg(long)]
    pub out: PathBuf,
    /// Override the tier's data seeds (comma separated).
    #[arg(long, value_delimiter = ',')]
    pub data_seeds: Option<Vec<u64>>,
    /// Override the tier's sample seeds (comma separated); never changes rows or columns.
    #[arg(long, value_delimiter = ',')]
    pub sample_seeds: Option<Vec<u64>>,
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// Recompute the SHA-256 of X.f32, y.f32 and dup.i32 before planning.
    #[arg(long)]
    pub verify_checksums: bool,
}

fn status_ok() -> String {
    STATUS_OK.to_string()
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct Cell {
    pub cell_id: String,
    pub panel_sha256: String,
    pub dataset_hash: String,
    pub table_id: usize,
    pub family_id: String,
    pub task: String,
    pub n_classes: u32,
    pub view: String,
    pub cols: Vec<usize>,
    pub data_seed: u64,
    pub stratum: String,
    pub a_idx: Vec<usize>,
    pub b_idx: Vec<usize>,
    pub c_idx: Vec<usize>,
    pub t_idx: Vec<usize>,
    pub sample_seeds: Vec<u64>,
    pub multipliers: Vec<u32>,
    #[serde(default = "status_ok")]
    pub status: String,
}

#[derive(Debug, Serialize)]
struct Summary {
    command: &'static str,
    panel_id: String,
    panel_sha256: String,
    tier: &'static str,
    data_seeds: Vec<u64>,
    sample_seeds: Vec<u64>,
    multipliers: Vec<u32>,
    tables: usize,
    families: usize,
    cells: usize,
    supported: usize,
    unsupported: usize,
    supported_headline: usize,
    supported_small_n: usize,
    cells_by_view: BTreeMap<String, usize>,
    schedule_sha256: String,
}

fn contract<T>(message: String) -> ArenaResult<T> {
    Err(ArenaError::Contract(message))
}

/// The tier's seeds `0..count`, or a non-empty duplicate-free override.
fn seed_list(name: &str, count: u64, given: &Option<Vec<u64>>) -> ArenaResult<Vec<u64>> {
    let Some(seeds) = given else {
        return Ok((0..count).collect());
    };
    let distinct: BTreeSet<&u64> = seeds.iter().collect();
    if seeds.is_empty() || distinct.len() != seeds.len() {
        return contract(format!("--{name} must list distinct seeds"));
    }
    Ok(seeds.clone())
}

/// Tables to schedule: all of them, or per family the one with the most rows
/// (ties broken by the smaller dataset hash).
pub fn select_tables(panel: &PanelManifest, tier: Tier) -> Vec<&PanelDatasetEntry> {
    if !tier.one_table_per_family() {
        return panel.datasets.iter().collect();
    }
    let mut best: BTreeMap<&str, &PanelDatasetEntry> = BTreeMap::new();
    for entry in &panel.datasets {
        let better = best.get(entry.family_id.as_str()).is_none_or(|held| {
            entry.rows > held.rows
                || (entry.rows == held.rows && entry.dataset_id < held.dataset_id)
        });
        if better {
            best.insert(&entry.family_id, entry);
        }
    }
    best.into_values().collect()
}

/// The panel and the families file must describe exactly this arena.
fn check_inputs(arena: &Arena, panel: &PanelManifest, families: &[FamilyRow]) -> ArenaResult<()> {
    if families.len() != arena.tables().len() {
        return contract("families file does not cover the arena".into());
    }
    for row in families {
        let table = arena
            .table_id_of(&row.dataset_hash)
            .map(|id| &arena.tables()[id]);
        if !table.is_some_and(|t| t.split == row.split && t.task == row.task) {
            return contract(format!(
                "families row {} disagrees with the arena",
                row.dataset_hash
            ));
        }
    }
    for entry in &panel.datasets {
        let table = arena
            .table_id_of(&entry.dataset_id)
            .map(|id| &arena.tables()[id]);
        let same = table.is_some_and(|t| {
            t.canonical_sha256 == entry.source_sha256
                && t.n_rows == entry.rows
                && t.n_cols == entry.features
                && t.task == entry.task
        });
        if !same || entry.target_column != TARGET_COLUMN {
            return contract(format!(
                "panel table {} disagrees with the arena",
                entry.dataset_id
            ));
        }
    }
    verify_panel_families(families, panel)
}

struct Plan<'a> {
    arena: &'a Arena,
    panel_id: &'a str,
    panel_sha256: &'a str,
    data_seeds: &'a [u64],
    sample_seeds: &'a [u64],
}

impl Plan<'_> {
    /// Every cell of one panel table: each data seed times each view.
    fn table_cells(&self, entry: &PanelDatasetEntry) -> ArenaResult<Vec<Cell>> {
        let Some(table_id) = self.arena.table_id_of(&entry.dataset_id) else {
            return contract(format!(
                "panel table {} is not in the arena",
                entry.dataset_id
            ));
        };
        let table = self.arena.table(table_id)?;
        let dup = self.arena.read_dup(table_id)?;
        let seed_of = |data_seed, purpose| {
            structural_seed(self.panel_sha256, &table.dataset_hash, data_seed, purpose)
        };
        let mut cells = Vec::new();
        for &data_seed in self.data_seeds {
            let split = split_roles(&dup, seed_of(data_seed, PURPOSE_SPLIT));
            let mut views = vec![(VIEW_FULL, (0..table.n_cols).collect::<Vec<usize>>())];
            if table.n_cols > VIEW_COLUMNS {
                let cols = view_columns(table.n_cols, seed_of(data_seed, PURPOSE_COLS));
                views.push((VIEW_P48, cols));
            }
            let supported = split.supported();
            let rows = |role: &[usize]| if supported { role.to_vec() } else { Vec::new() };
            for (view, cols) in views {
                let headline = supported && split.headline();
                cells.push(Cell {
                    cell_id: format!(
                        "{}:{}:{view}:d{data_seed}",
                        self.panel_id, table.dataset_hash
                    ),
                    panel_sha256: self.panel_sha256.to_string(),
                    dataset_hash: table.dataset_hash.clone(),
                    table_id,
                    family_id: entry.family_id.clone(),
                    task: table.task.clone(),
                    n_classes: table.n_classes,
                    view: view.to_string(),
                    cols,
                    data_seed,
                    stratum: if headline {
                        STRATUM_HEADLINE
                    } else {
                        STRATUM_SMALL_N
                    }
                    .to_string(),
                    a_idx: rows(&split.a),
                    b_idx: rows(&split.b),
                    c_idx: rows(&split.c),
                    t_idx: rows(&split.t),
                    sample_seeds: self.sample_seeds.to_vec(),
                    multipliers: MULTIPLIERS.to_vec(),
                    status: if supported {
                        STATUS_OK
                    } else {
                        STATUS_UNSUPPORTED
                    }
                    .to_string(),
                });
            }
        }
        Ok(cells)
    }
}

pub fn run(args: &PlanArgs) -> ArenaResult<String> {
    let arena = Arena::open(&args.arena)?;
    if args.verify_checksums {
        arena.verify_checksums()?;
    }
    let panel = PanelManifest::load(&args.panel)?;
    let families: Vec<FamilyRow> = read_jsonl(&args.families)?;
    check_inputs(&arena, &panel, &families)?;
    let panel_sha256 = panel.canonical_sha256()?;
    let (data_count, sample_count) = args.tier.seed_counts();
    let data_seeds = seed_list("data-seeds", data_count, &args.data_seeds)?;
    let sample_seeds = seed_list("sample-seeds", sample_count, &args.sample_seeds)?;

    let selected = select_tables(&panel, args.tier);
    let plan = Plan {
        arena: &arena,
        panel_id: &panel.panel_id,
        panel_sha256: &panel_sha256,
        data_seeds: &data_seeds,
        sample_seeds: &sample_seeds,
    };
    let per_table = par_map(args.threads, selected.len(), |i| {
        plan.table_cells(selected[i])
    });
    let mut cells: Vec<Cell> = Vec::new();
    for table_cells in per_table {
        cells.extend(table_cells?);
    }
    cells.sort_by(|a, b| a.cell_id.cmp(&b.cell_id));
    let schedule_sha256 = write_jsonl(&args.out, &cells)?;

    let supported: Vec<&Cell> = cells.iter().filter(|c| c.status == STATUS_OK).collect();
    let headline = supported
        .iter()
        .filter(|c| c.stratum == STRATUM_HEADLINE)
        .count();
    let mut cells_by_view = BTreeMap::new();
    for cell in &cells {
        *cells_by_view.entry(cell.view.clone()).or_default() += 1;
    }
    canonical_json(&Summary {
        command: "plan",
        panel_id: panel.panel_id.clone(),
        panel_sha256: panel_sha256.clone(),
        tier: args.tier.name(),
        data_seeds: data_seeds.clone(),
        sample_seeds: sample_seeds.clone(),
        multipliers: MULTIPLIERS.to_vec(),
        tables: selected.len(),
        families: selected
            .iter()
            .map(|e| e.family_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        cells: cells.len(),
        supported: supported.len(),
        unsupported: cells.len() - supported.len(),
        supported_headline: headline,
        supported_small_n: supported.len() - headline,
        cells_by_view,
        schedule_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dope_arena::panel::PanelRole;

    fn entry(hash: &str, family: &str, rows: usize) -> PanelDatasetEntry {
        PanelDatasetEntry {
            dataset_id: hash.into(),
            family_id: family.into(),
            target_column: TARGET_COLUMN.into(),
            task: "regression".into(),
            group_column: None,
            time_column: None,
            source_sha256: "0".repeat(64),
            rows,
            features: 3,
        }
    }

    #[test]
    fn tiers_set_seed_counts_and_table_selection() {
        assert_eq!(Tier::Lite.seed_counts(), (1, 1));
        assert_eq!(Tier::Full.seed_counts(), (2, 2));
        assert_eq!(Tier::Confirm.seed_counts(), (3, 2));
        assert_eq!(Tier::Sealed.seed_counts(), (3, 3));
        let panel = PanelManifest {
            protocol: "dope-fidelity/v3".into(),
            panel_id: "dev".into(),
            role: PanelRole::Development,
            datasets: vec![
                entry("h5", "lf-a", 100),
                entry("h2", "lf-a", 300),
                entry("h1", "lf-a", 300),
                entry("h9", "lf-b", 50),
                entry("h7", "lf-c", 80),
                entry("h8", "lf-c", 80),
            ],
            manifest_sha256: None,
        };
        let ids = |tier| -> Vec<&str> {
            select_tables(&panel, tier)
                .iter()
                .map(|e| e.dataset_id.as_str())
                .collect()
        };
        // Most rows per family; ties go to the smaller dataset hash.
        assert_eq!(ids(Tier::Lite), ["h1", "h9", "h7"]);
        for tier in [Tier::Full, Tier::Confirm, Tier::Sealed] {
            assert_eq!(ids(tier), ["h5", "h2", "h1", "h9", "h7", "h8"]);
        }
    }

    #[test]
    fn seed_lists_default_to_the_tier_and_reject_bad_overrides() {
        assert_eq!(seed_list("data-seeds", 3, &None).unwrap(), [0, 1, 2]);
        assert_eq!(
            seed_list("data-seeds", 3, &Some(vec![7, 5])).unwrap(),
            [7, 5]
        );
        assert!(seed_list("sample-seeds", 2, &Some(Vec::new())).is_err());
        let err = seed_list("sample-seeds", 2, &Some(vec![4, 4])).unwrap_err();
        assert!(err.to_string().contains("--sample-seeds"));
    }

    #[test]
    fn cells_serialise_with_sorted_keys_and_status_defaults_to_ok() {
        let cell = Cell {
            cell_id: "dev:h1:full:d0".into(),
            panel_sha256: "p".into(),
            dataset_hash: "h1".into(),
            table_id: 4,
            family_id: "lf-h1".into(),
            task: "binary".into(),
            n_classes: 2,
            view: VIEW_FULL.into(),
            cols: vec![0, 1],
            data_seed: 0,
            stratum: STRATUM_SMALL_N.into(),
            a_idx: vec![1],
            b_idx: vec![2],
            c_idx: Vec::new(),
            t_idx: vec![0],
            sample_seeds: vec![0],
            multipliers: MULTIPLIERS.to_vec(),
            status: STATUS_OK.into(),
        };
        let line = canonical_json(&cell).unwrap();
        assert_eq!(
            line,
            "{\"a_idx\":[1],\"b_idx\":[2],\"c_idx\":[],\"cell_id\":\"dev:h1:full:d0\",\"cols\":[0,1],\
             \"data_seed\":0,\"dataset_hash\":\"h1\",\"family_id\":\"lf-h1\",\"multipliers\":[4,1],\
             \"n_classes\":2,\"panel_sha256\":\"p\",\"sample_seeds\":[0],\"status\":\"ok\",\
             \"stratum\":\"small_n\",\"t_idx\":[0],\"table_id\":4,\"task\":\"binary\",\"view\":\"full\"}"
        );
        let without_status = line.replace("\"status\":\"ok\",", "");
        assert_eq!(serde_json::from_str::<Cell>(&without_status).unwrap(), cell);
        assert_eq!(serde_json::from_str::<Cell>(&line).unwrap(), cell);
    }
}
