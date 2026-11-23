//! Inputs of `primary`, `compare` and `null`: one run directory written by
//! `real-ref` and `score` (formats.md section 7), checked once.
//!
//! `<run-dir>/realref.jsonl`, the schedule the arms were scored on and
//! `<run-dir>/score/<arm>/rows.jsonl` per arm. A rows file must hold exactly
//! one entry per scheduled (cell, sample seed, multiplier) and auditor of the
//! real reference, carry the cell metadata of the schedule and repeat the
//! real-only columns of the real reference. Arms read from one run directory
//! are therefore paired on identical cells, splits and informative flags.
//! Errors name files, lines and cell ids, never a value.

use super::emitlog::check_arm_id;
use super::jsonl::{parse_jsonl, read_file, sha256_hex};
use dope_arena::{ArenaError, ArenaResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const REALREF_FILE: &str = "realref.jsonl";
pub const FAMILIES_FILE: &str = "families.jsonl";
pub const SCORE_DIR: &str = "score";
pub const ROWS_FILE: &str = "rows.jsonl";
pub const TASKS: [&str; 3] = ["regression", "binary", "multiclass"];
pub const STATUS_OK: &str = "ok";
pub const STATUS_UNSUPPORTED: &str = "unsupported";
const STATUSES: [&str; 4] = ["ok", "failed", "over_budget", "unsupported"];

pub fn contract<T>(message: impl Into<String>) -> ArenaResult<T> {
    Err(ArenaError::Contract(message.into()))
}

/// One entry of `rows.jsonl`: the `score::ScoreRow` fields the statistic
/// reads, plus guard fields a later `score` may add (absent in every rows
/// file so far; a guard that needs an absent field is not evaluated).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct KpiRow {
    pub cell_id: String,
    pub family_id: String,
    pub task: String,
    pub view: String,
    pub stratum: String,
    pub auditor: String,
    pub data_seed: u64,
    pub sample_seed: u64,
    pub multiplier: u32,
    pub l0: Option<f64>,
    pub l_a: Option<f64>,
    pub l_b: Option<f64>,
    pub r_s: Option<f64>,
    pub r_b: Option<f64>,
    pub informative: bool,
    pub status: String,
    pub packet_bytes: u64,
    pub artifact_class: String,
    pub c2st_auc: Option<f64>,
    pub c2st_real_ref: Option<f64>,
    /// `evaluate_v3` realism gate of the entry.
    #[serde(default)]
    pub realism_pass: Option<bool>,
    /// `evaluate_v3` empirical privacy gate of the entry.
    #[serde(default)]
    pub privacy_pass: Option<bool>,
    /// `max(0, rate(S in A) - rate(B in A))` of the entry's file (`dvs::v3::privacy`).
    #[serde(default)]
    pub exact_copy_excess: Option<f64>,
}

/// The table of a cell id `<panel>:<dataset_hash>:<view>:d<data_seed>`.
pub fn table_of(cell_id: &str) -> Option<&str> {
    let mut parts = cell_id.rsplitn(3, ':');
    let (_seed, _view) = (parts.next()?, parts.next()?);
    parts.next()?.rsplit_once(':').map(|(_, table)| table)
}

#[derive(Deserialize)]
struct ScheduleLine {
    cell_id: String,
    panel_sha256: String,
    dataset_hash: String,
    family_id: String,
    task: String,
    view: String,
    data_seed: u64,
    stratum: String,
    sample_seeds: Vec<u64>,
    multipliers: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ScheduleCell {
    pub dataset_hash: String,
    pub family_id: String,
    pub task: String,
    pub view: String,
    pub stratum: String,
    pub data_seed: u64,
    pub sample_seeds: Vec<u64>,
    pub multipliers: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub file_name: String,
    pub sha256: String,
    pub panel_sha256: String,
    pub cells: BTreeMap<String, ScheduleCell>,
}

impl Schedule {
    pub fn parse(file_name: &str, bytes: &[u8]) -> ArenaResult<Schedule> {
        let lines: Vec<ScheduleLine> = parse_jsonl(file_name, bytes)?;
        let Some(panel_sha256) = lines.first().map(|l| l.panel_sha256.clone()) else {
            return contract(format!("{file_name}: no cell"));
        };
        let mut cells = BTreeMap::new();
        for line in lines {
            let distinct_seeds: BTreeSet<&u64> = line.sample_seeds.iter().collect();
            let distinct_mults: BTreeSet<&u32> = line.multipliers.iter().collect();
            let well_formed = line.panel_sha256 == panel_sha256
                && TASKS.contains(&line.task.as_str())
                && table_of(&line.cell_id) == Some(line.dataset_hash.as_str())
                && !line.sample_seeds.is_empty()
                && distinct_seeds.len() == line.sample_seeds.len()
                && !line.multipliers.is_empty()
                && distinct_mults.len() == line.multipliers.len();
            if !well_formed {
                return contract(format!("{file_name}: cell {} is malformed", line.cell_id));
            }
            let cell = ScheduleCell {
                dataset_hash: line.dataset_hash,
                family_id: line.family_id,
                task: line.task,
                view: line.view,
                stratum: line.stratum,
                data_seed: line.data_seed,
                sample_seeds: line.sample_seeds,
                multipliers: line.multipliers,
            };
            if cells.insert(line.cell_id.clone(), cell).is_some() {
                return contract(format!("{file_name}: cell {} repeats", line.cell_id));
            }
        }
        Ok(Schedule {
            file_name: file_name.to_string(),
            sha256: sha256_hex(bytes),
            panel_sha256,
            cells,
        })
    }

    /// `dataset_hash -> family_id` of the scheduled tables.
    pub fn table_families(&self) -> BTreeMap<&str, &str> {
        self.cells
            .values()
            .map(|c| (c.dataset_hash.as_str(), c.family_id.as_str()))
            .collect()
    }
}

/// Real-only columns of one (cell, auditor) of `realref.jsonl`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
struct RefLine {
    cell_id: String,
    auditor: String,
    informative: bool,
    l0: Option<f64>,
    l_a: Option<f64>,
    l_b: Option<f64>,
    r_b: Option<f64>,
    c2st_real_ref: Option<f64>,
}

/// One arm's rows, checked against the run directory.
#[derive(Clone, Debug)]
pub struct Arm {
    pub id: String,
    pub rows_sha256: String,
    /// Sorted by (cell, auditor, multiplier, sample seed).
    pub rows: Vec<KpiRow>,
}

impl Arm {
    pub fn sorted(id: &str, rows_sha256: String, mut rows: Vec<KpiRow>) -> Arm {
        rows.sort_by(|a, b| key(a).cmp(&key(b)));
        Arm {
            id: id.to_string(),
            rows_sha256,
            rows,
        }
    }
}

fn key(row: &KpiRow) -> (&str, &str, u32, u64) {
    (&row.cell_id, &row.auditor, row.multiplier, row.sample_seed)
}

/// Two copies of one real-only number (rows repeat the real reference after
/// a JSON round trip, which may move the last bit).
fn same(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0),
        _ => false,
    }
}

#[derive(Clone, Debug)]
pub struct RunDir {
    pub dir: PathBuf,
    pub schedule: Schedule,
    pub realref_sha256: String,
    /// Auditor roster of the real reference, sorted.
    pub auditors: Vec<String>,
    refs: BTreeMap<(String, String), RefLine>,
}

/// The schedule given, else `<dir>/schedule.jsonl`, else the one `schedule*.jsonl` in `dir`.
fn schedule_path(dir: &Path, given: Option<&Path>) -> ArenaResult<PathBuf> {
    if let Some(path) = given {
        return Ok(path.to_path_buf());
    }
    let plain = dir.join("schedule.jsonl");
    if plain.is_file() {
        return Ok(plain);
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| super::jsonl::file_error(dir, &e))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("schedule") && name.ends_with(".jsonl") && path.is_file()
        })
        .collect();
    found.sort();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => contract("the run directory holds no schedule*.jsonl; pass --schedule"),
        _ => contract("the run directory holds several schedule*.jsonl; pass --schedule"),
    }
}

impl RunDir {
    pub fn open(dir: &Path, schedule: Option<&Path>) -> ArenaResult<RunDir> {
        let path = schedule_path(dir, schedule)?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("schedule.jsonl");
        let schedule = Schedule::parse(name, &read_file(&path)?)?;
        let realref = read_file(&dir.join(REALREF_FILE))?;
        let lines: Vec<RefLine> = parse_jsonl(REALREF_FILE, &realref)?;
        let mut refs = BTreeMap::new();
        for line in lines {
            if !schedule.cells.contains_key(&line.cell_id)
                || (line.informative && line.r_b.is_none())
            {
                return contract(format!(
                    "{REALREF_FILE}: cell {} does not fit the schedule",
                    line.cell_id
                ));
            }
            let at = (line.cell_id.clone(), line.auditor.clone());
            if refs.insert(at, line).is_some() {
                return contract(format!("{REALREF_FILE}: a (cell, auditor) repeats"));
            }
        }
        let auditors: BTreeSet<String> = refs.keys().map(|(_, a)| a.clone()).collect();
        if auditors.is_empty() || refs.len() != schedule.cells.len() * auditors.len() {
            return contract(format!("{REALREF_FILE} does not cover the schedule"));
        }
        Ok(RunDir {
            dir: dir.to_path_buf(),
            schedule,
            realref_sha256: sha256_hex(&realref),
            auditors: auditors.into_iter().collect(),
            refs,
        })
    }

    pub fn rows_path(&self, arm_id: &str) -> PathBuf {
        self.dir.join(SCORE_DIR).join(arm_id).join(ROWS_FILE)
    }

    pub fn load_arm(&self, arm_id: &str) -> ArenaResult<Arm> {
        check_arm_id(arm_id)?;
        let bytes = read_file(&self.rows_path(arm_id))?;
        let label = format!("{SCORE_DIR}/{arm_id}/{ROWS_FILE}");
        let rows: Vec<KpiRow> = parse_jsonl(&label, &bytes)?;
        self.check_rows(&label, &rows)?;
        Ok(Arm::sorted(arm_id, sha256_hex(&bytes), rows))
    }

    /// Require exactly one entry per scheduled file and auditor, each with the
    /// schedule's cell metadata and the real reference's real-only columns.
    pub fn check_rows(&self, label: &str, rows: &[KpiRow]) -> ArenaResult<()> {
        let mut seen = BTreeSet::new();
        for row in rows {
            let cell = self.schedule.cells.get(&row.cell_id);
            let scheduled = cell.is_some_and(|c| {
                c.family_id == row.family_id
                    && c.task == row.task
                    && c.view == row.view
                    && c.stratum == row.stratum
                    && c.data_seed == row.data_seed
                    && c.sample_seeds.contains(&row.sample_seed)
                    && c.multipliers.contains(&row.multiplier)
            });
            if !scheduled || !STATUSES.contains(&row.status.as_str()) {
                return contract(format!(
                    "{label}: an entry of cell {} is not in the schedule",
                    row.cell_id
                ));
            }
            let reference = self.refs.get(&(row.cell_id.clone(), row.auditor.clone()));
            let agrees = reference.is_some_and(|r| {
                r.informative == row.informative
                    && same(r.l0, row.l0)
                    && same(r.l_a, row.l_a)
                    && same(r.l_b, row.l_b)
                    && same(r.r_b, row.r_b)
                    && same(r.c2st_real_ref, row.c2st_real_ref)
            });
            if !agrees {
                return contract(format!(
                    "{label}: cell {} was not scored against this real reference",
                    row.cell_id
                ));
            }
            if !seen.insert(key(row)) {
                return contract(format!("{label}: an entry of cell {} repeats", row.cell_id));
            }
        }
        let files: usize = self
            .schedule
            .cells
            .values()
            .map(|c| c.sample_seeds.len() * c.multipliers.len())
            .sum();
        if rows.len() != files * self.auditors.len() {
            return contract(format!(
                "{label} does not cover the schedule: {} entries, {} scheduled",
                rows.len(),
                files * self.auditors.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE: &str = r#"{"cell_id":"p:t1:full:d0","panel_sha256":"ps","dataset_hash":"t1","family_id":"f1","task":"binary","view":"full","data_seed":0,"stratum":"headline","sample_seeds":[0],"multipliers":[4,1],"a_idx":[1,2]}"#;

    fn reference(auditor: &str) -> String {
        format!(
            r#"{{"cell_id":"p:t1:full:d0","auditor":"{auditor}","informative":true,"l0":1.0,"l_a":0.5,"l_b":0.5,"r_b":1.0,"c2st_real_ref":0.5,"y_true":[0.0,1.0]}}"#
        )
    }

    fn row(auditor: &str, multiplier: u32) -> String {
        format!(
            r#"{{"artifact_class":"source_free","auditor":"{auditor}","c2st_auc":0.6,"c2st_real_ref":0.5,"cell_id":"p:t1:full:d0","data_seed":0,"family_id":"f1","informative":true,"l0":1.0,"l_a":0.5,"l_b":0.5,"l_s":0.6,"ledger_id":"x","multiplier":{multiplier},"packet_bytes":900,"r_b":1.0,"r_s":0.8,"reason":"","rows":40,"sample_seed":0,"status":"ok","stratum":"headline","task":"binary","view":"full"}}"#
        )
    }

    fn run_dir(name: &str, rows: &[String]) -> PathBuf {
        let dir = crate::kpi::unit_test_dir(name);
        std::fs::write(dir.join("schedule-lite.jsonl"), format!("{LINE}\n")).unwrap();
        let refs = format!("{}\n{}\n", reference("gbdt"), reference("linear-seeded-v1"));
        std::fs::write(dir.join(REALREF_FILE), refs).unwrap();
        let arm = dir.join(SCORE_DIR).join("arm");
        std::fs::create_dir_all(&arm).unwrap();
        std::fs::write(arm.join(ROWS_FILE), rows.join("\n")).unwrap();
        dir
    }

    fn full_rows() -> Vec<String> {
        vec![
            row("linear-seeded-v1", 4),
            row("gbdt", 1),
            row("gbdt", 4),
            row("linear-seeded-v1", 1),
        ]
    }

    #[test]
    fn table_ids_come_from_the_cell_id() {
        assert_eq!(
            table_of("baseline-v250:009662904881da54:full:d0"),
            Some("009662904881da54")
        );
        assert_eq!(table_of("a:b:c:p48:d3"), Some("c"));
        assert_eq!(table_of("t1:full:d0"), None);
    }

    #[test]
    fn rows_are_checked_against_schedule_and_real_reference() {
        let dir = run_dir("rundir-ok", &full_rows());
        let run = RunDir::open(&dir, None).unwrap();
        assert_eq!(run.schedule.file_name, "schedule-lite.jsonl");
        assert_eq!(run.auditors, ["gbdt", "linear-seeded-v1"]);
        let arm = run.load_arm("arm").unwrap();
        let order: Vec<(&str, u32)> = arm
            .rows
            .iter()
            .map(|r| (r.auditor.as_str(), r.multiplier))
            .collect();
        assert_eq!(
            order,
            [
                ("gbdt", 1),
                ("gbdt", 4),
                ("linear-seeded-v1", 1),
                ("linear-seeded-v1", 4)
            ]
        );
        assert_eq!(arm.rows[0].realism_pass, None);
        assert_eq!(arm.rows_sha256.len(), 64);
        assert!(run.load_arm("../arm").is_err());
    }

    type RowsEdit = Box<dyn Fn(&mut Vec<String>)>;

    #[test]
    fn incomplete_or_foreign_rows_are_refused_without_values() {
        let cases: [(&str, RowsEdit, &str); 4] = [
            (
                "rundir-missing",
                Box::new(|rows| {
                    rows.pop();
                }),
                "does not cover",
            ),
            (
                "rundir-repeat",
                Box::new(|rows| rows[3] = row("gbdt", 4)),
                "repeats",
            ),
            (
                "rundir-ref",
                Box::new(|rows| rows[0] = rows[0].replace("\"l_a\":0.5", "\"l_a\":0.25")),
                "real reference",
            ),
            (
                "rundir-cell",
                Box::new(|rows| rows[0] = rows[0].replace("\"view\":\"full\"", "\"view\":\"p48\"")),
                "not in the schedule",
            ),
        ];
        for (name, edit, expected) in cases {
            let mut rows = full_rows();
            edit(&mut rows);
            let run = RunDir::open(&run_dir(name, &rows), None).unwrap();
            let err = run.load_arm("arm").unwrap_err().to_string();
            assert!(err.contains(expected), "{name}: {err}");
            assert!(!err.contains("0.25") && !err.contains("0.5"), "{err}");
        }
    }

    #[test]
    fn the_schedule_must_be_unique_or_named() {
        let dir = run_dir("rundir-two-schedules", &full_rows());
        std::fs::write(dir.join("schedule-full.jsonl"), format!("{LINE}\n")).unwrap();
        let err = RunDir::open(&dir, None).unwrap_err().to_string();
        assert!(err.contains("several"), "{err}");
        let named = dir.join("schedule-full.jsonl");
        assert_eq!(
            RunDir::open(&dir, Some(&named)).unwrap().schedule.file_name,
            "schedule-full.jsonl"
        );
    }
}
