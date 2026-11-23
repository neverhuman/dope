//! Synthetic ledgers for the unit tests of the statistic commands.

use super::jsonl::{canonical_json, write_jsonl};
use super::panelsel::PanelKind;
use super::receipt::{Context, RunArgs};
use super::rundir::{table_of, Arm, KpiRow, REALREF_FILE, ROWS_FILE, SCORE_DIR};
use super::spec::Contract;
use super::trg::Protocol;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const LINEAR: &str = "linear-seeded-v1";
pub const GBDT: &str = "gbdt";
pub const MLP: &str = "mlp-seeded-v1";

pub fn contract() -> Contract {
    Contract::embedded().unwrap().0
}

pub fn protocol() -> Protocol {
    contract().protocol()
}

/// One scored, informative headline entry at multiplier 4 with `R_B = 1`.
pub fn row(table: &str, family: &str, task: &str, auditor: &str, r_s: f64) -> KpiRow {
    KpiRow {
        cell_id: format!("p:{table}:full:d0"),
        family_id: family.to_string(),
        task: task.to_string(),
        view: "full".to_string(),
        stratum: "headline".to_string(),
        auditor: auditor.to_string(),
        data_seed: 0,
        sample_seed: 0,
        multiplier: 4,
        l0: Some(1.0),
        l_a: Some(0.5),
        l_b: Some(0.5),
        r_s: Some(r_s),
        r_b: Some(1.0),
        informative: true,
        status: "ok".to_string(),
        packet_bytes: 1000,
        artifact_class: "source_free".to_string(),
        c2st_auc: Some(0.6),
        c2st_real_ref: Some(0.5),
        ledger_id: None,
        realism_pass: None,
        privacy_pass: None,
        exact_copy_excess: None,
    }
}

pub fn arm(rows: Vec<KpiRow>) -> Arm {
    Arm::sorted("arm", "rows-sha".to_string(), rows)
}

/// A cell scored by all three auditors with the same retention; its TRG is `r_s - 1`.
pub fn cell(table: &str, family: &str, task: &str, r_s: f64) -> Vec<KpiRow> {
    [LINEAR, GBDT, MLP]
        .iter()
        .map(|auditor| row(table, family, task, auditor, r_s))
        .collect()
}

/// An arm named `id` with one cell per `(table, family, task, r_s)`.
pub fn arm_of(id: &str, cells: &[(&str, &str, &str, f64)]) -> Arm {
    let rows = cells
        .iter()
        .flat_map(|&(table, family, task, r_s)| cell(table, family, task, r_s))
        .collect();
    Arm::sorted(id, String::new(), rows)
}

/// Every guard field present and passing.
pub fn with_guard_fields(mut arm: Arm) -> Arm {
    for row in &mut arm.rows {
        row.realism_pass = Some(true);
        row.privacy_pass = Some(true);
        row.exact_copy_excess = Some(0.0);
    }
    arm
}

/// Apply `edit` to every row of `arm` whose cell holds `table`.
pub fn edit(mut arm: Arm, table: &str, edit: impl Fn(&mut KpiRow)) -> Arm {
    for row in arm
        .rows
        .iter_mut()
        .filter(|r| table_of(&r.cell_id) == Some(table))
    {
        edit(row);
    }
    arm
}

/// A run directory whose schedule and real reference are those of `arms[0]`.
pub fn write_run(name: &str, arms: &[Arm]) -> PathBuf {
    let dir = crate::kpi::unit_test_dir(name);
    let first = &arms[0].rows;
    let mut cells: BTreeMap<&str, (&KpiRow, BTreeSet<u64>, BTreeSet<u32>)> = BTreeMap::new();
    for row in first {
        let held = cells
            .entry(&row.cell_id)
            .or_insert((row, BTreeSet::new(), BTreeSet::new()));
        held.1.insert(row.sample_seed);
        held.2.insert(row.multiplier);
    }
    let schedule: Vec<_> = cells
        .iter()
        .map(|(cell_id, (row, seeds, multipliers))| {
            json!({"cell_id": cell_id, "panel_sha256": "ps", "dataset_hash": table_of(cell_id),
                "family_id": row.family_id, "task": row.task, "view": row.view,
                "data_seed": row.data_seed, "stratum": row.stratum,
                "sample_seeds": seeds, "multipliers": multipliers})
        })
        .collect();
    write_jsonl(&dir.join("schedule.jsonl"), &schedule).unwrap();
    let mut refs: BTreeMap<(&str, &str), serde_json::Value> = BTreeMap::new();
    for row in first {
        refs.entry((&row.cell_id, &row.auditor)).or_insert_with(|| {
            json!({"cell_id": row.cell_id, "auditor": row.auditor, "informative": row.informative,
                "l0": row.l0, "l_a": row.l_a, "l_b": row.l_b, "r_b": row.r_b,
                "c2st_real_ref": row.c2st_real_ref})
        });
    }
    write_jsonl(
        &dir.join(REALREF_FILE),
        &refs.into_values().collect::<Vec<_>>(),
    )
    .unwrap();
    for arm in arms {
        let lines: Vec<String> = arm
            .rows
            .iter()
            .map(|r| canonical_json(r).unwrap())
            .collect();
        let path = dir.join(SCORE_DIR).join(&arm.id).join(ROWS_FILE);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, lines.join("\n") + "\n").unwrap();
    }
    dir
}

pub fn run_args(dir: &std::path::Path) -> RunArgs {
    RunArgs {
        run_dir: dir.to_path_buf(),
        schedule: None,
        panel: PanelKind::All,
        dev_tables: None,
        contract: None,
        seed: None,
    }
}

/// Write `arms` as a run directory and open every one of them.
pub fn context(name: &str, arms: &[Arm]) -> Context {
    let dir = write_run(name, arms);
    let ids: Vec<&str> = arms.iter().map(|a| a.id.as_str()).collect();
    Context::open(&run_args(&dir), &ids).unwrap()
}
