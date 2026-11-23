//! Panel filter of the statistic commands: `--panel all|dev|confirm`.
//!
//! DEV is every family that holds one of the tables listed in `--dev-tables`
//! (one table hash per line); CONFIRM is every other family of the schedule.
//! A listed table maps to its family through the schedule, or through
//! `<run-dir>/families.jsonl` (written by `build-panels`) for a table the
//! schedule does not hold. The receipt records the list's hash, how many
//! tables matched, and the hash of the family set read.

use super::jsonl::{parse_jsonl, read_file, sha256_hex};
use super::rundir::{contract, RunDir, FAMILIES_FILE};
use dope_arena::ArenaResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PanelKind {
    All,
    Dev,
    Confirm,
}

impl PanelKind {
    pub fn id(self) -> &'static str {
        match self {
            PanelKind::All => "all",
            PanelKind::Dev => "dev",
            PanelKind::Confirm => "confirm",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PanelInfo {
    pub id: String,
    pub dev_tables_sha256: Option<String>,
    /// Distinct table hashes listed.
    pub dev_tables: usize,
    /// Listed tables the schedule holds.
    pub dev_tables_scheduled: usize,
    /// Listed tables found neither in the schedule nor in `families.jsonl`.
    pub dev_tables_unmatched: usize,
    /// `families.jsonl` when it was read.
    pub families_sha256: Option<String>,
    /// Schedule families that are DEV.
    pub dev_families: usize,
    /// Schedule families this panel keeps.
    pub families: usize,
    /// SHA-256 of the kept family ids, sorted, one per line.
    pub family_set_sha256: String,
}

#[derive(Clone, Debug)]
pub struct Panel {
    pub info: PanelInfo,
    families: BTreeSet<String>,
}

impl Panel {
    pub fn contains(&self, family: &str) -> bool {
        self.families.contains(family)
    }
}

#[derive(Deserialize)]
struct FamilyLine {
    dataset_hash: String,
    family_id: String,
}

/// Distinct table hashes of a `--dev-tables` file.
fn dev_tables(bytes: &[u8]) -> ArenaResult<BTreeSet<String>> {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return contract("--dev-tables is not UTF-8");
    };
    let mut tables = BTreeSet::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let hash = line.len() <= 64
            && line
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
        if !hash {
            return contract(format!(
                "--dev-tables: line {} is not a table hash",
                number + 1
            ));
        }
        tables.insert(line.to_string());
    }
    if tables.is_empty() {
        return contract("--dev-tables lists no table");
    }
    Ok(tables)
}

pub fn resolve(kind: PanelKind, dev_list: Option<&Path>, run: &RunDir) -> ArenaResult<Panel> {
    let scheduled = run.schedule.table_families();
    let all: BTreeSet<&str> = scheduled.values().copied().collect();
    let mut info = PanelInfo {
        id: kind.id().to_string(),
        dev_tables_sha256: None,
        dev_tables: 0,
        dev_tables_scheduled: 0,
        dev_tables_unmatched: 0,
        families_sha256: None,
        dev_families: 0,
        families: 0,
        family_set_sha256: String::new(),
    };
    let kept: BTreeSet<&str> = match (kind, dev_list) {
        (PanelKind::All, None) => all.clone(),
        (PanelKind::All, Some(_)) => return contract("--dev-tables needs --panel dev or confirm"),
        (_, None) => return contract("--panel dev and confirm need --dev-tables"),
        (_, Some(path)) => {
            let bytes = read_file(path)?;
            let listed = dev_tables(&bytes)?;
            info.dev_tables_sha256 = Some(sha256_hex(&bytes));
            info.dev_tables = listed.len();
            let mut by_hash: BTreeMap<String, String> = BTreeMap::new();
            let families_path = run.dir.join(FAMILIES_FILE);
            if families_path.is_file() {
                let bytes = read_file(&families_path)?;
                info.families_sha256 = Some(sha256_hex(&bytes));
                for line in parse_jsonl::<FamilyLine>(FAMILIES_FILE, &bytes)? {
                    by_hash.insert(line.dataset_hash, line.family_id);
                }
            }
            for (table, family) in &scheduled {
                if by_hash.get(*table).is_some_and(|f| f != family) {
                    return contract(format!(
                        "{FAMILIES_FILE} and the schedule disagree on table {table}"
                    ));
                }
            }
            let mut dev = BTreeSet::new();
            for table in &listed {
                let family = scheduled.get(table.as_str()).copied();
                info.dev_tables_scheduled += usize::from(family.is_some());
                match family.or_else(|| by_hash.get(table).map(String::as_str)) {
                    Some(family) => {
                        dev.insert(family.to_string());
                    }
                    None => info.dev_tables_unmatched += 1,
                }
            }
            let dev_in_schedule: BTreeSet<&str> =
                all.iter().copied().filter(|f| dev.contains(*f)).collect();
            info.dev_families = dev_in_schedule.len();
            if kind == PanelKind::Dev {
                dev_in_schedule
            } else {
                all.difference(&dev_in_schedule).copied().collect()
            }
        }
    };
    let listing: String = kept.iter().map(|f| format!("{f}\n")).collect();
    info.families = kept.len();
    info.family_set_sha256 = sha256_hex(listing.as_bytes());
    Ok(Panel {
        info,
        families: kept.into_iter().map(str::to_string).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_dir(name: &str) -> std::path::PathBuf {
        let dir = crate::kpi::unit_test_dir(name);
        let line = |table: &str, family: &str| {
            format!(
                r#"{{"cell_id":"p:{table}:full:d0","panel_sha256":"ps","dataset_hash":"{table}","family_id":"{family}","task":"binary","view":"full","data_seed":0,"stratum":"headline","sample_seeds":[0],"multipliers":[4]}}"#
            )
        };
        let schedule = [line("aa", "f1"), line("bb", "f2"), line("cc", "f3")].join("\n");
        std::fs::write(dir.join("schedule.jsonl"), schedule).unwrap();
        let refs: Vec<String> = ["aa", "bb", "cc"]
            .iter()
            .map(|t| format!(r#"{{"cell_id":"p:{t}:full:d0","auditor":"gbdt","informative":false,"l0":null,"l_a":null,"l_b":null,"r_b":null,"c2st_real_ref":null}}"#))
            .collect();
        std::fs::write(dir.join("realref.jsonl"), refs.join("\n")).unwrap();
        // Table dd is not scheduled; its family f2 is.
        let families = r#"{"dataset_hash":"dd","family_id":"f2","canon_stem":"x","evidence":"name","split":"val","task":"binary"}"#;
        std::fs::write(dir.join(FAMILIES_FILE), families).unwrap();
        dir
    }

    #[test]
    fn dev_is_every_family_holding_a_listed_table_and_confirm_the_rest() {
        let dir = run_dir("panelsel");
        let run = RunDir::open(&dir, None).unwrap();
        let list = dir.join("dev.txt");
        std::fs::write(&list, "aa\n\ndd\nee\n").unwrap();
        let dev = resolve(PanelKind::Dev, Some(&list), &run).unwrap();
        assert!(dev.contains("f1") && dev.contains("f2") && !dev.contains("f3"));
        let i = &dev.info;
        assert_eq!(
            (i.dev_tables, i.dev_tables_scheduled, i.dev_tables_unmatched),
            (3, 1, 1)
        );
        assert_eq!((i.dev_families, i.families), (2, 2));
        assert!(i.families_sha256.is_some() && i.dev_tables_sha256.is_some());
        let confirm = resolve(PanelKind::Confirm, Some(&list), &run).unwrap();
        assert!(confirm.contains("f3") && !confirm.contains("f1"));
        assert_eq!(confirm.info.id, "confirm");
        let all = resolve(PanelKind::All, None, &run).unwrap();
        assert_eq!(all.info.families, 3);
        assert_ne!(all.info.family_set_sha256, dev.info.family_set_sha256);
    }

    #[test]
    fn malformed_lists_and_flag_mixes_are_refused() {
        let dir = run_dir("panelsel-errors");
        let run = RunDir::open(&dir, None).unwrap();
        let list = dir.join("dev.txt");
        std::fs::write(&list, "aa\nSecret Column\n").unwrap();
        let err = resolve(PanelKind::Dev, Some(&list), &run)
            .unwrap_err()
            .to_string();
        assert!(err.contains("line 2") && !err.contains("Secret"), "{err}");
        assert!(resolve(PanelKind::Dev, None, &run).is_err());
        assert!(resolve(PanelKind::All, Some(&list), &run).is_err());
    }
}
