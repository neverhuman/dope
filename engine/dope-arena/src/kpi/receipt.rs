//! `dope.kpi-receipt/v2`: the receipt of `primary`, `compare` and `null`,
//! and the run context the three commands share.
//!
//! A receipt is a pure function of its inputs and of the binary: no clock,
//! no path, sorted keys, and every float written with 12 decimals, so the
//! same inputs give the same bytes. It names the contract (SHA-256), the
//! panel filter, the schedule, the real reference and every rows file read
//! (SHA-256 each), the tool version and git commit, and the seeds.

use super::compare::ComparisonResult;
use super::jsonl::{canonical_pretty, sha256_hex, write_atomic};
use super::null::NullResult;
use super::panelsel::{resolve, Panel, PanelInfo, PanelKind};
use super::primary::ArmSummary;
use super::rundir::{contract, Arm, RunDir};
use super::spec::{Contract, ContractRef};
use dope_arena::ArenaResult;
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const RECEIPT_SCHEMA: &str = "dope.kpi-receipt/v2";
/// Decimals kept for every float of a receipt.
pub const DECIMALS: usize = 12;

#[derive(clap::Args, Clone, Debug)]
pub struct RunArgs {
    /// Run directory holding `realref.jsonl`, the schedule and `score/<arm>/rows.jsonl`.
    #[arg(long)]
    pub run_dir: PathBuf,
    /// Schedule the arms were scored on (default: the run directory's one `schedule*.jsonl`).
    #[arg(long)]
    pub schedule: Option<PathBuf>,
    /// Families read: `all`, `dev` (families holding a `--dev-tables` table) or `confirm` (the rest).
    #[arg(long, value_enum, default_value_t = PanelKind::All)]
    pub panel: PanelKind,
    /// Table hashes, one per line, defining the DEV families.
    #[arg(long)]
    pub dev_tables: Option<PathBuf>,
    /// Contract to apply (default: the compiled `contracts/dope-kpi-v2.json`).
    #[arg(long)]
    pub contract: Option<PathBuf>,
    /// Bootstrap seed (default: the contract's).
    #[arg(long)]
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Tool {
    pub name: String,
    pub version: String,
    /// `DOPE_KPI_GIT_SHA` at build time (`unset` for a build that did not set it).
    pub git_sha: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Inputs {
    pub schedule_file: String,
    pub schedule_sha256: String,
    pub schedule_panel_sha256: String,
    pub realref_sha256: String,
    /// SHA-256 of every `rows.jsonl` read, by arm.
    pub rows_sha256: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BootstrapInfo {
    pub draws: usize,
    pub seed: u64,
    pub lower_quantile: f64,
    pub upper_quantile: f64,
    pub lower_rank: usize,
    pub upper_rank: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NoiseSd {
    pub value: f64,
    /// `default` (the contract's) or `input` (`--noise-sd`).
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Receipt {
    pub schema: String,
    pub command: String,
    pub tool: Tool,
    pub contract: ContractRef,
    pub inputs: Inputs,
    pub panel: PanelInfo,
    pub bootstrap: BootstrapInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_sd: Option<NoiseSd>,
    #[serde(default)]
    pub arms: Vec<ArmSummary>,
    #[serde(default)]
    pub comparisons: Vec<ComparisonResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null: Option<NullResult>,
}

pub fn tool() -> Tool {
    Tool {
        name: "dope_kpi".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: option_env!("DOPE_KPI_GIT_SHA")
            .unwrap_or("unset")
            .to_string(),
    }
}

/// What every statistic command reads: run directory, contract, panel and arms.
pub struct Context {
    pub run: RunDir,
    pub contract: Contract,
    pub contract_ref: ContractRef,
    pub panel: Panel,
    pub seed: u64,
    pub arms: BTreeMap<String, Arm>,
}

impl Context {
    pub fn open(args: &RunArgs, arm_ids: &[&str]) -> ArenaResult<Context> {
        let (contract, contract_ref) = Contract::load(args.contract.as_deref())?;
        let run = RunDir::open(&args.run_dir, args.schedule.as_deref())?;
        let panel = resolve(args.panel, args.dev_tables.as_deref(), &run)?;
        let mut arms = BTreeMap::new();
        for id in arm_ids {
            if !arms.contains_key(*id) {
                arms.insert(id.to_string(), run.load_arm(id)?);
            }
        }
        Ok(Context {
            seed: args.seed.unwrap_or(contract.bootstrap.seed),
            run,
            contract,
            contract_ref,
            panel,
            arms,
        })
    }

    pub fn arm(&self, id: &str) -> &Arm {
        &self.arms[id]
    }

    pub fn noise_sd(&self, given: Option<f64>) -> ArenaResult<NoiseSd> {
        match given {
            Some(value) if !(value.is_finite() && value >= 0.0) => {
                contract("--noise-sd must be a finite number >= 0")
            }
            Some(value) => Ok(NoiseSd {
                value,
                source: "input".to_string(),
            }),
            None => Ok(NoiseSd {
                value: self.contract.promotion.default_noise_sd,
                source: "default".to_string(),
            }),
        }
    }

    /// A receipt with every input recorded and no result yet.
    pub fn receipt(&self, command: &str) -> Receipt {
        let b = &self.contract.bootstrap;
        let (lower_rank, upper_rank) =
            super::boot::ranks(b.draws, b.lower_quantile, b.upper_quantile);
        Receipt {
            schema: RECEIPT_SCHEMA.to_string(),
            command: command.to_string(),
            tool: tool(),
            contract: self.contract_ref.clone(),
            inputs: Inputs {
                schedule_file: self.run.schedule.file_name.clone(),
                schedule_sha256: self.run.schedule.sha256.clone(),
                schedule_panel_sha256: self.run.schedule.panel_sha256.clone(),
                realref_sha256: self.run.realref_sha256.clone(),
                rows_sha256: self
                    .arms
                    .values()
                    .map(|arm| (arm.id.clone(), arm.rows_sha256.clone()))
                    .collect(),
            },
            panel: self.panel.info.clone(),
            bootstrap: BootstrapInfo {
                draws: b.draws,
                seed: self.seed,
                lower_quantile: b.lower_quantile,
                upper_quantile: b.upper_quantile,
                lower_rank,
                upper_rank,
            },
            noise_sd: None,
            arms: Vec::new(),
            comparisons: Vec::new(),
            null: None,
        }
    }
}

/// `x` with `DECIMALS` decimals (and no negative zero).
fn round(x: f64) -> f64 {
    let rounded: f64 = format!("{x:.DECIMALS$}").parse().unwrap_or(x);
    if rounded == 0.0 {
        0.0
    } else {
        rounded
    }
}

fn round_floats(value: &mut Value) {
    match value {
        Value::Number(number) if number.is_f64() => {
            let x = number.as_f64().map_or(0.0, round);
            *value = Number::from_f64(x).map_or(Value::Null, Value::Number);
        }
        Value::Array(items) => items.iter_mut().for_each(round_floats),
        Value::Object(map) => map.values_mut().for_each(round_floats),
        _ => {}
    }
}

/// Receipt bytes: sorted keys, two-space indent, rounded floats, trailing newline.
pub fn receipt_bytes(receipt: &Receipt) -> ArenaResult<Vec<u8>> {
    let mut value = serde_json::to_value(receipt)?;
    round_floats(&mut value);
    Ok(canonical_pretty(&value)?.into_bytes())
}

/// Write `receipt` to `path` and return the SHA-256 of the bytes written.
pub fn write_receipt(path: &Path, receipt: &Receipt) -> ArenaResult<String> {
    let bytes = receipt_bytes(receipt)?;
    write_atomic(path, &bytes)?;
    Ok(sha256_hex(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_keep_twelve_decimals_and_integers_stay_integers() {
        let mut value = serde_json::json!({
            "a": 0.123_456_789_012_345_6,
            "b": [-1e-15, 14782.0, 2048],
            "c": {"d": -0.030_496_235_321_696_142},
        });
        round_floats(&mut value);
        assert_eq!(
            super::super::jsonl::canonical_json(&value).unwrap(),
            r#"{"a":0.123456789012,"b":[0.0,14782.0,2048],"c":{"d":-0.030496235322}}"#
        );
    }

    #[test]
    fn the_tool_names_itself() {
        let tool = tool();
        assert_eq!(tool.name, "dope_kpi");
        assert!(!tool.version.is_empty() && !tool.git_sha.is_empty());
    }
}
