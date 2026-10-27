//! Typed view of the native landmark report fields read by
//! `descriptors.extract_report_descriptors` (Python reads `report.json` dicts).
//!
//! Build it from the native Rust [`Signature`] or from the JSON dict
//! (`Signature::private_report()` output or a Python-written report).

use crate::error::{DopeError, Result};
use crate::landmark::{CellState, Signature};
use serde_json::Value;

/// One `cells[]` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportCellView {
    pub id: i64,
    /// `"complete" | "uninformative" | "unsupported" | "over_budget" | "failed" | ...`
    pub state: String,
    pub gain: Option<f64>,
    pub se: Option<f64>,
    pub stable: bool,
    pub lambda_indices: Vec<f64>,
}

/// One `spectrum[]` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumView {
    pub slot: i64,
    /// `None` == key missing / null (Python substitutes `[None]*9`).
    pub values: Option<Vec<Option<f64>>>,
}

/// Fields of a native report consumed by the MLC-v3 descriptors.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NativeReportView {
    pub cells: Vec<ReportCellView>,
    pub target_stats: Option<Vec<Option<f64>>>,
    pub rows: Option<i64>,
    pub features: Option<i64>,
    pub classes: Option<i64>,
    pub task: Option<String>,
    pub spectrum: Vec<SpectrumView>,
}

fn state_str(s: CellState) -> &'static str {
    match s {
        CellState::Complete => "complete",
        CellState::Uninformative => "uninformative",
        CellState::Unsupported => "unsupported",
        CellState::OverBudget => "over_budget",
        CellState::Failed => "failed",
        CellState::NotRequested => "not_requested",
    }
}

fn bad(msg: &str) -> DopeError {
    DopeError::Input(format!("prepare_report_{msg}"))
}

/// Python `float(x)` for JSON scalars (numbers, bools, numeric strings).
fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Python `int(x)` for JSON scalars (truncates floats).
fn as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::Bool(b) => Some(*b as i64),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

/// Python truthiness of a JSON value.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn opt_f64(v: Option<&Value>) -> Option<f64> {
    match v {
        None | Some(Value::Null) => None,
        Some(x) => as_f64(x),
    }
}

impl NativeReportView {
    /// A report with no cells, no spectrum and no shape (Python `{}`).
    pub fn empty() -> Self {
        Self::default()
    }

    /// From the native Rust signature (same content as `private_report()`).
    pub fn from_signature(sig: &Signature) -> Self {
        NativeReportView {
            cells: sig
                .cells
                .iter()
                .map(|c| ReportCellView {
                    id: c.id.0 as i64,
                    state: state_str(c.state).to_string(),
                    gain: c.gain,
                    se: c.se,
                    stable: c.stable,
                    lambda_indices: c.lambda_indices.iter().map(|&l| l as f64).collect(),
                })
                .collect(),
            target_stats: Some(sig.target_stats.to_vec()),
            rows: Some(sig.rows as i64),
            features: Some(sig.features as i64),
            classes: Some(sig.classes as i64),
            task: Some(sig.task.to_string()),
            spectrum: sig
                .spectrum
                .iter()
                .map(|s| SpectrumView {
                    slot: s.slot as i64,
                    values: Some(s.values.to_vec()),
                })
                .collect(),
        }
    }

    /// From a report JSON object as read by the Python path.
    pub fn from_json(v: &Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| bad("not_object"))?;
        let mut out = NativeReportView::default();
        if let Some(cells) = obj.get("cells").filter(|c| !c.is_null()) {
            for c in cells.as_array().ok_or_else(|| bad("cells_not_array"))? {
                let co = c.as_object().ok_or_else(|| bad("cell_not_object"))?;
                let id = co
                    .get("id")
                    .and_then(as_i64)
                    .ok_or_else(|| bad("cell_id"))?;
                let state = co
                    .get("state")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let lambda_indices = match co.get("lambda_indices") {
                    Some(Value::Array(a)) => a.iter().filter_map(as_f64).collect(),
                    _ => Vec::new(),
                };
                out.cells.push(ReportCellView {
                    id,
                    state,
                    gain: opt_f64(co.get("gain")),
                    se: opt_f64(co.get("se")),
                    stable: co.get("stable").is_some_and(truthy),
                    lambda_indices,
                });
            }
        }
        out.target_stats = match obj.get("target_stats") {
            Some(Value::Array(a)) if !a.is_empty() => Some(
                a.iter()
                    .map(|x| if x.is_null() { None } else { as_f64(x) })
                    .collect(),
            ),
            _ => None,
        };
        out.rows = obj.get("rows").and_then(as_i64);
        out.features = obj.get("features").and_then(as_i64);
        out.classes = obj.get("classes").and_then(as_i64);
        out.task = obj.get("task").and_then(Value::as_str).map(str::to_string);
        if let Some(Value::Array(spec)) = obj.get("spectrum") {
            for item in spec {
                let slot = item.get("slot").and_then(as_i64).unwrap_or(-1);
                let values = match item.get("values") {
                    Some(Value::Array(a)) if !a.is_empty() => Some(
                        a.iter()
                            .map(|x| if x.is_null() { None } else { as_f64(x) })
                            .collect(),
                    ),
                    _ => None,
                };
                out.spectrum.push(SpectrumView { slot, values });
            }
        }
        Ok(out)
    }
}
