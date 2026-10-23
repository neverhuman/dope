//! Bounded CSV context selection by physical row identity, followed by fit-only transforms.
use super::manifest::{bad, check, Column, Graph, Schema, PREPROCESSING};
use crate::{levels::digest, stats::quantile_sorted, Result};
use candle_core::{Device, Tensor};
use std::{
    collections::{BTreeSet, BinaryHeap},
    path::Path,
};

pub struct Context {
    pub schema: Schema,
    pub values: Tensor,
    pub observed: Tensor,
    /// Physical data-row offsets, kept local and never included in exported code.
    pub physical_rows: Vec<usize>,
}
fn missing(value: &str) -> bool {
    matches!(value.trim(), "" | "?" | "NA" | "NaN" | "null")
}
fn number(value: &str) -> Result<f64> {
    let v = value
        .trim()
        .parse::<f64>()
        .map_err(|_| bad("learned numeric cell invalid"))?;
    check(v.is_finite(), "learned numeric cell nonfinite")?;
    Ok(v)
}

/// The input is a declared fit/context CSV, never a train/test union. Kinds refer to
/// source column order; the explicitly declared target is moved last exactly once.
pub fn read_context(
    path: &Path,
    target: &str,
    kinds: &[String],
    task: &str,
    graph: &Graph,
    seed: u64,
) -> Result<Context> {
    graph.validate()?;
    let metadata = std::fs::symlink_metadata(path)?;
    check(
        metadata.is_file() && metadata.len() <= 200000000,
        "learned input file envelope",
    )?;
    let mut csv = csv::ReaderBuilder::new()
        .flexible(false)
        .from_path(path)
        .map_err(|_| bad("learned CSV open failed"))?;
    let headers = csv
        .headers()
        .map_err(|_| bad("learned CSV header invalid"))?
        .clone();
    let unique: BTreeSet<_> = headers.iter().collect();
    check(
        unique.len() == headers.len()
            && headers.len() >= 2
            && headers.len() <= graph.max_features + 1
            && kinds.len() == headers.len()
            && kinds
                .iter()
                .all(|k| matches!(k.as_str(), "numeric" | "categorical")),
        "learned declared input schema invalid",
    )?;
    let target = headers
        .iter()
        .position(|h| h == target)
        .or_else(|| target.parse::<usize>().ok())
        .filter(|&i| i < headers.len())
        .ok_or_else(|| bad("learned declared target absent"))?;
    check(
        matches!(
            (task, kinds[target].as_str()),
            ("regression", "numeric") | ("classification", "categorical")
        ),
        "learned declared task mismatch",
    )?;
    let count = graph.max_context.min(graph.cell_cap / (headers.len() - 1));
    check(count >= 4, "learned context envelope too small")?;
    let mut selected = BinaryHeap::new();
    for (row, record) in csv.records().enumerate() {
        check(
            row < 200000 && (row + 1) * headers.len() <= 2000000,
            "learned CSV materialization envelope",
        )?;
        let record = record.map_err(|_| bad("learned CSV record invalid"))?;
        // Hash only seed/physical position: target values cannot affect context selection.
        let key = digest(&[seed.to_le_bytes(), (row as u64).to_le_bytes()].concat());
        if selected.len() < count || selected.peek().is_some_and(|(k, _, _)| key < *k) {
            if selected.len() == count {
                selected.pop();
            }
            selected.push((
                key,
                row,
                record.iter().map(str::to_owned).collect::<Vec<_>>(),
            ));
        }
    }
    let mut selected = selected.into_vec();
    selected.sort_by_key(|(_, row, _)| *row);
    check(
        selected.len() >= 4,
        "learned context has too few physical rows",
    )?;
    let order: Vec<_> = (0..headers.len())
        .filter(|&j| j != target)
        .chain([target])
        .collect();
    let mut columns = Vec::new();
    let mut mappings = Vec::new();
    for &j in &order {
        let cells: Vec<_> = selected
            .iter()
            .map(|(_, _, r)| r[j].as_str())
            .filter(|v| !missing(v))
            .collect();
        check(!cells.is_empty(), "learned context empty column")?;
        let (center, scale, labels) = if kinds[j] == "numeric" {
            let mut values = cells
                .iter()
                .map(|v| number(v))
                .collect::<Result<Vec<_>>>()?;
            values.sort_by(f64::total_cmp);
            let center = quantile_sorted(&values, 0.5);
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / values.len() as f64)
                .sqrt();
            let scale = (quantile_sorted(&values, 0.75) - quantile_sorted(&values, 0.25))
                .max(std)
                .max(1e-6);
            check(
                center.is_finite() && scale.is_finite(),
                "learned fit transform nonfinite",
            )?;
            (center, scale, Vec::new())
        } else {
            let labels = cells
                .into_iter()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            check(
                labels.len() + 2 <= graph.max_categories,
                "learned category envelope exceeded",
            )?;
            (0., 1., labels)
        };
        columns.push(Column {
            slot: if j == target { 0 } else { (j + 1) as u32 },
            kind: kinds[j].clone(),
            center,
            scale,
            categories: if kinds[j] == "categorical" {
                labels.len() + 2
            } else {
                0
            },
        });
        mappings.push(labels);
    }
    let schema = Schema {
        preprocessing: PREPROCESSING.into(),
        task: Some(task.into()),
        target_index: Some(columns.len() - 1),
        output: "positional_numeric_and_category_ids".into(),
        columns,
    };
    schema.validate(graph)?;
    let mut values = Vec::new();
    let mut observed = Vec::new();
    for (_, _, row) in &selected {
        for (i, &j) in order.iter().enumerate() {
            let present = !missing(&row[j]);
            check(present || j != target, "learned missing target unsupported")?;
            observed.push(u8::from(present) as f32);
            let value = if !present {
                0.
            } else if kinds[j] == "numeric" {
                (number(&row[j])? - schema.columns[i].center) / schema.columns[i].scale
            } else {
                (mappings[i]
                    .binary_search(&row[j])
                    .map_err(|_| bad("learned category binding invalid"))?
                    + 2) as f64
            };
            check(
                (value as f32).is_finite(),
                "learned normalized value overflow",
            )?;
            values.push(value as f32);
        }
    }
    let shape = (selected.len(), order.len());
    Ok(Context {
        schema,
        values: Tensor::from_vec(values, shape, &Device::Cpu)?,
        observed: Tensor::from_vec(observed, shape, &Device::Cpu)?,
        physical_rows: selected.into_iter().map(|(_, row, _)| row).collect(),
    })
}
