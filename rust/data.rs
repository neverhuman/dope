use std::fs::File;
use std::path::Path;

use crate::contract::{AnonymizationTier, ReleasePolicy};
use crate::error::{DopeError, Result, io_error};
use crate::model::Task;

#[derive(Clone, Debug)]
pub struct Table {
    pub rows: usize,
    pub features: usize,
    pub columns: Vec<Vec<f32>>,
    pub target: Vec<f32>,
}

impl Table {
    pub fn from_arrays(
        features: &[f32],
        target: &[f32],
        rows: usize,
        columns: usize,
        task: Task,
    ) -> Result<Self> {
        Self::from_arrays_with_policy(
            features,
            target,
            rows,
            columns,
            task,
            &ReleasePolicy::new(AnonymizationTier::L0, None, false)?,
        )
    }

    pub fn from_arrays_with_policy(
        features: &[f32],
        target: &[f32],
        rows: usize,
        columns: usize,
        task: Task,
        policy: &ReleasePolicy,
    ) -> Result<Self> {
        policy.validate()?;
        if rows == 0 || columns == 0 {
            return Err(DopeError::Data(
                "a data kernel requires at least one row and one feature".into(),
            ));
        }
        if rows.checked_mul(columns) != Some(features.len()) || target.len() != rows {
            return Err(DopeError::Data(
                "array dimensions do not match their buffers".into(),
            ));
        }
        let mut transposed = vec![Vec::with_capacity(rows); columns];
        for (row_index, row) in features.chunks_exact(columns).enumerate() {
            for (column, value) in row.iter().enumerate() {
                if value.is_infinite() {
                    return Err(DopeError::Data(format!(
                        "row {}, column {}: infinite feature; normalize input to [0,1]",
                        row_index + 1,
                        column + 1
                    )));
                }
                if policy.tier == AnonymizationTier::L3
                    && !value.is_nan()
                    && !(0.0..=1.0).contains(value)
                {
                    return Err(DopeError::Data(format!(
                        "row {}, column {}: feature outside [0,1]; normalize input before fitting",
                        row_index + 1,
                        column + 1
                    )));
                }
                transposed[column].push(if value.is_nan() {
                    f32::NAN
                } else {
                    value.clamp(0.0, 1.0)
                });
            }
        }
        let mut y = Vec::with_capacity(rows);
        for (row_index, &value) in target.iter().enumerate() {
            if !value.is_finite() {
                return Err(DopeError::Data(format!(
                    "row {}, column {}: target must be finite",
                    row_index + 1,
                    columns + 1
                )));
            }
            if policy.tier == AnonymizationTier::L3 && !(0.0..=1.0).contains(&value) {
                return Err(DopeError::Data(format!(
                    "row {}, column {}: target outside [0,1]; normalize input before fitting",
                    row_index + 1,
                    columns + 1
                )));
            }
            if policy.tier == AnonymizationTier::L3
                && task == Task::Binary
                && value != 0.0
                && value != 1.0
            {
                return Err(DopeError::Data(format!(
                    "row {}, column {}: binary target must be 0 or 1",
                    row_index + 1,
                    columns + 1
                )));
            }
            let clipped = value.clamp(0.0, 1.0);
            y.push(if task == Task::Binary {
                (clipped >= 0.5) as u8 as f32
            } else {
                clipped
            });
        }
        Ok(Self {
            rows,
            features: columns,
            columns: transposed,
            target: y,
        })
    }

    pub fn read_dataset_dir(path: &Path, task: Task) -> Result<Self> {
        Self::read_csv(&path.join("train.csv"), task)
    }

    pub fn read_dataset_dir_with_policy(
        path: &Path,
        task: Task,
        policy: &ReleasePolicy,
    ) -> Result<Self> {
        Self::read_csv_with_policy(&path.join("train.csv"), task, policy)
    }

    pub fn read_csv(path: &Path, task: Task) -> Result<Self> {
        Self::read_csv_with_policy(
            path,
            task,
            &ReleasePolicy::new(AnonymizationTier::L0, None, false)?,
        )
    }

    pub fn read_csv_with_policy(path: &Path, task: Task, policy: &ReleasePolicy) -> Result<Self> {
        let file = File::open(path).map_err(|error| io_error(path, error))?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(false)
            .from_reader(file);
        let mut flat = Vec::<f32>::new();
        let mut width = None;
        let mut rows = 0usize;
        for record in reader.records() {
            let record = record?;
            let record_width = record.len();
            if record_width < 2 {
                return Err(DopeError::Data(
                    "CSV needs at least one feature and one target".into(),
                ));
            }
            if width
                .replace(record_width)
                .is_some_and(|prior| prior != record_width)
            {
                return Err(DopeError::Data("ragged CSV rows are unsupported".into()));
            }
            for (column_index, field) in record.iter().enumerate() {
                let value = if field.trim().is_empty() || field.eq_ignore_ascii_case("nan") {
                    f32::NAN
                } else {
                    field.parse::<f32>().map_err(|_| {
                        DopeError::Data(format!(
                            "row {}, column {}: expected numeric input without a header",
                            rows + 1,
                            column_index + 1
                        ))
                    })?
                };
                flat.push(value);
            }
            rows += 1;
        }
        let width = width.ok_or_else(|| DopeError::Data("empty CSV".into()))?;
        let features = width - 1;
        let mut x = Vec::with_capacity(rows * features);
        let mut y = Vec::with_capacity(rows);
        for row in flat.chunks_exact(width) {
            x.extend_from_slice(&row[..features]);
            y.push(row[features]);
        }
        Self::from_arrays_with_policy(&x, &y, rows, features, task, policy)
    }

    pub fn completed_column(&self, column: usize, impute: f32) -> Vec<f32> {
        self.columns[column]
            .iter()
            .map(|value| if value.is_nan() { impute } else { *value })
            .collect()
    }
}
