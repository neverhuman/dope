//! Rows of one scheduled cell and the A-only unit space every arm is scored in.
//!
//! Features are standardised with the mean and standard deviation of the cell's
//! A rows and missing cells take the A median; a regression target is
//! standardised with the A target statistics. Nothing from B, C, T or a
//! synthetic table enters the transform, so every arm sees the same units.

use super::cellplan::{Cell, STATUS_OK};
use super::io::{Arena, TableData};
use super::mat::Mat;
use dope_arena::{ArenaError, ArenaResult};
use dope_core::dvs::v3::bundle::TaskKind;
use dope_core::eval::Matrix;
use dope_core::target::Head;

/// Reasons a cell cannot be scored for any arm (decided on real rows only).
pub const REASON_SCHEDULE: &str = "schedule_unsupported";
pub const REASON_REAL_TARGET: &str = "invalid_real_target";
pub const REASON_FIT_TARGET: &str = "degenerate_fit_target";
pub const REASON_QUERY_TARGET: &str = "degenerate_constant_query_target";

/// Rows of one role (or one synthetic table) restricted to the cell's view columns.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub n: usize,
    pub p: usize,
    /// Row-major raw features, `n * p` values; NaN marks a missing cell.
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

impl Rows {
    pub fn from_table(table: &TableData, cols: &[usize], idx: &[usize]) -> ArenaResult<Self> {
        if cols.iter().any(|&c| c >= table.n_cols) {
            return Err(ArenaError::Contract(
                "schedule column outside the table".into(),
            ));
        }
        let mut x = Vec::with_capacity(idx.len() * cols.len());
        let mut y = Vec::with_capacity(idx.len());
        for &row in idx {
            let full = table.row(row)?;
            x.extend(cols.iter().map(|&c| full[c]));
            y.push(table.y[row]);
        }
        Ok(Self {
            n: idx.len(),
            p: cols.len(),
            x,
            y,
        })
    }

    pub fn from_mat(mat: &Mat) -> Self {
        let width = mat.width();
        let mut x = Vec::with_capacity(mat.rows * mat.x_cols);
        let mut y = Vec::with_capacity(mat.rows);
        for row in mat.data.chunks_exact(width.max(1)) {
            x.extend_from_slice(&row[..mat.x_cols]);
            y.push(row[mat.x_cols]);
        }
        Self {
            n: mat.rows,
            p: mat.x_cols,
            x,
            y,
        }
    }

    /// The rows listed in `picks` (repeats allowed), features first and the target last.
    pub fn to_mat(&self, picks: &[usize]) -> ArenaResult<Mat> {
        let mut data = Vec::with_capacity(picks.len() * (self.p + 1));
        for &row in picks {
            data.extend_from_slice(self.row(row));
            data.push(self.y[row]);
        }
        Mat::new(picks.len(), self.p, data)
    }

    pub fn row(&self, row: usize) -> &[f32] {
        &self.x[row * self.p..(row + 1) * self.p]
    }
}

pub fn head_of(task: &str, n_classes: u32) -> ArenaResult<Head> {
    match task {
        "regression" => Ok(Head::Regression),
        "binary" => Ok(Head::Binary),
        "multiclass" if n_classes >= 2 => Ok(Head::Multiclass(n_classes as usize)),
        _ => Err(ArenaError::Contract("unsupported task in schedule".into())),
    }
}

pub fn task_kind(head: Head) -> TaskKind {
    match head {
        Head::Regression => TaskKind::Regression,
        Head::Binary => TaskKind::Binary,
        Head::Multiclass(_) => TaskKind::Multiclass,
    }
}

/// Every target is finite and, for classification, a class index of the head.
pub fn targets_valid(head: Head, y: &[f32]) -> bool {
    y.iter().all(|&v| match head {
        Head::Regression => v.is_finite(),
        Head::Binary => v == 0.0 || v == 1.0,
        Head::Multiclass(k) => v.is_finite() && v.fract() == 0.0 && v >= 0.0 && (v as usize) < k,
    })
}

/// At least two distinct target values.
pub fn target_varies(y: &[f32]) -> bool {
    y.first().is_some_and(|first| y.iter().any(|v| v != first))
}

/// The frozen unit space of one cell, fitted on A rows only.
#[derive(Clone, Debug)]
pub struct ScaleCard {
    center: Vec<f64>,
    scale: Vec<f64>,
    /// Raw value given to a missing cell: the A median (0 when A has no value).
    fill: Vec<f64>,
    y_center: f64,
    y_scale: f64,
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    // A float32 column cannot vary by less than 1e-9 of its level unless it is constant.
    let usable = std.is_finite() && std > 0.0 && std > 1e-9 * mean.abs();
    (mean, if usable { std } else { 1.0 })
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

impl ScaleCard {
    pub fn fit(a: &Rows, head: Head) -> Self {
        let (mut center, mut scale, mut fill) = (Vec::new(), Vec::new(), Vec::new());
        let mut column = Vec::with_capacity(a.n);
        for col in 0..a.p {
            column.clear();
            column.extend(
                (0..a.n)
                    .map(|r| f64::from(a.x[r * a.p + col]))
                    .filter(|v| v.is_finite()),
            );
            let (mean, std) = mean_std(&column);
            center.push(mean);
            scale.push(std);
            fill.push(median(&mut column));
        }
        let (y_center, y_scale) = match head {
            Head::Regression => {
                let y: Vec<f64> = a.y.iter().map(|&v| f64::from(v)).collect();
                mean_std(&y)
            }
            _ => (0.0, 1.0),
        };
        Self {
            center,
            scale,
            fill,
            y_center,
            y_scale,
        }
    }

    /// Imputed, standardised features of `rows`.
    pub fn features(&self, rows: &Rows) -> Matrix {
        let mut data = Vec::with_capacity(rows.x.len());
        for row in rows.x.chunks_exact(rows.p.max(1)) {
            for (col, &raw) in row.iter().enumerate() {
                let value = if raw.is_finite() {
                    f64::from(raw)
                } else {
                    self.fill[col]
                };
                data.push((value - self.center[col]) / self.scale[col]);
            }
        }
        Matrix {
            rows: rows.n,
            cols: rows.p,
            data,
        }
    }

    /// Targets in scoring units: A-standardised regression values or class indices.
    pub fn targets(&self, rows: &Rows) -> Vec<f64> {
        rows.y
            .iter()
            .map(|&v| (f64::from(v) - self.y_center) / self.y_scale)
            .collect()
    }

    /// Feature and target of every row in unit space (`p + 1` values per row).
    pub fn unit_rows(&self, rows: &Rows) -> Vec<Vec<f64>> {
        let x = self.features(rows);
        let y = self.targets(rows);
        (0..rows.n)
            .map(|r| {
                let mut row = x.row(r).to_vec();
                row.push(y[r]);
                row
            })
            .collect()
    }
}

/// Real rows of one supported cell plus its unit space.
#[derive(Clone, Debug)]
pub struct CellData {
    pub head: Head,
    pub a: Rows,
    pub b: Rows,
    pub c: Rows,
    pub t: Rows,
    /// Duplicate-group id of each T row when a group holds more than one of them.
    pub t_donors: Option<Vec<u64>>,
    pub card: ScaleCard,
}

impl CellData {
    /// Load a cell, or name why no arm can be scored on it.
    pub fn load(arena: &Arena, cell: &Cell) -> ArenaResult<Result<Self, &'static str>> {
        let head = head_of(&cell.task, cell.n_classes)?;
        if cell.status != STATUS_OK {
            return Ok(Err(REASON_SCHEDULE));
        }
        let table = arena.read_table(cell.table_id)?;
        let role = |idx: &[usize]| Rows::from_table(&table, &cell.cols, idx);
        let (a, b, c, t) = (
            role(&cell.a_idx)?,
            role(&cell.b_idx)?,
            role(&cell.c_idx)?,
            role(&cell.t_idx)?,
        );
        if ![&a, &b, &c, &t]
            .iter()
            .all(|rows| targets_valid(head, &rows.y))
        {
            return Ok(Err(REASON_REAL_TARGET));
        }
        if !target_varies(&a.y) {
            return Ok(Err(REASON_FIT_TARGET));
        }
        if !target_varies(&t.y) {
            return Ok(Err(REASON_QUERY_TARGET));
        }
        let dup = arena.read_dup(cell.table_id)?;
        let mut groups: Vec<i32> = cell.t_idx.iter().map(|&r| dup[r]).collect();
        let donors: Vec<u64> = groups.iter().map(|&g| u64::from(g as u32)).collect();
        groups.sort_unstable();
        let repeated = groups.windows(2).any(|w| w[0] == w[1]);
        let card = ScaleCard::fit(&a, head);
        Ok(Ok(Self {
            head,
            a,
            b,
            c,
            t,
            t_donors: repeated.then_some(donors),
            card,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(p: usize, x: &[f32], y: &[f32]) -> Rows {
        Rows {
            n: y.len(),
            p,
            x: x.to_vec(),
            y: y.to_vec(),
        }
    }

    #[test]
    fn unit_space_uses_a_statistics_only() {
        // Column 0: values 1, 3, missing, 5 -> mean 3, std sqrt(8/3), median 3.
        // Column 1 is constant in A, so it is centred and left unscaled.
        let a = rows(
            2,
            &[1.0, 7.0, 3.0, 7.0, f32::NAN, 7.0, 5.0, 7.0],
            &[10.0, 20.0, 30.0, 40.0],
        );
        let card = ScaleCard::fit(&a, Head::Regression);
        let std0 = (8.0f64 / 3.0).sqrt();
        let xa = card.features(&a);
        assert!((xa.at(0, 0) + 2.0 / std0).abs() < 1e-12);
        assert_eq!(xa.at(2, 0), 0.0, "a missing cell takes the A median");
        assert_eq!(xa.at(3, 1), 0.0);
        // Other rows are mapped with the same card, whatever their own statistics.
        let other = rows(2, &[103.0, 9.0, f32::INFINITY, 7.0], &[25.0, 1025.0]);
        let xo = card.features(&other);
        assert!((xo.at(0, 0) - 100.0 / std0).abs() < 1e-9);
        assert_eq!(xo.at(0, 1), 2.0);
        assert_eq!(xo.at(1, 0), 0.0, "an infinite real cell counts as missing");
        let y_std = 125.0f64.sqrt();
        let yo = card.targets(&other);
        assert!((yo[0] - 0.0).abs() < 1e-12 && (yo[1] - 1000.0 / y_std).abs() < 1e-9);
        let unit = card.unit_rows(&other);
        assert_eq!(unit[0].len(), 3);
        assert_eq!(unit[1][2], yo[1]);
        // Classification targets stay class indices.
        let card = ScaleCard::fit(&a, Head::Multiclass(50));
        assert_eq!(card.targets(&a), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn all_missing_and_empty_columns_stay_finite() {
        let a = rows(1, &[f32::NAN, f32::NAN], &[0.0, 1.0]);
        let card = ScaleCard::fit(&a, Head::Binary);
        let x = card.features(&rows(1, &[f32::NAN, 4.0], &[1.0, 0.0]));
        assert_eq!(x.data, vec![0.0, 4.0]);
        assert_eq!(mean_std(&[]), (0.0, 1.0));
        assert_eq!(mean_std(&[1e9, 1e9, 1e9]), (1e9, 1.0));
        assert_eq!(median(&mut []), 0.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn target_checks_follow_the_head() {
        assert!(targets_valid(Head::Regression, &[1.5, -2.0]));
        assert!(!targets_valid(Head::Regression, &[1.5, f32::NAN]));
        assert!(targets_valid(Head::Binary, &[0.0, 1.0]));
        assert!(!targets_valid(Head::Binary, &[0.0, 2.0]));
        assert!(targets_valid(Head::Multiclass(3), &[0.0, 2.0]));
        assert!(!targets_valid(Head::Multiclass(3), &[3.0]));
        assert!(!targets_valid(Head::Multiclass(3), &[1.5]));
        assert!(!targets_valid(Head::Multiclass(3), &[-1.0]));
        assert!(target_varies(&[1.0, 1.0, 2.0]));
        assert!(!target_varies(&[1.0, 1.0]));
        assert!(!target_varies(&[]));
        assert!(head_of("multiclass", 1).is_err());
        assert!(head_of("ranking", 0).is_err());
        assert_eq!(head_of("multiclass", 4).unwrap(), Head::Multiclass(4));
        assert_eq!(task_kind(Head::Binary), TaskKind::Binary);
    }

    #[test]
    fn rows_round_trip_through_the_matrix_container() {
        let r = rows(2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]);
        let mat = r.to_mat(&[2, 0, 2]).unwrap();
        assert_eq!(mat.data, vec![5.0, 6.0, 9.0, 1.0, 2.0, 7.0, 5.0, 6.0, 9.0]);
        let back = Rows::from_mat(&r.to_mat(&[0, 1, 2]).unwrap());
        assert_eq!(back, r);
        assert_eq!(back.row(1), &[3.0, 4.0]);
    }
}
