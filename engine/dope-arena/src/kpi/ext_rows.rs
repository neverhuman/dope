//! Rows exchanged with an external generator (`dope_kpi emit-cli`).
//!
//! Fit rows are the A rows of a cell in the cell's column view, features then
//! target. Text form: header `c0,...,c{p-1},target`, floats in the shortest
//! decimal form that reads back to the same float32, NaN as an empty field,
//! classification targets as integer class indices. Binary form: `DOPEMAT1`.
//! The generator's output is read back with columns matched by name and is
//! rejected, with a fixed reason code, unless it holds exactly the requested
//! rows with a usable target.

use super::ext_arm::ArmIo;
use super::ext_reason::Reason;
use super::io::TableData;
use super::mat::Mat;
use dope_arena::{ArenaError, ArenaResult};
use std::fmt::Write;

pub const TARGET_NAME: &str = "target";

/// The fit rows of one cell: `rows` of `table` restricted to `cols`.
pub struct FitView<'a> {
    pub table: &'a TableData,
    pub cols: &'a [usize],
    pub rows: &'a [usize],
    /// 0 for regression, else the number of classes.
    pub classes: u32,
}

/// Shape a generator's output must have.
#[derive(Clone, Copy, Debug)]
pub struct OutShape {
    pub features: usize,
    /// 0 for regression, else the number of classes.
    pub classes: u32,
    pub rows: usize,
}

impl OutShape {
    /// Upper bound on a valid output file: 256 bytes per value and a 4 KiB
    /// allowance. Anything longer is refused before it is read.
    pub fn max_bytes(&self) -> u64 {
        let values = (self.rows as u64).saturating_mul(self.features as u64 + 1);
        values.saturating_mul(256).saturating_add(4096)
    }
}

/// `value` as a class index when it is an integer in `0..classes`.
fn class_index(value: f32, classes: u32) -> Option<u32> {
    let in_range = value >= 0.0 && value < classes as f32 && value.fract() == 0.0;
    in_range.then_some(value as u32)
}

fn target_of(view: &FitView, row: usize) -> ArenaResult<f32> {
    let bad = |what: &str| ArenaError::Contract(format!("fit rows: {what}"));
    let target = *view
        .table
        .y
        .get(row)
        .ok_or_else(|| bad("row index outside the table"))?;
    if view.classes > 0 && class_index(target, view.classes).is_none() {
        return Err(bad("a classification target is not a class index"));
    }
    Ok(target)
}

fn feature_of(row: &[f32], col: usize) -> ArenaResult<f32> {
    row.get(col)
        .copied()
        .ok_or_else(|| ArenaError::Contract("fit rows: column index outside the table".into()))
}

fn fit_csv(view: &FitView) -> ArenaResult<String> {
    let mut out = String::with_capacity((view.rows.len() + 1) * (view.cols.len() + 1) * 10);
    for position in 0..view.cols.len() {
        let _ = write!(out, "c{position},");
    }
    out.push_str(TARGET_NAME);
    out.push('\n');
    let push = |out: &mut String, value: f32| {
        if !value.is_nan() {
            // `Display` for f32 is the shortest decimal that parses back to the same bits.
            let _ = write!(out, "{value}");
        }
    };
    for &row in view.rows {
        let features = view.table.row(row)?;
        for &col in view.cols {
            push(&mut out, feature_of(features, col)?);
            out.push(',');
        }
        let target = target_of(view, row)?;
        if view.classes > 0 {
            let _ = write!(out, "{}", target as u32);
        } else {
            push(&mut out, target);
        }
        out.push('\n');
    }
    Ok(out)
}

fn fit_mat(view: &FitView) -> ArenaResult<Vec<u8>> {
    let mut data = Vec::with_capacity(view.rows.len() * (view.cols.len() + 1));
    for &row in view.rows {
        let features = view.table.row(row)?;
        for &col in view.cols {
            data.push(feature_of(features, col)?);
        }
        data.push(target_of(view, row)?);
    }
    Mat::new(view.rows.len(), view.cols.len(), data)?.to_bytes()
}

/// The bytes written to `{fit_rows}`.
pub fn fit_rows_bytes(io: ArmIo, view: &FitView) -> ArenaResult<Vec<u8>> {
    match io {
        ArmIo::Csv => fit_csv(view).map(String::into_bytes),
        ArmIo::Mat => fit_mat(view),
    }
}

fn check_target(value: f32, classes: u32) -> Result<f32, Reason> {
    if !value.is_finite() {
        return Err(Reason::NonFiniteTarget);
    }
    if classes > 0 && class_index(value, classes).is_none() {
        return Err(Reason::LabelOutOfRange);
    }
    Ok(value)
}

/// Position of a header name in a matrix row: `c<j>` is feature `j`, `target` is last.
fn column_slot(name: &str, features: usize) -> Option<usize> {
    if name == TARGET_NAME {
        return Some(features);
    }
    let digits = name.strip_prefix('c')?;
    let index: usize = digits.parse().ok()?;
    // Only the canonical spelling: no sign, no leading zero.
    (index < features && index.to_string() == digits).then_some(index)
}

fn read_csv(bytes: &[u8], shape: &OutShape) -> Result<Mat, Reason> {
    let width = shape.features + 1;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .trim(csv::Trim::All)
        .from_reader(bytes);
    let headers = reader.headers().map_err(|_| Reason::ParseError)?;
    let mut slots = Vec::with_capacity(width);
    let mut seen = vec![false; width];
    for name in headers {
        let slot = column_slot(name, shape.features).ok_or(Reason::HeaderMismatch)?;
        if std::mem::replace(&mut seen[slot], true) {
            return Err(Reason::HeaderMismatch);
        }
        slots.push(slot);
    }
    if slots.len() != width {
        return Err(Reason::HeaderMismatch);
    }
    // A value takes at least two bytes of text, so the input bounds the allocation.
    let mut data = Vec::with_capacity(shape.rows.saturating_mul(width).min(bytes.len()));
    let mut row = vec![0.0f32; width];
    let mut record = csv::StringRecord::new();
    let mut count = 0usize;
    while reader
        .read_record(&mut record)
        .map_err(|_| Reason::ParseError)?
    {
        if count == shape.rows {
            return Err(Reason::RowCountMismatch);
        }
        if record.len() != width {
            return Err(Reason::ParseError);
        }
        for (field, &slot) in record.iter().zip(&slots) {
            row[slot] = if slot == shape.features {
                if field.is_empty() {
                    return Err(Reason::NonFiniteTarget);
                }
                let value = field.parse().map_err(|_| Reason::ParseError)?;
                check_target(value, shape.classes)?
            } else if field.is_empty() {
                f32::NAN
            } else {
                field.parse().map_err(|_| Reason::ParseError)?
            };
        }
        data.extend_from_slice(&row);
        count += 1;
    }
    if count != shape.rows {
        return Err(Reason::RowCountMismatch);
    }
    Mat::new(count, shape.features, data).map_err(|_| Reason::MatInvalid)
}

fn read_mat(bytes: &[u8], shape: &OutShape) -> Result<Mat, Reason> {
    let mat = Mat::from_bytes(bytes).map_err(|_| Reason::MatInvalid)?;
    if mat.x_cols != shape.features {
        return Err(Reason::HeaderMismatch);
    }
    if mat.rows != shape.rows {
        return Err(Reason::RowCountMismatch);
    }
    for row in mat.data.chunks_exact(mat.width()) {
        check_target(row[shape.features], shape.classes)?;
    }
    Ok(mat)
}

/// Parse the bytes a generator wrote to `{out}` into the matrix to keep.
pub fn read_output(io: ArmIo, bytes: &[u8], shape: &OutShape) -> Result<Mat, Reason> {
    match io {
        ArmIo::Csv => read_csv(bytes, shape),
        ArmIo::Mat => read_mat(bytes, shape),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::fixture::{write_arena, FixtureTable};
    use crate::kpi::io::Arena;

    const TRICKY: [f32; 16] = [
        0.1,
        -0.0,
        0.0,
        1.0,
        -3.0,
        16_777_216.0,
        1.0e-45,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        3.0e-12,
        123_456.79,
        0.333_333_34,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];

    /// A 4-row, 4-column table holding every tricky value, as `task` with `classes`.
    fn table(task: &str, classes: u32, y: Vec<f32>) -> TableData {
        let dir = crate::kpi::unit_test_dir(&format!("ext-rows-{task}"));
        let mut fixture = FixtureTable::new("00000000000000aa", "t", task, "val", 4)
            .with_rows(TRICKY.to_vec(), y);
        fixture.n_classes = classes;
        write_arena(&dir, &[fixture]).unwrap();
        let table = Arena::open(&dir).unwrap().read_table(0).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        table
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn csv_fit_rows_use_the_documented_text_and_read_back_bit_exact() {
        let table = table("regression", 0, vec![0.5, -2.0, 1.0e9, 7.0]);
        let (cols, rows) = ([0usize, 1, 3], [3usize, 0, 2]);
        let view = FitView {
            table: &table,
            cols: &cols,
            rows: &rows,
            classes: 0,
        };
        let csv = String::from_utf8(fit_rows_bytes(ArmIo::Csv, &view).unwrap()).unwrap();
        assert_eq!(
            csv,
            "c0,c1,c2,target\n\
             0.33333334,inf,,7\n\
             0.1,-0,1,0.5\n\
             340282350000000000000000000000000000000,-340282350000000000000000000000000000000,123456.79,1000000000\n"
        );
        let shape = OutShape {
            features: 3,
            classes: 0,
            rows: 3,
        };
        let back = read_output(ArmIo::Csv, csv.as_bytes(), &shape).unwrap();
        let expected = [
            [TRICKY[12], TRICKY[13], TRICKY[15], 7.0],
            [TRICKY[0], TRICKY[1], TRICKY[3], 0.5],
            [TRICKY[8], TRICKY[9], TRICKY[11], 1.0e9],
        ];
        assert_eq!(bits(&back.data), bits(&expected.concat()));
        assert_eq!((back.rows, back.x_cols, back.y_cols), (3, 3, 1));

        // Every tricky value survives the text form, including the smallest denormal.
        let all = [0usize, 1, 2, 3];
        let view = FitView {
            table: &table,
            cols: &all,
            rows: &all,
            classes: 0,
        };
        let csv = fit_rows_bytes(ArmIo::Csv, &view).unwrap();
        let shape = OutShape {
            features: 4,
            classes: 0,
            rows: 4,
        };
        let back = read_output(ArmIo::Csv, &csv, &shape).unwrap();
        let features: Vec<f32> = back.data.chunks(5).flat_map(|r| r[..4].to_vec()).collect();
        assert_eq!(bits(&features), bits(&TRICKY));

        // The binary form carries the same rows without any text.
        let mat = fit_rows_bytes(ArmIo::Mat, &view).unwrap();
        let back = read_output(ArmIo::Mat, &mat, &shape).unwrap();
        assert_eq!(back.to_bytes().unwrap(), mat);
        assert_eq!(bits(&back.data[..4]), bits(&TRICKY[..4]));
    }

    #[test]
    fn classification_targets_are_written_as_integers_and_checked() {
        let table = table("multiclass", 3, vec![2.0, 0.0, 1.0, 2.0]);
        let (cols, rows) = ([1usize], [0usize, 1, 2]);
        let view = FitView {
            table: &table,
            cols: &cols,
            rows: &rows,
            classes: 3,
        };
        let csv = String::from_utf8(fit_rows_bytes(ArmIo::Csv, &view).unwrap()).unwrap();
        assert_eq!(
            csv,
            "c0,target\n-0,2\n16777216,0\n-340282350000000000000000000000000000000,1\n"
        );
        let mat = Mat::from_bytes(&fit_rows_bytes(ArmIo::Mat, &view).unwrap()).unwrap();
        assert_eq!(mat.row(0).unwrap()[1], 2.0);

        // A table whose stored labels are not class indices is refused, without its values.
        for bad in [3.0, 1.5, -1.0, f32::NAN] {
            let table = self::table("multiclass", 3, vec![bad, 0.0, 1.0, 2.0]);
            let view = FitView {
                table: &table,
                ..view
            };
            for io in [ArmIo::Csv, ArmIo::Mat] {
                let err = fit_rows_bytes(io, &view).unwrap_err().to_string();
                assert!(err.ends_with("a classification target is not a class index"));
            }
        }
        let outside = FitView { rows: &[4], ..view };
        assert!(fit_rows_bytes(ArmIo::Csv, &outside).is_err());
        let outside = FitView { cols: &[4], ..view };
        assert!(fit_rows_bytes(ArmIo::Mat, &outside).is_err());
    }

    #[test]
    fn csv_output_is_matched_by_name_and_rejected_with_fixed_reasons() {
        let shape = |classes, rows| OutShape {
            features: 2,
            classes,
            rows,
        };
        let read = |text: &str, classes, rows| {
            read_output(ArmIo::Csv, text.as_bytes(), &shape(classes, rows))
        };
        // Any column order, quoted fields, padding, CRLF and a missing final newline.
        let mat = read("target, c1 ,c0\r\n1,\"2.5\",\r\n 0 ,nan,-7e-1", 2, 2).unwrap();
        assert!(mat.data[0].is_nan() && mat.data[4].is_nan());
        assert_eq!(mat.data[1..4], [2.5, 1.0, -0.7]);
        assert_eq!(mat.data[5], 0.0);
        assert_eq!(
            read("c0,c1,target\n1,2,1.0\n", 2, 1).unwrap().data,
            [1.0, 2.0, 1.0]
        );
        assert_eq!(read("c0,c1,target\n", 0, 0).unwrap().rows, 0);

        let expect = |text: &str, classes, rows, reason| {
            assert_eq!(read(text, classes, rows).unwrap_err(), reason, "{text:?}");
        };
        for header in [
            "",
            "c0,target",
            "c0,c1,c2,target",
            "c0,c1,target,extra",
            "c0,c0,target",
        ] {
            expect(&format!("{header}\n1,2,3\n"), 0, 1, Reason::HeaderMismatch);
        }
        for header in [
            "c0,c1,y",
            "c0,c01,target",
            "c0,C1,target",
            "c0,c+1,target",
            "c0,c2,target",
        ] {
            expect(&format!("{header}\n1,2,3\n"), 0, 1, Reason::HeaderMismatch);
        }
        expect("c0,c1,target\n1,x,3\n", 0, 1, Reason::ParseError);
        expect("c0,c1,target\n1,2,three\n", 0, 1, Reason::ParseError);
        expect("c0,c1,target\n1,2\n", 0, 1, Reason::ParseError);
        expect("c0,c1,target\n1,2,3,4\n", 0, 1, Reason::ParseError);
        expect("c0,c1,target\n1,NA,3\n", 0, 1, Reason::ParseError);
        let not_utf8 = read_output(ArmIo::Csv, b"c0,c1,target\n1,2,\xff\n", &shape(0, 1));
        assert_eq!(not_utf8.unwrap_err(), Reason::ParseError);
        expect("c0,c1,target\n1,2,3\n", 0, 2, Reason::RowCountMismatch);
        expect(
            "c0,c1,target\n1,2,3\n4,5,6\n",
            0,
            1,
            Reason::RowCountMismatch,
        );
        expect("c0,c1,target\n", 0, 1, Reason::RowCountMismatch);
        for target in ["", "nan", "inf", "-inf", "1e60"] {
            let text = format!("c0,c1,target\n1,2,{target}\n");
            expect(&text, 0, 1, Reason::NonFiniteTarget);
            expect(&text, 3, 1, Reason::NonFiniteTarget);
        }
        for target in ["3", "-1", "0.5", "1e3"] {
            let text = format!("c0,c1,target\n1,2,{target}\n");
            expect(&text, 3, 1, Reason::LabelOutOfRange);
            assert!(read(&text, 0, 1).is_ok());
        }
        // The first problem in row order decides.
        expect(
            "c0,c1,target\n1,2,9\n1,x,0\n",
            3,
            2,
            Reason::LabelOutOfRange,
        );
        expect("c0,c1,target\n1,x,0\n1,2,9\n", 3, 2, Reason::ParseError);
        assert_eq!(shape(0, 10).max_bytes(), 10 * 3 * 256 + 4096);
    }

    #[test]
    fn mat_output_is_checked_for_shape_and_target() {
        let shape = |features, classes, rows| OutShape {
            features,
            classes,
            rows,
        };
        let bytes = Mat::new(2, 2, vec![1.0, f32::NAN, 0.0, f32::INFINITY, 4.0, 1.0])
            .unwrap()
            .to_bytes()
            .unwrap();
        let read = |shape: OutShape| read_output(ArmIo::Mat, &bytes, &shape).map(|mat| mat.rows);
        assert_eq!(read(shape(2, 0, 2)), Ok(2));
        assert_eq!(read(shape(2, 2, 2)), Ok(2));
        assert_eq!(read(shape(3, 0, 2)), Err(Reason::HeaderMismatch));
        assert_eq!(read(shape(2, 0, 3)), Err(Reason::RowCountMismatch));
        assert_eq!(read(shape(2, 1, 2)), Err(Reason::LabelOutOfRange));
        let truncated = read_output(ArmIo::Mat, &bytes[..30], &shape(2, 0, 2));
        assert_eq!(truncated.unwrap_err(), Reason::MatInvalid);
        let text = read_output(ArmIo::Mat, b"c0,c1,target\n", &shape(2, 0, 0));
        assert_eq!(text.unwrap_err(), Reason::MatInvalid);
        for (target, reason) in [
            (f32::NAN, Reason::NonFiniteTarget),
            (f32::NEG_INFINITY, Reason::NonFiniteTarget),
            (0.25, Reason::LabelOutOfRange),
        ] {
            let bytes = Mat::new(1, 1, vec![5.0, target])
                .unwrap()
                .to_bytes()
                .unwrap();
            let out = read_output(ArmIo::Mat, &bytes, &shape(1, 2, 1));
            assert_eq!(out.unwrap_err(), reason);
        }
    }
}
