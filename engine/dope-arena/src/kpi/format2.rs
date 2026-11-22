//! `rust_format2`: the engine's Format 2 artifact as a source-free arm.
//!
//! Per cell, `dope_core::fit_v2` sees the A rows only; the synthetic rows come
//! from `decode_format2_program(bytes)` alone, so the scored object is
//! `decode(unpack(pack(encode(A))))` and `packet_bytes` is the artifact length.
//! The engine takes text tables: A is rendered with positional column names
//! (`x0..`, `y`), a missing cell as an empty field and a class as its index.
//!
//! Arm ids: `rust_format2` (the engine's default byte budget) or
//! `rust_format2_b<bytes>` for an explicit budget.

use super::celldata::CellData;
use super::cellplan::Cell;
use super::mat::Mat;
use super::seeds::{crn_seed, STRUCTURAL_SAMPLE_SEED};
use dope_core::marginal::atoms::DEFAULT_SUPPORT_K;
use dope_core::table::parse_number;
use dope_core::target::Head;
use dope_core::{
    decode_format2_program, fit_v2, EmissionMode, EmitOptions, FitV2Options, RawTable, Task,
};

pub const ARM: &str = "rust_format2";
pub const ARCH_ID: &str = concat!("dope-core-format2@", env!("CARGO_PKG_VERSION"));
pub const REASON_FIT: &str = "format2_fit_failed";
pub const REASON_SAMPLE: &str = "format2_sample_failed";
/// Smallest fit-row support the engine keeps behind a discrete level.
pub const SUPPORT_FLOOR: u64 = DEFAULT_SUPPORT_K as u64;

/// Byte budget named by an arm id: `Some(None)` for the engine default,
/// `Some(Some(bytes))` for `rust_format2_b<bytes>`, `None` for any other arm.
pub fn budget_of(arm_id: &str) -> Option<Option<usize>> {
    let rest = arm_id.strip_prefix(ARM)?;
    if rest.is_empty() {
        return Some(None);
    }
    let digits = rest.strip_prefix("_b")?;
    let plain = !digits.is_empty() && !digits.starts_with('0') && digits.len() <= 9;
    let bytes = digits.parse::<usize>().ok().filter(|_| plain)?;
    Some(Some(bytes))
}

fn cell_text(value: f32) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        String::new()
    }
}

/// The A rows as the text table the engine fits: features `x0..`, target `y` last.
fn fit_table(data: &CellData) -> RawTable {
    let a = &data.a;
    let mut headers: Vec<String> = (0..a.p).map(|col| format!("x{col}")).collect();
    headers.push("y".to_string());
    let mut columns: Vec<Vec<String>> = (0..a.p)
        .map(|col| {
            (0..a.n)
                .map(|row| cell_text(a.x[row * a.p + col]))
                .collect()
        })
        .collect();
    columns.push(a.y.iter().map(|&v| cell_text(v)).collect());
    RawTable { headers, columns }
}

/// Fit the artifact on the cell's A rows; the bytes are the whole encoding.
pub fn encode(
    cell: &Cell,
    data: &CellData,
    budget: Option<usize>,
) -> Result<Vec<u8>, &'static str> {
    let seed = crn_seed(
        &cell.panel_sha256,
        &cell.dataset_hash,
        cell.data_seed,
        STRUCTURAL_SAMPLE_SEED,
        "format2-fit",
    );
    let options = FitV2Options {
        budget,
        seed,
        task: Some(match data.head {
            Head::Regression => Task::Regression,
            _ => Task::Classification,
        }),
        ..Default::default()
    };
    fit_v2(&fit_table(data), data.a.p, &options)
        .map(|fitted| fitted.bytes)
        .map_err(|_| REASON_FIT)
}

/// Decode `packet` and draw `rows` rows; nothing but the bytes and seeds is read.
pub fn sample(
    packet: &[u8],
    cell: &Cell,
    x_cols: usize,
    rows: usize,
    sample_seed: u64,
    multiplier: u32,
) -> Result<Mat, &'static str> {
    let program = decode_format2_program(packet).map_err(|_| REASON_SAMPLE)?;
    let seed = crn_seed(
        &cell.panel_sha256,
        &cell.dataset_hash,
        cell.data_seed,
        sample_seed,
        &format!("format2-sample|m{multiplier}"),
    );
    let options = EmitOptions {
        mode: if multiplier == 1 {
            EmissionMode::Matched
        } else {
            EmissionMode::Faithful
        },
        rows,
        seed,
        ..Default::default()
    };
    let emitted = program
        .sample_with_mode(&options)
        .map_err(|_| REASON_SAMPLE)?;
    let mut text = Vec::new();
    program
        .write_raw(&emitted.samples, &mut text)
        .map_err(|_| REASON_SAMPLE)?;
    let table = RawTable::from_reader(text.as_slice()).map_err(|_| REASON_SAMPLE)?;
    if table.rows() != rows {
        return Err(REASON_SAMPLE);
    }
    // Columns are matched by their positional names; a feature the artifact
    // does not carry, and a level it cannot name as a number, read as missing.
    let column = |name: &str| table.headers.iter().position(|h| h == name);
    let target = column("y").ok_or(REASON_SAMPLE)?;
    let features: Vec<Option<usize>> = (0..x_cols).map(|col| column(&format!("x{col}"))).collect();
    let mut data = Vec::with_capacity(rows * (x_cols + 1));
    for row in 0..rows {
        for feature in &features {
            let value = feature.and_then(|at| parse_number(&table.columns[at][row]));
            data.push(value.map_or(f32::NAN, |v| v as f32));
        }
        let label = parse_number(&table.columns[target][row]).ok_or(REASON_SAMPLE)?;
        data.push(label as f32);
    }
    Mat::new(rows, x_cols, data).map_err(|_| REASON_SAMPLE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::celldata::{Rows, ScaleCard};
    use crate::kpi::seeds::KpiRng;

    fn cell() -> Cell {
        Cell {
            cell_id: "dev:h1:full:d0".into(),
            panel_sha256: "p".into(),
            dataset_hash: "h1".into(),
            table_id: 0,
            family_id: "lf-h1".into(),
            task: "binary".into(),
            n_classes: 2,
            view: "full".into(),
            cols: vec![0, 1, 2],
            data_seed: 0,
            stratum: "headline".into(),
            a_idx: Vec::new(),
            b_idx: Vec::new(),
            c_idx: Vec::new(),
            t_idx: Vec::new(),
            sample_seeds: vec![0],
            multipliers: vec![4, 1],
            status: "ok".into(),
        }
    }

    /// 400 rows: two informative features, one with missing cells, a binary target.
    fn data() -> CellData {
        let mut rng = KpiRng::new(5);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for row in 0..400 {
            let (u, v) = (rng.normal(), rng.normal());
            x.push((40.0 + 9.0 * u) as f32);
            x.push(if row % 11 == 0 {
                f32::NAN
            } else {
                (0.5 * v) as f32
            });
            x.push((row % 3) as f32);
            y.push(f32::from(u - v + 0.5 * rng.normal() > 0.0));
        }
        let a = Rows { n: 400, p: 3, x, y };
        let empty = Rows {
            n: 0,
            p: 3,
            x: Vec::new(),
            y: Vec::new(),
        };
        CellData {
            head: Head::Binary,
            card: ScaleCard::fit(&a, Head::Binary),
            b: empty.clone(),
            c: empty.clone(),
            t: empty,
            t_donors: None,
            a,
        }
    }

    #[test]
    fn arm_ids_name_their_budget() {
        assert_eq!(budget_of("rust_format2"), Some(None));
        assert_eq!(budget_of("rust_format2_b2048"), Some(Some(2048)));
        for other in [
            "rust_format2_b",
            "rust_format2_b02",
            "rust_format2x",
            "copy_a",
            "rust_format2_b-4",
        ] {
            assert_eq!(budget_of(other), None, "{other}");
        }
    }

    #[test]
    fn the_fit_table_carries_values_and_no_names() {
        let table = fit_table(&data());
        assert_eq!(table.headers, ["x0", "x1", "x2", "y"]);
        assert_eq!((table.rows(), table.width()), (400, 4));
        assert_eq!(table.columns[1][0], "", "a missing cell is an empty field");
        assert_eq!(table.columns[2][4], "1");
        assert!(table.columns[3].iter().all(|c| c == "0" || c == "1"));
        assert_eq!(cell_text(0.1), "0.1");
        assert_eq!(cell_text(f32::INFINITY), "");
    }

    #[test]
    fn rows_are_decoded_from_the_packet_alone() {
        let (cell, data) = (cell(), data());
        let packet = encode(&cell, &data, Some(4096)).unwrap();
        assert!(
            !packet.is_empty() && packet.len() <= 4096,
            "{} bytes",
            packet.len()
        );
        assert_eq!(encode(&cell, &data, Some(4096)).unwrap(), packet);
        let mat = sample(&packet, &cell, 3, 800, 0, 4).unwrap();
        assert_eq!((mat.rows, mat.x_cols), (800, 3));
        let bits = |mat: &Mat| mat.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let again = sample(&packet, &cell, 3, 800, 0, 4).unwrap();
        assert_eq!(bits(&again), bits(&mat), "same packet and seed, same rows");
        let labels: Vec<f32> = (0..800).map(|r| mat.row(r).unwrap()[3]).collect();
        assert!(labels.iter().all(|&v| v == 0.0 || v == 1.0));
        let ones = labels.iter().filter(|&&v| v == 1.0).count();
        assert!((200..600).contains(&ones), "{ones} positive labels of 800");
        let mean_x0 = (0..800)
            .map(|r| f64::from(mat.row(r).unwrap()[0]))
            .sum::<f64>()
            / 800.0;
        assert!(
            (mean_x0 - 40.0).abs() < 3.0,
            "raw units are kept: {mean_x0}"
        );
        let other = sample(&packet, &cell, 3, 800, 1, 4).unwrap();
        assert_ne!(bits(&other), bits(&mat), "another sample seed, other rows");
        assert_eq!(
            sample(&packet[..packet.len() / 2], &cell, 3, 8, 0, 4),
            Err(REASON_SAMPLE)
        );
    }
}
