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
//!
//! An artifact longer than the arm's budget is not sampled: its files are
//! logged `over_budget` with the artifact length, whether the engine returned
//! the artifact or refused it with a budget error (it does once its fit is
//! fail-closed, #390); the length a refusal names is kept, and a refusal that
//! names none leaves the size unmeasured. Any other fit error is `failed`.

use super::celldata::CellData;
use super::cellplan::Cell;
use super::emitlog::{EMIT_FAILED, EMIT_OVER_BUDGET};
use super::mat::Mat;
use super::seeds::{crn_seed, STRUCTURAL_SAMPLE_SEED};
use dope_core::marginal::atoms::DEFAULT_SUPPORT_K;
use dope_core::table::parse_number;
use dope_core::target::Head;
use dope_core::{
    decode_format2_program, fit_v2, DopeError, EmissionMode, EmitOptions, FitV2Options, RawTable,
    Task,
};

pub const ARM: &str = "rust_format2";
pub const ARCH_ID: &str = concat!("dope-core-format2@", env!("CARGO_PKG_VERSION"));
pub const REASON_FIT: &str = "format2_fit_failed";
pub const REASON_OVER_BUDGET: &str = "format2_over_budget";
pub const REASON_SAMPLE: &str = "format2_sample_failed";

/// Why `encode` produced no artifact: the emit status and reason, and the
/// artifact length when it is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub status: &'static str,
    pub reason: &'static str,
    pub bytes: Option<u64>,
}

fn over_budget(bytes: Option<u64>) -> Refusal {
    Refusal {
        status: EMIT_OVER_BUDGET,
        reason: REASON_OVER_BUDGET,
        bytes,
    }
}

/// The artifact length named by the engine's budget refusal
/// (`format2 artifact (<bytes> B) exceeds requested budget (<budget> B)`).
pub fn refused_size(message: &str) -> Option<u64> {
    let rest = message.strip_prefix("format2 artifact (")?;
    let (digits, _) = rest.split_once(" B)")?;
    digits.parse().ok()
}

/// The outcome of one fit against the arm's budget.
pub fn fit_outcome(
    fitted: dope_core::Result<Vec<u8>>,
    budget: Option<usize>,
) -> Result<Vec<u8>, Refusal> {
    match fitted {
        Ok(bytes) if budget.is_some_and(|cap| bytes.len() > cap) => {
            Err(over_budget(Some(bytes.len() as u64)))
        }
        Ok(bytes) => Ok(bytes),
        Err(DopeError::Budget(message)) => Err(over_budget(refused_size(&message))),
        Err(_) => Err(Refusal {
            status: EMIT_FAILED,
            reason: REASON_FIT,
            bytes: None,
        }),
    }
}
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
pub fn encode(cell: &Cell, data: &CellData, budget: Option<usize>) -> Result<Vec<u8>, Refusal> {
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
    let fitted = fit_v2(&fit_table(data), data.a.p, &options).map(|fitted| fitted.bytes);
    fit_outcome(fitted, budget)
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

    #[test]
    fn a_budget_the_artifact_cannot_meet_is_over_budget_with_its_length() {
        let refusal = encode(&cell(), &data(), Some(64)).unwrap_err();
        assert_eq!(
            (refusal.status, refusal.reason),
            (EMIT_OVER_BUDGET, REASON_OVER_BUDGET)
        );
        assert!(refusal.bytes.is_some_and(|bytes| bytes > 64), "{refusal:?}");
    }

    #[test]
    fn an_engine_budget_refusal_keeps_the_length_it_names() {
        let message = "format2 artifact (4084 B) exceeds requested budget (2048 B)";
        assert_eq!(refused_size(message), Some(4084));
        let refused =
            |message: &str| fit_outcome(Err(DopeError::Budget(message.into())), Some(2048));
        assert_eq!(refused(message).unwrap_err(), over_budget(Some(4084)));
        assert_eq!(refused("budget too small").unwrap_err(), over_budget(None));
        let other = fit_outcome(Err(DopeError::Budget(String::new())), None).unwrap_err();
        assert_eq!(other.status, EMIT_OVER_BUDGET);
        let failed = fit_outcome(Err(DopeError::Artifact("x".into())), Some(2048)).unwrap_err();
        assert_eq!(
            (failed.status, failed.reason, failed.bytes),
            (EMIT_FAILED, REASON_FIT, None)
        );
        assert_eq!(
            fit_outcome(Ok(vec![0; 2048]), Some(2048)),
            Ok(vec![0; 2048])
        );
        assert_eq!(
            fit_outcome(Ok(vec![0; 2049]), Some(2048)).unwrap_err(),
            over_budget(Some(2049))
        );
        assert_eq!(fit_outcome(Ok(vec![0; 9000]), None), Ok(vec![0; 9000]));
    }
}
