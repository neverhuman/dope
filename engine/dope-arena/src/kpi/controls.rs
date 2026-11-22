//! Native control arms, written as `DOPEMAT1` files plus an emission log and
//! then scored through the same reader as any external arm.
//!
//! | arm | synthetic table |
//! |---|---|
//! | `twin_c` | the cell's C rows (cells with at least 16 of them) |
//! | `copy_a` | the A rows themselves |
//! | `bootstrap_a` | `multiplier x n_A` rows of A drawn with replacement |
//! | `indep_marginals` | every A column permuted on its own, target included |
//! | `label_shuffle` | A with only the target permuted |
//!
//! `rust_format2[_b<bytes>]` (see `format2.rs`) is generated here too, as a
//! source-free arm with a measured packet.
//!
//! Streams follow formats.md section 4. A column's permutation is keyed by
//! its arena column index, so the `full` and `p48` views of a table agree on
//! every column they share. Only `bootstrap_a` grows with the multiplier.

use super::celldata::{CellData, Rows};
use super::cellplan::Cell;
use super::emitlog::{
    EmitRow, CLASS_CONTROL, CLASS_SOURCE_FREE, EMIT_FAILED, EMIT_OK, EMIT_UNSUPPORTED,
};
use super::format2;
use super::io::Arena;
use super::jsonl::sha256_hex;
use super::mat::{mat_relpath, write_mat, Mat};
use super::roles::MIN_ROLE_ROWS;
use super::seeds::{crn_seed, KpiRng};
use dope_arena::{ArenaError, ArenaResult};
use std::path::Path;

pub const TWIN_C: &str = "twin_c";
pub const COPY_A: &str = "copy_a";
pub const BOOTSTRAP_A: &str = "bootstrap_a";
pub const INDEP_MARGINALS: &str = "indep_marginals";
pub const LABEL_SHUFFLE: &str = "label_shuffle";
pub const ARMS: [&str; 5] = [TWIN_C, COPY_A, BOOTSTRAP_A, INDEP_MARGINALS, LABEL_SHUFFLE];
pub const ARCH_ID: &str = "native-control";
pub const REASON_NO_TWIN: &str = "c_rows_below_minimum";

/// Arms this binary generates itself: the controls and the engine's Format 2 arm.
pub fn is_native(arm_id: &str) -> bool {
    ARMS.contains(&arm_id) || format2::budget_of(arm_id).is_some()
}

fn permutation(cell: &Cell, sample_seed: u64, purpose: &str, n: usize) -> Vec<usize> {
    let seed = crn_seed(
        &cell.panel_sha256,
        &cell.dataset_hash,
        cell.data_seed,
        sample_seed,
        purpose,
    );
    let mut order: Vec<usize> = (0..n).collect();
    KpiRng::new(seed).shuffle(&mut order);
    order
}

/// A with its target column reordered by `order`.
fn with_targets(a: &Rows, order: &[usize]) -> Rows {
    Rows {
        y: order.iter().map(|&r| a.y[r]).collect(),
        ..a.clone()
    }
}

/// The synthetic table of one control arm, or the reason the cell has none.
pub fn generate(
    arm_id: &str,
    cell: &Cell,
    data: &CellData,
    sample_seed: u64,
    multiplier: u32,
) -> ArenaResult<Result<Mat, &'static str>> {
    let a = &data.a;
    let all: Vec<usize> = (0..a.n).collect();
    let mat = match arm_id {
        TWIN_C => {
            if data.c.n < MIN_ROLE_ROWS {
                return Ok(Err(REASON_NO_TWIN));
            }
            data.c.to_mat(&(0..data.c.n).collect::<Vec<_>>())?
        }
        COPY_A => a.to_mat(&all)?,
        BOOTSTRAP_A => {
            let seed = crn_seed(
                &cell.panel_sha256,
                &cell.dataset_hash,
                cell.data_seed,
                sample_seed,
                &format!("bootstrap|m{multiplier}"),
            );
            let mut rng = KpiRng::new(seed);
            let picks: Vec<usize> = (0..a.n * multiplier as usize)
                .map(|_| rng.below(a.n))
                .collect();
            a.to_mat(&picks)?
        }
        LABEL_SHUFFLE => {
            with_targets(a, &permutation(cell, sample_seed, "label", a.n)).to_mat(&all)?
        }
        INDEP_MARGINALS => {
            let mut shuffled = with_targets(a, &permutation(cell, sample_seed, "label", a.n));
            for (position, &col) in cell.cols.iter().enumerate() {
                let order = permutation(cell, sample_seed, &format!("indep|col{col}"), a.n);
                for (row, &from) in order.iter().enumerate() {
                    shuffled.x[row * a.p + position] = a.x[from * a.p + position];
                }
            }
            shuffled.to_mat(&all)?
        }
        _ => return Err(ArenaError::Contract("unknown control arm".into())),
    };
    Ok(Ok(mat))
}

/// How one scheduled file ended: written, or not produced for a named reason.
enum Outcome<'a> {
    Written {
        rows: u64,
        file: String,
    },
    Refused {
        status: &'static str,
        reason: &'a str,
    },
}

fn emit_row(
    arm_id: &str,
    cell: &Cell,
    (sample_seed, multiplier): (u64, u32),
    outcome: Outcome,
    packet: Option<&[u8]>,
) -> EmitRow {
    let (status, reason, rows, file) = match outcome {
        Outcome::Written { rows, file } => (EMIT_OK, "", rows, file),
        Outcome::Refused { status, reason } => (status, reason, 0, String::new()),
    };
    // A Format 2 arm is source free: its rows are decoded from the packet alone.
    let format2 = format2::budget_of(arm_id).is_some();
    EmitRow {
        cell_id: cell.cell_id.clone(),
        sample_seed,
        multiplier,
        arm_id: arm_id.to_string(),
        arch_id: if format2 { format2::ARCH_ID } else { ARCH_ID }.to_string(),
        weights_sha256: String::new(),
        artifact_class: if format2 {
            CLASS_SOURCE_FREE
        } else {
            CLASS_CONTROL
        }
        .to_string(),
        packet_bytes: packet.map_or(0, |bytes| bytes.len() as u64),
        packet_sha256: packet.map_or_else(String::new, sha256_hex),
        cold_bytes: 0,
        rows,
        file,
        status: status.to_string(),
        reason: reason.to_string(),
        encode_ms: 0.0,
        decode_ms: 0.0,
        min_emitted_support: format2.then_some(format2::SUPPORT_FLOOR),
    }
}

/// Write every scheduled file of one cell and return its log rows.
fn emit_cell(
    arena: &Arena,
    cell: &Cell,
    arm_id: &str,
    emit_dir: &Path,
) -> ArenaResult<Vec<EmitRow>> {
    let data = CellData::load(arena, cell)?;
    // One Format 2 artifact per cell, fitted on A; each file is decoded from it.
    let packet = match (&data, format2::budget_of(arm_id)) {
        (Ok(data), Some(budget)) => Some(format2::encode(cell, data, budget)),
        _ => None,
    };
    let bytes = packet.as_ref().and_then(|p| p.as_deref().ok());
    let mut rows = Vec::new();
    for &sample_seed in &cell.sample_seeds {
        for &multiplier in &cell.multipliers {
            let refused = |status, reason| Err(Outcome::Refused { status, reason });
            let generated = match (&data, &packet) {
                (Err(reason), _) => refused(EMIT_UNSUPPORTED, reason),
                (Ok(_), Some(Err(reason))) => refused(EMIT_FAILED, reason),
                (Ok(data), Some(Ok(packet))) => {
                    let count = data.a.n * multiplier as usize;
                    format2::sample(packet, cell, data.a.p, count, sample_seed, multiplier)
                        .or_else(|reason| refused(EMIT_FAILED, reason))
                }
                (Ok(data), None) => generate(arm_id, cell, data, sample_seed, multiplier)?
                    .or_else(|reason| refused(EMIT_UNSUPPORTED, reason)),
            };
            let outcome = match generated {
                Ok(mat) => {
                    let file = mat_relpath(arm_id, &cell.cell_id, sample_seed, multiplier);
                    write_mat(&emit_dir.join(&file), &mat)?;
                    Outcome::Written {
                        rows: mat.rows as u64,
                        file,
                    }
                }
                Err(outcome) => outcome,
            };
            let key = (sample_seed, multiplier);
            rows.push(emit_row(arm_id, cell, key, outcome, bytes));
        }
    }
    Ok(rows)
}

/// Generate a native arm for the cells at `positions`: files under
/// `<emit-dir>/<arm>/` and one log row per scheduled file.
pub fn emit_cells(
    arena: &Arena,
    cells: &[Cell],
    positions: &[usize],
    arm_id: &str,
    emit_dir: &Path,
) -> ArenaResult<Vec<EmitRow>> {
    let mut rows = Vec::new();
    for &position in positions {
        rows.extend(emit_cell(arena, &cells[position], arm_id, emit_dir)?);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::celldata::ScaleCard;
    use dope_core::target::Head;
    use std::collections::BTreeMap;

    fn cell(cols: Vec<usize>) -> Cell {
        Cell {
            cell_id: "dev:h1:full:d0".into(),
            panel_sha256: "p".into(),
            dataset_hash: "h1".into(),
            table_id: 0,
            family_id: "lf-h1".into(),
            task: "regression".into(),
            n_classes: 0,
            view: "full".into(),
            cols,
            data_seed: 0,
            stratum: "small_n".into(),
            a_idx: (0..40).collect(),
            b_idx: Vec::new(),
            c_idx: Vec::new(),
            t_idx: Vec::new(),
            sample_seeds: vec![0],
            multipliers: vec![4, 1],
            status: "ok".into(),
        }
    }

    /// 40 A rows whose every value is unique: column `j` of row `r` is `100 j + r`.
    fn data(cols: usize, c_rows: usize) -> CellData {
        let table = |n: usize, offset: f32| Rows {
            n,
            p: cols,
            x: (0..n * cols)
                .map(|i| offset + (100 * (i % cols) + i / cols) as f32)
                .collect(),
            y: (0..n).map(|r| offset + 9000.0 + r as f32).collect(),
        };
        let a = table(40, 0.0);
        CellData {
            head: Head::Regression,
            card: ScaleCard::fit(&a, Head::Regression),
            b: table(40, 0.5),
            c: table(c_rows, 0.25),
            t: table(20, 0.75),
            t_donors: None,
            a,
        }
    }

    fn column(mat: &Mat, col: usize) -> Vec<u32> {
        (0..mat.rows)
            .map(|r| mat.row(r).unwrap()[col].to_bits())
            .collect()
    }

    fn sorted(mut values: Vec<u32>) -> Vec<u32> {
        values.sort_unstable();
        values
    }

    #[test]
    fn each_arm_builds_the_table_it_names() {
        let (cell, data) = (cell(vec![0, 1, 2]), data(3, 20));
        let make = |arm: &str, seed: u64, m: u32| generate(arm, &cell, &data, seed, m).unwrap();
        let copy = make(COPY_A, 0, 4).unwrap();
        assert_eq!(copy, data.a.to_mat(&(0..40).collect::<Vec<_>>()).unwrap());
        assert_eq!(make(COPY_A, 3, 1).unwrap(), copy);
        let twin = make(TWIN_C, 0, 4).unwrap();
        assert_eq!((twin.rows, twin.row(0).unwrap()[0]), (20, 0.25));

        // Bootstrap: m x n rows, each an A row, with repeats and misses.
        let boot = make(BOOTSTRAP_A, 0, 4).unwrap();
        assert_eq!(boot.rows, 160);
        let mut seen: BTreeMap<u32, usize> = BTreeMap::new();
        for r in 0..boot.rows {
            let row = boot.row(r).unwrap();
            let source = row[0] as usize;
            assert_eq!(row, copy.row(source).unwrap());
            *seen.entry(source as u32).or_default() += 1;
        }
        assert!(seen.len() <= 40 && seen.values().any(|&count| count >= 6));
        assert_eq!(make(BOOTSTRAP_A, 0, 1).unwrap().rows, 40);
        assert_eq!(make(BOOTSTRAP_A, 0, 4).unwrap(), boot);
        assert_ne!(make(BOOTSTRAP_A, 1, 4).unwrap(), boot);

        // Label shuffle keeps the features and permutes the target.
        let shuffle = make(LABEL_SHUFFLE, 0, 4).unwrap();
        for col in 0..3 {
            assert_eq!(column(&shuffle, col), column(&copy, col));
        }
        assert_ne!(column(&shuffle, 3), column(&copy, 3));
        assert_eq!(sorted(column(&shuffle, 3)), sorted(column(&copy, 3)));

        // Independent marginals: every column a different permutation of itself.
        let indep = make(INDEP_MARGINALS, 0, 1).unwrap();
        for col in 0..4 {
            assert_ne!(column(&indep, col), column(&copy, col), "column {col}");
            assert_eq!(sorted(column(&indep, col)), sorted(column(&copy, col)));
        }
        assert_eq!(column(&indep, 3), column(&shuffle, 3), "one label stream");
        let source = |mat: &Mat, col: usize| -> Vec<u32> {
            column(mat, col)
                .iter()
                .map(|bits| (f32::from_bits(*bits) as u32) % 100)
                .collect()
        };
        assert_ne!(source(&indep, 0), source(&indep, 1));
        assert!(generate("rust_format9", &cell, &data, 0, 1).is_err());
    }

    #[test]
    fn views_share_column_streams_and_small_twins_are_refused() {
        // A view holding arena columns 5 and 9 permutes them like any other view would.
        let (wide, narrow) = (cell(vec![2, 5, 9]), cell(vec![5, 9]));
        let full = generate(INDEP_MARGINALS, &wide, &data(3, 0), 0, 1)
            .unwrap()
            .unwrap();
        let part = generate(INDEP_MARGINALS, &narrow, &data(2, 0), 0, 1)
            .unwrap()
            .unwrap();
        let order = |mat: &Mat, col: usize| -> Vec<u32> {
            column(mat, col)
                .iter()
                .map(|bits| (f32::from_bits(*bits) as u32) % 100)
                .collect()
        };
        assert_eq!(order(&full, 1), order(&part, 0));
        assert_eq!(order(&full, 2), order(&part, 1));
        assert_eq!(
            generate(TWIN_C, &wide, &data(3, 15), 0, 4).unwrap(),
            Err(REASON_NO_TWIN)
        );
        assert!(is_native("twin_c") && !is_native("twin"));
        assert!(is_native("rust_format2_b2048"));
        let refusal = Outcome::Refused {
            status: EMIT_UNSUPPORTED,
            reason: REASON_NO_TWIN,
        };
        let refused = emit_row(TWIN_C, &wide, (0, 4), refusal, None);
        assert_eq!(
            (refused.status.as_str(), refused.rows, refused.file.as_str()),
            ("unsupported", 0, "")
        );
        let written = |arm: &str, packet| {
            let outcome = Outcome::Written {
                rows: 40,
                file: format!("{arm}/x.mat"),
            };
            emit_row(arm, &wide, (2, 1), outcome, packet)
        };
        let copy = written(COPY_A, None);
        assert_eq!(
            (copy.status.as_str(), copy.artifact_class.as_str()),
            ("ok", "control")
        );
        assert_eq!((copy.packet_bytes, copy.min_emitted_support), (0, None));
        let packed = written("rust_format2_b4096", Some(b"abc".as_slice()));
        assert_eq!(packed.artifact_class, "source_free");
        assert_eq!((packed.packet_bytes, packed.packet_sha256.len()), (3, 64));
        assert_eq!(packed.min_emitted_support, Some(10));
    }
}
