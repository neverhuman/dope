//! Score mapped fit, synthetic, and holdout tables with `score_holdout_tables`.
//!
//! The pack carries numeric rows only. This binary does not open `test.csv`.

use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::ExitCode;

use dope_kernel::certification::{retention_lower_bound, score_holdout_tables};
use dope_kernel::data::Table;
use dope_kernel::model::Task;

const MAGIC: &[u8] = b"MFS3H1\n";
const MAX_LABEL: usize = 64;
const MAX_ROWS: usize = 100_000;
const MAX_WIDTH: usize = 256;

struct Cell {
    dataset: String,
    method: String,
    configuration: String,
    sample_seed: u32,
    fit: Table,
    synthetic: Table,
    holdout: Table,
}

fn usage() -> ExitCode {
    eprintln!("usage: mfs-v3-holdout score <pack> | lower <retention>...");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("score") => {
            let Some(path) = args.next() else {
                return usage();
            };
            if args.next().is_some() {
                return usage();
            }
            match score_file(Path::new(&path)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("holdout score failed: {error}");
                    ExitCode::from(1)
                }
            }
        }
        Some("lower") => {
            let mut values = Vec::new();
            for text in args {
                let Ok(value) = text.parse::<f64>() else {
                    return usage();
                };
                values.push(value);
            }
            match retention_lower_bound(&values) {
                Some(value) if value.is_finite() => {
                    println!("{value:.17}");
                    ExitCode::SUCCESS
                }
                _ => {
                    println!("null");
                    ExitCode::SUCCESS
                }
            }
        }
        _ => usage(),
    }
}

fn score_file(path: &Path) -> io::Result<()> {
    if path.file_name().and_then(|name| name.to_str()) == Some("test.csv") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "official test files stay sealed",
        ));
    }
    let mut input = File::open(path)?;
    let mut magic = [0_u8; 7];
    input.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unrecognized holdout pack",
        ));
    }
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut count = 0_u64;
    loop {
        match read_cell(&mut input) {
            Ok(Some(cell)) => {
                let scores = score_holdout_tables(
                    &cell.fit,
                    &cell.synthetic,
                    &cell.holdout,
                    u64::from(cell.sample_seed),
                );
                serde_json::to_writer(
                    &mut out,
                    &serde_json::json!({
                        "attribute_inference_advantage": scores.attribute_inference_advantage,
                        "configuration": cell.configuration,
                        "counts_as_dope_win": false,
                        "coverage_realism": scores.coverage_realism,
                        "dataset": cell.dataset,
                        "dependence_fidelity": scores.dependence_fidelity,
                        "driver_fidelity": scores.driver_fidelity,
                        "driver_importance": scores.driver_importance,
                        "marginal_fidelity": scores.marginal_fidelity,
                        "membership_auc": scores.membership_auc,
                        "method": cell.method,
                        "mmd_fidelity": scores.mmd_fidelity,
                        "official_tests_opened": false,
                        "sample_seed": cell.sample_seed,
                        "sliced_wasserstein_fidelity": scores.sliced_wasserstein_fidelity,
                        "superiority": null,
                    }),
                )
                .map_err(io::Error::other)?;
                out.write_all(b"\n")?;
                count += 1;
                if count % 100 == 0 {
                    eprintln!("holdout {count}");
                }
            }
            Ok(None) => break,
            Err(error) => return Err(error),
        }
    }
    out.flush()?;
    eprintln!("holdout cells {count}");
    Ok(())
}

fn read_cell(input: &mut File) -> io::Result<Option<Cell>> {
    let mut header = [0_u8; 4];
    match input.read(&mut header) {
        Ok(0) => return Ok(None),
        Ok(4) => {}
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated holdout cell",
            ));
        }
        Err(error) => return Err(error),
    }
    let fit_rows = u32::from_le_bytes(header) as usize;
    let hold_rows = read_u32(input)? as usize;
    let syn_rows = read_u32(input)? as usize;
    let width = read_u32(input)? as usize;
    if fit_rows == 0
        || hold_rows == 0
        || syn_rows == 0
        || width < 2
        || fit_rows > MAX_ROWS
        || hold_rows > MAX_ROWS
        || syn_rows > MAX_ROWS
        || width > MAX_WIDTH
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "holdout cell shape is outside the panel bound",
        ));
    }
    let dataset = read_label(input)?;
    let method = read_label(input)?;
    let configuration = read_label(input)?;
    let sample_seed = read_u32(input)?;
    let features = width - 1;
    Ok(Some(Cell {
        dataset,
        method,
        configuration,
        sample_seed,
        fit: read_table(input, fit_rows, features)?,
        holdout: read_table(input, hold_rows, features)?,
        synthetic: read_table(input, syn_rows, features)?,
    }))
}

fn read_table(input: &mut File, rows: usize, features: usize) -> io::Result<Table> {
    let width = features + 1;
    let mut bytes = vec![0_u8; rows * width * 4];
    input.read_exact(&mut bytes)?;
    let mut flat = Vec::with_capacity(rows * features);
    let mut target = Vec::with_capacity(rows);
    for row in bytes.chunks_exact(width * 4) {
        for column in 0..features {
            let start = column * 4;
            flat.push(read_unit(&row[start..start + 4])?);
        }
        let start = features * 4;
        target.push(read_unit(&row[start..start + 4])?);
    }
    Table::from_arrays(&flat, &target, rows, features, Task::Regression)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "mapped table was rejected"))
}

fn read_unit(bytes: &[u8]) -> io::Result<f32> {
    let Ok(bytes) = <[u8; 4]>::try_from(bytes) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mapped value is outside the unit interval",
        ));
    };
    let value = f32::from_le_bytes(bytes);
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mapped value is outside the unit interval",
        ))
    }
}

fn read_u32(input: &mut File) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_label(input: &mut File) -> io::Result<String> {
    let length = read_u32(input)? as usize;
    if length == 0 || length > MAX_LABEL {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "holdout label length is outside the panel bound",
        ));
    }
    let mut bytes = vec![0_u8; length];
    input.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "holdout label is not utf-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_table(rows: usize) -> Table {
        let values: Vec<f32> = (0..rows)
            .map(|index| (index as f32 + 0.5) / rows as f32)
            .collect();
        Table::from_arrays(&values, &values, rows, 1, Task::Regression).unwrap()
    }

    #[test]
    fn identical_tables_keep_full_marginal_fidelity() {
        let table = unit_table(8);
        let scores = score_holdout_tables(&table, &table, &table, 101);
        assert!((scores.marginal_fidelity - 1.0).abs() < 1e-6);
        assert!((scores.dependence_fidelity - 1.0).abs() < 1e-6);
        assert!(scores.membership_auc.is_finite());
        assert_eq!(scores.driver_importance, "absolute_target_correlation");
    }

    #[test]
    fn retention_bound_matches_the_student_fixture() {
        let value = retention_lower_bound(&[0.2, 0.5, 0.8]).unwrap();
        assert!((value + 0.0057563382544248975).abs() < 1e-9);
    }

    #[test]
    fn a_unit_interval_value_is_accepted_and_an_endpoint_stays_inside() {
        assert_eq!(read_unit(&0.5_f32.to_le_bytes()).unwrap(), 0.5);
        assert!(read_unit(&f32::NAN.to_le_bytes()).is_err());
        assert!(read_unit(&1.5_f32.to_le_bytes()).is_err());
    }
}
