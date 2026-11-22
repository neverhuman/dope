//! Summary of one `dope_kpi emit-cli` run: printed on stdout as one JSON
//! object and written to `<out>/<arm_id>/emit-summary.json`. Every count is
//! derived from the final rows of `emit.jsonl`, so the summary cannot
//! disagree with the log.

use super::ext_cell::{EmitRow, STATUS_OK};
use super::ext_reason::Reason;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Default, Serialize)]
pub struct Summary {
    pub command: &'static str,
    pub arm_id: String,
    pub arch_id: &'static str,
    pub artifact_class: &'static str,
    pub io: &'static str,
    pub weights_sha256: String,
    pub cold_bytes: u64,
    pub budget_bytes: Option<u64>,
    pub emit_file: String,
    /// Cells in the schedule file, and cells of this run (`--limit-cells`).
    pub schedule_cells: usize,
    pub limit_cells: Option<usize>,
    pub cells: usize,
    pub cells_ok: usize,
    pub cells_failed: usize,
    pub cells_unsupported: usize,
    /// Cells taken from the earlier log by `--resume`, and cells run now.
    pub cells_kept: usize,
    pub cells_run: usize,
    pub rows: usize,
    pub rows_ok: usize,
    pub rows_failed: usize,
    pub rows_unsupported: usize,
    /// Failed rows, and failed cells by their first failed row, per reason code.
    pub rows_failed_by_reason: BTreeMap<String, usize>,
    pub cells_failed_by_reason: BTreeMap<String, usize>,
    /// Over cells whose fit produced a packet (source-free arms only).
    pub packet_bytes_median: Option<f64>,
    pub packet_bytes_max: Option<u64>,
    /// Fit time over cells whose fit produced a packet; sample time over ok rows.
    pub encode_ms_median: Option<f64>,
    pub decode_ms_median: Option<f64>,
}

/// Median with the mean of the two middle values for an even count.
fn median(mut values: Vec<f64>) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(values[n / 2]),
        _ => Some((values[n / 2 - 1] + values[n / 2]) / 2.0),
    }
}

fn count(map: &mut BTreeMap<String, usize>, reason: &str) {
    *map.entry(reason.to_string()).or_default() += 1;
}

impl Summary {
    /// Fill the counts and timing statistics from the final rows, which must
    /// be sorted so that the rows of a cell are adjacent.
    pub fn tally(&mut self, rows: &[EmitRow]) {
        let (mut packets, mut encode, mut decode) = (Vec::new(), Vec::new(), Vec::new());
        for cell_rows in rows.chunk_by(|a, b| a.cell_id == b.cell_id) {
            let first_failed = cell_rows.iter().find(|row| row.status == "failed");
            if let Some(row) = first_failed {
                self.cells_failed += 1;
                count(&mut self.cells_failed_by_reason, &row.reason);
            } else if cell_rows.iter().all(|row| row.status == STATUS_OK) {
                self.cells_ok += 1;
            } else {
                self.cells_unsupported += 1;
            }
            // One packet and one fit time per cell, whatever the number of its files.
            let fitted = cell_rows.iter().find(|row| {
                row.status == STATUS_OK
                    || Reason::from_code(&row.reason).is_some_and(Reason::after_fit)
            });
            if let Some(row) = fitted {
                encode.push(row.encode_ms);
                if !row.packet_sha256.is_empty() {
                    packets.push(row.packet_bytes);
                }
            }
            for row in cell_rows {
                match row.status.as_str() {
                    STATUS_OK => {
                        self.rows_ok += 1;
                        decode.push(row.decode_ms);
                    }
                    "failed" => {
                        self.rows_failed += 1;
                        count(&mut self.rows_failed_by_reason, &row.reason);
                    }
                    _ => self.rows_unsupported += 1,
                }
            }
        }
        self.rows = rows.len();
        self.packet_bytes_max = packets.iter().copied().max();
        self.packet_bytes_median = median(packets.iter().map(|&bytes| bytes as f64).collect());
        self.encode_ms_median = median(encode);
        self.decode_ms_median = median(decode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::jsonl::canonical_json;

    fn row(
        cell: &str,
        seed: u64,
        status: &str,
        reason: &str,
        bytes: u64,
        ms: (f64, f64),
    ) -> EmitRow {
        EmitRow {
            cell_id: cell.into(),
            sample_seed: seed,
            multiplier: 1,
            arm_id: "gen".into(),
            arch_id: "ext-cli/v1".into(),
            weights_sha256: "w".into(),
            artifact_class: "source_free".into(),
            packet_bytes: bytes,
            packet_sha256: if bytes > 0 { "p".into() } else { String::new() },
            cold_bytes: 1,
            rows: 0,
            file: String::new(),
            status: status.into(),
            reason: reason.into(),
            encode_ms: ms.0,
            decode_ms: ms.1,
        }
    }

    #[test]
    fn median_uses_the_middle_or_the_mean_of_the_two_middle_values() {
        assert_eq!(median(Vec::new()), None);
        assert_eq!(median(vec![3.0]), Some(3.0));
        assert_eq!(median(vec![9.0, 1.0, 5.0]), Some(5.0));
        assert_eq!(median(vec![4.0, 1.0, 2.0, 100.0]), Some(3.0));
    }

    #[test]
    fn tally_counts_rows_and_cells_and_takes_one_packet_per_cell() {
        let rows = vec![
            // Two ok files from one 1000-byte packet.
            row("a", 0, "ok", "", 1000, (10.0, 1.0)),
            row("a", 1, "ok", "", 1000, (10.0, 3.0)),
            // The fit failed: no packet, no fit time in the statistics.
            row("b", 0, "failed", "fit_exit", 0, (99.0, 0.0)),
            row("b", 1, "failed", "fit_exit", 0, (99.0, 0.0)),
            // The packet exists; one file failed after it, one is ok.
            row("c", 0, "failed", "sample_timeout", 3000, (30.0, 500.0)),
            row("c", 1, "ok", "", 3000, (30.0, 5.0)),
            row("d", 0, "failed", "row_count_mismatch", 2000, (20.0, 7.0)),
            row("d", 1, "failed", "parse_error", 2000, (20.0, 7.0)),
            row("e", 0, "unsupported", "schedule_unsupported", 0, (0.0, 0.0)),
            row("e", 1, "unsupported", "schedule_unsupported", 0, (0.0, 0.0)),
            // Not finished: counted as failed, and not as a fit.
            row("f", 0, "failed", "run_interrupted", 0, (0.0, 0.0)),
        ];
        let mut summary = Summary {
            arm_id: "gen".into(),
            arch_id: "ext-cli/v1",
            ..Summary::default()
        };
        summary.tally(&rows);
        assert_eq!(
            (
                summary.cells_ok,
                summary.cells_failed,
                summary.cells_unsupported
            ),
            (1, 4, 1)
        );
        assert_eq!(
            (
                summary.rows,
                summary.rows_ok,
                summary.rows_failed,
                summary.rows_unsupported
            ),
            (11, 3, 6, 2)
        );
        let by = |map: &BTreeMap<String, usize>| {
            map.iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        assert_eq!(
            by(&summary.rows_failed_by_reason),
            "fit_exit=2,parse_error=1,row_count_mismatch=1,run_interrupted=1,sample_timeout=1"
        );
        assert_eq!(
            by(&summary.cells_failed_by_reason),
            "fit_exit=1,row_count_mismatch=1,run_interrupted=1,sample_timeout=1"
        );
        assert_eq!(summary.packet_bytes_median, Some(2000.0));
        assert_eq!(summary.packet_bytes_max, Some(3000));
        assert_eq!(summary.encode_ms_median, Some(20.0));
        assert_eq!(summary.decode_ms_median, Some(3.0));
        let json = canonical_json(&summary).unwrap();
        assert!(json.starts_with("{\"arch_id\":\"ext-cli/v1\",\"arm_id\":\"gen\""));
        assert!(json.contains("\"limit_cells\":null"));
    }
}
