//! One schedule cell through an external generator (`dope_kpi emit-cli`).
//!
//! Fit: the A rows of the cell are written to a private scratch file, the fit
//! command runs, and the scratch file is removed as soon as the command
//! returns, whatever its outcome. Sample: one command per (sample seed,
//! multiplier); its output is checked, stored as `DOPEMAT1` and removed.
//! Every attempt becomes exactly one `emit.jsonl` row (formats.md section 6).
//! Rows and messages built here never contain a cell value or child output.

use super::cellplan::Cell;
use super::ext_arm::{Arm, ArmIo, ArtifactClass, ARCH_ID};
use super::ext_proc::{run_logged, Exit};
use super::ext_reason::Reason;
use super::ext_rows::{fit_rows_bytes, read_output, FitView, OutShape};
use super::ext_scratch::{measure_packet, remove_path, write_private, Scratch};
use super::ext_template::{Phase, Values};
use super::io::Arena;
use super::jsonl::file_error;
use super::mat::{mat_relpath, write_mat};
use super::seeds::crn_seed;
use dope_arena::{ArenaError, ArenaResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Per-cell command logs, under `<run_dir>/<arm_id>/`. Never read back.
pub const LOG_DIR: &str = "_logs";
/// Scratch space of running cells; empty or absent after a run.
pub const WORK_DIR: &str = "_work";
/// Packets (and cards) kept for the scorer and for audits.
pub const PACKET_DIR: &str = "packets";
pub const PURPOSE_FIT: &str = "ext-fit";
pub const PURPOSE_SAMPLE: &str = "ext-sample";
pub const STATUS_OK: &str = "ok";

/// One line of `emit.jsonl`. `encode_ms` and `decode_ms` are wall-clock
/// measurements and the only fields that differ between identical runs.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EmitRow {
    pub cell_id: String,
    pub sample_seed: u64,
    pub multiplier: u32,
    pub arm_id: String,
    pub arch_id: String,
    pub weights_sha256: String,
    pub artifact_class: String,
    pub packet_bytes: u64,
    pub packet_sha256: String,
    pub cold_bytes: u64,
    pub rows: u64,
    pub file: String,
    pub status: String,
    pub reason: String,
    pub encode_ms: f64,
    pub decode_ms: f64,
}

/// What the fit of a cell left behind, shared by all rows of the cell.
#[derive(Clone, Debug, Default)]
struct Fitted {
    packet_bytes: u64,
    packet_sha256: String,
    encode_ms: f64,
}

pub struct CellJob<'a> {
    pub arena: &'a Arena,
    pub arm: &'a Arm,
    /// Absolute run directory; the `file` of a row is relative to it.
    pub run_dir: &'a Path,
    /// Limit of each fit and each sample command.
    pub timeout: Duration,
}

/// File-name stem of a cell: its id with `:` replaced by `__`.
pub fn file_stem(cell_id: &str) -> String {
    cell_id.replace(':', "__")
}

/// The (sample seed, multiplier) pairs of a cell, in schedule order.
pub fn cell_keys(cell: &Cell) -> Vec<(u64, u32)> {
    let pairs = cell.sample_seeds.iter().flat_map(|&seed| {
        cell.multipliers
            .iter()
            .map(move |&multiplier| (seed, multiplier))
    });
    pairs.collect()
}

/// Synthetic rows of one file: `multiplier * len(a_idx)`.
pub fn requested_rows(cell: &Cell, multiplier: u32) -> u64 {
    (cell.a_idx.len() as u64).saturating_mul(u64::from(multiplier))
}

/// 31-bit command seed: the low 31 bits of the section 4 stream seed.
fn command_seed(cell: &Cell, sample_seed: u64, purpose: &str) -> u32 {
    let seed = crn_seed(
        &cell.panel_sha256,
        &cell.dataset_hash,
        cell.data_seed,
        sample_seed,
        purpose,
    );
    (seed & 0x7fff_ffff) as u32
}

fn utf8(path: &Path) -> ArenaResult<&str> {
    path.to_str()
        .ok_or_else(|| ArenaError::Contract("run directory path is not UTF-8".into()))
}

impl CellJob<'_> {
    pub fn arm_dir(&self) -> PathBuf {
        self.run_dir.join(&self.arm.arm_id)
    }

    fn row(
        &self,
        cell: &Cell,
        key: (u64, u32),
        fitted: &Fitted,
        decode_ms: f64,
        outcome: Result<u64, Reason>,
    ) -> EmitRow {
        let (sample_seed, multiplier) = key;
        let (status, reason, rows, file) = match outcome {
            Ok(rows) => {
                let file = mat_relpath(&self.arm.arm_id, &cell.cell_id, sample_seed, multiplier);
                (STATUS_OK, "", rows, file)
            }
            Err(reason) => (reason.status(), reason.code(), 0, String::new()),
        };
        EmitRow {
            cell_id: cell.cell_id.clone(),
            sample_seed,
            multiplier,
            arm_id: self.arm.arm_id.clone(),
            arch_id: ARCH_ID.to_string(),
            weights_sha256: self.arm.weights_sha256.clone(),
            artifact_class: self.arm.artifact_class.name().to_string(),
            packet_bytes: fitted.packet_bytes,
            packet_sha256: fitted.packet_sha256.clone(),
            cold_bytes: self.arm.cold_bytes,
            rows,
            file,
            status: status.to_string(),
            reason: reason.to_string(),
            encode_ms: fitted.encode_ms,
            decode_ms,
        }
    }

    /// The rows of a cell for which no sample was attempted, all with `reason`.
    pub fn rows_without_samples(
        &self,
        cell: &Cell,
        reason: Reason,
        encode_ms: f64,
    ) -> Vec<EmitRow> {
        let fitted = Fitted {
            encode_ms,
            ..Fitted::default()
        };
        cell_keys(cell)
            .into_iter()
            .map(|key| self.row(cell, key, &fitted, 0.0, Err(reason)))
            .collect()
    }

    /// Absolute path of one emitted matrix (formats.md section 5).
    pub fn mat_path(&self, cell: &Cell, key: (u64, u32)) -> PathBuf {
        self.run_dir
            .join(mat_relpath(&self.arm.arm_id, &cell.cell_id, key.0, key.1))
    }

    /// Fit and sample one supported cell. An error means the run itself broke
    /// (arena, disk); every failure of the generator is a row.
    pub fn run(&self, cell: &Cell) -> ArenaResult<Vec<EmitRow>> {
        let stem = file_stem(&cell.cell_id);
        let work = self.arm_dir().join(WORK_DIR).join(&stem);
        remove_path(&work)?;
        std::fs::create_dir_all(&work).map_err(|e| file_error(&work, &e))?;
        let rows = self.run_in(cell, &stem, &work);
        // Also removes whatever the commands left in their working directory.
        let cleaned = remove_path(&work);
        let rows = rows?;
        cleaned?;
        Ok(rows)
    }

    fn run_in(&self, cell: &Cell, stem: &str, work: &Path) -> ArenaResult<Vec<EmitRow>> {
        let arm_dir = self.arm_dir();
        let log = arm_dir.join(LOG_DIR).join(format!("{stem}.log"));
        let packet = arm_dir.join(PACKET_DIR).join(format!("{stem}.packet"));
        let card = arm_dir.join(PACKET_DIR).join(format!("{stem}.card"));
        let keys = cell_keys(cell);
        // Nothing of an earlier attempt at this cell may pass for a result of this one.
        for stale in [&log, &packet, &card] {
            remove_path(stale)?;
        }
        for &key in &keys {
            remove_path(&self.mat_path(cell, key))?;
        }

        let io = self.arm.io;
        let classes = if cell.task == "regression" {
            0
        } else {
            cell.n_classes
        };
        let fit_rows = Scratch(work.join(format!("fit_rows.{}", io.name())));
        {
            let table = self.arena.read_table(cell.table_id)?;
            let view = FitView {
                table: &table,
                cols: &cell.cols,
                rows: &cell.a_idx,
                classes,
            };
            write_private(&fit_rows.0, &fit_rows_bytes(io, &view)?)?;
        }
        let out = Scratch(work.join(format!("out.{}", io.name())));
        let mut values = Values {
            bin: &self.arm.binary,
            fit_rows: utf8(&fit_rows.0)?,
            packet: utf8(&packet)?,
            card: utf8(&card)?,
            out: utf8(&out.0)?,
            rows: 0,
            seed: 0,
            fit_seed: command_seed(cell, 0, PURPOSE_FIT),
            task: &cell.task,
            classes,
            budget: self.arm.budget_bytes,
        };
        let fit = run_logged(
            &self.arm.command(Phase::Fit, &values),
            work,
            &log,
            "fit",
            self.timeout,
        );
        // The real rows leave the disk before anything else is looked at.
        fit_rows.remove()?;
        let fit = fit?;
        let failure = match fit.exit {
            Exit::SpawnFailed => Some(Reason::FitSpawn),
            Exit::Failed => Some(Reason::FitExit),
            Exit::TimedOut => Some(Reason::FitTimeout),
            Exit::Success if packet.is_file() => None,
            Exit::Success => Some(Reason::PacketMissing),
        };
        if let Some(reason) = failure {
            // A half-written packet of a failed fit is not kept.
            remove_path(&packet)?;
            remove_path(&card)?;
            return Ok(self.rows_without_samples(cell, reason, fit.wall_ms));
        }
        let mut fitted = Fitted {
            encode_ms: fit.wall_ms,
            ..Fitted::default()
        };
        // formats.md section 6: only a source-free arm has an encoding to measure.
        if self.arm.artifact_class == ArtifactClass::SourceFree {
            (fitted.packet_bytes, fitted.packet_sha256) = measure_packet(&packet, &card)?;
        }

        let mut rows = Vec::with_capacity(keys.len());
        for key in keys {
            let (sample_seed, multiplier) = key;
            values.rows = requested_rows(cell, multiplier);
            values.seed = command_seed(cell, sample_seed, PURPOSE_SAMPLE);
            out.remove()?;
            let sample = run_logged(
                &self.arm.command(Phase::Sample, &values),
                work,
                &log,
                &format!("sample s{sample_seed} m{multiplier}"),
                self.timeout,
            );
            let shape = OutShape {
                features: cell.cols.len(),
                classes,
                rows: values.rows as usize,
            };
            let outcome = sample.and_then(|ran| {
                let kept =
                    self.keep_output(io, ran.exit, &out.0, &shape, &self.mat_path(cell, key));
                kept.map(|outcome| (ran.wall_ms, outcome))
            });
            out.remove()?;
            let (decode_ms, outcome) = outcome?;
            let outcome = outcome.map(|()| values.rows);
            rows.push(self.row(cell, key, &fitted, decode_ms, outcome));
        }
        Ok(rows)
    }

    /// Check the output of one sample command and store it as `DOPEMAT1` at `dest`.
    fn keep_output(
        &self,
        io: ArmIo,
        exit: Exit,
        out: &Path,
        shape: &OutShape,
        dest: &Path,
    ) -> ArenaResult<Result<(), Reason>> {
        let failure = match exit {
            Exit::SpawnFailed => Some(Reason::SampleSpawn),
            Exit::Failed => Some(Reason::SampleExit),
            Exit::TimedOut => Some(Reason::SampleTimeout),
            Exit::Success => None,
        };
        if let Some(reason) = failure {
            return Ok(Err(reason));
        }
        let length = match std::fs::metadata(out) {
            Ok(meta) if meta.is_file() => meta.len(),
            _ => return Ok(Err(Reason::OutputMissing)),
        };
        if length > shape.max_bytes() {
            return Ok(Err(Reason::OutputTooLarge));
        }
        let Ok(bytes) = std::fs::read(out) else {
            return Ok(Err(Reason::OutputMissing));
        };
        match read_output(io, &bytes, shape) {
            Ok(mat) => write_mat(dest, &mat).map(Ok),
            Err(reason) => Ok(Err(reason)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::cellplan::{MULTIPLIERS, STATUS_UNSUPPORTED};
    use crate::kpi::jsonl::canonical_json;

    fn cell() -> Cell {
        Cell {
            cell_id: "dev:00000000000000aa:p48:d1".into(),
            panel_sha256: "ab".repeat(32),
            dataset_hash: "00000000000000aa".into(),
            table_id: 0,
            family_id: "lf-00000000000000aa".into(),
            task: "binary".into(),
            n_classes: 2,
            view: "p48".into(),
            cols: vec![0, 2],
            data_seed: 1,
            stratum: "small_n".into(),
            a_idx: vec![4, 9, 11],
            b_idx: vec![1, 2, 3],
            c_idx: Vec::new(),
            t_idx: vec![0],
            sample_seeds: vec![5, 3],
            multipliers: MULTIPLIERS.to_vec(),
            status: STATUS_UNSUPPORTED.into(),
        }
    }

    #[test]
    fn keys_rows_stems_and_seeds_follow_the_schedule() {
        let cell = cell();
        assert_eq!(cell_keys(&cell), [(5, 4), (5, 1), (3, 4), (3, 1)]);
        assert_eq!(requested_rows(&cell, 4), 12);
        assert_eq!(file_stem(&cell.cell_id), "dev__00000000000000aa__p48__d1");
        let full = crn_seed(&cell.panel_sha256, &cell.dataset_hash, 1, 3, "ext-sample");
        let seed = command_seed(&cell, 3, PURPOSE_SAMPLE);
        assert_eq!(u64::from(seed), full % (1 << 31));
        assert!(seed < 1 << 31);
        assert_ne!(seed, command_seed(&cell, 5, PURPOSE_SAMPLE));
        assert_ne!(seed, command_seed(&cell, 3, PURPOSE_FIT));
        assert_eq!((PURPOSE_FIT, PURPOSE_SAMPLE), ("ext-fit", "ext-sample"));
    }

    #[test]
    fn emit_rows_serialise_every_field_of_the_format() {
        let row = EmitRow {
            cell_id: "dev:h:full:d0".into(),
            sample_seed: 1,
            multiplier: 4,
            arm_id: "gen".into(),
            arch_id: ARCH_ID.into(),
            weights_sha256: "w".into(),
            artifact_class: "source_free".into(),
            packet_bytes: 1500,
            packet_sha256: "p".into(),
            cold_bytes: 9,
            rows: 64,
            file: "gen/dev__h__full__d0.s1.m4.mat".into(),
            status: STATUS_OK.into(),
            reason: String::new(),
            encode_ms: 12.5,
            decode_ms: 0.25,
        };
        let line = canonical_json(&row).unwrap();
        assert_eq!(
            line,
            "{\"arch_id\":\"ext-cli/v1\",\"arm_id\":\"gen\",\"artifact_class\":\"source_free\",\
             \"cell_id\":\"dev:h:full:d0\",\"cold_bytes\":9,\"decode_ms\":0.25,\"encode_ms\":12.5,\
             \"file\":\"gen/dev__h__full__d0.s1.m4.mat\",\"multiplier\":4,\"packet_bytes\":1500,\
             \"packet_sha256\":\"p\",\"reason\":\"\",\"rows\":64,\"sample_seed\":1,\"status\":\"ok\",\
             \"weights_sha256\":\"w\"}"
        );
        assert_eq!(serde_json::from_str::<EmitRow>(&line).unwrap(), row);
    }
}
