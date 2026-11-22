//! `DOPEMAT1` synthetic-row container (formats.md section 5).
//!
//! 24-byte header (`"DOPEMAT1"`, rows, x_cols, y_cols, reserved, all u32
//! little-endian) followed by `rows * (x_cols + y_cols)` float32 values,
//! row-major, features first and the target last. Values are raw table units.

use super::jsonl::{read_file, write_atomic};
use dope_arena::{ArenaError, ArenaResult};
use std::path::Path;

pub const MAT_MAGIC: &[u8; 8] = b"DOPEMAT1";
const HEADER_BYTES: usize = 24;

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub x_cols: usize,
    pub y_cols: usize,
    /// Row-major, `rows * (x_cols + y_cols)` values; each row ends with its target.
    pub data: Vec<f32>,
}

fn contract(message: &str) -> ArenaError {
    ArenaError::Contract(format!("DOPEMAT1: {message}"))
}

impl Mat {
    /// Wrap row-major values with a single trailing target column.
    pub fn new(rows: usize, x_cols: usize, data: Vec<f32>) -> ArenaResult<Self> {
        let mat = Self {
            rows,
            x_cols,
            y_cols: 1,
            data,
        };
        mat.check_shape()?;
        Ok(mat)
    }

    pub fn width(&self) -> usize {
        self.x_cols + self.y_cols
    }

    /// Features and target of one row.
    pub fn row(&self, row: usize) -> ArenaResult<&[f32]> {
        if row >= self.rows {
            return Err(contract("row index outside the matrix"));
        }
        let width = self.width();
        Ok(&self.data[row * width..(row + 1) * width])
    }

    fn check_shape(&self) -> ArenaResult<()> {
        if self.y_cols != 1 {
            return Err(contract("y_cols must be 1"));
        }
        let cells = self.rows.checked_mul(self.width());
        if cells != Some(self.data.len()) {
            return Err(contract("value count disagrees with the header"));
        }
        if [self.rows, self.x_cols]
            .iter()
            .any(|&n| n > u32::MAX as usize)
        {
            return Err(contract("dimension exceeds the u32 header field"));
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> ArenaResult<Vec<u8>> {
        self.check_shape()?;
        let mut out = Vec::with_capacity(HEADER_BYTES + self.data.len() * 4);
        out.extend_from_slice(MAT_MAGIC);
        for field in [self.rows, self.x_cols, self.y_cols, 0] {
            out.extend_from_slice(&(field as u32).to_le_bytes());
        }
        for value in &self.data {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> ArenaResult<Self> {
        if bytes.len() < HEADER_BYTES || &bytes[..8] != MAT_MAGIC {
            return Err(contract("missing magic or truncated header"));
        }
        let field = |index: usize| {
            let at = 8 + index * 4;
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
        };
        let (rows, x_cols, y_cols, reserved) = (field(0), field(1), field(2), field(3));
        if reserved != 0 {
            return Err(contract("reserved header field must be 0"));
        }
        if y_cols != 1 {
            return Err(contract("y_cols must be 1"));
        }
        let payload = rows
            .checked_mul(x_cols + y_cols)
            .and_then(|cells| cells.checked_mul(4));
        if payload != Some(bytes.len() - HEADER_BYTES) {
            return Err(contract("payload length disagrees with the header"));
        }
        let data = bytes[HEADER_BYTES..]
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        Ok(Self {
            rows,
            x_cols,
            y_cols,
            data,
        })
    }
}

pub fn write_mat(path: &Path, mat: &Mat) -> ArenaResult<()> {
    write_atomic(path, &mat.to_bytes()?)
}

pub fn read_mat(path: &Path) -> ArenaResult<Mat> {
    Mat::from_bytes(&read_file(path)?).map_err(|err| match err {
        ArenaError::Contract(message) => {
            ArenaError::Contract(format!("{}: {message}", path.display()))
        }
        other => other,
    })
}

/// Relative path of one emitted matrix:
/// `<arm_id>/<cell_id with ':' replaced by '__'>.s<sample_seed>.m<multiplier>.mat`.
pub fn mat_relpath(arm_id: &str, cell_id: &str, sample_seed: u64, multiplier: u32) -> String {
    format!(
        "{arm_id}/{}.s{sample_seed}.m{multiplier}.mat",
        cell_id.replace(':', "__")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Mat {
        let data = vec![
            1.5,
            -2.25,
            f32::NAN,
            0.0,
            3.0e-12,
            f32::MAX,
            f32::MIN_POSITIVE,
            1.0,
            -0.0,
            7.0,
            8.0,
            2.0,
        ];
        Mat::new(3, 3, data).unwrap()
    }

    #[test]
    fn round_trip_is_bit_exact_on_disk() {
        let dir = crate::kpi::unit_test_dir("mat-round-trip");
        let rel = mat_relpath("stat-packet", "dev:0123456789abcdef:p48:d2", 1, 4);
        assert_eq!(rel, "stat-packet/dev__0123456789abcdef__p48__d2.s1.m4.mat");
        let path = dir.join(&rel);
        let mat = sample();
        write_mat(&path, &mat).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 24 + 12 * 4);
        assert_eq!(&bytes[..8], b"DOPEMAT1");
        assert_eq!(
            &bytes[8..24],
            &[3, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(&bytes[24..28], &1.5f32.to_le_bytes());

        let back = read_mat(&path).unwrap();
        assert_eq!((back.rows, back.x_cols, back.y_cols), (3, 3, 1));
        let bits = |m: &Mat| m.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&back), bits(&mat));
        assert_eq!(back.to_bytes().unwrap(), bytes);
        assert_eq!(back.row(2).unwrap(), &[-0.0, 7.0, 8.0, 2.0]);
        assert!(back.row(3).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_matrix_round_trips() {
        let mat = Mat::new(0, 5, Vec::new()).unwrap();
        let bytes = mat.to_bytes().unwrap();
        assert_eq!(bytes.len(), 24);
        assert_eq!(Mat::from_bytes(&bytes).unwrap(), mat);
    }

    #[test]
    fn rejects_malformed_containers() {
        let good = sample().to_bytes().unwrap();
        let expect = |bytes: &[u8], needle: &str| {
            let err = Mat::from_bytes(bytes).unwrap_err().to_string();
            assert!(err.contains(needle), "{err}");
        };
        expect(&good[..20], "magic");
        let mut bad = good.clone();
        bad[0] = b'X';
        expect(&bad, "magic");
        let mut bad = good.clone();
        bad[20] = 1;
        expect(&bad, "reserved");
        let mut bad = good.clone();
        bad[16] = 2;
        expect(&bad, "y_cols");
        expect(&good[..good.len() - 1], "payload length");
        let mut bad = good.clone();
        bad.extend_from_slice(&[0; 4]);
        expect(&bad, "payload length");
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        bad[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        expect(&bad, "payload length");
        assert!(Mat::new(2, 3, vec![0.0; 7]).is_err());
    }

    #[test]
    fn read_errors_name_the_file() {
        let dir = crate::kpi::unit_test_dir("mat-errors");
        let path = dir.join("broken.mat");
        std::fs::write(&path, b"DOPEMAT0").unwrap();
        let err = read_mat(&path).unwrap_err().to_string();
        assert!(err.contains("broken.mat") && err.contains("magic"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
