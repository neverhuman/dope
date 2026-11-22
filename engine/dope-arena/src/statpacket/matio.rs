//! `DOPEMAT1` row files (`docs/plans/v2-gap-closure/formats.md`, section 5):
//! magic, `u32` rows, x columns, y columns, reserved, then row-major `f32`
//! little-endian with features first and the target last.

use super::StatError;
use std::path::Path;

pub const MAT_MAGIC: &[u8; 8] = b"DOPEMAT1";
const MAT_HEADER: usize = 24;

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub x_cols: usize,
    pub y_cols: usize,
    /// Row-major `rows x (x_cols + y_cols)`.
    pub data: Vec<f32>,
}

impl Mat {
    pub fn width(&self) -> usize {
        self.x_cols + self.y_cols
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, StatError> {
        let fits = |v: usize| u32::try_from(v).map_err(|_| StatError::Invalid("matrix too large"));
        if self.data.len() != self.rows * self.width() {
            return Err(StatError::Invalid("matrix data does not match its shape"));
        }
        let mut out = Vec::with_capacity(MAT_HEADER + 4 * self.data.len());
        out.extend(MAT_MAGIC);
        for v in [self.rows, self.x_cols, self.y_cols, 0] {
            out.extend(fits(v)?.to_le_bytes());
        }
        for v in &self.data {
            out.extend(v.to_le_bytes());
        }
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, StatError> {
        let bad = |why: &'static str| StatError::Invalid(why);
        if bytes.len() < MAT_HEADER || &bytes[..8] != MAT_MAGIC {
            return Err(bad("not a DOPEMAT1 file"));
        }
        let word = |i: usize| {
            let at = 8 + 4 * i;
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
        };
        let (rows, x_cols, y_cols) = (word(0), word(1), word(2));
        if word(3) != 0 {
            return Err(bad("DOPEMAT1 reserved field is not zero"));
        }
        let cells = rows
            .checked_mul(x_cols + y_cols)
            .and_then(|c| c.checked_mul(4))
            .ok_or(bad("DOPEMAT1 shape overflows"))?;
        if bytes.len() != MAT_HEADER + cells {
            return Err(bad("DOPEMAT1 length does not match its shape"));
        }
        let data = bytes[MAT_HEADER..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(Self {
            rows,
            x_cols,
            y_cols,
            data,
        })
    }

    pub fn read(path: &Path) -> Result<Self, StatError> {
        let bytes = std::fs::read(path).map_err(|e| StatError::Io(e.kind()))?;
        Self::from_bytes(&bytes)
    }

    pub fn write(&self, path: &Path) -> Result<(), StatError> {
        std::fs::write(path, self.to_bytes()?).map_err(|e| StatError::Io(e.kind()))
    }

    /// Features as `f64`, row-major `rows x x_cols`.
    pub fn features(&self) -> Vec<f64> {
        let width = self.width();
        self.data
            .chunks(width.max(1))
            .flat_map(|row| row[..self.x_cols].iter().map(|&v| v as f64))
            .collect()
    }

    /// Target column `k` (0-based among the y columns) as `f64`.
    pub fn target(&self, k: usize) -> Vec<f64> {
        let width = self.width();
        self.data
            .chunks(width.max(1))
            .map(|row| row[self.x_cols + k] as f64)
            .collect()
    }
}
