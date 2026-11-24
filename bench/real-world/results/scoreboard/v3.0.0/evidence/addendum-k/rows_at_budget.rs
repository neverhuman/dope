//! Research-only byte-capped row reference for DOPEMAT1 target-last matrices.
//! Fit stores a seeded random subset. Sample draws complete rows with replacement.

use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::{env, fs, process::ExitCode};

const HEADER: usize = 24;
const MAGIC: &[u8; 8] = b"DOPEMAT1";

struct Matrix<'a> {
    bytes: &'a [u8],
    len: usize,
    stride: usize,
}

impl<'a> Matrix<'a> {
    fn read(bytes: &'a [u8]) -> Option<Self> {
        let h = bytes.get(..HEADER)?;
        let field =
            |at: usize| u32::from_le_bytes(h[at..at + 4].try_into().expect("four header bytes"));
        let len = field(8) as usize;
        let stride = (field(12) as usize).checked_add(1)?.checked_mul(4)?;
        if &h[..8] != MAGIC || field(16) != 1 || field(20) != 0 || len == 0 {
            return None;
        }
        (HEADER.checked_add(stride.checked_mul(len)?)? == bytes.len()).then_some(Self {
            bytes,
            len,
            stride,
        })
    }

    fn row(&self, index: usize) -> &[u8] {
        let at = HEADER + index * self.stride;
        &self.bytes[at..at + self.stride]
    }

    fn write(&self, count: usize, indices: impl Iterator<Item = usize>) -> Option<Vec<u8>> {
        let row_count = u32::try_from(count).ok()?;
        let mut result = Vec::with_capacity(HEADER.checked_add(count.checked_mul(self.stride)?)?);
        result.extend_from_slice(&self.bytes[..HEADER]);
        result[8..12].copy_from_slice(&row_count.to_le_bytes());
        for i in indices {
            result.extend_from_slice(self.row(i));
        }
        Some(result)
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, count: usize) -> usize {
        let count = count as u64;
        let threshold = count.wrapping_neg() % count;
        loop {
            let value = self.next();
            if value >= threshold {
                return (value % count) as usize;
            }
        }
    }
}

fn fit(bytes: &[u8], budget: usize, seed: u64) -> Option<Vec<u8>> {
    let matrix = Matrix::read(bytes)?;
    let keep = (budget.checked_sub(HEADER)? / matrix.stride).min(matrix.len);
    if keep == 0 {
        return None;
    }
    let mut rows: Vec<_> = (0..matrix.len).collect();
    let mut rng = SplitMix64(seed);
    for i in 0..keep {
        rows.swap(i, i + rng.index(matrix.len - i));
    }
    matrix.write(keep, rows.into_iter().take(keep))
}

fn sample(bytes: &[u8], rows: usize, seed: u64) -> Option<Vec<u8>> {
    let matrix = Matrix::read(bytes)?;
    let mut rng = SplitMix64(seed);
    matrix.write(rows, (0..rows).map(|_| rng.index(matrix.len)))
}

fn run(args: &[String]) -> Result<(), &'static str> {
    let [cmd, input, amount, seed, output] = args else {
        return Err("expected fit|sample INPUT AMOUNT SEED OUTPUT");
    };
    let amount = amount.parse().map_err(|_| "invalid amount")?;
    let seed = seed.parse().map_err(|_| "invalid seed")?;
    let bytes = fs::read(input).map_err(|_| "could not read matrix")?;
    let result = match cmd.as_str() {
        "fit" => fit(&bytes, amount, seed).ok_or("invalid matrix or no complete row fits")?,
        "sample" => sample(&bytes, amount, seed).ok_or("invalid matrix or row count")?,
        _ => return Err("expected fit or sample"),
    };
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(output)
        .and_then(|mut file| file.write_all(&result))
        .map_err(|_| "could not write matrix")
}

fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        for field in [6_u32, 2, 1, 0] {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        for row in 0..6 {
            for value in [row as f32, f32::NAN, (row % 2) as f32] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn exact_cap_and_deterministic_unique_rows() {
        let source = fixture();
        assert!(fit(&source, 35, 0).is_none());
        let packet = fit(&source, 60, 3).unwrap();
        assert_eq!(packet, fit(&source, 60, 3).unwrap());
        assert_eq!(packet.len(), 60);
        let parsed = Matrix::read(&packet).unwrap();
        assert_eq!(parsed.len, 3);
        let rows: std::collections::BTreeSet<_> = (0..parsed.len).map(|i| parsed.row(i)).collect();
        let original = Matrix::read(&source).unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows
            .iter()
            .all(|row| (0..original.len).any(|i| *row == original.row(i))));
    }

    #[test]
    fn sample_replays_and_preserves_the_packet_rows() {
        let packet = fit(&fixture(), 60, 3).unwrap();
        let output = sample(&packet, 24, 5).unwrap();
        assert_eq!(output, sample(&packet, 24, 5).unwrap());
        let reference = Matrix::read(&packet).unwrap();
        let output = Matrix::read(&output).unwrap();
        assert_eq!(output.len, 24);
        assert!(
            (0..output.len).all(|i| (0..reference.len).any(|j| output.row(i) == reference.row(j)))
        );
    }

    #[test]
    fn rejects_drifted_matrix_shapes() {
        let mut source = fixture();
        assert!(fit(&source[..source.len() - 1], 60, 3).is_none());
        source[16] = 2;
        assert!(fit(&source, 60, 3).is_none());
    }
}
