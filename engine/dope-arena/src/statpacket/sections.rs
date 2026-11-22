//! Byte layout of sections S1..S10 of `dope.single-encoding/v1`, kind 0.
//!
//! There is no directory: every length follows from the header (task, K, P,
//! flags) and from the S2 flag plane. Readers take a bounds-checked cursor and
//! reject every non-canonical encoding (reserved bits, padding bits, orderings),
//! so `pack(unpack(bytes)) == bytes` for every accepted input.

use super::packet::{
    AtomTable, AtomValues, ColumnMarginal, Packet, PairTerm, MAIN_BINS, MAX_RANK, PAIRS, RANK_BLOCK,
};
use super::StatError;
use dope_core::quant;

const KIND_CONTINUOUS: u8 = 0;
const KIND_DISCRETE: u8 = 1;
const KIND_CONSTANT: u8 = 2;
const FLAG_LATTICE: u8 = 0x40;

/// Bounds-checked reader; running past the end is a truncated file.
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], StatError> {
        let end = self.pos.checked_add(n).ok_or(StatError::Truncated)?;
        let out = self.bytes.get(self.pos..end).ok_or(StatError::Truncated)?;
        self.pos = end;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, StatError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, StatError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn i8s(&mut self, n: usize) -> Result<Vec<i8>, StatError> {
        Ok(self.take(n)?.iter().map(|&b| b as i8).collect())
    }
}

fn push_i8s(out: &mut Vec<u8>, v: &[i8]) {
    out.extend(v.iter().map(|&c| c as u8));
}

/// `n` 6-bit codes, LSB first (the `dope_core::quant` bit order).
fn push_code6(out: &mut Vec<u8>, codes: &[u8]) {
    let wide: Vec<u32> = codes.iter().map(|&c| c as u32).collect();
    out.extend(quant::pack(&wide, 6));
}

fn read_code6(cur: &mut Cursor, n: usize, section: u8) -> Result<Vec<u8>, StatError> {
    let raw = cur.take(quant::packed_len(n, 6))?;
    let codes: Vec<u8> = quant::unpack(raw, n, 6)
        .into_iter()
        .map(|c| c as u8)
        .collect();
    let mut canonical = Vec::with_capacity(raw.len());
    push_code6(&mut canonical, &codes);
    if canonical != raw {
        return Err(StatError::Section(section, "padding bits are not zero"));
    }
    Ok(codes)
}

fn push_bitmap(out: &mut Vec<u8>, mask: &[bool]) {
    let mut bytes = vec![0u8; mask.len().div_ceil(8)];
    for (j, &on) in mask.iter().enumerate() {
        if on {
            bytes[j / 8] |= 1 << (j % 8);
        }
    }
    out.extend(bytes);
}

fn read_bitmap(cur: &mut Cursor, p: usize, section: u8) -> Result<Vec<bool>, StatError> {
    let raw = cur.take(p.div_ceil(8))?;
    let mask: Vec<bool> = (0..p).map(|j| raw[j / 8] >> (j % 8) & 1 == 1).collect();
    let mut canonical = Vec::with_capacity(raw.len());
    push_bitmap(&mut canonical, &mask);
    if canonical != raw {
        return Err(StatError::Section(section, "padding bits are not zero"));
    }
    Ok(mask)
}

/// Appends section `index` (1..=10) of `p`.
pub fn write_section(p: &Packet, index: usize, out: &mut Vec<u8>) {
    let (cols, o) = (p.cols(), p.outputs());
    match index {
        1 => {
            out.extend(&p.prior);
            out.push(p.resid_sd);
            out.push(p.temperature);
        }
        2 => {
            out.extend(p.columns.iter().map(flag_byte));
            for col in &p.columns {
                write_marginal(col, out);
            }
        }
        3 => {
            if p.target_sparse {
                push_bitmap(out, &p.target_mask);
            }
            for k in 0..o {
                out.extend(p.target_scale[k].to_le_bytes());
                out.extend(p.target_intercept[k].to_le_bytes());
            }
            for k in 0..o {
                let row = &p.target_weight[k * cols..(k + 1) * cols];
                let kept = row.iter().zip(&p.target_mask).filter(|(_, &on)| on);
                out.extend(kept.map(|(&w, _)| w as u8));
            }
        }
        4 | 6 | 9 => {
            let first = rank_block(index) * RANK_BLOCK;
            for row in p.loading.chunks(MAX_RANK) {
                push_i8s(out, &row[first..first + RANK_BLOCK]);
            }
            for row in p.class_mean.chunks(MAX_RANK) {
                push_i8s(out, &row[first..first + RANK_BLOCK]);
            }
        }
        5 | 7 => {
            for col in &p.columns {
                if let ColumnMarginal::Continuous { mid9, mid17, .. } = col {
                    push_code6(out, if index == 5 { mid9 } else { mid17 });
                }
            }
        }
        8 => {
            push_bitmap(out, &p.main_mask);
            let width = o * MAIN_BINS;
            for (j, _) in p.main_mask.iter().enumerate().filter(|(_, &on)| on) {
                push_i8s(out, &p.main_effect[j * width..(j + 1) * width]);
            }
        }
        _ => {
            for term in &p.pairs {
                if p.pair_index_len() == 1 {
                    out.extend([term.a as u8, term.b as u8]);
                } else {
                    out.extend(term.a.to_le_bytes());
                    out.extend(term.b.to_le_bytes());
                }
                push_i8s(out, &term.coef);
            }
        }
    }
}

fn rank_block(section: usize) -> usize {
    match section {
        4 => 0,
        6 => 1,
        _ => 2,
    }
}

fn flag_byte(col: &ColumnMarginal) -> u8 {
    match col {
        ColumnMarginal::Continuous { .. } => KIND_CONTINUOUS,
        ColumnMarginal::Constant => KIND_CONSTANT,
        ColumnMarginal::Discrete(t) => {
            let lattice = matches!(t.values, AtomValues::Lattice { .. });
            KIND_DISCRETE
                | (((t.prob.len() - 1) as u8) << 2)
                | if lattice { FLAG_LATTICE } else { 0 }
        }
    }
}

fn write_marginal(col: &ColumnMarginal, out: &mut Vec<u8>) {
    match col {
        ColumnMarginal::Constant => {}
        ColumnMarginal::Continuous { lo, hi, split, .. } => {
            // 24 bits little-endian: `lo` in bits 0..12, `hi` in bits 12..24.
            let halves = *lo as u32 | (*hi as u32) << 12;
            out.extend(&halves.to_le_bytes()[..3]);
            out.extend(split);
        }
        ColumnMarginal::Discrete(t) => {
            match &t.values {
                AtomValues::Lattice { base, step, index } => {
                    out.extend(base.to_le_bytes());
                    out.extend(step.to_le_bytes());
                    out.extend(index);
                }
                AtomValues::Explicit(values) => {
                    for v in values {
                        out.extend(v.to_le_bytes());
                    }
                }
            }
            push_code6(out, &t.prob);
        }
    }
}

fn read_marginal(cur: &mut Cursor, flag: u8) -> Result<ColumnMarginal, StatError> {
    let bad = |why: &'static str| StatError::Section(2, why);
    let (kind, count, lattice) = (
        flag & 3,
        ((flag >> 2) & 15) as usize,
        flag & FLAG_LATTICE != 0,
    );
    if flag & 0x80 != 0 {
        return Err(bad("reserved flag bit is set"));
    }
    if kind != KIND_DISCRETE && (count != 0 || lattice) {
        return Err(bad("atom bits set on a column without atoms"));
    }
    match kind {
        KIND_CONSTANT => Ok(ColumnMarginal::Constant),
        KIND_CONTINUOUS => {
            let b = cur.take(5)?;
            let halves = u32::from_le_bytes([b[0], b[1], b[2], 0]);
            Ok(ColumnMarginal::Continuous {
                lo: (halves & 0xfff) as u16,
                hi: (halves >> 12) as u16,
                split: [b[3], b[4]],
                mid9: [0; 4],
                mid17: [0; 8],
            })
        }
        KIND_DISCRETE => {
            let atoms = count + 1;
            if atoms < 2 {
                return Err(bad("a discrete column needs two atoms"));
            }
            let values = if lattice {
                let (base, step) = (cur.u16()?, cur.u16()?);
                let index = cur.take(atoms)?.to_vec();
                AtomValues::Lattice { base, step, index }
            } else {
                let mut v = Vec::with_capacity(atoms);
                for _ in 0..atoms {
                    v.push(cur.u16()?);
                }
                AtomValues::Explicit(v)
            };
            let prob = read_code6(cur, atoms, 2)?;
            Ok(ColumnMarginal::Discrete(AtomTable { values, prob }))
        }
        _ => Err(bad("unknown column kind")),
    }
}

/// Reads section `index` (1..=10) into `p`, whose header fields are already set.
pub fn read_section(p: &mut Packet, index: usize, cur: &mut Cursor) -> Result<(), StatError> {
    let (cols, o) = (p.cols(), p.outputs());
    let section = index as u8;
    match index {
        1 => {
            p.prior = cur.take(p.prior_len())?.to_vec();
            p.resid_sd = cur.u8()?;
            p.temperature = cur.u8()?;
        }
        2 => {
            let flags = cur.take(cols)?;
            p.columns = flags
                .iter()
                .map(|&f| read_marginal(cur, f))
                .collect::<Result<_, _>>()?;
        }
        3 => {
            p.target_mask = if p.target_sparse {
                read_bitmap(cur, cols, section)?
            } else {
                vec![true; cols]
            };
            for k in 0..o {
                p.target_scale[k] = cur.u16()?;
                p.target_intercept[k] = cur.u16()?;
            }
            let kept: Vec<usize> = (0..cols).filter(|&j| p.target_mask[j]).collect();
            for k in 0..o {
                let row = cur.i8s(kept.len())?;
                for (&j, w) in kept.iter().zip(row) {
                    p.target_weight[k * cols + j] = w;
                }
            }
        }
        4 | 6 | 9 => {
            let first = rank_block(index) * RANK_BLOCK;
            for row in p.loading.chunks_mut(MAX_RANK) {
                row[first..first + RANK_BLOCK].copy_from_slice(&cur.i8s(RANK_BLOCK)?);
            }
            for row in p.class_mean.chunks_mut(MAX_RANK) {
                row[first..first + RANK_BLOCK].copy_from_slice(&cur.i8s(RANK_BLOCK)?);
            }
        }
        5 | 7 => {
            for col in &mut p.columns {
                if let ColumnMarginal::Continuous { mid9, mid17, .. } = col {
                    if index == 5 {
                        mid9.copy_from_slice(&read_code6(cur, 4, section)?);
                    } else {
                        mid17.copy_from_slice(&read_code6(cur, 8, section)?);
                    }
                }
            }
        }
        8 => {
            p.main_mask = read_bitmap(cur, cols, section)?;
            let width = o * MAIN_BINS;
            for j in (0..cols).filter(|&j| p.main_mask[j]) {
                p.main_effect[j * width..(j + 1) * width].copy_from_slice(&cur.i8s(width)?);
            }
        }
        _ => {
            let wide = p.pair_index_len() == 2;
            p.pairs = Vec::with_capacity(PAIRS);
            for _ in 0..PAIRS {
                let (a, b) = if wide {
                    (cur.u16()?, cur.u16()?)
                } else {
                    (cur.u8()? as u16, cur.u8()? as u16)
                };
                if a as usize >= cols || b as usize >= cols {
                    return Err(StatError::Section(section, "pair column out of range"));
                }
                let coef = cur.i8s(o)?;
                p.pairs.push(PairTerm { a, b, coef });
            }
        }
    }
    Ok(())
}

/// Length in bytes of section `index` (1..=10) for the structure of `p`.
pub fn section_len(p: &Packet, index: usize) -> usize {
    let (cols, o) = (p.cols(), p.outputs());
    match index {
        1 => p.prior_len() + 2,
        2 => cols + p.columns.iter().map(|c| c.payload_len()).sum::<usize>(),
        3 => {
            let kept = p.target_mask.iter().filter(|&&on| on).count();
            let bitmap = if p.target_sparse { p.bitmap_len() } else { 0 };
            bitmap + 4 * o + o * kept
        }
        4 | 6 | 9 => RANK_BLOCK * (cols + p.regime_means()),
        5 => 3 * p.continuous_cols(),
        7 => 6 * p.continuous_cols(),
        8 => {
            let kept = p.main_mask.iter().filter(|&&on| on).count();
            p.bitmap_len() + MAIN_BINS * o * kept
        }
        _ => PAIRS * (2 * p.pair_index_len() + o),
    }
}

/// S2 split for the ledger: flag plane, knot payloads, atom payloads.
pub fn s2_breakdown(p: &Packet) -> (usize, usize, usize) {
    let knots = 5 * p.continuous_cols();
    let atoms = p
        .columns
        .iter()
        .filter(|c| matches!(c, ColumnMarginal::Discrete(_)))
        .map(|c| c.payload_len())
        .sum();
    (p.cols(), knots, atoms)
}
