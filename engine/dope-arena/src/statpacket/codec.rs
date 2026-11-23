//! `dope.single-encoding/v1` container: 24-byte header, sections S1..S10 in
//! nested order, CRC-32 trailer; truncation, byte ledger and budget fitting.
//!
//! Header (little-endian): magic `DSE1` | version u8 | kind u8 | task u8 | K u8 |
//! P u16 | level u8 | bit profile u8 | section count u8 | floor(log2 n_fit) u8 |
//! flags u8 | reserved u8 | decoder sha8 [8].

use super::linpred::importance_order;
use super::packet::{
    ColumnMarginal, Packet, PacketLevel, PacketTask, BIT_PROFILE_V1, CRC_LEN, FLAG_SPARSE_TARGET,
    HEADER_LEN, KIND_STAT, LOG2_N_MAX, LOG2_N_MIN, MAGIC, VERSION,
};
use super::sections::{read_section, s2_breakdown, section_len, write_section, Cursor};
use super::StatError;
use dope_core::mlc_v3::inflate::crc32_ieee;
use sha2::{Digest, Sha256};

/// Decode recipe of kind 0. Any change to a decode rule must change this text,
/// which changes `decoder_sha8` and makes earlier readers refuse the packets.
pub const RECIPE: &str = concat!(
    r#"{"codec":"dope.single-encoding/v1","decoder":"stat-packet-v0","#,
    r#""knots":"probit-spaced/nested-5-9-17/origin-pinned/half-span-6x2^((c-4095)/128)/split-over-255/code-over-63","#,
    r#""interp":"probit-linear","tail":"geometric-capped-linear/clip-6","#,
    r#""atoms":"f16-unit/6bit-mass/threshold-probit","#,
    r#""dependence":"loading-over-127/mean-6over127/psi-floor-0.02/mixture-cdf","#,
    r#""regime":"first-cumulative-prior>u/else-last-class-with-prior","#,
    r#""target":"eta=b+w.x/w=f16scale*code-over-127/class0-reference/zero-prior-class-never-drawn","#,
    r#""effects":"g(c)=4*sign(c)*(c-over-127)^2/bins-at-decoded-quartiles-count-t<=x/levels-D-E","#,
    r#""pairs":"15-products-g(c)*xa*xb/entry-15-constant/level-E","#,
    r#""noise":"regression-sd=2^((n0-240)-over-32)*2^(-d-over-5)@D*2^(-e-over-5)@E/clip-6/classes-T=2^((n1-128)-over-32)","#,
    r#""labels":"sampled-softmax(T*eta)-first-cumulative>u*total/multilabel-u<sigmoid(T*eta)-code-0-never-255-always/never-argmax/option-regime-label","#,
    r#""rng":"hash3(seed,purpose<<32|draw,row)/regime-1/factor-2/noise-3/target-4"}"#
);

/// First 8 bytes of SHA-256 of [`RECIPE`].
pub fn decoder_sha8() -> [u8; 8] {
    let digest = Sha256::digest(RECIPE.as_bytes());
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

/// Length of `pack(packet, level)` without building it.
pub fn packed_len(packet: &Packet, level: PacketLevel) -> usize {
    let body: usize = (1..=level.sections()).map(|i| section_len(packet, i)).sum();
    HEADER_LEN + body + CRC_LEN
}

/// Serialises the first `level.sections()` sections of `packet`.
pub fn pack(packet: &Packet, level: PacketLevel) -> Result<Vec<u8>, StatError> {
    packet.validate()?;
    if level > packet.level {
        return Err(StatError::Invalid(
            "requested level exceeds the packet level",
        ));
    }
    let mut out = Vec::with_capacity(packed_len(packet, level));
    out.extend(MAGIC);
    out.extend([VERSION, KIND_STAT, packet.task as u8, packet.k]);
    out.extend(packet.p.to_le_bytes());
    out.extend([
        level as u8,
        BIT_PROFILE_V1,
        level.sections() as u8,
        packet.log2_n,
        if packet.target_sparse {
            FLAG_SPARSE_TARGET
        } else {
            0
        },
        0,
    ]);
    out.extend(decoder_sha8());
    for index in 1..=level.sections() {
        write_section(packet, index, &mut out);
    }
    let crc = crc32_ieee(&out);
    out.extend(crc.to_le_bytes());
    Ok(out)
}

/// Parses header and sections; returns the packet and each section's length.
fn parse(bytes: &[u8]) -> Result<(Packet, Vec<usize>), StatError> {
    if bytes.len() < HEADER_LEN + CRC_LEN {
        return Err(StatError::Truncated);
    }
    if bytes[..4] != MAGIC {
        return Err(StatError::BadMagic);
    }
    if bytes[4] != VERSION {
        return Err(StatError::Version);
    }
    if bytes[5] != KIND_STAT {
        return Err(StatError::Kind);
    }
    let task = PacketTask::from_u8(bytes[6]).ok_or(StatError::Header("unknown task"))?;
    let k = bytes[7];
    if !task.accepts_k(k) {
        return Err(StatError::Header("class count is not valid for the task"));
    }
    let p = u16::from_le_bytes([bytes[8], bytes[9]]);
    if p == 0 {
        return Err(StatError::Header("zero columns"));
    }
    let level = PacketLevel::from_u8(bytes[10]).ok_or(StatError::Header("unknown level"))?;
    if bytes[11] != BIT_PROFILE_V1 {
        return Err(StatError::Header("unknown bit profile"));
    }
    if bytes[12] as usize != level.sections() {
        return Err(StatError::Header("section count does not match the level"));
    }
    let log2_n = bytes[13];
    if !(LOG2_N_MIN..=LOG2_N_MAX).contains(&log2_n) {
        return Err(StatError::Header("row-count exponent out of range"));
    }
    if bytes[14] & !FLAG_SPARSE_TARGET != 0 || bytes[15] != 0 {
        return Err(StatError::Header("reserved bits are set"));
    }
    if bytes[16..HEADER_LEN] != decoder_sha8() {
        return Err(StatError::Decoder);
    }
    // The S2 flag plane alone needs P bytes: refuse before allocating for P columns.
    if bytes.len() < HEADER_LEN + p as usize {
        return Err(StatError::Truncated);
    }
    let blank = vec![ColumnMarginal::Constant; p as usize];
    let mut packet = Packet::skeleton(task, k, log2_n, blank);
    packet.target_sparse = bytes[14] & FLAG_SPARSE_TARGET != 0;
    packet.level = level;
    packet.main_mask.fill(false);
    packet.pairs.clear();
    let mut cur = Cursor::new(bytes, HEADER_LEN);
    let mut lens = Vec::with_capacity(level.sections());
    for index in 1..=level.sections() {
        let start = cur.pos;
        read_section(&mut packet, index, &mut cur)?;
        lens.push(cur.pos - start);
    }
    Ok((packet, lens))
}

fn unpack_with_lens(bytes: &[u8]) -> Result<(Packet, Vec<usize>), StatError> {
    let (packet, lens) = parse(bytes)?;
    let end = HEADER_LEN + lens.iter().sum::<usize>();
    let trailer = bytes.get(end..end + CRC_LEN).ok_or(StatError::Truncated)?;
    let stored = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    if crc32_ieee(&bytes[..end]) != stored {
        return Err(StatError::Crc);
    }
    if bytes.len() != end + CRC_LEN {
        return Err(StatError::Trailing);
    }
    packet.validate()?;
    Ok((packet, lens))
}

/// Validates magic, version, kind, header, section lengths, CRC and canonical
/// form; rejects trailing bytes. Errors carry no data values.
pub fn unpack(bytes: &[u8]) -> Result<Packet, StatError> {
    unpack_with_lens(bytes).map(|(packet, _)| packet)
}

/// Drops whole trailing sections down to `level` and rewrites the level, the
/// section count and the CRC. `truncate(pack(p, E), k) == pack(p, k)`.
pub fn truncate(bytes: &[u8], level: PacketLevel) -> Result<Vec<u8>, StatError> {
    let (packet, lens) = unpack_with_lens(bytes)?;
    if level > packet.level {
        return Err(StatError::Invalid(
            "requested level exceeds the packet level",
        ));
    }
    let end = HEADER_LEN + lens[..level.sections()].iter().sum::<usize>();
    let mut out = bytes[..end].to_vec();
    out[10] = level as u8;
    out[12] = level.sections() as u8;
    let crc = crc32_ieee(&out);
    out.extend(crc.to_le_bytes());
    Ok(out)
}

/// Measured bytes of one packed file.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Ledger {
    pub level: PacketLevel,
    pub total: usize,
    pub header: usize,
    pub crc: usize,
    /// S1.. in wire order.
    pub sections: Vec<usize>,
    pub s2_flag_plane: usize,
    pub s2_knot_payload: usize,
    pub s2_atom_payload: usize,
    pub continuous_columns: usize,
    pub discrete_columns: usize,
    pub constant_columns: usize,
}

pub fn ledger(bytes: &[u8]) -> Result<Ledger, StatError> {
    let (packet, sections) = unpack_with_lens(bytes)?;
    let (flags, knots, atoms) = s2_breakdown(&packet);
    let continuous = packet.continuous_cols();
    let constant = packet
        .columns
        .iter()
        .filter(|c| matches!(c, ColumnMarginal::Constant))
        .count();
    Ok(Ledger {
        level: packet.level,
        total: bytes.len(),
        header: HEADER_LEN,
        crc: CRC_LEN,
        sections,
        s2_flag_plane: flags,
        s2_knot_payload: knots,
        s2_atom_payload: atoms,
        continuous_columns: continuous,
        discrete_columns: packet.cols() - continuous - constant,
        constant_columns: constant,
    })
}

/// Which rule of the fallback order produced the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum BudgetStep {
    /// A whole level with every section dense.
    Level,
    /// Level D with S8 kept for the most important columns only.
    PartialMainEffects,
    /// Level A with S3 weights kept for the most important columns only.
    SparseTarget,
    /// Even level A with no target weights exceeds the budget.
    OverBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetFit {
    pub packet: Packet,
    pub level: PacketLevel,
    pub bytes: usize,
    pub step: BudgetStep,
    pub fits: bool,
    pub target_cols: usize,
    pub main_effect_cols: usize,
}

fn finish(packet: Packet, level: PacketLevel, step: BudgetStep, max_bytes: usize) -> BudgetFit {
    // `level` never exceeds the source level here, so the fallback arm is unreachable.
    let packet = match packet.at_level(level) {
        Ok(cut) => cut,
        Err(_) => packet,
    };
    let bytes = packed_len(&packet, level);
    let count = |mask: &[bool]| mask.iter().filter(|&&on| on).count();
    BudgetFit {
        level,
        bytes,
        step,
        fits: bytes <= max_bytes,
        target_cols: count(&packet.target_mask),
        main_effect_cols: count(&packet.main_mask),
        packet,
    }
}

/// Deterministic degradation to at most `max_bytes`:
/// 1. the highest level E..A that fits with dense sections, except that
/// 2. level D is tried with S8 on the top-m columns before falling to C;
/// 3. below level A, S3 keeps weights for the top-m columns (m may be 0);
/// 4. otherwise the smallest packet is returned with `fits == false`.
pub fn fit_to_budget(packet: &Packet, max_bytes: usize) -> BudgetFit {
    let order = importance_order(packet);
    let o = packet.outputs();
    for level in PacketLevel::ALL
        .into_iter()
        .rev()
        .filter(|&l| l <= packet.level)
    {
        if packed_len(packet, level) <= max_bytes {
            return finish(packet.clone(), level, BudgetStep::Level, max_bytes);
        }
        if level == PacketLevel::D {
            let mut trial = packet.clone();
            trial.main_mask.fill(false);
            trial.main_effect.fill(0);
            let empty = packed_len(&trial, level);
            let per_col = 4 * o;
            let kept: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&j| packet.main_mask[j])
                .collect();
            let room = max_bytes.saturating_sub(empty) / per_col;
            let m = room.min(kept.len());
            if empty <= max_bytes && m >= 1 {
                for &j in &kept[..m] {
                    trial.main_mask[j] = true;
                    let cell = j * per_col..(j + 1) * per_col;
                    trial.main_effect[cell.clone()].copy_from_slice(&packet.main_effect[cell]);
                }
                return finish(trial, level, BudgetStep::PartialMainEffects, max_bytes);
            }
        }
    }
    let p = packet.cols();
    let mut trial = packet.clone();
    trial.target_sparse = true;
    trial.target_mask.fill(false);
    trial.target_weight.fill(0);
    let empty = packed_len(&trial, PacketLevel::A);
    let kept: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&j| packet.target_mask[j])
        .collect();
    let m = (max_bytes.saturating_sub(empty) / o).min(kept.len());
    if empty > max_bytes {
        return finish(trial, PacketLevel::A, BudgetStep::OverBudget, max_bytes);
    }
    for &j in &kept[..m] {
        trial.target_mask[j] = true;
        for k in 0..o {
            trial.target_weight[k * p + j] = packet.target_weight[k * p + j];
        }
    }
    finish(trial, PacketLevel::A, BudgetStep::SparseTarget, max_bytes)
}
