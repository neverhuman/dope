//! Pure-Rust IEEE CRC-32, RFC 1950/1951 zlib/deflate, and brotli decompression for `DML3` wire envelopes.

use crate::{DopeError, Result};
use std::io::Read;

fn err(msg: &str) -> DopeError {
    DopeError::Artifact(msg.into())
}

pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if (crc & 1) != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

pub fn adler32_checksum(bytes: &[u8]) -> u32 {
    let (mut s1, mut s2) = (1u32, 0u32);
    for &b in bytes {
        s1 = (s1 + b as u32) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    (s2 << 16) | s1
}

pub fn decompress_brotli(input: &[u8], max_out: usize) -> Result<Vec<u8>> {
    let mut reader = brotli::Decompressor::new(input, 4096);
    let mut out = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = reader
            .read(&mut chunk)
            .map_err(|_| err("invalid brotli stream"))?;
        if n == 0 {
            break;
        }
        if out.len().saturating_add(n) > max_out {
            return Err(err("decompressed payload exceeds envelope limit"));
        }
        out.extend_from_slice(&chunk[..n]);
    }
    Ok(out)
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], start: usize) -> Self {
        Self {
            data,
            byte_pos: start,
            bit_pos: 0,
        }
    }
    fn read_bits(&mut self, count: u8) -> Result<u32> {
        let mut val = 0u32;
        for i in 0..count {
            if self.byte_pos >= self.data.len() {
                return Err(err("truncated deflate bit stream"));
            }
            let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
            val |= (bit as u32) << i;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Ok(val)
    }
    fn align_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }
}

struct HuffTable {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl HuffTable {
    fn build(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l as usize >= 16 {
                return Err(err("invalid huffman code length"));
            }
            if l > 0 {
                counts[l as usize] += 1;
            }
        }
        let mut offsets = [0u16; 16];
        let mut total = 0u16;
        for bits in 1..=15 {
            offsets[bits] = total;
            total = total.saturating_add(counts[bits]);
        }
        let mut symbols = vec![0u16; total as usize];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                let idx = offsets[l as usize] as usize;
                offsets[l as usize] += 1;
                symbols[idx] = sym as u16;
            }
        }
        Ok(Self { counts, symbols })
    }
    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..=15 {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(err("unmatched huffman code in deflate stream"))
    }
}

pub fn decompress_zlib(input: &[u8], max_out: usize) -> Result<Vec<u8>> {
    if input.len() < 6 {
        return Err(err("zlib stream too short"));
    }
    let (cmf, flg) = (input[0] as u16, input[1] as u16);
    if (cmf & 0x0F) != 8 || (cmf >> 4) > 7 || (flg & 0x20) != 0 || ((cmf * 256 + flg) % 31) != 0 {
        return Err(err("invalid zlib header"));
    }
    let mut br = BitReader::new(&input[..input.len() - 4], 2);
    let mut out = Vec::new();
    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];
    const CL_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        if btype == 0 {
            br.align_byte();
            let len = br.read_bits(16)? as usize;
            let nlen = br.read_bits(16)? as usize;
            if (len ^ 0xFFFF) != nlen
                || br.byte_pos + len > br.data.len()
                || out.len() + len > max_out
            {
                return Err(err("invalid deflate stored block"));
            }
            out.extend_from_slice(&br.data[br.byte_pos..br.byte_pos + len]);
            br.byte_pos += len;
        } else if btype == 1 || btype == 2 {
            let (lit_table, dist_table) = if btype == 1 {
                let mut ll = vec![8u8; 288];
                ll[144..256].fill(9);
                ll[256..280].fill(7);
                (HuffTable::build(&ll)?, HuffTable::build(&[5u8; 32])?)
            } else {
                let hlit = br.read_bits(5)? as usize + 257;
                let hdist = br.read_bits(5)? as usize + 1;
                let hclen = br.read_bits(4)? as usize + 4;
                let mut cl_lens = [0u8; 19];
                for i in 0..hclen {
                    cl_lens[CL_ORDER[i]] = br.read_bits(3)? as u8;
                }
                let cl_table = HuffTable::build(&cl_lens)?;
                let mut lens = Vec::with_capacity(hlit + hdist);
                while lens.len() < hlit + hdist {
                    let sym = cl_table.decode(&mut br)?;
                    match sym {
                        0..=15 => lens.push(sym as u8),
                        16 => {
                            let prev = *lens
                                .last()
                                .ok_or_else(|| err("invalid repeat in huffman tree"))?;
                            let rep = br.read_bits(2)? as usize + 3;
                            lens.extend(std::iter::repeat_n(prev, rep));
                        }
                        17 => lens.extend(std::iter::repeat_n(0, br.read_bits(3)? as usize + 3)),
                        18 => lens.extend(std::iter::repeat_n(0, br.read_bits(7)? as usize + 11)),
                        _ => return Err(err("invalid huffman code length symbol")),
                    }
                }
                if lens.len() != hlit + hdist {
                    return Err(err("huffman length table overflow"));
                }
                (
                    HuffTable::build(&lens[..hlit])?,
                    HuffTable::build(&lens[hlit..])?,
                )
            };
            loop {
                let sym = lit_table.decode(&mut br)? as usize;
                if sym < 256 {
                    if out.len() >= max_out {
                        return Err(err("decompressed payload exceeds envelope limit"));
                    }
                    out.push(sym as u8);
                } else if sym == 256 {
                    break;
                } else if sym <= 285 {
                    let l_idx = sym - 257;
                    let length =
                        LEN_BASE[l_idx] as usize + br.read_bits(LEN_EXTRA[l_idx])? as usize;
                    let d_sym = dist_table.decode(&mut br)? as usize;
                    if d_sym >= 30 {
                        return Err(err("invalid deflate distance symbol"));
                    }
                    let dist =
                        DIST_BASE[d_sym] as usize + br.read_bits(DIST_EXTRA[d_sym])? as usize;
                    if dist == 0 || dist > out.len() || out.len() + length > max_out {
                        return Err(err("invalid deflate back-reference"));
                    }
                    for _ in 0..length {
                        let b = out[out.len() - dist];
                        out.push(b);
                    }
                } else {
                    return Err(err("invalid deflate literal/length symbol"));
                }
            }
        } else {
            return Err(err("reserved deflate block type"));
        }
        if bfinal == 1 {
            break;
        }
    }
    let expected_adler = u32::from_be_bytes(input[input.len() - 4..].try_into().unwrap());
    if adler32_checksum(&out) != expected_adler {
        return Err(err("zlib adler32 checksum mismatch"));
    }
    Ok(out)
}
