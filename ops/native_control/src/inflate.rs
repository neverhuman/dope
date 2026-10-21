//! Bounded authenticated runtime/control gzip framing, no heap.
use crate::{hash, sys::check};
pub const MAX: usize = 16 << 20;
pub fn inflate<'a>(raw: &[u8], out: &'a mut [u8]) -> &'a [u8] {
    let header = b"TDL1\x01\0\0\0\x0f\0\0\0literal_payload";
    check(raw.starts_with(header) && raw.len() >= header.len() + 8);
    let at = header.len();
    let size = u64::from_le_bytes(raw[at..at + 8].try_into().unwrap());
    let gz = &raw[at + 8..];
    check(size == gz.len() as u64 && gz.len() >= 18);
    check(gz[0..3] == [0x1f, 0x8b, 8] && gz[3] & 0xe0 == 0);
    let flags = gz[3];
    let mut pos = 10;
    // Accept the ordinary gzip vocabulary, including optional fields, while
    // rejecting concatenated streams/trailing bytes and validating header CRC.
    if flags & 4 != 0 {
        check(pos + 2 <= gz.len() - 8);
        let n = u16::from_le_bytes(gz[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        check(n <= gz.len() - 8 - pos);
        pos += n;
    }
    for flag in [8, 16] {
        if flags & flag != 0 {
            while pos < gz.len() - 8 && gz[pos] != 0 {
                pos += 1;
            }
            check(pos < gz.len() - 8);
            pos += 1;
        }
    }
    if flags & 2 != 0 {
        check(pos + 2 <= gz.len() - 8);
        check(
            hash::crc32(&gz[..pos]) as u16
                == u16::from_le_bytes(gz[pos..pos + 2].try_into().unwrap()),
        );
        pos += 2;
    }
    check(pos < gz.len() - 8);
    let trailer = gz.len() - 8;
    let (used, written) = deflate(&gz[pos..trailer], out);
    check(used == trailer - pos && written <= MAX);
    check(
        hash::crc32(&out[..written])
            == u32::from_le_bytes(gz[trailer..trailer + 4].try_into().unwrap()),
    );
    check(written as u32 == u32::from_le_bytes(gz[trailer + 4..].try_into().unwrap()));
    &out[..written]
}
// Bounded RFC1951 decoder. Canonical tables use counts/ordered symbols, no heap
// or large general-purpose fast tables. Every bit, code, run and backreference
// is checked before access; gzip framing/CRC/ISIZE are checked by inflate().
struct Bits<'a> {
    raw: &'a [u8],
    pos: usize,
    bit: u32,
}
impl Bits<'_> {
    fn read(&mut self, n: u32) -> usize {
        let mut value = 0;
        for i in 0..n {
            check(self.pos < self.raw.len());
            value |= (((self.raw[self.pos] as usize) >> self.bit) & 1) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        value
    }
    fn align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }
}
struct Huffman {
    counts: [u16; 16],
    symbols: [u16; 288],
}
impl Huffman {
    fn new(lengths: &[u8]) -> Self {
        check(lengths.len() <= 288);
        let mut table = Self {
            counts: [0; 16],
            symbols: [0; 288],
        };
        for &n in lengths {
            check(n <= 15);
            table.counts[n as usize] += 1;
        }
        let mut left: i32 = 1;
        for n in 1..16 {
            left = (left << 1) - table.counts[n] as i32;
            check(left >= 0);
        }
        let mut offsets = [0usize; 16];
        for n in 1..15 {
            offsets[n + 1] = offsets[n] + table.counts[n] as usize;
        }
        for (symbol, &n) in lengths.iter().enumerate() {
            if n != 0 {
                table.symbols[offsets[n as usize]] = symbol as u16;
                offsets[n as usize] += 1;
            }
        }
        table
    }
    fn symbol(&self, input: &mut Bits<'_>) -> usize {
        let (mut code, mut first, mut index) = (0usize, 0usize, 0usize);
        for n in 1..16 {
            code |= input.read(1);
            let count = self.counts[n] as usize;
            if code >= first && code - first < count {
                return self.symbols[index + code - first] as usize;
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        crate::sys::fail()
    }
}
fn deflate(raw: &[u8], output: &mut [u8]) -> (usize, usize) {
    let mut input = Bits {
        raw,
        pos: 0,
        bit: 0,
    };
    let mut written = 0usize;
    loop {
        let final_block = input.read(1) != 0;
        let kind = input.read(2);
        if kind == 0 {
            input.align();
            let length = input.read(16);
            let inverse = input.read(16);
            check(length ^ inverse == 65535 && length <= output.len() - written);
            for _ in 0..length {
                output[written] = input.read(8) as u8;
                written += 1;
            }
        } else {
            check(kind == 1 || kind == 2);
            let mut literals = [0u8; 288];
            let mut distances = [0u8; 32];
            let (nl, nd) = if kind == 1 {
                literals[..144].fill(8);
                literals[144..256].fill(9);
                literals[256..280].fill(7);
                literals[280..].fill(8);
                distances.fill(5);
                (288, 32)
            } else {
                let nl = input.read(5) + 257;
                let nd = input.read(5) + 1;
                let nc = input.read(4) + 4;
                check(nl <= 286 && nd <= 32);
                let order = [
                    16usize, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut codes = [0u8; 19];
                for i in 0..nc {
                    codes[order[i]] = input.read(3) as u8;
                }
                let codes = Huffman::new(&codes);
                let mut lengths = [0u8; 318];
                let mut n = 0;
                while n < nl + nd {
                    let symbol = codes.symbol(&mut input);
                    if symbol < 16 {
                        lengths[n] = symbol as u8;
                        n += 1;
                    } else {
                        let (value, count) = match symbol {
                            16 => {
                                check(n > 0);
                                (lengths[n - 1], input.read(2) + 3)
                            }
                            17 => (0, input.read(3) + 3),
                            18 => (0, input.read(7) + 11),
                            _ => crate::sys::fail(),
                        };
                        check(count <= nl + nd - n);
                        lengths[n..n + count].fill(value);
                        n += count;
                    }
                }
                literals[..nl].copy_from_slice(&lengths[..nl]);
                distances[..nd].copy_from_slice(&lengths[nl..nl + nd]);
                (nl, nd)
            };
            check(literals[256] != 0);
            let lit = Huffman::new(&literals[..nl]);
            let dist = Huffman::new(&distances[..nd]);
            loop {
                let symbol = lit.symbol(&mut input);
                if symbol < 256 {
                    check(written < output.len());
                    output[written] = symbol as u8;
                    written += 1;
                } else if symbol == 256 {
                    break;
                } else {
                    check(symbol <= 285);
                    const LEN: [usize; 29] = [
                        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
                        67, 83, 99, 115, 131, 163, 195, 227, 258,
                    ];
                    const EXTRA: [u32; 29] = [
                        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5,
                        5, 5, 5, 0,
                    ];
                    let n = symbol - 257;
                    let length = LEN[n] + input.read(EXTRA[n]);
                    let d = dist.symbol(&mut input);
                    check(d <= 29);
                    let extra = if d < 4 { 0 } else { (d / 2 - 1) as u32 };
                    let base = if d < 4 {
                        d + 1
                    } else {
                        ((2 + (d & 1)) << extra) + 1
                    };
                    let distance = base + input.read(extra);
                    check(
                        distance > 0
                            && distance <= written
                            && distance <= 32768
                            && length <= output.len() - written,
                    );
                    for _ in 0..length {
                        output[written] = output[written - distance];
                        written += 1;
                    }
                }
            }
        }
        if final_block {
            return (input.pos + usize::from(input.bit != 0), written);
        }
    }
}
