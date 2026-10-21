//! Strict decoder for the actual TDL1/literal_payload/gzip control document.
//! Fixed-size output slices, no heap, recursive JSON objects or unknown fields.
use crate::sys::check;
pub const MAX: usize = 16 << 20;
pub const VALUES: usize = 4096 * 64;
#[derive(Clone, Copy)]
pub struct Matrix {
    pub start: usize,
    pub rows: usize,
    pub widths: [usize; 4096],
}
impl Matrix {
    fn empty() -> Self {
        Self {
            start: 0,
            rows: 0,
            widths: [0; 4096],
        }
    }
}
pub struct Model {
    pub method: u32,
    pub columns: usize,
    pub count: usize,
    pub rows: Matrix,
    pub quantiles: Matrix,
    pub factor: Matrix,
}

fn at(raw: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap())
}
pub fn parse(raw: &[u8], expected: u32, values: &mut [f64]) -> Model {
    check(raw.len() >= 40 && &raw[..8] == b"DNCB2\0\0\0");
    let method = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let columns = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    let fit_rows = at(raw, 16) as usize;
    let _fit_seed = at(raw, 24);
    let count = u32::from_le_bytes(raw[32..36].try_into().unwrap()) as usize;
    let stored = u32::from_le_bytes(raw[36..40].try_into().unwrap()) as usize;
    check(
        method == expected
            && (method == 1 || method == 2)
            && columns > 0
            && columns <= 64
            && fit_rows > 0
            && fit_rows <= 4096,
    );
    let mut model = Model {
        method,
        columns,
        count,
        rows: Matrix::empty(),
        quantiles: Matrix::empty(),
        factor: Matrix::empty(),
    };
    let total = if method == 1 {
        check(stored > 0 && stored <= fit_rows && count == 0);
        model.rows.rows = stored;
        model.rows.widths[..stored].fill(columns);
        stored * columns
    } else {
        check(stored == 0 && fit_rows >= 3 && [257, 129, 65, 33, 17, 9, 5, 3].contains(&count));
        model.quantiles.rows = columns;
        model.quantiles.widths[..columns].fill(count);
        model.factor.rows = columns;
        model.factor.start = columns * count;
        for i in 0..columns {
            model.factor.widths[i] = i + 1;
        }
        1 + columns * count + columns * (columns + 1) / 2
    };
    check(total <= values.len() && raw.len() == 40 + total * 8);
    let skip = if method == 2 {
        check(f64::from_bits(at(raw, 40)) == 1e-6);
        1
    } else {
        0
    };
    for i in skip..total {
        let v = f64::from_bits(at(raw, 40 + i * 8));
        check(v.is_finite());
        values[i - skip] = v;
    }
    if method == 2 {
        for i in 0..columns {
            let q = &values[i * count..(i + 1) * count];
            check(q.windows(2).all(|p| p[0] <= p[1]));
            let start = model.factor.start + i * (i + 1) / 2;
            let f = &values[start..start + i + 1];
            check(f.iter().all(|v| libm::fabs(*v) <= 1.00000001) && f[i] > 0.0);
            let mut products = [0.0; 64];
            for (a, v) in products.iter_mut().zip(f) {
                *a = v * v;
            }
            check(libm::fabs(crate::sample::fsum(&products[..f.len()]) - 1.0) < 1e-7);
        }
    }
    model
}
