//! Small statistics helpers that follow the arithmetic of the original analyses: NumPy's
//! pairwise summation for means, population standard deviation, the median, and Spearman
//! correlation (average ranks for ties).

/// NumPy's pairwise summation (`np.add.reduce` on a contiguous float64 array).
pub fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

/// `np.mean`.
pub fn mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// `np.std` (population, ddof = 0).
pub fn std(a: &[f64]) -> f64 {
    let m = mean(a);
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    (np_sum(&sq) / a.len() as f64).sqrt()
}

/// `np.median` / `statistics.median`: the middle value, or the mean of the two middle values.
pub fn median(a: &[f64]) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Ranks from 1 with tied values sharing their average rank.
///
/// The prototype used `a.argsort().argsort()`, whose order among tied values depends on
/// NumPy's sort kernel (an AVX-512 sort on the probe machine), so its Spearman values were
/// not portable. Average ranks are the standard definition and are deterministic.
pub fn ranks(a: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..a.len()).collect();
    order.sort_by(|&i, &j| a[i].total_cmp(&a[j]));
    let mut r = vec![0.0; a.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && a[order[end]] == a[order[start]] {
            end += 1;
        }
        let avg = (start + end + 1) as f64 / 2.0;
        for &i in &order[start..end] {
            r[i] = avg;
        }
        start = end;
    }
    r
}

/// Pearson correlation of two equal-length samples.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    (sab / (saa.sqrt() * sbb.sqrt())).clamp(-1.0, 1.0)
}

/// Spearman correlation as the analyses computed it; `None` below three points or when
/// either side is constant.
pub fn spearman(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() < 3 || a.iter().all(|&x| x == a[0]) || b.iter().all(|&y| y == b[0]) {
        return None;
    }
    Some(pearson(&ranks(a), &ranks(b)))
}
