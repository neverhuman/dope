//! Exact O(n log n) Fenwick rank statistics and O(n) Hartigan Dip for `dope_core::evidence`.

/// Binary Indexed Tree (Fenwick) for O(n log n) Kendall tau-b and Huo-Szekely dCor.
struct Fenwick {
    tree: Vec<f64>,
}

impl Fenwick {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![0.0; size + 2],
        }
    }
    fn add(&mut self, mut idx: usize, val: f64) {
        idx += 1;
        while idx < self.tree.len() {
            self.tree[idx] += val;
            idx += idx & idx.wrapping_neg();
        }
    }
    fn sum(&self, mut idx: usize) -> f64 {
        let mut s = 0.0;
        idx += 1;
        while idx > 0 {
            s += self.tree[idx];
            idx &= idx - 1;
        }
        s
    }
}

/// Exact O(n log n) Kendall tau-b with full tie correction (deterministic row-index tie-break).
pub fn kendall_tau_b(x: &[f64], y: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64, usize)> = x
        .iter()
        .zip(y)
        .enumerate()
        .filter(|(_, (a, b))| a.is_finite() && b.is_finite())
        .map(|(i, (&a, &b))| (a, b, i))
        .collect();
    let n = pairs.len();
    if n < 4 {
        return 0.0;
    }
    let mut y_vals: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    y_vals.sort_by(f64::total_cmp);
    y_vals.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * b.abs().max(1.0));
    let m = y_vals.len();
    if m <= 1 {
        return 0.0;
    }

    let mut order: Vec<(f64, usize)> = pairs
        .iter()
        .map(|&(xv, yv, _)| {
            let yr = y_vals
                .partition_point(|&v| v < yv - 1e-12 * yv.abs().max(1.0))
                .min(m - 1);
            (xv, yr)
        })
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut bit = Fenwick::new(m);
    let mut y_counts = vec![0usize; m];
    let mut ties_x = 0.0f64;
    let mut concordant = 0.0f64;
    let mut discordant = 0.0f64;
    let mut seen_total = 0.0f64;

    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (order[j].0 - order[i].0).abs() <= 1e-12 * order[i].0.abs().max(1.0) {
            j += 1;
        }
        let block_len = (j - i) as f64;
        ties_x += block_len * (block_len - 1.0) * 0.5;
        for &(_, yr) in &order[i..j] {
            let less = if yr > 0 { bit.sum(yr - 1) } else { 0.0 };
            let leq = bit.sum(yr);
            concordant += less;
            discordant += seen_total - leq;
        }
        for &(_, yr) in &order[i..j] {
            bit.add(yr, 1.0);
            y_counts[yr] += 1;
        }
        seen_total += block_len;
        i = j;
    }
    let ties_y: f64 = y_counts
        .iter()
        .map(|&c| (c as f64) * (c as f64 - 1.0) * 0.5)
        .sum();
    let total_pairs = (n as f64) * (n as f64 - 1.0) * 0.5;
    let denom = ((total_pairs - ties_x).max(0.0) * (total_pairs - ties_y).max(0.0)).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        ((concordant - discordant) / denom).clamp(-1.0, 1.0)
    }
}

/// Exact O(n log n) Huo-Szekely (2016) 1D unbiased distance covariance.
fn u_centered_dcov_1d(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 {
        return 0.0;
    }
    let nf = n as f64;
    let row_sums = |arr: &[f64]| -> (Vec<f64>, f64) {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| arr[a].total_cmp(&arr[b]).then(a.cmp(&b)));
        let total: f64 = arr.iter().sum();
        let mut pref = 0.0;
        let mut sums = vec![0.0; n];
        for (pos, &i) in idx.iter().enumerate() {
            let v = arr[i];
            let left = (pos as f64) * v - pref;
            let right = (total - pref - v) - ((n - 1 - pos) as f64) * v;
            sums[i] = left + right;
            pref += v;
        }
        let grand: f64 = sums.iter().sum();
        (sums, grand)
    };
    let (a_dot, a_grand) = row_sums(x);
    let (b_dot, b_grand) = row_sums(y);

    let mut y_sorted = y.to_vec();
    y_sorted.sort_by(f64::total_cmp);
    y_sorted.dedup_by(|a, b| *a == *b);
    let m = y_sorted.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        x[a].total_cmp(&x[b])
            .then(y[a].total_cmp(&y[b]))
            .then(a.cmp(&b))
    });

    let mut f_cnt = Fenwick::new(m);
    let mut f_x = Fenwick::new(m);
    let mut f_y = Fenwick::new(m);
    let mut f_xy = Fenwick::new(m);

    let mut tot_cnt = 0.0;
    let mut tot_x = 0.0;
    let mut tot_y = 0.0;
    let mut tot_xy = 0.0;
    let mut s1 = 0.0;

    for &i in &order {
        let xi = x[i];
        let yi = y[i];
        let yr = y_sorted.partition_point(|&v| v < yi).min(m - 1);
        let c_lo = f_cnt.sum(yr);
        let x_lo = f_x.sum(yr);
        let y_lo = f_y.sum(yr);
        let xy_lo = f_xy.sum(yr);
        let part_lo = xi * yi * c_lo - xi * y_lo - yi * x_lo + xy_lo;
        let c_hi = tot_cnt - c_lo;
        let x_hi = tot_x - x_lo;
        let y_hi = tot_y - y_lo;
        let xy_hi = tot_xy - xy_lo;
        let part_hi = -xi * yi * c_hi + xi * y_hi + yi * x_hi - xy_hi;
        s1 += 2.0 * (part_lo + part_hi);

        f_cnt.add(yr, 1.0);
        f_x.add(yr, xi);
        f_y.add(yr, yi);
        f_xy.add(yr, xi * yi);
        tot_cnt += 1.0;
        tot_x += xi;
        tot_y += yi;
        tot_xy += xi * yi;
    }

    let s2: f64 = a_dot.iter().zip(&b_dot).map(|(&a, &b)| a * b).sum();
    let s3 = a_grand * b_grand;
    let denom = nf * (nf - 3.0);
    if denom <= 0.0 {
        return 0.0;
    }
    (s1 - 2.0 * s2 / (nf - 2.0) + s3 / ((nf - 1.0) * (nf - 2.0))) / denom
}

/// Exact O(n log n) unbiased 1D distance correlation in `[0, 1]`.
pub fn dcor_unbiased_1d(x: &[f64], y: &[f64]) -> f64 {
    let mut xf = Vec::with_capacity(x.len());
    let mut yf = Vec::with_capacity(y.len());
    for (&a, &b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            xf.push(a);
            yf.push(b);
        }
    }
    if xf.len() < 4 {
        return 0.0;
    }
    let dxy = u_centered_dcov_1d(&xf, &yf);
    if dxy <= 0.0 {
        return 0.0;
    }
    let dxx = u_centered_dcov_1d(&xf, &xf);
    let dyy = u_centered_dcov_1d(&yf, &yf);
    let denom = (dxx.max(0.0) * dyy.max(0.0)).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        (dxy / denom).clamp(0.0, 1.0).sqrt()
    }
}

/// Exact O(n) Hartigan Dip Test statistic D_n in [0, 0.5] on ascending sorted values.
pub fn hartigan_dip_stat(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 8 || (sorted[n - 1] - sorted[0]).abs() <= 1e-12 {
        return 0.0;
    }
    let nf = n as f64;
    let mut x: Vec<f64> = Vec::new();
    let mut f_lo: Vec<f64> = Vec::new();
    let mut f_hi: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && sorted[j] == sorted[i] {
            j += 1;
        }
        x.push(sorted[i]);
        f_lo.push(i as f64 / nf);
        f_hi.push(j as f64 / nf);
        i = j;
    }
    let m = x.len();
    if m <= 2 {
        return 0.0;
    }
    let gcm = |lo: usize, hi: usize| -> Vec<usize> {
        let mut hull: Vec<usize> = Vec::new();
        for idx in lo..=hi {
            while hull.len() >= 2 {
                let p1 = hull[hull.len() - 2];
                let p2 = hull[hull.len() - 1];
                let s1 = (f_lo[p2] - f_lo[p1]) * (x[idx] - x[p2]);
                let s2 = (f_lo[idx] - f_lo[p2]) * (x[p2] - x[p1]);
                if s1 >= s2 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(idx);
        }
        hull
    };
    let lcm = |lo: usize, hi: usize| -> Vec<usize> {
        let mut hull: Vec<usize> = Vec::new();
        for idx in lo..=hi {
            while hull.len() >= 2 {
                let p1 = hull[hull.len() - 2];
                let p2 = hull[hull.len() - 1];
                let s1 = (f_hi[p2] - f_hi[p1]) * (x[idx] - x[p2]);
                let s2 = (f_hi[idx] - f_hi[p2]) * (x[p2] - x[p1]);
                if s1 <= s2 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(idx);
        }
        hull
    };
    let eval_piecewise = |hull: &[usize], f: &[f64], target_idx: usize| -> f64 {
        let tx = x[target_idx];
        let pos = hull.partition_point(|&h| h < target_idx);
        if pos == 0 {
            return f[hull[0]];
        }
        if pos >= hull.len() {
            return f[hull[hull.len() - 1]];
        }
        let a = hull[pos - 1];
        let b = hull[pos];
        let dx = (x[b] - x[a]).max(1e-12);
        f[a] + (f[b] - f[a]) * ((tx - x[a]) / dx)
    };

    let mut lo = 0usize;
    let mut hi = m - 1;
    let mut dip = 0.0f64;
    for _ in 0..32 {
        if hi <= lo + 1 {
            break;
        }
        let g = gcm(lo, hi);
        let l = lcm(lo, hi);
        let mut d_gcm = 0.0f64;
        let mut ig = lo;
        for &k in &g {
            let gap = eval_piecewise(&l, &f_hi, k) - f_lo[k];
            if gap > d_gcm {
                d_gcm = gap;
                ig = k;
            }
        }
        let mut d_lcm = 0.0f64;
        let mut il = hi;
        for &k in &l {
            let gap = f_hi[k] - eval_piecewise(&g, &f_lo, k);
            if gap > d_lcm {
                d_lcm = gap;
                il = k;
            }
        }
        let d_now = d_gcm.max(d_lcm);
        if d_now <= dip + 1e-12 {
            break;
        }
        let (new_lo, new_hi) = if d_gcm >= d_lcm {
            let next_l = l.iter().copied().find(|&k| k >= ig).unwrap_or(hi);
            (ig, next_l)
        } else {
            let prev_g = g.iter().rev().copied().find(|&k| k <= il).unwrap_or(lo);
            (prev_g, il)
        };
        let mut left_max = 0.0f64;
        for k in lo..=new_lo {
            left_max = left_max.max((f_hi[k] - eval_piecewise(&g, &f_lo, k)).abs());
        }
        let mut right_max = 0.0f64;
        for k in new_hi..=hi {
            right_max = right_max.max((eval_piecewise(&l, &f_hi, k) - f_lo[k]).abs());
        }
        dip = dip.max(left_max.max(right_max));
        if new_lo == lo && new_hi == hi {
            dip = dip.max(d_now);
            break;
        }
        lo = new_lo;
        hi = new_hi;
    }
    (0.5 * dip).clamp(0.0, 0.5)
}
