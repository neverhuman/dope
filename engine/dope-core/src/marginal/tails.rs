//! Wasserstein-1 dynamic-programming interior knots and log-spaced tails to `k`-th order stats.

use crate::stats::quantile_sorted;

/// Build monotone `(u, x)` knots minimizing Wasserstein-1 error with log-spaced tails anchored
/// at the `k`-th order statistics (`k = min(support_k, n/10)`) — never raw min/max!
pub fn fit_w1_knots(sorted: &[f64], max_knots: usize, support_k: usize) -> (Vec<f64>, Vec<f64>) {
    let n = sorted.len();
    if n == 0 {
        return (vec![0.0, 1.0], vec![0.0, 0.0]);
    }
    if n < 4 || max_knots < 4 {
        let lo = sorted[0];
        let hi = sorted[n - 1];
        return (vec![0.0, 0.5, 1.0], vec![lo, sorted[n / 2], hi]);
    }

    // Privacy-safe tail anchors at k-th order statistics (support floor k = 10 when n >= 100).
    let k_ord = support_k.min(n / 10).max(1);
    let u_lo = (k_ord as f64 / n as f64).clamp(1e-5, 0.05);
    let u_hi = (1.0 - u_lo).clamp(0.95, 1.0 - 1e-5);
    let u_lo2 = (u_lo * 0.02_f64.max(u_lo * 2.0)).sqrt().min(0.10);
    let u_hi2 = (1.0 - ((1.0 - u_hi) * 0.02_f64.max((1.0 - u_hi) * 2.0)).sqrt()).max(0.90);

    // Build fine empirical candidate grid of size M <= 64 spanning [u_lo2, u_hi2].
    let m_grid = 48usize;
    let mut cand_u = Vec::with_capacity(m_grid + 4);
    cand_u.push(u_lo);
    if u_lo2 > u_lo + 1e-6 {
        cand_u.push(u_lo2);
    }
    for i in 1..m_grid {
        let u = u_lo2 + (u_hi2 - u_lo2) * (i as f64 / m_grid as f64);
        cand_u.push(u);
    }
    if u_hi2 < u_hi - 1e-6 {
        cand_u.push(u_hi2);
    }
    cand_u.push(u_hi);
    let cand_x: Vec<f64> = cand_u.iter().map(|&u| quantile_sorted(sorted, u)).collect();
    let m = cand_u.len();
    let k_budget = max_knots.clamp(5, m);

    // Precompute exact piecewise-linear W1 segment cost between candidate grid points a < b.
    let mut seg_cost = vec![vec![0.0_f64; m]; m];
    for a in 0..m {
        for b in (a + 1)..m {
            let du_tot = (cand_u[b] - cand_u[a]).max(1e-12);
            let mut err = 0.0;
            for j in a..b {
                let u_mid = 0.5 * (cand_u[j] + cand_u[j + 1]);
                let x_emp = 0.5 * (cand_x[j] + cand_x[j + 1]);
                let t = (u_mid - cand_u[a]) / du_tot;
                let x_lin = cand_x[a] + t * (cand_x[b] - cand_x[a]);
                err += (x_emp - x_lin).abs() * (cand_u[j + 1] - cand_u[j]);
            }
            seg_cost[a][b] = err;
        }
    }

    // Exact 1-D DP: dp[s][j] = min W1 cost using `s` segments ending at candidate `j`.
    let segs = k_budget - 1;
    let mut dp = vec![vec![f64::INFINITY; m]; segs + 1];
    let mut prev = vec![vec![0usize; m]; segs + 1];
    dp[0][0] = 0.0;
    for s in 1..=segs {
        for j in s..m {
            let mut best = f64::INFINITY;
            let mut best_i = s - 1;
            for i in (s - 1)..j {
                let c = dp[s - 1][i] + seg_cost[i][j];
                if c < best {
                    best = c;
                    best_i = i;
                }
            }
            dp[s][j] = best;
            prev[s][j] = best_i;
        }
    }

    // Backtrack chosen knot indices from m - 1 to 0.
    let mut chosen = Vec::with_capacity(k_budget);
    let mut cur = m - 1;
    for s in (1..=segs).rev() {
        chosen.push(cur);
        cur = prev[s][cur];
    }
    chosen.push(0);
    chosen.reverse();

    // Ensure both log-spaced tail interior knots (index 1 and m-2) are retained.
    for &tail_idx in &[1usize, m - 2] {
        if !chosen.contains(&tail_idx) {
            chosen.push(tail_idx);
        }
    }
    chosen.sort_unstable();
    chosen.dedup();

    let us: Vec<f64> = chosen.iter().map(|&i| cand_u[i]).collect();
    let xs: Vec<f64> = chosen.iter().map(|&i| cand_x[i]).collect();
    (us, xs)
}

/// Monotone piecewise-linear interpolation on arbitrary knot grid `(us, xs)` with
/// clamped/linear tail extension up to the `k`-th order-statistic bounds.
pub fn interp_uv(us: &[f64], xs: &[f64], u: f64) -> f64 {
    let k = us.len();
    if k == 0 {
        return 0.0;
    }
    if k == 1 || u <= us[0] {
        return xs[0];
    }
    if u >= us[k - 1] {
        return xs[k - 1];
    }
    let idx = us
        .partition_point(|&ui| ui <= u)
        .saturating_sub(1)
        .min(k - 2);
    let du = (us[idx + 1] - us[idx]).max(1e-12);
    let t = ((u - us[idx]) / du).clamp(0.0, 1.0);
    xs[idx] + t * (xs[idx + 1] - xs[idx])
}

/// Inverse CDF lookup on `(us, xs)`.
pub fn invert_uv(us: &[f64], xs: &[f64], x: f64) -> f64 {
    let k = xs.len();
    if k == 0 {
        return 0.5;
    }
    if k == 1 || x <= xs[0] {
        return us[0];
    }
    if x >= xs[k - 1] {
        return us[k - 1];
    }
    let idx = xs
        .partition_point(|&xi| xi <= x)
        .saturating_sub(1)
        .min(k - 2);
    let dx = xs[idx + 1] - xs[idx];
    let t = if dx > 1e-12 {
        ((x - xs[idx]) / dx).clamp(0.0, 1.0)
    } else {
        0.5
    };
    us[idx] + t * (us[idx + 1] - us[idx])
}
