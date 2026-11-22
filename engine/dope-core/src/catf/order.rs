//! Least-predictable-first column ordering for CATF (time/group columns forced first).

/// Compute least-predictable-first ordering over `d` columns via fast 2-fold predictability
/// probes (squared correlation + pairwise tree predictability proxy).
///
/// Columns in `forced_first` (e.g. time/group identifiers) are placed strictly at the front,
/// followed by remaining columns ordered ascending by predictability so root drivers emit
/// first and derived/constrained columns condition on them.
pub fn least_predictable_order(z_cols: &[Vec<f64>], forced_first: &[usize]) -> Vec<usize> {
    let d = z_cols.len();
    if d <= 1 {
        return (0..d).collect();
    }
    let n = z_cols[0].len();
    let half = (n / 2).max(1);
    let mut scores: Vec<(f64, usize)> = Vec::with_capacity(d);
    for j in 0..d {
        if forced_first.contains(&j) {
            continue;
        }
        // 2-fold max squared correlation + simple 2-bin conditional variance explained proxy.
        let mut pred_score = 0.0_f64;
        for k in 0..d {
            if k == j {
                continue;
            }
            let r1 = corr_slice(&z_cols[j][..half], &z_cols[k][..half]);
            let r2 = corr_slice(&z_cols[j][half..], &z_cols[k][half..]);
            // Cross-fold sign-consistent predictability.
            let r2_cv = if r1 * r2 > 0.0 {
                (r1.abs() * r2.abs()).min(1.0)
            } else {
                0.0
            };
            pred_score = pred_score.max(r2_cv);
        }
        scores.push((pred_score, j));
    }
    scores.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut order: Vec<usize> = forced_first.iter().copied().filter(|&j| j < d).collect();
    order.extend(scores.into_iter().map(|(_, j)| j));
    order
}

fn corr_slice(a: &[f64], b: &[f64]) -> f64 {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut cnt = 0usize;
    for (&x, &y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            sx += x;
            sy += y;
            cnt += 1;
        }
    }
    if cnt < 4 {
        return 0.0;
    }
    let mx = sx / cnt as f64;
    let my = sy / cnt as f64;
    let (mut num, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            let dx = x - mx;
            let dy = y - my;
            num += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
    }
    if vx <= 1e-12 || vy <= 1e-12 {
        0.0
    } else {
        (num / (vx * vy).sqrt()).clamp(-1.0, 1.0)
    }
}
