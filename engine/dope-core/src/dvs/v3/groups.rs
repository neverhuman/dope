//! Correlated feature grouping (`|Spearman| >= 0.70`) keeping parent + indicators together.

use crate::dvs::{ensure, DvsError};

/// Compute midranks for Spearman rank correlation (NaNs mapped to median rank).
fn midranks(col: &[f64]) -> Vec<f64> {
    let n = col.len();
    let mut finite: Vec<(usize, f64)> = col
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    finite.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut ranks = vec![(n.saturating_sub(1) as f64) * 0.5; n];
    let mut i = 0;
    while i < finite.len() {
        let mut j = i + 1;
        while j < finite.len() && (finite[j].1 - finite[i].1).abs() <= 1e-15 {
            j += 1;
        }
        let avg = ((i + j - 1) as f64) * 0.5;
        for &(row, _) in &finite[i..j] {
            ranks[row] = avg;
        }
        i = j;
    }
    ranks
}

pub fn spearman_corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 3 || a.len() != b.len() {
        return 0.0;
    }
    let ra = midranks(a);
    let rb = midranks(b);
    let n = ra.len() as f64;
    let ma = ra.iter().sum::<f64>() / n;
    let mb = rb.iter().sum::<f64>() / n;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in ra.iter().zip(&rb) {
        let dx = x - ma;
        let dy = y - mb;
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da <= 1e-12 || db <= 1e-12 {
        0.0
    } else {
        (num / (da * db).sqrt()).clamp(-1.0, 1.0)
    }
}

/// Partition feature columns into connected components where `|Spearman| >= threshold`
/// or two columns share the same `parent_ids[j]` (e.g. one-hot levels and missing indicator).
pub fn correlated_driver_groups(
    cols: &[Vec<f64>],
    parent_ids: Option<&[usize]>,
    threshold: f64,
) -> Result<Vec<Vec<usize>>, DvsError> {
    let d = cols.len();
    ensure(d >= 1, "empty_driver_columns")?;
    let n = cols[0].len();
    for c in cols {
        ensure(c.len() == n, "driver_column_length_mismatch")?;
    }
    if let Some(p) = parent_ids {
        ensure(p.len() == d, "driver_parent_length_mismatch")?;
    }
    let mut parent: Vec<usize> = (0..d).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut union = |a: usize, b: usize| {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent[hi] = lo;
        }
    };
    if let Some(pids) = parent_ids {
        for i in 0..d {
            for j in (i + 1)..d {
                if pids[i] == pids[j] {
                    union(i, j);
                }
            }
        }
    }
    for i in 0..d {
        for j in (i + 1)..d {
            if spearman_corr(&cols[i], &cols[j]).abs() >= threshold {
                union(i, j);
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..d {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    Ok(groups.into_values().collect())
}
