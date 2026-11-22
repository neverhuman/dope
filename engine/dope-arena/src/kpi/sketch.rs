//! Content evidence for lineage families: bottom-k sketches of column value sets.
//!
//! A column with more than `MIN_DISTINCT` distinct finite values (rounded to
//! six significant digits) is summarised by its `SKETCH_K` smallest value
//! hashes. Two columns match when the bottom-k Jaccard estimate is at least
//! one half. Candidate pairs come from an index on each sketch's smallest
//! hashes, never from an all-pairs scan. Two tables are linked when, in both
//! directions, at least `MIN_MATCHED_COLUMNS` columns have a match in the
//! other table and those columns are at least half of the eligible columns
//! named by `ShareBasis`: of the table that has fewer of them, or of each table.

use super::io::TableData;
use super::par::par_map;
use dope_arena::ArenaResult;
use dope_core::rng::splitmix64;
use serde::Serialize;
use std::fmt::Write as _;

pub const SKETCH_K: usize = 64;
/// A column is eligible when it has more than this many distinct finite values.
pub const MIN_DISTINCT: usize = 16;
pub const MIN_MATCHED_COLUMNS: usize = 3;
/// Each sketch is indexed under this many of its smallest hashes.
pub const LSH_KEYS: usize = 8;
/// Index buckets above this size compare sorted neighbours only.
pub const BUCKET_CAP: usize = 256;
pub const NEIGHBOR_WINDOW: usize = 64;

/// The eligible-column count that the matched share of a table link is measured against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ShareBasis {
    /// Half of the eligible columns of the table that has fewer of them.
    Smaller,
    /// Half of the eligible columns of each of the two tables.
    Each,
}

impl ShareBasis {
    pub fn name(self) -> &'static str {
        match self {
            ShareBasis::Smaller => "smaller",
            ShareBasis::Each => "each",
        }
    }
}

/// Hash of a finite value rounded to six significant digits; `None` for NaN
/// and infinities. The key packs sign, decimal exponent and the six digits of
/// the `{:.5e}` rendering, so +0 and -0 collide and nothing else does.
pub fn value_hash(value: f32, scratch: &mut String) -> Option<u64> {
    if !value.is_finite() {
        return None;
    }
    scratch.clear();
    write!(scratch, "{:.5e}", f64::from(value)).ok()?;
    let text = scratch.as_bytes();
    let (negative, body) = match text.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, text),
    };
    let split = body.iter().position(|&b| b == b'e')?;
    let digits = body[..split]
        .iter()
        .filter(|b| b.is_ascii_digit())
        .fold(0u64, |acc, b| acc * 10 + u64::from(b - b'0'));
    if digits == 0 {
        return Some(splitmix64(0));
    }
    let exponent: i64 = std::str::from_utf8(&body[split + 1..]).ok()?.parse().ok()?;
    let key = (u64::from(negative) << 63) | (((exponent + 1024) as u64 & 0x7ff) << 40) | digits;
    Some(splitmix64(key))
}

/// Bottom-k sketch (ascending, at most `SKETCH_K` hashes) of an eligible column.
pub fn column_sketch(values: &[f32], scratch: &mut String) -> Option<Vec<u64>> {
    let mut bits: Vec<u32> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| if *v == 0.0 { 0 } else { v.to_bits() })
        .collect();
    bits.sort_unstable();
    bits.dedup();
    if bits.len() <= MIN_DISTINCT {
        return None;
    }
    let mut hashes: Vec<u64> = bits
        .iter()
        .filter_map(|b| value_hash(f32::from_bits(*b), scratch))
        .collect();
    hashes.sort_unstable();
    hashes.dedup();
    if hashes.len() <= MIN_DISTINCT {
        return None;
    }
    hashes.truncate(SKETCH_K);
    Some(hashes)
}

/// Sketches of the eligible columns of one table: features in order, then the target.
pub fn table_sketches(table: &TableData) -> ArenaResult<Vec<Vec<u64>>> {
    let mut sketches = Vec::new();
    let mut column = Vec::with_capacity(table.n_rows);
    let mut scratch = String::new();
    for col in 0..table.n_cols {
        table.column_into(col, &mut column)?;
        sketches.extend(column_sketch(&column, &mut scratch));
    }
    sketches.extend(column_sketch(&table.y, &mut scratch));
    Ok(sketches)
}

/// `(shared, considered)` among the `SKETCH_K` smallest hashes of the union of
/// two bottom-k sketches; `shared / considered` estimates the Jaccard index
/// (and is exact when both sketches hold complete value sets).
pub fn jaccard_counts(a: &[u64], b: &[u64]) -> (usize, usize) {
    let (mut i, mut j, mut shared, mut considered) = (0, 0, 0, 0);
    while considered < SKETCH_K && (i < a.len() || j < b.len()) {
        considered += 1;
        match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) if x == y => {
                shared += 1;
                i += 1;
                j += 1;
            }
            (Some(x), Some(y)) if x < y => i += 1,
            (Some(_), None) => i += 1,
            _ => j += 1,
        }
    }
    (shared, considered)
}

/// Jaccard estimate of at least one half.
pub fn sketches_match(a: &[u64], b: &[u64]) -> bool {
    let (shared, considered) = jaccard_counts(a, b);
    considered > 0 && 2 * shared >= considered
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ContentStats {
    pub eligible_columns: usize,
    pub sketch_classes: usize,
    pub pair_checks: usize,
    pub matched_class_pairs: usize,
    pub windowed_buckets: usize,
    pub linked_table_pairs: usize,
}

/// Per table: `(sketch class, number of columns in that class)`, ascending by class.
type ClassCounts = Vec<(u32, u32)>;

/// Group equal sketches into classes: `(sorted distinct sketches, per-table class counts)`.
fn classify(tables: &[Vec<Vec<u64>>]) -> (Vec<&[u64]>, Vec<ClassCounts>) {
    let mut classes: Vec<&[u64]> = tables.iter().flatten().map(Vec::as_slice).collect();
    classes.sort_unstable();
    classes.dedup();
    let per_table = tables
        .iter()
        .map(|sketches| {
            let mut ids: Vec<u32> = sketches
                .iter()
                .map(|s| classes.partition_point(|c| *c < s.as_slice()) as u32)
                .collect();
            ids.sort_unstable();
            let mut counted: Vec<(u32, u32)> = Vec::new();
            for id in ids {
                match counted.last_mut() {
                    Some((last, columns)) if *last == id => *columns += 1,
                    _ => counted.push((id, 1)),
                }
            }
            counted
        })
        .collect();
    (classes, per_table)
}

/// Matching class pairs `(low, high)` found through the smallest-hash index.
fn matched_classes(
    classes: &[&[u64]],
    holders: &[Vec<u32>],
    stats: &mut ContentStats,
) -> Vec<(u32, u32)> {
    let mut keyed: Vec<(u64, u32)> = classes
        .iter()
        .enumerate()
        .flat_map(|(id, sketch)| sketch.iter().take(LSH_KEYS).map(move |&h| (h, id as u32)))
        .collect();
    keyed.sort_unstable();
    let same_single_holder = |a: u32, b: u32| {
        let (ha, hb) = (&holders[a as usize], &holders[b as usize]);
        ha.len() == 1 && hb.len() == 1 && ha[0] == hb[0]
    };
    let mut matched = Vec::new();
    for bucket in keyed.chunk_by(|x, y| x.0 == y.0) {
        let windowed = bucket.len() > BUCKET_CAP;
        stats.windowed_buckets += usize::from(windowed);
        for (position, &(_, a)) in bucket.iter().enumerate() {
            let end = if windowed {
                bucket.len().min(position + 1 + NEIGHBOR_WINDOW)
            } else {
                bucket.len()
            };
            for &(_, b) in &bucket[position + 1..end] {
                // Classes are ids in sketch order, so `a < b` within a bucket.
                if same_single_holder(a, b) {
                    continue;
                }
                stats.pair_checks += 1;
                if sketches_match(classes[a as usize], classes[b as usize]) {
                    matched.push((a, b));
                }
            }
        }
    }
    matched.sort_unstable();
    matched.dedup();
    matched
}

/// Content links `(low table, high table)` between tables given their column sketches.
pub fn content_links(
    tables: &[Vec<Vec<u64>>],
    basis: ShareBasis,
    threads: usize,
) -> (Vec<(usize, usize)>, ContentStats) {
    let (classes, per_table) = classify(tables);
    let mut stats = ContentStats {
        eligible_columns: tables.iter().map(Vec::len).sum(),
        sketch_classes: classes.len(),
        ..ContentStats::default()
    };
    let mut holders: Vec<Vec<u32>> = vec![Vec::new(); classes.len()];
    for (table, counted) in per_table.iter().enumerate() {
        for &(class, _) in counted {
            holders[class as usize].push(table as u32);
        }
    }
    let matched = matched_classes(&classes, &holders, &mut stats);
    stats.matched_class_pairs = matched.len();
    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); classes.len()];
    for &(a, b) in &matched {
        neighbors[a as usize].push(b);
        neighbors[b as usize].push(a);
    }

    // For each table, the other tables in which enough of its columns have a match.
    let passes: Vec<Vec<u32>> = par_map(threads, tables.len(), |t1| {
        let mut last_slot = vec![u32::MAX; tables.len()];
        let mut columns = vec![0usize; tables.len()];
        let mut touched: Vec<u32> = Vec::new();
        for (slot, &(class, multiplicity)) in per_table[t1].iter().enumerate() {
            let related = std::iter::once(class).chain(neighbors[class as usize].iter().copied());
            for other in related {
                for &t2 in &holders[other as usize] {
                    let at = t2 as usize;
                    if at == t1 || last_slot[at] == slot as u32 {
                        continue;
                    }
                    if columns[at] == 0 {
                        touched.push(t2);
                    }
                    last_slot[at] = slot as u32;
                    columns[at] += multiplicity as usize;
                }
            }
        }
        touched.sort_unstable();
        touched.retain(|&t2| {
            let matched_columns = columns[t2 as usize];
            // The reverse direction is checked from the other table's side below.
            let of_columns = match basis {
                ShareBasis::Smaller => tables[t1].len().min(tables[t2 as usize].len()),
                ShareBasis::Each => tables[t1].len(),
            };
            matched_columns >= MIN_MATCHED_COLUMNS && 2 * matched_columns >= of_columns
        });
        touched
    });
    let mut links = Vec::new();
    for (t1, others) in passes.iter().enumerate() {
        for &t2 in others {
            let t2 = t2 as usize;
            if t1 < t2 && passes[t2].binary_search(&(t1 as u32)).is_ok() {
                links.push((t1, t2));
            }
        }
    }
    stats.linked_table_pairs = links.len();
    (links, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(value: f32) -> Option<u64> {
        value_hash(value, &mut String::new())
    }

    #[test]
    fn values_hash_at_six_significant_digits() {
        assert_eq!(hash(1234.567), hash(1234.568));
        assert_ne!(hash(1234.56), hash(1234.57));
        assert_eq!(hash(0.1), hash(0.100_000_04));
        assert_eq!(hash(0.0), hash(-0.0));
        assert_ne!(hash(1.5), hash(-1.5));
        assert_ne!(hash(1.5), hash(15.0));
        assert_ne!(hash(1.5), hash(0.15));
        assert_ne!(hash(0.0), hash(f32::MIN_POSITIVE));
        assert!(hash(f32::MAX).is_some() && hash(1.0e-45).is_some());
        assert_eq!(hash(f32::NAN), None);
        assert_eq!(hash(f32::INFINITY), None);
        assert_eq!(hash(f32::NEG_INFINITY), None);
        // Known answer: 1234.567 renders as 1.23457e3 -> key (3 + 1024) << 40 | 123457.
        assert_eq!(hash(1234.567), Some(splitmix64((1027u64 << 40) | 123_457)));
        assert_eq!(
            hash(-0.000_123_456_79),
            Some(splitmix64((1u64 << 63) | (1020u64 << 40) | 123_457))
        );
    }

    #[test]
    fn eligibility_needs_more_than_sixteen_distinct_values() {
        let mut scratch = String::new();
        let mut values: Vec<f32> = (0..16).map(|i| i as f32).collect();
        values.extend([3.0, f32::NAN, f32::INFINITY, -0.0, 0.0]);
        assert_eq!(column_sketch(&values, &mut scratch), None);
        values.push(16.0);
        let sketch = column_sketch(&values, &mut scratch).unwrap();
        assert_eq!(sketch.len(), 17);
        assert!(sketch.windows(2).all(|w| w[0] < w[1]));

        // Distinct raw values that collapse at six digits do not count twice.
        let near: Vec<f32> = (0..40).map(|i| 1.0 + i as f32 * 1.0e-7).collect();
        assert_eq!(column_sketch(&near, &mut scratch), None);

        let wide: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        let sketch = column_sketch(&wide, &mut scratch).unwrap();
        assert_eq!(sketch.len(), SKETCH_K);
        let mut all: Vec<u64> = wide.iter().filter_map(|v| hash(*v)).collect();
        all.sort_unstable();
        assert_eq!(sketch, all[..SKETCH_K]);
    }

    #[test]
    fn jaccard_is_exact_for_complete_sets_and_bounded_for_sketches() {
        let range = |lo: u64, hi: u64| (lo..=hi).collect::<Vec<u64>>();
        assert_eq!(jaccard_counts(&range(1, 20), &range(11, 30)), (10, 30));
        assert!(!sketches_match(&range(1, 20), &range(11, 30)));
        assert_eq!(jaccard_counts(&range(1, 20), &range(1, 30)), (20, 30));
        assert!(sketches_match(&range(1, 20), &range(1, 30)));
        assert_eq!(jaccard_counts(&range(1, 64), &range(1, 64)), (64, 64));
        assert_eq!(jaccard_counts(&range(1, 64), &range(65, 128)), (0, 64));
        // Interleaved sketches: only the 64 smallest of the union are considered.
        let even: Vec<u64> = (0..64).map(|i| 2 * i).collect();
        let odd: Vec<u64> = (0..64).map(|i| 2 * i + 1).collect();
        assert_eq!(jaccard_counts(&even, &odd), (0, 64));
        assert_eq!(jaccard_counts(&[], &[]), (0, 0));
        assert!(!sketches_match(&[], &[]));
        assert_eq!(jaccard_counts(&range(1, 20), &[]), (0, 20));
        // Exactly one half matches (threshold is inclusive).
        assert_eq!(jaccard_counts(&range(1, 30), &range(11, 40)), (20, 40));
        assert!(sketches_match(&range(1, 30), &range(11, 40)));
    }

    fn sketch_of(values: impl Iterator<Item = f32>) -> Vec<u64> {
        column_sketch(&values.collect::<Vec<_>>(), &mut String::new()).unwrap()
    }

    #[test]
    fn links_need_three_columns_and_half_of_the_smaller_table() {
        // Distinct value ranges per column id; equal ids give equal sketches.
        let col = |id: u32| sketch_of((0..200).map(move |i| (id * 1000 + i) as f32 + 0.25));
        let table = |ids: &[u32]| ids.iter().map(|&id| col(id)).collect::<Vec<_>>();
        let tables = vec![
            table(&[1, 2, 3, 4]),                             // 0
            table(&[1, 2, 3, 9]),                             // 1: three shared columns with 0
            table(&[1, 2, 7, 8]),                             // 2: only two shared columns
            table(&[1, 2, 3, 10, 11, 12, 13]), // 3: three of table 0's (and 1's) four columns
            table(&[20, 21, 22, 23, 24, 25, 26, 1, 2, 3]), // 4: three of the four columns of 0 and 1
            table(&[20, 21, 22, 23, 24, 25, 26, 30, 31, 32]), // 5: 7 of 10 with 4
            table(&[40, 40, 40, 40]),                      // 6: one repeated column
            table(&[40, 41, 42, 43]), // 7: a single shared class is not three columns
            Vec::new(),               // 8: no eligible column
        ];
        let (links, stats) = content_links(&tables, ShareBasis::Smaller, 1);
        // (3, 4) share three columns, but that is under half of table 3's seven.
        assert_eq!(links, vec![(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (4, 5)]);
        assert_eq!(stats.linked_table_pairs, 6);
        assert_eq!(stats.eligible_columns, 47);
        assert_eq!(stats.windowed_buckets, 0);
        assert_eq!(
            content_links(&tables, ShareBasis::Smaller, 4),
            (links, stats)
        );
        // Measured against each table, three shared columns no longer tie a
        // four-column table to a seven- or ten-column one.
        let (links, stats) = content_links(&tables, ShareBasis::Each, 2);
        assert_eq!(links, vec![(0, 1), (4, 5)]);
        assert_eq!(stats.linked_table_pairs, 2);
    }

    #[test]
    fn overlapping_value_sets_link_through_the_index() {
        // Columns of table 1 hold 70% of the values of table 0's columns: Jaccard 0.7.
        let full = |id: u32| sketch_of((0..1000).map(move |i| (id * 5000 + i) as f32));
        let part = |id: u32| sketch_of((0..700).map(move |i| (id * 5000 + i) as f32));
        let far = |id: u32| sketch_of((0..1000).map(move |i| (id * 5000 + i) as f32 + 0.5));
        let tables = vec![
            (1..=5).map(full).collect::<Vec<_>>(),
            (1..=5).map(part).collect(),
            (1..=5).map(far).collect(),
        ];
        let (links, stats) = content_links(&tables, ShareBasis::Each, 2);
        assert_eq!(links, vec![(0, 1)]);
        assert_eq!(content_links(&tables, ShareBasis::Smaller, 1).0, links);
        assert_eq!(stats.matched_class_pairs, 5);
        assert_eq!(stats.sketch_classes, 15);
    }
}
