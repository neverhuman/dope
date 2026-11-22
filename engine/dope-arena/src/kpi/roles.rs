//! Row roles of one cell (formats.md section 4): disjoint A / B / C / T rows
//! that keep every duplicate group inside a single role, and seeded column views.
//!
//! With `G` duplicate groups: T takes `round(0.30 G)` groups, clamped to
//! [200, 2000] once the table can still give A and B 200 groups each
//! (`G >= 600`); A and B take `min(1024, floor(rest / 2))` groups each; C
//! takes up to as many of what is left. A and B are then thinned by seeded
//! row sampling to the same row count (at most 1024), C to at most that count.

use super::seeds::KpiRng;
use std::collections::{BTreeMap, BTreeSet};

pub const T_MIN_GROUPS: usize = 200;
pub const T_MAX_GROUPS: usize = 2000;
pub const AB_MAX: usize = 1024;
/// A cell is `headline` when A, B and T each hold at least this many duplicate groups.
pub const HEADLINE_MIN: usize = 200;
/// A cell is supported when A, B and T each hold at least this many rows.
pub const MIN_ROLE_ROWS: usize = 16;
pub const VIEW_COLUMNS: usize = 48;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoleSplit {
    /// Ascending row indices per role; `a.len() == b.len()`.
    pub a: Vec<usize>,
    pub b: Vec<usize>,
    pub c: Vec<usize>,
    pub t: Vec<usize>,
    /// Duplicate groups present in A and in B (the smaller of the two counts).
    pub ab_groups: usize,
    /// Duplicate groups in T.
    pub t_groups: usize,
}

impl RoleSplit {
    pub fn supported(&self) -> bool {
        self.a.len() >= MIN_ROLE_ROWS && self.t.len() >= MIN_ROLE_ROWS
    }

    pub fn headline(&self) -> bool {
        self.ab_groups >= HEADLINE_MIN && self.t_groups >= HEADLINE_MIN
    }
}

/// Duplicate groups given to T for a table with `groups` of them.
pub fn t_group_count(groups: usize) -> usize {
    let share = (3 * groups + 5) / 10;
    if groups >= T_MIN_GROUPS + 2 * HEADLINE_MIN {
        share.clamp(T_MIN_GROUPS, T_MAX_GROUPS)
    } else {
        share
    }
}

/// Keep `keep` of `rows` by seeded sampling without replacement; ascending output.
fn thin(mut rows: Vec<usize>, keep: usize, rng: &mut KpiRng) -> Vec<usize> {
    if rows.len() > keep {
        let picks = rng.sample_without_replacement(rows.len(), keep);
        rows = picks.into_iter().map(|at| rows[at]).collect();
    }
    rows.sort_unstable();
    rows
}

fn distinct_groups(dup: &[i32], rows: &[usize]) -> usize {
    rows.iter().map(|&r| dup[r]).collect::<BTreeSet<_>>().len()
}

/// Split the rows of one table (`dup[r]` = duplicate-group id of row `r`).
pub fn split_roles(dup: &[i32], seed: u64) -> RoleSplit {
    let mut by_group: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (row, group) in dup.iter().enumerate() {
        by_group.entry(*group).or_default().push(row);
    }
    let groups: Vec<Vec<usize>> = by_group.into_values().collect();
    let mut order: Vec<usize> = (0..groups.len()).collect();
    let mut rng = KpiRng::new(seed);
    rng.shuffle(&mut order);

    let t_groups = t_group_count(groups.len());
    let rest = groups.len() - t_groups;
    let ab = AB_MAX.min(rest / 2);
    let c_groups = ab.min(rest - 2 * ab);
    let mut start = 0;
    let mut take = |count: usize| -> Vec<usize> {
        let rows = order[start..start + count]
            .iter()
            .flat_map(|&g| groups[g].iter().copied())
            .collect();
        start += count;
        rows
    };
    let (t_rows, a_rows, b_rows, c_rows) = (take(t_groups), take(ab), take(ab), take(c_groups));
    let rows = a_rows.len().min(b_rows.len()).min(AB_MAX);
    let a = thin(a_rows, rows, &mut rng);
    let b = thin(b_rows, rows, &mut rng);
    let c = thin(c_rows, rows, &mut rng);
    let mut t = t_rows;
    t.sort_unstable();
    RoleSplit {
        ab_groups: distinct_groups(dup, &a).min(distinct_groups(dup, &b)),
        t_groups,
        a,
        b,
        c,
        t,
    }
}

/// The `p48` view: `VIEW_COLUMNS` seeded column indices of a wider table, ascending.
pub fn view_columns(n_cols: usize, seed: u64) -> Vec<usize> {
    let mut cols = KpiRng::new(seed).sample_without_replacement(n_cols, VIEW_COLUMNS);
    cols.sort_unstable();
    cols
}

#[cfg(test)]
mod tests {
    use super::*;

    fn singletons(rows: usize) -> Vec<i32> {
        (0..rows as i32).collect()
    }

    /// Roles are ascending, disjoint, in range, and no duplicate group straddles two roles.
    fn check_invariants(dup: &[i32], split: &RoleSplit) {
        let mut owner: BTreeMap<i32, usize> = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for (role, rows) in [&split.a, &split.b, &split.c, &split.t]
            .into_iter()
            .enumerate()
        {
            assert!(
                rows.windows(2).all(|w| w[0] < w[1]),
                "role {role} not ascending"
            );
            for &row in rows {
                assert!(row < dup.len());
                assert!(seen.insert(row), "row {row} is in two roles");
                assert_eq!(
                    *owner.entry(dup[row]).or_insert(role),
                    role,
                    "group straddles roles"
                );
            }
        }
        assert_eq!(split.a.len(), split.b.len());
        assert!(split.a.len() <= AB_MAX && split.c.len() <= split.a.len());
    }

    #[test]
    fn t_share_is_thirty_percent_with_the_stated_clamp() {
        let cases = [
            (0, 0),
            (1, 0),
            (2, 1),
            (25, 8),
            (51, 15),
            (52, 16),
            (599, 180),
            (600, 200),
            (666, 200),
            (667, 200),
            (700, 210),
            (6666, 2000),
            (6667, 2000),
            (100_000, 2000),
        ];
        for (groups, expected) in cases {
            assert_eq!(t_group_count(groups), expected, "G = {groups}");
        }
    }

    #[test]
    fn large_table_fills_every_role() {
        let dup = singletons(3000);
        let split = split_roles(&dup, 9);
        check_invariants(&dup, &split);
        assert_eq!(
            (split.t.len(), split.a.len(), split.b.len(), split.c.len()),
            (900, 1024, 1024, 52)
        );
        assert_eq!((split.t_groups, split.ab_groups), (900, 1024));
        assert!(split.supported() && split.headline());
        assert_eq!(split, split_roles(&dup, 9));
        assert_ne!(split.t, split_roles(&dup, 10).t);
        // The split is a shuffle, not a prefix.
        assert_ne!(split.t, (0..900).collect::<Vec<_>>());
    }

    #[test]
    fn stratum_and_support_follow_the_table_size() {
        let sizes = |rows: usize| {
            let dup = singletons(rows);
            let s = split_roles(&dup, 1);
            check_invariants(&dup, &s);
            (s.t.len(), s.a.len(), s.c.len(), s.supported(), s.headline())
        };
        assert_eq!(sizes(0), (0, 0, 0, false, false));
        assert_eq!(sizes(10), (3, 3, 1, false, false));
        assert_eq!(sizes(51), (15, 18, 0, false, false));
        assert_eq!(sizes(52), (16, 18, 0, true, false));
        assert_eq!(sizes(400), (120, 140, 0, true, false));
        assert_eq!(sizes(599), (180, 209, 1, true, false));
        assert_eq!(sizes(600), (200, 200, 0, true, true));
        assert_eq!(sizes(800), (240, 280, 0, true, true));
        assert_eq!(sizes(8192), (2000, 1024, 1024, true, true));
    }

    #[test]
    fn duplicate_groups_stay_whole_and_a_equals_b() {
        // 300 groups of 1..=5 rows, ids interleaved through the table.
        let mut dup = Vec::new();
        for round in 0..5 {
            for group in 0..300 {
                if group % 5 >= round {
                    dup.push(group * 7 - 900);
                }
            }
        }
        assert_eq!(dup.len(), 900);
        for seed in 0..20 {
            let split = split_roles(&dup, seed);
            check_invariants(&dup, &split);
            // T keeps whole groups: 90 groups of 3 rows on average.
            assert_eq!(split.t_groups, 90);
            assert_eq!(distinct_groups(&dup, &split.t), 90);
            let t_rows: usize = split
                .t
                .iter()
                .map(|&r| dup[r])
                .collect::<BTreeSet<_>>()
                .iter()
                .map(|g| dup.iter().filter(|d| *d == g).count())
                .sum();
            assert_eq!(split.t.len(), t_rows);
            assert!(split.ab_groups <= 105 && split.ab_groups > 80);
            assert!(split.supported() && !split.headline());
        }
    }

    #[test]
    fn heavy_duplication_is_capped_and_judged_by_groups() {
        // 2000 distinct rows, each stored four times: 8000 rows, 2000 groups.
        let dup: Vec<i32> = (0..8000).map(|r| r % 2000).collect();
        let split = split_roles(&dup, 4);
        check_invariants(&dup, &split);
        assert_eq!(split.t_groups, 600);
        assert_eq!(split.t.len(), 2400);
        assert_eq!(split.a.len(), 1024);
        assert!(split.ab_groups >= 500 && split.ab_groups <= 700);
        assert!(split.headline());

        // 100 distinct rows stored 50 times: plenty of rows, too few groups for the headline.
        let dup: Vec<i32> = (0..5000).map(|r| r % 100).collect();
        let split = split_roles(&dup, 4);
        check_invariants(&dup, &split);
        assert_eq!(
            (split.t_groups, split.t.len(), split.a.len()),
            (30, 1500, 1024)
        );
        assert!(split.ab_groups <= 35);
        assert!(split.supported() && !split.headline());
    }

    #[test]
    fn column_views_are_sorted_seeded_subsets() {
        let cols = view_columns(60, 3);
        assert_eq!(cols.len(), VIEW_COLUMNS);
        assert!(cols.windows(2).all(|w| w[0] < w[1]));
        assert!(cols.iter().all(|&c| c < 60));
        assert_eq!(cols, view_columns(60, 3));
        assert_ne!(cols, view_columns(60, 4));
        assert_ne!(cols, (0..48).collect::<Vec<_>>());
        assert_eq!(view_columns(48, 3), (0..48).collect::<Vec<_>>());
    }
}
