//! Lineage families (formats.md section 2): union-find over name and content links.
//!
//! Name evidence: two tables share a canonical stem that is not generic.
//! Content evidence: `sketch::content_links`. A family never spans the
//! training split and a panel; `verify_panel_families` enforces that.

use super::io::{Arena, IndexRow, SPLIT_TRAIN};
use super::par::par_map;
use super::sketch::{content_links, table_sketches, ContentStats, ShareBasis};
use dope_arena::panel::PanelManifest;
use dope_arena::{ArenaError, ArenaResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Stems that say nothing about the source; they never merge tables by name.
pub const GENERIC_STEMS: [&str; 9] = [
    "rows",
    "export",
    "unknown",
    "en",
    "csv_layers",
    "data",
    "train",
    "test",
    "dataset",
];

pub const EVIDENCE_NAME: &str = "name";
pub const EVIDENCE_CONTENT: &str = "content";
pub const EVIDENCE_SINGLETON: &str = "singleton";

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct FamilyRow {
    pub dataset_hash: String,
    pub family_id: String,
    pub canon_stem: String,
    pub evidence: String,
    pub split: String,
    pub task: String,
}

/// Cut `stem` at the first `marker` that is followed by an optional `_` and a digit.
fn cut_at_numbered(stem: &mut String, marker: &str) {
    let cut = stem.match_indices(marker).find_map(|(at, _)| {
        let rest = stem[at + marker.len()..].trim_start_matches('_');
        let underscores = stem.len() - at - marker.len() - rest.len();
        (underscores <= 1 && rest.starts_with(|c: char| c.is_ascii_digit())).then_some(at)
    });
    if let Some(at) = cut {
        stem.truncate(at);
    }
}

/// Canonical family stem: lower case with sibling suffixes removed
/// (`_target_<anything>`, `_seed_<n>...`, `_nrows<n>...`, `_train`, `_test`,
/// `_fold<n>`, `_split<n>`, `_v<n>`, trailing digits and underscores).
pub fn canon_family(stem: &str) -> String {
    let mut s: String = stem
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if matches!(c, '-' | '.' | ' ') { '_' } else { c })
        .collect();
    loop {
        let before = s.len();
        if let Some(at) = s.find("_target_") {
            s.truncate(at);
        }
        cut_at_numbered(&mut s, "_seed");
        cut_at_numbered(&mut s, "_nrows");
        for suffix in ["_target", "_train", "_test", "_fold", "_split"] {
            if s.ends_with(suffix) {
                s.truncate(s.len() - suffix.len());
            }
        }
        // One trailing run per pass, so a `_v<n>` marker is still visible when reached.
        let no_underscores = s.trim_end_matches('_').len();
        s.truncate(no_underscores);
        let no_digits = s.trim_end_matches(|c: char| c.is_ascii_digit()).len();
        if no_digits < s.len() && s[..no_digits].ends_with("_v") {
            s.truncate(no_digits - 2);
        } else {
            s.truncate(no_digits);
        }
        if s.len() == before {
            return s;
        }
    }
}

/// Empty and catch-all canonical stems carry no name evidence.
pub fn is_generic(canon_stem: &str) -> bool {
    canon_stem.is_empty() || GENERIC_STEMS.contains(&canon_stem)
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    // The smaller index becomes the root, so the result is order independent.
    parent[ra.max(rb)] = ra.min(rb);
}

/// One `FamilyRow` per table, in table order. `links` are content links by table index.
pub fn cluster_families(tables: &[IndexRow], links: &[(usize, usize)]) -> Vec<FamilyRow> {
    let canon: Vec<String> = tables
        .iter()
        .map(|t| canon_family(&t.family_stem))
        .collect();
    let mut by_stem: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, stem) in canon.iter().enumerate() {
        if !is_generic(stem) {
            by_stem.entry(stem).or_default().push(index);
        }
    }
    let mut parent: Vec<usize> = (0..tables.len()).collect();
    for members in by_stem.values() {
        for &member in &members[1..] {
            union(&mut parent, members[0], member);
        }
    }
    for &(a, b) in links {
        union(&mut parent, a, b);
    }
    let roots: Vec<usize> = (0..tables.len()).map(|i| find(&mut parent, i)).collect();
    let mut size = vec![0usize; tables.len()];
    let mut family_hash: Vec<&str> = tables.iter().map(|t| t.dataset_hash.as_str()).collect();
    for (index, &root) in roots.iter().enumerate() {
        size[root] += 1;
        if tables[index].dataset_hash.as_str() < family_hash[root] {
            family_hash[root] = &tables[index].dataset_hash;
        }
    }
    tables
        .iter()
        .enumerate()
        .map(|(index, table)| {
            let named = by_stem
                .get(canon[index].as_str())
                .is_some_and(|m| m.len() > 1);
            let evidence = match (size[roots[index]], named) {
                (1, _) => EVIDENCE_SINGLETON,
                (_, true) => EVIDENCE_NAME,
                (_, false) => EVIDENCE_CONTENT,
            };
            FamilyRow {
                dataset_hash: table.dataset_hash.clone(),
                family_id: format!("lf-{}", family_hash[roots[index]]),
                canon_stem: canon[index].clone(),
                evidence: evidence.to_string(),
                split: table.split.clone(),
                task: table.task.clone(),
            }
        })
        .collect()
}

/// Sketch every table of the arena and cluster it into lineage families.
pub fn build_families(
    arena: &Arena,
    basis: ShareBasis,
    threads: usize,
) -> ArenaResult<(Vec<FamilyRow>, ContentStats)> {
    let sketches = par_map(threads, arena.tables().len(), |table_id| {
        arena.read_table(table_id).and_then(|t| table_sketches(&t))
    })
    .into_iter()
    .collect::<ArenaResult<Vec<_>>>()?;
    let (links, stats) = content_links(&sketches, basis, threads);
    Ok((cluster_families(arena.tables(), &links), stats))
}

/// Family ids that contain at least one training table.
pub fn train_families(families: &[FamilyRow]) -> BTreeSet<&str> {
    families
        .iter()
        .filter(|row| row.split == SPLIT_TRAIN)
        .map(|row| row.family_id.as_str())
        .collect()
}

/// Every panel table must carry its lineage family, and no family of the panel
/// may contain a training table.
pub fn verify_panel_families(families: &[FamilyRow], panel: &PanelManifest) -> ArenaResult<()> {
    let by_hash: BTreeMap<&str, &FamilyRow> = families
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect();
    if by_hash.len() != families.len() {
        return Err(ArenaError::Contract(
            "families file lists a dataset_hash twice".into(),
        ));
    }
    let with_train = train_families(families);
    for entry in &panel.datasets {
        let row = by_hash.get(entry.dataset_id.as_str()).ok_or_else(|| {
            ArenaError::Contract(format!(
                "panel table {} is missing from the families file",
                entry.dataset_id
            ))
        })?;
        if row.family_id != entry.family_id {
            return Err(ArenaError::Contract(format!(
                "panel table {} disagrees with the families file on family_id",
                entry.dataset_id
            )));
        }
        if row.split == SPLIT_TRAIN || with_train.contains(entry.family_id.as_str()) {
            return Err(ArenaError::Contract(format!(
                "family {} spans the training split and panel {}",
                entry.family_id, panel.panel_id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::fixture::{write_arena, FixtureTable};
    use crate::kpi::seeds::KpiRng;
    use dope_arena::panel::{PanelDatasetEntry, PanelRole};

    #[test]
    fn siblings_collapse_to_one_canonical_stem() {
        let siblings = "x_target_a x_target_b x_seed_3 x_2 X_Target_long_name_7 x_nrows2000 \
            x_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_true x_nrows_500_extra x_train \
            x_test x_fold3 x_fold_3 x_split_2 x_v2 x_v2_3 x_test_2 x__ x_target x_seed7 x-2 \
            x.2 x_v10_target_y_seed_1";
        for stem in siblings.split(' ').chain([" x 2 "]) {
            assert_eq!(canon_family(stem), "x", "{stem}");
        }
        assert_eq!(
            canon_family("openml_gina_prior2_1041_target_pixel"),
            "openml_gina_prior"
        );
        assert_eq!(canon_family("pmlb_197_cpu_act_target"), "pmlb_197_cpu_act");
        assert_eq!(
            canon_family("forex_eurnok_minute_close"),
            "forex_eurnok_minute_close"
        );
        assert_eq!(
            canon_family("par_yield_curve_rates_1990"),
            "par_yield_curve_rates"
        );
        assert_eq!(canon_family("3b582d0708894d4b"), "3b582d0708894d4b");
        // Markers need their separator and number; ordinary words stay intact.
        assert_eq!(canon_family("heart_seed"), "heart_seed");
        assert_eq!(canon_family("seedling_survey"), "seedling_survey");
        assert_eq!(canon_family("moving_targets"), "moving_targets");
        assert_eq!(canon_family("river_v"), "river_v");
        assert_eq!(canon_family("contest"), "contest");
        assert_eq!(canon_family("x_seed__3"), "x_seed");
    }

    #[test]
    fn generic_stems_are_recognised_after_canonicalisation() {
        let generic = "rows export unknown en csv_layers data train test dataset Rows rows_3 \
            data_v2 export_target_y test_1 data_train 12345 _ _target_y 2020_01";
        for stem in generic.split(' ').chain([""]) {
            assert!(is_generic(&canon_family(stem)), "{stem}");
        }
        for stem in ["download", "rows_of_data", "dataset_sales", "x"] {
            assert!(!is_generic(&canon_family(stem)), "{stem}");
        }
    }

    fn index_row(table_id: usize, hash: &str, stem: &str, split: &str) -> IndexRow {
        IndexRow {
            table_id,
            dataset_hash: hash.into(),
            canonical_sha256: "0".repeat(64),
            family_stem: stem.into(),
            task: "regression".into(),
            n_classes: 0,
            n_rows: 10,
            n_cols: 2,
            split: split.into(),
            x_off: 0,
            row_off: 0,
        }
    }

    #[test]
    fn name_links_merge_siblings_but_never_generic_stems() {
        let tables = vec![
            index_row(0, "h9", "x_target_a", "train"),
            index_row(1, "h3", "x_target_b", "val"),
            index_row(2, "h5", "x_seed_3", "val"),
            index_row(3, "h7", "x_2", "val"),
            index_row(4, "h1", "rows", "train"),
            index_row(5, "h2", "rows", "val"),
            index_row(6, "h4", "export_2", "val"),
            index_row(7, "h6", "12345", "val"),
            index_row(8, "h8", "lonely", "val"),
        ];
        let rows = cluster_families(&tables, &[]);
        assert_eq!(rows.len(), 9);
        for row in &rows[..4] {
            assert_eq!(
                (row.family_id.as_str(), row.evidence.as_str()),
                ("lf-h3", "name")
            );
            assert_eq!(row.canon_stem, "x");
        }
        for (row, hash) in rows[4..].iter().zip(["h1", "h2", "h4", "h6", "h8"]) {
            assert_eq!(row.family_id, format!("lf-{hash}"));
            assert_eq!(row.evidence, "singleton");
        }
        assert_eq!(rows[5].canon_stem, "rows");
        assert_eq!(rows[6].canon_stem, "export");
        assert_eq!(rows[7].canon_stem, "");
        assert_eq!(
            (rows[0].split.as_str(), rows[1].split.as_str()),
            ("train", "val")
        );

        // Content links join generic stems and chain through name links.
        let rows = cluster_families(&tables, &[(4, 5), (3, 6)]);
        assert_eq!(
            (rows[4].family_id.as_str(), rows[4].evidence.as_str()),
            ("lf-h1", "content")
        );
        assert_eq!(
            (rows[5].family_id.as_str(), rows[5].evidence.as_str()),
            ("lf-h1", "content")
        );
        assert_eq!(
            (rows[6].family_id.as_str(), rows[6].evidence.as_str()),
            ("lf-h3", "content")
        );
        assert_eq!(
            (rows[3].family_id.as_str(), rows[3].evidence.as_str()),
            ("lf-h3", "name")
        );
        assert_eq!(rows[8].evidence, "singleton");
        // Link order and direction do not matter.
        assert_eq!(rows, cluster_families(&tables, &[(6, 3), (5, 4)]));
    }

    /// `rows` x `cols` measurements on a per-seed grid (about a hundred levels per
    /// column), so two seeds share almost no values while a row subset keeps most.
    fn measurements(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
        let mut rng = KpiRng::new(seed);
        let step = 0.05 + 0.013 * seed as f64;
        (0..rows * cols)
            .map(|i| ((rng.normal() * 20.0).round() * step + (i % cols) as f64 * 100.0) as f32)
            .collect()
    }

    fn take_rows(x: &[f32], cols: usize, rows: impl Iterator<Item = usize>) -> Vec<f32> {
        rows.flat_map(|r| x[r * cols..(r + 1) * cols].iter().copied())
            .collect()
    }

    #[test]
    fn renamed_half_copy_merges_and_unrelated_tables_do_not() {
        let cols = 6;
        let source = measurements(1, 600, cols);
        let half = take_rows(&source, cols, (0..600).step_by(2));
        let table = |hash: &str, stem: &str, split: &str, x: Vec<f32>| {
            let rows = x.len() / cols;
            FixtureTable::new(hash, stem, "regression", split, cols)
                .with_rows(x, (0..rows).map(|r| (r % 7) as f32).collect())
        };
        let tables = vec![
            table("00000000000000a1", "sensor_log", "train", source),
            table("00000000000000a2", "renamed_extract", "val", half),
            table(
                "00000000000000a3",
                "other_plant",
                "val",
                measurements(2, 600, cols),
            ),
            table(
                "00000000000000a4",
                "rows",
                "val",
                measurements(3, 300, cols),
            ),
        ];
        let dir = crate::kpi::unit_test_dir("families-content");
        write_arena(&dir, &tables).unwrap();
        let arena = Arena::open(&dir).unwrap();
        let (rows, stats) = build_families(&arena, ShareBasis::Smaller, 1).unwrap();
        assert_eq!(rows[0].family_id, "lf-00000000000000a1");
        assert_eq!(rows[1].family_id, "lf-00000000000000a1");
        assert_eq!(
            (rows[0].evidence.as_str(), rows[1].evidence.as_str()),
            ("content", "content")
        );
        assert_eq!(
            (rows[2].family_id.as_str(), rows[2].evidence.as_str()),
            ("lf-00000000000000a3", "singleton")
        );
        assert_eq!(
            (rows[3].family_id.as_str(), rows[3].evidence.as_str()),
            ("lf-00000000000000a4", "singleton")
        );
        assert_eq!(stats.linked_table_pairs, 1);
        assert_eq!(stats.eligible_columns, 4 * cols);
        let again = build_families(&arena, ShareBasis::Each, 3).unwrap();
        assert_eq!(again, (rows, stats));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn panel_verification_rejects_training_overlap() {
        let family = |hash: &str, id: &str, split: &str| FamilyRow {
            dataset_hash: hash.into(),
            family_id: id.into(),
            canon_stem: "s".into(),
            evidence: "name".into(),
            split: split.into(),
            task: "binary".into(),
        };
        let entry = |hash: &str, id: &str| PanelDatasetEntry {
            dataset_id: hash.into(),
            family_id: id.into(),
            target_column: "__target__".into(),
            task: "binary".into(),
            group_column: None,
            time_column: None,
            source_sha256: "0".repeat(64),
            rows: 10,
            features: 2,
        };
        let panel = |datasets| PanelManifest {
            protocol: "dope-fidelity/v3".into(),
            panel_id: "dev".into(),
            role: PanelRole::Development,
            datasets,
            manifest_sha256: None,
        };
        let families = vec![
            family("t1", "lf-a", "train"),
            family("v1", "lf-a", "val"),
            family("v2", "lf-b", "val"),
            family("v3", "lf-b", "val"),
        ];
        assert_eq!(train_families(&families), BTreeSet::from(["lf-a"]));
        verify_panel_families(
            &families,
            &panel(vec![entry("v2", "lf-b"), entry("v3", "lf-b")]),
        )
        .unwrap();
        let failure = |datasets| {
            verify_panel_families(&families, &panel(datasets))
                .unwrap_err()
                .to_string()
        };
        assert!(failure(vec![entry("v1", "lf-a")]).contains("spans the training split"));
        assert!(failure(vec![entry("t1", "lf-a")]).contains("spans the training split"));
        assert!(failure(vec![entry("v2", "lf-z")]).contains("disagrees"));
        assert!(failure(vec![entry("v9", "lf-b")]).contains("missing from the families file"));
        let mut doubled = families.clone();
        doubled.push(family("v2", "lf-b", "val"));
        let err = verify_panel_families(&doubled, &panel(vec![entry("v2", "lf-b")])).unwrap_err();
        assert!(err.to_string().contains("twice"));
    }
}
