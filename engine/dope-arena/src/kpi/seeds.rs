//! Common random numbers for the scoreboard (formats.md section 4).
//!
//! Every stream is keyed by
//! `sha256("dope-kpi-v2|" + panel_sha256 + "|" + dataset_hash + "|" + data_seed + "|" + sample_seed + "|" + purpose)`,
//! first 8 bytes little-endian. Row splits and column subsets are structural:
//! they are always derived with `STRUCTURAL_SAMPLE_SEED`, so no sample seed
//! can move a row between roles or change a view.

use dope_core::rng::Rng;
use sha2::{Digest, Sha256};

pub const PURPOSE_SPLIT: &str = "split";
pub const PURPOSE_COLS: &str = "cols";

/// The sample-seed slot used by streams that must not depend on the sample seed.
pub const STRUCTURAL_SAMPLE_SEED: u64 = 0;

pub fn crn_seed(
    panel_sha256: &str,
    dataset_hash: &str,
    data_seed: u64,
    sample_seed: u64,
    purpose: &str,
) -> u64 {
    let key =
        format!("dope-kpi-v2|{panel_sha256}|{dataset_hash}|{data_seed}|{sample_seed}|{purpose}");
    let digest = Sha256::digest(key.as_bytes());
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(head)
}

/// Seed of a split or column stream; identical for every sample seed.
pub fn structural_seed(
    panel_sha256: &str,
    dataset_hash: &str,
    data_seed: u64,
    purpose: &str,
) -> u64 {
    crn_seed(
        panel_sha256,
        dataset_hash,
        data_seed,
        STRUCTURAL_SAMPLE_SEED,
        purpose,
    )
}

/// Sequential generator over `dope_core::rng::Rng` (counter-based SplitMix64
/// mixing). The known-answer tests below pin its output, so a change in the
/// shared generator cannot silently move frozen scoreboard cells.
#[derive(Clone, Debug)]
pub struct KpiRng(Rng);

impl KpiRng {
    pub fn new(seed: u64) -> Self {
        Self(Rng::new(seed))
    }

    // Uniform and normal draws serve the control arms and bootstrap of later stages.
    #[allow(dead_code)]
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    /// Uniform on the open interval (0, 1).
    #[allow(dead_code)]
    pub fn uniform(&mut self) -> f64 {
        self.0.uniform()
    }

    /// Standard normal by inversion.
    #[allow(dead_code)]
    pub fn normal(&mut self) -> f64 {
        self.0.normal()
    }

    /// Uniform index in `0..n` (0 when `n` is 0).
    pub fn below(&mut self, n: usize) -> usize {
        self.0.below(n)
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        self.0.shuffle(items);
    }

    /// `k` distinct indices from `0..n` in draw order (all of them when `k >= n`).
    pub fn sample_without_replacement(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        let take = k.min(n);
        for slot in 0..take {
            let pick = slot + self.below(n - slot);
            pool.swap(slot, pick);
        }
        pool.truncate(take);
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PANEL: &str = "abababababababababababababababababababababababababababababababab";
    const TABLE: &str = "0123456789abcdef";

    /// Independent recomputation with `sha2`, spelled out byte by byte.
    fn reference(text: &str) -> u64 {
        let digest = Sha256::digest(text.as_bytes());
        (0..8).fold(0u64, |acc, i| acc | (u64::from(digest[i]) << (8 * i)))
    }

    #[test]
    fn crn_seed_matches_known_answers() {
        // Constants computed outside this crate with Python `hashlib` + `struct.unpack('<Q')`.
        assert_eq!(crn_seed(PANEL, TABLE, 3, 7, "split"), 17531045900753394662);
        assert_eq!(crn_seed(PANEL, TABLE, 3, 0, "split"), 14285463600360393667);
        assert_eq!(crn_seed(PANEL, TABLE, 0, 0, "cols"), 6088917845452204365);
        assert_eq!(crn_seed("", "", 0, 0, ""), 9133687623537898423);
        assert_eq!(
            crn_seed(PANEL, TABLE, 3, 7, "split"),
            reference(&format!("dope-kpi-v2|{PANEL}|{TABLE}|3|7|split"))
        );
        assert_eq!(
            crn_seed(PANEL, TABLE, 18446744073709551615, 12, "auditor"),
            reference(&format!(
                "dope-kpi-v2|{PANEL}|{TABLE}|18446744073709551615|12|auditor"
            ))
        );
    }

    #[test]
    fn every_key_part_changes_the_seed() {
        let base = crn_seed(PANEL, TABLE, 1, 2, "sample");
        assert_ne!(
            base,
            crn_seed(&PANEL.replace('a', "c"), TABLE, 1, 2, "sample")
        );
        assert_ne!(base, crn_seed(PANEL, "0123456789abcdee", 1, 2, "sample"));
        assert_ne!(base, crn_seed(PANEL, TABLE, 2, 2, "sample"));
        assert_ne!(base, crn_seed(PANEL, TABLE, 1, 3, "sample"));
        assert_ne!(base, crn_seed(PANEL, TABLE, 1, 2, "label"));
    }

    #[test]
    fn structural_streams_ignore_the_sample_seed() {
        // The split stream has no sample-seed input at all, so any sample seed
        // list yields the same permutation; the generic stream does depend on it.
        let split = structural_seed(PANEL, TABLE, 3, PURPOSE_SPLIT);
        assert_eq!(
            split,
            crn_seed(PANEL, TABLE, 3, STRUCTURAL_SAMPLE_SEED, "split")
        );
        assert_eq!(split, 14285463600360393667);
        assert_ne!(split, structural_seed(PANEL, TABLE, 4, PURPOSE_SPLIT));
        assert_ne!(split, structural_seed(PANEL, TABLE, 3, PURPOSE_COLS));
        assert_ne!(
            crn_seed(PANEL, TABLE, 3, 1, "sample"),
            crn_seed(PANEL, TABLE, 3, 2, "sample")
        );
        let order = |seed: u64| {
            let mut rows: Vec<usize> = (0..50).collect();
            KpiRng::new(seed).shuffle(&mut rows);
            rows
        };
        assert_eq!(
            order(split),
            order(structural_seed(PANEL, TABLE, 3, PURPOSE_SPLIT))
        );
        assert_ne!(
            order(split),
            order(structural_seed(PANEL, TABLE, 4, PURPOSE_SPLIT))
        );
    }

    #[test]
    fn generator_matches_known_answers() {
        // First three outputs for seed 42, recomputed in Python from the published
        // SplitMix64 constants (hash3(seed, 0xF17, counter), counter from 1).
        let mut rng = KpiRng::new(42);
        assert_eq!(rng.next_u64(), 0x48df_88e7_37a0_beaf);
        assert_eq!(rng.next_u64(), 0x2e0f_134d_ff33_baa4);
        assert_eq!(rng.next_u64(), 0xa67d_f28d_0478_4f52);
        let mut rng = KpiRng::new(42);
        let u = rng.uniform();
        assert_eq!(
            u,
            ((0x48df_88e7_37a0_beaf_u64 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        );
        assert_eq!(KpiRng::new(42).below(1000), (u * 1000.0) as usize);
    }

    #[test]
    fn draws_are_deterministic_and_well_formed() {
        let mut a = KpiRng::new(7);
        let mut b = KpiRng::new(7);
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for _ in 0..4000 {
            let (u, z) = (a.uniform(), a.normal());
            assert_eq!(u, b.uniform());
            assert_eq!(z, b.normal());
            assert!(u > 0.0 && u < 1.0);
            sum += z;
            sum_sq += z * z;
        }
        let mean: f64 = sum / 4000.0;
        assert!(mean.abs() < 0.06, "normal mean {mean}");
        assert!((sum_sq / 4000.0 - 1.0).abs() < 0.1);
        assert_eq!(KpiRng::new(1).below(0), 0);
        assert!((0..200).all(|_| a.below(3) < 3));
    }

    #[test]
    fn shuffle_and_sampling_are_permutations() {
        let mut rng = KpiRng::new(11);
        let mut items: Vec<usize> = (0..100).collect();
        rng.shuffle(&mut items);
        assert_ne!(items, (0..100).collect::<Vec<_>>());
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());

        let picks = rng.sample_without_replacement(60, 48);
        assert_eq!(picks.len(), 48);
        let mut unique = picks.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 48);
        assert!(unique.iter().all(|&i| i < 60));
        assert_eq!(
            KpiRng::new(5).sample_without_replacement(60, 48),
            KpiRng::new(5).sample_without_replacement(60, 48)
        );
        assert_ne!(
            KpiRng::new(5).sample_without_replacement(60, 48),
            KpiRng::new(6).sample_without_replacement(60, 48)
        );
        let mut all = KpiRng::new(5).sample_without_replacement(4, 9);
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3]);
        assert!(KpiRng::new(5).sample_without_replacement(0, 3).is_empty());

        // Every index is reachable: 2000 draws of 1-of-8 hit all eight values.
        let mut seen = [false; 8];
        let mut rng = KpiRng::new(3);
        for _ in 0..2000 {
            seen[rng.sample_without_replacement(8, 1)[0]] = true;
        }
        assert!(seen.iter().all(|&hit| hit));
    }
}
