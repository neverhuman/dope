//! Behavioral tests for Phase P3 Column Realism (`T06`, `T07`, `T08`, `T27`, `T28`).

use dope_core::dvs::v3::realism::column_marginal_auc;
use dope_core::marginal::atoms::DEFAULT_SUPPORT_K;
use dope_core::marginal::v2::{tier_discrete_cap, ColumnMarginalV2};
use dope_core::rng::{row_keyed_uniform, DEFAULT_NAMESPACE_128};

fn draw_column(m: &ColumnMarginalV2, n: usize, seed: u64) -> Vec<f64> {
    (0..n)
        .map(|r| {
            let u_main = row_keyed_uniform(&DEFAULT_NAMESPACE_128, seed, 0, r as u64, 0, 1, 0);
            let u_mix = row_keyed_uniform(&DEFAULT_NAMESPACE_128, seed, 0, r as u64, 0, 2, 0);
            let u_miss = row_keyed_uniform(&DEFAULT_NAMESPACE_128, seed, 0, r as u64, 0, 3, 0);
            m.sample_from_uniform(u_main, u_mix, u_miss)
        })
        .collect()
}

#[test]
fn t06_missingness_rates_are_preserved_accurately() {
    for &rate in &[0.0_f64, 0.15, 0.60] {
        let n = 1000usize;
        let col: Vec<f64> = (0..n)
            .map(|i| {
                if (i as f64 / n as f64) < rate {
                    f64::NAN
                } else {
                    (i as f64) * 0.037
                }
            })
            .collect();
        let m = ColumnMarginalV2::fit(&col, 24, 24, DEFAULT_SUPPORT_K);
        let synth = draw_column(&m, 4000, 11);
        let obs_miss = synth.iter().filter(|v| !v.is_finite()).count() as f64 / synth.len() as f64;
        assert!(
            (obs_miss - rate).abs() < 0.03,
            "expected missingness {rate}, got {obs_miss}"
        );
    }
}

#[test]
fn t07_atoms_respect_support_floor_k10_and_preserve_point_masses() {
    // 200 zeros (exact atom >= k=10), 150 ones (exact atom), 5 private rare 999.0 (< k=10),
    // plus 645 smooth values spanning (0.01..9.99).
    let mut col = Vec::with_capacity(1000);
    col.extend(std::iter::repeat_n(0.0, 200));
    col.extend(std::iter::repeat_n(1.0, 150));
    col.extend(std::iter::repeat_n(999.0, 5)); // below k=10 -> must NOT be stored as atom!
    for i in 0..645 {
        col.push(0.01 + (i as f64 / 645.0) * 9.5);
    }
    let m = ColumnMarginalV2::fit(&col, 24, 24, DEFAULT_SUPPORT_K);
    match &m {
        ColumnMarginalV2::Continuous { atoms, .. } => {
            assert!(atoms.iter().any(|a| (a.value - 0.0).abs() < 1e-12));
            assert!(atoms.iter().any(|a| (a.value - 1.0).abs() < 1e-12));
            assert!(
                !atoms.iter().any(|a| (a.value - 999.0).abs() < 1e-12),
                "rare value < k=10 leaked into atoms"
            );
        }
        _ => panic!("expected continuous marginal"),
    }
    let synth = draw_column(&m, 4000, 23);
    let zero_share = synth.iter().filter(|&&v| v == 0.0).count() as f64 / synth.len() as f64;
    assert!((zero_share - 0.20).abs() < 0.03);
    // k-th order stat tail anchor ensures rare 999.0 outlier is not emitted.
    assert!(synth.iter().all(|&v| v < 50.0));
}

#[test]
fn t08_w1_tails_capture_heavy_skew_with_low_marginal_auc() {
    // Heavy right-skewed lognormal-like column (1500 rows) + extreme singleton outlier.
    let mut col: Vec<f64> = (1..=1500)
        .map(|i| {
            let u = i as f64 / 1501.0;
            let z = dope_core::stats::norm_ppf(u);
            (1.2 * z).exp()
        })
        .collect();
    col.push(1.0e9); // single private extreme outlier must NOT anchor tail!
    let m = ColumnMarginalV2::fit(&col, 28, 24, DEFAULT_SUPPORT_K);
    let synth = draw_column(&m, 1501, 41);
    assert!(synth.iter().all(|&v| v < 1.0e6), "singleton leaked to tail");
    let auc = column_marginal_auc(&col, &synth).unwrap();
    assert!(
        auc <= 0.60,
        "W1 heavy-tail marginal must keep MargAUC <= 0.60, got {auc}"
    );
}

#[test]
fn t27_decimal_lattice_and_heaping_preserve_exact_decimals_and_f32_heritage() {
    // 1. Exact decimal cents lattice (q0 = 0.01) with heaping at multiples of 10 (0.10).
    let mut cents = Vec::with_capacity(1200);
    for i in 0..1200 {
        let base = if i % 3 == 0 {
            ((i % 50) * 10) as f64 * 0.01
        } else {
            (i % 500) as f64 * 0.01
        };
        cents.push((base * 100.0).round() / 100.0);
    }
    let m = ColumnMarginalV2::fit(&cents, 24, 24, DEFAULT_SUPPORT_K);
    match &m {
        ColumnMarginalV2::Continuous {
            lattice: Some(lat),
            heaping: Some(heap),
            ..
        } => {
            assert!((lat.step - 0.01).abs() < 1e-12);
            assert_eq!(lat.pi_offgrid, 0.0);
            assert_eq!(heap.multiple, 10);
        }
        other => panic!("expected decimal lattice + heaping, got {other:?}"),
    }
    let synth = draw_column(&m, 1200, 7);
    for &v in &synth {
        let rounded = (v * 100.0).round() / 100.0;
        assert!((v - rounded).abs() < 1e-12, "non-decimal drift {v}");
    }
    let auc = column_marginal_auc(&cents, &synth).unwrap();
    assert!(auc <= 0.60, "decimal lattice MargAUC too high: {auc}");

    // 2. f32-heritage values (0.1f32, 0.2f32, ...) round-trip to canonical f64::from(f32).
    let f32_col: Vec<f64> = (0..600)
        .map(|i| f64::from(((i % 200) as f32) * 0.1_f32))
        .collect();
    let m32 = ColumnMarginalV2::fit(&f32_col, 24, 24, DEFAULT_SUPPORT_K);
    match &m32 {
        ColumnMarginalV2::Continuous {
            lattice: Some(lat), ..
        } => {
            assert!(lat.f32_heritage);
            assert!((lat.step - 0.1).abs() < 1e-9);
        }
        other => panic!("expected f32-heritage lattice, got {other:?}"),
    }
    let s32 = draw_column(&m32, 400, 9);
    for &v in &s32 {
        assert_eq!(v.to_bits(), f64::from(v as f32).to_bits());
    }
}

#[test]
fn t28_hybrid_lattice_preserves_nonzero_offgrid_law_and_tier_discrete_caps() {
    assert_eq!(tier_discrete_cap(4096), 24);
    assert_eq!(tier_discrete_cap(16384), 64);
    assert_eq!(tier_discrete_cap(65536), 256);
    assert_eq!(tier_discrete_cap(1_048_576), 4096);

    // 90% integers (q0 = 1.0) + 10% continuous off-grid values (never silently snapped away!).
    let mut col = Vec::with_capacity(1000);
    for i in 0..900 {
        col.push((i % 100) as f64);
    }
    for i in 0..100 {
        col.push((i % 100) as f64 + 0.371 + (i as f64) * 0.0011);
    }
    let m = ColumnMarginalV2::fit(&col, 24, 24, DEFAULT_SUPPORT_K);
    match &m {
        ColumnMarginalV2::Continuous {
            lattice: Some(lat), ..
        } => {
            assert!((lat.step - 1.0).abs() < 1e-12);
            assert!(
                (lat.pi_offgrid - 0.10).abs() < 0.02,
                "expected pi_offgrid ~ 0.10, got {}",
                lat.pi_offgrid
            );
        }
        other => panic!("expected hybrid lattice, got {other:?}"),
    }
    let synth = draw_column(&m, 2000, 17);
    let offgrid_frac = synth
        .iter()
        .filter(|&&v| (v - v.round()).abs() > 1e-3)
        .count() as f64
        / synth.len() as f64;
    assert!(
        (offgrid_frac - 0.10).abs() < 0.03,
        "expected ~10% off-grid draws, got {offgrid_frac}"
    );
    let auc = column_marginal_auc(&col, &synth).unwrap();
    assert!(auc <= 0.60, "hybrid lattice MargAUC too high: {auc}");
}
