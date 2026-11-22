//! Scale card tests: parity with the Python scaler, exact inverse, constant
//! columns, grid snapping and JSON round trip.

use super::scale::{ColumnScale, ScaleCard, TargetScale};
use super::testkit::{n01, table, u01};

const NAN: f64 = f64::NAN;

/// FIT rows of the reference run (4 columns: mixed, outlier + missing, constant, one finite cell).
const FIT: [[f64; 4]; 10] = [
    [1.5, 10.0, 7.0, NAN],
    [2.25, 12.0, 7.0, NAN],
    [-0.75, 11.0, 7.0, NAN],
    [3.0, NAN, 7.0, 4.0],
    [0.5, 10.0, 7.0, NAN],
    [250.0, 13.0, 7.0, NAN],
    [1.0, 10.0, 7.0, NAN],
    [-40.0, 14.0, 7.0, NAN],
    [2.0, 9.0, 7.0, NAN],
    [1.25, 1.0e6, 7.0, NAN],
];
const QUERY: [[f64; 4]; 3] = [
    [0.0, 11.5, 7.0, 9.0],
    [1e9, NAN, 7.5, NAN],
    [-3.0, -5.0, 6.0, 1.0],
];
/// Output of `_fit_only_nc2_scale(FIT, QUERY)` (numpy 1.26.4, float64 before the float32 cast).
const FIT_UNIT: [[f64; 4]; 10] = [
    [0.0016239158555705034, -3.1820159152291567e-6, 0.0, 0.0],
    [0.011367410988993524, 3.1820159152291567e-6, 0.0, 0.0],
    [-0.027606569544698558, 0.0, 0.0, 0.0],
    [0.021110906122416544, 0.0, 0.0, 0.0],
    [-0.011367410988993524, -3.1820159152291567e-6, 0.0, 0.0],
    [3.227988406560174, 6.3640318304583135e-6, 0.0, 0.0],
    [-0.00487174756671151, -3.1820159152291567e-6, 0.0, 0.0],
    [-0.5375161481938366, 9.54604774568747e-6, 0.0, 0.0],
    [0.008119579277852517, -6.3640318304583135e-6, 0.0, 0.0],
    [-0.0016239158555705034, 3.1809911481423963, 0.0, 0.0],
];
const QUERY_UNIT: [[f64; 4]; 3] = [
    [-0.017863074411275538, 1.5910079576145784e-6, 0.0, 0.0],
    [6.0, 0.0, 6.0, 0.0],
    [-0.05683705494496762, -5.091225464366651e-5, -6.0, 0.0],
];

fn reference_card() -> ScaleCard {
    let flat: Vec<f64> = FIT.iter().flatten().copied().collect();
    ScaleCard::fit(&flat, 10, 4).unwrap()
}

#[test]
fn to_unit_matches_the_python_scaler() {
    let card = reference_card();
    let mut worst = 0.0f64;
    for (rows, want) in [(&FIT[..], &FIT_UNIT[..]), (&QUERY[..], &QUERY_UNIT[..])] {
        for (row, expect) in rows.iter().zip(want) {
            let flat: Vec<f64> = row.to_vec();
            let got = card.to_unit_rows(&flat).unwrap();
            for (g, e) in got.iter().zip(expect) {
                worst = worst.max((g - e).abs() / e.abs().max(1e-12));
                assert!((g - e).abs() <= 1e-12 * e.abs().max(1.0), "{g} vs {e}");
                // The Python pipeline casts to float32 last; the casts agree as well.
                assert_eq!(*g as f32, *e as f32);
            }
        }
    }
    println!("scale parity: worst relative difference to numpy {worst:.2e}");
    assert!(card.columns[2].constant && card.columns[3].constant);
    assert_eq!((card.columns[2].scale, card.columns[3].scale), (1e-6, 0.0));
}

#[test]
fn to_raw_inverts_to_unit_inside_the_clip() {
    let (n, p) = (4000, 3);
    // Bulk, heavy right tail, and a shifted column with a small spread.
    let raw = table(n, p, |i, j| match j {
        0 => 3.0 * n01(1, 0, i) - 7.0,
        1 => (1.2 * n01(1, 1, i)).exp(),
        _ => 1.0e6 + 0.01 * n01(1, 2, i),
    });
    let card = ScaleCard::fit(&raw, n, p).unwrap();
    let (mut worst, mut checked, mut tail) = (0.0f64, 0usize, 0usize);
    for (i, &x) in raw.iter().enumerate() {
        let col = &card.columns[i % p];
        let u = col.to_unit(x);
        if u.abs() >= 6.0 {
            continue;
        }
        checked += 1;
        tail += (u.abs() > 3.0) as usize;
        // Both the float64 unit value and its float32 cast must invert.
        for unit in [u, u as f32 as f64] {
            let rel = (col.to_raw(unit) - x).abs() / x.abs().max(col.scale);
            worst = worst.max(rel);
        }
    }
    println!("scale round trip: worst relative error {worst:.2e} over {checked} cells ({tail} in the asinh tail)");
    assert!(worst < 1e-5 && checked > 11_000 && tail > 50);
    // Exact formula checks around the tail start and at the clip.
    let col = ColumnScale {
        median: 0.0,
        scale: 1.0,
        constant: false,
        grid: 0.0,
    };
    assert_eq!(col.to_unit(3.0), 3.0);
    assert!((col.to_unit(5.0) - (3.0 + 2f64.asinh())).abs() < 1e-15);
    assert!((col.to_unit(-5.0) + (3.0 + 2f64.asinh())).abs() < 1e-15);
    assert_eq!((col.to_unit(1e12), col.to_unit(-1e12)), (6.0, -6.0));
    assert!((col.to_raw(6.0) - (3.0 + 3f64.sinh())).abs() < 1e-12);
    assert_eq!(col.to_raw(9.0), col.to_raw(6.0));
    assert_eq!(col.to_unit(f64::NAN), 0.0);
    assert_eq!(col.to_unit(f64::INFINITY), 0.0);
}

#[test]
fn constant_columns_return_their_constant() {
    let card = reference_card();
    for u in [-6.0, -0.3, 0.0, 2.0, 6.0] {
        assert_eq!(card.columns[2].to_raw(u), 7.0);
        assert_eq!(card.columns[3].to_raw(u), 4.0);
    }
    // A column with no finite FIT cell falls back to 0.
    let empty = ColumnScale::fit([f64::NAN, f64::NAN, f64::NAN].into_iter());
    assert!(empty.constant && empty.to_raw(1.0) == 0.0 && empty.to_unit(5.0) == 0.0);
}

#[test]
fn decimal_grids_are_detected_and_restored() {
    let n = 500;
    let integers = table(n, 1, |i, _| (40.0 + 12.0 * n01(2, 0, i)).round());
    // Cents stored as float32, the arena cell type.
    let cents = table(n, 1, |i, _| {
        ((1999.0 * u01(2, 1, i)).round() / 100.0) as f32 as f64
    });
    let thousands = table(n, 1, |i, _| (30.0 + 200.0 * u01(2, 2, i)).round() * 1000.0);
    let smooth = table(n, 1, |i, _| n01(2, 3, i));
    let step = |v: &[f64]| ColumnScale::fit(v.iter().copied()).grid;
    assert_eq!(step(&integers), 1.0);
    assert_eq!(step(&cents), 0.01);
    assert_eq!(step(&thousands), 1000.0);
    assert_eq!(step(&smooth), 0.0);
    for values in [&integers, &cents, &thousands] {
        let col = ColumnScale::fit(values.iter().copied());
        for &x in values.iter() {
            let u = col.to_unit(x);
            if u.abs() < 6.0 {
                // Even after the float32 cast of the unit value, the raw cell comes back bit-exact.
                assert_eq!(col.to_raw(u as f32 as f64) as f32, x as f32);
            }
        }
        // Any synthetic unit value lands on the grid.
        let x = col.to_raw(0.123_456_789);
        assert!(((x / col.grid).round() * col.grid - x).abs() < 1e-9);
    }
}

#[test]
fn json_round_trip_is_bit_exact() {
    let raw = table(64, 3, |i, j| match j {
        0 => 0.1 * n01(3, 0, i) + 1.0 / 3.0,
        1 => (i % 5) as f64,
        _ => 2.5,
    });
    let y: Vec<f64> = (0..64).map(|i| n01(3, 9, i) * 1e-7 + 123.456).collect();
    let card = ScaleCard::fit(&raw, 64, 3)
        .unwrap()
        .with_regression_target(&y)
        .unwrap();
    let text = card.to_json().unwrap();
    let back = ScaleCard::from_json(&text).unwrap();
    assert_eq!(back, card);
    for (a, b) in card.columns.iter().zip(&back.columns) {
        assert_eq!(a.median.to_bits(), b.median.to_bits());
        assert_eq!(a.scale.to_bits(), b.scale.to_bits());
    }
    let TargetScale::Regression { mean, sd } = back.target else {
        panic!("target kind lost");
    };
    assert!((card.target_to_raw(card.target_to_unit(y[5])) - y[5]).abs() < 1e-9);
    assert!((mean - 123.456).abs() < 1e-6 && sd > 0.0);
    assert_eq!(card.target_to_unit(1e9), 6.0);
    let classes = card.clone().with_classes(vec![0.0, 1.0, 5.0]);
    let back = ScaleCard::from_json(&classes.to_json().unwrap()).unwrap();
    assert_eq!(back, classes);
    assert_eq!(
        (back.target_to_raw(2.0), back.target_to_raw(0.0)),
        (5.0, 0.0)
    );
    assert!(ScaleCard::from_json("{}").is_err());
    assert!(ScaleCard::from_json(&text.replace("scale-card/v1", "scale-card/v9")).is_err());
}
