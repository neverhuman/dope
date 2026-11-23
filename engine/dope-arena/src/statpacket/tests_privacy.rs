//! Privacy-by-construction tests: planted canaries (a unique extreme value, a
//! level carried by fewer than 10 rows) never reach the packet bytes or the
//! decoded rows, and the packet does not depend on how extreme an extreme is.

use super::codec::pack;
use super::decode::Decoder;
use super::marginals::{atom_values, decoded_knots, fit_column, knot_plan, smoothed_knots};
use super::packet::{ColumnMarginal, PacketLevel};
use super::testkit::{column, fit_regression, n01, sorted, table, to_unit, u01};

const DECODED_ROWS: usize = 100_000;

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// Little- and big-endian float32 and float64 encodings of `v`.
fn float_encodings(v: f64) -> Vec<Vec<u8>> {
    vec![
        v.to_le_bytes().to_vec(),
        v.to_be_bytes().to_vec(),
        (v as f32).to_le_bytes().to_vec(),
        (v as f32).to_be_bytes().to_vec(),
    ]
}

#[test]
fn a_planted_unique_extreme_never_appears() {
    let (n, p) = (2000usize, 6usize);
    // Three canaries: a far maximum, a far minimum, and a maximum inside the unit clip.
    let canaries = [
        (777usize, 2usize, 987_654.312_5f64),
        (1234, 4, -31_337.125),
        (55, 0, 7.75),
    ];
    let raw = table(n, p, |i, j| {
        if let Some(c) = canaries.iter().find(|c| (c.0, c.1) == (i, j)) {
            return c.2;
        }
        match j {
            0 => n01(31, 0, i),
            1 => (0.7 * n01(31, 1, i)).exp(),
            2 => 50.0 + 10.0 * n01(31, 2, i) + 4.0 * n01(31, 0, i),
            3 => u01(31, 3, i),
            4 => -(-u01(31, 4, i).ln()) * 3.0,
            _ => n01(31, 5, i) - 0.5 * n01(31, 1, i),
        }
    });
    let (card, unit) = to_unit(&raw, n, p);
    let packet = fit_regression(&unit, n, p);
    let bytes = pack(&packet, PacketLevel::E).unwrap();
    let decoder = Decoder::new(&packet).unwrap();
    let raw_rows = decoder.rows_raw(&card, 4, 0, DECODED_ROWS).unwrap();
    let unit_rows = decoder.rows_unit(4, 0, DECODED_ROWS);
    for (row, j, value) in canaries {
        let col = column(&raw, p, j);
        let extreme = if value > 0.0 {
            col.iter().all(|&v| v <= value)
        } else {
            col.iter().all(|&v| v >= value)
        };
        assert!(extreme && col.iter().filter(|&&v| v == value).count() == 1);
        let in_unit = unit[row * p + j];
        // 1. No float encoding of the raw value, or of its unit value, is in the packet.
        let mut needles = float_encodings(value);
        if in_unit.abs() < 6.0 {
            needles.extend(float_encodings(in_unit));
        }
        for needle in &needles {
            assert!(!contains(&bytes, needle), "canary bytes found (column {j})");
        }
        // 2. Every stored knot lies strictly inside the canary.
        let knots = decoded_knots(&packet.columns[j], 17);
        assert_eq!(knots.len(), 17);
        assert!(
            knots.iter().all(|k| (k - in_unit).abs() > 0.5),
            "column {j}"
        );
        // 3. No decoded cell equals it, in raw or in unit space.
        let decoded_raw = column(&raw_rows, p, j);
        let decoded_unit = column(&unit_rows, p, j);
        assert!(decoded_raw.iter().all(|&v| v != value as f32 as f64));
        assert!(decoded_unit.iter().all(|&v| v != in_unit as f32 as f64) || in_unit.abs() == 6.0);
        // Far canaries are not even approached.
        if value.abs() > 1000.0 {
            assert!(decoded_raw.iter().all(|&v| v.abs() < value.abs() / 2.0));
        }
    }
    println!(
        "extreme canaries: packet {} B, {DECODED_ROWS} rows decoded, none found",
        bytes.len()
    );
}

#[test]
fn a_level_with_fewer_than_ten_rows_never_appears() {
    let (n, p) = (3000usize, 3usize);
    // Column 0: levels 0..3 plus 7.25 on 9 rows (rare and extreme) and 4 on exactly 10 rows.
    // Column 1: levels 1, 2, 3 plus an interior 2.5 on 9 rows.
    let raw = table(n, p, |i, j| match j {
        0 if i < 9 => 7.25,
        0 if i < 19 => 4.0,
        0 => [0.4, 0.7, 0.9]
            .iter()
            .filter(|&&c| u01(32, 0, i) >= c)
            .count() as f64,
        1 if (100..109).contains(&i) => 2.5,
        1 => 1.0 + [0.3, 0.8].iter().filter(|&&c| u01(32, 1, i) >= c).count() as f64,
        _ => n01(32, 2, i),
    });
    let (card, unit) = to_unit(&raw, n, p);
    let packet = fit_regression(&unit, n, p);
    let bytes = pack(&packet, PacketLevel::E).unwrap();
    let decoder = Decoder::new(&packet).unwrap();
    let raw_rows = decoder.rows_raw(&card, 6, 0, DECODED_ROWS).unwrap();
    let unit_rows = decoder.rows_unit(6, 0, DECODED_ROWS);
    for (j, rare, kept) in [
        (0usize, 7.25f64, vec![0.0, 1.0, 2.0, 3.0, 4.0]),
        (1, 2.5, vec![1.0, 2.0, 3.0]),
    ] {
        let rare_unit = card.columns[j].to_unit(rare);
        let ColumnMarginal::Discrete(atoms) = &packet.columns[j] else {
            panic!("column {j} should be discrete");
        };
        // The atom table holds exactly the supported levels (a 10-row level survives).
        let values = atom_values(atoms);
        assert_eq!(values.len(), kept.len());
        for (v, level) in values.iter().zip(&kept) {
            assert!((v - card.columns[j].to_unit(*level)).abs() < 0.005);
        }
        assert!(values.iter().all(|v| (v - rare_unit).abs() > 0.05));
        for needle in float_encodings(rare)
            .iter()
            .chain(&float_encodings(rare_unit))
        {
            assert!(
                !contains(&bytes, needle),
                "rare level bytes found (column {j})"
            );
        }
        // Decoded rows hold supported levels only.
        let decoded = column(&raw_rows, p, j);
        assert!(decoded.iter().all(|v| kept.contains(v)), "column {j}");
        assert!(kept.iter().all(|l| decoded.contains(l)));
        let decoded_unit = column(&unit_rows, p, j);
        assert!(decoded_unit.iter().all(|v| (v - rare_unit).abs() > 0.05));
    }
    println!(
        "rare-level canaries: packet {} B, {DECODED_ROWS} rows decoded, none found",
        bytes.len()
    );
}

#[test]
fn knot_windows_never_contain_the_extreme_rows() {
    for n in [
        16usize, 17, 25, 31, 32, 63, 64, 100, 127, 128, 255, 256, 1000, 1024, 5000,
    ] {
        let plan = knot_plan(n.ilog2() as u8);
        // Every row is 0 except the two extremes: any window touching them moves a knot.
        let mut col = vec![0.0; n];
        col[0] = -6.0;
        col[n - 1] = 6.0;
        let knots = smoothed_knots(&col, &plan);
        assert!(knots.iter().all(|&k| k == 0.0), "n = {n}");
    }
}

#[test]
fn the_packet_ignores_how_extreme_the_extremes_are() {
    let (n, p) = (1500usize, 5usize);
    let raw = table(n, p, |i, j| n01(33, j, i) + 0.5 * n01(33, 9, i));
    let (_, unit) = to_unit(&raw, n, p);
    let baseline = pack(&fit_regression(&unit, n, p), PacketLevel::E).unwrap();
    // Push each column's smallest and largest cell to the clip: ranks are unchanged.
    let mut pushed = unit.clone();
    for j in 0..p {
        let col = column(&unit, p, j);
        let lo = (0..n).min_by(|&a, &b| col[a].total_cmp(&col[b])).unwrap();
        let hi = (0..n).max_by(|&a, &b| col[a].total_cmp(&col[b])).unwrap();
        pushed[lo * p + j] = -6.0;
        pushed[hi * p + j] = 6.0;
    }
    assert_ne!(pushed, unit);
    assert_eq!(
        pack(&fit_regression(&pushed, n, p), PacketLevel::E).unwrap(),
        baseline
    );
}

#[test]
fn one_row_moves_a_knot_by_at_most_a_local_order_statistic_gap() {
    let n = 2000usize;
    let raw: Vec<f64> = (0..n).map(|i| (0.5 * n01(34, 0, i)).exp()).collect();
    let (_, values) = to_unit(&raw, n, 1);
    let ordered = sorted(values.clone());
    let (plan, plan_less) = (knot_plan(10), knot_plan((n - 1).ilog2() as u8));
    assert_eq!(plan, plan_less);
    let before = smoothed_knots(&ordered, &plan);
    // Removing one row shifts every window by at most one rank, so a knot moves by at
    // most the largest gap between neighbouring order statistics (checked for all rows).
    let gap = ordered.windows(2).map(|w| w[1] - w[0]);
    let widest_inner_gap = gap.skip(1).take(n - 3).fold(0.0, f64::max);
    let mut worst = 0.0f64;
    for drop in 0..n {
        let mut without = ordered.clone();
        without.remove(drop);
        let after = smoothed_knots(&without, &plan);
        for (a, b) in before.iter().zip(&after) {
            worst = worst.max((a - b).abs());
        }
    }
    assert!(worst <= widest_inner_gap, "{worst} > {widest_inner_gap}");
    // The stored (quantised) knots add at most one code step per nesting level.
    let all = decoded_knots(&fit_column(&values, &plan), 17);
    let mut worst_stored = 0.0f64;
    for target in [
        ordered[0],
        ordered[n - 1],
        ordered[n / 2],
        values[3],
        values[1500],
    ] {
        let without: Vec<f64> = values.iter().copied().filter(|&v| v != target).collect();
        assert_eq!(without.len(), n - 1);
        let knots = decoded_knots(&fit_column(&without, &plan), 17);
        for (a, b) in all.iter().zip(&knots) {
            worst_stored = worst_stored.max((a - b).abs());
        }
    }
    println!(
        "leave-one-out: smoothed knots move at most {worst:.4} (widest inner gap \
         {widest_inner_gap:.4}); stored knots moved at most {worst_stored:.4}"
    );
    assert!(worst_stored <= 0.15);
}
