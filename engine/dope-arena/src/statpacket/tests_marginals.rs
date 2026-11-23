//! Marginal tests: knot plan, nesting, quantile fidelity at 5 / 9 / 17 knots,
//! atom tables and level frequencies, constant columns.

use super::codec::{pack, truncate, unpack};
use super::decode::Decoder;
use super::marginals::{
    atom_masses, atom_values, decoded_knots, fit_column, knot_plan, knot_probability,
    simplex_codes, ColumnSampler,
};
use super::packet::{AtomValues, ColumnMarginal, PacketLevel};
use super::testkit::{column, fit_regression, n01, quantile, sorted, table, to_unit, u01};

#[test]
fn knot_plan_is_a_function_of_the_row_count_exponent() {
    for (e, k, half, u_lo) in [
        (4u8, 3usize, 2usize, 0.1875),
        (5, 3, 2, 0.09375),
        (6, 6, 3, 0.09375),
        (7, 10, 5, 0.078125),
        (8, 10, 5, 0.0390625),
        (9, 10, 5, 0.01953125),
        (10, 10, 5, 0.01),
        (30, 10, 5, 0.01),
    ] {
        let plan = knot_plan(e);
        assert_eq!(
            (plan.window_k, plan.half, plan.u_lo),
            (k, half, u_lo),
            "e = {e}"
        );
        assert!((knot_probability(&plan, 0) - u_lo).abs() < 1e-6);
        assert!((knot_probability(&plan, 16) - (1.0 - u_lo)).abs() < 1e-6);
        assert!((knot_probability(&plan, 8) - 0.5).abs() < 1e-6);
    }
    assert!((knot_plan(12).z_hi - 2.326_347_874_040_841).abs() < 1e-12);
}

#[test]
fn knots_are_exactly_nested_and_monotone() {
    let n = 2000;
    let raw: Vec<f64> = (0..n).map(|i| (0.6 * n01(4, 0, i)).exp()).collect();
    let (_, values) = to_unit(&raw, n, 1);
    let plan = knot_plan(10);
    let col = fit_column(&values, &plan);
    assert!(col.is_continuous());
    let (k5, k9, k17) = (
        decoded_knots(&col, 5),
        decoded_knots(&col, 9),
        decoded_knots(&col, 17),
    );
    assert_eq!((k5.len(), k9.len(), k17.len()), (5, 9, 17));
    for i in 0..5 {
        assert_eq!(k5[i].to_bits(), k9[2 * i].to_bits());
        assert_eq!(k5[i].to_bits(), k17[4 * i].to_bits());
    }
    for i in 0..9 {
        assert_eq!(k9[i].to_bits(), k17[2 * i].to_bits());
    }
    assert!(k17.windows(2).all(|w| w[0] <= w[1]));
    // The curve passes through its knots at every level.
    for (level, knots) in [
        (PacketLevel::A, &k5),
        (PacketLevel::C, &k9),
        (PacketLevel::E, &k17),
    ] {
        let curve = ColumnSampler::new(&col, level, &plan);
        let last = knots.len() - 1;
        for (i, k) in knots.iter().enumerate() {
            let z = plan.z_hi * (2.0 * i as f64 / last as f64 - 1.0);
            assert!((curve.eval(z) - k).abs() < 1e-9);
        }
    }
}

/// Largest quantile gap over the probabilities 0.02, 0.03, .. 0.98.
fn quantile_gap(fit: &[f64], decoded: &[f64]) -> f64 {
    let (a, b) = (sorted(fit.to_vec()), sorted(decoded.to_vec()));
    (2..=98)
        .map(|q| (quantile(&a, q as f64 / 100.0) - quantile(&b, q as f64 / 100.0)).abs())
        .fold(0.0, f64::max)
}

fn gaps_by_level(unit: &[f64], n: usize, p: usize) -> Vec<[f64; 3]> {
    let bytes = pack(&fit_regression(unit, n, p), PacketLevel::E).unwrap();
    let mut out = vec![[0.0; 3]; p];
    for (slot, level) in [PacketLevel::B, PacketLevel::C, PacketLevel::E]
        .into_iter()
        .enumerate()
    {
        let decoder = Decoder::new(&unpack(&truncate(&bytes, level).unwrap()).unwrap()).unwrap();
        let rows = decoder.rows_unit(77, 0, 100_000);
        for (j, gaps) in out.iter_mut().enumerate() {
            gaps[slot] = quantile_gap(&column(unit, p, j), &column(&rows, p, j));
        }
    }
    out
}

#[test]
fn decoded_quantiles_match_the_fit_rows() {
    let (n, p) = (2000, 5);
    // Gaussian, right-skewed, flat, one-sided, heavy-tailed.
    let raw = table(n, p, |i, j| match j {
        0 => n01(5, 0, i),
        1 => (0.8 * n01(5, 1, i)).exp(),
        2 => u01(5, 2, i),
        3 => -u01(5, 3, i).ln(),
        _ => n01(5, 4, i) / (0.25 + u01(5, 5, i)).sqrt(),
    });
    let (_, unit) = to_unit(&raw, n, p);
    let gaps = gaps_by_level(&unit, n, p);
    let worst = |slot: usize| gaps.iter().map(|g| g[slot]).fold(0.0, f64::max);
    println!(
        "quantile gap (unit space, q in 0.02..0.98, n_fit 2000, 100k decoded): \
         5 knots {:.4}, 9 knots {:.4}, 17 knots {:.4}; per column at 17: {:?}",
        worst(0),
        worst(1),
        worst(2),
        gaps.iter()
            .map(|g| (g[2] * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    assert!(worst(2) <= 0.1, "17 knots: {}", worst(2));
    assert!(worst(1) <= 0.2 && worst(0) <= 0.4);
    assert!(worst(2) <= worst(1) && worst(1) <= worst(0));
}

#[test]
fn a_gap_between_two_modes_is_the_known_weak_case() {
    let n = 2000;
    let raw = table(n, 1, |i, _| {
        let side = if u01(6, 0, i) < 0.5 { -2.0 } else { 2.0 };
        side + 0.5 * n01(6, 1, i)
    });
    let (_, unit) = to_unit(&raw, n, 1);
    let gap = gaps_by_level(&unit, n, 1)[0];
    println!(
        "bimodal quantile gap: 5 knots {:.3}, 9 knots {:.3}, 17 knots {:.3}",
        gap[0], gap[1], gap[2]
    );
    // Piecewise-linear quantiles put mass into an empty gap; bounded, not small.
    assert!(gap[2] < 0.6);
}

fn level_shares(values: &[f64], levels: &[f64]) -> Vec<f64> {
    levels
        .iter()
        .map(|l| values.iter().filter(|&&v| v == *l).count() as f64 / values.len() as f64)
        .collect()
}

#[test]
fn discrete_columns_reproduce_level_frequencies() {
    let (n, p, decoded_rows) = (2000usize, 4usize, 50_000usize);
    let pick = |u: f64, cuts: &[f64]| cuts.iter().filter(|&&c| u >= c).count() as f64;
    let raw = table(n, p, |i, j| match j {
        0 => pick(u01(7, 0, i), &[0.4, 0.7, 0.9]),
        1 => pick(u01(7, 1, i), &[0.7]),
        2 => 5.0 * pick(u01(7, 2, i), &[0.15, 0.35, 0.6, 0.8, 0.92]),
        _ => n01(7, 3, i),
    });
    let (card, unit) = to_unit(&raw, n, p);
    let packet = fit_regression(&unit, n, p);
    let lattice = |j: usize| match &packet.columns[j] {
        ColumnMarginal::Discrete(t) => {
            Some((t.prob.len(), matches!(t.values, AtomValues::Lattice { .. })))
        }
        _ => None,
    };
    assert_eq!(
        (lattice(0), lattice(1), lattice(2), lattice(3)),
        (Some((4, false)), Some((2, false)), Some((6, true)), None)
    );
    let decoder = Decoder::new(&packet).unwrap();
    let rows = decoder.rows_raw(&card, 3, 0, decoded_rows).unwrap();
    let (mut worst_sampler, mut worst_quant, mut worst_total) = (0.0f64, 0.0f64, 0.0f64);
    for (j, levels) in [
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0, 1.0],
        vec![0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
    ]
    .iter()
    .enumerate()
    {
        let fit = level_shares(&column(&raw, p, j), levels);
        let got = level_shares(&column(&rows, p, j), levels);
        // Every decoded cell is exactly one of the FIT levels.
        assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-12, "column {j}");
        let ColumnMarginal::Discrete(t) = &packet.columns[j] else {
            panic!("column {j} is not discrete");
        };
        let mass = atom_masses(t);
        for a in 0..levels.len() {
            let se_sampler = (mass[a] * (1.0 - mass[a]) / decoded_rows as f64).sqrt();
            let se_total =
                (fit[a] * (1.0 - fit[a]) * (1.0 / n as f64 + 1.0 / decoded_rows as f64)).sqrt();
            worst_sampler = worst_sampler.max((got[a] - mass[a]).abs() / se_sampler);
            worst_quant = worst_quant.max((mass[a] - fit[a]).abs());
            worst_total = worst_total.max((got[a] - fit[a]).abs() / se_total);
        }
    }
    println!(
        "level frequencies: decoded vs packet mass {worst_sampler:.2} SE, decoded vs FIT \
         {worst_total:.2} SE, mass quantisation error {worst_quant:.4}"
    );
    assert!(worst_sampler <= 3.0 && worst_total <= 3.0 && worst_quant <= 0.01);
}

#[test]
fn rare_levels_merge_and_weak_tables_fall_back_to_knots() {
    let plan = knot_plan(9);
    let build = |counts: &[(f64, usize)]| -> Vec<f64> {
        counts
            .iter()
            .flat_map(|&(v, c)| std::iter::repeat_n(v, c))
            .collect()
    };
    // Level 0.2 (9 rows) is nearer to 0.1 than to 0.5; level 0.3 ties and goes down to 0.1 too.
    let col = fit_column(
        &build(&[(0.0, 500), (0.1, 291), (0.2, 9), (0.3, 9), (0.5, 191)]),
        &plan,
    );
    let ColumnMarginal::Discrete(t) = &col else {
        panic!("expected atoms");
    };
    let values = atom_values(t);
    assert_eq!(values.len(), 3);
    for (v, want) in values.iter().zip([0.0, 0.1, 0.5]) {
        assert!((v - want).abs() < 1e-3);
    }
    let mass = atom_masses(t);
    for (m, want) in mass.iter().zip([0.5, 0.309, 0.191]) {
        assert!((m - want).abs() < 0.01, "{m} vs {want}");
    }
    // One surviving level: continuous. Seventeen levels: continuous. One level: constant.
    let lone = build(&[(0.0, 990), (1.0, 5), (2.0, 5)]);
    assert!(fit_column(&lone, &plan).is_continuous());
    let many: Vec<f64> = (0..17 * 40).map(|i| (i % 17) as f64 / 10.0).collect();
    assert!(fit_column(&many, &plan).is_continuous());
    assert_eq!(fit_column(&[1.25; 64], &plan), ColumnMarginal::Constant);
    // Sixteen well-supported levels are still atoms.
    let sixteen: Vec<f64> = (0..16 * 40).map(|i| (i % 16) as f64 / 10.0).collect();
    assert!(matches!(
        fit_column(&sixteen, &plan),
        ColumnMarginal::Discrete(_)
    ));
}

#[test]
fn constant_columns_decode_to_the_constant() {
    let (n, p) = (300, 3);
    let raw = table(n, p, |i, j| match j {
        0 => 42.5,
        1 => n01(8, 1, i),
        _ => {
            if i == 17 {
                -3.0
            } else {
                f64::NAN
            }
        }
    });
    let (card, unit) = to_unit(&raw, n, p);
    let packet = fit_regression(&unit, n, p);
    assert_eq!(packet.columns[0], ColumnMarginal::Constant);
    assert_eq!(packet.columns[2], ColumnMarginal::Constant);
    let decoder = Decoder::new(&packet).unwrap();
    let unit_rows = decoder.rows_unit(1, 0, 5000);
    let raw_rows = decoder.rows_raw(&card, 1, 0, 5000).unwrap();
    assert!(column(&unit_rows, p, 0).iter().all(|&v| v == 0.0));
    assert!(column(&raw_rows, p, 0).iter().all(|&v| v == 42.5));
    assert!(column(&raw_rows, p, 2).iter().all(|&v| v == -3.0));
    let spread = sorted(column(&raw_rows, p, 1));
    assert!(spread[0] < -1.5 && spread[4999] > 1.5);
}

#[test]
fn simplex_codes_track_the_shares() {
    assert_eq!(simplex_codes(&[1.0, 1.0, 1.0, 1.0], 0), vec![1, 1, 1, 1]);
    assert_eq!(simplex_codes(&[0.0, 0.0], 0), vec![0, 0]);
    assert_eq!(simplex_codes(&[3.0, 1.0], 1), vec![3, 1]);
    assert!(simplex_codes(&[1000.0, 1.0], 1).iter().all(|&c| c >= 1));
    let mut worst = 0.0f64;
    for trial in 0..200 {
        let w: Vec<f64> = (0..2 + trial % 15).map(|i| u01(9, trial, i)).collect();
        let codes = simplex_codes(&w, 0);
        let (total, sum) = (
            w.iter().sum::<f64>(),
            codes.iter().map(|&c| c as f64).sum::<f64>(),
        );
        assert!(codes.iter().all(|&c| c <= 63));
        for (c, v) in codes.iter().zip(&w) {
            worst = worst.max((*c as f64 / sum - v / total).abs());
        }
    }
    println!("simplex codes: worst share error {worst:.4}");
    assert!(worst < 0.02);
}
