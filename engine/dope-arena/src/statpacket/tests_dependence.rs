//! Dependence and sampler tests: correlation recovery by rank, tie-aware
//! scores, discrete association, class regimes, determinism and row locality.

use super::codec::{fit_to_budget, pack, truncate, unpack};
use super::decode::Decoder;
use super::dependence::{tie_scores, untied_score_variance};
use super::fit::{class_prior_codes, fit_features, fit_features_for_budget, FitInput, Labels};
use super::packet::{Level, Packet, Task};
use super::testkit::{
    column, fit_regression, max_corr_gap, n01, pearson, quantile, sorted, table, to_unit, u01,
};
use super::StatError;
use dope_core::stats::{norm_pdf, norm_ppf};

fn widen(rows: &[f32]) -> Vec<f64> {
    rows.iter().map(|&v| v as f64).collect()
}

#[test]
fn gaussian_correlations_are_recovered_and_degrade_with_rank() {
    let (n, p, rho) = (2000usize, 12usize, 0.7f64);
    // Columns form an AR(1) chain: corr(j, k) = 0.7^|j - k|, a slowly decaying spectrum.
    let mut raw = vec![0.0; n * p];
    for i in 0..n {
        raw[i * p] = n01(11, 0, i);
        for j in 1..p {
            raw[i * p + j] = rho * raw[i * p + j - 1] + (1.0 - rho * rho).sqrt() * n01(11, j, i);
        }
    }
    let (_, unit) = to_unit(&raw, n, p);
    let bytes = pack(&fit_regression(&unit, n, p), Level::E).unwrap();
    let gap = |level: Level| {
        let decoder = Decoder::new(&unpack(&truncate(&bytes, level).unwrap()).unwrap()).unwrap();
        max_corr_gap(&unit, &widen(&decoder.rows_unit(5, 0, 50_000)), p)
    };
    let (g12, g8, g4) = (gap(Level::E), gap(Level::C), gap(Level::B));
    println!(
        "max |corr gap| vs FIT rows (P=12, n=2000, 50k decoded): rank 12 {g12:.4}, rank 8 {g8:.4}, rank 4 {g4:.4}"
    );
    assert!(g12 <= 0.05, "rank 12: {g12}");
    assert!(g12 < g8 && g8 < g4, "not monotone: {g12} {g8} {g4}");
    // Level A carries no dependence section: columns are independent.
    let flat = Decoder::new(&unpack(&truncate(&bytes, Level::A).unwrap()).unwrap()).unwrap();
    let rows = widen(&flat.rows_unit(5, 0, 50_000));
    let worst = (0..p - 1)
        .map(|j| pearson(&column(&rows, p, j), &column(&rows, p, j + 1)).abs())
        .fold(0.0, f64::max);
    assert!(worst < 0.03, "level A neighbours correlate: {worst}");
}

#[test]
fn tie_scores_are_conditional_normal_means() {
    // 30 % zeros, 70 % ones.
    let col: Vec<f64> = (0..1000)
        .map(|i| if i % 10 < 3 { 0.0 } else { 1.0 })
        .collect();
    let s = tie_scores(&col);
    let cut = norm_pdf(norm_ppf(0.3));
    for (v, score) in col.iter().zip(&s) {
        let want = if *v == 0.0 { -cut / 0.3 } else { cut / 0.7 };
        assert!((score - want).abs() < 1e-12);
    }
    assert!(s.iter().sum::<f64>().abs() < 1e-9);
    // Untied rows keep the classical (rank - 0.5) / n scores, in input order.
    let col: Vec<f64> = (0..500).map(|i| ((i * 37) % 500) as f64).collect();
    let s = tie_scores(&col);
    for (v, score) in col.iter().zip(&s) {
        assert_eq!(*score, norm_ppf((*v + 0.5) / 500.0));
    }
    let var = s.iter().map(|v| v * v).sum::<f64>() / 500.0;
    assert!((var - untied_score_variance(500)).abs() < 1e-12);
    assert!(var > 0.97 && var < 1.0);
}

#[test]
fn discrete_columns_keep_their_association() {
    let (n, p) = (4000usize, 4usize);
    // One common factor: latent correlation 0.6 between every pair.
    let latent =
        |i: usize, j: usize| 0.6f64.sqrt() * n01(12, 99, i) + 0.4f64.sqrt() * n01(12, j, i);
    let raw = table(n, p, |i, j| {
        let z = latent(i, j);
        match j {
            0 => z,
            1 => (z > 0.4) as u8 as f64,
            2 => [-0.8, 0.1, 1.0].iter().filter(|&&c| z > c).count() as f64,
            _ => (z > -1.0) as u8 as f64,
        }
    });
    let (card, unit) = to_unit(&raw, n, p);
    let packet = fit_regression(&unit, n, p);
    assert!(packet.columns[0].is_continuous());
    assert!(packet.columns[1..].iter().all(|c| !c.is_continuous()));
    let decoder = Decoder::new(&packet).unwrap();
    let rows = widen(&decoder.rows_raw(&card, 9, 0, 50_000).unwrap());
    let gap = max_corr_gap(&raw, &rows, p);
    println!("mixed continuous / binary / ordinal table: max |corr gap| vs FIT rows {gap:.4}");
    assert!(gap <= 0.05, "{gap}");
}

#[test]
fn principal_axes_reproduce_a_factor_structure_wider_than_the_rank() {
    let (n, p) = (3000usize, 30usize);
    // Three common factors with mixed-sign loadings plus unit noise.
    let raw = table(n, p, |i, j| {
        let load = |f: usize| ((j * (f + 2) + f) % 7) as f64 / 4.0 - 0.75;
        (0..3).map(|f| load(f) * n01(15, 200 + f, i)).sum::<f64>() + n01(15, j, i)
    });
    let (_, unit) = to_unit(&raw, n, p);
    let bytes = pack(&fit_regression(&unit, n, p), Level::E).unwrap();
    let gap = |level: Level| {
        let decoder = Decoder::new(&unpack(&truncate(&bytes, level).unwrap()).unwrap()).unwrap();
        max_corr_gap(&unit, &widen(&decoder.rows_unit(2, 0, 50_000)), p)
    };
    let (g12, g8, g4) = (gap(Level::E), gap(Level::C), gap(Level::B));
    println!(
        "three factors in 30 columns (n=3000): max |corr gap| rank 12 {g12:.4}, rank 8 {g8:.4}, rank 4 {g4:.4}"
    );
    assert!(g12 <= 0.05 && g8 <= 0.05 && g4 <= 0.05, "{g12} {g8} {g4}");
}

#[test]
fn a_budget_without_room_for_dependence_skips_the_factor_fit_exactly() {
    let (n, p) = (300usize, 12usize);
    let (raw, _) = class_fixture(n, p);
    let (_, unit) = to_unit(&raw, n, p);
    let input = FitInput {
        x: &unit,
        rows: n,
        cols: p,
        task: Task::Regression,
        k: 0,
        labels: Labels::None,
    };
    let full = fit_features(&input).unwrap();
    for max_bytes in [60usize, 118, 150, 165, 166, 400, 2048] {
        let lean = fit_features_for_budget(&input, max_bytes).unwrap();
        let (a, b) = (
            fit_to_budget(&full, max_bytes),
            fit_to_budget(&lean, max_bytes),
        );
        assert_eq!((a.level, a.bytes, a.fits), (b.level, b.bytes, b.fits));
        assert_eq!(
            pack(&a.packet, a.level).unwrap(),
            pack(&b.packet, b.level).unwrap()
        );
        // The skip happens exactly when no dependence section fits.
        assert_eq!(lean.loading.iter().all(|&l| l == 0), a.level == Level::A);
    }
}

fn class_fixture(n: usize, p: usize) -> (Vec<f64>, Vec<u16>) {
    let ids: Vec<u16> = (0..n).map(|i| (u01(13, 50, i) < 0.35) as u16).collect();
    let raw = table(n, p, |i, j| {
        let shift = [2.5, -1.0, 0.0, 0.5, 0.0, 1.5][j % 6] * ids[i] as f64;
        shift + 0.7 * n01(13, 60, i) * (j % 2) as f64 + n01(13, j, i)
    });
    (raw, ids)
}

fn fit_binary(unit: &[f64], n: usize, p: usize, ids: &[u16]) -> Packet {
    fit_features(&FitInput {
        x: unit,
        rows: n,
        cols: p,
        task: Task::Binary,
        k: 2,
        labels: Labels::Classes(ids),
    })
    .unwrap()
}

#[test]
fn class_regimes_keep_marginals_and_correlations() {
    let (n, p) = (2000usize, 6usize);
    let (raw, ids) = class_fixture(n, p);
    let (_, unit) = to_unit(&raw, n, p);
    let packet = fit_binary(&unit, n, p, &ids);
    // 700 of 2000 rows are class 1 in expectation; codes sum to 255.
    let ones = ids.iter().filter(|&&c| c == 1).count();
    assert_eq!(packet.prior, vec![class_prior_codes(&[n - ones, ones])[0]]);
    assert!(packet.class_mean.iter().any(|&m| m != 0));
    let decoder = Decoder::new(&packet).unwrap();
    let rows = widen(&decoder.rows_unit(21, 0, 50_000));
    let corr = max_corr_gap(&unit, &rows, p);
    let mut quant = 0.0f64;
    for j in 0..p {
        let (a, b) = (sorted(column(&unit, p, j)), sorted(column(&rows, p, j)));
        for q in 2..=98 {
            quant =
                quant.max((quantile(&a, q as f64 / 100.0) - quantile(&b, q as f64 / 100.0)).abs());
        }
    }
    println!("binary regimes (P=6, n=2000): max |corr gap| {corr:.4}, max quantile gap {quant:.4}");
    assert!(corr <= 0.06 && quant <= 0.1, "{corr} {quant}");
}

#[test]
fn fit_and_decode_are_deterministic_and_row_local() {
    let (n, p) = (600usize, 9usize);
    let (raw, ids) = class_fixture(n, p);
    let (card, unit) = to_unit(&raw, n, p);
    let three: Vec<u16> = (0..n).map(|i| (i % 3) as u16).collect();
    let multiclass = fit_features(&FitInput {
        x: &unit,
        rows: n,
        cols: p,
        task: Task::Multiclass,
        k: 3,
        labels: Labels::Classes(&three),
    })
    .unwrap();
    for packet in [
        fit_regression(&unit, n, p),
        fit_binary(&unit, n, p, &ids),
        multiclass,
    ] {
        let again = match packet.task {
            Task::Regression => fit_regression(&unit, n, p),
            Task::Binary => fit_binary(&unit, n, p, &ids),
            _ => packet.clone(),
        };
        assert_eq!(
            pack(&packet, Level::E).unwrap(),
            pack(&again, Level::E).unwrap()
        );
        for level in Level::ALL {
            let decoder = Decoder::new(&packet.at_level(level).unwrap()).unwrap();
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
            let whole = bits(decoder.rows_unit(8, 0, 1001));
            let mut parts = bits(decoder.rows_unit(8, 0, 500));
            parts.extend(bits(decoder.rows_unit(8, 500, 501)));
            assert_eq!(whole, parts, "{:?} level {}", packet.task, level.name());
            assert_eq!(whole, bits(decoder.rows_unit(8, 0, 1001)));
            // A row far into the stream equals the same row decoded alone.
            let far = bits(decoder.rows_unit(8, 1 << 40, 3));
            assert_eq!(
                far[p..2 * p],
                bits(decoder.rows_unit(8, (1 << 40) + 1, 1))[..]
            );
            assert_ne!(whole, bits(decoder.rows_unit(9, 0, 1001)));
            let raw_whole = bits(decoder.rows_raw(&card, 8, 0, 400).unwrap());
            let mut raw_parts = bits(decoder.rows_raw(&card, 8, 0, 123).unwrap());
            raw_parts.extend(bits(decoder.rows_raw(&card, 8, 123, 277).unwrap()));
            assert_eq!(raw_whole, raw_parts);
        }
    }
}

#[test]
fn class_prior_codes_sum_to_255_and_keep_small_classes() {
    for counts in [
        vec![1300usize, 700],
        vec![1, 1999],
        vec![5, 5, 5, 5, 1980],
        vec![0, 10, 990],
        vec![7; 255],
        vec![1, 1, 1, 100_000],
    ] {
        let codes = class_prior_codes(&counts);
        assert_eq!(codes.iter().map(|&c| c as u32).sum::<u32>(), 255);
        for (code, count) in codes.iter().zip(&counts) {
            assert_eq!(*code == 0, *count == 0);
        }
    }
    assert_eq!(class_prior_codes(&[1300, 700]), vec![166, 89]);
}

fn regression_input(x: &[f64], rows: usize) -> FitInput<'_> {
    FitInput {
        x,
        rows,
        cols: 2,
        task: Task::Regression,
        k: 0,
        labels: Labels::None,
    }
}

#[test]
fn fit_rejects_bad_input_without_echoing_values() {
    let good = vec![0.25; 16 * 2];
    assert!(fit_features(&regression_input(&good, 16)).is_ok());
    let few = fit_features(&regression_input(&good[..30], 15)).unwrap_err();
    assert_eq!(few, StatError::Invalid("fewer than 16 FIT rows"));
    for poison in [f64::NAN, f64::INFINITY, 6.5, -123_456.789] {
        let mut bad = good.clone();
        bad[7] = poison;
        let text = fit_features(&regression_input(&bad, 16))
            .unwrap_err()
            .to_string();
        assert_eq!(
            text,
            "stat packet: features are not finite unit-space values"
        );
    }
    // Task, class count and labels must agree.
    let ids = vec![0u16; 16];
    let with = |task: Task, k: u8, labels: Labels<'_>| {
        fit_features(&FitInput {
            task,
            k,
            labels,
            ..regression_input(&good, 16)
        })
        .map(|_| ())
    };
    let mismatch = Err(StatError::Invalid("labels do not match the task"));
    assert_eq!(with(Task::Multiclass, 3, Labels::Classes(&ids)), Ok(()));
    assert_eq!(with(Task::Binary, 2, Labels::None), mismatch);
    assert_eq!(with(Task::Regression, 0, Labels::Classes(&ids)), mismatch);
    assert_eq!(
        with(Task::Multiclass, 3, Labels::Classes(&[3u16; 16])),
        mismatch
    );
    assert_eq!(
        with(Task::Multilabel, 2, Labels::Multilabel(&[1u8; 32])),
        Ok(())
    );
    assert_eq!(
        with(Task::Multilabel, 2, Labels::Multilabel(&[2u8; 32])),
        mismatch
    );
    assert_eq!(
        with(Task::Regression, 3, Labels::None),
        Err(StatError::Invalid("class count is not valid for the task"))
    );
}
