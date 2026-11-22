//! Residual sections and noise: S8 finds a planted non-linear main effect,
//! S10 a planted interaction, both stay empty on a linear table, and the
//! noise codes follow the residual of every level.

use super::codec::{fit_to_budget, pack, unpack, BudgetStep};
use super::effects::gate;
use super::fit::{fit_packet, Size, Target};
use super::linpred::{
    effect_code, effect_value, noise_sd, ratio_code, ratio_value, resid_sd_code, resid_sd_value,
    temperature_code, temperature_value, TargetModel, PAIR_OFFSET_SLOT,
};
use super::packet::{Level, Packet, Task};
use super::target::column_samplers;
use super::testkit::{fit_full, n01, table, u01};

/// Root mean square of `y - eta` on the FIT rows for the model of `level`.
fn residual_rms(packet: &Packet, level: Level, unit: &[f64], y: &[f64]) -> f64 {
    let samplers = column_samplers(packet);
    let eta = TargetModel::new(packet, level, &samplers).eta_rows(unit, packet.cols());
    let sse: f64 = y.iter().zip(&eta).map(|(y, e)| (y - e) * (y - e)).sum();
    (sse / y.len() as f64).sqrt()
}

fn gaussian(seed: u64, n: usize, p: usize) -> Vec<f64> {
    table(n, p, |i, j| n01(seed, j, i) + 0.4 * n01(seed, 90, i))
}

#[test]
fn quantisers_are_monotone_and_nearest() {
    for c in i8::MIN + 1..=i8::MAX {
        assert_eq!(effect_code(effect_value(c)), c);
    }
    assert!((-127..127).all(|c: i8| effect_value(c) < effect_value(c + 1)));
    assert_eq!(
        (effect_value(127), effect_code(9.0), effect_code(-9.0)),
        (4.0, 127, -127)
    );
    // Halfway between two values rounds to the nearer one by value.
    let mid = 0.5 * (effect_value(40) + effect_value(41));
    assert_eq!((effect_code(mid - 1e-9), effect_code(mid + 1e-9)), (40, 41));
    for code in 1..=255u8 {
        assert_eq!(resid_sd_code(resid_sd_value(code)), code);
        assert_eq!(temperature_code(temperature_value(code)), code);
    }
    assert_eq!(
        (resid_sd_value(0), resid_sd_code(0.0), resid_sd_value(240)),
        (0.0, 0, 1.0)
    );
    assert_eq!((temperature_value(128), temperature_code(1.0)), (1.0, 128));
    assert!((0..16).all(|c| ratio_code(ratio_value(c)) == c));
    assert_eq!(
        (ratio_value(0), ratio_code(1.3), ratio_code(0.0)),
        (1.0, 0, 15)
    );
    // The gate is the upper 5 % / candidates point of a chi-square.
    assert!((gate(1.0, 1) - 3.84).abs() < 0.15, "{}", gate(1.0, 1));
    assert!((gate(3.0, 1) - 7.81).abs() < 0.15, "{}", gate(3.0, 1));
    assert!(gate(1.0, 136) > 12.0 && gate(1.0, 136) < 16.0);
}

#[test]
fn main_effects_and_pairs_find_planted_structure_and_ignore_a_linear_table() {
    let (n, p) = (1200usize, 10usize);
    let raw = gaussian(71, n, p);
    let noise = |i: usize| 0.4 * n01(72, 0, i);
    let linear = |r: &[f64]| 1.0 * r[0] - 0.8 * r[1] + 0.5 * r[2];
    // (a) linear table; (b) a U-shaped effect of column 3; (c) an interaction of columns 4 and 5.
    let ys: Vec<Vec<f64>> = [0usize, 1, 2]
        .iter()
        .map(|&case| {
            raw.chunks(p)
                .enumerate()
                .map(|(i, r)| {
                    let extra = match case {
                        1 => 0.9 * (r[3] * r[3] - 1.16),
                        2 => 0.8 * r[4] * r[5],
                        _ => 0.0,
                    };
                    linear(r) + extra + noise(i)
                })
                .collect()
        })
        .collect();
    let fits: Vec<(Packet, Vec<f64>, Vec<f64>)> = ys
        .iter()
        .map(|y| {
            let (packet, card, unit) =
                fit_full(&raw, Target::Regression(y), Task::Regression, 0, n);
            let target = y.iter().map(|&v| card.target_to_unit(v)).collect();
            (packet, unit, target)
        })
        .collect();
    let live_pairs = |packet: &Packet| -> Vec<(u16, u16)> {
        packet.pairs[..PAIR_OFFSET_SLOT]
            .iter()
            .filter(|t| t.coef.iter().any(|&c| c != 0))
            .map(|t| (t.a, t.b))
            .collect()
    };
    let main_cols =
        |packet: &Packet| -> Vec<usize> { (0..p).filter(|&j| packet.main_mask[j]).collect() };
    let mut report = Vec::new();
    for (name, (packet, unit, target)) in ["linear", "u-shape", "interaction"].iter().zip(&fits) {
        let rms: Vec<f64> = [Level::A, Level::D, Level::E]
            .iter()
            .map(|&l| residual_rms(packet, l, unit, target))
            .collect();
        let sd: Vec<f64> = [Level::A, Level::D, Level::E]
            .iter()
            .map(|&l| noise_sd(packet, l))
            .collect();
        report.push(format!(
            "{name}: S8 columns {:?}, S10 pairs {:?}, residual RMS A/D/E {:.3}/{:.3}/{:.3}, stored SD {:.3}/{:.3}/{:.3}",
            main_cols(packet),
            live_pairs(packet),
            rms[0], rms[1], rms[2], sd[0], sd[1], sd[2]
        ));
        // Stored noise follows the residual of each level: 2.2 % steps for
        // level A, 15 % steps (2^(1/5)) for the two ratios, and the degrees of
        // freedom correction (at most 1 + 10 + 40 + 16 codes on 1,200 rows).
        assert!((sd[0] / rms[0] - 1.0).abs() < 0.03, "{name} {sd:?} {rms:?}");
        assert!((sd[1] / rms[1] - 1.0).abs() < 0.10, "{name} {sd:?} {rms:?}");
        assert!((sd[2] / rms[2] - 1.0).abs() < 0.10, "{name} {sd:?} {rms:?}");
    }
    println!("{}", report.join("\n"));
    let (linear_fit, u_fit, pair_fit) = (&fits[0].0, &fits[1].0, &fits[2].0);
    // A linear table stores neither main effects nor pairs, and its packet is smaller.
    assert!(main_cols(linear_fit).is_empty() && live_pairs(linear_fit).is_empty());
    assert!(linear_fit.pairs.iter().all(|t| (t.a, t.b) == (0, 0)));
    assert_eq!(linear_fit.temperature, 0);
    let len = |packet: &Packet| pack(packet, Level::E).unwrap().len();
    assert!(len(linear_fit) < len(u_fit));
    // The U shape is a main effect of column 3 alone: outer bins up, inner bins down.
    assert_eq!(main_cols(u_fit), vec![3]);
    let cell: Vec<f64> = u_fit.main_effect[3 * 4..4 * 4]
        .iter()
        .map(|&c| effect_value(c))
        .collect();
    assert!(
        cell[0] > 0.1 && cell[3] > 0.1 && cell[1] < -0.1 && cell[2] < -0.1,
        "{cell:?}"
    );
    let u = &fits[1];
    let (a, d) = (
        residual_rms(&u.0, Level::A, &u.1, &u.2),
        residual_rms(&u.0, Level::D, &u.1, &u.2),
    );
    assert!(d < 0.85 * a, "{a} {d}");
    assert!(u_fit.temperature >> 4 >= 1);
    // The interaction is the first pair, with the planted sign; it removes most of the residual.
    assert_eq!(live_pairs(pair_fit).first(), Some(&(4, 5)));
    assert!(pair_fit.pairs[0].coef[0] > 0 && main_cols(pair_fit).is_empty());
    let f = &fits[2];
    let (d, e) = (
        residual_rms(&f.0, Level::D, &f.1, &f.2),
        residual_rms(&f.0, Level::E, &f.1, &f.2),
    );
    assert!(e < 0.6 * d, "{d} {e}");
    assert!(pair_fit.temperature & 15 >= 3);
    // Truncation removes the sections and the decoder falls back to the level-A noise.
    let cut = unpack(&pack(pair_fit, Level::C).unwrap()).unwrap();
    assert_eq!(noise_sd(&cut, Level::C), resid_sd_value(pair_fit.resid_sd));
}

#[test]
fn classification_effects_are_gated_too() {
    let (n, p) = (1500usize, 8usize);
    let raw = gaussian(81, n, p);
    let label = |extra: fn(&[f64]) -> f64, seed: u64| -> Vec<u16> {
        raw.chunks(p)
            .enumerate()
            .map(|(i, r)| {
                let logit = -0.5 + 1.1 * r[0] - 0.7 * r[1] + extra(r);
                (u01(seed, 0, i) < 1.0 / (1.0 + (-logit).exp())) as u16
            })
            .collect()
    };
    let plain = label(|_| 0.0, 82);
    let crossed = label(|r| 1.4 * r[2] * r[3], 83);
    let (linear_fit, ..) = fit_full(&raw, Target::Classes(&plain), Task::Binary, 2, n);
    let (pair_fit, ..) = fit_full(&raw, Target::Classes(&crossed), Task::Binary, 2, n);
    let live = |packet: &Packet| -> Vec<(u16, u16)> {
        packet.pairs[..PAIR_OFFSET_SLOT]
            .iter()
            .filter(|t| t.coef[0] != 0)
            .map(|t| (t.a, t.b))
            .collect()
    };
    println!(
        "binary: linear table pairs {:?} S8 columns {}, crossed table pairs {:?} (first coefficient {:.3})",
        live(&linear_fit),
        linear_fit.main_mask.iter().filter(|&&m| m).count(),
        live(&pair_fit),
        effect_value(pair_fit.pairs[0].coef[0])
    );
    assert!(live(&linear_fit).is_empty() && linear_fit.main_mask.iter().all(|&m| !m));
    assert_eq!(live(&pair_fit).first(), Some(&(2, 3)));
    assert!(effect_value(pair_fit.pairs[0].coef[0]) > 0.5);
}

#[test]
fn the_budget_step_refreshes_the_noise_of_the_final_shape() {
    // A wide regression table with real main effects: at a budget that keeps
    // level D with only some S8 columns, the stored ratio is recomputed.
    let (n, p) = (1000usize, 40usize);
    let raw = gaussian(91, n, p);
    let y: Vec<f64> = raw
        .chunks(p)
        .enumerate()
        .map(|(i, r)| {
            let curved: f64 = (0..6).map(|j| 0.5 * (r[j] * r[j] - 1.16)).sum();
            r[0] - r[7] + curved + 0.3 * n01(92, 0, i)
        })
        .collect();
    let fit = |size: Size| {
        fit_packet(&raw, Target::Regression(&y), Task::Regression, 0, size, n).unwrap()
    };
    let full = fit(Size::level(Level::E));
    let packet = unpack(&full.bytes).unwrap();
    let kept = packet.main_mask.iter().filter(|&&m| m).count();
    assert!(kept >= 6, "{kept}");
    // The budget is one S8 column short of the dense level-D packet.
    let dense_d = pack(&packet, Level::D).unwrap().len();
    let tight = fit(Size::budget(dense_d - 1));
    assert_eq!(tight.budget.step, BudgetStep::PartialMainEffects);
    assert_eq!(
        (tight.budget.level, tight.budget.main_effect_cols),
        (Level::D, kept - 1)
    );
    assert!(tight.bytes.len() < dense_d && tight.budget.fits);
    let small = unpack(&tight.bytes).unwrap();
    // Without the refresh the packet would be the budget cut of the full one.
    let cut = fit_to_budget(&packet, dense_d - 1).packet;
    assert_eq!(small.main_effect, cut.main_effect);
    let (with, without) = (noise_sd(&small, Level::D), noise_sd(&cut, Level::D));
    println!("partial S8 ({} of {kept} columns): level-D noise SD {with:.4} after the refresh, {without:.4} before", kept - 1);
    assert!(with >= without);
}
