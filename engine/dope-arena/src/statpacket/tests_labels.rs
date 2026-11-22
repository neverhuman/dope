//! Target sampling: labels are draws (never an arg-max), rows are local and
//! reproducible with the target attached, the regime rule ties the label to
//! the feature regime, and the support floor keeps rare classes, rare labels
//! and a planted extreme target out of the packet and out of every sample.

use super::codec::{pack, truncate, unpack};
use super::decode::{Decoder, LabelRule};
use super::fit::{fit_packet, Size, Target};
use super::glm::sigmoid;
use super::linpred::TargetModel;
use super::packet::{Level, Packet, Task};
use super::scale::ScaleCard;
use super::target::column_samplers;
use super::testkit::{fit_full, mean_sd, n01, table, u01};

const DECODED_ROWS: usize = 100_000;

fn features(seed: u64, n: usize, p: usize) -> Vec<f64> {
    table(n, p, |i, j| match j % 3 {
        0 => n01(seed, j, i) + 0.5 * n01(seed, 70, i),
        1 => (0.4 * n01(seed, j, i)).exp(),
        _ => (3.0 * u01(seed, j, i)).floor(),
    })
}

fn binary_labels(raw: &[f64], p: usize, seed: u64) -> Vec<u16> {
    raw.chunks(p)
        .enumerate()
        .map(|(i, r)| (u01(seed, 0, i) < sigmoid(1.3 * r[0] - 0.9 * r[3] - 0.6)) as u16)
        .collect()
}

fn three_tasks(n: usize, p: usize) -> Vec<(Packet, ScaleCard)> {
    let raw = features(101, n, p);
    let y: Vec<f64> = raw
        .chunks(p)
        .enumerate()
        .map(|(i, r)| 5.0 + 2.0 * r[0] - r[3] + 0.6 * r[0] * r[3] + 0.5 * n01(102, 0, i))
        .collect();
    let two = binary_labels(&raw, p, 103);
    let four: Vec<u16> = raw
        .chunks(p)
        .enumerate()
        .map(|(i, r)| ((r[0] + 0.8 * n01(104, 0, i) > 0.0) as u16) * 2 + (r[2] > 0.5) as u16)
        .collect();
    let fit = |y: Target, task: Task, k: u8| {
        let (packet, card, _) = fit_full(&raw, y, task, k, n);
        (packet, card)
    };
    vec![
        fit(Target::Regression(&y), Task::Regression, 0),
        fit(Target::Classes(&two), Task::Binary, 2),
        fit(Target::Classes(&four), Task::Multiclass, 4),
    ]
}

#[test]
fn rows_with_their_target_are_local_and_reproducible() {
    let (n, p) = (700usize, 6usize);
    for (packet, card) in three_tasks(n, p) {
        let bytes = pack(&packet, Level::E).unwrap();
        for level in Level::ALL {
            let cut = unpack(&truncate(&bytes, level).unwrap()).unwrap();
            for rule in [LabelRule::Sampled, LabelRule::Regime] {
                let decoder = Decoder::new(&cut).unwrap().with_label_rule(rule);
                let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
                for card in [None, Some(&card)] {
                    let whole = bits(decoder.table(card, 9, 0, 801).unwrap());
                    let mut parts = bits(decoder.table(card, 9, 0, 400).unwrap());
                    parts.extend(bits(decoder.table(card, 9, 400, 401).unwrap()));
                    assert_eq!(whole, parts, "{:?} level {}", packet.task, level.name());
                    assert_eq!(whole, bits(decoder.table(card, 9, 0, 801).unwrap()));
                    assert_ne!(whole, bits(decoder.table(card, 10, 0, 801).unwrap()));
                    // A row far into the stream equals the same row decoded alone.
                    let far = bits(decoder.table(card, 9, 1 << 40, 3).unwrap());
                    let alone = bits(decoder.table(card, 9, (1 << 40) + 1, 1).unwrap());
                    assert_eq!(far[p + 1..2 * (p + 1)], alone[..]);
                }
                // The features do not depend on the target or on the label rule.
                let table = decoder.table(None, 9, 0, 50).unwrap();
                let only = decoder.rows_unit(9, 0, 50);
                for (row, x) in table.chunks(p + 1).zip(only.chunks(p)) {
                    assert_eq!(&row[..p], x);
                }
            }
        }
    }
}

#[test]
fn labels_are_drawn_from_the_conditional_never_arg_maxed() {
    let (n, p) = (2000usize, 6usize);
    let raw = features(111, n, p);
    let ids = binary_labels(&raw, p, 112);
    let (packet, _, _) = fit_full(&raw, Target::Classes(&ids), Task::Binary, 2, n);
    let decoder = Decoder::new(&packet).unwrap();
    let rows = decoder.table(None, 3, 0, DECODED_ROWS).unwrap();
    let model = TargetModel::new(&packet, Level::E, &column_samplers(&packet));
    // Ten probability bins: the share of label 1 follows the model's probability.
    let (mut count, mut ones, mut mass) = ([0.0f64; 10], [0.0f64; 10], [0.0f64; 10]);
    let mut eta = [0.0];
    for row in rows.chunks(p + 1) {
        let x: Vec<f64> = row[..p].iter().map(|&v| v as f64).collect();
        model.eta(&x, &mut eta);
        let q = sigmoid(eta[0]);
        let bin = ((q * 10.0) as usize).min(9);
        count[bin] += 1.0;
        ones[bin] += row[p] as f64;
        mass[bin] += q;
        assert!(row[p] == 0.0 || row[p] == 1.0);
    }
    let mut worst = 0.0f64;
    for bin in 0..10 {
        if count[bin] >= 500.0 {
            let (share, want) = (ones[bin] / count[bin], mass[bin] / count[bin]);
            let se = (want * (1.0 - want) / count[bin]).sqrt();
            worst = worst.max((share - want).abs() / se);
            // An arg-max would give 0 below one half and 1 above it.
            assert!(share > 0.01 && share < 0.99, "bin {bin}: {share}");
        }
    }
    let total = ones.iter().sum::<f64>() / DECODED_ROWS as f64;
    let fit_share = ids.iter().filter(|&&c| c == 1).count() as f64 / n as f64;
    println!(
        "binary labels, 100k rows: worst calibration gap over probability bins {worst:.2} SE; share of label 1 {total:.4} (FIT {fit_share:.4})"
    );
    assert!(worst < 4.0 && (total - fit_share).abs() < 0.02);
}

#[test]
fn the_regime_rule_ties_the_label_to_the_feature_regime() {
    let (n, p) = (1500usize, 6usize);
    let (packet, _) = three_tasks(n, p).remove(2);
    let prior: Vec<f64> = {
        let mut q: Vec<f64> = packet.prior.iter().map(|&c| c as f64 / 255.0).collect();
        q.push(1.0 - q.iter().sum::<f64>());
        q
    };
    let shares = |decoder: &Decoder| -> Vec<f64> {
        let rows = decoder.table(None, 5, 0, DECODED_ROWS).unwrap();
        let mut share = vec![0.0; 4];
        rows.chunks(p + 1)
            .for_each(|r| share[r[p] as usize] += 1.0 / DECODED_ROWS as f64);
        share
    };
    let tied = Decoder::new(&packet)
        .unwrap()
        .with_label_rule(LabelRule::Regime);
    let got = shares(&tied);
    for (g, q) in got.iter().zip(&prior) {
        assert!((g - q).abs() < 4.0 * (q * (1.0 - q) / DECODED_ROWS as f64).sqrt() + 1e-9);
    }
    // Level A has no regimes: the rule falls back to sampled labels.
    let flat = packet.at_level(Level::A).unwrap();
    let (a, b) = (
        Decoder::new(&flat).unwrap(),
        Decoder::new(&flat)
            .unwrap()
            .with_label_rule(LabelRule::Regime),
    );
    assert_eq!(
        a.table(None, 5, 0, 300).unwrap(),
        b.table(None, 5, 0, 300).unwrap()
    );
    // With regimes the two rules give different labels on the same features.
    let sampled = Decoder::new(&packet)
        .unwrap()
        .table(None, 5, 0, 300)
        .unwrap();
    assert_ne!(sampled, tied.table(None, 5, 0, 300).unwrap());
}

#[test]
fn regression_targets_keep_their_moments_and_a_planted_extreme_never_appears() {
    let (n, p) = (2000usize, 6usize);
    let raw = features(121, n, p);
    let canary = 987_654.25f64;
    let y: Vec<f64> = raw
        .chunks(p)
        .enumerate()
        .map(|(i, r)| {
            if i == 777 {
                canary
            } else {
                50.0 + 8.0 * (r[0] - 0.7 * r[3] + 0.6 * n01(122, 0, i))
            }
        })
        .collect();
    let (packet, card, _) = fit_full(&raw, Target::Regression(&y), Task::Regression, 0, n);
    let bytes = pack(&packet, Level::E).unwrap();
    for needle in [
        canary.to_le_bytes().to_vec(),
        (canary as f32).to_le_bytes().to_vec(),
        (canary as f32).to_be_bytes().to_vec(),
    ] {
        assert!(!bytes.windows(needle.len()).any(|w| w == needle));
    }
    let rows = Decoder::new(&packet)
        .unwrap()
        .table(Some(&card), 2, 0, DECODED_ROWS)
        .unwrap();
    let sampled: Vec<f64> = rows.chunks(p + 1).map(|r| r[p] as f64).collect();
    let top = sampled.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    // The standardised target is clipped at 6: no sample can reach the planted value.
    assert!(top < canary / 4.0, "{top}");
    assert!(sampled.iter().all(|v| v.is_finite()));

    // Without the outlier: mean and SD of the sampled target match the FIT rows.
    let clean: Vec<f64> = y
        .iter()
        .map(|&v| if v == canary { 50.0 } else { v })
        .collect();
    let (packet, card, _) = fit_full(&raw, Target::Regression(&clean), Task::Regression, 0, n);
    let rows = Decoder::new(&packet)
        .unwrap()
        .table(Some(&card), 2, 0, DECODED_ROWS)
        .unwrap();
    let sampled: Vec<f64> = rows.chunks(p + 1).map(|r| r[p] as f64).collect();
    let ((m_fit, s_fit), (m_syn, s_syn)) = (mean_sd(&clean), mean_sd(&sampled));
    println!(
        "regression target, 100k rows: mean {m_syn:.3} (FIT {m_fit:.3}), SD {s_syn:.3} (FIT {s_fit:.3})"
    );
    assert!((m_syn - m_fit).abs() < 0.05 * s_fit && (s_syn / s_fit - 1.0).abs() < 0.05);
}

#[test]
fn classes_and_labels_below_the_support_floor_are_never_emitted() {
    let (n, p) = (900usize, 5usize);
    let raw = features(131, n, p);
    // Class 2 is carried by 9 rows only; class 3 by none.
    let ids: Vec<u16> = (0..n)
        .map(|i| {
            if i % 100 == 7 {
                2
            } else {
                (raw[i * p] > 0.2) as u16
            }
        })
        .collect();
    assert_eq!(ids.iter().filter(|&&c| c == 2).count(), 9);
    let (packet, card, _) = fit_full(&raw, Target::Classes(&ids), Task::Multiclass, 4, n);
    // No prior mass, no weights, no factor means for the rare class.
    assert_eq!(packet.prior[2], 0);
    assert_eq!(packet.prior.iter().map(|&c| c as u32).sum::<u32>(), 255);
    assert!(packet.target_weight[p..2 * p].iter().all(|&w| w == 0));
    assert!(packet.class_mean[2 * 12..4 * 12].iter().all(|&m| m == 0));
    for rule in [LabelRule::Sampled, LabelRule::Regime] {
        for level in [Level::A, Level::E] {
            let cut = packet.at_level(level).unwrap();
            let rows = Decoder::new(&cut)
                .unwrap()
                .with_label_rule(rule)
                .table(Some(&card), 6, 0, DECODED_ROWS)
                .unwrap();
            let mut seen = [0usize; 4];
            rows.chunks(p + 1).for_each(|r| seen[r[p] as usize] += 1);
            assert_eq!((seen[2], seen[3]), (0, 0), "{rule:?} {level:?}");
            assert!(seen[0] > 0 && seen[1] > 0);
        }
    }
    // One more row lifts the class over the floor: it gets prior mass and weights.
    let ten: Vec<u16> = (0..n)
        .map(|i| if i % 100 == 7 || i == 0 { 2 } else { ids[i] })
        .collect();
    let (lifted, ..) = fit_full(&raw, Target::Classes(&ten), Task::Multiclass, 4, n);
    assert!(lifted.prior[2] > 0 && lifted.target_weight[p..2 * p].iter().any(|&w| w != 0));

    // Binary with 9 positives: a single-class target, no weights.
    let few: Vec<u16> = (0..n).map(|i| (i % 100 == 7) as u16).collect();
    let (packet, card, _) = fit_full(&raw, Target::Classes(&few), Task::Binary, 2, n);
    assert_eq!(packet.prior, vec![255]);
    assert!(packet.target_weight.iter().all(|&w| w == 0));
    let rows = Decoder::new(&packet)
        .unwrap()
        .table(Some(&card), 6, 0, 20_000)
        .unwrap();
    assert!(rows.chunks(p + 1).all(|r| r[p] == 0.0));

    // Multilabel: label 0 is common, label 1 has 9 positives, label 2 has 9 negatives.
    let cells: Vec<u8> = (0..n)
        .flat_map(|i| {
            [
                (raw[i * p] > 0.0) as u8,
                (i % 100 == 7) as u8,
                (i % 100 != 7) as u8,
            ]
        })
        .collect();
    let fitted = fit_packet(
        &raw,
        Target::Multilabel(&cells),
        Task::Multilabel,
        3,
        Size::level(Level::E),
        n,
    )
    .unwrap();
    let packet = unpack(&fitted.bytes).unwrap();
    assert_eq!((packet.prior[1], packet.prior[2]), (0, 255));
    assert!(packet.prior[0] > 100 && packet.prior[0] < 160);
    assert!(packet.target_weight[p..].iter().all(|&w| w == 0));
    let decoder = Decoder::new(&packet).unwrap();
    assert_eq!(decoder.targets(), 3);
    let rows = decoder.table(None, 6, 0, 20_000).unwrap();
    let mut ones = 0.0;
    for r in rows.chunks(p + 3) {
        assert_eq!((r[p + 1], r[p + 2]), (0.0, 1.0));
        ones += r[p] as f64 / 20_000.0;
    }
    let rate = cells.chunks(3).filter(|c| c[0] == 1).count() as f64 / n as f64;
    println!("multilabel: sampled rate of the common label {ones:.4} (FIT {rate:.4})");
    assert!((ones - rate).abs() < 0.03);
}
