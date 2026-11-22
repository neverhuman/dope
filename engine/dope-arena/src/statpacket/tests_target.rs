//! Target fit: the solvers reach stationary points of their penalised
//! objectives, and the stored S3 model is the plain evaluator's model
//! (`Ridge(alpha = 1)`, `LogisticRegression(C = 1)` on standardised inputs)
//! up to the int8 quantisation.

use super::fit::Target;
use super::glm::{logistic, ridge, sigmoid, softmax, softmax_row, Basis, Dense, SKIP_ROW};
use super::linpred::{temperature_value, weight_value, TargetModel, TEMPERATURE_UNIT};
use super::packet::{f16_to_f64, Level, Packet, Task};
use super::testkit::{column, fit_full, mean_sd, n01, table, u01};

/// `[1 | x]` with the columns of `x` standardised, and unit ridge weights.
fn standard_design(unit: &[f64], n: usize, p: usize) -> (Dense, Vec<(f64, f64)>) {
    let stats: Vec<(f64, f64)> = (0..p).map(|j| mean_sd(&column(unit, p, j))).collect();
    let mut a = Vec::with_capacity(n * (p + 1));
    for row in unit.chunks(p) {
        a.push(1.0);
        a.extend(row.iter().zip(&stats).map(|(v, (m, s))| (v - m) / s));
    }
    let mut penalty = vec![1.0; p + 1];
    penalty[0] = 0.0;
    let basis = Dense {
        a,
        n,
        d: p + 1,
        penalty,
    };
    (basis, stats)
}

/// Largest absolute entry of `A' r - D c`.
fn gradient_size(basis: &Dense, resid: &[f64], c: &[f64]) -> f64 {
    let g = basis.project(resid);
    g.iter()
        .zip(basis.penalty.iter().zip(c))
        .map(|(g, (pen, c))| (g - pen * c).abs())
        .fold(0.0, f64::max)
}

fn features(seed: u64, n: usize, p: usize) -> Vec<f64> {
    table(n, p, |i, j| match j % 4 {
        0 => n01(seed, j, i),
        1 => 0.6 * n01(seed, j - 1, i) + 0.8 * n01(seed, j, i),
        2 => (0.5 * n01(seed, j, i)).exp(),
        _ => (u01(seed, j, i) < 0.4) as u8 as f64 + 0.3 * n01(seed, 0, i),
    })
}

#[test]
fn solvers_reach_stationary_points() {
    let (n, p) = (500usize, 7usize);
    let raw = features(41, n, p);
    let (basis, _) = standard_design(&raw, n, p);
    let signal: Vec<f64> = raw
        .chunks(p)
        .map(|r| 0.9 * r[0] - 0.7 * r[1] + 0.4 * r[3])
        .collect();
    // Ridge: the normal equations hold.
    let y: Vec<f64> = (0..n).map(|i| signal[i] + 0.5 * n01(42, 0, i)).collect();
    let c = ridge(&basis, &y);
    let fitted = basis.apply(&c);
    let resid: Vec<f64> = y.iter().zip(&fitted).map(|(y, f)| y - f).collect();
    let g_ridge = gradient_size(&basis, &resid, &c);
    // Logistic: the penalised score vanishes.
    let labels: Vec<f64> = (0..n)
        .map(|i| (u01(43, 0, i) < sigmoid(signal[i] - 0.8)) as u8 as f64)
        .collect();
    let c = logistic(&basis, &vec![0.0; n], &labels, &vec![0.0; p + 1]);
    let prob: Vec<f64> = basis.apply(&c).iter().map(|&e| sigmoid(e)).collect();
    let resid: Vec<f64> = labels.iter().zip(&prob).map(|(y, q)| y - q).collect();
    let g_logit = gradient_size(&basis, &resid, &c);
    // Softmax (three classes, some rows skipped): every class block is
    // stationary and the penalised coefficients are centred.
    let classes = 3usize;
    let label: Vec<usize> = (0..n)
        .map(|i| {
            if i % 37 == 0 {
                return SKIP_ROW;
            }
            let logit = [0.0, signal[i], -signal[i] + 0.5 * raw[i * p + 4]];
            let mut q = [0.0; 3];
            softmax_row(&logit, &mut q);
            let u = u01(44, 0, i);
            (u > q[0]) as usize + (u > q[0] + q[1]) as usize
        })
        .collect();
    let d = p + 1;
    let c = softmax(
        &basis,
        &vec![0.0; n * classes],
        &label,
        classes,
        &vec![0.0; classes * d],
    );
    let mut g_soft = 0.0f64;
    for k in 0..classes {
        let lin: Vec<Vec<f64>> = (0..classes)
            .map(|l| basis.apply(&c[l * d..(l + 1) * d]))
            .collect();
        let resid: Vec<f64> = (0..n)
            .map(|i| {
                if label[i] == SKIP_ROW {
                    return 0.0;
                }
                let logit: Vec<f64> = lin.iter().map(|l| l[i]).collect();
                let mut q = vec![0.0; classes];
                softmax_row(&logit, &mut q);
                (label[i] == k) as u8 as f64 - q[k]
            })
            .collect();
        g_soft = g_soft.max(gradient_size(&basis, &resid, &c[k * d..(k + 1) * d]));
    }
    let centre = (1..d)
        .map(|j| (0..classes).map(|k| c[k * d + j]).sum::<f64>().abs())
        .fold(0.0, f64::max);
    println!(
        "stationarity (largest |gradient| entry, n = {n}): ridge {g_ridge:.2e}, logistic {g_logit:.2e}, softmax {g_soft:.2e}; softmax weight sum over classes {centre:.2e}"
    );
    assert!(g_ridge < 1e-8 && g_logit < 1e-6 && g_soft < 1e-5 && centre < 1e-6);
}

/// Dequantised weights of output `k` in standardised units, and the intercept
/// the packet implies for standardised inputs.
fn standardised_weights(packet: &Packet, k: usize, stats: &[(f64, f64)]) -> (f64, Vec<f64>) {
    let p = packet.cols();
    let w: Vec<f64> = (0..p).map(|j| weight_value(packet, k, j)).collect();
    let shift: f64 = w.iter().zip(stats).map(|(w, (m, _))| w * m).sum();
    let b = f16_to_f64(packet.target_intercept[k]) + shift;
    (b, w.iter().zip(stats).map(|(w, (_, s))| w * s).collect())
}

/// Largest gap between the stored and the reference weights, in units of the
/// quantisation step of each weight (half a step is the rounding bound).
fn worst_step_gap(packet: &Packet, k: usize, stats: &[(f64, f64)], want: &[f64]) -> f64 {
    let (_, got) = standardised_weights(packet, k, stats);
    let step = f16_to_f64(packet.target_scale[k]) / 127.0;
    got.iter()
        .zip(want.iter().zip(stats))
        .map(|(g, (w, (_, s)))| (g - w).abs() / (step * s))
        .fold(0.0, f64::max)
}

#[test]
fn the_linear_target_is_the_plain_ridge_model() {
    let (n, p) = (800usize, 8usize);
    let raw = features(51, n, p);
    let y: Vec<f64> = raw
        .chunks(p)
        .enumerate()
        .map(|(i, r)| {
            30.0 + 4.0 * (1.5 * r[0] - r[1] + 0.6 * r[3] + 0.3 * r[6] + 0.7 * n01(52, 0, i))
        })
        .collect();
    let (packet, card, unit) = fit_full(&raw, Target::Regression(&y), Task::Regression, 0, n);
    let (basis, stats) = standard_design(&unit, n, p);
    let target: Vec<f64> = y.iter().map(|&v| card.target_to_unit(v)).collect();
    let want = ridge(&basis, &target);
    let gap = worst_step_gap(&packet, 0, &stats, &want[1..]);
    let (b, _) = standardised_weights(&packet, 0, &stats);
    println!(
        "ridge parity: worst weight gap {gap:.3} quantisation steps, intercept gap {:.2e}",
        (b - want[0]).abs()
    );
    // Rounding to the nearest int8 code is half a step; the scale itself is a binary16.
    assert!(gap <= 0.51, "{gap}");
    assert!((b - want[0]).abs() < 5e-3);
    // No temperature for regression: the second noise byte holds ratio nibbles.
    assert!(packet.resid_sd > 0);
}

#[test]
fn the_binary_and_multiclass_targets_are_the_plain_logistic_models() {
    let (n, p) = (900usize, 6usize);
    let raw = features(61, n, p);
    let score: Vec<f64> = raw
        .chunks(p)
        .map(|r| 1.2 * r[0] - 0.9 * r[1] + 0.8 * r[3])
        .collect();
    // Binary.
    let ids: Vec<u16> = (0..n)
        .map(|i| (u01(62, 0, i) < sigmoid(score[i] - 1.2)) as u16)
        .collect();
    let (packet, _, unit) = fit_full(&raw, Target::Classes(&ids), Task::Binary, 2, n);
    let (basis, stats) = standard_design(&unit, n, p);
    let labels: Vec<f64> = ids.iter().map(|&c| c as f64).collect();
    let want = logistic(&basis, &vec![0.0; n], &labels, &vec![0.0; p + 1]);
    let gap = worst_step_gap(&packet, 0, &stats, &want[1..]);
    let (b, _) = standardised_weights(&packet, 0, &stats);
    let t = temperature_value(packet.temperature);
    println!(
        "logistic parity: worst weight gap {gap:.3} steps, intercept gap {:.2e}, temperature code {} (T = {t:.4})",
        (b - want[0]).abs(),
        packet.temperature
    );
    assert!(gap <= 0.51 && (b - want[0]).abs() < 2e-2, "{gap}");
    assert!(packet.temperature.abs_diff(TEMPERATURE_UNIT) <= 1 && packet.resid_sd == 0);
    // The intercept is refitted against the stored weights: the mean fitted
    // probability on the FIT rows equals the FIT share.
    let eta = TargetModel::new(&packet, Level::A, &[]).eta_rows(&unit, p);
    let mean: f64 = eta.iter().map(|&e| sigmoid(e)).sum::<f64>() / n as f64;
    let share = labels.iter().sum::<f64>() / n as f64;
    assert!((mean - share).abs() < 2e-3, "{mean} {share}");

    // Multiclass, K = 4: wire outputs are logits against class 0.
    let ids: Vec<u16> = (0..n)
        .map(|i| {
            let logit = [0.0, score[i], -score[i], 0.8 * raw[i * p + 4] - 0.5];
            let mut q = [0.0; 4];
            softmax_row(&logit, &mut q);
            let u = u01(63, 0, i);
            (0..3).filter(|&c| u > q[..=c].iter().sum::<f64>()).count() as u16
        })
        .collect();
    let (packet, _, unit) = fit_full(&raw, Target::Classes(&ids), Task::Multiclass, 4, n);
    let (basis, stats) = standard_design(&unit, n, p);
    let label: Vec<usize> = ids.iter().map(|&c| c as usize).collect();
    let d = p + 1;
    let sym = softmax(&basis, &vec![0.0; n * 4], &label, 4, &vec![0.0; 4 * d]);
    let mut worst = 0.0f64;
    for class in 1..4 {
        let want: Vec<f64> = (1..d).map(|j| sym[class * d + j] - sym[j]).collect();
        worst = worst.max(worst_step_gap(&packet, class - 1, &stats, &want));
    }
    // Mean fitted class probabilities on the FIT rows equal the FIT shares.
    let eta = TargetModel::new(&packet, Level::A, &[]).eta_rows(&unit, p);
    let mut mean = [0.0; 4];
    for row in eta.chunks(3) {
        let mut q = [0.0; 4];
        softmax_row(&[0.0, row[0], row[1], row[2]], &mut q);
        mean.iter_mut().zip(q).for_each(|(m, q)| *m += q / n as f64);
    }
    let mut calibration = 0.0f64;
    for (class, m) in mean.iter().enumerate() {
        let share = ids.iter().filter(|&&c| c as usize == class).count() as f64 / n as f64;
        calibration = calibration.max((m - share).abs());
    }
    println!(
        "multinomial parity (K = 4): worst weight gap {worst:.3} steps, worst mean-probability gap {calibration:.2e}"
    );
    assert!(worst <= 0.6 && calibration < 3e-3, "{worst} {calibration}");
}
