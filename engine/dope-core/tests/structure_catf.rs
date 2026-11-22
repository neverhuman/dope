//! Behavioral tests for Phase P5 Structural Rules & CATF Covariate Dependence
//! (`T05`, `T16`, `T26`, `T32`, discrete-latent correlation fix, and Gaussian zero-regression).

use dope_core::catf::{CatfConfig, CatfModel};
use dope_core::distil::{DistilConfig, DistilledStudent};
use dope_core::rng::Rng;
use dope_core::structure::{RuleMode, RuleSet, StructuralRule};
use dope_core::target::Head;
use dope_core::teacher::{CrossFittedTeacher, TeacherConfig};

#[test]
fn t05_nominal_permutation_invariance_in_catf() {
    let mut rng = Rng::new(71);
    let n = 600usize;
    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);
    for _ in 0..n {
        let a = (rng.below(4)) as f64;
        // Nonlinear nominal mapping: b depends strongly on a.
        let b = match a as usize {
            0 => {
                if rng.uniform() < 0.85 {
                    2.0
                } else {
                    0.0
                }
            }
            1 => {
                if rng.uniform() < 0.85 {
                    0.0
                } else {
                    1.0
                }
            }
            2 => {
                if rng.uniform() < 0.85 {
                    3.0
                } else {
                    2.0
                }
            }
            _ => {
                if rng.uniform() < 0.85 {
                    1.0
                } else {
                    3.0
                }
            }
        };
        c0.push(a);
        c1.push(b);
    }
    let perm = [3.0, 1.0, 0.0, 2.0];
    let c0_p: Vec<f64> = c0.iter().map(|&v| perm[v as usize]).collect();
    let cfg = CatfConfig {
        max_trees_per_col: 10,
        max_depth: 2,
        rule_mode: RuleMode::None,
        forced_first: vec![0],
        ..Default::default()
    };
    let bx = vec![c0.clone(), c1.clone()];
    let bx_p = vec![c0_p.clone(), c1.clone()];
    let m1 = CatfModel::fit(&bx, &bx, &[true, true], &cfg);
    let m2 = CatfModel::fit(&bx_p, &bx_p, &[true, true], &cfg);
    let s1 = m1.sample_x(2000, 13, 0);
    let s2 = m2.sample_x(2000, 13, 0);

    // Compare joint table P(a, b) after inverting permutation on s2.
    let inv = [2usize, 1, 3, 0];
    let mut p1 = [[0.0_f64; 4]; 4];
    let mut p2 = [[0.0_f64; 4]; 4];
    for r in 0..2000 {
        p1[s1[0][r].round() as usize][s1[1][r].round() as usize] += 1.0 / 2000.0;
        let orig_a = inv[s2[0][r].round() as usize];
        p2[orig_a][s2[1][r].round() as usize] += 1.0 / 2000.0;
    }
    let tv: f64 = (0..4)
        .flat_map(|a| (0..4).map(move |b| (p1[a][b] - p2[a][b]).abs()))
        .sum::<f64>()
        * 0.5;
    assert!(tv < 0.06, "nominal permutation TV drift too large: {tv}");
}

#[test]
fn t16_catf_depth2_lookahead_cracks_three_bit_parity_xor() {
    let mut rng = Rng::new(16);
    let n = 1600usize;
    let mut cols: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let b0 = rng.below(2);
        let b1 = rng.below(2);
        let b2 = rng.below(2);
        let b3 = b0 ^ b1 ^ b2; // pure 3-bit parity: zero pairwise or 2-way correlation!
        cols[0].push(b0 as f64);
        cols[1].push(b1 as f64);
        cols[2].push(b2 as f64);
        cols[3].push(b3 as f64);
    }
    let cfg = CatfConfig {
        max_trees_per_col: 25,
        max_depth: 3,
        eta: 0.5,
        alpha_shrink: 1.0,
        rule_mode: RuleMode::None,
        forced_first: vec![0, 1, 2],
        ..Default::default()
    };
    let model = CatfModel::fit(&cols, &cols, &[true, true, true, true], &cfg);
    let synth = model.sample_x(2000, 16, 0);
    let mut err = 0usize;
    for r in 0..2000 {
        let b0 = synth[0][r].round() as usize;
        let b1 = synth[1][r].round() as usize;
        let b2 = synth[2][r].round() as usize;
        let b3 = synth[3][r].round() as usize;
        if b3 != (b0 ^ b1 ^ b2) {
            err += 1;
        }
    }
    let err_rate = err as f64 / 2000.0;
    assert!(
        err_rate <= 0.02,
        "3-bit parity error rate {err_rate} exceeded 0.02"
    );
}

#[test]
fn t26_256_copied_binary_columns_preserve_row_level_identity() {
    let mut rng = Rng::new(26);
    let n = 300usize;
    let d = 256usize;
    let base: Vec<f64> = (0..n).map(|_| rng.below(2) as f64).collect();
    let cols: Vec<Vec<f64>> = (0..d).map(|_| base.clone()).collect();
    let is_disc = vec![true; d];
    let rules = RuleSet::mine(&cols, &cols, &is_disc, RuleMode::Structural, 10);
    assert_eq!(
        rules.rules.len(),
        255,
        "all 255 copied columns should mine FD-1"
    );
    let mut row = vec![0.0; d];
    row[0] = 1.0;
    rules.apply_row(&mut row, 0.5, 0.5);
    assert!(row.iter().all(|&v| (v - 1.0).abs() < 1e-9));
}

#[test]
fn t32_target_evaluates_at_post_rule_covariates_and_all_rules_mine() {
    let mut rng = Rng::new(32);
    let n = 500usize;
    let mut x0 = Vec::with_capacity(n); // continuous base
    let mut x1 = Vec::with_capacity(n); // x1 >= x0 (INEQ)
    let mut x2 = Vec::with_capacity(n); // x2 = x0 + x1 (ARITH)
    let mut x3 = Vec::with_capacity(n); // binary BIN(x0 > 0.5)
    let mut x4 = Vec::with_capacity(n); // EXCL with x5
    let mut x5 = Vec::with_capacity(n); // EXCL with x4
    let mut x6 = Vec::with_capacity(n); // has 15% missing
    let mut x7 = Vec::with_capacity(n); // MISS gated on x6
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let a = rng.uniform();
        let b = a + rng.uniform() * 0.5;
        let c = a + b;
        let bn = if a > 0.5 { 1.0 } else { 0.0 };
        let (e4, e5) = if i % 3 == 0 {
            (1.0, 0.0)
        } else if i % 3 == 1 {
            (0.0, 1.0)
        } else {
            (0.0, 0.0)
        };
        let (m6, m7) = if i % 7 == 0 {
            (f64::NAN, f64::NAN)
        } else {
            (rng.uniform(), rng.uniform())
        };
        x0.push(a);
        x1.push(b);
        x2.push(c);
        x3.push(bn);
        x4.push(e4);
        x5.push(e5);
        x6.push(m6);
        x7.push(m7);
        y.push(3.0 * c + 0.02 * rng.normal());
    }
    let build_x = vec![x0, x1, x2, x3, x4, x5, x6, x7];
    let is_nom = vec![false, false, false, true, true, true, false, false];
    let catf = CatfModel::fit(
        &build_x,
        &build_x,
        &is_nom,
        &CatfConfig {
            max_trees_per_col: 6,
            max_depth: 2,
            rule_mode: RuleMode::Structural,
            ..Default::default()
        },
    );
    assert!(catf
        .rules
        .rules
        .iter()
        .any(|r| matches!(r, StructuralRule::Arith { .. })));
    assert!(catf
        .rules
        .rules
        .iter()
        .any(|r| matches!(r, StructuralRule::Ineq { .. })));
    assert!(catf
        .rules
        .rules
        .iter()
        .any(|r| matches!(r, StructuralRule::Excl { .. })));
    assert!(catf
        .rules
        .rules
        .iter()
        .any(|r| matches!(r, StructuralRule::Miss { .. })));

    let teacher = CrossFittedTeacher::fit(
        Head::Regression,
        &build_x,
        &y,
        &TeacherConfig {
            folds: 3,
            max_trees_d4: 30,
            max_trees_d6: 30,
            seed: 3,
        },
    );
    let probes = catf.sample_x(600, 3, 1);
    let student = DistilledStudent::fit(
        &teacher,
        &build_x,
        &probes,
        &build_x,
        &y,
        &DistilConfig {
            max_trees: 80,
            depth: 5,
            seed: 3,
            ..Default::default()
        },
    );
    let samples = catf.sample_with_student(&student, 400, 7, 0);
    // Verify rules hold on sampled X AND y tracks post-rule x2 (= x0 + x1) closely (`T32`).
    assert!(catf.rules.violation_rate(&samples.x) <= 0.02);
    let mae: f64 = (0..400)
        .map(|r| (samples.y[r] - 3.0 * samples.x[2][r]).abs())
        .sum::<f64>()
        / 400.0;
    assert!(mae < 0.35, "post-rule target MAE too high: {mae}");
}

#[test]
fn discrete_latent_correlation_calibration_undoes_attenuation() {
    // Two balanced binary columns with empirical correlation ~0.70.
    let mut rng = Rng::new(88);
    let n = 1200usize;
    let mut b0 = Vec::with_capacity(n);
    let mut b1 = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.below(2) as f64;
        let y = if rng.uniform() < 0.85 { x } else { 1.0 - x };
        b0.push(x);
        b1.push(y);
    }
    let build_x = vec![b0, b1];
    let catf = CatfModel::fit(
        &build_x,
        &build_x,
        &[true, true],
        &CatfConfig {
            max_trees_per_col: 0, // Pure calibrated Gaussian copula anchor!
            rule_mode: RuleMode::None,
            ..Default::default()
        },
    );
    let s = catf.sample_x(4000, 88, 0);
    let agree = (0..4000)
        .filter(|&r| (s[0][r] - s[1][r]).abs() < 1e-6)
        .count() as f64
        / 4000.0;
    // Target agreement = 0.85 (corr = 0.70); uncalibrated 2/pi would drop corr to ~0.45!
    assert!(
        (agree - 0.85).abs() < 0.05,
        "calibrated copula agreement {agree} drifted from 0.85"
    );
}
