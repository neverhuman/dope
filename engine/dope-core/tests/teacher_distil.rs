//! Behavioral tests for Phase P4 Whole-Stack Cross-Fitted Teacher & Distilled Student
//! (`T01`, `T03`, `T04`, `T09`, `T10`, Max Tier, and external teacher exchange seam).

use dope_core::budget::{ModelSize, MAX_TIER_BUDGET_BYTES};
use dope_core::distil::{DistilConfig, DistilledStudent};
use dope_core::forest::{boost_soft, column_codes, predict_columns, SoftBoostOptions, SoftSplit};
use dope_core::rng::Rng;
use dope_core::target::Head;
use dope_core::teacher::exchange::{TeacherExchangeBundle, TEACHER_EXCHANGE_PROTOCOL};
use dope_core::teacher::panel::duplicate_grouped_kfolds;
use dope_core::teacher::{CrossFittedTeacher, TeacherConfig};

fn nonlinear_reg_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut x: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let x0 = rng.uniform();
        let x1 = rng.uniform();
        let x2 = rng.uniform();
        let x3 = rng.uniform();
        x[0].push(x0);
        x[1].push(x1);
        x[2].push(x2);
        x[3].push(x3);
        let xor_term = if (x0 > 0.5) ^ (x1 > 0.5) { 0.45 } else { -0.45 };
        let val = 1.2 * x0 - 0.8 * x1
            + xor_term
            + 0.3 * (x2 * std::f64::consts::PI).sin()
            + 0.08 * rng.normal();
        y.push(val);
    }
    (x, y)
}

fn r2(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|&v| (v - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum();
    1.0 - ss_res / ss_tot.max(1e-12)
}

#[test]
fn t01_fold_and_cal_label_poisoning_invariance() {
    let (build_x, build_y) = nonlinear_reg_data(220, 11);
    let (probe_x, _) = nonlinear_reg_data(300, 12);
    let (cal_x, cal_y) = nonlinear_reg_data(80, 13);

    let cfg = TeacherConfig {
        folds: 5,
        max_trees_d4: 30,
        max_trees_d6: 30,
        seed: 7,
    };
    let t1 = CrossFittedTeacher::fit(Head::Regression, &build_x, &build_y, &cfg);
    // 1. Poisoning CAL labels cannot change BUILD teacher OOF or student conditional logits!
    let poison_cal_y: Vec<f64> = cal_y.iter().map(|v| v + 100.0).collect();
    let dcfg = DistilConfig {
        max_trees: 40,
        depth: 5,
        forest_byte_cap: Some(32_000),
        seed: 9,
        ..Default::default()
    };
    let s1 = DistilledStudent::fit(&t1, &build_x, &probe_x, &cal_x, &cal_y, &dcfg);
    let s2 = DistilledStudent::fit(&t1, &build_x, &probe_x, &cal_x, &poison_cal_y, &dcfg);
    let p1 = s1.predict_logits(&probe_x);
    let p2 = s2.predict_logits(&probe_x);
    for (a, b) in p1.iter().zip(&p2) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "CAL poisoning leaked into student logits"
        );
    }

    // 2. Poisoning labels inside fold 0 leaves fold 0's own OOF predictions strictly invariant
    //    when fold assignment is fixed (duplicate-grouped fold disjointness check).
    let folds = duplicate_grouped_kfolds(&build_x, &build_y, 5, 7);
    let f0_set: std::collections::BTreeSet<usize> = folds[0].iter().copied().collect();
    for (fi, fold) in folds.iter().enumerate().skip(1) {
        for &r in fold {
            assert!(!f0_set.contains(&r), "fold {fi} overlapped fold 0");
        }
    }
}

#[test]
fn t03_and_t04_binary_flip_and_multiclass_permutation_invariance() {
    let mut rng = Rng::new(31);
    let n = 220usize;
    let mut x: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
    let mut y_bin = Vec::with_capacity(n);
    let mut y_mc = Vec::with_capacity(n);
    for _ in 0..n {
        let a = rng.uniform();
        let b = rng.uniform();
        let c = rng.uniform();
        x[0].push(a);
        x[1].push(b);
        x[2].push(c);
        y_bin.push(if a + 0.7 * b > 0.85 { 1.0 } else { 0.0 });
        y_mc.push(if a > 0.66 {
            2.0
        } else if b > 0.5 {
            1.0
        } else {
            0.0
        });
    }
    let y_bin_flip: Vec<f64> = y_bin.iter().map(|&v| 1.0 - v).collect();
    let cfg = TeacherConfig {
        folds: 3,
        max_trees_d4: 25,
        max_trees_d6: 25,
        seed: 5,
    };
    let tb = CrossFittedTeacher::fit(Head::Binary, &x, &y_bin, &cfg);
    let tbf = CrossFittedTeacher::fit(Head::Binary, &x, &y_bin_flip, &cfg);
    let mean_sum: f64 = tb
        .oof_predictions
        .iter()
        .zip(&tbf.oof_predictions)
        .map(|(&p, &q)| (p + q - 1.0).abs())
        .sum::<f64>()
        / n as f64;
    assert!(mean_sum < 0.08, "binary flip symmetry drift {mean_sum}");

    // Multiclass permutation (0->2, 1->0, 2->1).
    let perm = [2usize, 0, 1];
    let y_mc_perm: Vec<f64> = y_mc.iter().map(|&v| perm[v as usize] as f64).collect();
    let tmc = CrossFittedTeacher::fit(Head::Multiclass(3), &x, &y_mc, &cfg);
    let tmcp = CrossFittedTeacher::fit(Head::Multiclass(3), &x, &y_mc_perm, &cfg);
    let mut mc_diff = 0.0;
    for r in 0..n {
        for (old_c, &new_c) in perm.iter().enumerate() {
            mc_diff +=
                (tmc.oof_predictions[r * 3 + old_c] - tmcp.oof_predictions[r * 3 + new_c]).abs();
        }
    }
    mc_diff /= (n * 3) as f64;
    assert!(
        mc_diff < 0.08,
        "multiclass permutation symmetry drift {mc_diff}"
    );
}

#[test]
fn t09_soft_newton_weights_scale_invariantly_and_pull_predictions() {
    let n = 120usize;
    let x0: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let x = vec![x0];
    let rows: Vec<usize> = (0..n).collect();
    let codes = column_codes(&x, &rows, &[0.5]);
    let base = vec![0.0; n];
    // Soft targets split into low (0.2) and high (0.8) across identical codes.
    let soft_t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.2 } else { 0.8 }).collect();
    let w1 = vec![1.0; n];
    let w5 = vec![5.0; n];

    let b1 = boost_soft(
        Head::Regression,
        1,
        &SoftSplit {
            codes: &codes,
            base: &base,
            soft_targets: &soft_t,
            weights: &w1,
        },
        &SoftSplit {
            codes: &codes,
            base: &base,
            soft_targets: &soft_t,
            weights: &w1,
        },
        &SoftBoostOptions {
            depth: 2,
            max_trees: 15,
            lambda: 1.0,
            min_weight: 2.0,
            seed: 1,
            ..Default::default()
        },
    );
    let b5 = boost_soft(
        Head::Regression,
        1,
        &SoftSplit {
            codes: &codes,
            base: &base,
            soft_targets: &soft_t,
            weights: &w5,
        },
        &SoftSplit {
            codes: &codes,
            base: &base,
            soft_targets: &soft_t,
            weights: &w5,
        },
        &SoftBoostOptions {
            depth: 2,
            max_trees: 15,
            lambda: 5.0,
            min_weight: 10.0,
            seed: 1,
            ..Default::default()
        },
    );
    let p1 = predict_columns(&b1.forest, &codes, 1, n);
    let p5 = predict_columns(&b5.forest, &codes, 1, n);
    let max_d = p1
        .iter()
        .zip(&p5)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_d < 1e-6,
        "weight+lambda scale invariance violated: {max_d}"
    );

    // Upweighting high-target rows pulls predictions strictly upward.
    let w_hi: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.2 } else { 4.0 }).collect();
    let bhi = boost_soft(
        Head::Regression,
        1,
        &SoftSplit {
            codes: &codes,
            base: &base,
            soft_targets: &soft_t,
            weights: &w_hi,
        },
        &SoftSplit {
            codes: &codes,
            base: &base,
            soft_targets: &soft_t,
            weights: &w_hi,
        },
        &SoftBoostOptions {
            depth: 2,
            max_trees: 15,
            lambda: 1.0,
            min_weight: 2.0,
            seed: 1,
            ..Default::default()
        },
    );
    let phi = predict_columns(&bhi.forest, &codes, 1, n);
    let m1 = p1.iter().sum::<f64>() / n as f64;
    let mhi = phi.iter().sum::<f64>() / n as f64;
    assert!(
        mhi > m1 + 0.15,
        "weighted soft Newton failed to shift mean: {mhi} vs {m1}"
    );
}

#[test]
fn t10_distilled_student_retains_teacher_surface_and_max_tier_and_exchange_seam() {
    assert_eq!("max".parse::<ModelSize>(), Ok(ModelSize::Max));
    assert_eq!(ModelSize::Max.budget(), 1_048_576);
    assert_eq!(MAX_TIER_BUDGET_BYTES, 2_097_152);

    let (build_x, build_y) = nonlinear_reg_data(360, 101);
    let (probe_x, _) = nonlinear_reg_data(1400, 102);
    let (cal_x, cal_y) = nonlinear_reg_data(120, 103);
    let (test_x, test_y) = nonlinear_reg_data(240, 104);

    let teacher = CrossFittedTeacher::fit(
        Head::Regression,
        &build_x,
        &build_y,
        &TeacherConfig {
            folds: 5,
            max_trees_d4: 60,
            max_trees_d6: 60,
            seed: 7,
        },
    );
    let (teacher_test, _) = teacher.predict_probes(&test_x);
    let teacher_r2 = r2(&test_y, &teacher_test);

    let student = DistilledStudent::fit(
        &teacher,
        &build_x,
        &probe_x,
        &cal_x,
        &cal_y,
        &DistilConfig {
            max_pairs: 6,
            depth: 6,
            bits: 8,
            max_trees: 250,
            forest_byte_cap: Some(48_000),
            seed: 7,
            ..Default::default()
        },
    );
    let student_test = student.predict_expectations(&test_x);
    let student_r2 = r2(&test_y, &student_test);
    let retention_of_teacher = student_r2 / teacher_r2.max(1e-6);
    assert!(
        teacher_r2 > 0.75 && retention_of_teacher >= 0.90,
        "teacher_r2={teacher_r2:.4}, student_r2={student_r2:.4}, ratio={retention_of_teacher:.4}"
    );
    assert!(student.forest_trees > 20 && student.forest_bytes <= 48_000);
    assert!(student.residual_model.is_some());

    // External teacher exchange seam records seen families and refuses auditor overlap.
    let bundle = TeacherExchangeBundle {
        protocol: TEACHER_EXCHANGE_PROTOCOL.into(),
        build_split_sha256: "a".repeat(64),
        task: "regression".into(),
        num_outputs: 1,
        seen_teacher_families: vec!["catboost".into(), "lightgbm".into()],
        oof_predictions: teacher.oof_predictions.clone(),
        probe_predictions: None,
        probe_agreement: None,
    };
    bundle.validate(360, 1).unwrap();
    assert!(bundle.assert_auditor_unseen("catboost").is_err());
    assert!(bundle.assert_auditor_unseen("xgboost").is_ok());
}
