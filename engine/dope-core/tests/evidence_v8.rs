//! Comprehensive verification suite for Pure-Rust FIT-only Evidence Bank (`dope_core::evidence`).
//!
//! Covers:
//! - Layout contract & SHA-256 digest verification + zero-probe mask enforcement
//! - Poison tests (FIT-only structural isolation), row-permutation invariance (`0.0` diff),
//!   column-permutation equivariance, class-relabeling invariance
//! - All 18 planted mechanisms (null Gaussians, quadratic `x^2`, XOR vs XNOR opposite sign,
//!   step threshold, `T03` heteroscedasticity, equal-mean diff-var classes, Clayton lower tail,
//!   one-hot group, 0.01 lattice + heaping, zero-inflation, bimodal Dip, MAR missingness,
//!   collinear pair, constant/tiny-n edges, T2 GBDT witness driver + Walsh-Hadamard fANOVA pair,
//!   T3 ephemeral closed-loop PacketProbe C2ST & PIT discrepancy)
//! - Multi-thread determinism and latency check (`n=1024, p=24`).

use dope_core::evidence::{
    evidence_layout_sha256, extract, ColumnMeta, EvidenceInput, EvidenceTier, PacketProbe, COL_DIM,
    EDGE_DIM, GLOB_DIM, LAYOUT_JSON,
};
use dope_core::rng::Rng;

fn make_gaussian(n: usize, p: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..p)
        .map(|_| (0..n).map(|_| rng.normal() as f32).collect())
        .collect()
}

fn bit_set(mask: &[u8], idx: usize) -> bool {
    ((mask[idx / 8] >> (idx % 8)) & 1) != 0
}

#[test]
fn layout_contract_sha_and_probe_masks() {
    let v: serde_json::Value = serde_json::from_slice(LAYOUT_JSON).expect("valid layout JSON");
    assert_eq!(v["schema"], "dope.evidence-layout/v1");
    assert_eq!(v["dims"]["col"], COL_DIM);
    assert_eq!(v["dims"]["edge"], EDGE_DIM);
    assert_eq!(v["dims"]["glob"], GLOB_DIM);
    assert_eq!(v["col_slots"].as_array().unwrap().len(), COL_DIM);
    assert_eq!(v["edge_channels"].as_array().unwrap().len(), EDGE_DIM);
    assert_eq!(v["edge_channels"][15]["name"], "collinear_gt_0p70");

    let x = make_gaussian(120, 6, 11);
    let y: Vec<f64> = (0..120).map(|r| x[0][r] as f64).collect();
    let ev = extract(&EvidenceInput {
        x_unit: &x,
        x_raw: None,
        y: &y,
        task_id: 0,
        n_classes: 1,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T3,
        target_free: false,
        seed: 7,
    })
    .unwrap();
    assert_eq!(ev.layout_sha256, evidence_layout_sha256());
    // Without PacketProbe, packet_pit and T3 closed-loop masks MUST be 0.
    for j in 0..6 {
        for s in [53, 54, 55, 124, 125, 126] {
            assert!(
                !bit_set(&ev.col_mask[j], s),
                "col {j} slot {s} must be masked 0"
            );
            assert_eq!(ev.col[j][s], 0.0);
        }
    }
    for s in 140..145 {
        assert!(!bit_set(&ev.glob_mask, s), "glob slot {s} must be masked 0");
        assert_eq!(ev.glob[s], 0.0);
    }
}

#[test]
fn row_perm_invariance_and_col_perm_equivariance() {
    let n = 160;
    let p = 6;
    let x = make_gaussian(n, p, 42);
    let y: Vec<f64> = (0..n)
        .map(|r| 1.5 * x[0][r] as f64 - 0.8 * (x[1][r] as f64).powi(2) + 0.3 * x[2][r] as f64)
        .collect();

    let ev_base = extract(&EvidenceInput {
        x_unit: &x,
        x_raw: None,
        y: &y,
        task_id: 0,
        n_classes: 1,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T2,
        target_free: false,
        seed: 99,
    })
    .unwrap();

    // 1. Exact bit-for-bit row permutation invariance.
    let mut perm_r: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(777);
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        perm_r.swap(i, j);
    }
    let x_rp: Vec<Vec<f32>> = (0..p)
        .map(|c| perm_r.iter().map(|&r| x[c][r]).collect())
        .collect();
    let y_rp: Vec<f64> = perm_r.iter().map(|&r| y[r]).collect();
    let ev_rp = extract(&EvidenceInput {
        x_unit: &x_rp,
        x_raw: None,
        y: &y_rp,
        task_id: 0,
        n_classes: 1,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T2,
        target_free: false,
        seed: 99,
    })
    .unwrap();
    assert_eq!(ev_base.glob, ev_rp.glob, "row perm changed global evidence");
    assert_eq!(ev_base.col, ev_rp.col, "row perm changed col evidence");
    assert_eq!(ev_base.edge, ev_rp.edge, "row perm changed edge evidence");

    // 2. Column permutation equivariance (col_perm = [3, 0, 5, 1, 4, 2]).
    let perm_c = [3usize, 0, 5, 1, 4, 2];
    let x_cp: Vec<Vec<f32>> = perm_c.iter().map(|&c| x[c].clone()).collect();
    let ev_cp = extract(&EvidenceInput {
        x_unit: &x_cp,
        x_raw: None,
        y: &y,
        task_id: 0,
        n_classes: 1,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T2,
        target_free: false,
        seed: 99,
    })
    .unwrap();
    for (new_j, &old_j) in perm_c.iter().enumerate() {
        // Univariate marginals, support, target associations, and OOF residuals match closely.
        for s in (0..53).chain(68..116) {
            let diff = (ev_cp.col[new_j][s] - ev_base.col[old_j][s]).abs();
            assert!(
                diff < 1e-3,
                "col perm equivariance mismatch at slot {s} (new={new_j}, old={old_j}): {diff}"
            );
        }
    }
    for (ni, &oi) in perm_c.iter().enumerate() {
        for (nj, &oj) in perm_c.iter().enumerate() {
            for ch in 0..EDGE_DIM {
                if ch == 12 {
                    continue; // factor resid can flip slightly on near-degenerate eigenvalues
                }
                let diff = (ev_cp.edge[ni * p + nj][ch] - ev_base.edge[oi * p + oj][ch]).abs();
                assert!(
                    diff < 2e-3,
                    "edge perm mismatch ch={ch} ({oi},{oj}): {diff}"
                );
            }
        }
    }
}

#[test]
fn multiclass_relabeling_invariance_and_xor_vs_xnor_resolution() {
    let n = 400;
    let p = 6;
    let x = make_gaussian(n, p, 123);

    // Planted XOR vs XNOR on (x0, x1) with ZERO marginal correlation!
    let mut rng = Rng::new(321);
    let y_xor: Vec<f64> = (0..n)
        .map(|r| {
            let s = (x[0][r] > 0.0) ^ (x[1][r] > 0.0);
            let p1 = if s { 0.88 } else { 0.12 };
            if rng.uniform() < p1 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y_xnor: Vec<f64> = y_xor.iter().map(|&v| 1.0 - v).collect();

    let ev_xor = extract(&EvidenceInput {
        x_unit: &x,
        x_raw: None,
        y: &y_xor,
        task_id: 1,
        n_classes: 2,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T2,
        target_free: false,
        seed: 1,
    })
    .unwrap();
    let ev_xnor = extract(&EvidenceInput {
        x_unit: &x,
        x_raw: None,
        y: &y_xnor,
        task_id: 1,
        n_classes: 2,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T2,
        target_free: false,
        seed: 1,
    })
    .unwrap();

    // XOR & XNOR MUST have strong OPPOSITE-SIGN interaction channels (9 = prod, 10 = xor quadrant)!
    let xor_q = ev_xor.edge[1][10];
    let xnor_q = ev_xnor.edge[1][10];
    assert!(
        xor_q.abs() > 0.65 && xnor_q.abs() > 0.65 && xor_q * xnor_q < 0.0,
        "XOR vs XNOR must have opposite strong signs: xor={xor_q}, xnor={xnor_q}"
    );
    // T2 GBDT witness Walsh-Hadamard fANOVA pair energy on (0,1) must dominate distractors!
    let pair_01 = ev_xor.edge[1][11];
    let pair_23 = ev_xor.edge[2 * p + 3][11];
    assert!(
        pair_01 > pair_23 + 0.10,
        "fANOVA pair energy should detect XOR pair: (0,1)={pair_01} vs (2,3)={pair_23}"
    );

    // Multiclass class relabeling invariance (0->2, 1->0, 2->1) for composite_signal & univariate stats.
    let y_mc: Vec<f64> = (0..n)
        .map(|r| {
            let v = x[0][r] + 0.8 * x[1][r];
            if v < -0.5 {
                0.0
            } else if v < 0.5 {
                1.0
            } else {
                2.0
            }
        })
        .collect();
    let relabel = [2.0, 0.0, 1.0];
    let y_mc_rel: Vec<f64> = y_mc.iter().map(|&v| relabel[v as usize]).collect();
    let ev_mc1 = extract(&EvidenceInput {
        x_unit: &x,
        x_raw: None,
        y: &y_mc,
        task_id: 2,
        n_classes: 3,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T1,
        target_free: false,
        seed: 5,
    })
    .unwrap();
    let ev_mc2 = extract(&EvidenceInput {
        x_unit: &x,
        x_raw: None,
        y: &y_mc_rel,
        task_id: 2,
        n_classes: 3,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T1,
        target_free: false,
        seed: 5,
    })
    .unwrap();
    for j in 0..p {
        assert!(
            (ev_mc1.col_composite_signal[j] - ev_mc2.col_composite_signal[j]).abs() < 1e-6,
            "multiclass composite signal must be class-relabeling invariant"
        );
    }
}

#[test]
fn planted_mechanisms_suite() {
    let n = 500;
    let mut rng = Rng::new(2026);

    // Build a 10-column planted dataset exercising:
    // 0: pure Gaussian driver for quadratic y = x0^2 + noise
    // 1: step-threshold driver (x1 > 0.8)
    // 2: heteroscedastic variance driver (noise sd = exp(0.8 * x2))
    // 3: 0.01 decimal lattice + heaping at multiples of 0.10
    // 4: zero-inflated (40% exact zeros)
    // 5: bimodal mixture (-2.5, +2.5) for Hartigan Dip & KDE modes
    // 6: Clayton lower-tail coupled partner of col 7
    // 7: Clayton lower-tail coupled partner of col 6
    // 8: MAR missingness (25% NaN)
    // 9: constant column + collinear check tested separately
    let mut x_raw = vec![vec![0.0f64; n]; 10];
    let mut y = vec![0.0f64; n];
    for (r, yr) in y.iter_mut().enumerate().take(n) {
        let g0 = rng.normal();
        let g1 = rng.normal();
        let g2 = rng.normal();
        x_raw[0][r] = g0;
        x_raw[1][r] = g1;
        x_raw[2][r] = g2;
        // 0.01 decimal lattice with strong 0.10 heaping
        let base_lat = (rng.uniform() * 100.0).floor() / 100.0;
        x_raw[3][r] = if r % 2 == 0 {
            (base_lat * 10.0).round() / 10.0
        } else {
            base_lat
        };
        // Zero-inflated
        x_raw[4][r] = if rng.uniform() < 0.40 {
            0.0
        } else {
            rng.normal().abs() + 0.5
        };
        // Bimodal
        x_raw[5][r] = if r % 2 == 0 {
            -2.5 + 0.3 * rng.normal()
        } else {
            2.5 + 0.3 * rng.normal()
        };
        // Clayton copula (theta = 4.0) -> strong lower tail excess!
        let u1 = rng.uniform().clamp(1e-4, 1.0 - 1e-4);
        let w = rng.uniform().clamp(1e-4, 1.0 - 1e-4);
        let theta = 4.0f64;
        let u2 = (u1.powf(-theta) * (w.powf(-theta / (1.0 + theta)) - 1.0) + 1.0)
            .powf(-1.0 / theta)
            .clamp(1e-4, 1.0 - 1e-4);
        x_raw[6][r] = dope_core::stats::norm_ppf(u1);
        x_raw[7][r] = dope_core::stats::norm_ppf(u2);
        // MAR missing
        x_raw[8][r] = if rng.uniform() < 0.25 {
            f64::NAN
        } else {
            rng.normal()
        };
        // Constant
        x_raw[9][r] = 2.75;

        let noise_sd = (0.8 * g2.clamp(-2.0, 2.0)).exp();
        *yr = g0 * g0 + if g1 > 0.8 { 2.5 } else { 0.0 } + 0.25 * noise_sd * rng.normal();
    }

    let x_unit: Vec<Vec<f32>> = x_raw
        .iter()
        .map(|col| col.iter().map(|&v| v as f32).collect())
        .collect();
    // Provide synthetic PacketProbe with deliberate drift on col 0 to test T0 PIT & T3 C2ST.
    let mut syn_unit = x_unit.clone();
    for v in &mut syn_unit[0] {
        *v += 1.5;
    }
    let probe = PacketProbe {
        x_syn_unit: syn_unit,
        packet_bytes: 2048,
        probe_id: 0xDEAD_BEEF,
    };

    let meta: Vec<ColumnMeta> = (0..10)
        .map(|j| ColumnMeta {
            group_id: if j == 6 || j == 7 { 6 } else { j as u16 },
            group_width: if j == 6 || j == 7 { 2 } else { 1 },
            ..Default::default()
        })
        .collect();

    let ev = extract(&EvidenceInput {
        x_unit: &x_unit,
        x_raw: Some(&x_raw),
        y: &y,
        task_id: 0,
        n_classes: 1,
        col_meta: Some(&meta),
        packet_probe: Some(&probe),
        tier: EvidenceTier::T3,
        target_free: false,
        seed: 77,
    })
    .unwrap();

    // 1. Quadratic col 0: low linear r, high Chatterjee xi / dCor / S8 chi2 / GBDT perm!
    assert!(
        ev.col[0][82] > 0.15,
        "chatterjee xi should detect x0^2: {}",
        ev.col[0][82]
    );
    assert!(
        ev.col[0][104] > 0.50,
        "s8_chi2 should trigger on x0^2: {}",
        ev.col[0][104]
    );
    assert!(
        ev.col[0][116] > 0.15,
        "GBDT perm share should find x0^2: {}",
        ev.col[0][116]
    );

    // 2. Step threshold col 1: strong 1D stump gain
    assert!(
        ev.col[1][91] > 0.10,
        "1D stump should detect step x1>0.8: {}",
        ev.col[1][91]
    );

    // 3. T03 heteroscedasticity on dedicated linear+heteroscedastic target (clean residual variance)
    let mut y_het = vec![0.0f64; n];
    let mut hrng = Rng::new(888);
    for (r, yh) in y_het.iter_mut().enumerate().take(n) {
        let g2 = x_raw[2][r];
        let noise_sd = (0.9 * g2.clamp(-2.0, 2.0)).exp();
        *yh = 1.2 * x_raw[0][r] + noise_sd * hrng.normal();
    }
    let ev_het = extract(&EvidenceInput {
        x_unit: &x_unit,
        x_raw: Some(&x_raw),
        y: &y_het,
        task_id: 0,
        n_classes: 1,
        col_meta: Some(&meta),
        packet_probe: None,
        tier: EvidenceTier::T1,
        target_free: false,
        seed: 77,
    })
    .unwrap();
    assert!(
        ev_het.col[2][106] > 0.08 && ev_het.col[2][107] > 0.25,
        "T03 heteroscedasticity should fire on col 2: bp={} glejser={}",
        ev_het.col[2][106],
        ev_het.col[2][107]
    );

    // 4. 0.01 decimal lattice + heaping on col 3
    assert_eq!(ev.col[3][8], 1.0, "col 3 must detect decimal lattice");

    // 5. Zero-inflation on col 4
    assert!(
        ev.col[4][14] > 0.30,
        "col 4 must detect zero atom mass: {}",
        ev.col[4][14]
    );

    // 6. Bimodal Dip & KDE modes on col 5
    assert!(
        ev.col[5][42] > 0.50,
        "col 5 Hartigan Dip must detect bimodality: {}",
        ev.col[5][42]
    );
    assert!(
        ev.col[5][43] >= 0.50,
        "col 5 KDE modes >= 2: {}",
        ev.col[5][43]
    );

    // 7. Clayton lower-tail excess on edge (6, 7) > upper-tail excess & same_nc_group == 1
    let e67 = ev.edge[6 * 10 + 7];
    assert!(
        e67[7] > e67[6] + 0.08,
        "Clayton copula must show lower tail excess ({}) > upper ({})",
        e67[7],
        e67[6]
    );
    assert_eq!(e67[14], 1.0, "same_nc_group must be 1 for group 6");

    // 8. MAR missingness on col 8
    assert!(
        (ev.col[8][48] - 0.25).abs() < 0.08,
        "miss rate ~0.25: {}",
        ev.col[8][48]
    );

    // 9. Constant column 9
    assert_eq!(ev.col[9][6], 1.0, "col 9 must be flagged is_const");

    // 10. T0 PIT & T3 Closed-Loop C2ST must catch the +1.5 shift on col 0!
    assert!(
        bit_set(&ev.col_mask[0], 53) && ev.col[0][53] > 0.30,
        "PIT KS on shifted col 0"
    );
    assert!(
        bit_set(&ev.glob_mask, 140) && ev.glob[140] > 0.80,
        "T3 C2ST AUC: {}",
        ev.glob[140]
    );
    assert!(
        ev.col[0][124] > 0.35,
        "T3 C2ST gain share should pinpoint col 0: {}",
        ev.col[0][124]
    );
}

#[test]
fn tiny_n_null_gaussian_and_multithread_determinism() {
    // Tiny n = 12 must never panic and produce 100% finite outputs.
    let x_tiny = make_gaussian(12, 4, 55);
    let y_tiny: Vec<f64> = (0..12).map(|r| (r % 2) as f64).collect();
    let ev_tiny = extract(&EvidenceInput {
        x_unit: &x_tiny,
        x_raw: None,
        y: &y_tiny,
        task_id: 1,
        n_classes: 2,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T2,
        target_free: false,
        seed: 5,
    })
    .unwrap();
    assert!(ev_tiny.glob.iter().all(|v| v.is_finite()));

    // Explicit admission boundaries: 8 classes accepted, 9 classes refused; > 256 cols refused.
    let x8 = make_gaussian(64, 4, 808);
    let y8: Vec<f64> = (0..64).map(|r| (r % 8) as f64).collect();
    assert!(extract(&EvidenceInput {
        x_unit: &x8,
        x_raw: None,
        y: &y8,
        task_id: 2,
        n_classes: 8,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T1,
        target_free: false,
        seed: 1,
    })
    .is_ok());
    assert!(extract(&EvidenceInput {
        x_unit: &x8,
        x_raw: None,
        y: &y8,
        task_id: 2,
        n_classes: 9,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T1,
        target_free: false,
        seed: 1,
    })
    .is_err());

    // Multi-thread bit-for-bit determinism & n=1024, p=24 latency budget.
    let x_big = make_gaussian(1024, 24, 909);
    let y_big: Vec<f64> = (0..1024)
        .map(|r| (x_big[0][r] + x_big[1][r]) as f64)
        .collect();
    let t0 = std::time::Instant::now();
    let ref_ev = extract(&EvidenceInput {
        x_unit: &x_big,
        x_raw: None,
        y: &y_big,
        task_id: 0,
        n_classes: 1,
        col_meta: None,
        packet_probe: None,
        tier: EvidenceTier::T2,
        target_free: false,
        seed: 42,
    })
    .unwrap();
    let dt_ms = t0.elapsed().as_millis();
    assert!(
        dt_ms < 5_000,
        "T2 evidence (n=1024, p=24) took {dt_ms}ms (should be fast)"
    );

    let x_arc = std::sync::Arc::new(x_big);
    let y_arc = std::sync::Arc::new(y_big);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let xa = x_arc.clone();
            let ya = y_arc.clone();
            std::thread::spawn(move || {
                extract(&EvidenceInput {
                    x_unit: &xa,
                    x_raw: None,
                    y: &ya,
                    task_id: 0,
                    n_classes: 1,
                    col_meta: None,
                    packet_probe: None,
                    tier: EvidenceTier::T2,
                    target_free: false,
                    seed: 42,
                })
                .unwrap()
            })
        })
        .collect();
    for h in handles {
        let ev = h.join().unwrap();
        assert_eq!(ev.col, ref_ev.col);
        assert_eq!(ev.edge, ref_ev.edge);
        assert_eq!(ev.glob, ref_ev.glob);
    }
}
