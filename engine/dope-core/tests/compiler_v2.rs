//! Behavioral tests for Phase P6 Closed-Loop Exact-Artifact Compiler & Audit Witnesses
//! (`T19`, `T20`, `T21`, `T31`, `BUILD/SELECT/CAL` roles, and Ridge Certificate).

use dope_core::audit::{ridge_certificate, LearningSignatureV1, MAX_WITNESSES};
use dope_core::compile::v2::{
    project_moments_damped_newton, resolve_carry_forward_tiers, run_exact_artifact_loop,
    select_best_decoded, CandidateScore, CompilerMove, CompilerRoles, DecodedCandidateArtifact,
    MIN_PROJECTION_ESS,
};
use dope_core::rng::Rng;
use std::collections::BTreeSet;

fn make_real_and_shifted(n: usize, shift: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut x: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let x0 = rng.normal() + shift;
        let x1 = 0.5 * x0 + 0.866 * rng.normal() - 0.5 * shift;
        let x2 = rng.normal();
        let x3 = rng.normal();
        x[0].push(x0);
        x[1].push(x1);
        x[2].push(x2);
        x[3].push(x3);
        y.push(1.4 * x0 - 0.7 * x1 + 0.3 * x2 + 0.25 * rng.normal());
    }
    (x, y)
}

#[test]
fn t19_projection_damped_newton_cuts_witness_gap_with_ess_ge_030() {
    let (rx, ry) = make_real_and_shifted(800, 0.0, 19);
    // Deliberately shifted synthetic distribution Q (`shift = 0.45`).
    let (sx, sy) = make_real_and_shifted(1200, 0.45, 20);
    let sig = LearningSignatureV1::fit(&rx, &ry);
    assert!(sig.real_witness_means.len() <= MAX_WITNESSES);
    assert!(sig.real_witness_means.len() >= 15);

    let proj = project_moments_damped_newton(&sig, &sx, &sy, None);
    let lad = &proj.ladder;
    assert!(
        lad.ess >= MIN_PROJECTION_ESS,
        "ESS {} violated minimum floor {}",
        lad.ess,
        MIN_PROJECTION_ESS
    );
    assert!(
        lad.gap_reweighted < lad.gap_raw_q * 0.45,
        "projection failed to cut witness gap substantially: raw={} -> rew={}",
        lad.gap_raw_q,
        lad.gap_reweighted
    );
    assert!(
        lad.gap_compiled < lad.gap_raw_q * 0.55,
        "compiled direct sampler failed to preserve projected moments: raw={} -> comp={}",
        lad.gap_raw_q,
        lad.gap_compiled
    );
    assert_eq!(proj.compiled_indices.len(), 1200);
    let sum_w: f64 = proj.weights.iter().sum();
    assert!((sum_w - 1.0).abs() < 1e-9);
}

#[test]
fn t20_bucketed_lexicographic_selector_is_strictly_transitive() {
    let mut rng = Rng::new(20);
    let cands: Vec<CandidateScore> = (0..60)
        .map(|i| CandidateScore {
            id: format!("cand_{i:02}"),
            bytes: 4096 + (rng.below(8) * 1024),
            v_max: (rng.below(10) as f64) * 0.011,
            v_mean: (rng.below(10) as f64) * 0.011,
            l_pred: 0.10 + (rng.below(20) as f64) * 0.0011,
            refit_lineage_sha256: "0".repeat(64),
        })
        .collect();
    // Exhaustive transitivity check on all triples (a <= b && b <= c => a <= c).
    for a in &cands {
        for b in &cands {
            if a.cmp_lex(b).is_le() {
                for c in &cands {
                    if b.cmp_lex(c).is_le() {
                        assert!(a.cmp_lex(c).is_le(), "transitivity broken");
                    }
                }
            }
        }
    }
    let best = select_best_decoded(&cands).unwrap();
    for other in &cands {
        assert!(best.cmp_lex(other).is_le());
    }
}

#[test]
fn t21_carry_forward_tiers_prevent_inversions_and_loop_is_deterministic() {
    let c_small = CandidateScore {
        id: "t4k".into(),
        bytes: 3800,
        v_max: 0.01,
        v_mean: 0.008,
        l_pred: 0.200,
        refit_lineage_sha256: "a".repeat(64),
    };
    // 16 KiB candidate improves by only 0.002 (< 0.005 margin) -> must carry forward 4 KiB!
    let c_med_marginal = CandidateScore {
        id: "t16k_tiny".into(),
        bytes: 15_000,
        v_max: 0.01,
        v_mean: 0.008,
        l_pred: 0.198,
        refit_lineage_sha256: "b".repeat(64),
    };
    // 64 KiB candidate improves by 0.020 (>= 0.005) -> replaces carried-forward!
    let c_large_real = CandidateScore {
        id: "t64k_win".into(),
        bytes: 52_000,
        v_max: 0.01,
        v_mean: 0.008,
        l_pred: 0.180,
        refit_lineage_sha256: "c".repeat(64),
    };
    // Max candidate regresses slightly -> must carry forward 64 KiB (zero tier inversion!).
    let c_max_worse = CandidateScore {
        id: "tmax_worse".into(),
        bytes: 900_000,
        v_max: 0.01,
        v_mean: 0.008,
        l_pred: 0.183,
        refit_lineage_sha256: "d".repeat(64),
    };
    let ladder = resolve_carry_forward_tiers(&[
        (4_096, vec![c_small.clone()]),
        (16_384, vec![c_med_marginal]),
        (65_536, vec![c_large_real.clone()]),
        (1_048_576, vec![c_max_worse]),
    ]);
    assert_eq!(ladder.len(), 4);
    assert_eq!(ladder[0].winner.id, "t4k");
    assert!(ladder[1].carried_forward && ladder[1].winner.id == "t4k");
    assert!(!ladder[2].carried_forward && ladder[2].winner.id == "t64k_win");
    assert!(ladder[3].carried_forward && ladder[3].winner.id == "t64k_win");
    // Monotone non-increasing l_pred across all ascending tiers!
    for w in ladder.windows(2) {
        assert!(w[1].winner.l_pred <= w[0].winner.l_pred + 1e-12);
    }

    // Closed-loop exact-artifact driver is deterministic and enforces hard byte caps.
    let moves = [
        CompilerMove::YPlus,
        CompilerMove::XjPlus,
        CompilerMove::PPlus,
    ];
    let run_once = || {
        run_exact_artifact_loop(16_384, "roles_sha", &moves, |mv, step, _| {
            let (sz, l_pred) = match mv {
                CompilerMove::YPlus => (8000, 0.21),
                CompilerMove::XjPlus => (20_000, 0.15), // over 16 KiB cap -> MUST be refused!
                CompilerMove::PPlus => (12_000, 0.18),
                _ => (4000, 0.25),
            };
            Some(DecodedCandidateArtifact {
                mv,
                bytes: vec![step as u8; sz],
                packet_sha256: String::new(),
                score: CandidateScore {
                    id: format!("m_{}", mv.code()),
                    bytes: sz,
                    v_max: 0.01,
                    v_mean: 0.01,
                    l_pred,
                    refit_lineage_sha256: String::new(),
                },
            })
        })
        .unwrap()
    };
    let (w1, l1) = run_once();
    let (w2, l2) = run_once();
    assert_eq!(w1.score.id, "m_P+");
    assert_eq!(w1.packet_sha256, w2.packet_sha256);
    assert_eq!(l1.lineage_sha256, l2.lineage_sha256);
    assert_eq!(
        l1.evaluated.len(),
        2,
        "over-budget Xj+ move must be rejected"
    );
}

#[test]
fn t31_finite_relative_risk_bound_ridge_certificate_and_role_disjointness() {
    let (rx, ry) = make_real_and_shifted(600, 0.0, 31);
    let (vx, _) = make_real_and_shifted(200, 0.0, 32);
    let (sx_good, sy_good) = make_real_and_shifted(600, 0.05, 33);
    let (sx_bad, sy_bad) = make_real_and_shifted(600, 0.80, 34);

    // 1. Exact `2*eps + delta` relative-risk witness bound (`T31`).
    let sig = LearningSignatureV1::fit(&rx, &ry);
    for (sx, sy) in [(&sx_good, &sy_good), (&sx_bad, &sy_bad)] {
        let rep = sig.evaluate_gaps(sx, sy, None);
        assert!(
            rep.realized_excess_risk <= rep.risk_transfer_bound_2eps_plus_delta + 1e-9,
            "realized excess risk {} exceeded 2*eps+delta bound {}",
            rep.realized_excess_risk,
            rep.risk_transfer_bound_2eps_plus_delta
        );
    }

    // 2. Ridge Certificate tracks fidelity monotonically.
    let rc_good = ridge_certificate(&rx, &ry, &sx_good, &sy_good, &vx, 0.1);
    let rc_bad = ridge_certificate(&rx, &ry, &sx_bad, &sy_bad, &vx, 0.1);
    assert!(rc_good.weight_l2_diff < rc_bad.weight_l2_diff);
    assert!(rc_good.pred_rmse_diff < rc_bad.pred_rmse_diff);

    // 3. BUILD / SELECT / CAL roles are strictly disjoint and cover all rows.
    let roles = CompilerRoles::split(&rx, &ry, 7);
    let mut all = BTreeSet::new();
    for &r in roles
        .build_rows
        .iter()
        .chain(&roles.select_rows)
        .chain(&roles.cal_rows)
    {
        assert!(
            all.insert(r),
            "row {r} leaked across BUILD/SELECT/CAL roles"
        );
    }
    assert_eq!(all.len(), 600);
}
