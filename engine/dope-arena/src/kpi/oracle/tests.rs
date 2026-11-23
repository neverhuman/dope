use super::*;
use crate::kpi::celldata::ScaleCard;

fn cell(task: &str, n_classes: u32) -> Cell {
    Cell {
        cell_id: "dev:h1:full:d0".into(),
        panel_sha256: "p".into(),
        dataset_hash: "h1".into(),
        table_id: 0,
        family_id: "lf-h1".into(),
        task: task.into(),
        n_classes,
        view: "full".into(),
        cols: vec![0, 1, 2, 3],
        data_seed: 0,
        stratum: "headline".into(),
        a_idx: (0..400).collect(),
        b_idx: Vec::new(),
        c_idx: Vec::new(),
        t_idx: Vec::new(),
        sample_seeds: vec![0, 1],
        multipliers: vec![1, 4],
        status: "ok".into(),
    }
}

/// 400 A rows, 4 columns (one a 5-level integer); the target depends on x0 and x1.
fn data(head: Head) -> CellData {
    let mut rng = KpiRng::new(42);
    let (n, p) = (400, 4);
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row = [
            rng.normal(),
            rng.normal(),
            rng.normal(),
            (rng.uniform() * 5.0).floor(),
        ];
        let s = 1.5 * row[0] - row[1] + 0.3 * rng.normal();
        x.extend(row.iter().map(|&v| v as f32));
        y.push(match head {
            Head::Regression => s as f32,
            Head::Binary => f32::from(u8::from(s > 0.0)),
            Head::Multiclass(_) => {
                (if s < -0.7 {
                    0
                } else if s < 0.7 {
                    1
                } else {
                    2
                }) as f32
            }
        });
    }
    let a = Rows { n, p, x, y };
    let empty = Rows {
        n: 0,
        p,
        x: Vec::new(),
        y: Vec::new(),
    };
    CellData {
        head,
        card: ScaleCard::fit(&a, head),
        b: empty.clone(),
        c: empty.clone(),
        t: empty,
        t_donors: None,
        a,
    }
}

fn heads() -> [(Head, &'static str, u32); 3] {
    [
        (Head::Regression, "regression", 0),
        (Head::Binary, "binary", 2),
        (Head::Multiclass(3), "multiclass", 3),
    ]
}

#[test]
fn every_arm_emits_valid_rows_deterministically() {
    for (head, task, k) in heads() {
        let (c, d) = (cell(task, k), data(head));
        for arm in ARMS {
            let mat = generate(arm, &c, &d, 0, 4).unwrap().unwrap();
            assert_eq!((mat.rows, mat.x_cols), (1600, 4), "{arm} {task}");
            assert_eq!(
                generate(arm, &c, &d, 0, 4).unwrap().unwrap(),
                mat,
                "{arm} {task}"
            );
            let other = generate(arm, &c, &d, 1, 4).unwrap().unwrap();
            assert_ne!(
                other, mat,
                "{arm} {task}: sample seed must change the draws"
            );
            for r in 0..mat.rows {
                let row = mat.row(r).unwrap();
                assert!(row[..4].iter().all(|v| v.is_finite()), "{arm} {task}");
                let y = row[4];
                match head {
                    Head::Regression => assert!(y.is_finite()),
                    _ => assert!(
                        y.fract() == 0.0 && y >= 0.0 && (y as u32) < k,
                        "{arm} {task} {y}"
                    ),
                }
            }
            if matches!(arm, EXACT_RANK12 | EXACT_FULL) {
                // exact marginals: every value is one of A's values in that column
                for r in 0..mat.rows {
                    let row = mat.row(r).unwrap();
                    for (j, v) in row[..4].iter().enumerate() {
                        assert!((0..400).any(|i| d.a.row(i)[j] == *v), "{arm}: col {j}");
                    }
                }
            }
            if matches!(arm, REAL_X_PACKET_Y | REAL_X_TREE_Y) {
                for r in 0..mat.rows {
                    assert_eq!(
                        &mat.row(r).unwrap()[..4],
                        d.a.row(r % 400),
                        "{arm}: real X rows"
                    );
                }
            }
        }
    }
}

/// Share of rows whose sampled binary label agrees with the planted sign of 1.5 x0 - x1.
fn agreement(mat: &Mat) -> f64 {
    let hits = (0..mat.rows)
        .filter(|&r| {
            let row = mat.row(r).unwrap();
            (1.5 * row[0] - row[1] > 0.0) == (row[4] > 0.5)
        })
        .count();
    hits as f64 / mat.rows as f64
}

#[test]
fn oracles_carry_the_planted_conditional() {
    let (c, d) = (cell("binary", 2), data(Head::Binary));
    for arm in ARMS {
        let got = agreement(&generate(arm, &c, &d, 0, 4).unwrap().unwrap());
        // A Gaussian copula sees a binary target through attenuated (point-biserial) scores.
        let bar = if matches!(arm, EXACT_RANK12 | EXACT_FULL) {
            0.65
        } else {
            0.75
        };
        assert!(got > bar, "{arm}: label agreement {got:.3}");
    }
    // The tree ceiling on real X is at least as sharp as the packet's linear decoder there.
    let tree = agreement(&generate(REAL_X_TREE_Y, &c, &d, 0, 4).unwrap().unwrap());
    assert!(tree > 0.85, "{tree:.3}");
}
