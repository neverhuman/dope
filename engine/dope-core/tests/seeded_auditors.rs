//! Versioned CPU auditor controls must replay without a device RNG or process-global seed.
use dope_core::eval::{gbdt, nn, Matrix};
use dope_core::target::Head;

fn data() -> Matrix {
    Matrix {
        rows: 120,
        cols: 3,
        data: (0..120)
            .flat_map(|r| {
                let x = r as f64 / 20.0 - 3.0;
                [x, x.sin(), (x * 2.0).cos()]
            })
            .collect(),
    }
}

#[test]
fn seeded_neural_replays_after_other_seeds_and_across_thread_pools() {
    let x = data();
    for head in [Head::Regression, Head::Binary, Head::Multiclass(3)] {
        let y: Vec<_> = (0..x.rows)
            .map(|r| match head {
                Head::Regression => x.at(r, 0).powi(2) + x.at(r, 1),
                Head::Binary => f64::from(x.at(r, 0) > 0.0),
                Head::Multiclass(_) => (r % 3) as f64,
            })
            .collect();
        let mut p = nn::Params::mlp(123);
        p.hidden = vec![8];
        p.batch = 32;
        p.max_epochs = 5;
        p.patience = 3;
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| nn::fit_predict_seeded_cpu(&x, &y, head, &x, &p).unwrap())
        };
        let first = run(1);
        let mut other_seed = p.clone();
        other_seed.seed = 456;
        let other = nn::fit_predict_seeded_cpu(&x, &y, head, &x, &other_seed).unwrap();
        assert_ne!(first, other);
        assert_eq!(first, run(4));
        assert!(first.iter().all(|v| v.is_finite()));
        assert_eq!(first.len(), x.rows * head.outputs());
        if let Head::Multiclass(k) = head {
            for row in first.chunks(k) {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn seeded_neural_refuses_malformed_bounds_and_labels_without_panicking() {
    let x = data();
    let y = vec![0.0; x.rows];
    let mut p = nn::Params::mlp(0);
    p.batch = 0;
    assert!(nn::fit_predict_seeded_cpu(&x, &y, Head::Binary, &x, &p).is_err());
    p.batch = 32;
    let mut invalid = x.clone();
    invalid.data[0] = f64::NAN;
    assert!(nn::fit_predict_seeded_cpu(&invalid, &y, Head::Binary, &x, &p).is_err());
    let mut labels = y.clone();
    labels[0] = 2.0;
    assert!(nn::fit_predict_seeded_cpu(&x, &labels, Head::Binary, &x, &p).is_err());
    labels[0] = 0.5;
    assert!(nn::fit_predict_seeded_cpu(&x, &labels, Head::Multiclass(3), &x, &p).is_err());
    invalid = x.clone();
    invalid.data.pop();
    assert!(nn::fit_predict_seeded_cpu(&invalid, &y, Head::Regression, &x, &p).is_err());
}

#[test]
fn tree_reference_replays_with_parallel_feature_ties() {
    let mut x = data();
    for r in 0..x.rows {
        x.data[r * x.cols + 1] = x.at(r, 0);
    }
    let y: Vec<_> = (0..x.rows).map(|r| f64::from(x.at(r, 0) > 0.0)).collect();
    let p = gbdt::Params {
        rounds: 12,
        max_depth: 2,
        min_leaf: 4,
        bins: 16,
        seed: 321,
        ..Default::default()
    };
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| gbdt::fit_predict(&x, &y, Head::Binary, &x, &p))
    };
    assert_eq!(run(1), run(4));
}
