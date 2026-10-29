#[path = "../examples/resident_release_check/limits.rs"]
mod limits;

use limits::{canonical_device, summarize, StageTimes, RUNS};

fn sample(prepare_ms: f64, conditioning_ms: f64, forward_ms: f64) -> StageTimes {
    StageTimes {
        prepare_ms,
        conditioning_ms,
        forward_ms,
    }
}

#[test]
fn original_closed_bounds_include_the_single_read_and_discard_one_warmup() {
    let mut runs = [sample(20.0, 2.0, 3.0); RUNS];
    runs[0] = sample(10_000.0, 200.0, 300.0);
    let summary = summarize(5.0, &runs, 1_024, 8).unwrap();
    assert!(summary.passed);
    assert_eq!(summary.measured_runs, 5);
    assert_eq!(summary.warmup_runs_discarded, 1);
    assert_eq!(summary.overhead_max_ms, 5.0);
    assert_eq!(summary.none_p50_ms, 30.0);
    assert_eq!(summary.none_p95_ms, 30.0);
}

#[test]
fn one_slow_warm_forward_fails_even_when_median_overhead_is_small() {
    let mut runs = [sample(1.0, 0.5, 0.5); RUNS];
    runs[4].forward_ms = 4.500_001;
    let summary = summarize(1.0, &runs, 109, 8).unwrap();
    assert!(!summary.passed);
    assert!(summary.overhead_max_ms > 5.0);
    assert!(summary.none_p50_ms < 30.0);
}

#[test]
fn read_cost_can_make_an_otherwise_passing_fixture_fail() {
    let runs = [sample(20.0, 2.0, 3.0); RUNS];
    assert!(summarize(5.0, &runs, 100, 8).unwrap().passed);
    let summary = summarize(5.000_001, &runs, 100, 8).unwrap();
    assert!(!summary.passed);
    assert!(summary.none_p50_ms > 30.0);
}

#[test]
fn median_is_of_each_complete_trial_not_sum_of_stage_medians() {
    let runs = [
        sample(1.0, 0.0, 0.0),
        sample(30.0, 1.0, 0.0),
        sample(30.0, 1.0, 0.0),
        sample(30.0, 0.0, 1.0),
        sample(26.0, 0.0, 0.0),
        sample(26.0, 0.0, 0.0),
    ];
    let summary = summarize(0.0, &runs, 40, 8).unwrap();
    assert_eq!(summary.none_p50_ms, 31.0);
    assert!(!summary.passed);
}

#[test]
fn p95_reports_tail_without_turning_the_p50_bound_into_a_maximum() {
    let runs = [
        sample(999.0, 0.0, 0.0),
        sample(1.0, 0.0, 0.0),
        sample(2.0, 0.0, 0.0),
        sample(4.0, 0.0, 0.0),
        sample(8.0, 0.0, 0.0),
        sample(160.0, 0.0, 0.0),
    ];
    let summary = summarize(0.0, &runs, 1, 8).unwrap();
    assert!(summary.passed);
    assert_eq!(summary.none_p50_ms, 4.0);
    assert!((summary.none_p95_ms - 129.6).abs() < 1e-10);
}

#[test]
fn missing_extra_and_inapplicable_observations_never_pass() {
    let runs = [sample(1.0, 1.0, 1.0); RUNS];
    for count in 0..RUNS {
        assert!(summarize(1.0, &runs[..count], 40, 8).is_err());
    }
    let extra = [sample(1.0, 1.0, 1.0); RUNS + 1];
    assert!(summarize(1.0, &extra, 40, 8).is_err());
    for rows in [0, 1_025, usize::MAX] {
        assert!(summarize(1.0, &runs, rows, 8).is_err());
    }
    for threads in [0, 1, 2, 7, 9, 128] {
        assert!(summarize(1.0, &runs, 40, threads).is_err());
    }
}

#[test]
fn nonfinite_negative_and_overflowed_stages_refuse_even_in_warmup() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.000_001] {
        let mut runs = [sample(1.0, 1.0, 1.0); RUNS];
        assert!(summarize(invalid, &runs, 100, 8).is_err());
        for index in [0, RUNS - 1] {
            for field in 0..3 {
                runs = [sample(1.0, 1.0, 1.0); RUNS];
                match field {
                    0 => runs[index].prepare_ms = invalid,
                    1 => runs[index].conditioning_ms = invalid,
                    _ => runs[index].forward_ms = invalid,
                }
                assert!(summarize(1.0, &runs, 100, 8).is_err());
            }
        }
    }
    assert!(summarize(f64::MAX, &[sample(f64::MAX, 0.0, 0.0); RUNS], 100, 8).is_err());
    assert!(summarize(0.0, &[sample(0.0, f64::MAX, f64::MAX); RUNS], 100, 8).is_err());
}

#[test]
fn zero_total_trials_refuse_while_individual_zero_components_remain_valid() {
    let mut runs = [sample(0.0, 0.0, 0.0); RUNS];
    assert!(summarize(0.0, &runs, 40, 8).is_err());
    assert!(summarize(1.0, &runs, 40, 8).unwrap().passed);
    runs[0] = sample(0.0, 0.0, 0.0);
    for run in &mut runs[1..] {
        run.prepare_ms = 1.0;
    }
    assert!(summarize(0.0, &runs, 40, 8).is_err());
}

#[test]
fn device_selection_rejects_automatic_fallback_and_unsupported_aliases() {
    assert_eq!(canonical_device("cpu"), Ok("cpu".into()));
    assert_eq!(canonical_device("cuda:0"), Ok("cuda:0".into()));
    assert_eq!(canonical_device("cuda:002"), Ok("cuda:2".into()));
    for name in [
        "",
        "auto",
        "gpu",
        "metal",
        "cpu:0",
        "cuda",
        "cuda:",
        "cuda:-1",
        "cuda:+1",
        "cuda:1:2",
        " cuda:0",
        "cuda:0 ",
        "cuda:999999999999999999999999999999999999",
    ] {
        assert!(canonical_device(name).is_err());
    }
}
