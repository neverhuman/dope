//! Actual native compile/encode/decode coverage; synthetic engineering fixtures,
//! never independent-source transfer, equal-compute gain or tail qualification.

use dope_kernel::compiler::{CompileOptions, compile_kernel_from_arrays};
use dope_kernel::contract::{AnonymizationTier, ReleasePolicy};
use dope_kernel::model::{
    Dependence, KernelProgram, Noise, ResidualProfile, ResidualV1, Target, Task,
};

const ROWS: usize = 256;
const WIDTH: usize = 48;

fn unit(mut value: u64) -> f32 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((value ^ (value >> 31)) % 10000) as f32 / 9999.0
}

fn fixture(reverse: bool) -> (Vec<f32>, Vec<f32>, (u32, u32)) {
    let mut x = Vec::with_capacity(ROWS * WIDTH);
    let mut y = Vec::with_capacity(ROWS);
    for row in 0..ROWS {
        let rank0 = unit(row as u64 ^ 0x18af);
        let rank1 = unit(row as u64 ^ 0x792b);
        let mut values = (0..WIDTH)
            .map(|column| {
                if column < 24 {
                    // Two shared factors with weak idiosyncratic noise: dense,
                    // redundant nuisance structure rather than orthogonal dummies.
                    let mix = (column % 5) as f32 / 4.0;
                    0.9 * (mix * rank0 + (1.0 - mix) * rank1)
                        + 0.1 * unit(row as u64 ^ ((column as u64 + 1) << 24))
                } else {
                    unit(row as u64 ^ ((column as u64 + 1) << 24))
                }
            })
            .collect::<Vec<_>>();
        values[30] = (row % 2) as f32;
        values[31] = ((row / 2) % 2) as f32;
        values[40] = row as f32 / (ROWS - 1) as f32; // An ID-like distractor.
        values[41] = values[0];
        values[42] = values[1];
        values[46] = values[30];
        values[47] = values[31];
        if reverse {
            values.reverse();
        }
        x.extend(values);
        // Two-valued continuous regression remains regression, not binary.
        y.push(if (row % 2) == ((row / 2) % 2) {
            0.1
        } else {
            0.9
        });
    }
    // After reversal the duplicated parents occur first and become the exact
    // equivalence-class representatives, which must be reflected in the wire.
    (x, y, if reverse { (0, 1) } else { (30, 31) })
}

fn controls(cap: usize, adaptive: bool, residual: ResidualProfile) -> CompileOptions {
    CompileOptions {
        seed: Some(1729),
        beam_width: Some(52),
        quantization_profiles: vec![8],
        adaptive_allocation: adaptive,
        development_residual: Some(residual),
        deadline: Some(std::time::Duration::from_secs(60)),
        release_policy: ReleasePolicy::new(AnonymizationTier::L0, Some(cap), false).unwrap(),
        ..Default::default()
    }
}

#[test]
fn compiled_interaction_survives_realistic_distractors_feasible_rank_and_encoded_caps() {
    for (reverse, adaptive) in [(false, false), (false, true), (true, true)] {
        let (x, y, parents) = fixture(reverse);
        let cap = 4096;
        let result = compile_kernel_from_arrays(
            &x,
            &y,
            ROWS,
            WIDTH,
            Task::Regression,
            &controls(cap, adaptive, ResidualProfile::NoNoise),
        )
        .unwrap();
        assert_eq!(result.report.search_history.len(), 52);
        assert!(result.artifact.len() <= cap && !result.report.compliant);
        assert!(
            result
                .report
                .search_history
                .iter()
                .filter(|attempt| attempt.actual_artifact_bytes.is_some())
                .all(|attempt| attempt.elapsed_seconds.is_some())
        );
        let mut pure_pair = 0;
        let mut refined_pair = 0;
        let mut poet = 0;
        for (id, artifact) in &result.candidate_artifacts {
            assert!(artifact.len() <= cap, "over-cap candidate {id}");
            let kernel = dope_kernel::codec::decode_kernel(artifact).unwrap();
            assert_eq!(kernel.task, Task::Regression);
            let KernelProgram::Symbolic {
                dependence,
                target,
                noise,
            } = &kernel.program
            else {
                panic!("symbolic fixture");
            };
            assert!(matches!(
                noise,
                Noise::ResidualV1 {
                    distribution: ResidualV1::NoNoise
                }
            ));
            if let Dependence::Poet { rank, loadings, .. } = dependence {
                poet += 1;
                assert!((1..=4).contains(rank));
                assert!(loadings.iter().all(|loading| loading.factor < *rank && (loading.feature as usize) < WIDTH));
            }
            if let Target::Ga2m {
                logistic,
                interactions,
                ..
            } = target
            {
                assert!(!logistic);
                assert!(interactions.iter().all(|term| term.left != term.right));
                let pair = interactions
                    .iter()
                    .any(|term| (term.left, term.right) == parents);
                pure_pair += usize::from(pair);
                if pair && id.contains("-c1-") {
                    refined_pair += 1;
                    assert!(interactions.len() <= 2);
                }
                if !adaptive {
                    assert!(interactions.len() <= 4);
                }
            }
            let report = result
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == *id)
                .unwrap();
            assert_eq!(report.artifact_bytes, artifact.len());
            let trace = result
                .report
                .search_history
                .iter()
                .find(|attempt| attempt.candidate_id == *id)
                .unwrap();
            assert_eq!(trace.actual_artifact_bytes, Some(artifact.len()));
        }
        assert!(
            pure_pair > 0,
            "native encoded candidate bank omitted screened pair {parents:?}"
        );
        assert!(poet > 0, "feasible factor-rank candidate omitted");
        if adaptive {
            assert!(
                refined_pair > 0,
                "capacity-refitted GA2M omitted screened pair"
            );
        }
    }
}

#[test]
fn binary_valued_regression_keeps_explicit_residual_semantics_after_full_compilation() {
    let (x, y, _) = fixture(false);
    for profile in [
        ResidualProfile::StudentT4,
        ResidualProfile::GaussianMixture2,
        ResidualProfile::MonotoneQuantile17,
    ] {
        let mut options = controls(4096, false, profile);
        options.beam_width = Some(18);
        let result =
            compile_kernel_from_arrays(&x, &y, ROWS, WIDTH, Task::Regression, &options).unwrap();
        for (_, artifact) in result.candidate_artifacts {
            assert_eq!(&artifact[4..6], &[3, 3]);
            let kernel = dope_kernel::codec::decode_kernel(&artifact).unwrap();
            assert_eq!(kernel.task, Task::Regression);
            let KernelProgram::Symbolic { target, noise, .. } = kernel.program else {
                panic!("symbolic fixture");
            };
            assert!(!matches!(target, Target::SparseLogistic { .. }));
            let Noise::ResidualV1 { distribution } = noise else {
                panic!("explicit residual disappeared");
            };
            assert!(matches!(
                (profile, distribution),
                (ResidualProfile::StudentT4, ResidualV1::StudentT4 { .. })
                    | (
                        ResidualProfile::GaussianMixture2,
                        ResidualV1::GaussianMixture2 { .. }
                    )
                    | (
                        ResidualProfile::MonotoneQuantile17,
                        ResidualV1::MonotoneQuantile17 { .. }
                    )
            ));
        }
    }
}
