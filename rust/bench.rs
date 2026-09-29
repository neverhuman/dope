use std::time::Instant;

use dope_kernel::codec::{LoadedKernel, decode_kernel};
use dope_kernel::compiler::{CompileOptions, compile_kernel_from_arrays};
use dope_kernel::model::{
    ColumnSchema, CopulaEdge, Dependence, Kernel, KernelProgram, LinearTerm, Marginal, Noise,
    SchemaKind, Target, Task,
};
use dope_kernel::sample::{SampleOptions, sample_kernel};

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random(index: usize) -> f32 {
    (splitmix64(index as u64) >> 40) as f32 / (1u32 << 24) as f32
}

fn compile_benchmark() {
    let rows = 2_000;
    let columns = 256;
    let features: Vec<f32> = (0..rows * columns).map(random).collect();
    let target: Vec<f32> = (0..rows)
        .map(|row| {
            (0..4)
                .map(|column| features[row * columns + column])
                .sum::<f32>()
                * 0.25
        })
        .collect();
    let started = Instant::now();
    let result = compile_kernel_from_arrays(
        &features,
        &target,
        rows,
        columns,
        Task::Regression,
        &CompileOptions {
            seed: Some(7),
            ..Default::default()
        },
    )
    .unwrap();
    println!(
        "compile_2000x256_seconds={:.6} artifact_bytes={} candidates={}",
        started.elapsed().as_secs_f64(),
        result.artifact.len(),
        result.report.candidate_count
    );
    assert_eq!(
        dope_kernel::encode_kernel(&decode_kernel(&result.artifact).unwrap()).unwrap(),
        result.artifact
    );
}

fn sampling_kernel(features: usize, dependent: bool) -> LoadedKernel {
    let values: Vec<f32> = (0..17).map(|index| index as f32 / 16.0).collect();
    LoadedKernel::V2(Box::new(Kernel {
        task: Task::Regression,
        rows_fitted: 2_000,
        features: features as u32,
        seed: 7,
        seed_policy: 0,
        quantization_bits: 8,
        compliant: false,
        schema: vec![
            ColumnSchema {
                kind: SchemaKind::Continuous,
                missing_probability: 0.0,
                impute: 0.5,
                transform: dope_kernel::model::Transform::Identity,
            };
            features
        ],
        marginals: vec![Marginal::QuantileSpline { values }; features],
        program: KernelProgram::Symbolic {
            dependence: if dependent {
                Dependence::ChowLiu {
                    root: 0,
                    edges: (1..features)
                        .map(|child| CopulaEdge {
                            parent: (child - 1) as u32,
                            child: child as u32,
                            correlation: 0.5,
                        })
                        .collect(),
                }
            } else {
                Dependence::Independent
            },
            target: Target::SparseLinear {
                intercept: 0.2,
                terms: vec![LinearTerm {
                    feature: 0,
                    coefficient: 0.6,
                }],
            },
            noise: Noise::Homoscedastic { sigma: 0.05 },
        },
    }))
}

fn sample_benchmark(rows: usize, features: usize, dependent: bool) {
    let kernel = sampling_kernel(features, dependent);
    let started = Instant::now();
    let sampled = sample_kernel(
        &kernel,
        SampleOptions {
            rows,
            seed: Some(9),
        },
    )
    .unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let cells = rows * (features + 1);
    println!(
        "sample_{rows}x{features}_dependent={dependent}_seconds={seconds:.6} cells_per_second={:.0} checksum={:.6}",
        cells as f64 / seconds,
        sampled
            .iter()
            .step_by((cells / 100).max(1))
            .map(|value| f64::from(*value))
            .sum::<f64>()
    );
}

fn main() {
    compile_benchmark();
    sample_benchmark(100_000, 100, false);
    sample_benchmark(10_000, 1_000, false);
    sample_benchmark(100_000, 100, true);
    sample_benchmark(10_000, 1_000, true);
}
