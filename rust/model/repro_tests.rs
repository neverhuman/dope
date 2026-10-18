//! P02 observations of the historical sampler, before R03 changes semantics.
use super::*;
use crate::codec::{LoadedKernel, decode_kernel, encode_kernel};
use crate::sample::{SampleOptions, sample_kernel};
use serde_json::{Value, json};

pub(crate) fn fixture(dependence: Dependence, width: usize) -> Kernel {
    Kernel {
        task: Task::Regression,
        rows_fitted: 256,
        features: width as u32,
        seed: 1729,
        seed_policy: 0,
        quantization_bits: 16,
        compliant: false,
        schema: vec![
            ColumnSchema {
                kind: SchemaKind::Continuous,
                missing_probability: 0.0,
                impute: 0.5,
                transform: Transform::Identity,
            };
            width
        ],
        marginals: vec![
            Marginal::QuantileSpline {
                values: vec![0.0, 0.25, 0.5, 0.75, 1.0]
            };
            width
        ],
        program: KernelProgram::Symbolic {
            dependence,
            target: Target::SparseLinear {
                intercept: 0.5,
                terms: vec![],
            },
            noise: Noise::Homoscedastic { sigma: 0.0 },
        },
    }
}

pub(crate) fn draws(kernel: Kernel, rows: usize) -> Vec<f32> {
    sample_kernel(
        &LoadedKernel::V3(Box::new(kernel)),
        SampleOptions {
            rows,
            seed: Some(1729),
        },
    )
    .expect("bounded synthetic fixture samples")
}

pub(crate) fn mse(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / left.len() as f64
}

pub(crate) fn vine_observation() -> Value {
    // Gaussian C-vine: rho01=a, rho02=b, rho12|0=c. Its exact Cholesky
    // factor is a conditional-vine reference, independent of the legacy loop.
    let (a, b, c) = (0.6_f32, 0.5_f32, 0.4_f32);
    let edges = vec![
        VineEdge {
            left: 0,
            right: 1,
            conditioning_depth: 0,
            parameter: a,
        },
        VineEdge {
            left: 0,
            right: 2,
            conditioning_depth: 0,
            parameter: b,
        },
        VineEdge {
            left: 1,
            right: 2,
            conditioning_depth: 1,
            parameter: c,
        },
    ];
    let conditioned = fixture(
        Dependence::Vine {
            edges: edges.clone(),
        },
        3,
    );
    let mut shallow_edges = edges;
    shallow_edges[2].conditioning_depth = 0;
    let shallow = fixture(
        Dependence::Vine {
            edges: shallow_edges,
        },
        3,
    );
    let encoded = encode_kernel(&conditioned).expect("vine encodes");
    assert!(encoded != encode_kernel(&shallow).expect("shallow vine encodes"));
    let decoded = decode_kernel(&encoded).expect("vine decodes");
    assert!(
        decoded.program == conditioned.program,
        "depth survives decoding"
    );
    let conditioned_samples = draws(conditioned, 256);
    let shallow_samples = draws(shallow, 256);
    assert!(
        conditioned_samples == shallow_samples,
        "legacy ignores stored depth"
    );
    let reference = fixture(
        Dependence::GaussianCopula {
            cholesky: vec![
                1.0,
                0.0,
                0.0,
                a,
                (1.0 - a * a).sqrt(),
                0.0,
                b,
                c * (1.0 - b * b).sqrt(),
                ((1.0 - b * b) * (1.0 - c * c)).sqrt(),
            ],
        },
        3,
    );
    let reference_mse = mse(&conditioned_samples, &draws(reference, 256));
    assert!(
        reference_mse > 1e-4,
        "legacy differs from conditional Gaussian vine"
    );
    json!({
        "id": "P02-REPRO-VINE", "status": "reproduced_before_repair",
        "rows": 256, "width": 3, "seed": 1729,
        "depth_survives_wire": true, "depth_changes_wire": true,
        "depth_only_sample_mse": 0.0, "conditional_reference_sample_mse": reference_mse,
        "reference": "Gaussian C-vine analytic Cholesky; rho12=a*b+c*sqrt((1-a*a)*(1-b*b))",
        "scope": "stored depth alone is ignored; no general vine correctness claim"
    })
}

#[test]
fn p02_vine_conditioning_depth_semantics() {
    vine_observation();
}

#[test]
fn p02_vine_depth_bound_errors_are_source_free() {
    let mut kernel = fixture(
        Dependence::Vine {
            edges: vec![VineEdge {
                left: 0,
                right: 1,
                conditioning_depth: 9,
                parameter: 0.4,
            }],
        },
        2,
    );
    let first = kernel.validate().expect_err("invalid depth refuses");
    kernel.schema[0].impute = 0.123_456;
    let second = kernel.validate().expect_err("invalid depth still refuses");
    assert!(
        first == second,
        "diagnostic is independent of source-like values"
    );
    assert!(
        first == "invalid bounded dependence operator",
        "diagnostic stays typed"
    );
}
