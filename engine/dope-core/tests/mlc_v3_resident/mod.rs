use super::{build_deterministic_safetensors_bytes, max_diff_f32, MlcV3Model};
use candle_core::{DType, Device, Tensor};

fn inputs(batch: usize, device: &Device) -> (Tensor, Tensor) {
    let features: Vec<f32> = (0..batch * 256)
        .map(|i| ((i % 151) as f32 - 75.0) * 0.021)
        .collect();
    let columns: Vec<f32> = (0..batch * 64 * 17)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    (
        Tensor::from_vec(features, (batch, 256), device).unwrap(),
        Tensor::from_vec(columns, (batch, 64, 17), device).unwrap(),
    )
}

fn assert_matches_host(model: &MlcV3Model, features: &Tensor, columns: &Tensor) -> f32 {
    let got = model
        .encode_features_629_tensor(features, columns, "high")
        .unwrap();
    assert_eq!(got.dtype(), DType::F32);
    assert!(got.device().same_device(model.device()));
    let host =
        MlcV3Model::from_safetensors_bytes(&build_deterministic_safetensors_bytes()).unwrap();
    let x = features
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let col = columns
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let got = got
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let expected = host.encode_features_629_batch(&x, &col, "high").unwrap();
    let difference = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| max_diff_f32(a, b))
        .fold(0.0, f32::max);
    assert!(
        difference < 1e-5,
        "resident/host max difference {difference}"
    );
    assert!(got.iter().flatten().all(|v| v.is_finite()));
    difference
}

#[test]
fn tensor_features_match_existing_host_api_and_reject_bad_inputs() {
    let raw = build_deterministic_safetensors_bytes();
    let model = MlcV3Model::from_safetensors_bytes_on_device(&raw, &Device::Cpu).unwrap();
    assert!(model.device().is_cpu());
    assert_eq!(model.inventory.tensor_count, 56);
    assert_eq!(model.inventory.parameter_count, 444_168);
    for batch in [1, 4] {
        let (x, columns) = inputs(batch, &Device::Cpu);
        assert_matches_host(&model, &x, &columns);
    }
    let (x, col) = inputs(1, &Device::Cpu);
    for level in ["small", "ultra", "private-value"] {
        let error = model
            .encode_features_629_tensor(&x, &col, level)
            .unwrap_err();
        assert!(!error.to_string().contains(level));
    }
    assert!(model
        .encode_features_629_tensor(&x.to_dtype(DType::F64).unwrap(), &col, "high")
        .is_err());
    assert!(model
        .encode_features_629_tensor(&x, &col.narrow(1, 0, 63).unwrap(), "high")
        .is_err());
    assert!(model
        .encode_features_629_tensor(&x.narrow(1, 0, 255).unwrap(), &col, "high")
        .is_err());
    assert!(model
        .encode_features_629_tensor(&x, &col.to_dtype(DType::F64).unwrap(), "high")
        .is_err());
    let mut values = vec![0.0f32; 256];
    values[0] = f32::NAN;
    values[1] = f32::INFINITY;
    values[2] = f32::NEG_INFINITY;
    let bad = Tensor::from_vec(values, (1, 256), &Device::Cpu).unwrap();
    let cleaned = model
        .encode_features_629_tensor(&bad, &col, "high")
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert!(cleaned[0].iter().all(|v| v.is_finite()));
    assert!(cleaned[0][..3].iter().all(|v| v.to_bits() == 0));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU; run explicitly to qualify CUDA inference"]
fn cuda_resident_features_execute_and_match_cpu_and_released_reference() {
    let device = Device::new_cuda(0).expect("CUDA device required for explicit GPU qualification");
    let raw = build_deterministic_safetensors_bytes();
    let model = MlcV3Model::from_safetensors_bytes_on_device(&raw, &device).unwrap();
    assert!(model.device().is_cuda());
    assert_eq!(model.inventory.tensor_count, 56);
    for batch in [1, 4] {
        let (x, col) = inputs(batch, &device);
        let difference = assert_matches_host(&model, &x, &col);
        println!("CUDA deterministic whole model B={batch}, max_abs_diff={difference}");
    }
    let (cpu_x, cpu_col) = inputs(1, &Device::Cpu);
    assert!(model
        .encode_features_629_tensor(&cpu_x, &cpu_col, "high")
        .is_err());
    let (x, col) = inputs(1, &device);
    assert!(model
        .encode_features_629_tensor(&x, &cpu_col, "high")
        .is_err());
    let output = model.encode_features_629_tensor(&x, &col, "high").unwrap();
    assert!(output.device().is_cuda());
    let narrowed = output.narrow(1, 0, 512).unwrap();
    assert_eq!(narrowed.dims(), [1, 512]);
    assert!(narrowed.device().is_cuda());

    if let Ok(weights) = std::env::var("DOPE_ENCODER_WEIGHTS") {
        let released =
            MlcV3Model::load_safetensors_on_device(std::path::Path::new(&weights), &device)
                .unwrap();
        assert_eq!(released.inventory.sha256, super::EXPECTED_TRAINED_ST_SHA256);
        assert_eq!(released.inventory.tensor_count, 56);
        let fixture = std::env::var("DOPE_ENCODER_PARITY_FIXTURE")
            .expect("released GPU parity fixture required");
        #[derive(serde::Deserialize)]
        struct Reference {
            vec: Vec<Vec<f32>>,
            columns: Vec<Vec<Vec<f32>>>,
            expected: Vec<Vec<f32>>,
        }
        let reference: Reference =
            serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
        let batch = reference.vec.len();
        assert!(batch > 0);
        assert_eq!(reference.expected.len(), batch);
        assert!(reference
            .expected
            .iter()
            .all(|row| row.len() == 629 && row.iter().all(|v| v.is_finite())));
        let features = Tensor::from_vec(
            reference.vec.into_iter().flatten().collect::<Vec<_>>(),
            (batch, 256),
            &device,
        )
        .unwrap();
        let columns = Tensor::from_vec(
            reference
                .columns
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<_>>(),
            (batch, 64, 17),
            &device,
        )
        .unwrap();
        let got = released
            .encode_features_629_tensor(&features, &columns, "high")
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(got.iter().flatten().all(|v| v.is_finite()));
        let difference = got
            .iter()
            .zip(&reference.expected)
            .map(|(a, b)| max_diff_f32(a, b))
            .fold(0.0, f32::max);
        assert!(
            difference < 1e-5,
            "released CUDA/Python max difference {difference}"
        );
        for _ in 0..4 {
            released
                .encode_features_629_tensor(&features, &columns, "high")
                .unwrap();
        }
        device.synchronize().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..32 {
            released
                .encode_features_629_tensor(&features, &columns, "high")
                .unwrap();
        }
        device.synchronize().unwrap();
        println!("CUDA released whole checkpoint B={batch}, tensor_count=56, parameters=444168, max_abs_diff={difference}, synchronized_resident_mean_us={}",start.elapsed().as_secs_f64()*1e6/32.0);
    }
}
