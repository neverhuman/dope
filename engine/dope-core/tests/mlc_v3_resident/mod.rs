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

#[test]
fn tensor_512_features_are_bit_identical_to_629_prefix_and_reject_bad_inputs() {
    assert_eq!(dope_core::mlc_v3::MLC_V3_FEATURES_512_DIM, 512);
    let raw = build_deterministic_safetensors_bytes();
    let model = MlcV3Model::from_safetensors_bytes_on_device(&raw, &Device::Cpu).unwrap();
    let bits_1d = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    let bits_2d = |v: &[Vec<f32>], w: usize| -> Vec<Vec<u32>> {
        v.iter().map(|r| bits_1d(&r[..w])).collect()
    };

    for batch in [1, 4, 16] {
        let (x, columns) = inputs(batch, &Device::Cpu);
        for level in ["low", "small", "medium", "high", "ultra"] {
            let full_routed = model.router.forward_tensor(&x, level).unwrap();
            let (task_lat, dist_lat) = model.router.forward_latents_tensor(&x, level).unwrap();
            assert_eq!(
                bits_2d(&task_lat.to_vec2::<f32>().unwrap(), 128),
                bits_2d(&full_routed.task_latent.to_vec2::<f32>().unwrap(), 128)
            );
            assert_eq!(
                bits_2d(&dist_lat.to_vec2::<f32>().unwrap(), 128),
                bits_2d(&full_routed.dist_latent.to_vec2::<f32>().unwrap(), 128)
            );
        }

        let got_629 = model
            .encode_features_629_tensor(&x, &columns, "high")
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let got_512 = model
            .encode_features_512_tensor(&x, &columns, "high")
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let got_desc_512 = model
            .encode_descriptor_512_tensor(&x, "high")
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(got_512.len(), batch);
        assert_eq!(got_desc_512.len(), batch);
        assert!(got_512.iter().all(|r| r.len() == 512));
        assert!(got_desc_512.iter().all(|r| r.len() == 512));
        assert_eq!(bits_2d(&got_512, 512), bits_2d(&got_629, 512));
        assert_eq!(bits_2d(&got_desc_512, 512), bits_2d(&got_629, 512));
    }

    let (x1, col1) = inputs(1, &Device::Cpu);
    let x1_vec = x1.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
    let col1_vec = col1.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
    let mut col1_arr = [[0.0f32; 17]; 64];
    for (dst, src) in col1_arr.iter_mut().zip(&col1_vec) {
        dst.copy_from_slice(src);
    }
    let cond_629 = model
        .encode_conditioning_629(&x1_vec, &col1_arr, "high")
        .unwrap();
    let cond_512 = model
        .encode_conditioning_512(&x1_vec, &col1_arr, "high")
        .unwrap();
    assert_eq!(cond_512.len(), 512);
    assert_eq!(bits_1d(&cond_512), bits_1d(&cond_629[..512]));

    for level in ["small", "ultra", "private-value"] {
        let err = model
            .encode_features_512_tensor(&x1, &col1, level)
            .unwrap_err();
        assert!(!err.to_string().contains(level));
        let err_desc = model.encode_descriptor_512_tensor(&x1, level).unwrap_err();
        assert!(!err_desc.to_string().contains(level));
    }
    let (_, col4) = inputs(4, &Device::Cpu);
    for bad_res in [
        model.encode_features_512_tensor(&x1, &col4, "high"),
        model.encode_features_512_tensor(&x1.to_dtype(DType::F64).unwrap(), &col1, "high"),
        model.encode_features_512_tensor(&x1, &col1.narrow(1, 0, 63).unwrap(), "high"),
        model.encode_features_512_tensor(&x1.narrow(1, 0, 255).unwrap(), &col1, "high"),
        model.encode_features_512_tensor(&x1, &col1.to_dtype(DType::F64).unwrap(), "high"),
    ] {
        assert!(bad_res.is_err());
    }
    assert!(model
        .encode_conditioning_512(&x1_vec[..255], &col1_arr, "high")
        .is_err());
    assert!(model
        .encode_conditioning_512(&x1_vec, &col1_arr[..63], "high")
        .is_err());

    let mut values = vec![0.0f32; 256];
    values[0] = f32::NAN;
    values[1] = f32::INFINITY;
    values[2] = f32::NEG_INFINITY;
    let bad = Tensor::from_vec(values, (1, 256), &Device::Cpu).unwrap();
    let cleaned_629 = model
        .encode_features_629_tensor(&bad, &col1, "high")
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let cleaned_512 = model
        .encode_features_512_tensor(&bad, &col1, "high")
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert!(cleaned_512[0].iter().all(|v| v.is_finite()));
    assert!(cleaned_512[0][..3].iter().all(|v| v.to_bits() == 0));
    assert_eq!(bits_2d(&cleaned_512, 512), bits_2d(&cleaned_629, 512));
}

#[test]
fn prepared_table_conditioning_modes_and_lossless_k36_match_629_prefix_0_ulp() {
    use dope_core::landmark::{
        LandmarkBundle, LandmarkGroup, LandmarkMode, LandmarkOptions, PROGRESSIVE_ORDER,
    };
    use dope_core::prepare::{
        build_conditioning, prepare_table, prepare_table_cells_with_policy,
        progressive_landmarks_for_table, ClassVocabPolicy, DopeConditioning, NativeReportView,
        PrepTask,
    };

    let raw = build_deterministic_safetensors_bytes();
    let model = MlcV3Model::from_safetensors_bytes_on_device(&raw, &Device::Cpu).unwrap();
    let bits = |s: &[f32]| -> Vec<u32> { s.iter().map(|v| v.to_bits()).collect() };
    let assert_cond_0_ulp = |cond: &DopeConditioning| {
        let mut col_arr = [[0.0f32; 17]; 64];
        for (dst, src) in col_arr.iter_mut().zip(&cond.col_landmarks) {
            dst.copy_from_slice(src);
        }
        let host_629 = model
            .encode_conditioning_629(&cond.vec256, &col_arr, "high")
            .unwrap();
        let host_512 = model
            .encode_conditioning_512(&cond.vec256, &col_arr, "high")
            .unwrap();
        let desc_t = Tensor::from_slice(&cond.vec256, (1, 256), &Device::Cpu).unwrap();
        let col_t = Tensor::from_vec(
            col_arr.iter().flatten().copied().collect::<Vec<_>>(),
            (1, 64, 17),
            &Device::Cpu,
        )
        .unwrap();
        let res_629 = model
            .encode_features_629_tensor(&desc_t, &col_t, "high")
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let res_512 = model
            .encode_features_512_tensor(&desc_t, &col_t, "high")
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let desc_512 = model
            .encode_descriptor_512_tensor(&desc_t, "high")
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let expected = bits(&host_629[..512]);
        assert_eq!(bits(&host_512), expected);
        assert_eq!(bits(&res_629[0][..512]), expected);
        assert_eq!(bits(&res_512[0]), expected);
        assert_eq!(bits(&desc_512[0]), expected);
    };

    let x_reg: Vec<Vec<f64>> = (0..32)
        .map(|i| vec![(i as f64) * 0.25 - 3.5, ((i * 5) % 11) as f64])
        .collect();
    let y_reg: Vec<f64> = (0..32).map(|i| (i as f64) * 0.5).collect();
    let tbl_reg = prepare_table(&x_reg, &y_reg, PrepTask::Regression).unwrap();
    let zero = LandmarkBundle::zero();
    for mode in [LandmarkMode::None, LandmarkMode::Fast] {
        assert_cond_0_ulp(&build_conditioning(&tbl_reg, &zero, None, mode).unwrap());
    }
    let mut prog =
        progressive_landmarks_for_table(&tbl_reg, &LandmarkOptions::default(), None).unwrap();
    let mut last_bun = zero.clone();
    for grp in PROGRESSIVE_ORDER {
        let (bun, _) = prog.next_group(grp).unwrap();
        assert_cond_0_ulp(
            &build_conditioning(&tbl_reg, &bun, None, LandmarkMode::Progressive).unwrap(),
        );
        last_bun = bun;
    }
    let full_view = NativeReportView::from_signature(prog.report());
    assert_cond_0_ulp(
        &build_conditioning(&tbl_reg, &last_bun, Some(&full_view), LandmarkMode::Full).unwrap(),
    );

    let x_mc: Vec<Vec<String>> = (0..72)
        .map(|i| {
            vec![
                format!("{:.4}", (i as f64) * 0.15),
                format!("cat_{}", i % 9),
            ]
        })
        .collect();
    let y_mc: Vec<String> = (0..72)
        .map(|i| format!("{:.6}", (i / 2) as f64 * 0.25))
        .collect();
    let ids_mc: Vec<u64> = (0..72u64).map(|i| 100 + i).collect();
    let tbl_k36 = prepare_table_cells_with_policy(
        &x_mc,
        &y_mc,
        PrepTask::Multiclass,
        ids_mc,
        ClassVocabPolicy::LosslessV1,
    )
    .unwrap();
    assert_eq!(tbl_k36.classes, 36);
    for mode in [LandmarkMode::None, LandmarkMode::Fast] {
        assert_cond_0_ulp(&build_conditioning(&tbl_k36, &zero, None, mode).unwrap());
    }
    let mut prog_k36 =
        progressive_landmarks_for_table(&tbl_k36, &LandmarkOptions::default(), None).unwrap();
    let (bun_k36, _) = prog_k36.next_group(LandmarkGroup::Macro).unwrap();
    assert_cond_0_ulp(
        &build_conditioning(&tbl_k36, &bun_k36, None, LandmarkMode::Progressive).unwrap(),
    );
}

#[test]
#[ignore = "microbenchmark for 512D vs 629D resident CPU latency"]
fn bench_512_vs_629_resident_cpu() {
    let raw = build_deterministic_safetensors_bytes();
    let model = MlcV3Model::from_safetensors_bytes_on_device(&raw, &Device::Cpu).unwrap();
    let iters = 16usize;
    for batch in [1usize, 4, 16] {
        let (x, col) = inputs(batch, &Device::Cpu);
        for _ in 0..4 {
            let _ = model.encode_features_629_tensor(&x, &col, "high").unwrap();
            let _ = model.encode_features_512_tensor(&x, &col, "high").unwrap();
            let _ = model.encode_descriptor_512_tensor(&x, "high").unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = model.encode_features_629_tensor(&x, &col, "high").unwrap();
        }
        let us_629 = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = model.encode_features_512_tensor(&x, &col, "high").unwrap();
        }
        let us_512 = t1.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let t2 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = model.encode_descriptor_512_tensor(&x, "high").unwrap();
        }
        let us_desc_512 = t2.elapsed().as_secs_f64() * 1e6 / (iters as f64);
        println!(
            "B={batch}: 629D={us_629:.2}us, 512D={us_512:.2}us ({:.2}x), desc_512D={us_desc_512:.2}us ({:.2}x)",
            us_629 / us_512,
            us_629 / us_desc_512
        );
    }
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
    let (x, col) = inputs(1, &device);
    for bad in [
        model.encode_features_629_tensor(&cpu_x, &cpu_col, "high"),
        model.encode_features_512_tensor(&cpu_x, &cpu_col, "high"),
        model.encode_descriptor_512_tensor(&cpu_x, "high"),
        model.encode_features_629_tensor(&x, &cpu_col, "high"),
        model.encode_features_512_tensor(&x, &cpu_col, "high"),
    ] {
        assert!(bad.is_err());
    }
    let output = model.encode_features_629_tensor(&x, &col, "high").unwrap();
    let out_512 = model.encode_features_512_tensor(&x, &col, "high").unwrap();
    let desc_512 = model.encode_descriptor_512_tensor(&x, "high").unwrap();
    assert!(output.device().is_cuda() && out_512.device().is_cuda() && desc_512.device().is_cuda());
    let narrowed = output.narrow(1, 0, 512).unwrap();
    assert_eq!(narrowed.dims(), [1, 512]);
    assert_eq!(out_512.dims(), [1, 512]);
    assert_eq!(desc_512.dims(), [1, 512]);
    let to_cpu = |t: &Tensor| t.to_device(&Device::Cpu).unwrap().to_vec2::<f32>().unwrap();
    let (nar_v, out512_v, desc512_v) = (to_cpu(&narrowed), to_cpu(&out_512), to_cpu(&desc_512));
    for j in 0..512 {
        assert_eq!(out512_v[0][j].to_bits(), nar_v[0][j].to_bits());
        assert_eq!(desc512_v[0][j].to_bits(), nar_v[0][j].to_bits());
    }

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
        let got_512 = released
            .encode_features_512_tensor(&features, &columns, "high")
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for b in 0..batch {
            for j in 0..512 {
                assert_eq!(got_512[b][j].to_bits(), got[b][j].to_bits());
            }
        }
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
            released
                .encode_features_512_tensor(&features, &columns, "high")
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
        let us_629 = start.elapsed().as_secs_f64() * 1e6 / 32.0;
        let start_512 = std::time::Instant::now();
        for _ in 0..32 {
            released
                .encode_features_512_tensor(&features, &columns, "high")
                .unwrap();
        }
        device.synchronize().unwrap();
        let us_512 = start_512.elapsed().as_secs_f64() * 1e6 / 32.0;
        println!("CUDA released whole checkpoint B={batch}, tensor_count=56, parameters=444168, max_abs_diff={difference}, synchronized_resident_mean_us={us_629}, 512d_mean_us={us_512}");
    }
}
