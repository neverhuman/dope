use super::table;
use dope_kernel::compiler::CompileOptions;
use dope_kernel::model::{KernelProgram, Noise, ResidualProfile, ResidualV1, Target};
use dope_kernel::typed_data::workflow::{
    compile_raw, compile_raw_residual, inspect_raw, sample_raw,
};

#[test]
fn residual_cli_explicit_profile_runs_in_fresh_source_free_worker() {
    use std::{fs, process::Command};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/raw-residual-cli")
        .join(std::process::id().to_string());
    let empty = root.join("empty");
    fs::create_dir_all(&empty).unwrap();
    let schema = root.join("schema.json");
    let csv = root.join("fit.csv");
    let artifact = root.join("model.raw");
    fs::write(
        &schema,
        serde_json::to_vec(&serde_json::json!({
            "format":"dope-table-schema","version":1,"columns":[
                {"id":"c0001","source_name":"private_x","role":"feature","dtype":"continuous"},
                {"id":"c0002","source_name":"private_y","role":"target","dtype":"continuous"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let mut rows = String::from("private_x,private_y\n");
    for row in 0..64 {
        rows.push_str(&format!(
            "{},{}\n",
            10.0 + row as f64 / 63.0 * 20.0,
            100.0 + row as f64 / 63.0 * 200.0
        ));
    }
    fs::write(&csv, rows).unwrap();
    let binary = env!("CARGO_BIN_EXE_dope-kernel");
    let refused = Command::new(binary)
        .arg("compile-raw")
        .arg("--csv")
        .arg(&csv)
        .arg("--schema")
        .arg(&schema)
        .arg("--out")
        .arg(&artifact)
        .args(["--residual-profile", "no_noise_v1"])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(!artifact.exists());
    let output = Command::new(binary)
        .arg("compile-raw")
        .arg("--csv")
        .arg(&csv)
        .arg("--schema")
        .arg(&schema)
        .arg("--out")
        .arg(&artifact)
        .args([
            "--candidate",
            "development_symbolic",
            "--residual-profile",
            "no_noise_v1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let sampled = sample_raw(&fs::read(&artifact).unwrap(), 64, Some(11)).unwrap();
    let encoded = sampled
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let worker = Command::new(binary)
        .env_clear()
        .current_dir(&empty)
        .arg("sample-raw-isolated")
        .arg("--artifact")
        .arg(&artifact)
        .args(["--rows", "64", "--seed", "11"])
        .output()
        .unwrap();
    assert!(
        worker.status.success(),
        "{}",
        String::from_utf8_lossy(&worker.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&worker.stdout).unwrap();
    assert_eq!(receipt["inspection"]["profile"], "raw-numeric-residual-v1");
    assert_eq!(
        receipt["sample_sha256"],
        dope_kernel::bundle::content_fingerprint(&encoded)
    );
    let public = String::from_utf8(worker.stdout).unwrap();
    assert!(!public.contains("private_x") && !public.contains("private_y"));
}

fn residual_options() -> CompileOptions {
    CompileOptions {
        seed: Some(1729),
        beam_width: Some(1),
        quantization_profiles: vec![8],
        ..CompileOptions::default()
    }
}

fn replace_residual_payload(artifact: &[u8], noise: ResidualV1) -> Vec<u8> {
    let state_len = u64::from_le_bytes(artifact[8..16].try_into().unwrap()) as usize;
    let state = &artifact[56..56 + state_len];
    let mut kernel = dope_kernel::codec::decode_kernel(&artifact[56 + state_len..]).unwrap();
    let KernelProgram::Symbolic {
        target,
        noise: retained,
        ..
    } = &mut kernel.program
    else {
        panic!("symbolic residual profile")
    };
    *target = Target::SparseLinear {
        intercept: 0.5,
        terms: vec![],
    };
    *retained = Noise::ResidualV1 {
        distribution: noise,
    };
    let payload = dope_kernel::codec::encode_kernel(&kernel).unwrap();
    let mut content = state.to_vec();
    content.extend_from_slice(&payload);
    let mut output = b"DOPERAW2".to_vec();
    output.extend_from_slice(&(state.len() as u64).to_le_bytes());
    output.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    output.extend_from_slice(blake3::hash(&content).as_bytes());
    output.extend_from_slice(&content);
    output
}

#[test]
fn residual_profiles_reload_charge_actual_bytes_and_keep_v1_explicit() {
    let fit = (0..64).collect::<Vec<_>>();
    let data = table(1.0, 1.0, 99999.0);
    for profile in [
        ResidualProfile::NoNoise,
        ResidualProfile::StudentT4,
        ResidualProfile::GaussianMixture2,
        ResidualProfile::MonotoneQuantile17,
    ] {
        let result =
            compile_raw_residual(&data, &fit, &residual_options(), 32768, profile).unwrap();
        assert_eq!(&result.artifact[..8], b"DOPERAW2");
        assert_eq!(result.inspection.profile, "raw-numeric-residual-v1");
        let accounting = &result.inspection.accounting;
        assert_eq!(
            accounting.framing_bytes
                + accounting.transform_schema_bytes
                + accounting.native_payload_bytes,
            result.artifact.len()
        );
        assert_eq!(accounting.complete_sampling_bytes, result.artifact.len());
        assert!(result.report.selected_candidate.ends_with(profile.as_str()));
        assert!(!result.report.compliant);
        let state_len = u64::from_le_bytes(result.artifact[8..16].try_into().unwrap()) as usize;
        let payload = &result.artifact[56 + state_len..];
        assert_eq!(&payload[4..6], &[3, 3]);
        let kernel = dope_kernel::codec::decode_kernel(payload).unwrap();
        assert!(matches!(
            kernel.program,
            KernelProgram::Symbolic {
                noise: Noise::ResidualV1 { .. },
                ..
            }
        ));
        let samples = sample_raw(&result.artifact, 256, Some(11)).unwrap();
        assert_eq!(
            samples,
            sample_raw(&result.artifact, 256, Some(11)).unwrap()
        );
        assert!(samples.iter().all(|v| v.is_finite()));
        let exact = compile_raw_residual(
            &data,
            &fit,
            &residual_options(),
            result.artifact.len(),
            profile,
        )
        .unwrap();
        assert_eq!(exact.artifact, result.artifact);
        assert!(
            compile_raw_residual(
                &data,
                &fit,
                &residual_options(),
                result.artifact.len() - 1,
                profile
            )
            .is_err()
        );
        let mut mislabeled = result.artifact;
        mislabeled[..8].copy_from_slice(b"DOPERAW1");
        assert!(
            inspect_raw(&mislabeled)
                .unwrap_err()
                .to_string()
                .contains("raw_residual_profile_mismatch")
        );
        let public = serde_json::to_string(&result.inspection).unwrap();
        for private in [
            "secret_x",
            "secret_y",
            "\"location\":",
            "\"scale\":",
            "\"values\":",
        ] {
            assert!(!public.contains(private));
        }
    }
    let mut explicit = residual_options();
    explicit.development_residual = Some(ResidualProfile::NoNoise);
    assert!(
        compile_raw(&data, &fit, &explicit, 32768)
            .err()
            .unwrap()
            .to_string()
            .contains("raw_v1_refuses_development_residual")
    );
}

#[test]
fn constant_residual_successor_stays_constant_and_refuses_old_wire_label() {
    let data = table(1.0, 0.0, 9000.0);
    let fit = (0..64).collect::<Vec<_>>();
    for profile in [
        ResidualProfile::NoNoise,
        ResidualProfile::StudentT4,
        ResidualProfile::GaussianMixture2,
        ResidualProfile::MonotoneQuantile17,
    ] {
        let compiled =
            compile_raw_residual(&data, &fit, &residual_options(), 32768, profile).unwrap();
        assert!(
            sample_raw(&compiled.artifact, 128, Some(11))
                .unwrap()
                .chunks_exact(2)
                .all(|row| row[1] == 0.0)
        );
        let state_len = u64::from_le_bytes(compiled.artifact[8..16].try_into().unwrap()) as usize;
        let mut payload = compiled.artifact[56 + state_len..].to_vec();
        payload[5] = 2;
        assert!(
            dope_kernel::codec::decode_kernel(&payload)
                .unwrap_err()
                .to_string()
                .contains("residual v1 requires native DPK3.3 regression")
        );
    }
    assert!(
        ResidualV1::MonotoneQuantile17 {
            values: vec![0.0; 16]
        }
        .validate()
        .is_err()
    );
    assert!(
        ResidualV1::MonotoneQuantile17 {
            values: (0..17).rev().map(|v| v as f32).collect()
        }
        .validate()
        .is_err()
    );
    assert!(
        ResidualV1::StudentT4 {
            location: f32::NAN,
            scale: -1.0
        }
        .validate()
        .is_err()
    );
}

#[test]
fn residual_sampler_consumes_actual_components_and_unclipped_student_tail() {
    let base = compile_raw_residual(
        &table(1.0, 1.0, 999.0),
        &(0..64).collect::<Vec<_>>(),
        &residual_options(),
        32768,
        ResidualProfile::NoNoise,
    )
    .unwrap();
    let none = replace_residual_payload(&base.artifact, ResidualV1::NoNoise);
    assert!(
        sample_raw(&none, 1024, Some(17))
            .unwrap()
            .chunks_exact(2)
            .all(|row| row[1] == 200.0)
    );
    let mixture = replace_residual_payload(
        &base.artifact,
        ResidualV1::GaussianMixture2 {
            weight: 0.25,
            locations: [-0.25, 0.25],
            scales: [0.0, 0.0],
        },
    );
    let samples = sample_raw(&mixture, 1024, Some(17)).unwrap();
    assert!(
        samples
            .chunks_exact(2)
            .all(|row| row[1] == 150.0 || row[1] == 250.0)
    );
    let lower = samples
        .chunks_exact(2)
        .filter(|row| row[1] == 150.0)
        .count();
    assert!(
        (200..=320).contains(&lower),
        "deterministic component routing {lower}"
    );
    let student = replace_residual_payload(
        &base.artifact,
        ResidualV1::StudentT4 {
            location: 0.0,
            scale: 0.5,
        },
    );
    let samples = sample_raw(&student, 1024, Some(17)).unwrap();
    assert!(samples.chunks_exact(2).any(|row| row[1] < 100.0));
    assert!(samples.chunks_exact(2).any(|row| row[1] > 300.0));
    let quantile = replace_residual_payload(
        &base.artifact,
        ResidualV1::MonotoneQuantile17 {
            values: (0..17).map(|at| -0.25 + at as f32 / 32.0).collect(),
        },
    );
    let samples = sample_raw(&quantile, 1024, Some(17)).unwrap();
    assert!(
        samples
            .chunks_exact(2)
            .all(|row| (150.0..=250.0).contains(&row[1]))
    );
    assert!(
        samples
            .chunks_exact(2)
            .any(|row| row[1] != 150.0 && row[1] != 250.0)
    );
}

#[test]
fn residual_fits_are_outer_fit_only_affine_and_small_sample_unavailable() {
    let fit = (0..64).collect::<Vec<_>>();
    for profile in [
        ResidualProfile::NoNoise,
        ResidualProfile::StudentT4,
        ResidualProfile::GaussianMixture2,
        ResidualProfile::MonotoneQuantile17,
    ] {
        let a = compile_raw_residual(
            &table(1.0, 1.0, 99999.0),
            &fit,
            &residual_options(),
            32768,
            profile,
        )
        .unwrap();
        let b = compile_raw_residual(
            &table(1.0, 1.0, -99999.0),
            &fit,
            &residual_options(),
            32768,
            profile,
        )
        .unwrap();
        assert_eq!(a.artifact, b.artifact);
        let scaled = compile_raw_residual(
            &table(100.0, 0.001, 99999.0),
            &fit,
            &residual_options(),
            32768,
            profile,
        )
        .unwrap();
        assert_eq!(
            a.report.selected_candidate,
            scaled.report.selected_candidate
        );
        let sa = sample_raw(&a.artifact, 64, Some(9)).unwrap();
        let sb = sample_raw(&scaled.artifact, 64, Some(9)).unwrap();
        for (a, b) in sa.chunks_exact(2).zip(sb.chunks_exact(2)) {
            assert!((a[0] - b[0] / 100.0).abs() < 1e-4);
            assert!((a[1] - b[1] * 1000.0).abs() < 1e-3);
        }
    }
    let error = compile_raw_residual(
        &table(1.0, 1.0, 999.0),
        &(0..12).collect::<Vec<_>>(),
        &residual_options(),
        32768,
        ResidualProfile::StudentT4,
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("development_residual_oof_unavailable")
    );
}
