mod codec {
    pub use dope_kernel::codec::{
        HEADER_BYTES, LoadedKernel, MAGIC, V2_MAGIC, decode_kernel, encode_kernel,
    };
}

mod sample {
    pub use dope_kernel::sample::{SampleOptions, sample_kernel};
}

#[path = "../rust/bundle.rs"]
mod bundle;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use bundle::{BundleLimits, content_fingerprint, preflight_dpk, verify_bundle};
use dope_kernel::model::{
    ColumnSchema, CopulaEdge, Dependence, JointDecoder, JointGenerator, JointNetwork, Kernel,
    KernelProgram, LinearTerm, Marginal, NeuralArchitecture, NeuralProfile, Noise, QuantizedLinear,
    RankNormalization, SchemaKind, Target, Task, Transform,
};

fn scratch(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/vnext/pr07-tests")
        .join(name);
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn native_manifest(bundle_id: &str, name: &str, bytes: &[u8], kind: &str) -> Vec<u8> {
    let digest = content_fingerprint(bytes);
    format!(
        r#"{{"amortization_count":1,"bundle_id":"{bundle_id}","capabilities":["byte-inventory","native-dpk-sample"],"components":[{{"bytes":{},"consumed_by_sampler":true,"depends_on":[],"executable":false,"handler":"none","id":"payload","kind":"{kind}","path":"{name}","required":true,"role":"sampling_payload","sha256":"{digest}"}}],"format":"dope.generator-bundle/v1","sampling_handler":"native-inline-dpk","transport_framing":"unmeasured","version":1}}"#,
        bytes.len()
    )
    .into_bytes()
}

fn stage_native(name: &str, filename: &str, bytes: &[u8], kind: &str) -> (PathBuf, String) {
    let root = scratch(name);
    fs::write(root.join(filename), bytes).unwrap();
    let manifest = native_manifest(name, filename, bytes, kind);
    fs::write(root.join("manifest.json"), &manifest).unwrap();
    let digest = content_fingerprint(&manifest);
    (root, digest)
}

fn symbolic_kernel() -> Kernel {
    Kernel {
        task: Task::Regression,
        rows_fitted: 32,
        features: 2,
        seed: 9,
        seed_policy: 0,
        quantization_bits: 8,
        compliant: false,
        schema: vec![
            ColumnSchema {
                kind: SchemaKind::Continuous,
                missing_probability: 0.0,
                impute: 0.5,
                transform: Transform::Identity,
            },
            ColumnSchema {
                kind: SchemaKind::Binary,
                missing_probability: 0.0,
                impute: 0.0,
                transform: Transform::Identity,
            },
        ],
        marginals: vec![
            Marginal::QuantileSpline {
                values: vec![0.0, 0.2, 0.5, 0.8, 1.0],
            },
            Marginal::Bernoulli { probability: 0.25 },
        ],
        program: KernelProgram::Symbolic {
            dependence: Dependence::ChowLiu {
                root: 0,
                edges: vec![CopulaEdge {
                    parent: 0,
                    child: 1,
                    correlation: 0.2,
                }],
            },
            target: Target::SparseLinear {
                intercept: 0.1,
                terms: vec![LinearTerm {
                    feature: 0,
                    coefficient: 0.5,
                }],
            },
            noise: Noise::Homoscedastic { sigma: 0.05 },
        },
    }
}

fn zero_linear(input: usize, output: usize, biases: Vec<f32>) -> QuantizedLinear {
    QuantizedLinear {
        input_dim: input as u32,
        output_dim: output as u32,
        weights: vec![0; input * output],
        scales: vec![1.0; output],
        biases,
    }
}

fn dpk33_kernel() -> Kernel {
    let tokens = 2;
    let latent = 4;
    let hidden = 16;
    let mut value_biases = Vec::new();
    for _ in 0..tokens {
        value_biases.extend([0.0, -2.0]);
    }
    Kernel {
        task: Task::Binary,
        rows_fitted: 16,
        features: 1,
        seed: 7,
        seed_policy: 0,
        quantization_bits: 8,
        compliant: false,
        schema: vec![ColumnSchema {
            kind: SchemaKind::Binary,
            missing_probability: 0.0,
            impute: 0.0,
            transform: Transform::Identity,
        }],
        marginals: vec![Marginal::Bernoulli { probability: 0.5 }],
        program: KernelProgram::NeuralJoint(JointGenerator {
            architecture: NeuralArchitecture::Tvae,
            profile: NeuralProfile::MicroTvae4,
            feature_permutation: vec![0],
            target_marginal: Marginal::Bernoulli { probability: 0.4 },
            normalization: vec![
                RankNormalization {
                    location: 0.0,
                    scale: 1.0,
                    missing_probability: 0.0,
                    randomized_discrete: true,
                };
                tokens
            ],
            network: JointNetwork::Tvae {
                decoder: Box::new(JointDecoder {
                    hidden_1: zero_linear(latent, hidden, vec![0.0; hidden]),
                    hidden_2: zero_linear(hidden, hidden, vec![0.0; hidden]),
                    value_head: zero_linear(hidden, tokens * 2, value_biases),
                    missing_head: zero_linear(hidden, tokens, vec![-20.0; tokens]),
                    variance_floor: 0.01,
                }),
            },
            training_hash: "a".repeat(64),
            implementation_hash: "b".repeat(64),
        }),
    }
}

fn direct_sample(bytes: &[u8], rows: usize) -> Vec<f32> {
    let kernel = codec::decode_kernel(bytes).unwrap();
    let loaded = if bytes.starts_with(codec::MAGIC) {
        codec::LoadedKernel::V3(Box::new(kernel))
    } else {
        codec::LoadedKernel::V2(Box::new(kernel))
    };
    sample::sample_kernel(
        &loaded,
        sample::SampleOptions {
            rows,
            seed: Some(11),
        },
    )
    .unwrap()
}

fn assert_round_trip(name: &str, bytes: &[u8], magic: &[u8]) {
    assert!(bytes.starts_with(magic));
    let expected = direct_sample(bytes, 4);
    let (root, digest) = stage_native(name, "payload.dpk", bytes, "dpk3");
    let verified = verify_bundle(&root, &digest, &BundleLimits::default()).unwrap();
    assert_eq!(verified.retained_payload(), Some(bytes));
    let sampled = verified
        .sample(4, Some(11), &BundleLimits::default())
        .unwrap();
    assert_eq!(sampled, expected);
    fs::write(root.join("payload.dpk"), vec![0u8; 64]).unwrap();
    assert_eq!(
        verified
            .sample(4, Some(11), &BundleLimits::default())
            .unwrap(),
        expected
    );
    assert_eq!(fs::read(root.join("payload.dpk")).unwrap(), vec![0u8; 64]);
}

#[test]
fn dpk32_and_dpk33_sample_from_retained_bytes() {
    let symbolic = codec::encode_kernel(&symbolic_kernel()).unwrap();
    assert_eq!(&symbolic[..6], b"DPK3\x03\x02");
    assert_round_trip("dpk32", &symbolic, codec::MAGIC);
    let neural = codec::encode_kernel(&dpk33_kernel()).unwrap();
    assert_eq!(&neural[..6], b"DPK3\x03\x03");
    assert_round_trip("dpk33", &neural, codec::MAGIC);
}

#[test]
fn adversarial_headers_fail_before_a_large_allocation() {
    let mut bomb = vec![0u8; codec::HEADER_BYTES];
    bomb[..4].copy_from_slice(codec::MAGIC);
    bomb[4] = 3;
    bomb[5] = 2;
    bomb[6] = 1;
    bomb[8] = 8;
    bomb[12] = 1;
    let payload = [0x81u8, 0x01, 0x80, 0x80, 0x80, 0x80, 0x08, 0x00];
    bomb[32..40].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    bomb.extend_from_slice(&payload);
    assert!(bomb.len() < 128);
    assert_eq!(
        preflight_dpk(&bomb, &BundleLimits::default())
            .unwrap_err()
            .code(),
        "expansion_cap_exceeded"
    );
    let (root, digest) = stage_native("bomb", "payload.dpk", &bomb, "dpk3");
    assert_eq!(
        verify_bundle(&root, &digest, &BundleLimits::default())
            .unwrap_err()
            .code(),
        "expansion_cap_exceeded"
    );

    let mut flags = codec::encode_kernel(&symbolic_kernel()).unwrap();
    flags[10] = 2;
    let (root, digest) = stage_native("flags", "payload.dpk", &flags, "dpk3");
    assert_eq!(
        verify_bundle(&root, &digest, &BundleLimits::default())
            .unwrap_err()
            .code(),
        "invalid_flags"
    );

    let mut zero = codec::encode_kernel(&symbolic_kernel()).unwrap();
    zero[12..16].fill(0);
    let (root, digest) = stage_native("zero-features", "payload.dpk", &zero, "dpk3");
    assert_eq!(
        verify_bundle(&root, &digest, &BundleLimits::default())
            .unwrap_err()
            .code(),
        "invalid_dimensions"
    );

    let encoded = codec::encode_kernel(&symbolic_kernel()).unwrap();
    let (root, digest) = stage_native(
        "truncated",
        "payload.dpk",
        &encoded[..encoded.len() - 1],
        "dpk3",
    );
    let code = verify_bundle(&root, &digest, &BundleLimits::default())
        .unwrap_err()
        .code();
    assert!(
        code == "truncated_artifact"
            || code == "dependency_identity_mismatch"
            || code == "expansion_cap_exceeded"
    );
    let limits = BundleLimits {
        max_sample_rows: 4,
        ..BundleLimits::default()
    };
    let (root, digest) = stage_native("rows", "payload.dpk", &encoded, "dpk3");
    let verified = verify_bundle(&root, &digest, &limits).unwrap();
    assert_eq!(
        verified.sample(0, Some(1), &limits).unwrap_err().code(),
        "invalid_dimensions"
    );
    assert_eq!(
        verified.sample(5, Some(1), &limits).unwrap_err().code(),
        "sample_cap_exceeded"
    );
}

#[test]
fn symlink_and_case_collision_are_typed_and_path_free() {
    let bytes = codec::encode_kernel(&symbolic_kernel()).unwrap();
    let root = scratch("symlink");
    let outside = scratch("symlink-outside");
    fs::write(outside.join("zz-secret-path.dpk"), &bytes).unwrap();
    std::os::unix::fs::symlink(outside.join("zz-secret-path.dpk"), root.join("payload.dpk"))
        .unwrap();
    let manifest = native_manifest("symlink", "payload.dpk", &bytes, "dpk3");
    fs::write(root.join("manifest.json"), &manifest).unwrap();
    let error = verify_bundle(
        &root,
        &content_fingerprint(&manifest),
        &BundleLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.code(), "symlink_component");
    assert!(!error.to_string().contains("zz-secret-path"));

    let root = scratch("case");
    fs::write(root.join("A.bin"), b"ab").unwrap();
    fs::write(root.join("a.bin"), b"ab").unwrap();
    let digest = content_fingerprint(b"ab");
    let manifest = format!(
        r#"{{"amortization_count":1,"bundle_id":"case-collision","capabilities":["byte-inventory"],"components":[{{"bytes":2,"consumed_by_sampler":false,"depends_on":[],"executable":false,"handler":"none","id":"lower","kind":"opaque","path":"a.bin","required":true,"role":"dictionary","sha256":"{digest}"}},{{"bytes":2,"consumed_by_sampler":false,"depends_on":["lower"],"executable":false,"handler":"none","id":"upper","kind":"opaque","path":"A.bin","required":true,"role":"sampling_payload","sha256":"{digest}"}}],"format":"dope.generator-bundle/v1","sampling_handler":"inventory-only","transport_framing":"unmeasured","version":1}}"#
    );
    fs::write(root.join("manifest.json"), &manifest).unwrap();
    let error = verify_bundle(
        &root,
        &content_fingerprint(manifest.as_bytes()),
        &BundleLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.code(), "case_collision");
    assert!(!error.to_string().contains("A.bin"));
}

fn namespace_available() -> bool {
    Command::new("sudo")
        .args([
            "-n",
            "unshare",
            "--net",
            "--mount",
            "--propagation",
            "private",
            "--",
            "true",
        ])
        .status()
        .is_ok_and(|status| status.success())
}

fn interfaces(netdev: &str) -> Vec<String> {
    netdev
        .lines()
        .filter_map(|line| {
            let name = line.split(':').next()?.trim();
            if name.is_empty() || name.contains('|') {
                None
            } else {
                Some(name.to_string())
            }
        })
        .collect()
}

#[test]
fn native_bundle_worker() {
    if std::env::var_os("DOPE_BUNDLE_WORKER").is_none() {
        return;
    }
    let root = PathBuf::from(std::env::var("DOPE_BUNDLE_ROOT").unwrap());
    let source = PathBuf::from(std::env::var("DOPE_BUNDLE_SOURCE").unwrap());
    let out = PathBuf::from(std::env::var("DOPE_BUNDLE_OUT").unwrap());
    let digest = std::env::var("DOPE_BUNDLE_SHA").unwrap();
    let mounted = Command::new("mount")
        .args(["-t", "tmpfs", "-o", "nosuid,nodev,noexec", "tmpfs"])
        .arg(&source)
        .status()
        .is_ok_and(|status| status.success());
    if !mounted {
        panic!("mount_failed");
    }
    let hidden = fs::read_dir(&source)
        .map(|entries| entries.count() == 0)
        .unwrap_or(false);
    if !hidden {
        panic!("source_visible");
    }
    let net = fs::read_to_string("/proc/net/dev").unwrap_or_default();
    let blocked = std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([1, 1, 1, 1], 443)),
        std::time::Duration::from_secs(2),
    )
    .is_err();
    let verified = verify_bundle(&root, &digest, &BundleLimits::default())
        .unwrap_or_else(|_| panic!("verify_failed"));
    let sample = verified
        .sample(4, Some(11), &BundleLimits::default())
        .unwrap_or_else(|_| panic!("sample_failed"));
    let fingerprint = content_fingerprint(
        &sample
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let body = format!(
        "{{\"source_hidden\":{hidden},\"network_blocked\":{blocked},\"sample_sha256\":\"{fingerprint}\",\"sample_len\":{}}}",
        sample.len()
    );
    fs::write(&out, body).unwrap();
    fs::write(out.with_extension("netdev"), net).unwrap();
}

#[test]
fn source_free_sampling_uses_a_network_and_mount_namespace_when_available() {
    let receipt_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/vnext/receipts");
    fs::create_dir_all(&receipt_dir).unwrap();
    let receipt = receipt_dir.join("PR07-isolation.json");
    if !namespace_available() {
        fs::write(
            &receipt,
            b"{\"gate\":\"unmeasured\",\"mechanism\":null,\"reason\":\"network_mount_namespace_unavailable\"}\n",
        )
        .unwrap();
        return;
    }
    let bytes = codec::encode_kernel(&symbolic_kernel()).unwrap();
    let root = scratch("isolated-bundle");
    let source = scratch("isolated-source");
    fs::write(source.join("rows.csv"), b"synthetic-source-row\n").unwrap();
    fs::write(root.join("payload.dpk"), &bytes).unwrap();
    let manifest = native_manifest("isolated", "payload.dpk", &bytes, "dpk3");
    fs::write(root.join("manifest.json"), &manifest).unwrap();
    let digest = content_fingerprint(&manifest);
    let expected = {
        let verified = verify_bundle(&root, &digest, &BundleLimits::default()).unwrap();
        let sample = verified
            .sample(4, Some(11), &BundleLimits::default())
            .unwrap();
        content_fingerprint(
            &sample
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    };
    let out = root.join("worker-result.json");
    let exe = std::env::current_exe().unwrap();
    let status = Command::new("sudo")
        .arg("-n")
        .arg("unshare")
        .args(["--net", "--mount", "--propagation", "private", "--", "env"])
        .arg("DOPE_BUNDLE_WORKER=1")
        .arg(format!("DOPE_BUNDLE_ROOT={}", root.display()))
        .arg(format!("DOPE_BUNDLE_SOURCE={}", source.display()))
        .arg(format!("DOPE_BUNDLE_OUT={}", out.display()))
        .arg(format!("DOPE_BUNDLE_SHA={digest}"))
        .arg(&exe)
        .args(["--exact", "native_bundle_worker", "--nocapture"])
        .status()
        .unwrap();
    assert!(status.success(), "worker_failed");
    let body = fs::read_to_string(&out).unwrap();
    let net = fs::read_to_string(out.with_extension("netdev")).unwrap();
    assert!(body.contains("\"source_hidden\":true"));
    assert!(body.contains("\"network_blocked\":true"));
    assert!(body.contains(&format!("\"sample_sha256\":\"{expected}\"")));
    assert!(!body.contains("synthetic-source"));
    assert!(!net.contains("synthetic-source"));
    assert_eq!(interfaces(&net), vec!["lo".to_string()]);
    assert_eq!(
        fs::read(source.join("rows.csv")).unwrap(),
        b"synthetic-source-row\n"
    );
    let record = format!(
        "{{\"gate\":\"passed\",\"mechanism\":\"sudo -n unshare --net --mount --propagation private\",\"interfaces\":[\"lo\"],\"sample_sha256\":\"{expected}\"}}\n"
    );
    fs::write(&receipt, record).unwrap();
}
