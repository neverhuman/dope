//! The `dope` binary end to end on a small mixed-type CSV.

use std::path::{Path, PathBuf};
use std::process::Command;

fn dope(dir: &Path, args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "dope {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn workdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dope-cli-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut csv = String::from("age,city,score\n");
    for i in 0..400u32 {
        let city = ["paris", "lima", "oslo"][(i % 3) as usize];
        let age = 20 + (i * 7919) % 50;
        let score =
            age as f64 * 0.5 + if city == "oslo" { 10.0 } else { 0.0 } + ((i * 31) % 7) as f64;
        csv += &format!("{age},{city},{score}\n");
    }
    std::fs::write(dir.join("train.csv"), csv).unwrap();
    dir
}

#[test]
fn fit_inspect_sample_predict_encode() {
    let dir = workdir("journey");
    let fit = dope(
        &dir,
        &[
            "fit",
            "train.csv",
            "--target",
            "score",
            "--budget",
            "2k",
            "-o",
            "m.dope",
            "--device",
            "cpu",
        ],
    );
    assert!(fit.contains("compiled train.csv -> m.dope"));
    let size = std::fs::metadata(dir.join("m.dope")).unwrap().len();
    assert!(size <= 2048, "artifact {size} bytes over budget");
    let inspect: serde_json::Value =
        serde_json::from_str(&dope(&dir, &["inspect", "m.dope", "--json"])).unwrap();
    assert_eq!(inspect["target"], "score");
    assert_eq!(inspect["bytes"]["total"].as_u64(), Some(size));
    let synth = dope(
        &dir,
        &[
            "sample", "m.dope", "-n", "20", "--seed", "3", "--device", "cpu",
        ],
    );
    let mut lines = synth.lines();
    assert_eq!(lines.next(), Some("age,city,score"));
    assert_eq!(lines.count(), 20);
    let ml = dope(
        &dir,
        &[
            "sample", "m.dope", "-n", "5", "--space", "ml", "--device", "cpu",
        ],
    );
    assert!(ml.lines().next().unwrap().contains("city=oslo"));
    let pred = dope(&dir, &["predict", "m.dope", "train.csv"]);
    assert_eq!(pred.lines().count(), 401);
    let enc = dope(&dir, &["encode", "train.csv", "--model", "m.dope"]);
    assert_eq!(enc.lines().next().unwrap(), ml.lines().next().unwrap());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn errors_name_the_problem_without_echoing_cells() {
    let dir = workdir("errors");
    let out = Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(&dir)
        .args(["fit", "train.csv", "--target", "missing_column"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("target column not found"));
    assert!(!err.contains("paris"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn seeded_cpu_auditors_replay_across_fresh_processes_and_learn_signal() {
    let dir = workdir("seeded-auditors");
    let mut query = String::from("age,city,score\n");
    for i in 400..500u32 {
        let city = ["paris", "lima", "oslo"][(i % 3) as usize];
        let age = 20 + (i * 7919) % 50;
        let score =
            age as f64 * 0.5 + if city == "oslo" { 10.0 } else { 0.0 } + ((i * 31) % 7) as f64;
        query += &format!("{age},{city},{score}\n");
    }
    std::fs::write(dir.join("query.csv"), query).unwrap();
    dope(
        &dir,
        &[
            "fit",
            "train.csv",
            "--target",
            "score",
            "--budget",
            "2KiB",
            "--no-audit",
            "--seed",
            "17",
            "--device",
            "cpu",
            "-o",
            "m.dope",
        ],
    );
    let run = |threads, report| {
        dope(
            &dir,
            &[
                "--threads",
                threads,
                "eval",
                "--model",
                "m.dope",
                "--train",
                "train.csv",
                "--test",
                "query.csv",
                "--auditors",
                "linear-seeded-v1,gbdt,mlp-seeded-v1",
                "--factor",
                "1",
                "--seed",
                "123",
                "--json",
                report,
            ],
        );
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(dir.join(report)).unwrap())
            .unwrap()
    };
    let first = run("1", "first.json");
    assert_eq!(first, run("4", "second.json"));
    let auditors = first["auditors"].as_array().unwrap();
    assert_eq!(auditors.len(), 3);
    for auditor in auditors {
        assert!(auditor["real_loss"].as_f64().unwrap() < auditor["null_loss"].as_f64().unwrap());
        assert_eq!(auditor["informative"], true);
    }
    assert_eq!(auditors[2]["auditor"], "mlp-seeded-v1");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn size_parameter_caps_and_tier_catalog() {
    let dir = workdir("tiers");
    dope(
        &dir,
        &[
            "fit",
            "train.csv",
            "-t",
            "score",
            "--size",
            "small",
            "--max-parameters",
            "200",
            "--no-audit",
            "--report",
            "fit.json",
            "--device",
            "cpu",
        ],
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("fit.json")).unwrap()).unwrap();
    assert_eq!(report["budget"], 2048);
    assert!(report["parameters"]["total"].as_u64().unwrap() <= 200);
    let inspect: serde_json::Value =
        serde_json::from_str(&dope(&dir, &["inspect", "model.dope", "--json"])).unwrap();
    assert_eq!(report["parameters"], inspect["parameters"]);
    dope(
        &dir,
        &[
            "fit-tiers",
            "train.csv",
            "-t",
            "score",
            "-o",
            "zoo",
            "--dataset-id",
            "test-fixture",
            "--source",
            "fixture",
            "--data-license",
            "MIT",
            "--device",
            "cpu",
        ],
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("zoo/catalog.json")).unwrap()).unwrap();
    assert_eq!(manifest["kind"], "per-dataset-pretrained-generators");
    assert_eq!(manifest["release_authorized"], false);
    assert_eq!(manifest["anonymized"], false);
    assert_eq!(manifest["target_index"], 2);
    for item in manifest["artifacts"].as_array().unwrap() {
        let file = dir.join("zoo").join(item["file"].as_str().unwrap());
        let bytes = std::fs::read(&file).unwrap();
        assert_eq!(bytes.len() as u64, item["bytes"].as_u64().unwrap());
        assert!(item["bytes"].as_u64().unwrap() <= item["budget_bytes"].as_u64().unwrap());
        use sha2::{Digest, Sha256};
        assert_eq!(item["sha256"], format!("{:x}", Sha256::digest(&bytes)));
        let p = dope_core::decode(&bytes).unwrap();
        assert_eq!(
            item["parameters"],
            serde_json::to_value(p.parameters()).unwrap()
        );
    }
    let out = Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(&dir)
        .args([
            "fit-tiers",
            "train.csv",
            "-t",
            "score",
            "-o",
            "zoo",
            "--dataset-id",
            "fixture",
            "--source",
            "fixture",
            "--data-license",
            "MIT",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("already exists"));
    dope(
        &dir,
        &[
            "fit-tiers",
            "train.csv",
            "-t",
            "score",
            "-o",
            "zoo-anon",
            "--dataset-id",
            "test-fixture",
            "--source",
            "fixture",
            "--data-license",
            "MIT",
            "--device",
            "cpu",
            "--anonymize",
        ],
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("zoo-anon/catalog.json")).unwrap()).unwrap();
    assert_eq!(manifest["anonymized"], true);
    let p = dope_core::decode(&std::fs::read(dir.join("zoo-anon/small.dope")).unwrap()).unwrap();
    assert_ne!(p.schema.target.name, "score");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn color_flags_preserve_plain_pipes_and_enable_ansi_when_requested() {
    let dir = workdir("color-flags");
    let plain = dope(&dir, &["doctor"]);
    assert!(!plain.contains("\x1b["));
    assert_eq!(plain, dope(&dir, &["--color", "never", "doctor"]));
    let colored = dope(&dir, &["--color", "always", "doctor"]);
    assert!(colored.contains("\x1b["));
    assert!(colored.contains("native Rust"));

    // NO_COLOR takes precedence over CLICOLOR_FORCE in --color auto mode.
    let no_color = Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(&dir)
        .env("NO_COLOR", "1")
        .env("CLICOLOR_FORCE", "1")
        .arg("doctor")
        .output()
        .unwrap();
    assert!(no_color.status.success());
    assert!(!String::from_utf8_lossy(&no_color.stdout).contains("\x1b["));

    let clicolor = Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(&dir)
        .env_remove("NO_COLOR")
        .env("CLICOLOR_FORCE", "1")
        .arg("doctor")
        .output()
        .unwrap();
    assert!(clicolor.status.success());
    assert!(String::from_utf8_lossy(&clicolor.stdout).contains("\x1b["));

    // Fit & inspect: colored vs plain/JSON, and ensure sample notices never corrupt CSV stdout.
    let fit_col = dope(
        &dir,
        &[
            "--color",
            "always",
            "fit",
            "train.csv",
            "-t",
            "score",
            "-b",
            "2048",
            "-o",
            "m.dope",
            "--device",
            "cpu",
        ],
    );
    assert!(fit_col.contains("\x1b[") && fit_col.contains("0 source rows stored"));
    let inspect_json = dope(&dir, &["--color", "always", "inspect", "m.dope", "--json"]);
    assert!(!inspect_json.contains("\x1b["));
    let parsed_inspect: serde_json::Value = serde_json::from_str(&inspect_json).unwrap();
    assert_eq!(parsed_inspect["target"], "score");

    let sample_stdout = dope(
        &dir,
        &[
            "--color", "always", "sample", "m.dope", "-n", "6", "--device", "cpu",
        ],
    );
    assert!(!sample_stdout.contains("\x1b["));
    assert_eq!(sample_stdout.lines().count(), 7);
    assert!(sample_stdout.starts_with("age,city,score\n"));

    let sample_file_col = dope(
        &dir,
        &[
            "--color", "always", "sample", "m.dope", "-n", "6", "--device", "cpu", "-o", "syn.csv",
        ],
    );
    assert!(sample_file_col.contains("\x1b[") && sample_file_col.contains("sampled"));
    let syn_disk = std::fs::read_to_string(dir.join("syn.csv")).unwrap();
    assert!(!syn_disk.contains("\x1b[") && syn_disk.lines().count() == 7);

    let sample_file_plain = dope(
        &dir,
        &[
            "sample", "m.dope", "-n", "6", "--device", "cpu", "-o", "syn2.csv",
        ],
    );
    assert!(sample_file_plain.is_empty());

    // Build a minimal valid DML3 wire packet and verify dynamic metadata in plain, --json, and --color always modes.
    let mut core = Vec::new();
    core.extend_from_slice(b"DML2");
    core.extend_from_slice(&[2u8, 1u8, 0u8, 0u8]); // ver=2, high, regression, 0 trees
    core.extend_from_slice(&120u32.to_le_bytes()); // n_train = 120
    core.extend_from_slice(&2u32.to_le_bytes()); // p_use = 2
    for &u in &[1u16, 1, 2, 0, 1, 0] {
        core.extend_from_slice(&u.to_le_bytes());
    }
    for _ in 0..7 {
        core.extend_from_slice(&0i32.to_le_bytes());
    }
    for _ in 0..4 {
        core.extend_from_slice(&0.0f32.to_le_bytes());
    }
    // f16 blocks: reg_w(1) + reg_m(2) + reg_s(2) + chol(2) + diag(2) + dist(128) + task(128) + knots(4) + bin_p(2) + bin_lh(4) = 275 f16s
    core.resize(core.len() + 275 * 2, 0u8);
    core.push(0u8); // is_binary_col packed (1 byte for p_use=2)
    core.extend_from_slice(&[0u8, 0u8]); // w_basis (1 f16)
    let mut payload = Vec::new();
    payload.extend_from_slice(&[2u8, 0u8, 1u8, 0u8]); // high, regression, inc_v3=1
    payload.extend_from_slice(&(core.len() as u32).to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&core);
    let crc = dope_core::mlc_v3::inflate::crc32_ieee(&payload);
    let mut dml3 = Vec::new();
    dml3.extend_from_slice(b"DML3");
    dml3.extend_from_slice(&[3u8, 0u8]);
    dml3.extend_from_slice(&crc.to_le_bytes());
    dml3.extend_from_slice(&payload);
    std::fs::write(dir.join("test.dml3"), &dml3).unwrap();

    let wire_plain = dope(&dir, &["mlc-v3", "inspect-wire", "--wire", "test.dml3"]);
    assert!(!wire_plain.contains("\x1b["));
    let wire_val: serde_json::Value = serde_json::from_str(&wire_plain).unwrap();
    assert_eq!(wire_val["wire_format"], "DML3");
    assert_eq!(wire_val["wire_bytes"], dml3.len());
    assert_eq!(wire_val["n_train"], 120);
    assert_eq!(wire_val["p_use"], 2);

    let wire_json_forced = dope(
        &dir,
        &[
            "--color",
            "always",
            "mlc-v3",
            "inspect-wire",
            "--wire",
            "test.dml3",
            "--json",
        ],
    );
    assert!(!wire_json_forced.contains("\x1b["));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&wire_json_forced).unwrap(),
        wire_val
    );

    let wire_col = dope(
        &dir,
        &[
            "--color",
            "always",
            "mlc-v3",
            "inspect-wire",
            "--wire",
            "test.dml3",
        ],
    );
    assert!(wire_col.contains("\x1b["));
    assert!(wire_col.contains(&format!("{}", dml3.len())));
    assert!(wire_col.contains("120 train rows × 2 features"));
    assert!(wire_col.contains("0 source rows stored"));

    // Build a valid 56-tensor SafeTensors fixture and verify inspect-weights & mlc-v3 sample across modes.
    let specs: &[(&str, &[usize])] = &[
        ("router.level_embed.weight", &[3, 256]),
        ("router.small_embed", &[256]),
        ("router.input_norm.weight", &[140]),
        ("router.input_norm.bias", &[140]),
        ("router.v3_inter_norm.weight", &[116]),
        ("router.v3_inter_norm.bias", &[116]),
        ("router.backbone.0.weight", &[256, 140]),
        ("router.backbone.0.bias", &[256]),
        ("router.backbone.2.weight", &[256]),
        ("router.backbone.2.bias", &[256]),
        ("router.backbone.3.weight", &[256, 256]),
        ("router.backbone.3.bias", &[256]),
        ("router.backbone.5.weight", &[256]),
        ("router.backbone.5.bias", &[256]),
        ("router.inter_attn.token_proj.weight", &[32, 32]),
        ("router.inter_attn.token_proj.bias", &[32]),
        ("router.inter_attn.qkv.weight", &[96, 32]),
        ("router.inter_attn.qkv.bias", &[96]),
        ("router.inter_attn.bilinear_u.weight", &[128, 256]),
        ("router.inter_attn.bilinear_u.bias", &[128]),
        ("router.inter_attn.bilinear_v.weight", &[128, 256]),
        ("router.inter_attn.bilinear_v.bias", &[128]),
        ("router.inter_attn.out_proj.weight", &[256, 384]),
        ("router.inter_attn.out_proj.bias", &[256]),
        ("router.inter_attn.norm.weight", &[256]),
        ("router.inter_attn.norm.bias", &[256]),
        ("router.gate_head.weight", &[10, 256]),
        ("router.gate_head.bias", &[10]),
        ("router.v3_gate_head.weight", &[4, 256]),
        ("router.v3_gate_head.bias", &[4]),
        ("router.task_latent_head.weight", &[128, 256]),
        ("router.task_latent_head.bias", &[128]),
        ("router.dist_latent_head.weight", &[128, 256]),
        ("router.dist_latent_head.bias", &[128]),
        ("router.landmark_recon_head.weight", &[23, 256]),
        ("router.landmark_recon_head.bias", &[23]),
        ("router.bb_recon_head.weight", &[16, 256]),
        ("router.bb_recon_head.bias", &[16]),
        ("target_net.film_gamma.weight", &[64, 128]),
        ("target_net.film_gamma.bias", &[64]),
        ("target_net.film_beta.weight", &[64, 128]),
        ("target_net.film_beta.bias", &[64]),
        ("target_net.fc1.weight", &[128, 192]),
        ("target_net.fc1.bias", &[128]),
        ("target_net.fc2.weight", &[128, 128]),
        ("target_net.fc2.bias", &[128]),
        ("target_net.out_head.weight", &[1, 128]),
        ("target_net.out_head.bias", &[1]),
        ("target_net.col_gate_head.weight", &[1, 145]),
        ("target_net.col_gate_head.bias", &[1]),
        ("cov_net.net.0.weight", &[128, 192]),
        ("cov_net.net.0.bias", &[128]),
        ("cov_net.net.2.weight", &[128]),
        ("cov_net.net.2.bias", &[128]),
        ("cov_net.net.3.weight", &[64, 128]),
        ("cov_net.net.3.bias", &[64]),
    ];
    let mut header_map = serde_json::Map::new();
    let mut offset = 0usize;
    for &(name, shape) in specs {
        let len_bytes = shape.iter().product::<usize>() * 4;
        header_map.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + len_bytes],
            }),
        );
        offset += len_bytes;
    }
    let mut header_bytes = serde_json::to_vec(&header_map).unwrap();
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut st_bytes = Vec::with_capacity(8 + header_bytes.len() + offset);
    st_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    st_bytes.extend_from_slice(&header_bytes);
    st_bytes.resize(8 + header_bytes.len() + offset, 0u8);
    std::fs::write(dir.join("test.safetensors"), &st_bytes).unwrap();

    let st_plain = dope(
        &dir,
        &["mlc-v3", "inspect-weights", "--weights", "test.safetensors"],
    );
    assert!(!st_plain.contains("\x1b["));
    let st_val: serde_json::Value = serde_json::from_str(&st_plain).unwrap();
    assert_eq!(st_val["tensor_count"], 56);
    assert_eq!(st_val["parameter_count"], 444_168);
    assert_eq!(st_val["byte_len"], st_bytes.len());

    let st_json_forced = dope(
        &dir,
        &[
            "--color",
            "always",
            "mlc-v3",
            "inspect-weights",
            "--weights",
            "test.safetensors",
            "--json",
        ],
    );
    assert!(!st_json_forced.contains("\x1b["));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&st_json_forced).unwrap(),
        st_val
    );

    let st_col = dope(
        &dir,
        &[
            "--color",
            "always",
            "mlc-v3",
            "inspect-weights",
            "--weights",
            "test.safetensors",
        ],
    );
    assert!(st_col.contains("\x1b["));
    assert!(st_col.contains("56") && st_col.contains("444,168") && st_col.contains("parameters"));
    assert!(st_col.contains("rust-candle-cpu · 0 Python/LibTorch deps"));

    let v3_sample_stdout = dope(
        &dir,
        &[
            "--color",
            "always",
            "mlc-v3",
            "sample",
            "--weights",
            "test.safetensors",
            "--wire",
            "test.dml3",
            "-n",
            "4",
        ],
    );
    assert!(!v3_sample_stdout.contains("\x1b["));
    assert_eq!(v3_sample_stdout.lines().count(), 5);
    assert!(v3_sample_stdout.starts_with("f_000,f_001,target\n"));

    let v3_sample_file_col = dope(
        &dir,
        &[
            "--color",
            "always",
            "mlc-v3",
            "sample",
            "--weights",
            "test.safetensors",
            "--wire",
            "test.dml3",
            "-n",
            "4",
            "-o",
            "v3_syn.csv",
        ],
    );
    assert!(v3_sample_file_col.contains("\x1b[") && v3_sample_file_col.contains("sampled"));
    let v3_disk = std::fs::read_to_string(dir.join("v3_syn.csv")).unwrap();
    assert!(!v3_disk.contains("\x1b[") && v3_disk.lines().count() == 5);

    // Unsupported inputs fail clearly on stderr.
    for bad_args in [
        vec!["--color", "bogus", "doctor"],
        vec!["mlc-v3", "fetch-weights", "--tag", "latest"],
        vec!["mlc-v3", "inspect-wire", "--wire", "train.csv"],
        vec!["mlc-v3", "inspect-weights", "--weights", "train.csv"],
    ] {
        let reject = Command::new(env!("CARGO_BIN_EXE_dope"))
            .current_dir(&dir)
            .args(&bad_args)
            .output()
            .unwrap();
        assert!(!reject.status.success(), "unexpected success: {bad_args:?}");
        assert!(
            !reject.stderr.is_empty(),
            "empty stderr on failure: {bad_args:?}"
        );
    }
    std::fs::remove_dir_all(dir).ok();
}
