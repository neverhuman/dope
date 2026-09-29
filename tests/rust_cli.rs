use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use dope_kernel::RouterEvidence;
use dope_kernel::production::write_canonical;
use dope_kernel::router::{QuantizedLayer, ROUTER_OUTPUTS, RouterBundle};

fn run(binary: &str, args: &[&str]) {
    let output = Command::new(binary).args(args).output().unwrap();
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn qualified_router(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let hidden = 128;
    let sketch = 856;
    let candidate_ids = (0..24)
        .map(|index| match index {
            0 => "independent_quantile".into(),
            1 => "chow_liu".into(),
            _ => format!("candidate-{index:02}"),
        })
        .collect::<Vec<_>>();
    let bundle = RouterBundle {
        format: "dope-distilled-router".into(),
        version: 1,
        sketch_mean: vec![0.0; sketch],
        sketch_std: vec![1.0; sketch],
        input_scale: 0.01,
        candidate_embeddings: candidate_ids
            .iter()
            .enumerate()
            .map(|(index, _)| vec![index as f32 / 24.0])
            .collect(),
        candidate_ids,
        layers: vec![
            QuantizedLayer {
                input: sketch + 1,
                output: 3,
                weights: vec![1; (sketch + 1) * 3],
                biases: vec![0; 3],
                multiplier: 0.01,
            },
            QuantizedLayer {
                input: 3,
                output: hidden,
                weights: vec![1; 3 * hidden],
                biases: vec![0; hidden],
                multiplier: 0.5,
            },
            QuantizedLayer {
                input: hidden,
                output: ROUTER_OUTPUTS,
                weights: vec![1; hidden * ROUTER_OUTPUTS],
                biases: vec![0; ROUTER_OUTPUTS],
                multiplier: 0.5,
            },
        ],
        output_scale: vec![0.1; ROUTER_OUTPUTS],
        output_bias: vec![0.2; ROUTER_OUTPUTS],
        ga2m_router: None,
        trained: true,
        training_evidence_sha256: Some("a".repeat(64)),
    };
    let evidence = RouterEvidence {
        format: "dope-router-evidence".into(),
        version: 1,
        top_four_oracle_recall: 1.0,
        maximum_profile_regret_upper: 0.01,
        beats_random: true,
        beats_best_fixed: true,
        beats_ga2m: true,
        paired_hypervolume_improvement: 0.1,
        paired_hypervolume_ci_lower: 0.01,
        student_max_abs_difference: 0.0001,
        top_choice_agreement: 1.0,
        bundle_bytes: serde_json::to_vec(&bundle).unwrap().len() as u64,
        inference_p95_ms: 1.0,
        sketch_p95_ms: 10.0,
        training_evidence_sha256: "a".repeat(64),
        best_fixed_candidate_id: "independent_quantile".into(),
        student_mean_regret: 0.01,
        random_mean_regret: 0.02,
        best_fixed_mean_regret: 0.02,
        ga2m_mean_regret: 0.02,
    };
    let bundle_path = root.join("router.bundle.json");
    let evidence_path = root.join("router-evidence.json");
    write_canonical(&bundle_path, &bundle).unwrap();
    write_canonical(&evidence_path, &evidence).unwrap();
    (bundle_path, evidence_path)
}

#[test]
fn embed_dataset_cli_writes_canonical_json_and_vector_csv() {
    let root = std::env::temp_dir().join(format!("dope-embed-cli-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let csv = root.join("train.csv");
    fs::write(&csv, "feature,outcome\n10,-1\n20,2\n30,-1\n").unwrap();
    let (bundle, evidence) = qualified_router(&root);
    let out = root.join("embedding.json");
    let repeated_out = root.join("embedding-repeated.json");
    let vector = root.join("embedding.csv");
    let binary = env!("CARGO_BIN_EXE_dope-kernel");
    let csv_s = text(&csv);
    let bundle_s = text(&bundle);
    let evidence_s = text(&evidence);
    let out_s = text(&out);
    let vector_s = text(&vector);
    let base_args = [
        "embed-dataset",
        "--csv",
        csv_s.as_str(),
        "--target-column",
        "outcome",
        "--task",
        "binary",
        "--router-bundle",
        bundle_s.as_str(),
        "--router-evidence",
        evidence_s.as_str(),
    ];
    let output = Command::new(binary)
        .args(base_args)
        .args(["--out", out_s.as_str(), "--vector-out", vector_s.as_str()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["dimension"], 4168);
    assert_eq!(summary["candidates"].as_array().unwrap().len(), 24);
    assert_eq!(summary["candidates"][0], "independent_quantile");
    let embedding: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(embedding["format"], "dope-dataset-action-embedding");
    assert_eq!(embedding["dimension"], 4168);
    assert_eq!(embedding["rows"], 3);
    assert_eq!(embedding["features"], 1);
    assert_eq!(fs::read_to_string(&vector).unwrap().lines().count(), 2);

    run(
        binary,
        &[
            "embed-dataset",
            "--csv",
            &text(&csv),
            "--target-column",
            "outcome",
            "--task",
            "binary",
            "--router-bundle",
            &text(&bundle),
            "--router-evidence",
            &text(&evidence),
            "--out",
            &text(&repeated_out),
        ],
    );
    assert_eq!(fs::read(out).unwrap(), fs::read(repeated_out).unwrap());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn regression_corpus_cli_reconstructs_headers_shards_and_verifies_outputs() {
    let root =
        std::env::temp_dir().join(format!("dope-regression-batch-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let corpus = root.join("corpus");
    let output_dir = root.join("embeddings");
    fs::create_dir_all(&corpus).unwrap();
    let first = corpus.join("0000000000000001");
    fs::create_dir_all(&first).unwrap();
    fs::write(first.join("train.csv"), "1,2\n3,4\n").unwrap();
    fs::write(
        first.join("meta.json"),
        serde_json::to_vec(&serde_json::json!({
            "task_type": "regression",
            "target_column": "target",
            "target_is_final_column": true,
            "output_headers": ["feature", "target"],
            "feature_columns": ["feature"]
        }))
        .unwrap(),
    )
    .unwrap();
    let fallback = corpus.join("0000000000000002");
    fs::create_dir_all(&fallback).unwrap();
    fs::write(fallback.join("train.csv"), "5,6\n7,8\n").unwrap();
    fs::write(
        fallback.join("meta.json"),
        serde_json::to_vec(&serde_json::json!({
            "task_type": "regression",
            "target_column": "target",
            "target_is_final_column": true,
            "output_headers": ["stale", "feature", "target"],
            "feature_columns": ["feature,quoted"]
        }))
        .unwrap(),
    )
    .unwrap();
    let ignored = corpus.join("not-a-dataset");
    fs::create_dir_all(&ignored).unwrap();
    fs::write(ignored.join("train.csv"), "1,2\n").unwrap();
    fs::write(ignored.join("meta.json"), br#"{"task_type":"regression"}"#).unwrap();
    let target_only = corpus.join("0000000000000003");
    fs::create_dir_all(&target_only).unwrap();
    fs::write(target_only.join("train.csv"), "10\n20\n15\n").unwrap();
    fs::write(
        target_only.join("meta.json"),
        serde_json::to_vec(&serde_json::json!({
            "task_type": "regression",
            "target_column": "target",
            "target_is_final_column": true,
            "output_headers": ["stale", "target"],
            "feature_columns": ["stale-a", "stale-b"]
        }))
        .unwrap(),
    )
    .unwrap();

    let (bundle, evidence) = qualified_router(&root);
    let binary = env!("CARGO_BIN_EXE_dope-kernel");
    let qualification = Command::new(binary)
        .args([
            "qualify-embedding-router",
            "--router-bundle",
            &text(&bundle),
            "--router-evidence",
            &text(&evidence),
        ])
        .output()
        .unwrap();
    assert!(qualification.status.success());
    let qualification: serde_json::Value = serde_json::from_slice(&qualification.stdout).unwrap();
    assert_eq!(qualification["version"], 2);
    assert_eq!(qualification["embedding_eligible"], true);
    assert_eq!(qualification["promotion_qualified"], true);
    assert_eq!(
        qualification["promotion_failed_gates"],
        serde_json::json!([])
    );
    assert_eq!(qualification["dimension"], 4168);
    let inventory = Command::new(binary)
        .args([
            "inspect-regression-corpus",
            "--dataset-root",
            &text(&corpus),
        ])
        .output()
        .unwrap();
    assert!(inventory.status.success());
    let inventory: serde_json::Value = serde_json::from_slice(&inventory.stdout).unwrap();
    assert_eq!(inventory["discovered"], 3);
    assert_eq!(inventory["output_headers"], 1);
    assert_eq!(inventory["feature_column_fallbacks"], 2);
    assert_eq!(inventory["schema_mismatches"], 0);

    let manifest_path = root.join("manifest.json");
    run(
        binary,
        &[
            "embed-regression-corpus",
            "--dataset-root",
            &text(&corpus),
            "--output-dir",
            &text(&output_dir),
            "--router-bundle",
            &text(&bundle),
            "--router-evidence",
            &text(&evidence),
            "--method",
            "attention-small",
            "--expected-dimension",
            "4168",
            "--shard-index",
            "0",
            "--shard-count",
            "1",
            "--jobs",
            "2",
            "--determinism-checks",
            "3",
            "--manifest-out",
            &text(&manifest_path),
        ],
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["version"], 2);
    assert_eq!(manifest["discovered"], 3);
    assert_eq!(manifest["assigned"], 3);
    assert_eq!(manifest["success"], 3);
    assert_eq!(manifest["schema_mismatch"], 0);
    assert_eq!(manifest["failed"], 0);
    assert_eq!(manifest["determinism_checked"], 3);
    assert_eq!(manifest["records"][0]["header_source"], "output_headers");
    assert_eq!(
        manifest["records"][1]["header_source"],
        "feature_columns+target"
    );
    assert_eq!(
        manifest["records"][2]["header_source"],
        "truncated_feature_columns+target"
    );
    assert_eq!(manifest["records"][2]["features"], 0);
    let target_embedding: serde_json::Value = serde_json::from_slice(
        &fs::read(output_dir.join("0000000000000003_attention-small_4168.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(target_embedding["features"], 0);
    assert_eq!(
        target_embedding["normalization"]["features"],
        serde_json::json!([])
    );
    for id in ["0000000000000001", "0000000000000002", "0000000000000003"] {
        let vector = output_dir.join(format!("{id}_attention-small_4168.vs"));
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(vector)
            .unwrap();
        let records = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|record| record.len() == 4168));
    }
    let verification = Command::new(binary)
        .args([
            "verify-regression-embeddings",
            "--output-dir",
            &text(&output_dir),
            "--router-bundle",
            &text(&bundle),
            "--router-evidence",
            &text(&evidence),
            "--method",
            "attention-small",
            "--expected-dimension",
            "4168",
            "--expected-count",
            "3",
        ])
        .output()
        .unwrap();
    assert!(
        verification.status.success(),
        "{}",
        String::from_utf8_lossy(&verification.stderr)
    );
    let verification: serde_json::Value = serde_json::from_slice(&verification.stdout).unwrap();
    assert_eq!(verification["version"], 2);
    assert_eq!(verification["datasets"], 3);
    assert_eq!(verification["dimension"], 4168);
    assert_eq!(verification["zero_feature_datasets"], 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn regression_corpus_cli_admits_and_reports_promotion_failures() {
    let root = std::env::temp_dir().join(format!(
        "dope-regression-batch-gate-cli-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let (bundle, evidence_path) = qualified_router(&root);
    let mut evidence: RouterEvidence =
        serde_json::from_slice(&fs::read(&evidence_path).unwrap()).unwrap();
    evidence.beats_random = false;
    write_canonical(&evidence_path, &evidence).unwrap();
    let dataset = root.join("0000000000000001");
    fs::create_dir_all(&dataset).unwrap();
    fs::write(dataset.join("train.csv"), "1,2\n3,4\n").unwrap();
    fs::write(
        dataset.join("meta.json"),
        br#"{"task_type":"regression","target_column":"target","target_is_final_column":true,"output_headers":["feature","target"],"feature_columns":["feature"]}"#,
    )
    .unwrap();
    let output_dir = root.join("embeddings");
    let manifest = root.join("manifest.json");
    let output = Command::new(env!("CARGO_BIN_EXE_dope-kernel"))
        .args([
            "embed-regression-corpus",
            "--dataset-root",
            &text(&root),
            "--output-dir",
            &text(&output_dir),
            "--router-bundle",
            &text(&bundle),
            "--router-evidence",
            &text(&evidence_path),
            "--method",
            "attention-small",
            "--expected-dimension",
            "4168",
            "--shard-index",
            "0",
            "--shard-count",
            "1",
            "--manifest-out",
            &text(&manifest),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    assert_eq!(manifest["success"], 1);
    assert_eq!(manifest["router"]["embedding_eligible"], true);
    assert_eq!(manifest["router"]["promotion_qualified"], false);
    assert_eq!(
        manifest["router"]["promotion_failed_gates"],
        serde_json::json!(["beats_random"])
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_cli_end_to_end() {
    let root = std::env::temp_dir().join(format!("dope-rust-cli-{}", std::process::id()));
    let dataset = root.join("corpus/dataset");
    let packed = root.join("packed");
    fs::create_dir_all(&dataset).unwrap();
    let mut train = File::create(dataset.join("train.csv")).unwrap();
    let mut test = File::create(dataset.join("test.csv")).unwrap();
    for row in 0..120 {
        let x0 = row as f32 / 119.0;
        let x1 = (row % 2) as f32;
        let y = (0.1 + 0.7 * x0 + 0.1 * x1).clamp(0.0, 1.0);
        writeln!(train, "{x0},{x1},{y}").unwrap();
        if row < 30 {
            writeln!(test, "{x0},{x1},{y}").unwrap();
        }
    }
    let binary = env!("CARGO_BIN_EXE_dope-kernel");
    let artifact = root.join("kernel.dpk");
    let inspection = root.join("inspection.json");
    let synthetic = root.join("synthetic.csv");
    let artifact_s = text(&artifact);
    let dataset_s = text(&dataset);
    let inspection_s = text(&inspection);
    let synthetic_s = text(&synthetic);
    run(
        binary,
        &[
            "compile",
            "--dataset-dir",
            &dataset_s,
            "--task",
            "regression",
            "--out",
            &artifact_s,
            "--seed",
            "7",
        ],
    );
    run(
        binary,
        &["inspect", "--kernel", &artifact_s, "--out", &inspection_s],
    );
    run(
        binary,
        &[
            "sample",
            "--kernel",
            &artifact_s,
            "--rows",
            "25",
            "--out",
            &synthetic_s,
            "--seed",
            "8",
        ],
    );
    assert!(fs::read(&artifact).unwrap().starts_with(b"DPK3"));
    let inspected: serde_json::Value =
        serde_json::from_slice(&fs::read(&inspection).unwrap()).unwrap();
    assert_eq!(inspected["version"], 3);
    assert_eq!(inspected["target_position"], "last");
    assert_eq!(fs::read_to_string(&synthetic).unwrap().lines().count(), 25);

    let corpus_s = text(&root.join("corpus"));
    let packed_s = text(&packed);
    run(
        binary,
        &[
            "pack-corpus",
            "--corpus",
            &corpus_s,
            "--out",
            &packed_s,
            "--shard-mib",
            "0.001",
        ],
    );
    assert!(packed.join("manifest.json").is_file());
    assert!(packed.join("split-manifest.json").is_file());
    let calibration = root.join("language-run");
    let calibration_s = text(&calibration);
    run(
        binary,
        &[
            "calibrate-language",
            "--corpus",
            &packed_s,
            "--out",
            &calibration_s,
        ],
    );
    let language: serde_json::Value =
        serde_json::from_slice(&fs::read(calibration.join("language.json")).unwrap()).unwrap();
    assert_eq!(language["learned_from"], "training_lineages_only");
    assert_eq!(language["test_split_opened"], false);
    let language_kernel = root.join("kernel-language.dpk");
    run(
        binary,
        &[
            "compile",
            "--dataset-dir",
            &dataset_s,
            "--task",
            "regression",
            "--language",
            &text(&calibration.join("language.json")),
            "--out",
            &text(&language_kernel),
        ],
    );
    assert!(fs::read(language_kernel).unwrap().starts_with(b"DPK3"));

    let certification = root.join("certification.json");
    let certification_s = text(&certification);
    run(
        binary,
        &[
            "certify",
            "--real-dir",
            &dataset_s,
            "--kernel",
            &artifact_s,
            "--out",
            &certification_s,
        ],
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(&certification).unwrap()).unwrap();
    assert_eq!(report["format"], "dope-kernel-certification");
    assert_eq!(report["version"], 4);
    assert_eq!(report["certified"], false);
    assert_eq!(report["master_fitness"]["eligible"], false);
    assert!(report["master_fitness"]["score"].is_null());
    assert_eq!(report["master_fitness"]["privacy_soft_weight"], 0.0);
    assert_eq!(report["master_fitness"]["version"], 2);
    assert_eq!(report["auditors"].as_object().unwrap().len(), 6);
    assert_eq!(
        report["auditors"]["elastic_net_glm"]["metrics"]
            .as_array()
            .unwrap()
            .len(),
        36
    );
    assert_eq!(
        report["isolation"]["synthetic_sizes"],
        serde_json::json!([120, 240, 480, 960])
    );
    assert!(report["privacy"]["near_copy_count"].is_number());
    assert!(
        report["failed_gates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|gate| gate == "auditor_family_coverage")
    );

    let release = root.join("release");
    let conversion = Command::new(binary)
        .args([
            "convert",
            "--dataset-dir",
            &dataset_s,
            "--real-holdout-dir",
            &dataset_s,
            "--task",
            "regression",
            "--out",
            &text(&release),
            "--synthetic-seed",
            "23",
        ])
        .output()
        .unwrap();
    assert!(!conversion.status.success());
    assert!(String::from_utf8_lossy(&conversion.stderr).contains("auditor coverage"));
    assert!(!release.exists());

    let v1 = root.join("v1.dk.json");
    let v1_sample = root.join("v1.csv");
    fs::write(&v1, r#"{"format":"dope-kernel","version":1,"shell":{"task":"binary","n":20,"p":1,"seed":7},"marginals":[{"kind":"bern","params":{"p":0.4}}],"dependence":{"kind":"ind","params":{}},"target":{"kind":"lift_logit","params":{"intercept":0.0,"terms":[[0,2.0]]}},"residual":{"kind":"bernoulli_cal","params":{"base_rate":0.5}},"program":"(dk v=1)"}"#).unwrap();
    run(
        binary,
        &[
            "sample",
            "--kernel",
            &text(&v1),
            "--rows",
            "10",
            "--out",
            &text(&v1_sample),
        ],
    );
    assert_eq!(fs::read_to_string(&v1_sample).unwrap().lines().count(), 10);
    fs::remove_dir_all(&root).unwrap();
}
