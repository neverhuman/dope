//! Unit tests for `dope_kpi render-paper-tables`.

use super::render_paper::{build_tables, PaperTablesArgs, DEFAULT_CANDIDATE};
use std::path::PathBuf;

#[test]
fn paper_tables_match_committed_receipts_with_zero_prior_versions() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let primary_path = repo.join("bench/real-world/results/scoreboard/v3.0.0/primary-methods.json");
    let paper_dir = repo.join("bench/real-world/results/scoreboard/v3.0.0/evidence/paper");
    let compare_paths = vec![
        paper_dir.join("v3-vs-arfpy-arf.json"),
        paper_dir.join("v3-vs-sdv-gaussian-copula.json"),
        paper_dir.join("v3-vs-sdv-tvae.json"),
        paper_dir.join("v3-vs-sdv-ctgan.json"),
    ];
    let primary = std::fs::read(&primary_path).expect("primary-methods.json");
    let compares: Vec<Vec<u8>> = compare_paths
        .iter()
        .map(|p| std::fs::read(p).expect("re-reduced compare receipt"))
        .collect();
    let args = PaperTablesArgs {
        primary: primary_path,
        compares: compare_paths,
        ranking_out: PathBuf::from("unused-ranking.tex"),
        paired_out: PathBuf::from("unused-paired.tex"),
        candidate: DEFAULT_CANDIDATE.to_string(),
        methods: Vec::new(),
        references: Vec::new(),
        labels: Vec::new(),
    };
    let (ranking, paired, holm_map) = build_tables(&args, &primary, &compares).unwrap();
    assert!(!ranking.contains("\\label{") && !paired.contains("\\label{"));
    assert!(
        ranking.contains(r"\textbf{DOPE v3} & 87.9 [81.1, 95.1] & 69.1"),
        "{ranking}"
    );
    assert!(
        ranking.contains(r"ARF & 77.2 [58.4, 90.0]\textdagger{} & 63.0"),
        "{ranking}"
    );
    assert!(
        ranking.contains(r"SDV TVAE & \ensuremath{-}3.9 [\ensuremath{-}26.2, 17.8]\textdagger{}"),
        "{ranking}"
    );
    assert!(
        ranking.contains(r"Copy of fitting rows & 96.1 [93.1, 98.6]"),
        "{ranking}"
    );
    for banned in ["2.0.0", "1.3.0", "pr390", "dse8-p0"] {
        assert!(!ranking.contains(banned), "prior arm {banned} in Table 1");
        assert!(!paired.contains(banned), "prior arm {banned} in Table 2");
    }
    assert!(
        paired.contains(
            r"ARF & +10.8 [\ensuremath{-}3.5, +29.5] & +6.1 [\ensuremath{-}16.5, +28.7] & 0.114 & 0.114 & \textbf{HOLD} & tasks, mlp, realism"
        ),
        "{paired}"
    );
    assert!(
        paired.contains(
            r"SDV TVAE & +91.9 [+71.9, +112.3] & +85.2 [+60.0, +110.0] & \textless{}0.001 & \textless{}0.001 & \textbf{PROMOTE} & none"
        ),
        "{paired}"
    );
    assert!(
        paired.contains(
            r"SDV Gaussian copula & +47.8 [+33.7, +62.6] & +51.9 [+37.0, +69.1] & \textless{}0.001 & \textless{}0.001 & \textbf{HOLD} & tasks"
        ),
        "{paired}"
    );
    assert!(
        paired.contains(
            r"SDV CTGAN & +157.1 [+141.8, +173.6] & +138.5 [+123.2, +155.2] & \textless{}0.001 & \textless{}0.001 & \textbf{HOLD} & privacy"
        ),
        "{paired}"
    );
    assert!((holm_map["arfpy-arf"] - 0.113888611139).abs() < 1e-9);
    assert!(holm_map["sdv-tvae"] < 0.001);

    let committed_ranking =
        std::fs::read_to_string(repo.join("docs/whitepaper-v3/generated/v3-ranking-table.tex"))
            .expect("committed v3-ranking-table.tex");
    let committed_paired =
        std::fs::read_to_string(repo.join("docs/whitepaper-v3/generated/v3-paired-table.tex"))
            .expect("committed v3-paired-table.tex");
    assert_eq!(
        ranking, committed_ranking,
        "docs/whitepaper-v3/generated/v3-ranking-table.tex drifted from committed receipts"
    );
    assert_eq!(
        paired, committed_paired,
        "docs/whitepaper-v3/generated/v3-paired-table.tex drifted from committed receipts"
    );
}

#[test]
fn holm_flip_and_not_evaluated_guard_demote_outcome_to_hold() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let primary_path = repo.join("bench/real-world/results/scoreboard/v3.0.0/primary-methods.json");
    let paper_dir = repo.join("bench/real-world/results/scoreboard/v3.0.0/evidence/paper");
    let primary = std::fs::read(&primary_path).expect("primary-methods.json");
    let arf = std::fs::read(paper_dir.join("v3-vs-arfpy-arf.json")).expect("arf receipt");
    let tvae_raw = std::fs::read(paper_dir.join("v3-vs-sdv-tvae.json")).expect("tvae receipt");

    let mut tvae: serde_json::Value = serde_json::from_slice(&tvae_raw).unwrap();
    tvae["comparisons"][0]["p_one_sided"] = serde_json::json!(0.030);
    tvae["comparisons"][0]["p_holm"] = serde_json::json!(0.030);
    for cond in tvae["comparisons"][0]["decision"]["conditions"]
        .as_array_mut()
        .unwrap()
    {
        if cond["name"] == "holm_p" {
            cond["value"] = serde_json::json!(0.030);
            cond["check"] = serde_json::json!("pass");
        }
    }
    tvae["comparisons"][0]["decision"]["outcome"] = serde_json::json!("PROMOTE");
    let tvae_near = serde_json::to_vec(&tvae).unwrap();

    let args = PaperTablesArgs {
        primary: primary_path.clone(),
        compares: Vec::new(),
        ranking_out: PathBuf::from("unused-ranking.tex"),
        paired_out: PathBuf::from("unused-paired.tex"),
        candidate: DEFAULT_CANDIDATE.to_string(),
        methods: vec!["arfpy-arf".to_string(), "sdv-tvae".to_string()],
        references: Vec::new(),
        labels: Vec::new(),
    };
    let (_, paired, holm_map) = build_tables(&args, &primary, &[arf.clone(), tvae_near]).unwrap();
    assert!((holm_map["sdv-tvae"] - 0.060).abs() < 1e-9);
    assert!(
        paired.contains(r"& 0.030 & 0.060 & \textbf{HOLD} & none \\"),
        "{paired}"
    );
    assert!(!paired.contains(r"\textbf{PROMOTE}"), "{paired}");

    let mut tvae_ne: serde_json::Value = serde_json::from_slice(&tvae_raw).unwrap();
    for guard in tvae_ne["comparisons"][0]["decision"]["guards"]
        .as_array_mut()
        .unwrap()
    {
        if guard["name"] == "privacy_coverage_delta" {
            guard["check"] = serde_json::json!("not_evaluated");
            guard["value"] = serde_json::Value::Null;
        }
    }
    let tvae_ne_bytes = serde_json::to_vec(&tvae_ne).unwrap();
    let (_, paired_ne, _) = build_tables(&args, &primary, &[arf, tvae_ne_bytes]).unwrap();
    assert!(
        paired_ne.contains(r"& \textbf{HOLD} & privacy (n/e) \\"),
        "{paired_ne}"
    );
    assert!(!paired_ne.contains(r"\textbf{PROMOTE}"), "{paired_ne}");
}

#[test]
fn provenance_mismatches_between_primary_and_compare_fail_closed() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let primary_path = repo.join("bench/real-world/results/scoreboard/v3.0.0/primary-methods.json");
    let paper_dir = repo.join("bench/real-world/results/scoreboard/v3.0.0/evidence/paper");
    let primary = std::fs::read(&primary_path).expect("primary-methods.json");
    let tvae_raw = std::fs::read(paper_dir.join("v3-vs-sdv-tvae.json")).expect("tvae receipt");
    let args = PaperTablesArgs {
        primary: primary_path,
        compares: Vec::new(),
        ranking_out: PathBuf::from("unused-ranking.tex"),
        paired_out: PathBuf::from("unused-paired.tex"),
        candidate: DEFAULT_CANDIDATE.to_string(),
        methods: vec!["sdv-tvae".to_string()],
        references: Vec::new(),
        labels: Vec::new(),
    };
    type ReceiptEdit = fn(&mut serde_json::Value);
    let cases: [(&str, ReceiptEdit); 4] = [
        ("rows_sha256", |v| {
            v["inputs"]["rows_sha256"]["dope-v3b-auto-v8"] = serde_json::json!("deadbeef");
        }),
        ("guards_sha256", |v| {
            v["inputs"]["guards_sha256"]["sdv-tvae"] = serde_json::json!("deadbeef");
        }),
        ("guards_summary_sha256", |v| {
            v["inputs"]["guards_summary_sha256"]
                .as_object_mut()
                .unwrap()
                .remove("sdv-tvae");
        }),
        ("primary summary", |v| {
            v["arms"][0]["primary"]["trg"] = serde_json::json!(-0.999);
        }),
    ];
    for (needle, mutate) in cases {
        let mut val: serde_json::Value = serde_json::from_slice(&tvae_raw).unwrap();
        mutate(&mut val);
        let bytes = serde_json::to_vec(&val).unwrap();
        let err = match build_tables(&args, &primary, &[bytes]) {
            Ok(_) => panic!("expected failure for {needle}"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains(needle), "expected {needle} in {err}");
    }
}
