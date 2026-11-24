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
