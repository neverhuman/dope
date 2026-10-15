use std::{fs, path::Path, process::Command};

#[test]
fn native_development_pilot_cli_matches_exact_successor_without_admission() {
    let fixture: serde_json::Value =
        serde_json::from_slice(include_bytes!("../fixtures/n19-pilot-v2.json")).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/native-pilot-cli")
        .join(std::process::id().to_string());
    fs::create_dir_all(&root).unwrap();
    let plan = root.join("plan.json");
    let attempts = root.join("attempts.json");
    fs::write(&plan, serde_json::to_vec(&fixture["plan"]).unwrap()).unwrap();
    fs::write(&attempts, serde_json::to_vec(&fixture["attempts"]).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_dope-kernel"))
        .arg("inspect-development-pilot")
        .arg("--plan")
        .arg(&plan)
        .arg("--attempts")
        .arg(&attempts)
        .arg("--expected-plan-digest")
        .arg(fixture["expected_plan_digest"].as_str().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let inspection: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(inspection["summary"], fixture["expected_summary"]);
    assert_eq!(inspection["execution_authorized"], false);
    assert_eq!(inspection["scientific_admission"], false);
    assert!(
        !String::from_utf8(output.stdout)
            .unwrap()
            .contains("custodian-alpha")
    );
    let rejected = Command::new(env!("CARGO_BIN_EXE_dope-kernel"))
        .arg("inspect-development-pilot")
        .arg("--plan")
        .arg(&plan)
        .arg("--expected-plan-digest")
        .arg("0".repeat(64))
        .output()
        .unwrap();
    assert!(!rejected.status.success());
}
