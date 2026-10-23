use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};
fn dir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("dope-landmarks-{name}-{}", std::process::id()));
    if p.exists() {
        std::fs::remove_dir_all(&p).unwrap();
    }
    std::fs::create_dir(&p).unwrap();
    let mut text = "CANARY_header,z,y\n".to_string();
    for i in 0..120 {
        text += &format!(
            "{},{},{}\n",
            i % 43,
            (i * 19) % 71,
            2 * (i % 43) + (i * 19) % 71
        );
    }
    std::fs::write(p.join("fit.csv"), text).unwrap();
    p
}
fn run(p: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(p)
        .env("RAYON_NUM_THREADS", "2")
        .args(args)
        .output()
        .unwrap()
}
fn success(p: &Path, args: &[&str]) -> String {
    let o = run(p, args);
    assert!(
        o.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap()
}
#[test]
fn real_signature_is_private_no_headers_and_bad_panel_refuses() {
    let p = dir("real");
    let o = run(
        &p,
        &[
            "landmarks",
            "real",
            "fit.csv",
            "-t",
            "y",
            "--work-budget",
            "100000",
            "--spectrum-features",
            "0",
            "--json",
        ],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stderr).contains("private measurement"));
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["privacy"], "private");
    assert_eq!(v["cells"].as_array().unwrap().len(), 23);
    assert!(!String::from_utf8_lossy(&o.stdout).contains("CANARY"));
    let bad = run(
        &p,
        &[
            "landmarks",
            "real",
            "fit.csv",
            "-t",
            "y",
            "--panel",
            "CANARY_secret",
        ],
    );
    assert!(!bad.status.success());
    assert!(!String::from_utf8_lossy(&bad.stderr).contains("CANARY"));
    std::fs::remove_dir_all(p).unwrap();
}
#[test]
fn synthetic_descriptor_is_source_free_and_compare_refuses_overlap() {
    let p = dir("synthetic");
    success(
        &p,
        &[
            "fit",
            "fit.csv",
            "-t",
            "y",
            "--plan",
            "3,2,0",
            "--no-trees",
            "--no-audit",
            "-o",
            "m.dope",
            "--quiet",
        ],
    );
    let overlap = run(
        &p,
        &[
            "landmarks",
            "compare",
            "m.dope",
            "--train",
            "fit.csv",
            "--query",
            "fit.csv",
            "--json",
        ],
    );
    assert!(!overlap.status.success());
    assert!(String::from_utf8_lossy(&overlap.stderr).contains("overlapping rows"));
    std::fs::remove_file(p.join("fit.csv")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&success(
        &p,
        &[
            "landmarks",
            "synthetic",
            "m.dope",
            "--rows",
            "100",
            "--work-budget",
            "100000",
            "--spectrum-features",
            "0",
            "--json",
        ],
    ))
    .unwrap();
    assert_eq!(v["privacy"], "artifact_derived");
    std::fs::remove_dir_all(p).unwrap();
}
#[test]
fn release_inventory_accounts_all_bytes_and_never_overwrites() {
    let p = dir("release");
    success(
        &p,
        &[
            "fit-levels",
            "fit.csv",
            "-t",
            "y",
            "--compute-through",
            "low",
            "--export-levels",
            "small",
            "-o",
            "released",
        ],
    );
    let inventory: serde_json::Value =
        serde_json::from_slice(&std::fs::read(p.join("released/inventory.json")).unwrap()).unwrap();
    assert_eq!(inventory["objects"].as_array().unwrap().len(), 1);
    assert_eq!(inventory["objects"][0]["path"], "low.dope");
    let total: u64 = std::fs::read_dir(p.join("released"))
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert_eq!(inventory["total_bytes"], total);
    assert!(
        std::fs::metadata(p.join("released/low.dope"))
            .unwrap()
            .len()
            <= 2048
    );
    let text = inventory.to_string();
    assert!(!text.contains("self_audit"));
    assert!(!text.contains("CANARY"));
    let bad = run(
        &p,
        &[
            "fit-levels",
            "fit.csv",
            "-t",
            "y",
            "--export-levels",
            "all",
            "-o",
            "released",
        ],
    );
    assert!(!bad.status.success());
    std::fs::remove_dir_all(p).unwrap();
}

#[test]
fn combined_descriptors_bind_only_the_authorized_artifacts() {
    let p = dir("combined");
    success(
        &p,
        &[
            "fit-levels",
            "fit.csv",
            "-t",
            "y",
            "--compute-through",
            "medium",
            "--export-levels",
            "medium,low",
            "--descriptors",
            "--combine",
            "-o",
            "released",
        ],
    );
    let inventory: serde_json::Value =
        serde_json::from_slice(&std::fs::read(p.join("released/inventory.json")).unwrap()).unwrap();
    assert_eq!(inventory["objects"].as_array().unwrap().len(), 5);
    for object in inventory["objects"].as_array().unwrap() {
        let data =
            std::fs::read(p.join("released").join(object["path"].as_str().unwrap())).unwrap();
        assert_eq!(object["sha256"], dope_core::levels::digest(&data));
        assert_eq!(object["bytes"], data.len());
    }
    let combined: serde_json::Value =
        serde_json::from_slice(&std::fs::read(p.join("released/combined.embedding.json")).unwrap())
            .unwrap();
    assert_eq!(combined["values"].as_array().unwrap().len(), 192);
    assert_eq!(combined["mask"].as_array().unwrap().len(), 192);
    assert_eq!(combined["blocks"][0]["level"], "low");
    assert_eq!(combined["blocks"][1]["level"], "medium");
    let total: u64 = std::fs::read_dir(p.join("released"))
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert_eq!(inventory["total_bytes"], total);
    assert!(!p.join("released/high.dope").exists());
    assert!(!p.join("released/ultra.dope").exists());
    assert!(!inventory.to_string().contains("CANARY"));
    std::fs::remove_dir_all(p).unwrap();
}
