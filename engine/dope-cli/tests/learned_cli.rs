//! A recipient with decoder weights only can sample the exact admitted code.
use std::{path::PathBuf, process::Command};
#[test]
fn frozen_encode_inspect_and_decoder_only_sample_without_overwrite_or_source_echo() {
    let root = std::env::temp_dir().join(format!("dope-learned-cli-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../dope-core/tests/fixtures/learned-v1/model");
    let csv = root.join("fit.csv");
    let mut text = String::from("PRIVATE_HEADER,c,target\n");
    for i in 0..80 {
        text += &format!("{},PRIVATE_CATEGORY_{},{}\n", i, i % 2, i * 2);
    }
    std::fs::write(&csv, text).unwrap();
    let code = root.join("low.code");
    let invoke = |args: Vec<String>| {
        Command::new(env!("CARGO_BIN_EXE_dope"))
            .args(args)
            .output()
            .unwrap()
    };
    let encode = vec!["learned", "encode", "--model"]
        .into_iter()
        .map(str::to_string)
        .chain([
            fixture.display().to_string(),
            "--csv".into(),
            csv.display().to_string(),
            "--target".into(),
            "target".into(),
            "--kinds".into(),
            "numeric,categorical,numeric".into(),
            "--task".into(),
            "regression".into(),
            "--levels".into(),
            "low".into(),
            "--out".into(),
            code.display().to_string(),
        ])
        .collect::<Vec<_>>();
    let first = invoke(encode.clone());
    assert!(first.status.success(), "{first:?}");
    assert!(!invoke(encode).status.success());
    let bytes = std::fs::read(&code).unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("PRIVATE_"));
    let recipient = root.join("recipient");
    std::fs::create_dir(&recipient).unwrap();
    for file in ["model.json", "decoder.safetensors"] {
        std::fs::copy(fixture.join(file), recipient.join(file)).unwrap();
    }
    std::fs::remove_file(csv).unwrap();
    let out = root.join("sample.csv");
    let sampled = invoke(vec![
        "learned".into(),
        "sample".into(),
        "--model".into(),
        recipient.display().to_string(),
        "--code".into(),
        code.display().to_string(),
        "--level".into(),
        "low".into(),
        "--rows".into(),
        "8".into(),
        "--steps".into(),
        "8".into(),
        "--out".into(),
        out.display().to_string(),
    ]);
    assert!(sampled.status.success(), "{sampled:?}");
    assert_eq!(std::fs::read_to_string(out).unwrap().lines().count(), 9);
    let inspected = invoke(vec![
        "learned".into(),
        "inspect".into(),
        "--model".into(),
        recipient.display().to_string(),
        "--code".into(),
        code.display().to_string(),
    ]);
    assert!(inspected.status.success());
    let bad = root.join("bad.code");
    std::fs::write(&bad, b"PRIVATE_ERROR_CANARY").unwrap();
    let failure = invoke(vec![
        "learned".into(),
        "inspect".into(),
        "--model".into(),
        recipient.display().to_string(),
        "--code".into(),
        bad.display().to_string(),
    ]);
    let stderr = String::from_utf8_lossy(&failure.stderr);
    assert!(!failure.status.success());
    assert!(!stderr.contains("PRIVATE_ERROR_CANARY"));
    assert!(stderr.contains("hint:"));
    std::fs::remove_dir_all(root).unwrap();
}
