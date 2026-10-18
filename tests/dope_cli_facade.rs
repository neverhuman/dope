use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
fn root(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/dope-cli-facade")
        .join(std::process::id().to_string())
        .join(name);
    fs::create_dir_all(&path).unwrap();
    path
}
fn call(args: &[&str], code: i32) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_dope"))
        .args(args)
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(code),
        "args={args:?}\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}
fn fixture(root: &Path) -> (PathBuf, PathBuf) {
    let csv = root.join("private-source.csv");
    let schema = root.join("private-schema.json");
    fs::write(
        &csv,
        "SECRET_HEADER,SECRET_TARGET\n10,100\n20,200\n30,300\n40,400\n",
    )
    .unwrap();
    fs::write(&schema, serde_json::to_vec(&json!({
        "format":"dope-table-schema","version":1,"metadata_visibility":"approved_semantic_labels",
        "columns":[
            {"id":"c0001","source_name":"SECRET_HEADER","role":"feature","dtype":"continuous","semantic_label":"SECRET_LABEL","units":"SECRET_UNITS"},
            {"id":"c0002","source_name":"SECRET_TARGET","role":"target","dtype":"continuous"}
        ]
    })).unwrap()).unwrap();
    (csv, schema)
}
fn private_free(bytes: &[u8], root: &Path) {
    let text = String::from_utf8_lossy(bytes);
    for secret in [
        "SECRET_HEADER",
        "SECRET_TARGET",
        "SECRET_LABEL",
        "SECRET_UNITS",
        s(root),
    ] {
        assert!(!text.contains(secret), "leaked {secret}: {text}");
    }
}
#[test]
fn registry_help_and_aliases_are_available() {
    let output = call(&["--help"], 0);
    let help = String::from_utf8(output.stdout).unwrap();
    for name in "prepare validate-input compile inspect sample doctor evaluate report demo --offline --panel --max-input-bytes".split_whitespace() {
        assert!(help.contains(name), "missing {name}");
        if !name.starts_with('-') {
            call(&[name, "--help"], 0);
        }
    }
    let output = call(&["doctor", "--panel-tag=development", "--offline"], 0);
    assert!(String::from_utf8_lossy(&output.stderr).contains("use --panel"));
    let doctor: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(doctor["candidate_evaluation"], "unavailable");
    assert_eq!(doctor["weights_downloaded"], false);
    assert_eq!(doctor["legal_release_state"], "pending");
}
#[test]
fn typed_workflow_preserves_units_and_keeps_public_metadata_opaque() {
    let root = root("workflow");
    let (csv, schema) = fixture(&root);
    let receipt = root.join("intake.json");
    let alias = call(&["validate", "--csv", s(&csv), "--schema", s(&schema)], 0);
    private_free(&alias.stdout, &root);
    call(
        &[
            "prepare",
            "--csv",
            s(&csv),
            "--schema",
            s(&schema),
            "--out",
            s(&receipt),
        ],
        0,
    );
    let intake: Value = serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
    assert_eq!(intake["state"], "input_validated");
    assert_eq!(intake["fit_transform_state"], "not_fitted");
    assert_eq!(intake["qualified"], false);
    let artifact = root.join("raw.dpk");
    let report = root.join("compile.json");
    let output = call(
        &[
            "compile",
            "--csv",
            s(&csv),
            "--schema",
            s(&schema),
            "--out",
            s(&artifact),
            "--report",
            s(&report),
            "--seed",
            "7",
        ],
        0,
    );
    private_free(&output.stdout, &root);
    private_free(&fs::read(report).unwrap(), &root);
    let output = call(&["inspect", "--artifact", s(&artifact)], 0);
    private_free(&output.stdout, &root);
    let inspected: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        inspected["certification"],
        "unqualified_development_artifact"
    );
    assert_eq!(
        inspected["accounting"]["complete_sampling_bytes"],
        fs::metadata(&artifact).unwrap().len()
    );
    let declaration =
        dope_kernel::schema::TableSchemaV1::from_private_json(&fs::read(&schema).unwrap()).unwrap();
    let table = dope_kernel::typed_data::TypedTable::read_csv(
        fs::File::open(&csv).unwrap(),
        declaration,
        Default::default(),
    )
    .unwrap();
    let mixed = dope_kernel::typed_data::mixed::compile_mixed(
        &table,
        &[0, 1, 2, 3],
        &Default::default(),
        32768,
        dope_kernel::typed_data::mixed::MixedProfile::IndependentV1,
    )
    .unwrap();
    let mixed_path = root.join("mixed.dpk");
    fs::write(&mixed_path, mixed.artifact).unwrap();
    let output = call(&["inspect", "--artifact", s(&mixed_path)], 0);
    private_free(&output.stdout, &root);
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap()["format"],
        "dope-mixed-sampling-artifact"
    );
    let first = root.join("first.csv");
    let second = root.join("second.csv");
    for out in [&first, &second] {
        let output = call(
            &[
                "sample",
                "--artifact",
                s(&artifact),
                "--out",
                s(out),
                "--rows",
                "8",
                "--seed",
                "11",
            ],
            0,
        );
        private_free(&output.stdout, &root);
    }
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    let samples = fs::read_to_string(first).unwrap();
    assert_eq!(samples.lines().count(), 8);
    for line in samples.lines() {
        let row = line
            .split(',')
            .map(|v| v.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(row.len(), 2);
        assert!((10.0..=40.0).contains(&row[0]));
        assert!(row[1] > 1.0, "sampling must preserve original units");
    }
}
#[test]
fn parser_failures_never_echo_private_arguments() {
    let root = root("parser-redaction");
    for args in [
        vec!["SECRET_HEADER"],
        vec!["doctor", "--SECRET_HEADER", s(&root)],
        vec!["doctor", "--panel", "SECRET_HEADER"],
        vec!["doctor", "--max-input-bytes", "SECRET_HEADER"],
        vec![
            "sample",
            "--artifact",
            s(&root),
            "--rows",
            "NaN",
            "--out",
            "SECRET_HEADER",
        ],
        vec![
            "compile",
            "--deadline-seconds",
            "inf",
            "--out",
            "SECRET_HEADER",
        ],
        vec![
            "compile",
            "--deadline-seconds",
            "0",
            "--out",
            "SECRET_HEADER",
        ],
        vec![
            "compile",
            "--deadline-seconds",
            "601",
            "--out",
            "SECRET_HEADER",
        ],
    ] {
        let output = call(&args, 2);
        assert!(output.stdout.is_empty());
        private_free(&output.stderr, &root);
    }
}
#[test]
fn offline_and_missing_capabilities_fail_before_io() {
    let root = root("offline");
    let out = root.join("absent.json");
    for args in [
        vec!["doctor", "--network"],
        vec!["doctor", "--offline=false"],
        vec!["doctor", "--panel", "tdl-candidate"],
        vec!["evaluate", "--artifact", "SECRET_HEADER", "--out", s(&out)],
    ] {
        let output = call(&args, 3);
        assert!(output.stdout.is_empty());
        private_free(&output.stderr, &root);
        assert!(!out.exists());
    }
    call(
        &[
            "inspect",
            "--artifact",
            "https://private.invalid/SECRET_HEADER",
        ],
        2,
    );
}
#[test]
fn input_and_output_budgets_fail_without_artifacts() {
    let root = root("budget");
    let (csv, schema) = fixture(&root);
    for (index, limit) in [
        ["--max-input-bytes", "8"],
        ["--max-rows", "2"],
        ["--max-cells", "3"],
        ["--max-field-bytes", "2"],
    ]
    .iter()
    .enumerate()
    {
        let out = root.join(format!("invalid-{index}.dpk"));
        let output = call(
            &[
                "compile",
                "--csv",
                s(&csv),
                "--schema",
                s(&schema),
                "--out",
                s(&out),
                limit[0],
                limit[1],
            ],
            5,
        );
        private_free(&output.stderr, &root);
        assert!(!out.exists());
    }
    let out = root.join("too-small.dpk");
    call(
        &[
            "compile",
            "--csv",
            s(&csv),
            "--schema",
            s(&schema),
            "--out",
            s(&out),
            "--max-complete-bytes",
            "1",
        ],
        5,
    );
    assert!(!out.exists());
    call(&["demo", "--out", s(&out), "--max-output-bytes", "1"], 5);
    assert!(!out.exists());
    call(
        &[
            "compile",
            "--csv",
            s(&csv),
            "--schema",
            s(&schema),
            "--out",
            s(&out),
            "--require-formal-dp",
        ],
        8,
    );
}
#[test]
fn invalid_csv_missing_files_and_output_collisions_are_redacted() {
    let root = root("invalid-input");
    let (csv, schema) = fixture(&root);
    fs::write(
        &csv,
        "SECRET_HEADER,SECRET_TARGET\nSECRET_VALUE,99999999999999\n",
    )
    .unwrap();
    let output = call(
        &["validate-input", "--csv", s(&csv), "--schema", s(&schema)],
        2,
    );
    private_free(&output.stderr, &root);
    assert!(!String::from_utf8_lossy(&output.stderr).contains("SECRET_VALUE"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("99999999999999"));
    let missing = root.join("SECRET_HEADER");
    let output = call(&["inspect", "--artifact", s(&missing)], 6);
    private_free(&output.stderr, &root);
    call(&["demo", "--out", s(&csv)], 2);
    assert!(fs::read_to_string(&csv).unwrap().contains("SECRET_VALUE"));
    let out = root.join("same.dpk");
    call(&["demo", "--out", s(&out), "--report", s(&out)], 2);
    assert!(!out.exists());
    call(
        &["doctor", "--out", s(&root.join("missing-parent/out.json"))],
        70,
    );
}
#[test]
fn reports_keep_science_privacy_and_rights_separate_and_redact_strings() {
    let root = root("reports");
    let path = root.join("receipt.json");
    for (index, certified, gate, code) in [
        (0, false, "driver", 7),
        (1, false, "membership_inference", 8),
        (2, true, "driver", 4),
    ] {
        let out = root.join(format!("projected-{index}.json"));
        fs::write(&path, serde_json::to_vec(&json!({
            "format":"dope-kernel-certification","version":5,"certified":certified,
            "gates":{gate:certified},"failed_gates":if certified {vec![]} else {vec!["SECRET_HEADER"]},
            "private_path":s(&root), "extrema":[999999999999999i64],
        })).unwrap()).unwrap();
        let output = call(&["report", "--report", s(&path), "--out", s(&out)], code);
        private_free(&output.stderr, &root);
        private_free(&fs::read(&out).unwrap(), &root);
        let projected: Value = serde_json::from_slice(&fs::read(out).unwrap()).unwrap();
        assert_eq!(projected["tdl_qualified"], false);
        assert_eq!(projected["legal_release_state"], "pending");
        assert_eq!(projected["formal_dp_state"], "unverified");
    }
    fs::write(
        &path,
        r#"{"format":"dope-kernel-certification","format":"SECRET_HEADER"}"#,
    )
    .unwrap();
    let output = call(&["report", "--report", s(&path)], 2);
    assert!(String::from_utf8_lossy(&output.stderr).contains("duplicate_json_key"));
    fs::write(&path, r#"{"format":"dope-kernel-certification","version":5,"certified":true,"gates":{"driver":false},"failed_gates":[]}"#).unwrap();
    call(&["report", "--report", s(&path)], 2);
}
#[test]
fn demo_is_fixture_only_and_legacy_v1_inspection_hides_descriptions() {
    let root = root("demo-v1");
    let artifact = root.join("demo.dpk");
    let receipt = root.join("demo.json");
    call(&["demo", "--out", s(&artifact), "--report", s(&receipt)], 0);
    let report: Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
    assert_eq!(report["fixture_only"], true);
    assert_eq!(report["certified"], false);
    let v1 = root.join("v1.json");
    fs::write(&v1, r#"{"format":"dope-kernel","version":1,"shell":{"task":"binary","n":20,"p":1,"seed":7},"marginals":[{"kind":"bern","params":{"p":0.4}}],"dependence":{"kind":"ind","params":{}},"target":{"kind":"lift_logit","params":{"intercept":0.0,"terms":[[0,2.0]]}},"residual":{"kind":"bernoulli_cal","params":{"base_rate":0.5}},"program":"SECRET_HEADER"}"#).unwrap();
    let output = call(&["inspect", "--artifact", s(&v1)], 0);
    private_free(&output.stdout, &root);
    let samples = root.join("v1.csv");
    call(
        &[
            "sample",
            "--artifact",
            s(&v1),
            "--rows",
            "5",
            "--out",
            s(&samples),
        ],
        0,
    );
    assert_eq!(fs::read_to_string(samples).unwrap().lines().count(), 5);
    let mut invalid: Value = serde_json::from_slice(&fs::read(&v1).unwrap()).unwrap();
    let rejected = root.join("rejected.csv");
    for (width, terms, code) in [
        (json!(4_294_967_297u64), json!([[0, 2.0]]), 5),
        (json!(1), json!([[]]), 2),
    ] {
        invalid["shell"]["p"] = width;
        invalid["target"]["params"]["terms"] = terms;
        fs::write(&v1, serde_json::to_vec(&invalid).unwrap()).unwrap();
        call(
            &[
                "sample",
                "--artifact",
                s(&v1),
                "--rows",
                "1",
                "--out",
                s(&rejected),
            ],
            code,
        );
    }
    assert!(!rejected.exists());
    let out = root.join("oversize.csv");
    call(
        &[
            "sample",
            "--artifact",
            s(&artifact),
            "--rows",
            "100",
            "--max-cells",
            "4",
            "--out",
            s(&out),
        ],
        5,
    );
    assert!(!out.exists());
}
#[cfg(unix)]
#[test]
fn special_files_and_symlink_outputs_are_refused() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let root = root("special-files");
    let source = root.join("source.json");
    let link = root.join("link.json");
    fs::write(&source, "SECRET_HEADER").unwrap();
    symlink(&source, &link).unwrap();
    call(&["inspect", "--artifact", s(&link)], 6);
    call(&["doctor", "--out", s(&link)], 2);
    call(&["inspect", "--artifact", "/dev/zero"], 2);
    assert_eq!(fs::read_to_string(&source).unwrap(), "SECRET_HEADER");
    let out = root.join("doctor.json");
    call(&["doctor", "--out", s(&out)], 0);
    assert_eq!(
        fs::metadata(out).unwrap().permissions().mode() & 0o777,
        0o600
    );
}
#[test]
fn all_entrypoint_sections_stay_below_shape_limit() {
    fn check(directory: &Path) {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                check(&path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                assert!(
                    fs::read_to_string(&path).unwrap().lines().count() < 500,
                    "{} exceeds shape limit",
                    path.display()
                );
            }
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    check(&manifest.join("rust/main"));
    assert!(
        fs::read_to_string(manifest.join("rust/main.rs"))
            .unwrap()
            .lines()
            .count()
            < 500
    );
}
