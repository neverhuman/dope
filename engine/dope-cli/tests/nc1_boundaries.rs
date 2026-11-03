use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
fn directory() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "dope-nc1-boundaries-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}
fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}
fn write(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dope"))
        .args(["--threads", "1"])
        .args(args)
        .env("PATH", "")
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHON")
        .env_remove("VIRTUAL_ENV")
        .output()
        .unwrap()
}
fn table(source: &str, ids: &[u64], feature: &[f64], target: &[f64], category: &[&str]) -> Value {
    json!({"source_sha256":source.repeat(64),"row_ids":ids,
    "columns":[
        {"name":"CANARY_PRIVATE_HEADER_X","values":feature.iter().map(|v|json!({"kind":"number","value":v})).collect::<Vec<_>>()},
        {"name":"CANARY_PRIVATE_TARGET_LABEL","values":target.iter().map(|v|json!({"kind":"number","value":v})).collect::<Vec<_>>()},
        {"name":"CANARY_PRIVATE_CATEGORY_HEADER","values":category.iter().map(|v|json!({"kind":"category","value":v})).collect::<Vec<_>>()}
    ]})
}
fn fit_request() -> Value {
    json!({"table":table("a", &[13,33,73,103], &[0.,2.,4.,1.0e300], &[10.,20.,40.,1.0e300], &["CANARY_PRIVATE_CAT_A","CANARY_PRIVATE_CAT_B","CANARY_PRIVATE_CAT_A","CANARY_PRIVATE_CAT_QUERY"]),
      "declared":{"features":[{"column":0,"kind":"continuous"},{"column":2,"kind":"nominal"}],"target":{"column":1,"task":"regression"}},
      "fit_rows":[73,13,33]})
}

#[test]
fn actual_normalize_cli_private_fit_public_projection_query_roles_and_tail() {
    let root = directory();
    let request = root.join("fit.json");
    let out = root.join("fit-out");
    write(&request, &fit_request());
    let result = run(&[
        "normalize",
        "fit",
        "--request",
        path(&request),
        "--out",
        path(&out),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8_lossy(&result.stdout);
    for canary in [
        "CANARY_PRIVATE_HEADER_X",
        "CANARY_PRIVATE_TARGET_LABEL",
        "CANARY_PRIVATE_CAT_A",
        "1e300",
        "1e+300",
    ] {
        assert!(
            !stdout.contains(canary),
            "CLI stdout disclosed planted private canary"
        );
    }
    let private_card = std::fs::read(out.join("ScaleCard.private.json")).unwrap();
    let private = String::from_utf8_lossy(&private_card);
    for canary in [
        "CANARY_PRIVATE_HEADER_X",
        "CANARY_PRIVATE_TARGET_LABEL",
        "CANARY_PRIVATE_CAT_A",
    ] {
        assert!(
            private.contains(canary),
            "canary was not planted in successful private artifact"
        );
    }
    let private_value: Value = serde_json::from_slice(&private_card).unwrap();
    assert_eq!(private_value["features"][0]["numeric"]["hi"], 4.0);
    let public_bytes = std::fs::read(out.join("schema.public.json")).unwrap();
    let public = String::from_utf8(public_bytes.clone()).unwrap();
    for canary in [
        "CANARY_PRIVATE_HEADER_X",
        "CANARY_PRIVATE_TARGET_LABEL",
        "CANARY_PRIVATE_CATEGORY_HEADER",
        "CANARY_PRIVATE_CAT_A",
        "1e300",
        "1e+300",
    ] {
        assert!(
            !public.contains(canary),
            "public schema disclosed planted private canary"
        );
    }
    let parsed: Value = serde_json::from_slice(&public_bytes).unwrap();
    assert_eq!(parsed["target"], "regression");
    assert_eq!(parsed["columns"], 4); // numeric + A/B/OTHER nominal group
    for key in [
        "names", "features", "declared", "lo", "hi", "median", "extrema",
    ] {
        assert!(parsed.get(key).is_none());
    }

    let classification = root.join("classification.json");
    let numeric_values: Vec<Value> = [0., 1., 2., 3.]
        .iter()
        .map(|v| json!({"kind":"number","value":v}))
        .collect();
    let label_values: Vec<Value> = [
        "CANARY_TARGET_NEG",
        "CANARY_TARGET_POS",
        "CANARY_TARGET_NEG",
        "CANARY_TARGET_POS",
    ]
    .iter()
    .map(|v| json!({"kind":"category","value":v}))
    .collect();
    let unused_values: Vec<Value> = ["u", "u", "u", "u"]
        .iter()
        .map(|v| json!({"kind":"category","value":v}))
        .collect();

    write(
        &classification,
        &json!({
            "table":{"source_sha256":"c".repeat(64),"row_ids":[17,37,77,117],"columns":[
                {"name":"class-feature","values":numeric_values},
                {"name":"private-class-target","values":label_values},
                {"name":"unused-after-target","values":unused_values}
            ]},
            "declared":{"features":[{"column":0,"kind":"continuous"}],"target":{"column":1,"task":"binary","positive_label":"CANARY_TARGET_POS"}},
            "fit_rows":[17,37,77,117]
        }),
    );
    let class_out = root.join("class-out");
    let class_result = run(&[
        "normalize",
        "fit",
        "--request",
        path(&classification),
        "--out",
        path(&class_out),
    ]);
    assert!(
        class_result.status.success(),
        "{}",
        String::from_utf8_lossy(&class_result.stderr)
    );
    let class_private = std::fs::read_to_string(class_out.join("ScaleCard.private.json")).unwrap();
    assert!(class_private.contains("CANARY_TARGET_POS"));
    assert!(class_private.contains("CANARY_TARGET_NEG"));
    let class_public = std::fs::read_to_string(class_out.join("schema.public.json")).unwrap();
    assert!(!class_public.contains("CANARY_TARGET_POS"));
    assert!(!class_public.contains("CANARY_TARGET_NEG"));
    assert!(!String::from_utf8_lossy(&class_result.stdout).contains("CANARY_TARGET"));

    let query = json!({"table":{"source_sha256":"b".repeat(64),"row_ids":[901,401],"columns":[
        {"name":"CANARY_PRIVATE_CATEGORY_HEADER","values":[{"kind":"category","value":"CANARY_PRIVATE_CAT_QUERY"},{"kind":"category","value":"CANARY_PRIVATE_CAT_A"}]},
        {"name":"CANARY_PRIVATE_HEADER_X","values":[{"kind":"number","value":6.},{"kind":"number","value":2.}]}
    ]},"rows":[401,901],"features_only":true});
    let query_file = root.join("query.json");
    write(&query_file, &query);
    let query_out = root.join("query-out");
    let applied = run(&[
        "normalize",
        "apply",
        "--card",
        path(&out.join("ScaleCard.private.json")),
        "--request",
        path(&query_file),
        "--out",
        path(&query_out),
    ]);
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stderr)
    );
    let matrix: Value =
        serde_json::from_slice(&std::fs::read(query_out.join("matrix.unit.private.json")).unwrap())
            .unwrap();
    assert_eq!(matrix["row_ids"], json!([401, 901]));
    assert!(matrix["y"].is_null());
    assert_eq!(matrix["x"].as_array().unwrap().len(), 2);
    assert!(matrix["x"][1][0].as_f64().unwrap() > 1.0);

    let full = root.join("full.json");
    write(
        &full,
        &json!({"table":table("a", &[13,33,73,103], &[0.,2.,4.,1.0e300], &[10.,20.,40.,1.0e300], &["CANARY_PRIVATE_CAT_A","CANARY_PRIVATE_CAT_B","CANARY_PRIVATE_CAT_A","CANARY_PRIVATE_CAT_QUERY"]),"rows":[103],"features_only":false}),
    );
    let full_out = root.join("full-out");
    let full_apply = run(&[
        "normalize",
        "apply",
        "--card",
        path(&out.join("ScaleCard.private.json")),
        "--request",
        path(&full),
        "--out",
        path(&full_out),
    ]);
    assert!(
        full_apply.status.success(),
        "{}",
        String::from_utf8_lossy(&full_apply.stderr)
    );
    let tail: Value =
        serde_json::from_slice(&std::fs::read(full_out.join("matrix.unit.private.json")).unwrap())
            .unwrap();
    assert!(tail["y"][0].as_f64().unwrap() > 1.0);
    assert_eq!(tail["x"][0].as_array().unwrap().len(), 4);
    // CLI retains its existing structured x/y wire. SDK tests separately exercise
    // the actual native consumer target-last view, without inventing a CLI wire.

    let refused = root.join("refused.json");
    let mut invalid = fit_request();
    invalid["declared"]["features"][0]["column"] = json!(1); // target role collision
    write(&refused, &invalid);
    let bad = run(&[
        "normalize",
        "fit",
        "--request",
        path(&refused),
        "--out",
        path(&root.join("bad-out")),
    ]);
    assert!(!bad.status.success());
    let stderr = String::from_utf8_lossy(&bad.stderr);
    assert!(stderr.contains("invalid_feature_role"));
    for canary in [
        "CANARY_PRIVATE_HEADER_X",
        "CANARY_PRIVATE_TARGET_LABEL",
        "CANARY_PRIVATE_CAT_A",
        "1e300",
        "1e+300",
    ] {
        assert!(
            !stderr.contains(canary),
            "semantic error disclosed planted private canary"
        );
    }
    let malformed = root.join("malformed.json");
    std::fs::write(
        &malformed,
        br#"{"CANARY_PRIVATE_PARSE":"CANARY_PRIVATE_VALUE""#,
    )
    .unwrap();
    let parse = run(&[
        "normalize",
        "fit",
        "--request",
        path(&malformed),
        "--out",
        path(&root.join("parse-out")),
    ]);
    assert!(!parse.status.success());
    let parse_stderr = String::from_utf8_lossy(&parse.stderr);
    assert!(parse_stderr.contains("invalid_normalization_request"));
    assert!(!parse_stderr.contains("CANARY_PRIVATE"));
}
