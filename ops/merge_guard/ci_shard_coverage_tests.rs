use super::*;

fn hash(s: &str) -> String {
    sha(&normalize(&s.lines().collect::<Vec<_>>()))
}

#[test]
fn exact_inventory_and_body_hash_match() {
    let body = "echo native-test";
    let h = hash(body);
    let expected = BTreeMap::from([("native-check".into(), ("cpu-native".into(), h.clone()))]);
    let actual = BTreeMap::from([("native-check".into(), ("cpu-native".into(), h))]);
    assert_eq!(compare(&expected, &actual), Ok(()));
}
#[test]
fn missing_extra_duplicate_swapped_and_edited_commands_refuse() {
    let h = hash("echo x");
    let expected = BTreeMap::from([("x".into(), ("cpu-native".into(), h.clone()))]);
    assert_eq!(
        compare(&expected, &BTreeMap::new()),
        Err("command_coverage_mismatch")
    );
    let extra = BTreeMap::from([
        ("x".into(), ("cpu-native".into(), h.clone())),
        ("y".into(), ("cpu-native".into(), h.clone())),
    ]);
    assert_eq!(compare(&expected, &extra), Err("command_coverage_mismatch"));
    let swapped = BTreeMap::from([("x".into(), ("cpu-compat".into(), h.clone()))]);
    assert_eq!(
        compare(&expected, &swapped),
        Err("command_content_mismatch")
    );
    let edited = BTreeMap::from([("x".into(), ("cpu-native".into(), hash("echo changed")))]);
    assert_eq!(compare(&expected, &edited), Err("command_content_mismatch"));
    let h = hash("ci_run x check echo x");
    let duplicate = format!(
        "# ci-command:x {h}\nci_run x check echo x\n# ci-command:x {h}\nci_run x check echo x\n"
    );
    assert_eq!(marked(&duplicate), Err("command_duplicate"));
}
#[test]
fn marker_hash_is_recomputed_from_following_body() {
    let body = "  ci_run x check echo x\n";
    let h = hash(body);
    let source = format!("# ci-command:x {h}\n{body}");
    let parsed = marked(&source).unwrap();
    assert_eq!(
        parsed["x"],
        (
            "check".into(),
            h.clone(),
            h.clone(),
            "ci_run x check echo x".into()
        )
    );
    let tampered = format!("# ci-command:x {h}\nci_run x check echo y\n");
    assert_eq!(marked(&tampered), Err("command_content_mismatch"));
}
#[test]
fn argv_equivalence_handles_env_assignments_and_only_reviewed_bin_addition() {
    assert_eq!(
        same_invocation(
            "PYTHONPATH=. python3 -m unittest sample -v",
            "ci_run sample-tests unittest env PYTHONPATH=. python3 -m unittest sample -v",
            false
        ),
        Ok(true)
    );
    assert_eq!(
        same_invocation(
            "cargo test --locked",
            "ci_run root-cargo-test cargo-test cargo test --locked",
            false
        ),
        Ok(true)
    );
    assert_eq!(
        same_invocation(
            "cargo test --locked --bin alpha",
            "ci_run merge-test cargo-test cargo test --locked --bin alpha --bin ci-shard",
            true
        ),
        Ok(true)
    );
    assert_eq!(
        same_invocation(
            "cargo test --locked",
            "ci_run root-cargo-test cargo-test cargo test --locked --no-run",
            false
        ),
        Ok(false)
    );
    assert_eq!(
        same_invocation(
            "cargo test --locked",
            "ci_run other cargo-test cargo test --locked --bin ci-shard",
            false
        ),
        Ok(false)
    );
    assert_eq!(
        same_invocation(
            "cargo test --locked",
            "ci_run other cargo-test cargo test --locked --bin ci-shard --bin ci-shard",
            true
        ),
        Ok(false)
    );
}
#[test]
fn argv_equivalence_keeps_shell_glob_and_cleanup_semantics() {
    let command = "git diff --exit-code -- a/*.tex";
    assert_eq!(
        same_invocation(
            command,
            "ci_run drift check bash -c 'git diff --exit-code -- a/*.tex'",
            false
        ),
        Ok(true)
    );
    assert_eq!(
        same_invocation(
            "find target -delete || true",
            "ci_run clean check bash -c \"find target -delete || true\"",
            false
        ),
        Ok(true)
    );
}

#[test]
fn compiled_inventory_rejects_omission_and_reassignment() {
    let mut rows: Vec<Value> = super::super::inventory::EXPECTED_COMMANDS.iter().map(|(id, path, start, end)| {
        let shard = id.split_once(':').unwrap().0;
        serde_json::json!({"id": id, "shard": shard, "origin_path": path, "origin_start": start, "origin_end": end})
    }).collect();
    assert_eq!(super::inventory_matches(&rows), Ok(()));
    rows.pop();
    assert_eq!(
        super::inventory_matches(&rows),
        Err("coverage_inventory_mismatch")
    );
    let mut rows: Vec<Value> = super::super::inventory::EXPECTED_COMMANDS.iter().map(|(id, path, start, end)| {
        let shard = id.split_once(':').unwrap().0;
        serde_json::json!({"id": id, "shard": shard, "origin_path": path, "origin_start": start, "origin_end": end})
    }).collect();
    rows[0]["shard"] = Value::String("cpu-compat".into());
    assert_eq!(
        super::inventory_matches(&rows),
        Err("coverage_inventory_mismatch")
    );
}

#[test]
fn complete_packaged_coverage_preimages_and_sources_are_wired() {
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/ci-sharding/command-map.json")).unwrap();
    let manifest_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let frozen_root = manifest_root.join("fixtures/ci-sharding/source");
    let stage_root = manifest_root.join("../ci");
    let workflow = include_str!("../../.github/workflows/ci.yml");
    assert!(verify(&manifest, &frozen_root, &stage_root, workflow).is_ok());
    assert_eq!(
        verify(
            &manifest,
            &frozen_root.join("missing"),
            &stage_root,
            workflow
        ),
        Err("coverage_preimage_missing")
    );
}

#[test]
fn candidate_stage_pins_reject_control_flow_drift() {
    let original = include_bytes!("../ci/shard-runtime.sh");
    assert!(super::pinned_stage_matches("shard-runtime.sh", original));
    let mut changed = original.to_vec();
    changed.push(b' ');
    assert!(!super::pinned_stage_matches("shard-runtime.sh", &changed));
}
