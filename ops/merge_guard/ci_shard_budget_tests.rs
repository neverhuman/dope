use super::*;
const POLICY: &str = include_str!("../../agent/cost-budget.toml");
const CI_GRAPH: &str = "name: CI\non:\n  pull_request:\njobs:\n  cpu-native:\n    timeout-minutes: 90\n  cpu-compat:\n    timeout-minutes: 120\n  cpu-transfer-paper:\n    timeout-minutes: 120\n  ci-shard-portable-arm64:\n    timeout-minutes: 30\n  cpu:\n    timeout-minutes: 20\n  paper-clean:\n    timeout-minutes: 60\n  security:\n    timeout-minutes: 60\n";
const GPU: &str = "name: GPU\non:\n  workflow_dispatch:\npermissions:\n  contents: read\njobs:\n  gpu:\n    timeout-minutes: 120\n";
fn sources() -> Sources<'static> {
    Sources {
        ci: CI_GRAPH,
        audit: "jobs:\n  jankurai-audit:\n    timeout-minutes: 60\n",
        gpu_workflow: GPU,
        neural: "let deadline = Duration::from_secs(600);",
        train: "const GPU_MEMORY_LIMIT_BYTES: u128 = 16 * 1024 * 1024 * 1024;",
        fitness: "const MAX_ROWS: usize = 128;\nconst MAX_DIMENSIONS: usize = 32;",
        gpu_script: "test -n \"${SEALED_CORPUS_DIR:-}\"; test -n \"${SEALED_ARTIFACT:-}\";",
    }
}
#[test]
fn admitted_seven_role_graph_and_unchanged_legacy_bounds_pass() {
    assert_eq!(verify_sources(POLICY, &sources()), Ok(()));
}
#[test]
fn old_three_job_graph_extra_missing_duplicate_and_unbounded_roles_refuse() {
    let old = "jobs:\n  cpu:\n    timeout-minutes: 120\n  paper-clean:\n    timeout-minutes: 60\n  security:\n    timeout-minutes: 60\n";
    let extra = format!("{CI_GRAPH}  surprise:\n    timeout-minutes: 1\n");
    let flow = format!("{CI_GRAPH}  surprise: {{}}\n");
    let duplicate = format!("{CI_GRAPH}  cpu:\n    timeout-minutes: 20\n");
    let missing = CI_GRAPH.replace("  cpu-native:\n    timeout-minutes: 90\n", "");
    for bad in [old, &extra, &flow, &duplicate, &missing] {
        let mut input = sources();
        input.ci = bad;
        assert!(verify_sources(POLICY, &input).is_err());
    }
    for value in [
        "0",
        "91",
        "18446744073709551616",
        "\"90\"",
        "${{ 90 }}",
        "-1",
    ] {
        let bad = CI_GRAPH.replacen(
            "timeout-minutes: 90",
            &format!("timeout-minutes: {value}"),
            1,
        );
        let mut input = sources();
        input.ci = &bad;
        assert!(verify_sources(POLICY, &input).is_err(), "{value}");
    }
}
#[test]
fn missing_duplicate_and_step_timeout_declarations_refuse() {
    for bad in [
        CI_GRAPH.replace("    timeout-minutes: 90\n", ""),
        CI_GRAPH.replace(
            "    timeout-minutes: 90",
            "    timeout-minutes: 90\n    timeout-minutes: 90",
        ),
        format!("{CI_GRAPH}    steps:\n      - timeout-minutes: 1\n"),
        format!("{CI_GRAPH}      timeout-minutes: 1\n"),
    ] {
        let mut input = sources();
        input.ci = &bad;
        assert!(verify_sources(POLICY, &input).is_err());
    }
}
#[test]
fn workflow_dispatch_is_the_only_admitted_gpu_trigger() {
    for changed in [
        GPU.replace("  workflow_dispatch:", "  schedule:"),
        GPU.replace("  workflow_dispatch:", "  workflow_dispatch:\n  push:"),
        GPU.replace("on:\n  workflow_dispatch:", "on: [workflow_dispatch, push]"),
    ] {
        let mut input = sources();
        input.gpu_workflow = &changed;
        assert!(verify_sources(POLICY, &input).is_err());
    }
}
#[test]
fn audit_and_manual_gpu_quotas_stay_exact() {
    let mut input = sources();
    input.audit = "jobs:\n  jankurai-audit:\n    timeout-minutes: 61\n";
    assert!(verify_sources(POLICY, &input).is_err());
    input = sources();
    input.gpu_workflow = "on:\n  workflow_dispatch:\njobs:\n  gpu:\n    timeout-minutes: 121\n";
    assert!(verify_sources(POLICY, &input).is_err());
}
#[test]
fn every_preexisting_source_kill_switch_is_required() {
    let mut input = sources();
    input.neural = "Duration::from_secs(601)";
    assert_eq!(verify_sources(POLICY, &input), Err("budget_neural_seconds"));
    input = sources();
    input.train = "GPU_MEMORY_LIMIT_BYTES: u128 = 32 * 1024 * 1024 * 1024";
    assert_eq!(verify_sources(POLICY, &input), Err("budget_gpu_memory"));
    for fitness in [
        "const MAX_ROWS: usize = 129; const MAX_DIMENSIONS: usize = 32;",
        "const MAX_ROWS: usize = 128; const MAX_DIMENSIONS: usize = 33;",
    ] {
        input = sources();
        input.fitness = fitness;
        assert_eq!(verify_sources(POLICY, &input), Err("budget_fitness_bounds"));
    }
    for script in [
        "test -n \"${SEALED_CORPUS_DIR:-}\"",
        "test -n \"${SEALED_ARTIFACT:-}\"",
    ] {
        input = sources();
        input.gpu_script = script;
        assert_eq!(verify_sources(POLICY, &input), Err("budget_sealed_guards"));
    }
}
#[test]
fn malformed_or_unsupported_toml_never_becomes_a_partial_policy() {
    for bad in [
        "version = 1\nversion = 1\n[admission]",
        "version = 01\n[admission]",
        "version = -1\n[admission]",
        "version = 1.0\n[admission]",
        "version = inf\n[admission]",
        "version = 18446744073709551616\n[admission]",
        "version = [1]\n[admission]",
        "version = { n=1 }\n[admission]",
        "version = 1\n[other]",
        "version = 1\n[admission]\n[admission]",
        "version = 1\n[admission]\nx.y=true",
        "version = 1\n[admission]\n\"x\"=true",
        "retry_rule = \"unterminated\n[admission]",
    ] {
        assert!(profile::parse(bad).is_err(), "{bad}");
    }
}
#[test]
fn exact_policy_keys_values_and_admission_flags_are_not_optional() {
    for bad in [
        POLICY.replace("cpu_native_minutes = 90", "cpu_native_minutes = 91"),
        POLICY.replace(
            "single_neural_fit_seconds = 600",
            "single_neural_fit_seconds = 601",
        ),
        POLICY.replace(
            "gpu_memory_bytes = 17179869184",
            "gpu_memory_bytes = 34359738368",
        ),
        POLICY.replace("fitness_sample_rows = 128", "fitness_sample_rows = 129"),
        POLICY.replace(
            "fitness_sample_dimensions = 32",
            "fitness_sample_dimensions = 33",
        ),
        POLICY.replace("require_pinned_cuda = true", "require_pinned_cuda = false"),
        POLICY.replace(
            "require_sealed_corpus = true",
            "require_sealed_corpus = false",
        ),
        POLICY.replace(
            "require_candidate_artifact = true",
            "require_candidate_artifact = false",
        ),
        POLICY.replace("require_candidate_artifact = true", ""),
        format!("{POLICY}\nunknown = true\n"),
    ] {
        assert!(verify_sources(&bad, &sources()).is_err());
    }
}
#[test]
fn quoted_hash_is_preserved_and_comment_is_outside_the_string() {
    let parsed = profile::parse("text = \"a#b\" # trailing\n[admission]\nflag=true").unwrap();
    assert_eq!(parsed.get("text"), Some(&Scalar::Text("a#b".into())));
}
#[test]
fn recursive_split_sources_are_read_and_missing_paths_refuse() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Owned(std::path::PathBuf);
    impl Drop for Owned {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let parent = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/budget-tests");
    fs::create_dir_all(&parent).unwrap();
    let path = parent.join(format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    let owned = Owned(path);
    fs::write(owned.0.join("unit.rs"), "// root").unwrap();
    fs::create_dir_all(owned.0.join("unit/nested")).unwrap();
    fs::write(
        owned.0.join("unit/nested/bound.rs"),
        "Duration::from_secs(600)",
    )
    .unwrap();
    assert!(
        tree(&owned.0, "unit.rs")
            .unwrap()
            .contains("Duration::from_secs(600)")
    );
    assert!(tree(&owned.0, "missing.rs").is_err());
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(owned.0.join("unit.rs"), owned.0.join("unit/link.rs")).unwrap();
        assert_eq!(tree(&owned.0, "unit.rs"), Err("budget_source_path"));
    }
}
