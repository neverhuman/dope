use super::*;

const HEAD: &str = "0123456789abcdef0123456789abcdef01234567";
fn fake_summary(id: &str, kind: &str) -> String {
    match (kind, id) {
        ("cargo-test", _) => "test result: ok. 3 passed; 0 failed; 0 ignored; finished\n".into(),
        ("unittest", "engine-benchmark") => "Ran 5 tests in 0.001s\n\nOK\n".into(),
        ("unittest", "u04-admissions") => "Ran 7 tests in 0.001s\n\nOK\n".into(),
        ("unittest", "review-fixes") => "Ran 24 tests in 0.001s\n\nOK\nRan 1 test in 0.001s\n\nOK\nRan 2 tests in 0.001s\n\nOK\n".into(),
        ("unittest-required", "next-version-reference") => "Ran 11 tests in 0.001s\n\nOK\n".into(),
        ("unittest", _) => "Ran 3 tests in 0.001s\n\nOK\n".into(),
        ("pytest", _) => "3 passed in 0.01s\n".into(),
        ("readiness", _) => "CPU tool readiness: 12 real-entry refusals passed\n".into(),
        _ => String::new(),
    }
}
fn log(shard: &str) -> Vec<u8> {
    let mut out = String::new();
    for (_owner, id, kind) in inventory::RUNTIME_COMMANDS
        .iter()
        .filter(|(owner, _, _)| *owner == shard)
    {
        let summary = fake_summary(id, kind);
        out.push_str(&format!(
            "@@CI-BEGIN {id} {kind}@@\n{summary}@@CI-END {id} 0@@\n"
        ));
    }
    out.into_bytes()
}
fn block(id: &str, kind: &str, body: &str, status: u8) -> String {
    format!("@@CI-BEGIN {id} {kind}@@\n{body}@@CI-END {id} {status}@@\n")
}
fn row(shard: &str) -> (String, Value, Vec<u8>) {
    let bytes = log(shard);
    let receipt = make_receipt(shard, HEAD, 41, 2, "success", &bytes).unwrap();
    (shard.into(), receipt, bytes)
}
fn good_rows() -> Vec<(String, Value, Vec<u8>)> {
    SHARDS.iter().map(|shard| row(shard)).collect()
}
fn good_needs() -> Value {
    json!({"cpu-native":"success","cpu-compat":"success","cpu-transfer-paper":"success"})
}

#[test]
fn accepts_complete_fresh_receipts_and_sums_only_real_test_framework_counts() {
    let rows = good_rows();
    assert!(aggregate(&good_needs(), &rows, HEAD, 41, 2).is_ok());
    let native = parse_log(std::str::from_utf8(&rows[0].2).unwrap()).unwrap();
    assert!(
        native
            .iter()
            .filter(|command| command.kind == "check")
            .all(|command| command.passed == 0 && command.failed == 0)
    );
    let compat = parse_log(std::str::from_utf8(&rows[1].2).unwrap()).unwrap();
    assert_eq!(
        compat
            .iter()
            .find(|command| command.id == "engine-benchmark")
            .map(|c| c.passed),
        Some(5)
    );
    assert_eq!(
        compat
            .iter()
            .find(|command| command.id == "u04-admissions")
            .map(|c| c.passed),
        Some(7)
    );
    assert_eq!(
        compat
            .iter()
            .find(|command| command.id == "next-version-reference")
            .map(|c| c.passed),
        Some(11)
    );
    let transfer = parse_log(std::str::from_utf8(&rows[2].2).unwrap()).unwrap();
    assert_eq!(
        transfer
            .iter()
            .find(|command| command.id == "cpu-tool-readiness")
            .map(|c| c.passed),
        Some(12)
    );
    assert_eq!(
        transfer
            .iter()
            .find(|command| command.id == "review-fixes")
            .map(|c| c.passed),
        Some(27)
    );
    let expected: u64 = rows
        .iter()
        .map(|(_, receipt, _)| receipt["tests_executed"].as_u64().unwrap())
        .sum();
    assert!(expected > 0);
    assert_eq!(
        expected,
        rows.iter()
            .map(|(_, _, log)| {
                parse_log(std::str::from_utf8(log).unwrap())
                    .unwrap()
                    .iter()
                    .filter(|command| is_test_kind(&command.kind))
                    .map(|command| command.passed + command.failed)
                    .sum::<u64>()
            })
            .sum::<u64>()
    );
}
#[test]
fn refuses_missing_duplicate_and_unknown_receipts() {
    let mut rows = good_rows();
    rows.pop();
    assert_eq!(
        aggregate(&good_needs(), &rows, HEAD, 41, 2),
        Err("receipt_set_invalid")
    );
    let mut rows = good_rows();
    rows[1] = rows[0].clone();
    assert_eq!(
        aggregate(&good_needs(), &rows, HEAD, 41, 2),
        Err("receipt_set_invalid")
    );
    let mut rows = good_rows();
    rows[0].0 = "unexpected".into();
    assert_eq!(
        aggregate(&good_needs(), &rows, HEAD, 41, 2),
        Err("receipt_set_invalid")
    );
}
#[test]
fn refuses_failed_skipped_cancelled_and_extra_needs() {
    let mut needs = good_needs();
    needs["cpu-compat"] = json!("failure");
    assert_eq!(
        aggregate(&needs, &good_rows(), HEAD, 41, 2),
        Err("worker_not_success")
    );
    let mut needs = good_needs();
    needs["cpu-native"] = json!("skipped");
    assert_eq!(
        aggregate(&needs, &good_rows(), HEAD, 41, 2),
        Err("worker_not_success")
    );
    let mut needs = good_needs();
    needs["unexpected"] = json!("success");
    assert_eq!(
        aggregate(&needs, &good_rows(), HEAD, 41, 2),
        Err("worker_not_success")
    );
}
#[test]
fn refuses_stale_head_run_attempt_and_swapped_log() {
    let mut rows = good_rows();
    rows[0].1["head_sha"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    assert_eq!(
        aggregate(&good_needs(), &rows, HEAD, 41, 2),
        Err("receipt_identity_mismatch")
    );
    let mut rows = good_rows();
    rows[1].1["run_id"] = json!(42);
    assert_eq!(
        aggregate(&good_needs(), &rows, HEAD, 41, 2),
        Err("receipt_identity_mismatch")
    );
    let mut rows = good_rows();
    rows[2].1["run_attempt"] = json!(1);
    assert_eq!(
        aggregate(&good_needs(), &rows, HEAD, 41, 2),
        Err("receipt_identity_mismatch")
    );
    let mut rows = good_rows();
    rows[0].2.push(b' ');
    assert_eq!(
        aggregate(&good_needs(), &rows, HEAD, 41, 2),
        Err("receipt_identity_mismatch")
    );
}
#[test]
fn rejects_missing_extra_and_reordered_non_test_guard_commands() {
    let good = String::from_utf8(log("cpu-native")).unwrap();
    let first_end = good.find("@@CI-END frozen-evidence 0@@\n").unwrap()
        + "@@CI-END frozen-evidence 0@@\n".len();
    let missing = &good[first_end..];
    assert_eq!(
        make_receipt("cpu-native", HEAD, 41, 2, "success", missing.as_bytes()),
        Err("command_coverage_mismatch")
    );
    let extra = good
        .replace("@@CI-BEGIN frozen-evidence", "@@CI-BEGIN surprise")
        .replace("@@CI-END frozen-evidence", "@@CI-END surprise");
    assert_eq!(
        make_receipt("cpu-native", HEAD, 41, 2, "success", extra.as_bytes()),
        Err("command_order_or_kind_mismatch")
    );
    let second_end =
        good.find("@@CI-END ci-shard-build 0@@\n").unwrap() + "@@CI-END ci-shard-build 0@@\n".len();
    let first = &good[..first_end];
    let second = &good[first_end..second_end];
    let tail = &good[second_end..];
    let reordered = format!("{second}{first}{tail}");
    assert_ne!(reordered, good);
    assert_eq!(
        make_receipt("cpu-native", HEAD, 41, 2, "success", reordered.as_bytes()),
        Err("command_order_or_kind_mismatch")
    );
}
#[test]
fn rejects_zero_test_command_even_when_other_tests_pass() {
    let good = String::from_utf8(log("cpu-native")).unwrap();
    let zero = good.replacen(
        "@@CI-BEGIN landmark-cache unittest@@\nRan 3 tests in 0.001s\n\nOK",
        "@@CI-BEGIN landmark-cache unittest@@\nRan 0 tests in 0.001s\n\nOK",
        1,
    );
    assert_eq!(
        make_receipt("cpu-native", HEAD, 41, 2, "success", zero.as_bytes()),
        Err("zero_test_command")
    );
}
#[test]
fn a_failed_check_records_no_fabricated_test_count() {
    let failed = block("frozen-evidence", "check", "refused\n", 7);
    let receipt = make_receipt("cpu-native", HEAD, 41, 2, "failed", failed.as_bytes()).unwrap();
    assert_eq!(receipt["tests_executed"], 0);
    assert_eq!(receipt["commands"][0]["exit_code"], 7);
    assert_eq!(receipt["commands"][0]["tests_failed"], 0);
}
#[test]
fn framework_parsers_count_all_summaries_and_exclude_skipped_or_ignored_tests() {
    assert_eq!(
        summary("pytest", &["no test result"]),
        Err("test_summary_missing")
    );
    assert_eq!(
        summary("unittest", &["Ran 2 tests", "FAILED (failures=1)"]),
        Ok((1, 1))
    );
    assert_eq!(
        summary(
            "unittest",
            &[
                "Ran 4 tests",
                "OK (skipped=1)",
                "Ran 3 tests",
                "FAILED (errors=1)"
            ]
        ),
        Ok((5, 1))
    );
    assert_eq!(
        summary("pytest", &["2 passed, 1 failed, 4 skipped in 0.2s"]),
        Ok((2, 1))
    );
    assert_eq!(
        summary(
            "cargo-test",
            &["test result: ok. 1 passed; 0 failed; 4 ignored; done"]
        ),
        Ok((1, 0))
    );
}
#[test]
fn failed_test_command_without_summary_has_zero_tests_not_a_fake_failure_count() {
    let result =
        parse_log("@@CI-BEGIN failed-test unittest@@\ntraceback\n@@CI-END failed-test 1@@\n")
            .unwrap();
    assert_eq!(
        (result[0].passed, result[0].failed, result[0].exit_code),
        (0, 0, 1)
    );
}
#[test]
fn receipt_directory_rejects_missing_duplicate_and_extra_files() {
    let mut files = Vec::new();
    for shard in SHARDS {
        files.push(format!("{shard}.json"));
        files.push(format!("{shard}.log"));
    }
    assert_eq!(validate_filenames(&files), Ok(()));
    let mut missing = files.clone();
    missing.pop();
    assert_eq!(validate_filenames(&missing), Err("receipt_set_invalid"));
    let mut extra = files.clone();
    extra.push("cpu-other.json".into());
    assert_eq!(validate_filenames(&extra), Err("receipt_set_invalid"));
    let mut duplicate = files.clone();
    duplicate.push(files[0].clone());
    assert_eq!(validate_filenames(&duplicate), Err("receipt_set_invalid"));
}
#[test]
fn malformed_or_nonobject_receipts_refuse() {
    assert_eq!(parse_receipt(b"{"), Err("receipt_invalid"));
    assert_eq!(parse_receipt(b"[]"), Err("receipt_invalid"));
}

#[test]
fn refuses_test_command_with_wrong_framework_even_with_valid_summary() {
    let good = String::from_utf8(log("cpu-native")).unwrap();
    let wrong = good
        .replace("landmark-cache unittest", "landmark-cache cargo-test")
        .replace(
            "Ran 3 tests in 0.001s\n\nOK",
            "test result: ok. 3 passed; 0 failed; 0 ignored; finished",
        );
    assert_eq!(
        make_receipt("cpu-native", HEAD, 41, 2, "success", wrong.as_bytes()),
        Err("command_order_or_kind_mismatch")
    );
}

#[test]
fn framework_failures_refuse_even_when_a_wrapper_reports_exit_zero() {
    for (shard, needle, replacement) in [
        (
            "cpu-native",
            "test result: ok. 3 passed; 0 failed; 0 ignored; finished",
            "test result: FAILED. 2 passed; 1 failed; 0 ignored; finished",
        ),
        (
            "cpu-native",
            "Ran 3 tests in 0.001s\n\nOK",
            "Ran 3 tests in 0.001s\n\nFAILED (failures=1)",
        ),
        (
            "cpu-native",
            "Ran 3 tests in 0.001s\n\nOK",
            "Ran 3 tests in 0.001s\n\nFAILED (unexpected successes=1)",
        ),
        (
            "cpu-compat",
            "3 passed in 0.01s",
            "2 passed, 1 failed in 0.01s",
        ),
    ] {
        let good = String::from_utf8(log(shard)).unwrap();
        assert!(good.contains(needle));
        let failed = good.replacen(needle, replacement, 1);
        assert_eq!(
            make_receipt(shard, HEAD, 41, 2, "success", failed.as_bytes()),
            Err("test_command_failed")
        );
    }
}
#[test]
fn ignored_or_expected_failure_only_commands_cannot_satisfy_required_tests() {
    let good = String::from_utf8(log("cpu-native")).unwrap();
    for body in [
        "Ran 3 tests in 0.001s\n\nOK (skipped=3)",
        "Ran 3 tests in 0.001s\n\nOK (expected failures=3)",
    ] {
        let excluded = good.replacen("Ran 3 tests in 0.001s\n\nOK", body, 1);
        assert_eq!(
            make_receipt("cpu-native", HEAD, 41, 2, "success", excluded.as_bytes()),
            Err("zero_test_command")
        );
    }
}
#[test]
fn unexpected_successes_and_unrecognized_failed_results_are_not_passes() {
    assert_eq!(
        summary(
            "unittest",
            &["Ran 3 tests", "FAILED (unexpected successes=1)"]
        ),
        Ok((2, 1))
    );
    assert_eq!(
        summary("unittest", &["Ran 3 tests", "FAILED"]),
        Err("test_summary_invalid")
    );
    assert_eq!(
        summary("unittest", &["Ran 3 tests", "OK (expected failures=2)"]),
        Ok((1, 0))
    );
    assert_eq!(
        summary("pytest", &["1 failed in 0.1s", "2 passed in 0.1s"]),
        Ok((2, 1))
    );
    assert_eq!(
        summary(
            "unittest",
            &[
                "Ran 3 tests",
                "FAILED (expected failures=1, unexpected successes=1)"
            ]
        ),
        Ok((1, 1))
    );
}

#[test]
fn required_next_version_suite_rejects_skip_zero_and_failed_summary_even_with_end_zero() {
    let good = String::from_utf8(log("cpu-compat")).unwrap();
    assert!(make_receipt("cpu-compat", HEAD, 41, 2, "success", good.as_bytes()).is_ok());

    let skipped = good.replace(
        "Ran 11 tests in 0.001s\n\nOK",
        "Ran 11 tests in 0.001s\n\nOK (skipped=1)",
    );
    assert_ne!(skipped, good);
    assert_eq!(
        make_receipt("cpu-compat", HEAD, 41, 2, "success", skipped.as_bytes()),
        Err("required_test_skip_or_expected_failure")
    );

    let expected_failure = good.replace(
        "Ran 11 tests in 0.001s\n\nOK",
        "Ran 11 tests in 0.001s\n\nOK (expected failures=1)",
    );
    assert_ne!(expected_failure, good);
    assert_eq!(
        make_receipt(
            "cpu-compat",
            HEAD,
            41,
            2,
            "success",
            expected_failure.as_bytes()
        ),
        Err("required_test_skip_or_expected_failure")
    );

    let zero = good.replace(
        "Ran 11 tests in 0.001s\n\nOK",
        "Ran 0 tests in 0.001s\n\nOK",
    );
    assert_ne!(zero, good);
    assert_eq!(
        make_receipt("cpu-compat", HEAD, 41, 2, "success", zero.as_bytes()),
        Err("zero_test_command")
    );

    let failed_with_zero_exit = good.replace(
        "Ran 11 tests in 0.001s\n\nOK",
        "Ran 11 tests in 0.001s\n\nFAILED (failures=1)",
    );
    assert_ne!(failed_with_zero_exit, good);
    assert_eq!(
        make_receipt(
            "cpu-compat",
            HEAD,
            41,
            2,
            "success",
            failed_with_zero_exit.as_bytes()
        ),
        Err("test_command_failed")
    );
    assert_eq!(
        summary("unittest", &["Ran 11 tests", "OK (skipped=1)"]),
        Ok((10, 0))
    );
    assert_eq!(
        summary("unittest", &["Ran 11 tests", "OK (expected failures=1)"]),
        Ok((10, 0))
    );
}
