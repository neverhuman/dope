//! Source-bound admission for the portable ARM proof and its required CPU gate.
pub(super) fn verify_portable(source: &str) -> Result<(), &'static str> {
    const JOB: &str = "\n  ci-shard-portable-arm64:\n";
    if source.match_indices(JOB).count() != 1 {
        return Err("portable_workflow_invalid");
    }
    let block = source.split_once(JOB).ok_or("portable_workflow_invalid")?.1;
    let block = block
        .lines()
        .take_while(|line| line.is_empty() || line.starts_with("    "))
        .collect::<Vec<_>>()
        .join("\n");
    if !block.contains("    if: github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository")
        || !block.contains("labels: [self-hosted, Linux, ARM64, lan-ci, ubuntu24]")
        || !block.contains("cargo test --manifest-path ops/merge_guard/Cargo.toml")
        || !block.contains("--bin check-frozen-evidence --bin ci-shard")
        || block.contains("continue-on-error: true")
    {
        return Err("portable_workflow_invalid");
    }
    let cpu = source
        .split_once("\n  cpu:\n")
        .ok_or("portable_workflow_invalid")?
        .1;
    let cpu = cpu
        .lines()
        .take_while(|line| line.is_empty() || line.starts_with("    "))
        .collect::<Vec<_>>()
        .join("\n");
    if !cpu.contains("needs: [cpu-native, cpu-compat, cpu-transfer-paper, ci-shard-portable-arm64]")
        || !cpu.contains("PORTABLE_RESULT: ${{ needs.ci-shard-portable-arm64.result }}")
        || !cpu.contains("[[ \"$PORTABLE_RESULT\" == success ]]")
    {
        return Err("portable_workflow_invalid");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = include_str!("../../.github/workflows/ci.yml");
    #[test]
    fn portable_gate_requires_fork_guard_actual_tests_and_aggregate_dependency() {
        assert_eq!(verify_portable(SOURCE), Ok(()));
        for (before, after) in [
            (
                "ci-shard-portable-arm64:\n    if:",
                "ci-shard-portable-arm64:\n    removed-if:",
            ),
            (
                "cpu-transfer-paper, ci-shard-portable-arm64]",
                "cpu-transfer-paper]",
            ),
            (
                "--bin check-frozen-evidence --bin ci-shard",
                "--bin ci-shard",
            ),
            ("[[ \"$PORTABLE_RESULT\" == success ]]", "true"),
        ] {
            assert!(SOURCE.contains(before));
            assert_eq!(
                verify_portable(&SOURCE.replacen(before, after, 1)),
                Err("portable_workflow_invalid")
            );
        }
    }
}
