//! Framework result parsing. Ignored, skipped and expected failures are not passes.
use super::fail;
fn count_before(s: &str, word: &str) -> Option<u64> {
    let words: Vec<_> = s.split_ascii_whitespace().collect();
    words
        .iter()
        .position(|w| w.trim_matches(|c: char| !c.is_ascii_alphabetic()) == word)?
        .checked_sub(1)
        .and_then(|i| words[i].parse().ok())
}
fn count_after(s: &str, word: &str) -> Option<u64> {
    let words: Vec<_> = s.split_ascii_whitespace().collect();
    words
        .iter()
        .position(|w| *w == word)?
        .checked_add(1)
        .and_then(|i| words.get(i)?.parse().ok())
}
fn count_eq(s: &str, key: &str) -> Option<u64> {
    let at = s.match_indices(key).find_map(|(at, _)| {
        let previous = s[..at].trim_end().chars().last();
        (at == 0 || matches!(previous, Some('(' | ','))).then_some(at + key.len())
    })?;
    let digits: String = s[at..]
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}
pub(super) fn summary(kind: &str, lines: &[&str]) -> Result<(u64, u64), &'static str> {
    match kind {
        "cargo-test" => {
            let mut total = (0u64, 0u64);
            let mut seen = false;
            for line in lines.iter().filter(|l| l.contains("test result:")) {
                let passed = count_before(line, "passed").ok_or("test_summary_invalid")?;
                let failed = count_before(line, "failed").ok_or("test_summary_invalid")?;
                let _ignored = count_before(line, "ignored").ok_or("test_summary_invalid")?;
                total.0 = total.0.saturating_add(passed);
                total.1 = total.1.saturating_add(failed);
                seen = true;
            }
            if seen {
                Ok(total)
            } else {
                fail("test_summary_missing")
            }
        }
        "unittest" | "unittest-required" => {
            let required = kind == "unittest-required";
            let mut total = (0u64, 0u64);
            let starts: Vec<usize> = lines
                .iter()
                .enumerate()
                .filter_map(|(i, line)| line.trim_start().starts_with("Ran ").then_some(i))
                .collect();
            if starts.is_empty() {
                return fail("test_summary_missing");
            }
            for (index, start) in starts.iter().enumerate() {
                let end = starts.get(index + 1).copied().unwrap_or(lines.len());
                let block = &lines[*start..end];
                let n = count_after(block[0].trim(), "Ran").ok_or("test_summary_invalid")?;
                let ok = block
                    .iter()
                    .any(|line| line.trim() == "OK" || line.trim().starts_with("OK ("));
                let failed = block.iter().find(|line| line.trim().starts_with("FAILED"));
                if ok == failed.is_some() {
                    return fail("test_summary_invalid");
                }
                let (failure_count, errors, skipped) = if let Some(line) = failed {
                    (
                        count_eq(line, "failures=").unwrap_or(0),
                        count_eq(line, "errors=")
                            .unwrap_or(0)
                            .saturating_add(count_eq(line, "unexpected successes=").unwrap_or(0)),
                        count_eq(line, "skipped=").unwrap_or(0),
                    )
                } else {
                    (
                        0,
                        0,
                        block
                            .iter()
                            .find_map(|line| count_eq(line, "skipped="))
                            .unwrap_or(0),
                    )
                };
                let failures = failure_count.saturating_add(errors);
                if required && skipped != 0 {
                    return fail("required_test_skip_or_expected_failure");
                }
                let expected = block
                    .iter()
                    .find_map(|line| count_eq(line, "expected failures="))
                    .unwrap_or(0);
                if required && expected != 0 {
                    return fail("required_test_skip_or_expected_failure");
                }
                if (failed.is_some() && failures == 0)
                    || failures.saturating_add(skipped).saturating_add(expected) > n
                {
                    return fail("test_summary_invalid");
                }
                total.0 = total.0.saturating_add(n - failures - skipped - expected);
                total.1 = total.1.saturating_add(failures);
            }
            Ok(total)
        }
        "readiness" => {
            let count = lines
                .iter()
                .find_map(|line| {
                    let line = line.trim();
                    (line.starts_with("CPU tool readiness:")
                        && line.ends_with("real-entry refusals passed"))
                    .then(|| count_after(line, "readiness:"))
                    .flatten()
                })
                .ok_or("test_summary_missing")?;
            Ok((count, 0))
        }
        "pytest" => {
            let mut total = (0u64, 0u64);
            let mut seen = false;
            for line in lines {
                if line.contains(" passed") || line.contains(" failed") || line.contains(" skipped")
                {
                    let passed = count_before(line, "passed").unwrap_or(0);
                    let failed = count_before(line, "failed").unwrap_or(0);
                    let skipped = count_before(line, "skipped").unwrap_or(0);
                    if passed.saturating_add(failed).saturating_add(skipped) > 0 {
                        total.0 = total.0.saturating_add(passed);
                        total.1 = total.1.saturating_add(failed);
                        seen = true;
                    }
                }
            }
            if seen {
                Ok(total)
            } else {
                fail("test_summary_missing")
            }
        }
        _ => fail("command_kind_invalid"),
    }
}
