//! Tracked-source shape checks; no product or policy authority.
use std::path::Path;
use std::process::Command;

#[test]
fn tracked_source_length_caps_and_artifact_exclusion() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(&root)
        .output()
        .expect("tracked source inventory");
    assert!(output.status.success());
    for name in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|s| !s.is_empty())
    {
        let name = std::str::from_utf8(name).expect("UTF-8 repository path");
        assert!(
            !name.starts_with(".agent/") && !name.starts_with("target/"),
            "tracked artifact: {name}"
        );
        let python = name.ends_with(".py") && !super::PYTHON_LINE_EXCEPTIONS.contains(&name);
        let rust = name.ends_with(".rs")
            && !name.starts_with("ops/vendor/")
            && !name.starts_with("tests/")
            && !name.ends_with("tests.rs")
            && !name.contains("/tests/");
        if python || rust {
            let source = std::fs::read_to_string(root.join(name)).expect("tracked source");
            assert!(source.lines().count() < 500, "source length cap: {name}");
        }
    }
}
