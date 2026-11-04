use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
const TRANSFER: &str = include_str!("../../../tests/golden/transfer/retention_90.json");
const DRIVER: &str = include_str!("../../../tests/golden/drivers/v2/v2_additive.json");
const RANK: &str = include_str!("../../../tests/golden/drivers/informative-v1/rank-cases.json");

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let parent = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/golden-strict");
        fs::create_dir_all(&parent).unwrap();
        let root = parent.join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        for component in COMPONENTS {
            fs::create_dir(root.join(component)).unwrap();
        }
        fs::create_dir(root.join("drivers/v2")).unwrap();
        fs::create_dir(root.join("drivers/informative-v1")).unwrap();
        let fixture = Self(root);
        fixture.write("drivers/informative-v1/rank-cases.json", RANK);
        fixture.write("transfer/base.json", TRANSFER);
        fixture
    }
    fn write(&self, path: &str, data: &str) {
        let path = self.0.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, data).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn strict_discovery_replays_nested_v2_and_separate_seven_case_rank_array() {
    let f = Fixture::new();
    f.write("drivers/v2/nested/driver.json", DRIVER);
    let paths = discover_cases(&f.0).unwrap();
    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&f.0.join("drivers/v2/nested/driver.json")));
    assert_eq!(verify_golden_corpus(&f.0).unwrap(), 2);
    let driver = strict_parse_json(DRIVER).unwrap();
    assert_eq!(
        run_case(&driver).unwrap_err().reason,
        "invalid_golden_schema"
    );
    assert!(values_close(
        &run_versioned_case(&driver).unwrap(),
        &driver["expected"],
        GOLDEN_TOLERANCE
    ));
}

#[test]
fn missing_component_and_rank_fixture_cannot_silently_shrink_corpus() {
    let f = Fixture::new();
    fs::remove_dir(f.0.join("aggregate")).unwrap();
    assert_eq!(
        discover_cases(&f.0).unwrap_err().reason,
        "unreadable_golden_directory"
    );
    let f = Fixture::new();
    fs::remove_file(f.0.join("drivers/informative-v1/rank-cases.json")).unwrap();
    assert_eq!(
        discover_cases(&f.0).unwrap_err().reason,
        "missing_rank_fixture"
    );
    let f = Fixture::new();
    fs::remove_dir(f.0.join("drivers/v2")).unwrap();
    assert_eq!(
        discover_cases(&f.0).unwrap_err().reason,
        "unreadable_golden_directory"
    );
}

#[test]
fn malformed_json_duplicate_ids_wrong_component_and_bad_rank_refuse() {
    let f = Fixture::new();
    f.write("transfer/truncated.json", "{\"CANARY_PRIVATE\":");
    assert!(verify_golden_corpus(&f.0).is_err());
    let f = Fixture::new();
    f.write("transfer/nested/duplicate.json", TRANSFER);
    assert_eq!(
        verify_golden_corpus(&f.0).unwrap_err().reason,
        "duplicate_case_id"
    );
    let f = Fixture::new();
    fs::remove_file(f.0.join("transfer/base.json")).unwrap();
    f.write("charge/misplaced.json", TRANSFER);
    assert_eq!(
        verify_golden_corpus(&f.0).unwrap_err().reason,
        "golden_component_path_mismatch"
    );
    let f = Fixture::new();
    f.write("drivers/informative-v1/rank-cases.json", "[]");
    assert_eq!(
        discover_cases(&f.0).unwrap_err().reason,
        "invalid_rank_fixture"
    );
}

#[cfg(unix)]
#[test]
fn symlinked_case_and_directory_refuse_without_following_them() {
    let f = Fixture::new();
    std::os::unix::fs::symlink(
        f.0.join("transfer/base.json"),
        f.0.join("transfer/link.json"),
    )
    .unwrap();
    assert_eq!(
        discover_cases(&f.0).unwrap_err().reason,
        "invalid_golden_path"
    );
    let f = Fixture::new();
    std::os::unix::fs::symlink(f.0.join("transfer"), f.0.join("drivers/escape")).unwrap();
    assert_eq!(
        discover_cases(&f.0).unwrap_err().reason,
        "invalid_golden_path"
    );
}
