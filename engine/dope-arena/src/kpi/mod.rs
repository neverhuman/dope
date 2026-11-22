//! `dope-kpi/v2` scoreboard data plane (`docs/plans/v2-gap-closure/formats.md`).
//!
//! Pulled into the `dope_kpi` binary with `#[path]`, so the library, its
//! `main.rs` and the crate manifest stay untouched. Every artifact written
//! from here is a pure function of its inputs: no clock, no process id, no
//! unseeded randomness, and the same bytes for any `--threads` value.

pub mod cellplan;
pub mod families;
pub mod io;
pub mod jsonl;
// The `score` and control-arm stages consume this container; `build-panels`
// and `plan` only define it, so the binary does not call it yet.
#[allow(dead_code)]
pub mod mat;
pub mod panels_build;
pub mod par;
pub mod roles;
pub mod seeds;
pub mod sketch;

#[cfg(test)]
pub mod fixture;

/// Scratch directory for unit tests, inside the build tree and never in the sources.
#[cfg(test)]
pub fn unit_test_dir(name: &str) -> std::path::PathBuf {
    let exe = std::env::current_exe().expect("test binary path");
    let target = exe.ancestors().nth(3).expect("target directory");
    let dir = target.join("kpi-unit").join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clear scratch directory");
    }
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    dir
}
