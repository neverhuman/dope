//! `dope-kpi/v2` scoreboard data plane (`docs/plans/v2-gap-closure/formats.md`).
//!
//! Pulled into the `dope_kpi` binary with `#[path]`, so the library, its
//! `main.rs` and the crate manifest stay untouched. Every artifact written
//! from here is a pure function of its inputs: no clock, no process id, no
//! unseeded randomness, and the same bytes for any `--threads` value.

pub mod audit;
pub mod auditors;
pub mod boot;
pub mod celldata;
pub mod cellplan;
pub mod compare;
pub mod controls;
pub mod emitlog;
pub mod ext_arm;
pub mod ext_cell;
pub mod ext_emit;
pub mod ext_proc;
pub mod ext_reason;
pub mod ext_rows;
pub mod ext_scratch;
pub mod ext_summary;
pub mod ext_template;
pub mod families;
pub mod flip;
pub mod format2;
pub mod guard_eval;
pub mod guard_merge;
pub mod guards;
pub mod hero;
pub mod hero_labels;
pub mod hero_layout;
pub mod hero_model;
pub mod hero_svg;
pub mod io;
pub mod jsonl;
pub mod mat;
pub mod null;
pub mod oracle;
pub mod pairing;
pub mod panels_build;
pub mod panelsel;
pub mod par;
pub mod primary;
pub mod realref;
pub mod receipt;
pub mod reduce;
pub mod render;
pub mod render_paper;
mod render_paper_tex;
pub mod render_ranked;
pub mod roles;
pub mod rule;
pub mod rundir;
pub mod score;
pub mod scorecell;
pub mod seeds;
pub mod shard;
pub mod sketch;
pub mod spec;
pub mod trg;

#[cfg(test)]
pub mod fixture;
#[cfg(test)]
pub mod guard_fixture;
#[cfg(test)]
mod guard_tests;
#[cfg(test)]
mod hero_tests;
#[cfg(test)]
mod render_paper_tests;
#[cfg(test)]
pub mod stat_testing;
#[cfg(test)]
mod stat_tests;
#[cfg(test)]
mod trg_tests;

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
