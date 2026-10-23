//! `dope-probes`: exact probe inputs, the levels probe driver and receipt replay.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use dope_probes::inputs::{sha256_hex, split_table, Inputs};
use dope_probes::{landmarks, levels, Error, Result};

#[derive(Parser)]
#[command(
    name = "dope-probes",
    version,
    about = "Development probes for portable levels and landmarks (not shipped)"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Verify the pinned PMLB files and write each table's train.csv and test.csv.
    Split {
        /// Input manifest.
        #[arg(long, default_value = "bench/real-world/probes/inputs.json")]
        inputs: PathBuf,
        /// Directory holding <table>.tsv (decompressed) and, optionally, <table>.tsv.gz.
        #[arg(long)]
        raw: PathBuf,
        /// Output directory; each table gets <out>/<table>/{train,test}.csv.
        #[arg(long)]
        out: PathBuf,
    },
    /// Run the levels probe with a `dope` binary and append receipts to <out>/levels-probe.jsonl.
    Levels {
        /// The engine binary (the committed receipts used the release build of 65b1d2a).
        #[arg(long)]
        dope: PathBuf,
        /// Directory of <table>/{train,test}.csv written by `split`.
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Recompute the byte columns from saved level artifacts and compare them with receipts.
    LevelsBytes {
        /// Directory of <table>/{low,med,high,ultra}.dope (a levels run's output directory).
        #[arg(long)]
        artifacts: PathBuf,
        #[arg(
            long,
            default_value = "bench/real-world/probes/results/levels-probe.jsonl"
        )]
        receipts: PathBuf,
    },
    /// Compare two levels receipt files (for example a rerun against the committed receipts).
    CompareLevels { first: PathBuf, second: PathBuf },
    /// Print the levels tables from a levels receipt file.
    ReplayLevels {
        #[arg(default_value = "bench/real-world/probes/results/levels-probe.jsonl")]
        receipts: PathBuf,
    },
    /// Print tables A–C of the landmark plan's §2.2 from a landmark receipt file.
    ReplayLandmarks {
        #[arg(default_value = "bench/real-world/probes/results/landmark-probe.jsonl")]
        receipts: PathBuf,
    },
    /// Per-landmark and per-auditor retention of two programs of one table (§2.2 D).
    LandmarkDiff {
        #[arg(
            long,
            default_value = "bench/real-world/probes/results/landmark-probe.jsonl"
        )]
        receipts: PathBuf,
        #[arg(long)]
        table: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
    },
    /// The §5.5a representation comparison for the linear landmark.
    Rawlin {
        #[arg(
            long,
            default_value = "bench/real-world/probes/results/landmark-probe.jsonl"
        )]
        receipts: PathBuf,
        #[arg(
            long,
            default_value = "bench/real-world/probes/results/landmark-rawlin.jsonl"
        )]
        rawlin: PathBuf,
    },
}

fn split(inputs: &Path, raw: &Path, out: &Path) -> Result<()> {
    let inp = Inputs::load(inputs)?;
    let mut mismatched = Vec::new();
    for (name, pins) in &inp.tables {
        let gz = raw.join(format!("{name}.tsv.gz"));
        if gz.exists() {
            let bytes = fs::read(&gz).map_err(|e| Error::io(&gz, e))?;
            if sha256_hex(&bytes) != pins.tsv_gz {
                return Err(Error::msg(format!("digest mismatch for {}", gz.display())));
            }
        }
        let tsv = raw.join(format!("{name}.tsv"));
        let bytes = fs::read(&tsv).map_err(|e| {
            Error::msg(format!(
                "{}: {e}; download {}/{name}/{name}.tsv.gz and decompress it",
                tsv.display(),
                inp.pmlb_base
            ))
        })?;
        if sha256_hex(&bytes) != pins.tsv {
            return Err(Error::msg(format!("digest mismatch for {}", tsv.display())));
        }
        let (train, test) = split_table(&bytes, &inp.split, &out.join(name))?;
        let ok = train == pins.train_csv && test == pins.test_csv;
        if !ok {
            mismatched.push(name.clone());
        }
        println!(
            "{name} train {train} test {test} {}",
            if ok { "pinned" } else { "DIFFERS" }
        );
    }
    if mismatched.is_empty() {
        Ok(())
    } else {
        Err(Error::msg(format!(
            "split differs from the pins: {}",
            mismatched.join(", ")
        )))
    }
}

fn run_levels(dope: &Path, data: &Path, out: &Path) -> Result<()> {
    fs::create_dir_all(out).map_err(|e| Error::io(out, e))?;
    let res = out.join("levels-probe.jsonl");
    let done: Vec<String> = if res.exists() {
        levels::load(&res)?.into_iter().map(|r| r.table).collect()
    } else {
        Vec::new()
    };
    let mut dirs: Vec<PathBuf> = fs::read_dir(data)
        .map_err(|e| Error::io(data, e))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for tdir in dirs {
        let name = tdir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if done.contains(&name) {
            continue;
        }
        let row = levels::probe_table(dope, &tdir, out)?;
        let line = serde_json::to_string(&row).map_err(|e| Error::msg(e.to_string()))?;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&res)
            .map_err(|e| Error::io(&res, e))?;
        writeln!(f, "{line}").map_err(|e| Error::io(&res, e))?;
        println!(
            "{name} {} single {:.3} pooled {:.3}",
            levels::independent(&row)
                .map(|x| format!("{x:.3}"))
                .join("/"),
            row.single_at_sum.robust,
            row.pooled_lmh.robust
        );
    }
    Ok(())
}

/// Recompute the byte columns of saved artifacts. Program, inflated and metadata sizes must
/// match the receipts exactly. Brotli sizes are encoder-specific: the receipts used the C
/// reference encoder (Python `brotli` 1.2.1) and this tool uses the engine's Rust encoder, so
/// those columns are reported side by side with the low+med+high chain saving.
fn levels_bytes(artifacts: &Path, receipts: &Path) -> Result<()> {
    let mut differ = Vec::new();
    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    println!("| Table | exact columns | max brotli size difference | chain saving, receipts | chain saving, Rust encoder |");
    println!("|---|---|---:|---:|---:|");
    for r in levels::load(receipts)? {
        let got = levels::byte_columns(&artifacts.join(&r.table))?;
        let want = levels::ByteColumns::from_row(&r);
        let exact = got.bytes == want.bytes
            && got.inflated == want.inflated
            && got.metadata == want.metadata;
        let worst = got
            .brotli_alone
            .iter()
            .zip(&want.brotli_alone)
            .map(|(a, b)| (*a as f64 - *b as f64).abs() / *b as f64)
            .fold(0.0, f64::max);
        println!(
            "| {} | {} | {:.2}% | {:.1}% | {:.1}% |",
            r.table,
            if exact { "match" } else { "DIFFER" },
            100.0 * worst,
            100.0 * want.chain_saving(),
            100.0 * got.chain_saving()
        );
        ours.push(got.chain_saving());
        theirs.push(want.chain_saving());
        if !exact {
            differ.push(r.table);
        }
    }
    println!(
        "median chain saving: receipts {:.1}%, Rust encoder {:.1}%",
        100.0 * dope_probes::stats::median(&theirs),
        100.0 * dope_probes::stats::median(&ours)
    );
    if differ.is_empty() {
        Ok(())
    } else {
        Err(Error::msg(format!(
            "program sizes differ: {}",
            differ.join(", ")
        )))
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Cmd::Split { inputs, raw, out } => split(inputs, raw, out),
        Cmd::Levels { dope, data, out } => run_levels(dope, data, out),
        Cmd::LevelsBytes {
            artifacts,
            receipts,
        } => levels_bytes(artifacts, receipts),
        Cmd::CompareLevels { first, second } => levels::load(first).and_then(|a| {
            levels::load(second).map(|b| print!("{}", levels::render_compare(&a, &b)))
        }),
        Cmd::ReplayLevels { receipts } => {
            levels::load(receipts).map(|rows| print!("{}", levels::render(&rows)))
        }
        Cmd::ReplayLandmarks { receipts } => {
            landmarks::load(receipts).map(|rows| print!("{}", landmarks::render(&rows)))
        }
        Cmd::LandmarkDiff {
            receipts,
            table,
            from,
            to,
        } => landmarks::load(receipts).and_then(|rows| {
            landmarks::render_diff(&rows, table, from, to)
                .map(|s| print!("{s}"))
                .ok_or_else(|| Error::msg("unknown table or program"))
        }),
        Cmd::Rawlin { receipts, rawlin } => landmarks::load(receipts).and_then(|rows| {
            landmarks::load_rawlin(rawlin)
                .map(|raw| print!("{}", landmarks::render_rawlin(&rows, &raw)))
        }),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let r = e.repair();
            eprintln!(
                "dope-probes: {e}\n  reason: {}\n  fixes: {}\n  hint: {}\n  docs: {}",
                r.reason, r.common_fixes, r.repair_hint, r.docs_url
            );
            ExitCode::FAILURE
        }
    }
}
