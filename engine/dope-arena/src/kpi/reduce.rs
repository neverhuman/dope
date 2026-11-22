//! `dope_kpi reduce-ledger`: run `dvs::v3::evaluate_v3` on a score ledger.
//!
//! The existing `dope-arena reduce` fixes the kappa thresholds at their
//! defaults and refuses partial evidence. This reducer takes the thresholds
//! from a file, makes `allow_partial_diagnostic` a flag, and can restrict the
//! ledger to one multiplier or to some auditors. `--kappa-out` writes the
//! thresholds calibrated from the kappa points of this ledger, which is
//! meaningful for the real twin arm (`twin_c`) only.

use super::jsonl::{
    canonical_json, canonical_pretty, parse_json, read_file, sha256_hex, write_atomic,
};
use super::score::parse_ledger_id;
use dope_arena::ledger::read_ledger;
use dope_arena::{ArenaError, ArenaResult};
use dope_core::dvs::v3::equiv::KappaThresholds;
use dope_core::dvs::v3::{
    evaluate_v3, CoverageSummary, FidelityCellRequest, FidelityV3Receipt, FidelityV3Request,
    PROTOCOL_V3,
};
use serde::Serialize;
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub struct ReduceArgs {
    /// `ledger.jsonl` written by `score`.
    #[arg(long)]
    pub ledger: PathBuf,
    /// Output receipt (JSON).
    #[arg(long)]
    pub out: PathBuf,
    /// Emission mode recorded in the receipt: `faithful` (4n), `matched` (1n) or `distill`.
    #[arg(long, default_value = "faithful")]
    pub mode: String,
    /// Keep only entries of this multiplier.
    #[arg(long)]
    pub multiplier: Option<u32>,
    /// Keep only entries of these auditors (comma separated).
    #[arg(long, value_delimiter = ',')]
    pub auditors: Option<Vec<String>>,
    /// JSON file with `pred_kappa_star`, `lift_kappa_star`, `calib_kappa_star`, `driver_kappa_star`.
    #[arg(long)]
    pub kappa_file: Option<PathBuf>,
    /// Let absent driver, realism or privacy evidence pass (diagnostic receipts only).
    #[arg(long)]
    pub allow_partial: bool,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = 1000)]
    pub bootstrap_draws: usize,
    /// Write kappa thresholds calibrated from this ledger's informative cells.
    #[arg(long)]
    pub kappa_out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Filters<'a> {
    multiplier: Option<u32>,
    auditors: &'a Option<Vec<String>>,
}

#[derive(Serialize)]
struct ReceiptFile<'a> {
    ledger_sha256: &'a str,
    filters: Filters<'a>,
    ledger_entries: usize,
    seed: u64,
    bootstrap_draws: usize,
    kappa_thresholds: KappaThresholds,
    allow_partial_diagnostic: bool,
    receipt: &'a FidelityV3Receipt,
}

#[derive(Serialize)]
struct Summary<'a> {
    command: &'static str,
    ledger_sha256: &'a str,
    ledger_entries: usize,
    reduced_entries: usize,
    mode: &'a str,
    allow_partial_diagnostic: bool,
    kappa_thresholds: KappaThresholds,
    coverage: &'a CoverageSummary,
    receipt_sha256: String,
}

/// Entries that pass the multiplier and auditor filters, in ledger order.
pub fn select(
    cells: Vec<FidelityCellRequest>,
    multiplier: Option<u32>,
    auditors: &Option<Vec<String>>,
) -> ArenaResult<Vec<FidelityCellRequest>> {
    let mut kept = Vec::new();
    for cell in cells {
        if auditors
            .as_ref()
            .is_some_and(|names| !names.contains(&cell.auditor_id))
        {
            continue;
        }
        if let Some(wanted) = multiplier {
            let Some((_, _, found)) = parse_ledger_id(&cell.cell_id, &cell.auditor_id) else {
                return Err(ArenaError::Contract(
                    "--multiplier needs entry ids of the form <cell>:s<seed>:m<multiplier>:<auditor>"
                        .into(),
                ));
            };
            if found != wanted {
                continue;
            }
        }
        kept.push(cell);
    }
    Ok(kept)
}

/// Thresholds at the 95th percentile of the kappa points of the informative cells.
pub fn calibrate(receipt: &FidelityV3Receipt) -> KappaThresholds {
    let (mut pred, mut lift, mut calib) = (Vec::new(), Vec::new(), Vec::new());
    for cell in receipt.cells.iter().filter(|c| c.status == "complete") {
        if let Some(equivalence) = &cell.equivalence {
            pred.push(equivalence.pred.kappa_point);
            lift.push(equivalence.lift.kappa_point);
            calib.push(equivalence.calib.kappa_point);
        }
    }
    KappaThresholds::calibrate_from_real_c(&pred, &lift, &calib, &[])
}

pub fn run(args: &ReduceArgs) -> ArenaResult<String> {
    let kappa_thresholds = match &args.kappa_file {
        Some(path) => parse_json(&path.display().to_string(), &read_file(path)?)?,
        None => KappaThresholds::default(),
    };
    let (cells, ledger_sha256) = read_ledger(&args.ledger)?;
    let ledger_entries = cells.len();
    let cells = select(cells, args.multiplier, &args.auditors)?;
    if cells.is_empty() {
        return Err(ArenaError::Contract(
            "no ledger entry passes the filters".into(),
        ));
    }
    let receipt = evaluate_v3(&FidelityV3Request {
        protocol: PROTOCOL_V3.to_string(),
        mode: args.mode.clone(),
        seed: args.seed,
        bootstrap_draws: args.bootstrap_draws,
        kappa_thresholds,
        allow_partial_diagnostic: args.allow_partial,
        cells,
    })?;
    let text = canonical_pretty(&ReceiptFile {
        ledger_sha256: &ledger_sha256,
        filters: Filters {
            multiplier: args.multiplier,
            auditors: &args.auditors,
        },
        ledger_entries,
        seed: args.seed,
        bootstrap_draws: args.bootstrap_draws,
        kappa_thresholds,
        allow_partial_diagnostic: args.allow_partial,
        receipt: &receipt,
    })?;
    write_atomic(&args.out, text.as_bytes())?;
    if let Some(path) = &args.kappa_out {
        write_atomic(path, canonical_pretty(&calibrate(&receipt))?.as_bytes())?;
    }
    canonical_json(&Summary {
        command: "reduce-ledger",
        ledger_sha256: &ledger_sha256,
        ledger_entries,
        reduced_entries: receipt.cells.len(),
        mode: &args.mode,
        allow_partial_diagnostic: args.allow_partial,
        kappa_thresholds,
        coverage: &receipt.coverage,
        receipt_sha256: sha256_hex(text.as_bytes()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dope_core::dvs::v3::FidelityCellInput;

    fn entry(id: &str, auditor: &str) -> FidelityCellRequest {
        FidelityCellRequest {
            cell_id: id.to_string(),
            family_id: "lf".to_string(),
            auditor_id: auditor.to_string(),
            input: FidelityCellInput::Unsupported,
        }
    }

    #[test]
    fn filters_keep_ledger_order_and_reject_foreign_ids() {
        let ledger = || {
            vec![
                entry("p:h:full:d0:s0:m1:gbdt", "gbdt"),
                entry("p:h:full:d0:s0:m1:linear-seeded-v1", "linear-seeded-v1"),
                entry("p:h:full:d0:s0:m4:gbdt", "gbdt"),
                entry("p:h:full:d0:s1:m4:gbdt", "gbdt"),
            ]
        };
        let ids = |cells: Vec<FidelityCellRequest>| -> Vec<String> {
            cells.into_iter().map(|c| c.cell_id).collect()
        };
        assert_eq!(select(ledger(), None, &None).unwrap().len(), 4);
        assert_eq!(
            ids(select(ledger(), Some(4), &None).unwrap()),
            ["p:h:full:d0:s0:m4:gbdt", "p:h:full:d0:s1:m4:gbdt"]
        );
        let linear = Some(vec!["linear-seeded-v1".to_string()]);
        assert_eq!(
            ids(select(ledger(), Some(1), &linear).unwrap()),
            ["p:h:full:d0:s0:m1:linear-seeded-v1"]
        );
        assert!(select(ledger(), Some(2), &None).unwrap().is_empty());
        let foreign = vec![entry("cell-7", "gbdt")];
        assert!(select(foreign.clone(), None, &None).is_ok());
        let err = select(foreign, Some(4), &None).unwrap_err().to_string();
        assert!(err.contains("--multiplier"), "{err}");
    }
}
