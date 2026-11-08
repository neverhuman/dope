//! DVS receipt and Audit Card from explicitly supplied, local frozen evidence.
use crate::safe_io;
use clap::Args;
use dope_core::dvs::{self, DvsRequest};
use std::path::PathBuf;

#[derive(Args)]
pub(crate) struct DvsArgs {
    /// Frozen evidence request; no fitting, data discovery or downloads.
    #[arg(long)]
    pub(crate) request: PathBuf,
    /// NEW directory containing receipt.json and AuditCard.md.
    #[arg(short, long)]
    pub(crate) out: PathBuf,
}
pub(crate) fn run(args: DvsArgs) -> Result<(), String> {
    let input: DvsRequest =
        safe_io::read_json(&args.request, 16 * 1024 * 1024).map_err(|_| "invalid_dvs_request")?;
    let result = dvs::evaluate(&input).map_err(|e| e.to_string())?;
    let receipt = safe_io::json(&result)?;
    let card = dvs::receipt::audit_card(&result).into_bytes();
    safe_io::write_directory(
        &args.out,
        &[("receipt.json", receipt), ("AuditCard.md", card)],
    )?;
    println!(
        "{}",
        serde_json::json!({"protocol":dvs::PROTOCOL,"status":result.status,"scheduled_cells":result.scheduled_cells,"qualification_state":"not_run"})
    );
    Ok(())
}
