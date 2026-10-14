//! Standalone native operations guard. No approval or merge executor exists.
#[path = "lib.rs"]
mod guard;

use clap::{ArgGroup, Parser};
use guard::github::{GitHub, ProcessTransport};
use serde_json::{Value, json};
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name="dope-merge-guard",about="Read-only, fail-closed DOPE merge inspection",group(ArgGroup::new("input").required(true).multiple(false).args(["review_receipt","verify_merge","controls_only"])))]
pub(crate) struct Cli {
    #[arg(long, default_value = "neverhuman/dope")]
    repo: String,
    #[arg(long, required_unless_present = "controls_only")]
    pr: Option<i64>,
    #[arg(long, required_unless_present = "controls_only")]
    source: Option<String>,
    #[arg(long)]
    base: String,
    #[arg(
        long,
        default_value = "gh",
        help = "Existing gh executable; credentials are never switched"
    )]
    gh: String,
    #[arg(long)]
    review_receipt: Option<PathBuf>,
    #[arg(long, help = "Prior full premerge receipt")]
    verify_merge: Option<PathBuf>,
    #[arg(long, help = "Capture server rules without a review fixture")]
    controls_only: bool,
}
pub(crate) fn read_receipt(path: &std::path::Path) -> guard::Result<Value> {
    let mut data = Vec::new();
    let file = std::fs::File::open(path).map_err(|_| guard::Refused("receipt_unavailable"))?;
    file.take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut data)
        .map_err(|_| guard::Refused("receipt_unavailable"))?;
    guard::require(data.len() <= 16 * 1024 * 1024, "receipt_unavailable")?;
    serde_json::from_slice(&data).map_err(|_| guard::Refused("receipt_unavailable"))
}
pub(crate) fn execute(args: Cli) -> Value {
    let mut api = GitHub::new(args.gh, ProcessTransport::default());
    if args.controls_only {
        return guard::inspection::inspect_controls(&mut api, &args.repo, &args.base);
    }
    let path = args.review_receipt.as_ref().or(args.verify_merge.as_ref());
    let receipt = match path.and_then(|p| read_receipt(p).ok()) {
        Some(v) => v,
        None => {
            return json!({"version":1,"read_only":true,"state":"blocked","reason":"receipt_unavailable","merge_authorized":false});
        }
    };
    let pins = json!({"repo":args.repo,"pr":args.pr,"source_sha":args.source,"base_sha":args.base,"base_branch":"main"});
    guard::inspection::inspect(
        &mut api,
        &pins,
        &receipt,
        if args.verify_merge.is_some() {
            Some(&receipt)
        } else {
            None
        },
    )
}
fn main() -> ExitCode {
    let result = execute(Cli::parse());
    match serde_json::to_string_pretty(&result) {
        Ok(output) => println!("{output}"),
        Err(_) => {
            println!(
                "{{\"state\":\"blocked\",\"reason\":\"invalid_or_incomplete_evidence\",\"merge_authorized\":false}}"
            );
            return ExitCode::FAILURE;
        }
    }
    if result["state"].as_str().is_some_and(|s| {
        [
            "server_policy_observed",
            "ready_for_operator",
            "merged_verified",
        ]
        .contains(&s)
    }) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
