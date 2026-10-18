use super::exit_codes::{Code, Failure, Result};
use super::io;
use super::registry::{Cli, Command, Limits};
use super::workflow;
use serde_json::{Value, json};
use std::path::Path;

pub fn run(cli: Cli) -> Result<()> {
    if !cli.offline || cli.network {
        return Err(Failure::new(Code::Unsupported, "network_access_refused"));
    }
    if cli.panel != "development" {
        return Err(Failure::new(
            Code::Unsupported,
            "candidate_panel_not_qualified",
        ));
    }
    let limits = &cli.limits;
    match cli.command {
        Command::Prepare { input, out, report } => {
            let mut paths = vec![out.as_path()];
            paths.extend(report.as_deref());
            io::outputs(&paths)?;
            let (_, receipt) = workflow::table(&input, limits)?;
            io::emit(&receipt, Some(&out), limits)?;
            if let Some(report) = report {
                io::emit(&receipt, Some(&report), limits)?;
            }
            Ok(())
        }
        Command::ValidateInput { input, out } => {
            io::outputs(&out.as_deref().into_iter().collect::<Vec<_>>())?;
            let (_, receipt) = workflow::table(&input, limits)?;
            io::emit(&receipt, out.as_deref(), limits)
        }
        Command::Compile(args) => workflow::compile(args, limits),
        Command::Inspect { artifact, out } => {
            io::outputs(&out.as_deref().into_iter().collect::<Vec<_>>())?;
            io::emit(
                &workflow::inspect(&artifact, limits)?,
                out.as_deref(),
                limits,
            )
        }
        Command::Sample {
            artifact,
            rows,
            out,
            seed,
        } => workflow::sample(&artifact, rows, seed, &out, limits),
        Command::Doctor { out } => {
            io::outputs(&out.as_deref().into_iter().collect::<Vec<_>>())?;
            io::emit(
                &json!({"name":"dope","version":env!("CARGO_PKG_VERSION"),
                "implementation":"native_rust","offline":true,"network":"refused",
                "raw_compile":"fully_observed_numeric_regression",
                "legacy_v1_decode":"read_only_compatible",
                "prepared_view":"unavailable","candidate_evaluation":"unavailable",
                "source_free_isolation":"not_claimed_by_facade",
                "external_worker_qualification":"unavailable",
                "utility_qualification_state":"unassessed","empirical_privacy_state":"unassessed",
                "formal_dp_state":"unverified","legal_release_state":"pending",
                "legacy_certification":"dope-kernel","weights_downloaded":false}),
                out.as_deref(),
                limits,
            )
        }
        Command::Evaluate {
            artifact,
            real_dir,
            out,
            report,
            seed,
            tier,
        } => {
            let Some(directory) = real_dir else {
                return Err(Failure::unsupported());
            };
            let mut paths = vec![out.as_path()];
            paths.extend(report.as_deref());
            io::outputs(&paths)?;
            // Explicit legacy replay, kept for existing dope scripts. It does not
            // use a candidate panel or confer TDL qualification/release authority.
            eprintln!(
                "migration: --real-dir replays legacy certification; exit 0 means receipt written, inspect certified; use dope-kernel for the full legacy interface"
            );
            io::local(&directory)?;
            let artifact_bytes = io::read(&artifact, limits.max_input_bytes)?;
            super::artifacts::budget(&artifact_bytes, limits)?;
            let (_artifact_snapshot, artifact_path) = io::snapshot(&artifact_bytes)?;
            for name in ["train.csv", "test.csv"] {
                let bytes = io::read(&directory.join(name), limits.max_input_bytes)?;
                workflow::numeric_csv(&bytes, limits)?;
            }
            let policy = dope_kernel::contract::ReleasePolicy::new(
                tier.parse().map_err(|_| Failure::usage("tier_invalid"))?,
                None,
                false,
            )?;
            // Retain the historical writer in memory, then enforce the public
            // output boundary (exclusive creation, permissions and byte limit).
            let (_sink, captured) = io::sink()?;
            let receipt = dope_kernel::certification::certify_kernel_with_policy(
                &directory,
                &artifact_path,
                &captured,
                seed,
                0,
                1,
                &policy,
            )?;
            let projection = json!({"format":"dope-legacy-evaluation-summary","certified":receipt.certified,
                "tdl_qualified":false,"legal_release_state":"pending","formal_dp_state":"unverified"});
            io::emit(&receipt, Some(&out), limits)?;
            if let Some(report) = report {
                io::emit(&projection, Some(&report), limits)?;
            }
            io::emit(&projection, None, limits)
        }
        Command::Report { report, out } => {
            io::outputs(&out.as_deref().into_iter().collect::<Vec<_>>())?;
            let input = io::json(&io::read(&report, limits.max_input_bytes)?)?;
            let (projection, decision) = project_report(&input)?;
            io::emit(&projection, out.as_deref(), limits)?;
            decision
        }
        Command::Demo { out, report, seed } => demo(&out, report.as_deref(), seed, limits),
        Command::Version { json: as_json } => {
            if as_json {
                io::emit(
                    &json!({"name":"dope","version":env!("CARGO_PKG_VERSION")}),
                    None,
                    limits,
                )
            } else {
                io::stdout(
                    format!("dope {}", env!("CARGO_PKG_VERSION")).as_bytes(),
                    limits,
                )
            }
        }
    }
}

fn project_report(input: &Value) -> Result<(Value, Result<()>)> {
    // Never copy arbitrary strings, nested source metadata or path inventories.
    // These are display-only local receipts, not authenticated release authority.
    if input["format"] != "dope-kernel-certification"
        || !input["version"]
            .as_u64()
            .is_some_and(|v| (1..=5).contains(&v))
        || !input["certified"].is_boolean()
        || !input["gates"].is_object()
        || !input["failed_gates"].is_array()
    {
        return Err(Failure::usage("legacy_report_schema_invalid"));
    }
    let gates = input["gates"].as_object().unwrap();
    if gates.is_empty() || gates.values().any(|v| !v.is_boolean()) {
        return Err(Failure::usage("legacy_report_schema_invalid"));
    }
    let failures = input["failed_gates"].as_array().unwrap();
    if failures.iter().any(|v| !v.is_string())
        || (input["certified"] == true
            && (!failures.is_empty() || gates.values().any(|v| v == false)))
    {
        return Err(Failure::usage("legacy_report_inconsistent"));
    }
    let privacy_failed = gates.iter().any(|(key, value)| {
        value == false
            && matches!(
                key.as_str(),
                "membership_inference"
                    | "rare_slice_privacy_coverage"
                    | "rare_slice_membership"
                    | "rare_slice_attribute_inference"
                    | "attribute_inference"
                    | "exact_copies"
                    | "near_copies"
            )
    });
    let decision = if input["certified"] != true {
        Err(Failure::new(
            if privacy_failed {
                Code::Privacy
            } else {
                Code::Science
            },
            if privacy_failed {
                "legacy_privacy_gate_failed"
            } else {
                "legacy_scientific_gate_failed"
            },
        ))
    } else {
        // Historical certification has no independent legal authorization receipt.
        Err(Failure::new(
            Code::Rights,
            "release_authorization_unavailable",
        ))
    };
    Ok((
        json!({"format":"dope-report-summary","version":1,"legacy_certified":input["certified"],
        "failed_gate_count":gates.values().filter(|v| **v == false).count(),
        "utility_qualification_state":"unassessed","empirical_privacy_state":"unassessed",
        "formal_dp_state":"unverified","legal_release_state":"pending",
        "tdl_qualified":false,"receipt_authentication":"not_verified"}),
        decision,
    ))
}

fn demo(out: &Path, report: Option<&Path>, seed: u64, limits: &Limits) -> Result<()> {
    let mut paths = vec![out];
    paths.extend(report);
    io::outputs(&paths)?;
    if limits.max_rows < 16 || limits.max_cells < 32 {
        return Err(Failure::budget());
    }
    let x = (0..16).map(|i| i as f32 / 15.0).collect::<Vec<_>>();
    let y = x.iter().map(|v| 0.2 + 0.5 * v).collect::<Vec<_>>();
    let options = dope_kernel::CompileOptions {
        seed: Some(seed),
        backend_id: Some("chow_liu".into()),
        deadline: Some(std::time::Duration::from_secs(10)),
        ..Default::default()
    };
    let compiled = dope_kernel::compile_kernel_from_arrays(
        &x,
        &y,
        16,
        1,
        dope_kernel::model::Task::Regression,
        &options,
    )?;
    let summary = json!({"format":"dope-demo-receipt","version":1,"fixture_only":true,
        "qualified":false,"certified":false,"artifact_bytes":compiled.artifact.len(),
        "privacy":"unassessed","formal_dp":"unverified","legal_release":"pending","offline":true});
    if serde_json::to_vec_pretty(&summary).unwrap().len() as u64 > limits.max_output_bytes {
        return Err(Failure::budget());
    }
    io::write(out, &compiled.artifact, limits)?;
    if let Some(report) = report {
        io::emit(&summary, Some(report), limits)?;
    }
    io::emit(&summary, None, limits)
}
