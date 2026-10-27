use crate::args::*;
use dope_core::{
    landmark::{
        bundle::REPORT_RECIPE_OBSERVED, parse_landmark_ids, signature, signature_requested,
        LandmarkBundle, LandmarkFrame, LandmarkOptions, LandmarkScope, SubsampleSpec,
        TransferReference,
    },
    levels::{ensure_disjoint, remap_samples},
    schema::{infer, IntakeOptions, Task},
    RawTable, SampleOptions,
};
use std::io::Write;

fn opts(a: &LandmarkCommon) -> LandmarkOptions {
    LandmarkOptions {
        panel: a.panel.clone(),
        ridge_rows: a.ridge_rows,
        small_rows: a.small_rows,
        knn_reference: a.knn_reference,
        folds: a.folds,
        work_budget: a.work_budget,
        seed: a.seed,
        spectrum_features: a.spectrum_features,
    }
}
fn err(e: dope_core::DopeError) -> String {
    e.to_string()
}
fn emit(a: &LandmarkCommon, value: serde_json::Value) -> Result<(), String> {
    let text = if a.json {
        serde_json::to_string_pretty(&value).map_err(|_| "cannot encode landmark report")?
    } else {
        let mut s = format!(
            "{} · {}\n",
            value["panel"].as_str().unwrap_or("transfer"),
            value["privacy"].as_str().unwrap_or("private")
        );
        if let Some(cells) = value["cells"].as_array() {
            for c in cells {
                s += &format!(
                    "  {:<14} {:<15} gain {} retention {}\n",
                    c["name"]
                        .as_str()
                        .unwrap_or_else(|| c["id"].as_str().unwrap_or("")),
                    c["state"].as_str().unwrap_or(""),
                    c["gain"],
                    c["retention"]
                );
            }
        }
        s
    };
    let mut w = crate::writer(&a.out)?;
    w.write_all(text.as_bytes())
        .and_then(|_| w.write_all(b"\n"))
        .and_then(|_| w.flush())
        .map_err(|_| "cannot write landmark report".into())
}
pub(crate) fn run(command: LandmarkCommand) -> Result<(), String> {
    match command {
        LandmarkCommand::Real(a) => {
            eprintln!("private measurement: real signatures read authorized fit labels; keep this report local");
            let mut raw = RawTable::read(&a.csv).map_err(err)?;
            let target = raw.column_index(&a.target).map_err(err)?;
            let target_name = raw.headers[target].clone();
            let groups = if let Some(g) = &a.group {
                let j = raw.column_index(g).map_err(err)?;
                if j == target {
                    return Err("group column cannot be the target".into());
                }
                let mut ids = std::collections::BTreeMap::new();
                let groups: Vec<u64> = raw.columns[j]
                    .iter()
                    .map(|v| {
                        let next = ids.len() as u64;
                        *ids.entry(v.clone()).or_insert(next)
                    })
                    .collect();
                raw.headers.remove(j);
                raw.columns.remove(j);
                Some(groups)
            } else {
                None
            };
            let target = raw.column_index(&target_name).map_err(err)?;
            let schema = infer(
                &raw,
                target,
                &IntakeOptions {
                    task: a.task.as_deref().map(|t| {
                        if t == "regression" {
                            Task::Regression
                        } else {
                            Task::Classification
                        }
                    }),
                    ..Default::default()
                },
            )
            .map_err(err)?;
            let data = schema.to_model(&raw).map_err(err)?;
            let frame = LandmarkFrame::real(
                &schema,
                &data,
                &(0..data.rows()).collect::<Vec<_>>(),
                groups.as_deref(),
            )
            .map_err(err)?;
            let options = opts(&a.common);
            let requested_ids = a
                .ids
                .as_deref()
                .map(parse_landmark_ids)
                .transpose()
                .map_err(err)?;
            let subsample_spec = a
                .subsample
                .map(|n| SubsampleSpec::new(n, a.subsample_seed.unwrap_or(a.common.seed)));
            let scope = LandmarkScope::for_frame(&frame, &options, subsample_spec);
            let measured = match &subsample_spec {
                Some(spec) => frame.subsample(spec).map_err(err)?.0,
                None => frame,
            };
            let report = signature_requested(&measured, &options, requested_ids.as_deref())
                .map_err(err)?
                .private_report();
            if let Some(bundle_path) = &a.emit_bundle {
                let bundle = LandmarkBundle::from_report_with_scope(
                    &report,
                    scope.bundle_scope(),
                    REPORT_RECIPE_OBSERVED,
                )
                .map_err(err)?;
                let mut text = serde_json::to_string_pretty(&bundle)
                    .map_err(|_| "cannot encode landmark bundle".to_string())?;
                text.push('\n');
                std::fs::write(bundle_path, text)
                    .map_err(|_| "cannot write landmark bundle".to_string())?;
            }
            emit(&a.common, report)
        }
        LandmarkCommand::Synthetic(a) => {
            if !(20..=200_000).contains(&a.rows) {
                return Err("synthetic signature rows must be in 20..=200000".into());
            }
            let p =
                dope_core::decode(&std::fs::read(&a.model).map_err(|_| "cannot read artifact")?)
                    .map_err(err)?;
            let samples = dope_core::sample(
                &p,
                &SampleOptions {
                    rows: a.rows,
                    seed: a.common.seed,
                    ..Default::default()
                },
            )
            .map_err(err)?;
            let frame = LandmarkFrame::synthetic(&p, &samples).map_err(err)?;
            emit(
                &a.common,
                signature(&frame, &opts(&a.common))
                    .map_err(err)?
                    .private_report(),
            )
        }
        LandmarkCommand::Compare(a) => {
            eprintln!("private measurement: transfer scores use query labels for selection; they are not external certification");
            let p =
                dope_core::decode(&std::fs::read(&a.model).map_err(|_| "cannot read artifact")?)
                    .map_err(err)?;
            let train = RawTable::read(&a.train).map_err(err)?;
            let query = RawTable::read(&a.query).map_err(err)?;
            ensure_disjoint(&train, &query).map_err(err)?;
            let target = train.column_index(&p.schema.target.name).map_err(err)?;
            let schema = infer(
                &train,
                target,
                &IntakeOptions {
                    task: Some(p.schema.task),
                    ..Default::default()
                },
            )
            .map_err(err)?;
            let reference_program =
                dope_core::eval::schema_program(&schema, &train).map_err(err)?;
            let mut combined = train.clone();
            for (j, h) in train.headers.iter().enumerate() {
                let k = query.column_index(h).map_err(err)?;
                combined.columns[j].extend(query.columns[k].clone());
            }
            let data = schema.to_model(&combined).map_err(err)?;
            let reference = TransferReference::new(
                &reference_program,
                &data,
                &(0..train.rows()).collect::<Vec<_>>(),
                &(train.rows()..combined.rows()).collect::<Vec<_>>(),
                &opts(&a.common),
                a.broad,
            )
            .map_err(err)?;
            let s = dope_core::sample(
                &p,
                &SampleOptions {
                    rows: train.rows().min(4096),
                    seed: a.common.seed,
                    ..Default::default()
                },
            )
            .map_err(err)?;
            let score = reference
                .score(&remap_samples(&p, &s, &reference_program).map_err(err)?)
                .map_err(err)?;
            emit(&a.common, score.private_report())
        }
    }
}
