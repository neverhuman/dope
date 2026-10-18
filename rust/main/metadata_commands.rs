fn run_metadata(command: Command) -> Result<()> {
    match command {
        Command::Version { json } => {
            let value = serde_json::json!({
                "name": "dope-kernel",
                "version": DEVELOPMENT_VERSION,
                "kpi_contract": {"sha256": KPI_CONTRACT_SHA256, "blake3": KPI_CONTRACT_BLAKE3}
            });
            if json {
                println!("{}", serde_json::to_string(&value)?);
            } else {
                println!("dope-kernel {DEVELOPMENT_VERSION}");
            }
        }
        Command::FreezeContract => {
            let contract = KpiContract::embedded()?;
            contract.validate()?;
            println!(
                "{}",
                serde_json::json!({"frozen": true, "release_identity": DEVELOPMENT_VERSION, "kpi_contract_sha256": KPI_CONTRACT_SHA256, "kpi_contract_blake3": KPI_CONTRACT_BLAKE3})
            );
        }
        Command::BuildRc {
            release_dir,
            gate_evidence,
            coverage,
            out,
            source_commit,
            source_tag_object_sha256,
            signature_kind,
            public_key_sha256,
        } => {
            let evidence: GateEvidence = serde_json::from_slice(
                &fs::read(&gate_evidence)
                    .map_err(|error| dope_kernel::error::io_error(&gate_evidence, error))?,
            )?;
            let coverage: CoverageReport = serde_json::from_slice(
                &fs::read(&coverage)
                    .map_err(|error| dope_kernel::error::io_error(&coverage, error))?,
            )?;
            let manifest = build_rc_manifest(
                &release_dir,
                &evidence,
                &coverage,
                &BuildRcOptions {
                    source_commit,
                    source_tag_object_sha256,
                    signature_kind,
                    public_key_sha256,
                },
            )?;
            write_json(&out, &manifest)?;
            println!("{}", serde_json::to_string(&manifest)?);
        }
        Command::Campaign { command } => run_campaign(command)?,
        Command::EmbedDataset {
            csv,
            target_column,
            target_index,
            task,
            router_bundle,
            router_evidence,
            out,
            vector_out,
            restricted_metadata,
        } => {
            let selector = match (target_column.as_deref(), target_index) {
                (Some(name), None) => TargetSelector::Name(name),
                (None, Some(index)) => TargetSelector::Index(index),
                _ => {
                    return Err(DopeError::Data(
                        "exactly one target column selector is required".into(),
                    ));
                }
            };
            let embedding = embed_dataset_with_metadata(
                &csv,
                selector,
                parse_task(&task)?,
                &router_bundle,
                &router_evidence,
                if restricted_metadata {
                    MetadataMode::RestrictedResearch
                } else {
                    MetadataMode::Public
                },
            )?;
            dope_kernel::production::write_canonical(&out, &embedding)?;
            if let Some(path) = &vector_out {
                write_embedding_csv(path, &embedding)?;
            }
            println!(
                "{}",
                serde_json::json!({
                    "dimension": embedding.dimension,
                    "candidates": embedding.candidate_order,
                    "schema_sha256": embedding.schema_sha256,
                    "out": out,
                    "vector_out": vector_out,
                })
            );
        }
        Command::QualifyEmbeddingRouter {
            router_bundle,
            router_evidence,
        } => {
            let qualification = qualify_embedding_router(&router_bundle, &router_evidence)?;
            println!("{}", serde_json::to_string(&qualification)?);
        }
        Command::InspectRegressionCorpus { dataset_root } => {
            let inventory = inspect_regression_corpus(&dataset_root)?;
            println!("{}", serde_json::to_string(&inventory)?);
        }
        Command::EmbedRegressionCorpus {
            dataset_root,
            output_dir,
            router_bundle,
            router_evidence,
            method,
            expected_dimension,
            shard_index,
            shard_count,
            jobs,
            determinism_checks,
            manifest_out,
        } => {
            let manifest = embed_regression_corpus(&RegressionEmbeddingBatchOptions {
                dataset_root: &dataset_root,
                output_dir: &output_dir,
                router_bundle: &router_bundle,
                router_evidence: &router_evidence,
                method: &method,
                expected_dimension,
                shard_index,
                shard_count,
                jobs,
                determinism_checks,
            })?;
            dope_kernel::production::write_canonical(&manifest_out, &manifest)?;
            println!(
                "{}",
                serde_json::json!({
                    "discovered": manifest.discovered,
                    "assigned": manifest.assigned,
                    "success": manifest.success,
                    "schema_mismatch": manifest.schema_mismatch,
                    "failed": manifest.failed,
                    "determinism_checked": manifest.determinism_checked,
                    "dimension": manifest.router.dimension,
                    "manifest_out": manifest_out,
                })
            );
            if manifest.schema_mismatch != 0 || manifest.failed != 0 {
                return Err(DopeError::Data(format!(
                    "regression embedding shard has {} schema mismatches and {} failures",
                    manifest.schema_mismatch, manifest.failed
                )));
            }
        }
        Command::VerifyRegressionEmbeddings {
            output_dir,
            router_bundle,
            router_evidence,
            method,
            expected_dimension,
            expected_count,
        } => {
            let qualification = qualify_embedding_router(&router_bundle, &router_evidence)?;
            if qualification.dimension != expected_dimension {
                return Err(DopeError::Data(format!(
                    "qualified router dimension is {}, expected {expected_dimension}",
                    qualification.dimension
                )));
            }
            let verification =
                verify_regression_embeddings(&output_dir, &method, &qualification, expected_count)?;
            println!("{}", serde_json::to_string(&verification)?);
        }
        command => return run_release(command),
    }
    Ok(())
}
