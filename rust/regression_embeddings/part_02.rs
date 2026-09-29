

fn process_dataset(
    dataset: &DatasetInput,
    embedder: &DatasetEmbedder,
    temporary_dir: &Path,
    output_dir: &Path,
    method: &str,
) -> RegressionEmbeddingRecord {
    let mut record = RegressionEmbeddingRecord {
        dataset_id: dataset.id.clone(),
        global_index: dataset.global_index,
        status: "failed".into(),
        header_source: None,
        rows: None,
        features: None,
        json_output: None,
        vector_output: None,
        error: None,
    };
    let (headered_csv, target, header_source) = match prepare_headered_csv(dataset, temporary_dir) {
        Ok(prepared) => prepared,
        Err(error) => {
            record.status = "schema_mismatch".into();
            record.error = Some(error.to_string());
            return record;
        }
    };
    record.header_source = Some(header_source);
    let dimension = embedder.qualification().dimension;
    let stem = format!("{}_{}_{}", dataset.id, method, dimension);
    let json_output = output_dir.join(format!("{stem}.json"));
    let vector_output = output_dir.join(format!("{stem}.vs"));
    let partial_suffix = format!("partial-{}", std::process::id());
    let json_partial = output_dir.join(format!(".{stem}.json.{partial_suffix}"));
    let vector_partial = output_dir.join(format!(".{stem}.vs.{partial_suffix}"));
    let result = (|| -> Result<()> {
        let embedding = embedder.embed_csv(
            &headered_csv,
            TargetSelector::Name(&target),
            Task::Regression,
        )?;
        if embedding.dimension != dimension
            || embedding.component_names.len() != dimension
            || embedding.vector.len() != dimension
        {
            return Err(DopeError::Data(
                "embedding payload does not match its qualified dimension".into(),
            ));
        }
        record.rows = Some(embedding.rows);
        record.features = Some(embedding.features);
        write_canonical(&json_partial, &embedding)?;
        crate::embedding::write_embedding_csv(&vector_partial, &embedding)?;
        verify_embedding_vector(&vector_partial, dimension)?;
        fs::rename(&json_partial, &json_output).map_err(|error| io_error(&json_output, error))?;
        fs::rename(&vector_partial, &vector_output)
            .map_err(|error| io_error(&vector_output, error))?;
        Ok(())
    })();
    let _ = fs::remove_file(&headered_csv);
    let _ = fs::remove_file(&json_partial);
    let _ = fs::remove_file(&vector_partial);
    match result {
        Ok(()) => {
            record.status = "success".into();
            record.json_output = Some(json_output);
            record.vector_output = Some(vector_output);
        }
        Err(error) => record.error = Some(error.to_string()),
    }
    record
}

fn verify_determinism(
    dataset: &DatasetInput,
    record: &RegressionEmbeddingRecord,
    embedder: &DatasetEmbedder,
    temporary_dir: &Path,
) -> Result<()> {
    let (headered_csv, target, _) = prepare_headered_csv(dataset, temporary_dir)?;
    let vector_repeat = temporary_dir.join(format!("{}.repeat.vs", dataset.id));
    let result = (|| -> Result<()> {
        let repeated = embedder.embed_csv(
            &headered_csv,
            TargetSelector::Name(&target),
            Task::Regression,
        )?;
        let json_output = record
            .json_output
            .as_deref()
            .ok_or_else(|| DopeError::Data("successful record has no JSON output".into()))?;
        if fs::read(json_output).map_err(|error| io_error(json_output, error))?
            != canonical_json(&repeated)?
        {
            return Err(DopeError::Data(
                "repeated JSON embedding is not byte-stable".into(),
            ));
        }
        crate::embedding::write_embedding_csv(&vector_repeat, &repeated)?;
        let vector_output = record
            .vector_output
            .as_deref()
            .ok_or_else(|| DopeError::Data("successful record has no vector output".into()))?;
        if fs::read(vector_output).map_err(|error| io_error(vector_output, error))?
            != fs::read(&vector_repeat).map_err(|error| io_error(&vector_repeat, error))?
        {
            return Err(DopeError::Data(
                "repeated vector embedding is not byte-stable".into(),
            ));
        }
        Ok(())
    })();
    let _ = fs::remove_file(headered_csv);
    let _ = fs::remove_file(vector_repeat);
    result
}

pub fn embed_regression_corpus(
    options: &RegressionEmbeddingBatchOptions<'_>,
) -> Result<RegressionEmbeddingBatchManifest> {
    if options.shard_count == 0 || options.shard_index >= options.shard_count {
        return Err(DopeError::Data("invalid shard index or count".into()));
    }
    if options.jobs == 0 {
        return Err(DopeError::Data(
            "batch worker count must be positive".into(),
        ));
    }
    if !valid_method(options.method) {
        return Err(DopeError::Data(
            "method must contain only lowercase ASCII letters, digits, and hyphens".into(),
        ));
    }

    // Qualification deliberately precedes corpus discovery and all output writes.
    let embedder = DatasetEmbedder::load(options.router_bundle, options.router_evidence)?;
    let router = embedder.qualification().clone();
    if router.dimension != options.expected_dimension {
        return Err(DopeError::Data(format!(
            "qualified router dimension is {}, expected {}",
            router.dimension, options.expected_dimension
        )));
    }
    let discovered = discover(options.dataset_root)?;
    let assigned = discovered
        .iter()
        .filter(|dataset| dataset.global_index % options.shard_count == options.shard_index)
        .cloned()
        .collect::<Vec<_>>();
    let dataset_ids_sha256 = hashes(&canonical_json(
        &discovered
            .iter()
            .map(|dataset| dataset.id.as_str())
            .collect::<Vec<_>>(),
    )?)
    .sha256;
    fs::create_dir_all(options.output_dir).map_err(|error| io_error(options.output_dir, error))?;
    let temporary_dir = options
        .output_dir
        .join(format!(".tmp-shard-{}", options.shard_index));
    fs::create_dir_all(&temporary_dir).map_err(|error| io_error(&temporary_dir, error))?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.jobs)
        .build()
        .map_err(|error| DopeError::Data(format!("cannot build batch worker pool: {error}")))?;
    let mut records = pool.install(|| {
        assigned
            .par_iter()
            .map(|dataset| {
                process_dataset(
                    dataset,
                    &embedder,
                    &temporary_dir,
                    options.output_dir,
                    options.method,
                )
            })
            .collect::<Vec<_>>()
    });
    records.sort_by_key(|record| record.global_index);

    let successful_ids = records
        .iter()
        .filter(|record| record.status == "success")
        .take(options.determinism_checks)
        .map(|record| record.dataset_id.clone())
        .collect::<Vec<_>>();
    let mut determinism_checked = 0;
    for id in successful_ids {
        let dataset = assigned
            .iter()
            .find(|dataset| dataset.id == id)
            .expect("successful dataset remains assigned");
        let record = records
            .iter_mut()
            .find(|record| record.dataset_id == id)
            .expect("successful record remains present");
        match verify_determinism(dataset, record, &embedder, &temporary_dir) {
            Ok(()) => determinism_checked += 1,
            Err(error) => {
                record.status = "failed".into();
                record.error = Some(error.to_string());
            }
        }
    }
    let _ = fs::remove_dir(&temporary_dir);

    let success = records
        .iter()
        .filter(|record| record.status == "success")
        .count();
    let schema_mismatch = records
        .iter()
        .filter(|record| record.status == "schema_mismatch")
        .count();
    let failed = records.len() - success - schema_mismatch;
    Ok(RegressionEmbeddingBatchManifest {
        format: "dope-regression-embedding-batch".into(),
        version: 2,
        method: options.method.into(),
        dataset_root: options.dataset_root.into(),
        output_dir: options.output_dir.into(),
        shard_index: options.shard_index,
        shard_count: options.shard_count,
        discovered: discovered.len(),
        dataset_ids_sha256,
        assigned: assigned.len(),
        success,
        schema_mismatch,
        failed,
        determinism_checked,
        router,
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_and_method_names_are_restricted() {
        assert!(valid_dataset_id("0123456789abcdef"));
        assert!(valid_dataset_id("0123456789ABCDEF"));
        assert!(!valid_dataset_id("synth_0123456789"));
        assert!(valid_method("attention-small"));
        assert!(!valid_method("attention/small"));
    }
}
