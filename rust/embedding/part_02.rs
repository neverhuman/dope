

impl DatasetEmbedder {
    pub fn load(router_bundle_path: &Path, router_evidence_path: &Path) -> Result<Self> {
        let (bundle, qualification) =
            load_qualified_router(router_bundle_path, router_evidence_path)?;
        Ok(Self {
            router: QuantizedRouter::new(bundle)?,
            qualification,
        })
    }

    pub fn qualification(&self) -> &EmbeddingRouterQualification {
        &self.qualification
    }

    pub fn embed_csv(
        &self,
        csv_path: &Path,
        selector: TargetSelector<'_>,
        task: Task,
    ) -> Result<DatasetActionEmbedding> {
        self.embed_csv_with_metadata(csv_path, selector, task, MetadataMode::Public)
    }

    pub fn embed_csv_with_metadata(
        &self,
        csv_path: &Path,
        selector: TargetSelector<'_>,
        task: Task,
        metadata_mode: MetadataMode,
    ) -> Result<DatasetActionEmbedding> {
        let loaded = load_numeric_csv(csv_path, selector, task)?;
        let sketch = DatasetSketch::from_train(&loaded.table, task);
        if sketch.version != DATASET_SKETCH_VERSION {
            return Err(DopeError::Data("unsupported dataset sketch version".into()));
        }
        let router_embedding = self.router.action_embedding(&sketch)?;

        let mut vector = router_embedding.standardized_sketch;
        let mut component_names = (0..vector.len())
            .map(|index| format!("sketch.standardized.{index:03}"))
            .collect::<Vec<_>>();
        let sketch_range = ComponentRange {
            offset: 0,
            length: vector.len(),
        };
        let mut candidate_order = Vec::with_capacity(router_embedding.candidates.len());
        let mut candidate_offsets = Vec::with_capacity(router_embedding.candidates.len());
        let mut predictions = Vec::with_capacity(router_embedding.candidates.len());
        for candidate in router_embedding.candidates {
            candidate_order.push(candidate.candidate_id.clone());
            let hidden_state = ComponentRange {
                offset: vector.len(),
                length: candidate.hidden_state.len(),
            };
            component_names.extend((0..candidate.hidden_state.len()).map(|index| {
                format!(
                    "candidate.{}.hidden_state.{index:03}",
                    candidate.candidate_id
                )
            }));
            vector.extend(candidate.hidden_state);
            let standardized_actions = ComponentRange {
                offset: vector.len(),
                length: candidate.standardized_actions.len(),
            };
            component_names.extend(
                ROUTER_ACTION_NAMES
                    .iter()
                    .map(|name| format!("candidate.{}.action.{name}", candidate.candidate_id)),
            );
            vector.extend(candidate.standardized_actions);
            candidate_offsets.push(CandidateComponentOffsets {
                candidate_id: candidate.candidate_id,
                hidden_state,
                standardized_actions,
            });
            predictions.push(candidate.prediction);
        }
        let dimension = vector.len();
        if dimension != self.qualification.dimension {
            return Err(DopeError::Data(format!(
                "router produced dimension {dimension}, expected {}",
                self.qualification.dimension
            )));
        }
        let schema_sha256 = hashes(&canonical_json(&serde_json::json!({
            "format": "dope-dataset-action-embedding-schema",
            "version": ACTION_EMBEDDING_VERSION,
            "component_names": component_names,
        }))?)
        .sha256;
        Ok(DatasetActionEmbedding {
            format: "dope-dataset-action-embedding".into(),
            version: ACTION_EMBEDDING_VERSION,
            dimension,
            schema_sha256,
            router_bundle_sha256: self.qualification.router_bundle_sha256.clone(),
            router_evidence_sha256: self.qualification.router_evidence_sha256.clone(),
            task,
            rows: loaded.table.rows,
            features: loaded.table.features,
            normalization: if metadata_mode == MetadataMode::RestrictedResearch {
                loaded.normalization
            } else {
                DatasetNormalization::public_positions(loaded.table.features)
            },
            metadata_mode,
            candidate_order,
            components: EmbeddingComponentOffsets {
                standardized_sketch: sketch_range,
                candidates: candidate_offsets,
            },
            component_names,
            vector,
            predictions,
        })
    }
}

pub fn embed_dataset(
    csv_path: &Path,
    selector: TargetSelector<'_>,
    task: Task,
    router_bundle_path: &Path,
    router_evidence_path: &Path,
) -> Result<DatasetActionEmbedding> {
    DatasetEmbedder::load(router_bundle_path, router_evidence_path)?
        .embed_csv(csv_path, selector, task)
}

pub fn embed_dataset_with_metadata(
    csv_path: &Path,
    selector: TargetSelector<'_>,
    task: Task,
    router_bundle_path: &Path,
    router_evidence_path: &Path,
    metadata_mode: MetadataMode,
) -> Result<DatasetActionEmbedding> {
    DatasetEmbedder::load(router_bundle_path, router_evidence_path)?.embed_csv_with_metadata(
        csv_path,
        selector,
        task,
        metadata_mode,
    )
}

pub fn write_embedding_csv(path: &Path, embedding: &DatasetActionEmbedding) -> Result<()> {
    let file = File::create(path).map_err(|error| io_error(path, error))?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);
    writer.write_record(&embedding.component_names)?;
    let values = embedding
        .vector
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    writer.write_record(values)?;
    writer.flush().map_err(|error| io_error(path, error))?;
    Ok(())
}

#[cfg(test)]
include!("tests.rs");
