// Additive reference-only input transform. Historical embedding-v1 still uses
// its original per-table loader. Private headers and extrema are never public.

#[derive(Clone)]
pub struct FrozenNumericTransform {
    headers: Vec<String>,
    target_index: usize,
    task: Task,
    normalization: DatasetNormalization,
    binary_labels: Option<[f64; 2]>,
    digest: String,
}

fn frozen_input_error(code: &'static str) -> DopeError {
    DopeError::Data(code.into())
}

fn read_frozen_csv(path: &Path) -> Result<(Vec<String>, Vec<csv::StringRecord>)> {
    use std::io::Read;
    const MAX_BYTES: usize = 16 * 1024 * 1024;
    let file = File::open(path).map_err(|_| frozen_input_error("frozen_input_unreadable"))?;
    if !file
        .metadata()
        .map_err(|_| frozen_input_error("frozen_input_unreadable"))?
        .is_file()
    {
        return Err(frozen_input_error("frozen_input_carrier_invalid"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| frozen_input_error("frozen_input_unreadable"))?;
    if bytes.len() > MAX_BYTES {
        return Err(frozen_input_error("frozen_input_bytes_exceeded"));
    }
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(bytes.as_slice());
    let headers = reader
        .headers()
        .map_err(|_| frozen_input_error("frozen_input_csv_invalid"))?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if headers.is_empty() || headers.iter().collect::<BTreeSet<_>>().len() != headers.len() {
        return Err(frozen_input_error("frozen_input_schema_invalid"));
    }
    let rows = reader
        .records()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| frozen_input_error("frozen_input_csv_invalid"))?;
    if rows.is_empty() {
        return Err(frozen_input_error("frozen_input_empty"));
    }
    Ok((headers, rows))
}

impl FrozenNumericTransform {
    /// Fit once on the admitted development reference. No later input can
    /// change the feature/target ranges, label coding or reference column order.
    pub fn fit_csv(path: &Path, selector: TargetSelector<'_>, task: Task) -> Result<Self> {
        let (headers, rows) = read_frozen_csv(path)?;
        if task == Task::Binary && headers.len() < 2 {
            return Err(frozen_input_error("frozen_input_schema_invalid"));
        }
        let target_index = match selector {
            TargetSelector::Name(name) => headers.iter().position(|h| h == name),
            TargetSelector::Index(index) => (index < headers.len()).then_some(index),
        }
        .ok_or_else(|| frozen_input_error("frozen_target_missing"))?;
        let mut columns = vec![Vec::new(); headers.len()];
        for row in &rows {
            for (index, column) in columns.iter_mut().enumerate() {
                column.push(
                    parse_field(&row[index])
                        .map_err(|_| frozen_input_error("frozen_numeric_input_invalid"))?,
                );
            }
        }
        if columns[target_index].iter().any(Option::is_none) {
            return Err(frozen_input_error("frozen_target_missing"));
        }
        let ranges = columns
            .iter()
            .enumerate()
            .map(|(i, values)| {
                let range = extrema(values.iter().flatten().copied());
                NormalizationExtrema {
                    column: headers[i].clone(),
                    minimum: range.map(|r| r.0),
                    maximum: range.map(|r| r.1),
                }
            })
            .collect::<Vec<_>>();
        let binary_labels = if task == Task::Binary {
            let mut labels = columns[target_index]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            labels.sort_by(f64::total_cmp);
            labels.dedup_by(|a, b| *a == *b);
            if labels.len() != 2 {
                return Err(frozen_input_error("frozen_binary_labels_invalid"));
            }
            Some([labels[0], labels[1]])
        } else {
            None
        };
        let normalization = DatasetNormalization {
            features: ranges
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != target_index)
                .map(|(_, r)| r.clone())
                .collect(),
            target: ranges[target_index].clone(),
        };
        let body = serde_json::json!({"format":"dope-frozen-numeric-transform", "version":1,
            "headers":headers,"target_index":target_index,"task":task,
            "normalization":normalization,"binary_labels":binary_labels,
            "rule":"reference_minmax_clamped_v1"});
        let digest = hashes(&canonical_json(&body)?).sha256;
        Ok(Self {
            headers,
            target_index,
            task,
            normalization,
            binary_labels,
            digest,
        })
    }

    /// Public metadata omits source names, labels and fitted extrema. This
    /// content commitment is a coordinate identity, never a privacy certificate.
    pub fn descriptor(&self) -> serde_json::Value {
        serde_json::json!({"format":"dope-frozen-numeric-transform", "version":1,
            "state_digest":self.digest,"task":self.task,"features":self.normalization.features.len(),
            "normalization":"reference_minmax_clamped_v1","target_included":true,
            "privacy":"unmeasured","qualifies_transfer":false})
    }

    pub fn apply_csv(&self, path: &Path) -> Result<LoadedNumericDataset> {
        let (headers, rows) = read_frozen_csv(path)?;
        if headers.len() != self.headers.len() {
            return Err(frozen_input_error("frozen_input_schema_mismatch"));
        }
        let order = self
            .headers
            .iter()
            .map(|name| {
                headers
                    .iter()
                    .position(|h| h == name)
                    .ok_or_else(|| frozen_input_error("frozen_input_schema_mismatch"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut columns = vec![Vec::with_capacity(rows.len()); self.normalization.features.len()];
        let mut target = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut feature = 0;
            for (index, source_index) in order.iter().enumerate() {
                let value = parse_field(&row[*source_index])
                    .map_err(|_| frozen_input_error("frozen_numeric_input_invalid"))?;
                if index == self.target_index {
                    let value = value.ok_or_else(|| frozen_input_error("frozen_target_missing"))?;
                    target.push(if let Some(labels) = self.binary_labels {
                        if value == labels[0] {
                            0.0
                        } else if value == labels[1] {
                            1.0
                        } else {
                            return Err(frozen_input_error(
                                "frozen_binary_label_outside_reference",
                            ));
                        }
                    } else {
                        normalize(
                            value,
                            self.normalization.target.minimum.expect("fit target"),
                            self.normalization.target.maximum.expect("fit target"),
                        )
                    });
                } else {
                    let range = &self.normalization.features[feature];
                    let normalized = match (value, range.minimum, range.maximum) {
                        (None, _, _) => f32::NAN,
                        (Some(v), Some(lo), Some(hi)) => normalize(v, lo, hi),
                        _ => {
                            return Err(frozen_input_error(
                                "frozen_feature_without_reference_support",
                            ))
                        }
                    };
                    columns[feature].push(normalized);
                    feature += 1;
                }
            }
        }
        Ok(LoadedNumericDataset {
            table: Table {
                rows: rows.len(),
                features: columns.len(),
                columns,
                target,
            },
            normalization: self.normalization.clone(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrozenActionEmbedding {
    pub format: String,
    pub version: u8,
    pub frozen_transform: serde_json::Value,
    pub basis_sha256: String,
    pub capability: EncoderCapability,
    pub embedding: DatasetActionEmbedding,
}

impl DatasetEmbedder {
    /// Uses the same frozen router/action basis for every input. The additive
    /// wrapper distinguishes reference-only normalization from historical v1.
    pub fn embed_csv_frozen(
        &self,
        path: &Path,
        transform: &FrozenNumericTransform,
        metadata: MetadataMode,
    ) -> Result<FrozenActionEmbedding> {
        let embedding =
            self.embed_loaded_dataset(transform.apply_csv(path)?, transform.task, metadata)?;
        let basis = serde_json::json!({"format":"dope-frozen-action-basis","version":1,
            "transform_digest":transform.digest,"schema_sha256":embedding.schema_sha256,
            "router_bundle_sha256":embedding.router_bundle_sha256,
            "router_evidence_sha256":embedding.router_evidence_sha256,
            "candidate_order":embedding.candidate_order,"components":embedding.components,
            "normalization":"reference_minmax_clamped_then_frozen_router_zscore"});
        Ok(FrozenActionEmbedding {
            format: "dope-frozen-action-embedding".into(),
            version: 1,
            frozen_transform: transform.descriptor(),
            basis_sha256: hashes(&canonical_json(&basis)?).sha256,
            capability: EncoderCapability::native_action(&embedding),
            embedding,
        })
    }
}
