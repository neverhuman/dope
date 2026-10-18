pub(crate) fn prepared_error(code: &'static str) -> SchemaError {
    SchemaError::new(code, None, None)
}

pub(crate) fn prepared_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

impl PreparedMetadata {
    pub fn validate(&self) -> Result<()> {
        use std::collections::BTreeSet;
        let reject = || prepared_error("invalid_prepared_contract");
        if self.horizon.steps == 0 || self.columns.len() < 2 || self.columns.len() > MAX_COLUMNS {
            return Err(reject());
        }
        if serde_json::to_vec(self).map_err(|_| reject())?.len() > super::MAX_SCHEMA_BYTES {
            return Err(prepared_error("prepared_metadata_bytes_exceeded"));
        }
        let mut sources = BTreeSet::new();
        for (index, column) in self.columns.iter().enumerate() {
            let target = index + 1 == self.columns.len();
            if column.id != (index + 1) as u32
                || !super::opaque_id(&column.source_id)
                || !sources.insert(&column.source_id)
                || (column.role == PreparedRole::Target) != target
                || column
                    .bounds
                    .lower
                    .iter()
                    .chain(&column.bounds.upper)
                    .any(|x| !x.is_finite())
                || matches!((column.bounds.lower, column.bounds.upper), (Some(a), Some(b)) if a > b)
            {
                return Err(reject());
            }
            let transform_valid = match &column.transform {
                PreparedTransform::Affine { offset, scale } => {
                    column.dtype == PreparedType::Continuous
                        && offset.is_finite()
                        && scale.is_finite()
                        && *scale > 0.0
                        && (!target || self.task == TaskKind::Regression)
                }
                PreparedTransform::Identity {} => column.dtype != PreparedType::CategoryCode,
                PreparedTransform::CategoryIndex { categories } => {
                    !target
                        && column.dtype == PreparedType::CategoryCode
                        && !categories.is_empty()
                        && categories.len() <= 4096
                        && categories
                            .iter()
                            .all(|s| !s.is_empty() && s.len() <= super::DEFAULT_MAX_FIELD_BYTES)
                        && categories.iter().collect::<BTreeSet<_>>().len() == categories.len()
                        && column.bounds.lower.is_none()
                        && column.bounds.upper.is_none()
                }
            };
            if !transform_valid {
                return Err(reject());
            }
        }
        let mut group_ids = BTreeSet::new();
        let mut members = BTreeSet::new();
        let feature_count = self.columns.len() - 1;
        if self.group_map.is_empty() || self.group_map.len() > feature_count {
            return Err(prepared_error("prepared_group_mismatch"));
        }
        for group in &self.group_map {
            if group.id == 0
                || !group_ids.insert(group.id)
                || group.column_ids.is_empty()
                || group
                    .column_ids
                    .iter()
                    .any(|id| *id == 0 || *id as usize > feature_count || !members.insert(*id))
            {
                return Err(prepared_error("prepared_group_mismatch"));
            }
        }
        if members.len() != feature_count {
            return Err(prepared_error("prepared_group_mismatch"));
        }
        let target_column = self.columns.last().ok_or_else(reject)?;
        match (&self.task, &self.target) {
            (TaskKind::Regression, PreparedTarget::Regression { units }) => {
                if units.trim().is_empty()
                    || units.len() > 128
                    || target_column.dtype != PreparedType::Continuous
                    || !matches!(target_column.transform, PreparedTransform::Affine { .. })
                {
                    return Err(reject());
                }
            }
            (TaskKind::Binary | TaskKind::Multiclass, PreparedTarget::Discrete { class_order }) => {
                let minimum = if self.task == TaskKind::Binary { 2 } else { 3 };
                if class_order.len() < minimum
                    || class_order.len() > 4096
                    || (self.task == TaskKind::Binary && class_order.len() != 2)
                    || class_order.iter().collect::<BTreeSet<_>>().len() != class_order.len()
                    || target_column.dtype != PreparedType::Integer
                    || !matches!(target_column.transform, PreparedTransform::Identity {})
                {
                    return Err(prepared_error("prepared_class_mismatch"));
                }
            }
            _ => return Err(prepared_error("prepared_class_mismatch")),
        }
        let p = &self.provenance;
        if [
            &p.fit_digest,
            &p.entity_digest,
            &p.split_digest,
            &p.column_digest,
            &p.implementation_hash,
        ]
        .iter()
        .any(|x| !prepared_digest(x))
            || self.policies != PreparedPolicies::default()
            || self.projection.state != "private"
            || self.projection.public != "opaque_only"
            || self.capabilities != PreparedCapabilities::default()
            || self
                .rights
                .evidence_digest
                .as_ref()
                .is_some_and(|s| !prepared_digest(s))
            || (self.rights.state == RightsState::Approved && self.rights.evidence_digest.is_none())
        {
            return Err(reject());
        }
        Ok(())
    }
}
