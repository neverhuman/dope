// Keep the D00 ordered typed-cell digest identical across both boundaries.
fn fit_row_digest(table: &TypedTable, fit_rows: &[usize]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"prepared-view-v1:fit-rows\0");
    hash.update((fit_rows.len() as u64).to_le_bytes());
    hash.update((table.columns.len() as u64).to_le_bytes());
    for row in fit_rows {
        hash.update((*row as u64).to_le_bytes());
        for column in &table.columns {
            match &column[*row] {
                Cell::Number(x) => {
                    hash.update([0]);
                    hash.update(x.to_bits().to_le_bytes());
                }
                Cell::Boolean(x) => hash.update([1, u8::from(*x)]),
                Cell::Category(x) => {
                    hash.update([2]);
                    hash.update((x.len() as u64).to_le_bytes());
                    hash.update(x.as_bytes());
                }
                Cell::Mask(state) => hash.update([3, *state as u8]),
            }
        }
    }
    format!("{:x}", hash.finalize())
}

impl FittedTransforms {
    pub fn fit_transform(table: &TypedTable, fit_rows: &[usize]) -> Result<Self> {
        fit_transform(table, fit_rows)
    }

    /// Private provenance of ordered fitting row indices and original typed cells.
    /// Historical v1 states still decode, but have no attested fit partition.
    pub fn fit_row_digest(&self) -> Option<&str> {
        self.fit_row_digest.as_deref()
    }

    pub fn fit_only(&self) -> bool {
        self.fit_row_digest.is_some()
    }

    /// Admit a retained map against the experiment's independently frozen digest.
    /// Ordinary `reload` remains available for historical sampling-only states.
    pub fn reload_for_fit(bytes: &[u8], expected_fit_row_digest: &str) -> Result<Self> {
        let state = Self::reload(bytes)?;
        if !crate::schema::prepared_digest(expected_fit_row_digest)
            || state.fit_row_digest() != Some(expected_fit_row_digest)
        {
            return Err(SchemaError::new("fit_row_digest_mismatch", None, None));
        }
        Ok(state)
    }
}
