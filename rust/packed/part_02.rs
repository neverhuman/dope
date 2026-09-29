

impl PackedCorpus {
    pub fn open(root: &Path, verify: bool) -> Result<Self> {
        let manifest_path = root.join("manifest.json");
        let bytes = fs::read(&manifest_path).map_err(|error| io_error(&manifest_path, error))?;
        let manifest: PackedManifest = serde_json::from_slice(&bytes)?;
        if manifest.format != "dope-packed-corpus"
            || manifest.version != 2
            || manifest.checksum != manifest_checksum(&manifest)?
        {
            return Err(DopeError::Data(
                "packed manifest checksum or version invalid".into(),
            ));
        }
        let mut maps = Vec::with_capacity(manifest.shards.len());
        for shard in &manifest.shards {
            let path = root.join(&shard.file);
            let file = File::open(&path).map_err(|error| io_error(&path, error))?;
            // SAFETY: mappings are read-only, the file handle remains valid for map
            // creation, and the packed corpus is documented immutable after open.
            let map =
                unsafe { MmapOptions::new().map(&file) }.map_err(|error| io_error(&path, error))?;
            if verify
                && (map.len() != shard.bytes
                    || blake3::hash(&map).to_hex().as_str() != shard.checksum)
            {
                return Err(DopeError::Data(format!(
                    "packed shard checksum mismatch: {}",
                    shard.file
                )));
            }
            maps.push(map);
        }
        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            maps,
        })
    }

    pub fn dataset_ids(&self, split: Option<&str>) -> Vec<&str> {
        let mut ids: Vec<_> = self
            .manifest
            .datasets
            .iter()
            .filter(|record| {
                record.partition == "train" && split.is_none_or(|name| record.split == name)
            })
            .map(|record| record.dataset_id.as_str())
            .collect();
        ids.sort_unstable();
        ids
    }

    fn decode_column(&self, shard: usize, descriptor: &ColumnDescriptor) -> Result<Vec<f32>> {
        let map = &self.maps[shard];
        let start = descriptor.offset as usize;
        let data_end = start + descriptor.data_bytes as usize;
        let end = start + descriptor.length as usize;
        let data = map
            .get(start..data_end)
            .ok_or_else(|| DopeError::Data("packed column offset is outside shard".into()))?;
        let mut values: Vec<f32> = match &descriptor.encoding {
            ColumnEncoding::Bits => (0..descriptor.rows)
                .map(|index| ((data[index / 8] >> (index % 8)) & 1) as f32)
                .collect(),
            ColumnEncoding::Grid {
                values,
                index_bytes,
            } => {
                if *index_bytes == 1 {
                    data.iter()
                        .take(descriptor.rows)
                        .map(|index| values[*index as usize])
                        .collect()
                } else {
                    data.chunks_exact(2)
                        .take(descriptor.rows)
                        .map(|bytes| values[u16::from_le_bytes(bytes.try_into().unwrap()) as usize])
                        .collect()
                }
            }
            ColumnEncoding::Uint16 { .. } => data
                .chunks_exact(2)
                .take(descriptor.rows)
                .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()) as f32 / u16::MAX as f32)
                .collect(),
            ColumnEncoding::Float32 { .. } => data
                .chunks_exact(4)
                .take(descriptor.rows)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect(),
        };
        if descriptor.mask_bytes > 0 {
            let mask = map
                .get(data_end..end)
                .ok_or_else(|| DopeError::Data("packed missingness mask outside shard".into()))?;
            for index in 0..descriptor.rows {
                if (mask[index / 8] >> (index % 8)) & 1 == 1 {
                    values[index] = f32::NAN;
                }
            }
        }
        Ok(values)
    }

    pub fn read_partition(&self, dataset_id: &str, partition: &str) -> Result<Table> {
        let record = self
            .manifest
            .datasets
            .iter()
            .find(|record| record.dataset_id == dataset_id && record.partition == partition)
            .ok_or_else(|| {
                DopeError::Data(format!("unknown packed dataset {dataset_id}/{partition}"))
            })?;
        let mut columns: Vec<Vec<f32>> = record
            .columns
            .iter()
            .map(|descriptor| self.decode_column(record.shard, descriptor))
            .collect::<Result<_>>()?;
        let target = columns
            .pop()
            .ok_or_else(|| DopeError::Data("packed dataset has no target".into()))?;
        Ok(Table {
            rows: record.rows,
            features: columns.len(),
            columns,
            target,
        })
    }

    pub fn read(&self, dataset_id: &str) -> Result<Table> {
        self.read_partition(dataset_id, "train")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn bit_and_quantized_column_round_trips() {
        let binary = vec![0.0, 1.0, f32::NAN, 1.0, 0.0];
        let (payload, descriptor) = pack_column(&binary);
        assert!(matches!(descriptor.encoding, ColumnEncoding::Bits));
        assert_eq!(payload.len(), 2);
        let continuous: Vec<f32> = (0..100).map(|index| index as f32 / 99.0).collect();
        let (_, descriptor) = pack_column(&continuous);
        assert!(matches!(descriptor.encoding, ColumnEncoding::Uint16 { .. }));
    }

    #[test]
    fn mmap_corpus_round_trip_and_checksum() {
        let root = std::env::temp_dir().join(format!("dope-rust-pack-{}", std::process::id()));
        let dataset = root.join("corpus/dataset");
        let packed_path = root.join("packed");
        fs::create_dir_all(&dataset).unwrap();
        let mut file = File::create(dataset.join("train.csv")).unwrap();
        let mut test = File::create(dataset.join("test.csv")).unwrap();
        for row in 0..100 {
            let first = if row == 7 {
                "nan".to_string()
            } else {
                (row as f32 / 99.0).to_string()
            };
            writeln!(
                file,
                "{first},{},{}",
                row % 2,
                0.2 + 0.6 * row as f32 / 99.0
            )
            .unwrap();
            if row < 10 {
                writeln!(test, "0.1,0,0.2").unwrap();
            }
        }
        let split = crate::corpus::build_split_manifest(&root.join("corpus"), 7).unwrap();
        let manifest = pack_corpus(&split, &packed_path, 128, false).unwrap();
        assert!(!manifest.shards.is_empty());
        assert!(
            manifest
                .datasets
                .iter()
                .all(|record| { record.partition == "train" && record.split != "test" })
        );
        let reader = PackedCorpus::open(&packed_path, true).unwrap();
        let restored = reader.read("dataset").unwrap();
        assert_eq!(restored.rows, 100);
        assert_eq!(restored.features, 2);
        assert!(restored.columns[0][7].is_nan());
        assert!((restored.columns[0][50] - 50.0 / 99.0).abs() <= 1.0 / u16::MAX as f32 + 1e-6);
        fs::remove_dir_all(&root).unwrap();
    }
}
