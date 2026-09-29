use std::fs::{self, File};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};

use crate::access::{AccessBroker, AccessRole};
use crate::corpus::SplitManifest;
use crate::data::Table;
use crate::error::{DopeError, Result, io_error};
use crate::model::Task;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "encoding", rename_all = "snake_case")]
pub enum ColumnEncoding {
    Bits,
    Grid { values: Vec<f32>, index_bytes: u8 },
    Uint16 { quantization_utility: f32 },
    Float32 { quantization_utility: f32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColumnDescriptor {
    pub offset: u64,
    pub length: u64,
    pub data_bytes: u64,
    pub mask_bytes: u64,
    pub rows: usize,
    pub encoding: ColumnEncoding,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedDataset {
    pub dataset_id: String,
    pub partition: String,
    pub task: Task,
    pub split: String,
    pub rows: usize,
    pub cols: usize,
    pub shard: usize,
    pub columns: Vec<ColumnDescriptor>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardDescriptor {
    pub file: String,
    pub bytes: usize,
    pub checksum: String,
    pub archive: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedManifest {
    pub format: String,
    pub version: u8,
    pub target_position: String,
    pub shard_target_bytes: usize,
    pub split_manifest_checksum: String,
    pub datasets: Vec<PackedDataset>,
    pub shards: Vec<ShardDescriptor>,
    pub checksum: String,
}

fn pack_bits(values: impl Iterator<Item = bool>, rows: usize) -> Vec<u8> {
    let mut packed = vec![0u8; rows.div_ceil(8)];
    for (index, value) in values.enumerate() {
        if value {
            packed[index / 8] |= 1 << (index % 8);
        }
    }
    packed
}

fn unique_values(values: &[f32]) -> Vec<f32> {
    let mut unique = values.to_vec();
    unique.sort_by(f32::total_cmp);
    unique.dedup_by(|left, right| (*left - *right).abs() <= 1e-7);
    unique
}

fn correlation(left: &[f32], right: &[f32]) -> f64 {
    let pairs: Vec<(f64, f64)> = left
        .iter()
        .zip(right)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (f64::from(*x), f64::from(*y)))
        .collect();
    if pairs.len() < 3 {
        return 0.0;
    }
    let x_mean = pairs.iter().map(|(x, _)| x).sum::<f64>() / pairs.len() as f64;
    let y_mean = pairs.iter().map(|(_, y)| y).sum::<f64>() / pairs.len() as f64;
    let numerator = pairs
        .iter()
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>();
    let x_norm = pairs.iter().map(|(x, _)| (x - x_mean).powi(2)).sum::<f64>();
    let y_norm = pairs.iter().map(|(_, y)| (y - y_mean).powi(2)).sum::<f64>();
    if x_norm * y_norm <= 1e-18 {
        0.0
    } else {
        numerator / (x_norm * y_norm).sqrt()
    }
}

fn quantization_utility(values: &[f32], quantized: &[u16], target: &[f32]) -> f32 {
    let restored: Vec<f32> = quantized
        .iter()
        .map(|value| *value as f32 / u16::MAX as f32)
        .collect();
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len().max(1) as f32;
    let rmse = (values
        .iter()
        .zip(&restored)
        .map(|(value, restored)| (value - restored).powi(2))
        .sum::<f32>()
        / values.len().max(1) as f32)
        .sqrt();
    let numeric = (1.0 - rmse / variance.sqrt().max(1e-6)).clamp(0.0, 1.0);
    let original_signal = correlation(values, target);
    let restored_signal = correlation(&restored, target);
    let predictive = (1.0
        - (original_signal - restored_signal).abs() / original_signal.abs().max(0.05))
    .clamp(0.0, 1.0) as f32;
    numeric.min(predictive)
}

fn pack_column_for(column: &[f32], target: &[f32]) -> (Vec<u8>, ColumnDescriptor) {
    let observed: Vec<f32> = column
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let mut sorted = observed.clone();
    sorted.sort_by(f32::total_cmp);
    let fill = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let values: Vec<f32> = column
        .iter()
        .map(|value| {
            if value.is_nan() {
                fill
            } else {
                value.clamp(0.0, 1.0)
            }
        })
        .collect();
    let unique = unique_values(&values);
    let (mut payload, encoding) = if unique.len() <= 2
        && unique
            .iter()
            .all(|value| *value <= 1e-7 || *value >= 1.0 - 1e-7)
    {
        (
            pack_bits(values.iter().map(|value| *value >= 0.5), values.len()),
            ColumnEncoding::Bits,
        )
    } else if unique.len() <= 32 {
        let index_bytes = if unique.len() <= 256 { 1 } else { 2 };
        let mut encoded = Vec::with_capacity(values.len() * index_bytes as usize);
        for value in &values {
            let index = unique
                .binary_search_by(|probe| probe.total_cmp(value))
                .unwrap();
            if index_bytes == 1 {
                encoded.push(index as u8);
            } else {
                encoded.extend_from_slice(&(index as u16).to_le_bytes());
            }
        }
        (
            encoded,
            ColumnEncoding::Grid {
                values: unique,
                index_bytes,
            },
        )
    } else {
        let quantized: Vec<u16> = values
            .iter()
            .map(|value| (value * u16::MAX as f32).round() as u16)
            .collect();
        let utility = quantization_utility(&values, &quantized, target);
        if utility >= 0.999 {
            let mut encoded = Vec::with_capacity(2 * values.len());
            for value in quantized {
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            (
                encoded,
                ColumnEncoding::Uint16 {
                    quantization_utility: utility,
                },
            )
        } else {
            let mut encoded = Vec::with_capacity(4 * values.len());
            for value in values {
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            (
                encoded,
                ColumnEncoding::Float32 {
                    quantization_utility: utility,
                },
            )
        }
    };
    let data_bytes = payload.len();
    let mask = pack_bits(column.iter().map(|value| value.is_nan()), column.len());
    let mask_bytes = if column.iter().any(|value| value.is_nan()) {
        mask.len()
    } else {
        0
    };
    if mask_bytes > 0 {
        payload.extend_from_slice(&mask);
    }
    let descriptor = ColumnDescriptor {
        offset: 0,
        length: payload.len() as u64,
        data_bytes: data_bytes as u64,
        mask_bytes: mask_bytes as u64,
        rows: column.len(),
        encoding,
    };
    (payload, descriptor)
}

#[cfg(test)]
fn pack_column(column: &[f32]) -> (Vec<u8>, ColumnDescriptor) {
    pack_column_for(column, column)
}

fn manifest_checksum(manifest: &PackedManifest) -> Result<String> {
    let mut clone = manifest.clone();
    clone.checksum.clear();
    Ok(blake3::hash(&serde_json::to_vec(&clone)?)
        .to_hex()
        .to_string())
}

pub fn pack_corpus(
    manifest: &SplitManifest,
    out: &Path,
    shard_bytes: usize,
    archive_zstd: bool,
) -> Result<PackedManifest> {
    if shard_bytes == 0 {
        return Err(DopeError::Data("shard size must be positive".into()));
    }
    fs::create_dir_all(out).map_err(|error| io_error(out, error))?;
    let broker = AccessBroker::new(manifest, AccessRole::SurfaceTrainer, false)?;
    let mut records: Vec<_> = manifest
        .datasets
        .iter()
        .filter(|record| record.duplicate_of.is_none() && record.split != "test")
        .collect();
    records.sort_by_key(|record| {
        (
            record.task.code(),
            record.cols.next_power_of_two(),
            record.dataset_id.clone(),
        )
    });
    let mut payloads = vec![Vec::<u8>::new()];
    let mut datasets = Vec::new();
    for record in records {
        for (partition, path) in [("train", record.path.join("train.csv"))] {
            broker.authorize(record, partition)?;
            if !path.is_file() {
                continue;
            }
            let table = Table::read_csv(&path, record.task)?;
            let mut all_columns = table.columns.clone();
            all_columns.push(table.target.clone());
            let packed: Vec<_> = all_columns
                .iter()
                .map(|column| pack_column_for(column, &table.target))
                .collect();
            let required = packed
                .iter()
                .map(|(payload, _)| payload.len())
                .sum::<usize>();
            if !payloads.last().expect("one initial shard").is_empty()
                && payloads.last().expect("one initial shard").len() + required > shard_bytes
            {
                payloads.push(Vec::with_capacity(required.min(shard_bytes)));
            }
            let shard = payloads.len() - 1;
            let mut columns = Vec::with_capacity(packed.len());
            for (payload, mut descriptor) in packed {
                descriptor.offset = payloads[shard].len() as u64;
                payloads[shard].extend_from_slice(&payload);
                columns.push(descriptor);
            }
            datasets.push(PackedDataset {
                dataset_id: record.dataset_id.clone(),
                partition: partition.into(),
                task: record.task,
                split: record.split.clone(),
                rows: table.rows,
                cols: table.features + 1,
                shard,
                columns,
            });
        }
    }
    let mut shards = Vec::with_capacity(payloads.len());
    for (index, payload) in payloads.iter().enumerate() {
        let file_name = format!("shard-{index:05}.dps");
        let path = out.join(&file_name);
        fs::write(&path, payload).map_err(|error| io_error(&path, error))?;
        let archive = if archive_zstd {
            let archive_name = format!("{file_name}.zst");
            let archive_path = out.join(&archive_name);
            let compressed = zstd::bulk::compress(payload, 9)
                .map_err(|error| DopeError::Data(format!("zstd archive failure: {error}")))?;
            fs::write(&archive_path, compressed).map_err(|error| io_error(&archive_path, error))?;
            Some(archive_name)
        } else {
            None
        };
        shards.push(ShardDescriptor {
            file: file_name,
            bytes: payload.len(),
            checksum: blake3::hash(payload).to_hex().to_string(),
            archive,
        });
    }
    let mut packed = PackedManifest {
        format: "dope-packed-corpus".into(),
        version: 2,
        target_position: "last".into(),
        shard_target_bytes: shard_bytes,
        split_manifest_checksum: manifest.checksum.clone(),
        datasets,
        shards,
        checksum: String::new(),
    };
    packed.checksum = manifest_checksum(&packed)?;
    let manifest_path = out.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&packed)?)
        .map_err(|error| io_error(&manifest_path, error))?;
    Ok(packed)
}

pub struct PackedCorpus {
    pub root: PathBuf,
    pub manifest: PackedManifest,
    maps: Vec<Mmap>,
}

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
