/// Magic of a table artifact: a binary intake plan followed by a packed DPK3
/// kernel. Every byte that sampling reads is in the file.
pub const TABLE_MAGIC: &[u8; 8] = b"DOPETAB1";
/// Version 2 stores date bounds as f64 (version 1, never released, used f32).
const PLAN_VERSION: u8 = 2;

#[derive(Default)]
struct PlanWriter {
    bytes: Vec<u8>,
    dictionary_bytes: usize,
}

impl PlanWriter {
    fn varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.bytes.push((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        self.bytes.push(value as u8);
    }

    fn float(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn double(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn text(&mut self, value: &str) {
        let before = self.bytes.len();
        self.varint(value.len() as u64);
        self.bytes.extend_from_slice(value.as_bytes());
        self.dictionary_bytes += self.bytes.len() - before;
    }
}

struct PlanReader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> PlanReader<'a> {
    fn byte(&mut self) -> Result<u8> {
        let value = *self.data.get(self.at).ok_or_else(|| DopeError::Codec("table_plan_truncated".into()))?;
        self.at += 1;
        Ok(value)
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(DopeError::Codec("table_plan_varint_overflow".into()))
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(length).filter(|end| *end <= self.data.len());
        let end = end.ok_or_else(|| DopeError::Codec("table_plan_truncated".into()))?;
        let slice = &self.data[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn float(&mut self) -> Result<f32> {
        let value = f32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        if value.is_finite() { Ok(value) } else { Err(DopeError::Codec("table_plan_non_finite".into())) }
    }

    fn double(&mut self) -> Result<f64> {
        let value = f64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes"));
        if value.is_finite() { Ok(value) } else { Err(DopeError::Codec("table_plan_non_finite".into())) }
    }

    fn count(&mut self, bound: usize) -> Result<usize> {
        let value = self.varint()? as usize;
        if value > bound {
            return Err(DopeError::Codec("table_plan_count_exceeds_bound".into()));
        }
        Ok(value)
    }

    fn text(&mut self) -> Result<String> {
        let length = self.count(1 << 16)?;
        String::from_utf8(self.take(length)?.to_vec()).map_err(|_| DopeError::Codec("table_plan_text_not_utf8".into()))
    }
}

/// Binary plan; the second value counts the bytes of names and levels.
fn encode_plan(plan: &TablePlan) -> (Vec<u8>, usize) {
    let mut writer = PlanWriter::default();
    writer.bytes.push(PLAN_VERSION);
    writer.bytes.push(plan.task.code());
    writer.varint(plan.columns.len() as u64);
    writer.varint(plan.target as u64);
    for column in &plan.columns {
        writer.text(&column.name);
        match &column.encoding {
            Encoding::Numeric { integer, decimals, log, low, high } => {
                let flags = u8::from(*integer) | (u8::from(log.is_some()) << 1)
                    | (u8::from(log.is_some_and(|l| l.mirrored)) << 2);
                writer.bytes.extend_from_slice(&[0, flags, *decimals]);
                writer.float(*low);
                writer.float(*high);
                if let Some(scale) = log {
                    writer.float(scale.origin);
                    writer.float(scale.scale);
                }
            }
            Encoding::Category { levels } => {
                writer.bytes.push(1);
                writer.varint(levels.len() as u64);
                for level in levels {
                    writer.text(level);
                }
            }
            Encoding::Date { time, low, high } => {
                writer.bytes.extend_from_slice(&[2, u8::from(*time)]);
                writer.double(*low);
                writer.double(*high);
            }
        }
    }
    (writer.bytes, writer.dictionary_bytes)
}

fn decode_plan(data: &[u8]) -> Result<TablePlan> {
    let mut reader = PlanReader { data, at: 0 };
    if reader.byte()? != PLAN_VERSION {
        return Err(DopeError::Codec("table_plan_version_unsupported".into()));
    }
    let task = match reader.byte()? {
        0 => Task::Regression,
        1 => Task::Binary,
        _ => return Err(DopeError::Codec("table_plan_task_unknown".into())),
    };
    let count = reader.count(4096)?;
    let target = reader.count(count.saturating_sub(1))?;
    let mut columns = Vec::with_capacity(count);
    for _ in 0..count {
        let name = reader.text()?;
        let encoding = match reader.byte()? {
            0 => {
                let flags = reader.byte()?;
                let decimals = reader.byte()?.min(6);
                let (low, high) = (reader.float()?, reader.float()?);
                let log = if flags & 2 != 0 {
                    let (origin, scale) = (reader.float()?, reader.float()?);
                    Some(LogScale { origin, scale, mirrored: flags & 4 != 0 })
                } else {
                    None
                };
                Encoding::Numeric { integer: flags & 1 != 0, decimals, log, low, high }
            }
            1 => {
                let levels = (0..reader.count(MAX_LEVELS)?).map(|_| reader.text()).collect::<Result<Vec<_>>>()?;
                if levels.is_empty() {
                    return Err(DopeError::Codec("table_plan_category_empty".into()));
                }
                Encoding::Category { levels }
            }
            2 => {
                let time = reader.byte()? != 0;
                Encoding::Date { time, low: reader.double()?, high: reader.double()? }
            }
            _ => return Err(DopeError::Codec("table_plan_encoding_unknown".into())),
        };
        columns.push(ColumnPlan { name, encoding });
    }
    if reader.at != data.len() || columns.len() < 2 {
        return Err(DopeError::Codec("table_plan_malformed".into()));
    }
    Ok(TablePlan { task, columns, target })
}

/// Byte accounting of a table artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TableAccounting {
    pub total_bytes: usize,
    pub kernel_bytes: usize,
    pub plan_bytes: usize,
    /// Uncompressed size of the column names and category labels in the plan.
    pub dictionary_bytes: usize,
    pub framing_bytes: usize,
}

/// Plan section: one byte (0 raw, 1 zstd) then the plan, compressed when that
/// is smaller. Names and category labels usually dominate the plan.
fn plan_section(plan: &TablePlan) -> (Vec<u8>, usize) {
    let (raw, dictionary_bytes) = encode_plan(plan);
    let compressed = zstd::bulk::compress(&raw, 19).ok().filter(|c| c.len() + 4 < raw.len());
    let mut section = Vec::with_capacity(raw.len() + 5);
    match compressed {
        Some(compressed) => {
            section.push(1);
            let mut writer = PlanWriter::default();
            writer.varint(raw.len() as u64);
            section.extend_from_slice(&writer.bytes);
            section.extend_from_slice(&compressed);
        }
        None => {
            section.push(0);
            section.extend_from_slice(&raw);
        }
    }
    (section, dictionary_bytes)
}

fn read_plan_section(section: &[u8]) -> Result<TablePlan> {
    match section.split_first() {
        Some((0, raw)) => decode_plan(raw),
        Some((1, rest)) => {
            let mut reader = PlanReader { data: rest, at: 0 };
            let length = reader.count(1 << 24)?;
            let raw = zstd::bulk::decompress(&rest[reader.at..], length)
                .map_err(|_| DopeError::Codec("table_plan_decompression_failed".into()))?;
            decode_plan(&raw)
        }
        _ => Err(DopeError::Codec("table_plan_section_unknown".into())),
    }
}

pub fn write_table_artifact(plan: &TablePlan, kernel: &[u8]) -> (Vec<u8>, TableAccounting) {
    let (plan_bytes, dictionary_bytes) = plan_section(plan);
    let mut writer = PlanWriter::default();
    writer.bytes.extend_from_slice(TABLE_MAGIC);
    writer.varint(plan_bytes.len() as u64);
    let framing_bytes = writer.bytes.len();
    writer.bytes.extend_from_slice(&plan_bytes);
    writer.bytes.extend_from_slice(kernel);
    let accounting = TableAccounting {
        total_bytes: writer.bytes.len(),
        kernel_bytes: kernel.len(),
        plan_bytes: plan_bytes.len(),
        dictionary_bytes,
        framing_bytes,
    };
    (writer.bytes, accounting)
}

pub fn read_table_artifact(bytes: &[u8]) -> Result<(TablePlan, Kernel, TableAccounting)> {
    if !bytes.starts_with(TABLE_MAGIC) {
        return Err(DopeError::Codec("not_a_dope_table_artifact".into()));
    }
    let mut reader = PlanReader { data: bytes, at: TABLE_MAGIC.len() };
    let plan_length = reader.count(bytes.len())?;
    let framing_bytes = reader.at;
    let plan_data = reader.take(plan_length)?;
    let plan = read_plan_section(plan_data)?;
    let kernel_bytes = &bytes[reader.at..];
    let kernel = crate::codec::decode_kernel(kernel_bytes)?;
    if kernel.features as usize != plan.feature_count() || kernel.task != plan.task {
        return Err(DopeError::Codec("table_plan_kernel_mismatch".into()));
    }
    let (_, dictionary_bytes) = encode_plan(&plan);
    let accounting = TableAccounting {
        total_bytes: bytes.len(),
        kernel_bytes: kernel_bytes.len(),
        plan_bytes: plan_length,
        dictionary_bytes,
        framing_bytes,
    };
    Ok((plan, kernel, accounting))
}
