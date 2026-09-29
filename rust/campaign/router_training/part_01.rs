
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterBenchmarkReport {
    pub format: String,
    pub version: u8,
    pub rows: usize,
    pub features: usize,
    pub repeats: usize,
    pub sketch_p95_ms: f64,
    pub router_inference_p95_ms: Option<f64>,
    pub router_bundle_bytes: Option<u64>,
    pub deterministic_inference: Option<bool>,
}

pub fn benchmark_router(
    router_bundle: Option<&Path>,
    rows: usize,
    features: usize,
    repeats: usize,
) -> Result<RouterBenchmarkReport> {
    if rows == 0 || !(1..=2_000).contains(&features) || repeats < 2 {
        return Err(DopeError::Data(
            "router benchmark requires rows > 0, 1..=2000 features, and at least two repeats"
                .into(),
        ));
    }
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 40) as f32 / (1u32 << 24) as f32
    };
    let values = (0..rows * features).map(|_| next()).collect::<Vec<_>>();
    let target = (0..rows)
        .map(|row| {
            (0..features.min(8))
                .map(|feature| values[row * features + feature])
                .sum::<f32>()
                / features.min(8) as f32
        })
        .collect::<Vec<_>>();
    let table = Table::from_arrays(&values, &target, rows, features, Task::Regression)?;
    let mut sketches = Vec::with_capacity(repeats);
    let mut sketch_times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        sketches.push(DatasetSketch::from_train(&table, Task::Regression));
        sketch_times.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let sketch_p95_ms = float_percentile(&mut sketch_times, 0.95).unwrap_or(f64::INFINITY);
    let (router_inference_p95_ms, router_bundle_bytes, deterministic_inference) =
        if let Some(path) = router_bundle {
            let bundle: RouterBundle = read_json(path)?;
            let router = QuantizedRouter::new(bundle)?;
            let mut prediction_hashes = BTreeSet::new();
            let mut inference_times = Vec::with_capacity(repeats);
            for sketch in &sketches {
                let started = Instant::now();
                let prediction = router.predict(sketch)?;
                inference_times.push(started.elapsed().as_secs_f64() * 1_000.0);
                prediction_hashes.insert(hashes(&canonical_json(&prediction)?).sha256);
            }
            (
                float_percentile(&mut inference_times, 0.95),
                Some(
                    fs::metadata(path)
                        .map_err(|error| io_error(path, error))?
                        .len(),
                ),
                Some(prediction_hashes.len() == 1),
            )
        } else {
            (None, None, None)
        };
    Ok(RouterBenchmarkReport {
        format: "dope-router-benchmark".into(),
        version: 1,
        rows,
        features,
        repeats,
        sketch_p95_ms,
        router_inference_p95_ms,
        router_bundle_bytes,
        deterministic_inference,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct CachedSketch {
    format: String,
    version: u8,
    lineage_group_id: String,
    structural_profile: String,
    sketch: DatasetSketch,
    compute_ms: f64,
}

#[derive(Clone, Debug, Default)]
struct RouterCellAggregate {
    seen_replicates: u128,
    count: usize,
    retention_sum: f64,
    runtime_ms_sum: f64,
    peak_memory_bytes_sum: f64,
    artifact_bytes_sum: f64,
}

fn router_bounded_retention(cell: &JobEvidence) -> f64 {
    let improvement = cell.null_loss - cell.trtr_loss;
    let informative = improvement >= 0.01 * cell.null_loss.abs();
    if informative && improvement.abs() > f64::EPSILON {
        // Router utility is bounded: exceeding TRTR is a full success, while
        // negative retention has no extra routing value. KPI aggregation
        // retains the required unclamped statistic.
        ((cell.null_loss - cell.tstr_loss) / improvement).clamp(0.0, 1.0)
    } else {
        f64::from(cell.tstr_loss <= cell.trtr_loss + 0.01 * cell.null_loss)
    }
}

fn router_replicate_slot(cell: &JobEvidence, auditor_ids: &[String]) -> Result<usize> {
    let auditor = auditor_ids
        .iter()
        .position(|id| id == &cell.auditor_id)
        .ok_or_else(|| DopeError::Data(format!("unknown router auditor {}", cell.auditor_id)))?;
    let size = [1, 4]
        .iter()
        .position(|&value| value == cell.size_multiplier)
        .ok_or_else(|| DopeError::Data("router metric has a non-gold size multiplier".into()))?;
    let generation = [1829, 99238, 196647]
        .iter()
        .position(|&value| value == cell.generation_seed)
        .ok_or_else(|| DopeError::Data("router metric has an unfrozen generation seed".into()))?;
    let auditor_seed = [57721, 161803, 271828]
        .iter()
        .position(|&value| value == cell.auditor_seed)
        .ok_or_else(|| DopeError::Data("router metric has an unfrozen auditor seed".into()))?;
    Ok((((auditor * 2) + size) * 3 + generation) * 3 + auditor_seed)
}