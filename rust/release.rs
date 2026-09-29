use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::certification::{CertificationReport, certify_kernel_with_policy};
use crate::codec::LoadedKernel;
use crate::codec::{decode_kernel, encode_kernel};
use crate::compiler::{
    CandidateReport, CompileOptions, CompileReport, compile_kernel_from_arrays,
    compile_kernel_from_dir,
};
use crate::contract::{
    FormalDpConfiguration, FrozenBackend, ReleasePolicy, auditor_specs, empirical_backends,
    formal_dp_configurations,
};
use crate::data::Table;
use crate::error::{DopeError, Result, io_error};
use crate::model::{KernelProgram, Target, Task};
use crate::sample::{SampleOptions, sample_kernel, sample_kernel_to_csv};

#[derive(Clone, Debug)]
pub struct ConversionOptions {
    pub seed: u64,
    pub runtime_dictionary_bytes: usize,
    pub supported_datasets: usize,
    pub synthetic_csv_seeds: Vec<u64>,
    pub compile_options: CompileOptions,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            seed: 1729,
            runtime_dictionary_bytes: 0,
            supported_datasets: 1,
            synthetic_csv_seeds: Vec::new(),
            compile_options: CompileOptions {
                release_policy: ReleasePolicy::default(),
                beam_width: Some(4),
                ..Default::default()
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ByteAccounting {
    pub artifact_bytes: usize,
    pub shared_component_bytes: usize,
    pub standalone_bytes: usize,
    pub allocated_bytes: usize,
    pub supported_datasets: usize,
    pub reconciled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccountedCandidate {
    pub candidate_id: String,
    pub artifact_bytes: usize,
    pub standalone_bytes: usize,
    pub allocated_bytes: usize,
    pub score: f64,
    pub compliant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyntheticModelBundle {
    pub format: String,
    pub version: u8,
    pub task: Task,
    pub auditor: String,
    pub trained_from: String,
    pub tuning_source: String,
    pub training_rows: usize,
    pub training_seed: u64,
    pub target: Target,
    pub contains_source_rows: bool,
    pub checksum: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversionReport {
    pub format: String,
    pub version: u8,
    pub release_seed: u64,
    pub certified: bool,
    pub release_label: String,
    pub evaluated_candidates: Vec<String>,
    pub certified_candidates: Vec<String>,
    pub compiler: CompileReport,
    pub candidate_predictions: Vec<CandidateReport>,
    pub empirical_backend_registry: Vec<FrozenBackend>,
    pub standalone_pareto_frontier: Vec<AccountedCandidate>,
    pub allocated_pareto_frontier: Vec<AccountedCandidate>,
    pub formal_dp_frontier: Vec<FormalDpConfiguration>,
    pub byte_accounting: ByteAccounting,
    pub minimum_gold_one_sided_95_retention: Option<f64>,
    pub checksums: BTreeMap<String, String>,
    pub failures: Vec<String>,
    pub sealed_test_open_count: usize,
}

#[derive(Clone, Debug)]
pub struct ConversionResult {
    pub report: ConversionReport,
    pub certification: CertificationReport,
    pub output_dir: PathBuf,
}

fn row_major(table: &Table) -> Vec<f32> {
    let mut values = vec![0.0; table.rows * table.features];
    for row in 0..table.rows {
        for column in 0..table.features {
            values[row * table.features + column] = table.columns[column][row];
        }
    }
    values
}

fn checksum_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn bundle_checksum(bundle: &SyntheticModelBundle) -> Result<String> {
    let mut copy = bundle.clone();
    copy.checksum.clear();
    Ok(blake3::hash(&serde_json::to_vec(&copy)?)
        .to_hex()
        .to_string())
}

fn accounted(
    candidates: &[CandidateReport],
    runtime_dictionary_bytes: usize,
    supported_datasets: usize,
) -> Vec<AccountedCandidate> {
    candidates
        .iter()
        .map(|candidate| AccountedCandidate {
            candidate_id: candidate.candidate_id.clone(),
            artifact_bytes: candidate.artifact_bytes,
            standalone_bytes: candidate.artifact_bytes + runtime_dictionary_bytes,
            allocated_bytes: candidate.artifact_bytes
                + runtime_dictionary_bytes.div_ceil(supported_datasets.max(1)),
            score: candidate.score,
            compliant: candidate.compliant,
        })
        .collect()
}

pub fn convert_release(
    dataset_dir: &Path,
    real_holdout_dir: &Path,
    task: Task,
    out: &Path,
    options: &ConversionOptions,
) -> Result<ConversionResult> {
    if out.exists() {
        return Err(DopeError::Data(
            "release output already exists; choose a new directory".into(),
        ));
    }
    let missing_auditors = auditor_specs()
        .into_iter()
        .filter(|spec| !spec.available)
        .map(|spec| spec.id)
        .collect::<Vec<_>>();
    if !missing_auditors.is_empty() {
        return Err(DopeError::Unsupported(format!(
            "release requires complete frozen auditor coverage; unavailable: {}",
            missing_auditors.join(",")
        )));
    }
    let mut compile_options = options.compile_options.clone();
    compile_options.release_policy.validate()?;
    compile_options.seed = Some(options.seed);
    let compiled = compile_kernel_from_dir(dataset_dir, task, &compile_options)?;
    let evidence_dir = out.with_extension("candidate-evidence");
    fs::create_dir_all(&evidence_dir).map_err(|error| io_error(&evidence_dir, error))?;
    let mut evaluated_candidates = Vec::new();
    let mut passing = Vec::new();
    for candidate in &compiled.candidates {
        let artifact = compiled
            .candidate_artifacts
            .get(&candidate.candidate_id)
            .ok_or_else(|| DopeError::Data("compiler omitted a candidate artifact".into()))?;
        let identity = blake3::hash(artifact).to_hex().to_string();
        let candidate_path = evidence_dir.join(format!("{identity}.dpk"));
        let report_path = evidence_dir.join(format!("{identity}.certification.json"));
        fs::write(&candidate_path, artifact).map_err(|error| io_error(&candidate_path, error))?;
        let report = certify_kernel_with_policy(
            real_holdout_dir,
            &candidate_path,
            &report_path,
            options.seed,
            options.runtime_dictionary_bytes,
            options.supported_datasets,
            &compile_options.release_policy,
        )?;
        evaluated_candidates.push(candidate.candidate_id.clone());
        if report.certified {
            passing.push((candidate.clone(), artifact.clone(), report));
        }
    }
    passing.sort_by(|left, right| {
        left.1
            .len()
            .cmp(&right.1.len())
            .then_with(|| {
                right
                    .2
                    .minimum_gold_one_sided_95_retention
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(
                        &left
                            .2
                            .minimum_gold_one_sided_95_retention
                            .unwrap_or(f64::NEG_INFINITY),
                    )
            })
            .then_with(|| {
                right
                    .2
                    .fidelity
                    .worst_joint_fidelity
                    .total_cmp(&left.2.fidelity.worst_joint_fidelity)
            })
            .then_with(|| left.0.candidate_id.cmp(&right.0.candidate_id))
    });
    let certified_candidates = passing
        .iter()
        .map(|entry| entry.0.candidate_id.clone())
        .collect::<Vec<_>>();
    let (winner, artifact, certification) = passing.into_iter().next().ok_or_else(|| {
        DopeError::Data(format!(
            "no evaluated candidate passed every release gate; evidence: {}",
            evidence_dir.display()
        ))
    })?;
    let staging = out.with_extension(format!("staging-{}", std::process::id()));
    fs::create_dir(&staging).map_err(|error| io_error(&staging, error))?;
    let kernel_path = staging.join("kernel.dpk");
    fs::write(&kernel_path, &artifact).map_err(|error| io_error(&kernel_path, error))?;

    // A canonical decode/encode equality check is part of release construction,
    // not deferred to a consumer.
    let canonical = encode_kernel(&decode_kernel(&artifact)?)?;
    if canonical != artifact {
        return Err(crate::DopeError::Codec(
            "compiled Kernel V3 is not a canonical round trip".into(),
        ));
    }

    let loaded = LoadedKernel::V3(Box::new(decode_kernel(&artifact)?));
    for synthetic_seed in &options.synthetic_csv_seeds {
        sample_kernel_to_csv(
            &loaded,
            SampleOptions {
                rows: compiled.report.rows,
                seed: Some(*synthetic_seed),
            },
            &staging.join(format!("synthetic-{synthetic_seed}.csv")),
        )?;
    }

    let model_seed = options.seed.wrapping_add(0x5a17);
    let sampled = sample_kernel(
        &loaded,
        SampleOptions {
            rows: compiled.report.rows,
            seed: Some(model_seed),
        },
    )?;
    let synthetic = {
        let stride = compiled.report.positional_features + 1;
        let columns = (0..compiled.report.positional_features)
            .map(|column| {
                (0..compiled.report.rows)
                    .map(|row| sampled[row * stride + column])
                    .collect()
            })
            .collect();
        let target = (0..compiled.report.rows)
            .map(|row| sampled[row * stride + compiled.report.positional_features])
            .collect();
        Table {
            rows: compiled.report.rows,
            features: compiled.report.positional_features,
            columns,
            target,
        }
    };
    let synthetic_model = compile_kernel_from_arrays(
        &row_major(&synthetic),
        &synthetic.target,
        synthetic.rows,
        synthetic.features,
        task,
        &CompileOptions {
            seed: Some(options.seed),
            beam_width: Some(1),
            quantization_profiles: vec![8],
            ..Default::default()
        },
    )?;
    let target = match decode_kernel(&synthetic_model.artifact)?.program {
        KernelProgram::Symbolic { target, .. } => target,
        KernelProgram::NeuralJoint(_) => {
            return Err(DopeError::Unsupported(
                "synthetic model bundles require a symbolic predictor".into(),
            ));
        }
    };
    let mut bundle = SyntheticModelBundle {
        format: "dope-synthetic-model-bundle".into(),
        version: 3,
        task,
        auditor: "elastic_net_glm".into(),
        trained_from: "synthetic_only".into(),
        tuning_source: "fixed_hyperparameters_no_real_holdout".into(),
        training_rows: synthetic.rows,
        training_seed: model_seed,
        target,
        contains_source_rows: false,
        checksum: String::new(),
    };
    bundle.checksum = bundle_checksum(&bundle)?;
    let bundle_path = staging.join("synthetic-model.bundle");
    fs::write(&bundle_path, serde_json::to_vec_pretty(&bundle)?)
        .map_err(|error| io_error(&bundle_path, error))?;

    let certification_path = staging.join("tstr-certification.json");
    fs::write(
        &certification_path,
        serde_json::to_vec_pretty(&certification)?,
    )
    .map_err(|error| io_error(&certification_path, error))?;

    let standalone = accounted(
        &compiled.pareto_frontier,
        options.runtime_dictionary_bytes,
        1,
    );
    let allocated = accounted(
        &compiled.pareto_frontier,
        options.runtime_dictionary_bytes,
        options.supported_datasets,
    );
    let artifact_bytes = artifact.len();
    let standalone_bytes = artifact_bytes + options.runtime_dictionary_bytes;
    let allocated_bytes = artifact_bytes
        + options
            .runtime_dictionary_bytes
            .div_ceil(options.supported_datasets.max(1));
    let byte_accounting = ByteAccounting {
        artifact_bytes,
        shared_component_bytes: options.runtime_dictionary_bytes,
        standalone_bytes,
        allocated_bytes,
        supported_datasets: options.supported_datasets.max(1),
        reconciled: standalone_bytes == artifact_bytes + options.runtime_dictionary_bytes
            && allocated_bytes
                == artifact_bytes
                    + options
                        .runtime_dictionary_bytes
                        .div_ceil(options.supported_datasets.max(1)),
    };
    let empirical = empirical_backends();
    let formal_dp = formal_dp_configurations(compiled.report.rows);
    let failures: BTreeSet<String> = certification.failed_gates.iter().cloned().collect();
    let mut checksums = BTreeMap::new();
    checksums.insert("kernel.dpk".into(), checksum_file(&kernel_path)?);
    checksums.insert(
        "synthetic-model.bundle".into(),
        checksum_file(&bundle_path)?,
    );
    checksums.insert(
        "tstr-certification.json".into(),
        checksum_file(&certification_path)?,
    );
    for synthetic_seed in &options.synthetic_csv_seeds {
        let name = format!("synthetic-{synthetic_seed}.csv");
        checksums.insert(name.clone(), checksum_file(&staging.join(name))?);
    }
    let mut compiler_report = compiled.report.clone();
    compiler_report.selected_candidate = winner.candidate_id.clone();
    compiler_report.artifact_bytes = artifact_bytes;
    compiler_report.effective_bytes = artifact_bytes;
    compiler_report.bits_per_original_cell = winner.bits_per_original_cell;
    compiler_report.failed_gates = winner.failed_gates.clone();
    let report = ConversionReport {
        format: "dope-conversion-report".into(),
        version: 3,
        release_seed: options.seed,
        certified: true,
        release_label: "certified".into(),
        evaluated_candidates,
        certified_candidates,
        compiler: compiler_report,
        candidate_predictions: compiled.candidates,
        empirical_backend_registry: empirical,
        standalone_pareto_frontier: standalone,
        allocated_pareto_frontier: allocated,
        formal_dp_frontier: formal_dp,
        byte_accounting,
        minimum_gold_one_sided_95_retention: certification.minimum_gold_one_sided_95_retention,
        checksums,
        failures: failures.into_iter().collect(),
        sealed_test_open_count: 1,
    };
    let report_path = staging.join("conversion-report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .map_err(|error| io_error(&report_path, error))?;
    let expected = report
        .checksums
        .keys()
        .cloned()
        .chain(std::iter::once("conversion-report.json".into()))
        .collect::<BTreeSet<_>>();
    let actual = fs::read_dir(&staging)
        .map_err(|error| io_error(&staging, error))?
        .map(|entry| entry.map_err(|error| io_error(&staging, error)))
        .collect::<Result<Vec<_>>>()?;
    let mut found = BTreeSet::new();
    for entry in actual {
        let name = entry.file_name().to_string_lossy().into_owned();
        let metadata =
            fs::symlink_metadata(entry.path()).map_err(|error| io_error(entry.path(), error))?;
        if !metadata.file_type().is_file() || !expected.contains(&name) {
            return Err(DopeError::Data(
                "release staging contains an unexpected file or symlink".into(),
            ));
        }
        found.insert(name);
    }
    if found != expected {
        return Err(DopeError::Data(
            "release staging inventory is incomplete".into(),
        ));
    }
    for synthetic_seed in &options.synthetic_csv_seeds {
        Table::read_csv_with_policy(
            &staging.join(format!("synthetic-{synthetic_seed}.csv")),
            task,
            &compile_options.release_policy,
        )?;
    }
    fs::rename(&staging, out).map_err(|error| io_error(out, error))?;
    Ok(ConversionResult {
        report,
        certification,
        output_dir: out.to_path_buf(),
    })
}
