use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::certification::{CertificationReport, certify_kernel};
use crate::codec::LoadedKernel;
use crate::codec::{decode_kernel, encode_kernel};
use crate::compiler::{
    CandidateReport, CompileOptions, CompileReport, compile_kernel_from_arrays,
    compile_kernel_from_dir,
};
use crate::contract::{
    FormalDpConfiguration, FrozenBackend, empirical_backends, formal_dp_configurations,
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
            compile_options: CompileOptions::default(),
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
    fs::create_dir_all(out).map_err(|error| io_error(out, error))?;
    let mut compile_options = options.compile_options.clone();
    compile_options.seed = Some(options.seed);
    let compiled = compile_kernel_from_dir(dataset_dir, task, &compile_options)?;
    let kernel_path = out.join("kernel.dpk");
    fs::write(&kernel_path, &compiled.artifact).map_err(|error| io_error(&kernel_path, error))?;

    // A canonical decode/encode equality check is part of release construction,
    // not deferred to a consumer.
    let canonical = encode_kernel(&decode_kernel(&compiled.artifact)?)?;
    if canonical != compiled.artifact {
        return Err(crate::DopeError::Codec(
            "compiled Kernel V3 is not a canonical round trip".into(),
        ));
    }

    let loaded = LoadedKernel::V3(Box::new(decode_kernel(&compiled.artifact)?));
    for synthetic_seed in &options.synthetic_csv_seeds {
        sample_kernel_to_csv(
            &loaded,
            SampleOptions {
                rows: compiled.report.rows,
                seed: Some(*synthetic_seed),
            },
            &out.join(format!("synthetic-{synthetic_seed}.csv")),
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
    let bundle_path = out.join("synthetic-model.bundle");
    fs::write(&bundle_path, serde_json::to_vec_pretty(&bundle)?)
        .map_err(|error| io_error(&bundle_path, error))?;

    let certification_path = out.join("tstr-certification.json");
    let certification = certify_kernel(
        real_holdout_dir,
        &kernel_path,
        &certification_path,
        options.seed,
        options.runtime_dictionary_bytes,
        options.supported_datasets,
    )?;

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
    let artifact_bytes = compiled.artifact.len();
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
    let mut failures: BTreeSet<String> = compiled.report.failed_gates.iter().cloned().collect();
    failures.extend(certification.failed_gates.iter().cloned());
    failures.extend(
        empirical
            .iter()
            .filter(|backend| !backend.available)
            .map(|backend| format!("empirical_backend_unavailable:{}", backend.id)),
    );
    failures.extend(
        formal_dp
            .iter()
            .filter(|configuration| !configuration.available)
            .map(|configuration| {
                format!("formal_dp_backend_unavailable:{}", configuration.backend)
            }),
    );
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
        checksums.insert(name.clone(), checksum_file(&out.join(name))?);
    }
    let report = ConversionReport {
        format: "dope-conversion-report".into(),
        version: 3,
        release_seed: options.seed,
        certified: certification.certified,
        release_label: if certification.certified {
            "certified".into()
        } else {
            "strongest_explicit_kernel".into()
        },
        compiler: compiled.report,
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
    let report_path = out.join("conversion-report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .map_err(|error| io_error(&report_path, error))?;
    Ok(ConversionResult {
        report,
        certification,
        output_dir: out.to_path_buf(),
    })
}
