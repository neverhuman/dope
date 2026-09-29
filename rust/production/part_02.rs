

impl StructuralProfile {
    pub fn from_shape(task: Task, rows: usize, features: usize) -> Result<Self> {
        if !(1..=2_000).contains(&features) {
            return Err(DopeError::Data(
                "production profiles support 1 to 2,000 features".into(),
            ));
        }
        let row_band = match rows {
            0..=31 => "<32",
            32..=255 => "32-255",
            256..=1023 => "256-1023",
            _ => ">=1024",
        };
        let feature_band = match features {
            1..=16 => "1-16",
            17..=64 => "17-64",
            65..=256 => "65-256",
            _ => "257-2000",
        };
        Ok(Self {
            task: task.as_str().into(),
            row_band: row_band.into(),
            feature_band: feature_band.into(),
        })
    }

    pub fn id(&self) -> String {
        format!("{}/{}/{}", self.task, self.row_band, self.feature_band)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiCell {
    pub task: Task,
    pub train_rows: usize,
    pub features: usize,
    pub auditor: String,
    pub size_multiplier: usize,
    pub lineage_group_id: String,
    pub generation_seed: u64,
    pub auditor_seed: u64,
    pub null_loss: f64,
    pub trtr_loss: f64,
    pub tstr_loss: f64,
    pub calibration_degradation: Option<f64>,
    pub rare_class_or_tail_retention: Option<f64>,
    pub supported_subgroup_retention: Option<f64>,
    pub nominal_95_coverage: Option<f64>,
}

impl KpiCell {
    pub fn validate(&self, contract: &KpiContract) -> Result<()> {
        if self.train_rows == 0
            || !(1..=2_000).contains(&self.features)
            || !contract
                .scaling_size_multipliers
                .contains(&self.size_multiplier)
            || !contract.auditor_seeds.contains(&self.auditor_seed)
            || ![self.null_loss, self.trtr_loss, self.tstr_loss]
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0)
            || self.lineage_group_id.is_empty()
            || !contract.auditor_seeds.contains(&self.auditor_seed)
        {
            return Err(DopeError::Data("invalid KPI cell".into()));
        }
        if !auditor_specs()
            .iter()
            .any(|auditor| auditor.id == self.auditor)
        {
            return Err(DopeError::Data("KPI cell names an unfrozen auditor".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiAggregate {
    pub task: String,
    pub structural_profile: String,
    pub auditor: String,
    pub size_multiplier: usize,
    pub lineage_groups: usize,
    pub informative_groups: usize,
    pub low_signal_groups: usize,
    pub mean_unclamped_retention: Option<f64>,
    pub one_sided_95_lower_bound: Option<f64>,
    pub low_signal_compliance: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiReport {
    pub format: String,
    pub version: u8,
    pub primary_kpi: String,
    pub ptf_v1: Option<f64>,
    pub cells: Vec<KpiAggregate>,
    pub independently_gated_profiles: Vec<String>,
    pub folded_profiles: BTreeMap<String, String>,
    pub low_signal_compliance_overall: Option<f64>,
    pub low_signal_compliance_by_profile: BTreeMap<String, f64>,
    pub eligible_population: usize,
    pub evaluated_numerator: usize,
    pub missing_or_timeout_denominator: usize,
    pub lineage_weighting: String,
    pub coverage_tier: String,
    pub confidence_method: String,
    pub kpi_contract: ContentHashes,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct AuditorAvailabilitySummary {
    pub required: usize,
    pub available: usize,
    pub unavailable: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiSummary {
    pub format: String,
    pub version: u8,
    pub ptf_v1: Option<f64>,
    pub low_signal_compliance: Option<f64>,
    pub coverage: Option<f64>,
    pub missing_cells: usize,
    pub candidate_regret: Option<f64>,
    pub auditor_availability: AuditorAvailabilitySummary,
    pub production_score_available: bool,
}

impl KpiSummary {
    pub fn from_evidence(
        report: Option<&KpiReport>,
        coverage_report: Option<&CoverageReport>,
        candidate_regret: Option<f64>,
    ) -> Self {
        let auditors = auditor_specs();
        let unavailable = auditors
            .iter()
            .filter(|auditor| !auditor.available)
            .map(|auditor| auditor.id.clone())
            .collect::<Vec<_>>();
        let (coverage, missing_cells) = if let Some(coverage_report) = coverage_report {
            let eligible = coverage_report
                .entries
                .iter()
                .map(|entry| entry.eligible)
                .sum::<usize>();
            let evaluated = coverage_report
                .entries
                .iter()
                .map(|entry| entry.evaluated)
                .sum::<usize>();
            let missing = coverage_report
                .entries
                .iter()
                .map(|entry| entry.missing + entry.timed_out)
                .sum();
            (
                (eligible > 0).then_some(evaluated as f64 / eligible as f64),
                missing,
            )
        } else {
            (
                report.and_then(|report| {
                    (report.eligible_population > 0).then_some(
                        report.evaluated_numerator as f64 / report.eligible_population as f64,
                    )
                }),
                report.map_or(0, |report| report.missing_or_timeout_denominator),
            )
        };
        let ptf_v1 = report.and_then(|report| report.ptf_v1);
        let required_complete = coverage_report
            .is_some_and(|coverage| coverage.validate().is_ok() && coverage.required_complete);
        let production_score_available = ptf_v1.is_some()
            && required_complete
            && unavailable.is_empty()
            && empirical_backends().iter().all(|backend| backend.available);
        Self {
            format: "dope-kpi-summary".into(),
            version: 1,
            ptf_v1,
            low_signal_compliance: report.and_then(|report| report.low_signal_compliance_overall),
            coverage,
            missing_cells,
            candidate_regret,
            auditor_availability: AuditorAvailabilitySummary {
                required: auditors.len(),
                available: auditors.len() - unavailable.len(),
                unavailable,
            },
            production_score_available,
        }
    }
}

type AggregateKey = (String, String, String, usize);

#[derive(Clone, Copy)]
struct LossTriple {
    null: f64,
    trtr: f64,
    tstr: f64,
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<_> = values.collect();
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn one_sided_lower(values: &[f64], level: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return Some(average);
    }
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let distribution = StudentsT::new(0.0, 1.0, (values.len() - 1) as f64).ok()?;
    Some(average - distribution.inverse_cdf(level) * (variance / values.len() as f64).sqrt())
}

pub fn aggregate_kpis(
    cells: &[KpiCell],
    eligible_population: usize,
    missing_or_timeout_denominator: usize,
    coverage_tier: &str,
) -> Result<KpiReport> {
    let contract = KpiContract::embedded()?;
    contract.validate()?;
    for cell in cells {
        cell.validate(&contract)?;
    }

    let mut profile_lineages = BTreeMap::<String, BTreeMap<String, Vec<LossTriple>>>::new();
    for cell in cells {
        let profile =
            StructuralProfile::from_shape(cell.task, cell.train_rows, cell.features)?.id();
        profile_lineages
            .entry(profile)
            .or_default()
            .entry(cell.lineage_group_id.clone())
            .or_default()
            .push(LossTriple {
                null: cell.null_loss,
                trtr: cell.trtr_loss,
                tstr: cell.tstr_loss,
            });
    }
    let mut independently_gated = Vec::new();
    let mut folded_profiles = BTreeMap::new();
    for (profile, lineages) in &profile_lineages {
        let informative = lineages
            .values()
            .filter(|values| {
                let null = mean(values.iter().map(|value| value.null));
                let trtr = mean(values.iter().map(|value| value.trtr));
                null - trtr >= contract.low_signal.improvement_threshold_fraction * null.abs()
            })
            .count();
        if lineages.len() >= contract.profile_gating.minimum_groups
            && informative >= contract.profile_gating.minimum_informative_groups
        {
            independently_gated.push(profile.clone());
            folded_profiles.insert(profile.clone(), profile.clone());
        } else {
            let task = profile.split('/').next().unwrap_or("unknown");
            folded_profiles.insert(profile.clone(), format!("{task}/other"));
        }
    }
    independently_gated.sort();

    let mut grouped = BTreeMap::<AggregateKey, BTreeMap<String, Vec<LossTriple>>>::new();
    for cell in cells {
        let raw_profile =
            StructuralProfile::from_shape(cell.task, cell.train_rows, cell.features)?.id();
        let profile = folded_profiles[&raw_profile].clone();
        grouped
            .entry((
                cell.task.as_str().into(),
                profile,
                cell.auditor.clone(),
                cell.size_multiplier,
            ))
            .or_default()
            .entry(cell.lineage_group_id.clone())
            .or_default()
            .push(LossTriple {
                null: cell.null_loss,
                trtr: cell.trtr_loss,
                tstr: cell.tstr_loss,
            });
    }

    let mut aggregates = Vec::new();
    let mut all_low_signal = Vec::<(String, bool)>::new();
    for ((task, profile, auditor, size_multiplier), lineages) in grouped {
        let mut retentions = Vec::new();
        let mut low_signal = Vec::new();
        for values in lineages.values() {
            let null = mean(values.iter().map(|value| value.null));
            let trtr = mean(values.iter().map(|value| value.trtr));
            let tstr = mean(values.iter().map(|value| value.tstr));
            if null - trtr >= contract.low_signal.improvement_threshold_fraction * null.abs() {
                retentions.push((null - tstr) / (null - trtr));
            } else {
                let passed =
                    tstr <= trtr + contract.low_signal.noninferiority_fraction_of_null * null;
                low_signal.push(passed);
                all_low_signal.push((profile.clone(), passed));
            }
        }
        aggregates.push(KpiAggregate {
            task,
            structural_profile: profile,
            auditor,
            size_multiplier,
            lineage_groups: lineages.len(),
            informative_groups: retentions.len(),
            low_signal_groups: low_signal.len(),
            mean_unclamped_retention: (!retentions.is_empty())
                .then(|| retentions.iter().sum::<f64>() / retentions.len() as f64),
            one_sided_95_lower_bound: one_sided_lower(
                &retentions,
                contract.confidence.one_sided_level,
            ),
            low_signal_compliance: (!low_signal.is_empty()).then(|| {
                low_signal.iter().filter(|passed| **passed).count() as f64 / low_signal.len() as f64
            }),
        });
    }
    let required: BTreeSet<_> = contract.required_size_multipliers.iter().copied().collect();
    let ptf_v1 = aggregates
        .iter()
        .filter(|cell| required.contains(&cell.size_multiplier))
        .filter_map(|cell| cell.one_sided_95_lower_bound)
        .reduce(f64::min);
    let low_signal_compliance_overall = (!all_low_signal.is_empty()).then(|| {
        all_low_signal.iter().filter(|(_, passed)| *passed).count() as f64
            / all_low_signal.len() as f64
    });
    let mut by_profile_raw = BTreeMap::<String, (usize, usize)>::new();
    for (profile, passed) in all_low_signal {
        let entry = by_profile_raw.entry(profile).or_default();
        entry.0 += passed as usize;
        entry.1 += 1;
    }
    let low_signal_compliance_by_profile = by_profile_raw
        .into_iter()
        .map(|(profile, (passed, total))| (profile, passed as f64 / total as f64))
        .collect();
    let evaluated_numerator = cells
        .iter()
        .map(|cell| &cell.lineage_group_id)
        .collect::<BTreeSet<_>>()
        .len();
    Ok(KpiReport {
        format: "dope-kpi-aggregate".into(),
        version: 1,
        primary_kpi: "PTF-v1".into(),
        ptf_v1,
        cells: aggregates,
        independently_gated_profiles: independently_gated,
        folded_profiles,
        low_signal_compliance_overall,
        low_signal_compliance_by_profile,
        eligible_population,
        evaluated_numerator,
        missing_or_timeout_denominator,
        lineage_weighting: "equal_lineage_group".into(),
        coverage_tier: coverage_tier.into(),
        confidence_method: "paired_lineage_clustered_one_sided_student_t_95".into(),
        kpi_contract: ContentHashes {
            sha256: KPI_CONTRACT_SHA256.into(),
            blake3: KPI_CONTRACT_BLAKE3.into(),
        },
    })
}

include!("coverage_and_gates.rs");
