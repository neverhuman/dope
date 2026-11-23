//! The canonical contract `contracts/dope-kpi-v2.json` (`dope.kpi-contract/v2`):
//! the statistic, bootstrap, promotion thresholds, byte grid and null.
//!
//! The contract is compiled into the binary; `--contract FILE` reads another
//! one instead. Either way its SHA-256 goes into every receipt. A contract
//! this binary does not implement (another schema, task list, cell reduction,
//! auditor roster shape or inconsistent bounds) is refused without quoting it.

use super::jsonl::{parse_json, read_file, sha256_hex};
use super::rundir::{contract, TASKS};
use super::trg::Protocol;
use dope_arena::ArenaResult;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CONTRACT_SCHEMA: &str = "dope.kpi-contract/v2";
const EMBEDDED: &[u8] = include_bytes!("../../../../contracts/dope-kpi-v2.json");

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StatisticSpec {
    pub id: String,
    pub name: String,
    pub retention: String,
    pub entry_value: String,
    pub clip_low: f64,
    pub clip_high: f64,
    pub failed_value: f64,
    pub auditors: Vec<String>,
    pub cell_reduction: String,
    pub held_out_auditor: String,
    pub stratum: String,
    pub primary_multiplier: u32,
    pub matched_multiplier: u32,
    pub tasks: Vec<String>,
    pub aggregation: String,
    pub informative: String,
    pub not_applicable: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapSpec {
    pub kind: String,
    pub draws: usize,
    pub seed: u64,
    pub lower_quantile: f64,
    pub upper_quantile: f64,
    pub rank: String,
    pub p_value: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionSpec {
    pub multiplicity: String,
    pub coverage: String,
    pub exact_copy: String,
    pub ci_low_gt: f64,
    pub holm_p_lt: f64,
    pub min_delta: f64,
    pub default_noise_sd: f64,
    pub task_ci_low_gt: f64,
    pub mlp_delta_ge: f64,
    pub realism_coverage_delta_gt: f64,
    pub privacy_coverage_delta_gt: f64,
    pub exact_copy_excess_le: f64,
    pub min_training_seeds: usize,
    pub reject_ci_high_lt: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BytesSpec {
    pub grid: Vec<u64>,
    pub headline: u64,
    pub guard: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NullSpec {
    pub kind: String,
    pub min_replicates: usize,
    pub seed: u64,
    pub max_false_promote_rate: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Contract {
    pub schema: String,
    pub contract_id: String,
    pub status: String,
    pub protocol_frozen: bool,
    pub scope: String,
    pub statistic: StatisticSpec,
    pub bootstrap: BootstrapSpec,
    pub promotion: PromotionSpec,
    pub bytes: BytesSpec,
    pub null: NullSpec,
}

/// Which contract a receipt was computed under.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContractRef {
    pub id: String,
    pub sha256: String,
    /// `embedded` (compiled from `contracts/dope-kpi-v2.json`) or `file`.
    pub source: String,
    pub status: String,
    pub protocol_frozen: bool,
}

impl Contract {
    fn parse(label: &str, bytes: &[u8]) -> ArenaResult<Contract> {
        let c: Contract = parse_json(label, bytes)?;
        let s = &c.statistic;
        let distinct = |names: &[String]| {
            let mut sorted = names.to_vec();
            sorted.sort();
            sorted.dedup();
            sorted.len() == names.len()
        };
        let quantiles = 0.0 < c.bootstrap.lower_quantile
            && c.bootstrap.lower_quantile < c.bootstrap.upper_quantile
            && c.bootstrap.upper_quantile < 1.0;
        let valid = c.schema == CONTRACT_SCHEMA
            && s.tasks.iter().map(String::as_str).eq(TASKS)
            && s.cell_reduction == "minimum"
            && !s.auditors.is_empty()
            && distinct(&s.auditors)
            && !s.auditors.contains(&s.held_out_auditor)
            && s.clip_low.is_finite()
            && s.clip_high.is_finite()
            && s.clip_low < s.clip_high
            && s.failed_value.is_finite()
            && s.primary_multiplier >= 1
            && s.matched_multiplier >= 1
            && c.bootstrap.draws >= 200
            && quantiles
            && c.bytes.grid.windows(2).all(|w| w[0] < w[1])
            && c.bytes.grid.contains(&c.bytes.headline)
            && c.null.min_replicates >= 1
            && c.null.max_false_promote_rate > 0.0
            && c.null.max_false_promote_rate < 1.0
            && c.promotion.min_training_seeds >= 1;
        if !valid {
            return contract(format!(
                "{label} is not a {CONTRACT_SCHEMA} contract this binary implements"
            ));
        }
        Ok(c)
    }

    pub fn embedded() -> ArenaResult<(Contract, String)> {
        Ok((
            Contract::parse("embedded contract", EMBEDDED)?,
            sha256_hex(EMBEDDED),
        ))
    }

    /// The contract at `path`, else the embedded one.
    pub fn load(path: Option<&Path>) -> ArenaResult<(Contract, ContractRef)> {
        let (contract, sha256, source) = match path {
            Some(path) => {
                let bytes = read_file(path)?;
                let label = path.display().to_string();
                (Contract::parse(&label, &bytes)?, sha256_hex(&bytes), "file")
            }
            None => {
                let (contract, sha256) = Contract::embedded()?;
                (contract, sha256, "embedded")
            }
        };
        let reference = ContractRef {
            id: contract.contract_id.clone(),
            sha256,
            source: source.to_string(),
            status: contract.status.clone(),
            protocol_frozen: contract.protocol_frozen,
        };
        Ok((contract, reference))
    }

    pub fn protocol(&self) -> Protocol {
        Protocol {
            clip_low: self.statistic.clip_low,
            clip_high: self.statistic.clip_high,
            failed_value: self.statistic.failed_value,
            auditors: self.statistic.auditors.clone(),
            held_out: self.statistic.held_out_auditor.clone(),
            stratum: self.statistic.stratum.clone(),
        }
    }

    pub fn quantiles(&self) -> (f64, f64) {
        (self.bootstrap.lower_quantile, self.bootstrap.upper_quantile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_contract_is_the_repository_file() {
        let (c, sha) = Contract::embedded().unwrap();
        assert_eq!(c.schema, CONTRACT_SCHEMA);
        assert_eq!(c.bootstrap.draws, 10_000);
        assert_eq!(c.bytes.headline, 2048);
        assert_eq!(c.bytes.grid, [512, 1024, 2048, 4096]);
        assert_eq!((c.statistic.clip_low, c.statistic.clip_high), (-1.0, 1.25));
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../contracts/dope-kpi-v2.json"
        );
        assert_eq!(sha, sha256_hex(&std::fs::read(path).unwrap()));
        let protocol = c.protocol();
        assert_eq!(protocol.auditors, ["linear-seeded-v1", "gbdt"]);
        assert_eq!(protocol.held_out, "mlp-seeded-v1");
        let (_, reference) = Contract::load(Some(Path::new(path))).unwrap();
        assert_eq!((reference.sha256, reference.source.as_str()), (sha, "file"));
    }

    #[test]
    fn a_foreign_contract_is_refused_without_quoting_it() {
        let text = String::from_utf8(EMBEDDED.to_vec()).unwrap();
        let wrong = text.replace(
            "\"cell_reduction\": \"minimum\"",
            "\"cell_reduction\": \"SECRET\"",
        );
        assert_ne!(wrong, text);
        let err = Contract::parse("c.json", wrong.as_bytes())
            .unwrap_err()
            .to_string();
        assert!(err.contains("c.json") && !err.contains("SECRET"), "{err}");
        let extra = text.replacen('{', "{\"SECRET\": 1,", 1);
        let err = Contract::parse("c.json", extra.as_bytes())
            .unwrap_err()
            .to_string();
        assert!(!err.contains("SECRET"), "{err}");
    }
}
