//! The canonical contract `contracts/dope-kpi.json` (`dope.kpi-contract/v2`):
//! the statistic, bootstrap, promotion thresholds, byte grid and null.
//!
//! The contract is compiled into the binary; `--contract FILE` reads another
//! one instead. Either way its SHA-256 goes into every receipt. A contract
//! this binary does not implement (another schema, task list, cell reduction,
//! auditor roster shape or inconsistent bounds) is refused without quoting
//! it. A `--contract` file may change numbers (thresholds,
//! draws, seeds, quantiles, the byte grid, the auditor roster) but not the
//! texts that describe the method: those must be the compiled contract's,
//! so a receipt never names a method the binary did not run. Only a
//! contract whose SHA-256 is compiled in as frozen (an owner decision; none
//! yet) may declare `protocol_frozen: true` or a status other than
//! `candidate`.

use super::jsonl::{parse_json, read_file, sha256_hex};
use super::rundir::{contract, TASKS};
use super::trg::Protocol;
use dope_arena::ArenaResult;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CONTRACT_SCHEMA: &str = "dope.kpi-contract/v2";
const EMBEDDED: &[u8] = include_bytes!("../../../../contracts/dope-kpi.json");
/// SHA-256 of the contracts an owner decision froze (none yet).
const FROZEN: [&str; 0] = [];
const CANDIDATE: &str = "candidate";

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
    /// `embedded` (compiled from `contracts/dope-kpi.json`) or `file`.
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

    /// The texts that describe the method, by field name.
    fn method_texts(&self) -> [(&'static str, &str); 16] {
        let (s, b, p) = (&self.statistic, &self.bootstrap, &self.promotion);
        [
            ("contract_id", &self.contract_id),
            ("statistic.id", &s.id),
            ("statistic.name", &s.name),
            ("statistic.retention", &s.retention),
            ("statistic.entry_value", &s.entry_value),
            ("statistic.aggregation", &s.aggregation),
            ("statistic.informative", &s.informative),
            ("statistic.not_applicable", &s.not_applicable),
            ("bootstrap.kind", &b.kind),
            ("bootstrap.rank", &b.rank),
            ("bootstrap.p_value", &b.p_value),
            ("promotion.multiplicity", &p.multiplicity),
            ("promotion.coverage", &p.coverage),
            ("promotion.exact_copy", &p.exact_copy),
            ("bytes.guard", &self.bytes.guard),
            ("null.kind", &self.null.kind),
        ]
    }

    /// Refuse a method text other than the compiled one, and a frozen or
    /// non-candidate status the binary does not know as frozen.
    fn check_binding(&self, label: &str, sha256: &str, embedded: &Contract) -> ArenaResult<()> {
        let texts = self.method_texts().into_iter().zip(embedded.method_texts());
        for ((field, own), (_, compiled)) in texts {
            if own != compiled {
                return contract(format!(
                    "{label}: {field} describes a method this binary does not implement"
                ));
            }
        }
        let frozen = FROZEN.contains(&sha256);
        if (self.protocol_frozen || self.status != CANDIDATE) && !frozen {
            return contract(format!(
                "{label} declares a frozen or non-candidate status; only a contract frozen by an owner decision and compiled into dope_kpi may"
            ));
        }
        Ok(())
    }

    pub fn embedded() -> ArenaResult<(Contract, String)> {
        let label = "embedded contract";
        let (contract, sha256) = (Contract::parse(label, EMBEDDED)?, sha256_hex(EMBEDDED));
        contract.check_binding(label, &sha256, &contract)?;
        Ok((contract, sha256))
    }

    /// The contract at `path`, else the embedded one.
    pub fn load(path: Option<&Path>) -> ArenaResult<(Contract, ContractRef)> {
        let (contract, sha256, source) = match path {
            Some(path) => {
                let bytes = read_file(path)?;
                let label = path.display().to_string();
                let (contract, sha256) = (Contract::parse(&label, &bytes)?, sha256_hex(&bytes));
                contract.check_binding(&label, &sha256, &Contract::embedded()?.0)?;
                (contract, sha256, "file")
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

    /// The statistic of record: rows as `score` wrote them.
    pub fn protocol(&self) -> Protocol {
        Protocol {
            clip_low: self.statistic.clip_low,
            clip_high: self.statistic.clip_high,
            failed_value: self.statistic.failed_value,
            failed_at_floor: false,
            auditors: self.statistic.auditors.clone(),
            held_out: self.statistic.held_out_auditor.clone(),
            stratum: self.statistic.stratum.clone(),
            budget_cap: None,
        }
    }

    /// The statistic with every packet above the headline budget over budget
    /// (a diagnostic for arms scored with a larger budget).
    pub fn protocol_at_headline(&self) -> Protocol {
        Protocol {
            budget_cap: Some(self.bytes.headline),
            ..self.protocol()
        }
    }

    /// The statistic with a failed entry at `clip_low - r_b` instead of
    /// `failed_value` (a diagnostic; the protocol of record keeps the constant).
    pub fn protocol_failed_at_floor(&self) -> Protocol {
        Protocol {
            failed_at_floor: true,
            ..self.protocol()
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
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../contracts/dope-kpi.json");
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

    #[test]
    fn a_contract_file_may_change_numbers_but_not_the_method_or_its_status() {
        let dir = super::super::unit_test_dir("spec-binding");
        let text = String::from_utf8(EMBEDDED.to_vec()).unwrap();
        let load = |name: &str, body: &str| {
            let path = dir.join(name);
            std::fs::write(&path, body).unwrap();
            Contract::load(Some(&path)).map(|(c, r)| (c, r.source))
        };
        let numbers = text.replace("\"draws\": 10000", "\"draws\": 2000");
        let (c, source) = load("numbers.json", &numbers).unwrap();
        assert_eq!((c.bootstrap.draws, source.as_str()), (2000, "file"));
        let edits = [
            (
                "\"exact_copy\": \"mean over",
                "\"exact_copy\": \"SECRET maximum over",
            ),
            ("\"protocol_frozen\": false", "\"protocol_frozen\": true"),
            ("\"status\": \"candidate\"", "\"status\": \"frozen\""),
        ];
        for (index, (from, to)) in edits.into_iter().enumerate() {
            let body = text.replace(from, to);
            assert_ne!(body, text, "{from}");
            let err = load(&format!("edit-{index}.json"), &body)
                .unwrap_err()
                .to_string();
            assert!(!err.contains("SECRET"), "{err}");
        }
    }
}
