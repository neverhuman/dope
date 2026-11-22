//! Bounded closed-loop exact-artifact compiler driver (`Y+`, `Xj+`, `Mj+`, `R+`, `N+`, `C+`, `P+`).
//!
//! Every candidate move is accepted ONLY after `serialize -> parse -> sample` round-trip
//! evaluation on sealed `SELECT`, ordered by transitive bucketed lexicographic selection.

use super::select::{select_best_decoded, CandidateScore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompilerMove {
    /// Increase conditional target capacity (distilled student trees/pairs).
    YPlus,
    /// Increase CATF covariate dependence capacity on worst column.
    XjPlus,
    /// Refine column marginal knots/atoms/lattice on worst MargAUC column.
    MjPlus,
    /// Expand mined structural rule set.
    RPlus,
    /// Refit heteroscedastic residual noise / temperature on CAL.
    NPlus,
    /// Audit-and-correct pass on weakest selector learner.
    CPlus,
    /// Owner-side damped-Newton moment projection compiled to stateless sampler.
    PPlus,
}

impl CompilerMove {
    pub fn code(&self) -> &'static str {
        match self {
            Self::YPlus => "Y+",
            Self::XjPlus => "Xj+",
            Self::MjPlus => "Mj+",
            Self::RPlus => "R+",
            Self::NPlus => "N+",
            Self::CPlus => "C+",
            Self::PPlus => "P+",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DecodedCandidateArtifact {
    pub mv: CompilerMove,
    pub bytes: Vec<u8>,
    pub packet_sha256: String,
    pub score: CandidateScore,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompileLedger {
    pub budget_bytes: usize,
    pub roles_sha256: String,
    pub lineage_sha256: String,
    pub evaluated: Vec<CandidateScore>,
    pub winner: CandidateScore,
}

/// Hash the deterministic refit lineage from `(roles_sha256, move_sequence, packet_sha256)`.
pub fn lineage_hash(roles_sha256: &str, moves: &[CompilerMove], packet_bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(roles_sha256.as_bytes());
    for mv in moves {
        h.update(mv.code().as_bytes());
        h.update(b"|");
    }
    h.update(Sha256::digest(packet_bytes));
    format!("{:x}", h.finalize())
}

/// Run the bounded closed-loop exact-artifact compiler over a finite schedule of moves.
///
/// `eval_move` MUST perform `build -> serialize -> parse -> sample -> SELECT audit` so every
/// score strictly reflects the exact decoded bytes that receivers execute!
pub fn run_exact_artifact_loop<F>(
    budget_bytes: usize,
    roles_sha256: &str,
    moves: &[CompilerMove],
    mut eval_move: F,
) -> Option<(DecodedCandidateArtifact, CompileLedger)>
where
    F: FnMut(CompilerMove, usize, &str) -> Option<DecodedCandidateArtifact>,
{
    let mut accepted_artifacts: Vec<DecodedCandidateArtifact> = Vec::new();
    let mut evaluated_scores: Vec<CandidateScore> = Vec::new();
    let mut executed_moves: Vec<CompilerMove> = Vec::new();

    for (step, &mv) in moves.iter().enumerate() {
        executed_moves.push(mv);
        let Some(mut cand) = eval_move(mv, step, roles_sha256) else {
            continue;
        };
        // Exact-artifact byte-cap guard: refuse any candidate exceeding `budget_bytes`.
        if cand.bytes.len() > budget_bytes || cand.score.bytes != cand.bytes.len() {
            continue;
        }
        cand.packet_sha256 = format!("{:x}", Sha256::digest(&cand.bytes));
        cand.score.refit_lineage_sha256 = lineage_hash(roles_sha256, &executed_moves, &cand.bytes);
        evaluated_scores.push(cand.score.clone());
        accepted_artifacts.push(cand);
    }

    let best_score = select_best_decoded(&evaluated_scores)?.clone();
    let winner_art = accepted_artifacts
        .into_iter()
        .find(|a| a.score == best_score)?;
    let ledger = CompileLedger {
        budget_bytes,
        roles_sha256: roles_sha256.to_string(),
        lineage_sha256: winner_art.score.refit_lineage_sha256.clone(),
        evaluated: evaluated_scores,
        winner: best_score,
    };
    Some((winner_art, ledger))
}
