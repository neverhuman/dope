// Generated contract surface for the versioned TDL JSON schemas.
// Rust and the JSON schemas remain product truth; this package has no scorer.
export type EvidenceState = 'unmeasured' | 'incomplete' | 'complete' | 'failed' | 'unsupported' | 'missing';
export type PrivacyState = 'unassessed' | 'passed' | 'failed' | 'incomplete' | 'not_claimed' | 'unverified' | 'verified';
export type RightsState = 'unknown' | 'restricted' | 'cleared';

export interface TdlResultV2 {
  format: 'dope-tdl-result-v2'; contract_id: 'tdl-contract-v2.candidate'; version: 2;
  protocol_id: `protocol:${string}`; citable: false; certified: false;
  identities: Record<string, {kind: string; id: string; bindings: Record<string, string>}>;
  artifact_qualification_state: string; scheduled_cell_state: string;
  campaign_completeness_state: string; empirical_privacy_state: string;
  formal_dp_state: string; release_authorization_state: string; claim_provenance_state: string;
  measurements: {harm_ucb: {numerator:number;denominator:number}|null; driver_lcb:{numerator:number;denominator:number}|null; recipient_bytes:number|null};
}
export interface TdlLedgerEntry { id?: string; model?: string; state?: string; status?: string; evidence_url?: string; }
