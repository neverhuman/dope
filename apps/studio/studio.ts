const input = document.querySelector('#receipt-file');
const status = document.querySelector('#status');
const retry = document.querySelector('#retry');
const review = document.querySelector('#review');
const ledger = document.querySelector('#ledger');
let entries = [];
let shown = 0;
const text = (id, value) => { document.querySelector(`#${id}`).textContent = value; };
const safe = (v) => typeof v === 'string' ? v : v == null ? 'Unknown' : String(v);
const hash = (obj) => obj?.report_sha256 || obj?.receipt_sha256 || obj?.sha256 || 'Not provided';
function setState(node, value, qualified = false) {
  node.textContent = value;
  node.className = `state ${qualified ? 'good' : 'bad'}`;
}
function renderLedger() {
  const fragment = document.createDocumentFragment();
  for (const entry of entries.slice(shown, shown + 50)) {
    const li = document.createElement('li');
    li.textContent = `${safe(entry.model || entry.id || 'Evidence item')}: ${safe(entry.state || entry.status || 'unknown')}`;
    if (entry.evidence_url && /^https?:\/\//i.test(entry.evidence_url)) {
      const a = document.createElement('a'); a.href = entry.evidence_url; a.rel = 'noreferrer'; a.textContent = ' Evidence'; li.append(a);
    }
    fragment.append(li);
  }
  ledger.append(fragment); shown += 50;
  document.querySelector('#more').hidden = shown >= entries.length;
  text('ledger-count', `${entries.length} entries total; showing ${Math.min(shown, entries.length)}.`);
}
function render(data, file, fileHash) {
  review.hidden = false;
  const report = data.format === 'dope-tdl-report';
  const unknown = data.format !== 'dope-tdl-result-v2' && !report;
  const complete = (report ? data.evidence_status === 'sealed_test' : data.campaign_completeness_state === 'complete');
  const privacyPassed = report ? data.release_gates?.empirical_privacy === 'pass' : data.empirical_privacy_state === 'passed';
  const identity = report ? data.dataset?.data_sha256 : data.identities?.task?.bindings?.source_sha256;
  const pairs = [['Evidence type', report ? 'TDL report v1' : unknown ? safe(data.format) : 'TDL result v2'], ['Source commitment', safe(identity)], ['File', file.name], ['File size', `${file.size} bytes`], ['Receipt/report hash', hash(data)], ['Loaded file SHA-256', fileHash]];
  const summary = document.querySelector('#summary'); summary.replaceChildren();
  for (const [k,v] of pairs) { const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;summary.append(dt,dd); }
  text('schema-state', `Schema / format: ${safe(data.format)}${unknown ? ' (unsupported)' : ' (recognized)'}`);
  text('source-state', `Source binding: ${safe(data.identities?.task?.bindings?.source_sha256 || data.dataset?.data_sha256)}`);
  const rightsState = data.rights_state || data.release_gates?.rights_state || data.release_gates?.rights;
  text('rights-state', `Rights state: ${safe(rightsState)} · evidence commitment: ${safe(data.identities?.release_profile?.bindings?.rights_sha256 || data.release_gates?.rights_sha256)}`);
  text('progress', `Campaign: ${safe(data.campaign_completeness_state || data.evidence_status)} · scheduled job: ${safe(data.scheduled_cell_state)} · qualification: ${safe(data.artifact_qualification_state)}`);
  text('bytes', `Recipient bytes: ${safe(data.measurements?.recipient_bytes ?? data.byte_accounting?.recipient_bytes ?? data.byte_accounting?.total_bytes)}`);
  text('privacy', `Empirical privacy: ${safe(data.empirical_privacy_state || data.release_gates?.empirical_privacy)} · formal DP: ${safe(data.formal_dp_state || 'Unknown')}`);
  entries = Array.isArray(data.auditor_reports) ? data.auditor_reports : Array.isArray(data.ledger) ? data.ledger : [];
  ledger.replaceChildren(); shown = 0; renderLedger();
  const models = entries.map(e => `${safe(e.model || e.model_id || e.family || 'model')}: ${safe(e.state || e.status || 'unknown')}`);
  const denominator = data.mandatory_model_count ?? data.denominator ?? data.expected_models?.length ?? entries.length;
  const failures = data.mandatory_model_failures ?? entries.filter(e => !['passed','pass','complete','qualified'].includes(String(e.state || e.status).toLowerCase())).length;
  const rightsPassed = ['cleared','pass','passed'].includes(String(rightsState).toLowerCase());
  const artifactPassed = report ? data.citable === true : data.artifact_qualification_state === 'qualified';
  const qualified = !unknown && complete && privacyPassed && artifactPassed && rightsPassed && entries.length > 0 && entries.length >= Number(denominator) && Number(failures) === 0;
  setState(document.querySelector('#qualification'), unknown ? 'Unsupported evidence format — qualification unavailable' : qualified ? 'Qualified by the supplied evidence' : 'Not qualified · incomplete or blocked evidence', qualified);
  text('models', `Mandatory models: ${entries.length} reported of ${safe(denominator)} expected; ${safe(failures)} failures/incomplete. ${models.length ? models.join(' · ') : 'No model ledger supplied; state unknown.'}`);
  text('hash', `Receipt/report hash: ${hash(data)} · loaded file SHA-256: ${fileHash}`);
  const links = Array.isArray(data.evidence_links) ? data.evidence_links.filter(x => typeof x === 'string' && /^https?:\/\//i.test(x)) : [];
  document.querySelector('#links').replaceChildren(document.createTextNode('Evidence links: '));
  for (const url of links.slice(0, 50)) { const a=document.createElement('a');a.href=url;a.rel='noreferrer';a.textContent=url;a.style.marginRight='.8rem';document.querySelector('#links').append(a); }
}
async function openFile(file) {
  if (!file) return;
  try {
    const raw = await file.text();
    if (raw.length > 10_000_000) throw new Error('File exceeds the 10 MB viewer limit.');
    const data = JSON.parse(raw);
    if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Expected a JSON object.');
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
    const fileHash = [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('');
    render(data, file, fileHash); status.textContent = `Loaded ${file.name} locally.`; retry.hidden = true;
  } catch (error) {
    review.hidden = true; status.textContent = `Could not open evidence: ${error.message}`; retry.hidden = false;
  }
}
input.addEventListener('change', () => openFile(input.files[0]));
retry.addEventListener('click', () => openFile(input.files[0]));
document.querySelector('#more').addEventListener('click', renderLedger);
