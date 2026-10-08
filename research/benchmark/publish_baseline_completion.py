"""Recount baseline coverage and timeout dispositions from frozen small receipts."""
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELATIVE = 'research/benchmark/results/baseline-completion-20261008-v1'
LOCK_SHA256 = '3630991b75607ac02c8166c2090b6e4509ecd47cd169b86a11c2b32c02b6db98'
FOREST_ROUND_SHA256 = '15048401c2966534e5d051aa31da3614a8a30555d163a4f59f38c3eba5f5cea3'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def load_sources(repo=REPO):
    def verified(relative, expected):
        path = repo / relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts
                and path.resolve(strict=True).is_relative_to(repo.resolve())
                and not any(p.is_symlink() for p in (path, *path.parents)), 'invalid source path')
        require(path.stat().st_size <= 32_000_000, 'oversized metadata')
        raw = path.read_bytes()
        require(digest(raw) == expected, 'frozen metadata drift')
        return raw

    lock = json.loads(verified(RELATIVE+'/inputs.lock.json', LOCK_SHA256))
    require(lock['format'] == 'baseline-completion-input-lock-v1', 'input lock differs')
    docs = {}
    for name, ref in lock['sources'].items():
        raw = verified(ref['path'], ref['sha256'])
        require(len(raw) == ref['bytes'], 'metadata length differs')
        if ref['path'].endswith('.gz'):
            import io
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(8_000_001)
            require(len(raw) <= 8_000_000, 'oversized receipt archive')
        docs[name] = json.loads(raw)
    return lock, docs


def recount_forest(forest):
    require(forest['round']['sha256'] == FOREST_ROUND_SHA256, 'forest round differs')
    raw = forest['round_original'].encode()
    require(digest(raw) == FOREST_ROUND_SHA256 and len(raw) == forest['round']['bytes'],
            'original round bytes differ')
    round_lock = json.loads(raw)
    require(round_lock['fit_timeout_seconds'] == 600 and round_lock['official_tests_opened'] is False,
            'forest budget or seal differs')
    require(forest['runtime']['sha256'] == round_lock['runtime_sha256']
            and forest['source_files'] == round_lock['source_files'], 'captured source identities differ')
    jobs = {row['job_sha256']: row['file_sha256'] for row in round_lock['jobs']}
    seen, outcomes, timeouts = set(), collections.Counter(), {}
    for source, row in zip(forest['originals'], forest['fits'], strict=True):
        docs = {}
        for key in ('job', 'fit', 'close'):
            raw = source[key].encode()
            ref = source[key+'_ref']
            require(digest(raw) == ref['sha256'] and len(raw) == ref['bytes'],
                    'original receipt bytes differ')
            docs[key] = json.loads(raw)
        job, fit, close = (docs[k] for k in ('job', 'fit', 'close'))
        jid = canonical(job)
        require(jid not in seen and jobs.get(jid) == source['job_ref']['sha256'], 'duplicate or unbound job')
        seen.add(jid)
        require(fit['job_sha256'] == jid == row['job_sha256']
                and fit['round_sha256'] == close['round_sha256'] == FOREST_ROUND_SHA256
                and close['fit_receipt_sha256'] == source['fit_ref']['sha256'], 'closure differs')
        require(row['fit_receipt'] == source['fit_ref'] and row['job_receipt'] == source['job_ref']
                and row['closure_receipt'] == source['close_ref'], 'projection refs differ')
        require(type(job['fit_seed']) is int and job['fit_seed'] in (23, 37, 53, 71)
                and row['fit_seed'] == job['fit_seed'] and row['dataset'] == job['dataset']
                and row['status'] == fit['status'], 'projected identity differs')
        require(fit['status'] in ('ok', 'timeout') and fit['official_tests_opened'] is False
                and all(fit[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
                'unsupported outcome or claim')
        outcomes[fit['status']] += 1
        if fit['status'] == 'timeout':
            require(fit['worker_exit_code'] == close['worker_exit_code'] == 124, 'timeout closure differs')
            timeouts[jid] = row
    require(dict(outcomes) == forest['status_counts'] and len(seen) == forest['physical_closed'], 'counts differ')
    require({row['job_sha256'] for row in forest['timeout_dispositions']} == set(timeouts)
            and len(forest['timeout_dispositions']) == len(timeouts), 'timeout roster differs')
    for row in forest['timeout_dispositions']:
        source = timeouts[row['job_sha256']]
        require(row['dataset'] == source['dataset'] and row['fit_seed'] == source['fit_seed']
                and row['fit_receipt'] == source['fit_receipt']
                and row['job_receipt'] == source['job_receipt']
                and row['closure_receipt'] == source['closure_receipt']
                and row['fit_timeout_seconds'] == 600 and row['status'] == 'timeout'
                and row['accepted_artifact'] is False and row['retry_started'] is False
                and row['common_metrics'] is None and row['measured_elapsed_seconds'] is None,
                'timeout disposition differs')
    return outcomes


def build(lock, docs):
    forest = docs['forest_input']
    outcomes = recount_forest(forest)
    rate = docs['rate']
    start, end = rate['start'], rate['end']
    parse = datetime.datetime.fromisoformat
    hourly = (end['physical_closed']-start['physical_closed']) / ((parse(end['observed_utc'])-parse(start['observed_utc'])).total_seconds()/3600)
    require(hourly > 0 and hourly == rate['dispositions_per_hour'], 'rate projection differs')
    eta = parse(forest['observed_utc']) + datetime.timedelta(hours=(400-len(forest['fits']))/hourly)
    mt = datetime.timezone(datetime.timedelta(hours=-6))
    ts, native, initial_tabsyn, followon, td = (docs[k] for k in ('tabsyn_fits','tabsyn_native','tabsyn_old','followon','tabddpm'))
    for name, doc in docs.items():
        if name not in ('rate',):
            require(doc['official_tests_opened'] is False and doc['mfs_v2'] is None and doc['ptf_v1'] is None,
                    'publication gate differs')
    forests = [docs[k] for k in ('forest2','forest6','forest5')]
    require(all(c['status'] == 'fit_ok' for c in ts['cells']), 'TabSyn fit closure differs')
    require(followon['arf_common_n4n_five_fit_complete'] is True, 'ARF coverage differs')
    rows = [dict(method='ARF', logical_metrics_done=followon['arf_logical_cells'], logical_metrics_total=6000,
                 physical_metric_receipts=followon['arf_physical_metric_receipts'], aliases=followon['arf_alias_cells'],
                 completed_five_fit_lineages=followon['arf_lineages'], fit_timeouts=None, historical_failures_recounted=False,
                 eta='DONE: actual common-metric close Oct7 16:46 MDT; PR187 merged Oct7 20:36 MDT'),
            dict(method='TabSyn', default_fit11_ok=len(ts['cells']), default_fit11_total=100, default_fit11_non_ok=0,
                 logical_metrics_done=initial_tabsyn['sample_cells']+followon['tabsyn_new_cells'], logical_metrics_total=6000,
                 completed_n4n_lineages=initial_tabsyn['lineages'], completed_five_fit_lineages=0,
                 prior_unmeasured_allocations=ts['summary']['prior_unmeasured_allocations'],
                 prior_unmeasured_book_charge_seconds=ts['summary']['prior_unmeasured_book_charge_seconds'],
                 native_audit_status_counts=native['summary']['status_counts'], native_selection_complete=False,
                 prior_failed_native_attempts=native['summary']['prior_failed_native_attempts'],
                 eta='272.67 eligible GPU-slot-hour allowance: conditional Oct19 16:40 MDT for Oct8 08:00 start; no slot granted; native objective readiness required'),
            dict(method='TabDDPM', retained_models=td['retained_models'],
                 logical_metrics_done=td['measured_logical_slots'], logical_metrics_total=td['planned_logical_slots'],
                 physical_metric_receipts=td['physical_metric_receipts'], pending_metric_slots=td['pending_logical_slots'],
                 completed_five_fit_lineages=0, fit_timeouts=None, historical_failures_recounted=False,
                 eta='287.67 eligible GPU-slot-hour allowance: conditional Oct20 07:40 MDT for Oct8 08:00 start; no slot granted; original history reconciliation required'),
            dict(method='Forest-Flow', default_replication_dispositions=len(forest['fits']), default_replication_total=400,
                 default_replication_ok=outcomes['ok'], fit_timeouts=outcomes['timeout'],
                 logical_metrics_done=sum(p['common_sample_cells'] for p in forests), logical_metrics_total=6000,
                 default_only_metric_target=3000, published_models=sum(p['generator_fits'] for p in forests),
                 completed_five_fit_lineages=sum(p['lineages'] for p in forests),
                 fit_disposition_eta_mdt=eta.astimezone(mt).isoformat(), metric_allowance_hours=6,
                 conditional_metric_eta_mdt=(eta+datetime.timedelta(hours=6)).astimezone(mt).isoformat(),
                 eta='Rate-based disposition close; includes timeouts. Six-hour metric allowance is unmeasured; native replication deferred')]
    return dict(format='baseline-completion-checkpoint-v1', observed_utc=forest['observed_utc'], rows=rows,
                source_refs=lock['sources'], forest_round_sha256=FOREST_ROUND_SHA256,
                forest_fit_timeout_seconds=json.loads(forest['round_original'])['fit_timeout_seconds'],
                forest_recent_dispositions_per_hour=hourly, timeout_policy='retain_original_600s_no_selective_higher_budget_retry',
                new_fits=0, new_evaluations=0, official_tests_opened=False, mfs_v2=None, ptf_v1=None,
                release_safe_l3=None, superiority=None, production_certified=False)


def render(panel):
    rows = ['# Baseline completion checkpoint', '', 'Frozen '+panel['observed_utc']+'. Official tests sealed; gated scores null.', '',
            'Metric target: 100 lineages × default/native × five fit seeds × n/4n × three sample seeds = 6,000 logical cells.', '',
            '| Method | Published logical metric cells | Fit coverage / disposition | Failures / timeouts | ETA (MDT) |',
            '| --- | --- | --- | --- | --- |']
    for row in panel['rows']:
        method = row['method']
        scope = {'ARF':'100 lineages, all five seeds; 5,892 physical metric receipts +108 aliases',
                 'TabSyn':'100/100 scaled-default fit11 OK; 35 earlier unmeasured allocations; 8 complete n/4n lineages; native winner unavailable',
                 'TabDDPM':'21 retained models, fit11; other fit seeds pending; historical failures not recounted here',
                 'Forest-Flow':f"{row.get('default_replication_dispositions')}/400 additional default fit dispositions: {row.get('default_replication_ok')} OK, {row.get('fit_timeouts')} timeouts; published 13 five-fit lineages"}[method]
        eta = row.get('fit_disposition_eta_mdt', row['eta'])
        failures = {'ARF':'No missing current matrix; historical failed attempts not recounted',
                    'TabSyn':f"0 non-OK current default fits; {row.get('prior_failed_native_attempts')} earlier failed native-audit attempts; audit unavailable/input-unavailable remain distinct from fit failures",
                    'TabDDPM':f"Historical failures/timeouts not recounted; {row.get('pending_metric_slots')} pending slots are not failures",
                    'Forest-Flow':f"{row.get('fit_timeouts')} timeout; 0 other failures in this frozen additional-default round"}[method]
        rows.append(f"| {method} | {row['logical_metrics_done']}/{row['logical_metrics_total']} | {scope} | {failures} | {eta} |")
    rows += ['', 'Counts and input paths/digests: `panel.json` and `inputs.lock.json` in this directory.',
             'Timeouts remain missing and contribute no DOPE win. Forest default-only target is 3,000 metric cells.',
             'Native-objective unavailability is not a generator failure. Pending cells are not failures.',
             'No current bulk model hash, scientific auditor, or official test was reopened.', '']
    return '\n'.join(rows)


def render_budget_tex(panel):
    timeouts = next(r['fit_timeouts'] for r in panel['rows'] if r['method'] == 'Forest-Flow')
    return ('% Generated from the frozen baseline completion receipt; no hand-typed counts.\n'
            '\\newcommand{\\BaselineForestFitLimit}{'+str(panel['forest_fit_timeout_seconds'])+'}\n'
            '\\newcommand{\\BaselineForestTimeouts}{'+str(timeouts)+'}\n')


def main():
    lock, docs = load_sources()
    out = REPO/RELATIVE
    panel = build(lock, docs)
    for name, data in [('panel.json', panel), ('timeout-dispositions.json', docs['forest_input']['timeout_dispositions'])]:
        (out/name).write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n')
    (out/'completion.md').write_text(render(panel))
    (REPO/'docs/whitepaper/generated/baseline-budget.tex').write_text(render_budget_tex(panel))


if __name__ == '__main__':
    main()
