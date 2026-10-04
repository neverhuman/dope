"""Replay frozen native density choices and trial-cost metadata; no model loads."""
from pathlib import Path
import hashlib
import itertools
import json
import math

from research.benchmark import density_publication_inputs as owned

BASE = owned.BASE
ROOT = BASE / 'density-s3-population-native-custody-v1'
METHOD_SHA = '406ac543c2b6f139c6fc665a556c754af48392962bcbdf4d39d256243bb0102e'
METHOD_PATH = BASE / 'density-native-replay-preparation-v1/frozen-methods.lock.json'
require = owned.require
ROUNDS = (('compact-native-all-v1', 'f347643867949c186976bcf1de38b71123ef5922b2bd73dd82fd2fa0a1baf6a7', 'method_lock_sha256'),
              ('copula-native-all-v1', '9fa3528eb69605c5f03beea0750d911fbb3c811dcddb9b4f9ce3dc55738f7d64', 'methods_sha256'))


def identity(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def number(v):
    if type(v) not in (int, float):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def replay_native_metadata():
    """Verify supplied native metadata from fixed external anchors.

    The legacy method snapshot matches both original round hashes. No common
    utility participates in selection. Full native reference/runtime/worker/
    artifact custody remains a separate final publication prerequisite.
    """
    anchor = owned.bound_json(ROOT / 'receipt-lock-v1.json', owned.NATIVE)
    require(anchor['reconciliation_sha256'] == owned.NATIVE_REPORT, 'native anchor differs')
    refs = anchor['refs']
    consumed = {}

    def read(path, expected=None):
        p = str(path)
        require(p in refs and (expected is None or refs[p] == expected), 'unbound native metadata')
        value = owned.bound_json(p, refs[p])
        consumed[p] = refs[p]
        return value

    report = owned.bound_json(ROOT / 'reconciliation-v1.json', owned.NATIVE_REPORT)
    methods = owned.bound_json(METHOD_PATH, METHOD_SHA)['methods']
    require(report['complete_native_matrix'] is True
            and all(type(report[k]) is int and report[k] == n
                    for k, n in (('datasets', 100), ('native_cells', 300), ('trial_count', 1100)))
            and report['native_objective'] == 'mean_log_density' and report['direction'] == 'maximize'
            and report['official_tests_opened'] is False and report['shared_kpi_used_for_selection'] is False
            and report['counts_as_dope_win'] is False and report['native_kpis_never_ranked_across_methods'] is True
            and all(report.get(k) is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority'))
            and all(type(report[k]) is int and report[k] == 0 for k in
                    ('new_generator_fits_started', 'new_tuning_trials', 'new_samples_started'))
            and report['historical_full_runtime_closure_upgraded'] is False
            and report['reported_experiment_reproduction_claim'] is False,
            'native report scope differs')
    source_files = report['source_files']
    require(len(source_files) == 4 and {Path(p).name for p in source_files} ==
            {'adapters.py', 'native_objective.py', 'tune_density.py', 'tune_copula.py'}
            and all(refs.get(p) == h for p, h in source_files.items()), 'native source declarations differ')
    owned.verify_refs(source_files)
    owned.exact_tree(ROOT / 'source', source_files)
    cells = {(c['dataset'], c['method'], c['configuration']): c for c in report['default_native_cells']}
    trials = {(t['dataset'], t['method'], t['trial_index']): t for t in report['trials']}
    require(len(cells) == len(report['default_native_cells']) == 600
            and len(trials) == len(report['trials']) == 1100, 'native inventory differs')
    require(all(type(c['dataset']) is str and type(c['method']) is str
                and c['configuration'] in ('default', 'native_selected')
                and type(c['fit_seed']) is int and c['fit_seed'] == 11
                for c in cells.values())
            and all(type(t['trial_index']) is int for t in trials.values()), 'native numeric identities differ')
    seen, costs, used_trials = set(), [], set()
    legacy_selections_without_validation_hash = 0
    legacy_trials_without_source_identity = 0
    for name, lock_sha, method_key in ROUNDS:
        lock = read(BASE / name / 'round.lock.json', lock_sha)
        require(lock[method_key] == METHOD_SHA and lock['validation_only'] is True, 'original method lock differs')
        for job in lock['jobs']:
            dataset, method = job['dataset'], job.get('method', 'GaussianCopula')
            key = dataset, method
            require(type(dataset) is str and method in
                    ('GaussianCopula', 'independent_marginals', 'Chow-Liu')
                    and key not in seen, 'duplicate or unknown native lineage')
            seen.add(key)
            entry = methods[method]
            require(entry['native_objective']['implementation_sha256'] == next(
                        h for p, h in source_files.items() if Path(p).name == 'native_objective.py')
                    and entry['adapter_sha256'] == lock['adapter_sha256'] == next(
                        h for p, h in source_files.items() if Path(p).name == 'adapters.py'),
                    'native declared source binding differs')
            configs = []
            keys = sorted(entry['tuning_search_space'])
            for values in itertools.product(*(entry['tuning_search_space'][k] for k in keys)):
                configs.append(entry['default_config'] | dict(zip(keys, values)))
            require(identity(configs) == identity(job['configurations']) and 1 <= len(configs) <= 8
                    and type(job['fit_seed']) is int and job['fit_seed'] == 11
                    and type(job['budget_seconds']) is int and job['budget_seconds'] == 43200, 'native grid or budget differs')
            default, selected = cells[key + ('default',)], cells[key + ('native_selected',)]
            require(default['selection_receipt_path'] == selected['selection_receipt_path']
                    and default['selection_receipt_sha256'] == selected['selection_receipt_sha256'], 'native selection binding differs')
            selection = read(default['selection_receipt_path'], default['selection_receipt_sha256'])
            require(selection['format'] == 'dope-benchmark-validation-selection'
                    and type(selection['version']) is int and selection['version'] == 1
                    and selection['dataset'] == dataset and selection['method'] == method
                    and selection['partition'] == 'validation' and selection['test_opened'] is False
                    and selection['round_sha256'] == lock_sha
                    and type(selection['selected_trial_index']) is int
                    and identity(selection['objective']) == identity(entry['native_objective'])
                    and selection['objective']['name'] == 'mean_log_density'
                    and selection['objective']['direction'] == 'maximize'
                    and len(selection['trials']) == len(configs), 'native objective or receipt differs')
            # These externally anchored historical selections omit this field.
            # Every raw trial and metric below must still bind the validation hash.
            if 'validation_sha256' in selection:
                require(selection['validation_sha256'] == job['validation_sha256'],
                        'native selection partition differs')
            else:
                legacy_selections_without_validation_hash += 1
            rows, cell_costs = [], []
            for index, trial in enumerate(selection['trials']):
                trial_key = dataset, method, index
                summary = trials[trial_key]
                raw = read(summary['attempt_receipt_path'], summary['attempt_receipt_sha256'])
                require(raw['status'] == trial['status'] == summary['status'] == 'ok', 'native raw outcome differs')
                require(identity(raw['config']) == identity(trial['config']) == identity(configs[index])
                        and number(raw['native_kpi']) and len({identity(v) for v in
                            (raw['native_kpi'], trial['native_kpi'], summary['native_validation_kpi'])}) == 1
                        and type(raw['artifact_bytes']) is int and raw['artifact_bytes'] >= 0
                        and len({identity(v) for v in (raw['artifact_bytes'], trial['artifact_bytes'],
                                                      summary['charged_artifact_bytes'])}) == 1, 'native trial differs')
                proof = raw['identity']
                require(proof['dataset'] == dataset and proof['method'] == method
                        and type(proof['fit_seed']) is int and proof['fit_seed'] == 11
                        and proof['round_sha256'] == lock_sha
                        and identity(proof['config']) == identity(raw['config'])
                        and proof['train_sha256'] == job['train_sha256']
                        and proof['validation_sha256'] == job['validation_sha256'], 'native trial partition differs')
                for field, expected in (
                    ('native_objective_sha256', entry['native_objective']['implementation_sha256']),
                    ('adapter_sha256', lock['adapter_sha256'])):
                    if field in proof:
                        require(proof[field] == expected, 'native trial source identity differs')
                    else:
                        require(method == 'GaussianCopula', 'native trial source identity missing')
                if any(field not in proof for field in ('native_objective_sha256', 'adapter_sha256')):
                    legacy_trials_without_source_identity += 1
                inventory = raw['artifact_inventory']
                require(len(inventory) == 2 and {r['path'] for r in inventory} ==
                        {'model.json', 'projection.json'}
                        and all(type(r['bytes']) is int and r['bytes'] >= 0 for r in inventory)
                        and sum(r['bytes'] for r in inventory) == raw['artifact_bytes']
                        and next(r['sha256'] for r in inventory if r['path'] == 'model.json')
                        == raw['artifact_sha256']
                        and identity(inventory) == identity(trial['artifact_inventory']), 'native declared byte charge differs')
                for row in inventory:
                    owned.digest_string(row['sha256'])
                value = read(raw['metric_receipt_path'], raw['metric_receipt_sha256'])
                require(value['format'] == 'dope-benchmark-native-validation-kpi'
                        and type(value['version']) is int and value['version'] == 1
                        and value['partition'] == 'validation' and value['method'] == method
                        and value['objective'] == 'mean_log_density' and value['direction'] == 'maximize'
                        and value['implementation_sha256'] == entry['native_objective']['implementation_sha256']
                        and identity(value['value']) == identity(raw['native_kpi'])
                        and value['artifact_sha256'] == raw['artifact_sha256']
                        and value['validation_sha256'] == job['validation_sha256'], 'native metric binding differs')
                require(number(raw['wall_seconds']) and raw['wall_seconds'] >= 0
                        and len({identity(v) for v in (raw['wall_seconds'], trial['wall_seconds'],
                                                      summary['seconds'])}) == 1, 'native time differs')
                cell_costs.append(raw['wall_seconds']); costs.append(raw['wall_seconds'])
                rows.append(raw); used_trials.add(trial_key)
            total = math.fsum(cell_costs)
            require(number(selection['total_wall_seconds']) and total <= 43200
                    and math.isclose(total, selection['total_wall_seconds'], abs_tol=1e-9, rel_tol=1e-12), 'native cost cap or total differs')
            winner = min(range(len(rows)), key=lambda i: (-rows[i]['native_kpi'], rows[i]['artifact_bytes'], identity(rows[i]['config']), i))
            defaults = [i for i, r in enumerate(rows) if identity(r['config']) == identity(entry['default_config'])]
            require(len(defaults) == 1 and selection['selected_trial_index'] == winner
                    and identity(selection['selected_config']) == identity(rows[winner]['config']), 'native winner differs')
            for cell, index in ((default, defaults[0]), (selected, winner)):
                require(type(cell['selected_trial_index']) is int and cell['selected_trial_index'] == index
                        and identity(cell['config']) == identity(rows[index]['config'])
                        and identity(cell['native_validation_kpi']) == identity(rows[index]['native_kpi'])
                        and identity(cell['artifact_bytes']) == identity(rows[index]['artifact_bytes'])
                        and identity(cell['artifact_inventory']) == identity(rows[index]['artifact_inventory'])
                        and cell['model_sha256'] == rows[index]['artifact_sha256']
                        and identity(cell['native_objective']) == identity(selection['objective'])
                        and cell['fit_attempt_receipt_path'] == trials[(dataset, method, index)]['attempt_receipt_path']
                        and cell['fit_attempt_receipt_sha256'] == trials[(dataset, method, index)]['attempt_receipt_sha256']
                        and cell['shared_kpi_used_for_selection'] is False and cell['native_values_cross_ranked'] is False
                        and cell['projection_bytes_included'] is True
                        and cell['implementation_kind'] == ('author_library' if method == 'GaussianCopula' else 'study_reference')
                        and cell['historical_full_runtime_closure_upgraded'] is False
                        and cell['counts_as_dope_win'] is False, 'default or winner cell differs')
    require(len(seen) == 300 and used_trials == set(trials)
            and len({d for d, m in seen}) == 100
            and seen == {(d, m) for d in {d for d, m in seen}
                         for m in ('GaussianCopula', 'independent_marginals', 'Chow-Liu')}, 'native coverage differs')
    total = math.fsum(costs)
    require(number(report['native_trial_operation_seconds'])
            and math.isclose(total, report['native_trial_operation_seconds'], abs_tol=1e-9, rel_tol=1e-12), 'native overall time differs')
    owned.verify_refs(consumed)
    return dict(scope='native winner/default/grid and cost metadata replay only',
        datasets=100, native_lineages=300, default_native_cells=600, native_trial_receipts=1100,
        consumed_metadata_refs=len(consumed), consumed_metadata_refs_sha256=identity(consumed),
        native_receipt_lock_sha256=owned.NATIVE, native_report_sha256=owned.NATIVE_REPORT,
        original_methods_sha256=METHOD_SHA, unchanged_native_choices=True,
        legacy_selections_without_validation_hash=legacy_selections_without_validation_hash,
        legacy_trials_without_source_identity=legacy_trials_without_source_identity,
        full_native_reference_inventory_verified=False, runtime_worker_artifact_replay_required=True,
        native_likelihood_recomputed=False, new_generator_fits_started=0, native_selection_changed=False,
        official_tests_opened=False, sdv_v3_launched=False, publication_admitted=False,
        mfs_v2=None,ptf_v1=None,release_safe=None,superiority=None,counts_as_dope_win=False)
