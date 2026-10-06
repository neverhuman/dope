"""Use the existing checked APP recipes and explicit local custody map.

Metadata/source preparation only. Numeric/artifact payloads use a separate,
future admitted reader; they are never accepted by this custody reader.
"""
from pathlib import Path

from research.benchmark import tabsyn_metric_operation_inputs as inputs


def read_local(ref):
    path = Path(ref['path'])
    inputs.require(not any(p.is_symlink() for p in (path, *path.parents)), 'custody_alias_rejected')
    return path.read_bytes()


def make_custody_reader(mapping_ref, *, local_reader=read_local, adapter_ref=None):
    """Resolve exact (original path, digest); recheck the captured copy per read."""
    mapping = inputs.decode(mapping_ref, local_reader)
    inputs.require(mapping['format'] == 'dope-TS-frozen-closed143-verified-metadata-source-byte-map-v1'
                   and mapping['resolver_must_rehash_each_read'] is True
                   and mapping['sample_or_model_payloads_copied'] is False
                   and mapping['original_TS_parent_kernel_exit'] is None
                   and mapping['CSV_rows_decoded'] is False, 'custody_scope_changed')
    lookup = {}
    for row in mapping['entries']:
        key = (row['original_path'], row['sha256'])
        inputs.require(key not in lookup and type(row['bytes']) is int and row['bytes'] >= 0,
                       'duplicate_or_invalid_custody_entry')
        lookup[key] = {'path': row['local_path'], 'sha256': row['sha256'], 'bytes': row['bytes']}
    if adapter_ref is not None:
        inputs.require(adapter_ref['sha256'] == inputs.ADAPTER_SHA256, 'unreviewed_APP_adapter')

    def read(ref):
        inputs.require(type(ref) is dict and type(ref.get('path')) is str
                       and type(ref.get('sha256')) is str, 'invalid_custody_reference')
        name = Path(ref['path']).name
        inputs.require(Path(name).suffix in ('.json', '.py') and name not in ('model.json', 'projection.json'),
                       'learned_or_numeric_body_forbidden_in_metadata_reader')
        if adapter_ref is not None and inputs.canonical(ref) == inputs.canonical(adapter_ref):
            return inputs.checked(adapter_ref, local_reader)
        key = (ref['path'], ref['sha256'])
        if key not in lookup:
            raise FileNotFoundError('captured_metadata_reference_unmaterialized')
        data = inputs.checked(lookup[key], local_reader)
        if 'bytes' in ref:
            inputs.require(type(ref['bytes']) is int and ref['bytes'] == len(data), 'original_size_changed')
        return data

    return read


def key(job):
    inputs.require(type(job['lineage_sha256']) is str
                   and inputs.re.fullmatch('[0-9a-f]{64}', job['lineage_sha256']) is not None
                   and type(job['sample_seed']) is int and job['sample_seed'] in (101, 211, 307)
                   and type(job['row_multiplier']) is int and job['row_multiplier'] in (1, 4),
                   'invalid_recipe_identity')
    return job['lineage_sha256'], job['sample_seed'], job['row_multiplier']


def select_recipe(report, lineage_sha256, sample_seed, row_multiplier):
    inputs.require(report['format'] == 'dope-tabsyn-shared-validation-input-preparation-v1'
                   and report['execution_enabled'] is False and report['official_tests_opened'] is False
                   and report['historical_parent_exit'] is None and report['physical_samples_rehashed'] is False
                   and report['numeric_rows_decoded'] is False
                   and all(report[name] is None for name in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority')),
                   'recipe_report_scope_changed')
    identities = [key(job) for job in report['jobs']]
    inputs.require(type(report['sample_cells']) is int and report['sample_cells'] == 600
                   and len(identities) == len(set(identities)) == 143
                   and report['status_counts'] == {'admitted_new_CPU': 95, 'admitted_retained_GPU': 48,
                                                 'unavailable_aggregate_admission_gap': 8,
                                                 'unavailable_monitor': 31, 'unstarted': 418},
                   'closed143_recipe_population_changed')
    chosen = (lineage_sha256, sample_seed, row_multiplier)
    key({'lineage_sha256': lineage_sha256, 'sample_seed': sample_seed, 'row_multiplier': row_multiplier})
    if chosen not in identities:
        return None
    job = report['jobs'][identities.index(chosen)]
    inputs.require(job['column_order'] == 'projected_inputs_then_target'
                   and job['sample_CSV_to_common_permutation'] == list(range(len(job['column_kinds'])))
                   and all(type(x) is int for x in job['sample_CSV_to_common_permutation'])
                   and type(job['synthetic_csv_header_rows']) is int and job['synthetic_csv_header_rows'] == 1
                   and job['historical_parent_exit'] is None and job['execution_enabled'] is False
                   and job['official_tests_opened'] is False
                   and all(job[name] is None for name in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority')),
                   'recipe_order_or_scope_changed')
    return job


def prepare_recipe(report_ref, adapter_ref, lineage_sha256, sample_seed, row_multiplier,
                   *, metadata_reader, source_reader, report_reader=read_local):
    """Reconstruct the selected input from operation receipts, then join recipe."""
    report = inputs.decode(report_ref, report_reader)
    recipe = select_recipe(report, lineage_sha256, sample_seed, row_multiplier)
    prepared = inputs.prepare_operation(report['snapshot_ref'], adapter_ref, lineage_sha256,
                                        sample_seed, row_multiplier,
                                        metadata_reader=metadata_reader, source_reader=source_reader)
    if recipe is None:
        inputs.require(prepared['status'] == 'unavailable', 'admitted_operation_missing_from_recipe_report')
        return prepared
    if prepared['status'] == 'unavailable':
        return prepared
    inputs.require(inputs.canonical(prepared['evaluator_input']) == inputs.canonical(recipe),
                   'recipe_differs_from_operation_receipt_bindings')
    prepared['job_identity'].update(recipe_report_ref=report_ref, recipe_sha256=inputs.digest(recipe))
    prepared['job_sha256'] = inputs.digest(prepared['job_identity'])
    prepared['recipe_report_ref'] = report_ref
    prepared['execution_boundary'] = {
        'receipt_admission': 'TabSyn fit and sample operation receipts',
        'evaluator_API': 'evaluate(fit, validation, synthetic, column_kinds)',
        'train_csv_format': recipe['train_ref']['csv_format'],
        'validation_csv_format': recipe['validation_ref']['csv_format'],
        'synthetic_csv_format': 'headered_numeric',
        'synthetic_csv_header_rows': 1,
        'additional_column_permutation': None,
    }
    return prepared
