"""Bind rights-safe neural comparison artifacts to immutable scratch anchors."""
import hashlib
import json
from pathlib import Path

from research.benchmark import publish_sdv_matched as publisher
from research.benchmark.publish_s3_matched import schema


def build(results):
    results = Path(results)
    report = publisher.committed(results / (publisher.NAME + '.json'), publisher.REPORT_SHA256)
    csv, markdown = publisher.tables(report)
    serialize = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    for suffix, expected in {'.csv': csv.encode(), '.md': markdown.encode(),
                             '.schema.json': serialize(schema(report))}.items():
        path = results / (publisher.NAME + suffix)
        publisher.inputs.require(not any(p.is_symlink() for p in (path, *path.parents))
                                 and path.read_bytes() == expected, 'derived publication changed')
    artifacts = {}
    for suffix in ('.json', '.schema.json', '.csv', '.md', '.svg', '.pdf'):
        path = results / (publisher.NAME + suffix)
        publisher.inputs.require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                                 'publication artifact not owned')
        data = path.read_bytes()
        artifacts[path.name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    directory = Path(__file__).parent
    sources = {}
    for name in ('publish_sdv_matched.py', 'publish_sdv_matched_figure.py',
                 'publish_sdv_matched_manifest.py', 'sdv_matched_aggregate.py',
                 'density_publication_inputs.py', 'density_metric_replay.py',
                 'density_declared_runtime_replay.py', 'publish_sdv_population_native.py',
                 'sdv_native_accounting.py', 'sdv_native_continuation.py', 'reconcile_sdv_population.py'):
        path = directory / name
        publisher.inputs.require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                                 'publisher source not owned')
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(format='dope-sdv-matched-validation-publication-manifest', version=1,
        scope=report['scope'], datasets=100, logical_cells=4800, sdv_physical_batches=78,
        sdv_measured_logical_cells=486, sdv_sample_unavailable_logical_cells=1914,
        sdv_available_configuration_bindings=81, sdv_unavailable_configuration_bindings=319,
        original_sdv_native_classes=report['sdv_native_ledger_class_counts'],
        original_sdv_scheduling_cutoff_reasons=report['sdv_scheduling_cutoff_reasons'],
        artifacts=artifacts, publisher_sources=sources, source_locks=report['source_locks'],
        immutable_receipt_lock=dict(path=str(publisher.ROOT / 'receipt-lock-v1.json'), sha256=publisher.RECEIPTS),
        immutable_reconciliation=dict(path=str(publisher.ROOT / 'reconciliation-v1.json'), sha256=publisher.RECONCILIATION),
        immutable_auxiliary_inventory=dict(path=str(publisher.ROOT / 'matched-auxiliary.lock-v1.json'), sha256=publisher.AUXILIARY),
        citation_keys=['xu2019modeling', 'patki2016synthetic'], bibliography='DOPE_SYNTH_REFERENCES.bib',
        native_kpis_never_ranked_across_methods=True, shared_kpi_used_for_selection=False,
        method_conclusions_from_scheduling_cutoffs=False, fit_seeds=[11],
        sample_seeds=[101, 211, 307], sizes=[1, 4],
        new_generator_fits_started=0, new_samples_generated=0, sdv_v3_launched=False,
        official_tests_opened=False, production_certified=False, full_campaign_admitted=False,
        counts_as_dope_win=False, mfs_v2=None, ptf_v1=None, release_safe_l3=None, paired_superiority=None)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args(); value = build(args.results_dir)
    for suffix, data in (('.manifest.json', value), ('.manifest.schema.json', schema(value))):
        (args.results_dir / (publisher.NAME + suffix)).write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__': main()
