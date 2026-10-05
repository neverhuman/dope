"""Bind descriptive matched results to immutable receipt and artifact hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from . import publish_density_matched as publisher
from .publish_s3_matched import schema


def build(results, figure_directory, figure_manifest_sha256):
    results, figure_directory = Path(results), Path(figure_directory)
    publisher.inputs.digest_string(figure_manifest_sha256)
    proof_path = figure_directory / 'figures-manifest.json'
    proof_bytes = publisher.regular_bytes(proof_path)
    publisher.inputs.require(hashlib.sha256(proof_bytes).hexdigest() == figure_manifest_sha256,
                             'frozen figure receipt changed')
    figure_proof = json.loads(proof_bytes)
    figure_schema = proof_path.with_suffix('.schema.json')
    publisher.inputs.require(publisher.regular_bytes(figure_schema) == publisher.serialized(schema(figure_proof)),
                             'figure receipt schema changed')
    figure_source = Path(__file__).with_name('publish_density_matched_figure.py')
    publisher.inputs.require(figure_proof['report_sha256'] == publisher.REPORT_SHA256
        and figure_proof['source_sha256'] == hashlib.sha256(publisher.regular_bytes(figure_source)).hexdigest()
        and type(figure_proof['datasets']) is int and figure_proof['datasets'] == 100
        and type(figure_proof['logical_cells']) is int and figure_proof['logical_cells'] == 6000
        and figure_proof['official_tests_opened'] is False
        and figure_proof['production_certified'] is False
        and all(figure_proof[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
        'figure source, report or research scope changed')
    path = results / (publisher.NAME + '.json')
    report = publisher.load_report(path, publisher.REPORT_SHA256)
    table, markdown = publisher.tables(report)
    expected = {'.schema.json': publisher.serialized(schema(report)),
                '.csv': table.encode(), '.md': markdown.encode()}
    for suffix, data in expected.items():
        publisher.inputs.require(publisher.regular_bytes(results / (publisher.NAME + suffix)) == data,
                                 'matched publication shape or table changed')
    artifacts = {}
    paths = [path] + [results / (publisher.NAME + suffix) for suffix in expected]
    paths += [figure_directory / f'density-matched-retention-{size}n.{extension}'
              for size in (1, 4) for extension in ('svg', 'pdf')]
    publisher.inputs.require(set(figure_proof['artifacts']) == {p.name for p in paths[-4:]},
                             'complete figure inventory required')
    for item in paths[-4:]:
        ref = figure_proof['artifacts'][item.name]
        data = publisher.regular_bytes(item)
        publisher.inputs.require(type(ref['bytes']) is int
            and len(data) == ref['bytes']
            and hashlib.sha256(data).hexdigest() == ref['sha256'],
            'frozen figure bytes changed')
    paths.append(proof_path)
    paths.append(figure_schema)
    for item in paths:
        data = publisher.regular_bytes(item)
        artifacts[str(item.relative_to(results))] = {
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    here = Path(__file__).parent
    sources = {name: hashlib.sha256(publisher.regular_bytes(here / name)).hexdigest() for name in (
        'publish_density_matched.py', 'publish_density_matched_figure.py',
        'publish_density_matched_manifest.py', 'density_matched_aggregate.py',
        'density_declared_runtime_replay.py', 'density_publication_inputs.py',
        'density_metric_replay.py', 'density_custody_replay.py', 'density_native_replay.py',
        'publish_dope_population_validation.py', 'publish_dope_population_fits.py',
        'publish_s3_matched.py')}
    return dict(format='dope-density-matched-population-validation-manifest', version=1,
        scope=report['scope'], datasets=100, logical_cells=6000,
        density_physical_batches=502, density_logical_cells=3600,
        logical_status_counts=report['logical_status_counts'], artifacts=artifacts,
        publisher_sources=sources, source_locks=report['source_locks'],
        figure_receipt_sha256=figure_manifest_sha256,
        immutable_receipt_lock=dict(path=str(publisher.inputs.ROOT / 'receipt-lock-v1.json'),
                                   sha256=publisher.RECEIPT_SHA256),
        immutable_reconciliation=dict(path=str(publisher.inputs.ROOT / 'reconciliation-v1.json'),
                                     sha256=publisher.RECONCILIATION_SHA256),
        citation_keys=['patki2016synthetic', 'chow1968approximating'],
        bibliography='DOPE_SYNTH_REFERENCES.bib',
        native_kpis_never_ranked_across_methods=True,
        common_utility_used_for_baseline_selection=False,
        fit_seeds=[11], sample_seeds=[101, 211, 307], sizes=[1, 4],
        all_frozen_cells_accounted=True, full_campaign_admitted=False,
        global_family_selected=False, official_tests_opened=False,
        production_certified=False, historical_full_runtime_closure_upgraded=False,
        mfs_v2=None, ptf_v1=None, release_safe_l3=None, paired_superiority=None,
        counts_as_dope_win=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path, required=True)
    parser.add_argument('--figure-manifest-sha256', required=True)
    args = parser.parse_args()
    manifest = build(args.results_dir, args.figure_dir, args.figure_manifest_sha256)
    path = args.results_dir / (publisher.NAME + '.manifest.json')
    with path.open('xb') as stream:
        stream.write(publisher.serialized(manifest))
    with path.with_suffix('.schema.json').open('xb') as stream:
        stream.write(publisher.serialized(schema(manifest)))


if __name__ == '__main__':
    main()
