"""Guarded ARF fit/sample contract for training-derived research worker views."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import time

from research.benchmark import arf_native as native
from research.benchmark import arf_runtime_guard as runtime


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def seed(value):
    check(type(value) is int and 0 <= value < 2**32, 'invalid ARF seed')


def deadline(value):
    check(type(value) in (float, int) and math.isfinite(value)
          and time.time() < value, 'ARF deadline exceeded')


def numeric(data):
    try:
        rows = [[float(v) for v in row] for row in csv.reader(io.StringIO(data.decode()))]
    except (UnicodeError, ValueError, csv.Error):
        raise ValueError('invalid common-numeric input') from None
    check(len(rows) >= 2 and len(rows[0]) >= 2
          and all(len(r) == len(rows[0]) for r in rows)
          and all(math.isfinite(v) and 0 <= v <= 1 for r in rows for v in r),
          'invalid common-numeric input')
    return rows


def config(value):
    check(type(value) is dict and set(value) == {'num_trees', 'min_node_size', 'max_iters', 'alpha'},
          'invalid ARF configuration')
    for name, low, high in (('num_trees', 1, 100), ('min_node_size', 1, 100_000), ('max_iters', 0, 10)):
        check(type(value[name]) is int and low <= value[name] <= high, 'invalid ARF configuration')
    check(type(value['alpha']) is float and math.isfinite(value['alpha'])
          and 0 <= value['alpha'] <= 1, 'invalid ARF configuration')


def owned_bytes(path, expected, base):
    runtime.digest(expected)
    p = runtime.safe(path, base)
    check('evaluator' not in p.parts and p.name != 'test.csv'
          and not (p.parent / 'test.csv').exists(), 'unadmitted ARF input path')
    check(stat.S_ISREG(p.stat().st_mode), 'ARF input must be regular')
    data = p.read_bytes()
    check(hashlib.sha256(data).hexdigest() == expected, 'ARF input hash changed')
    return data


def destination(path, base):
    p = Path(path)
    check(p.is_absolute() and p.parent == p.parent.resolve(strict=True)
          and p.parent.is_relative_to(base) and not p.exists() and not p.is_symlink(),
          'ARF output path is not reserved')
    return p


def inventory(root):
    files = sorted(root.iterdir())
    check(all(stat.S_ISREG(p.lstat().st_mode) for p in files), 'ARF artifact alias or special entry')
    return {p.name: {'sha256': runtime.hash_file(p), 'bytes': p.stat().st_size} for p in files}


def check_runtime(path, expected, base):
    lock = runtime.verify(path, expected, base)
    for module in (__file__, native.__file__):
        source = runtime.safe(module, base)
        check(source.is_relative_to(lock['source_root'])
              and lock['source_files'].get(str(source), {}).get('sha256') == runtime.hash_file(source),
              'executing ARF adapter is not the frozen source')
    return lock


def fit(*, runtime_path, runtime_sha256, base, train, train_sha256,
        validation, validation_sha256, projection, projection_sha256,
        artifact, configuration, fit_seed, deadline_epoch):
    deadline(deadline_epoch)
    check(deadline_epoch - time.time() <= 600, 'ARF fit ceiling exceeds 600 seconds')
    check_runtime(runtime_path, runtime_sha256, base)
    seed(fit_seed); config(configuration)
    check(Path(train).name == 'train.csv' and Path(validation).name == 'validation.csv'
          and Path(projection).name == 'projection.json'
          and Path(train).parent == Path(validation).parent == Path(projection).parent,
          'ARF worker partition differs')
    training = numeric(owned_bytes(train, train_sha256, base))
    heldout = numeric(owned_bytes(validation, validation_sha256, base))
    projection_bytes = owned_bytes(projection, projection_sha256, base)
    check(len(training[0]) == len(heldout[0]), 'ARF worker widths differ')
    output = destination(artifact, base)
    deadline(deadline_epoch)
    import numpy as np
    from arfpy.arf import arf
    np.random.seed(fit_seed)
    frames = native.prepare_frames(np.asarray(training), np.asarray(heldout))
    start = time.monotonic()
    model = arf(frames[0], num_trees=configuration['num_trees'],
                min_node_size=configuration['min_node_size'], max_iters=configuration['max_iters'],
                delta=0, early_stop=True, verbose=False, random_state=fit_seed, n_jobs=16)
    model.forde(dist='truncnorm', oob=False, alpha=configuration['alpha'])
    value = native.heldout_mean_log_density(model, frames[1])
    deadline(deadline_epoch)
    native.save_sampler(model, output)
    (output / 'projection.json').write_bytes(projection_bytes)
    (output / 'adapter.json').write_text(json.dumps(dict(format='dope-arf-research-adapter-v1',
        runtime_sha256=runtime_sha256, fit_seed=fit_seed, configuration=configuration,
        source_rows_required=False, support='clip_to_unit_interval'), sort_keys=True) + '\n')
    files = inventory(output)
    check((output / 'projection.json').read_bytes() == projection_bytes, 'ARF projection copy differs')
    check_runtime(runtime_path, runtime_sha256, base)
    deadline(deadline_epoch)
    return dict(native_kpi=dict(objective='heldout_forde_mean_log_density', direction='maximize',
        value=value, partition='validation', validation_sha256=validation_sha256,
        preprocessing_fit_partition='train_only', fit_seed=fit_seed,
        not_comparable_across_methods=True), artifact_inventory=files,
        artifact_bytes=sum(r['bytes'] for r in files.values()),
        fit_and_density_seconds=time.monotonic()-start,
        training_rows=len(training), validation_rows=len(heldout), official_tests_opened=False)


def sample(*, runtime_path, runtime_sha256, base, artifact, expected_inventory,
           sample_seed, rows, output, deadline_epoch):
    deadline(deadline_epoch)
    check_runtime(runtime_path, runtime_sha256, base)
    seed(sample_seed)
    check(type(rows) is int and 1 <= rows <= 10_000_000, 'invalid ARF sample size')
    root = runtime.safe(artifact, base)
    names = {'model.json', 'bounds.csv', 'continuous.csv', 'categories.csv',
             'projection.json', 'adapter.json'}
    check(type(expected_inventory) is dict and set(expected_inventory) == names
          and set(p.name for p in root.iterdir()) == names, 'ARF artifact inventory differs')
    blobs = {}
    for name in sorted(names):
        row = expected_inventory[name]
        data = owned_bytes(root / name, row['sha256'], base)
        check(type(row['bytes']) is int and len(data) == row['bytes'], 'ARF artifact bytes differ')
        blobs[name] = data
    metadata = runtime.decode(blobs['model.json'])
    columns = metadata['names']
    check(metadata['format'] == 'dope-arf-forge-factors-v1' and metadata['source_rows_required'] is False
          and metadata['restricted_research_artifact'] is True
          and type(columns) is list and len(columns) >= 2
          and columns == [f'c{i}' for i in range(len(columns))]
          and type(metadata['num_trees']) is int and 1 <= metadata['num_trees'] <= 100
          and metadata['dist'] == 'truncnorm'
          and len(metadata['factor_columns']) == len(metadata['object_columns']) == len(columns)
          and all(type(v) is bool for v in metadata['factor_columns'] + metadata['object_columns'])
          and set(metadata['files']) == {'bounds.csv','continuous.csv','categories.csv'}
          and all(metadata['files'][n] == expected_inventory[n]['sha256'] for n in metadata['files']),
          'ARF factor declaration differs')
    adapter = runtime.decode(blobs['adapter.json'])
    check(adapter['format'] == 'dope-arf-research-adapter-v1'
          and adapter['support'] == 'clip_to_unit_interval'
          and adapter['runtime_sha256'] == runtime_sha256 and adapter['source_rows_required'] is False,
          'ARF artifact runtime differs')
    path = destination(output, base)
    deadline(deadline_epoch)
    import numpy as np
    import pandas as pd
    from arfpy.arf import arf
    model = arf.__new__(arf)
    model.p=len(columns); model.orig_colnames=columns; model.num_trees=metadata['num_trees']
    model.factor_cols=pd.Series(metadata['factor_columns'],index=columns,dtype=bool)
    model.object_cols=pd.Series(metadata['object_columns'],index=columns,dtype=bool)
    model.levels=metadata['levels']; model.dist=metadata['dist']
    for name, field in (('bounds.csv','bnds'),('continuous.csv','params'),('categories.csv','class_probs')):
        setattr(model,field,pd.read_csv(io.BytesIO(blobs[name]),float_precision='round_trip'))
    np.random.seed(sample_seed)
    generated=model.forge(rows).to_numpy(dtype=float)
    check(generated.shape==(rows,len(columns)) and np.isfinite(generated).all(),
          'ARF generated table is invalid')
    deadline(deadline_epoch)
    np.savetxt(path,np.clip(generated,0,1),delimiter=',',fmt='%.17g')
    sample_hash = runtime.hash_file(path)
    check(inventory(root) == expected_inventory, 'ARF artifact changed during sampling')
    check_runtime(runtime_path, runtime_sha256, base)
    deadline(deadline_epoch)
    return dict(rows=rows, columns=len(columns), sample_sha256=sample_hash,
                artifact_bytes=sum(r['bytes'] for r in expected_inventory.values()),
                official_tests_opened=False)
