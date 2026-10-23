"""Private TRAIN-only corpus loading; final families are never opened by this module."""
import csv
from collections import OrderedDict
from datetime import datetime
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import subprocess
import numpy as np
from .contracts import canonical, digest, require, validate_corpus
from .data import role_indices
from .priors import table


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def read_dataset(item, *, include_landmarks=True):
    require(item['meta_role']in('train','development') and item['source_role']=='official_train'
            and item['rights']['admitted']is True and bool(item['rights']['license_id'])
            and bool(item['family_id']) and not item['family_id'].startswith('unresolved:'),
            'source_role_not_allowed')
    path = Path(item["path"])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 200000000,
            "source_path_unsupported")
    require(file_sha(path) == item["train_sha256"], "source_digest_mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if item["has_header"]: next(reader)
        rows=[];cells=0
        for row in reader:
            cells+=len(row)
            require(len(rows)<200000 and cells<=2000000, "source_materialization_envelope")
            rows.append(row)
    kinds, target = item["column_kinds"], item["target_index"]
    require(0 <= target < len(kinds) and all(len(row) == len(kinds) for row in rows), "source_schema_drift")
    order = [j for j in range(len(kinds)) if j != target] + [target]
    # Content hashes include held-out targets. They bind lineage, never row selection.
    require(isinstance(item.get('dataset_id'), str) and bool(item['dataset_id']), 'physical_dataset_id_required')
    physical = digest(dict(family=item['family_id'], dataset=item['dataset_id']))
    row_ids = [physical+f":{i}" for i in range(len(rows))]
    policy = item["split_policy"]
    groups = [row[item["group_index"]] for row in rows] if policy == "group" else None
    times = None
    if policy == "time":
        require(item.get('time_kind') in ('numeric','iso8601'), 'explicit_time_kind_required')
        raw_times=[row[item['time_index']] for row in rows]
        if item['time_kind']=='numeric':times=[float(v) for v in raw_times]
        else:
            parsed=[datetime.fromisoformat(v) for v in raw_times]
            require(all(v.tzinfo is not None and v.utcoffset()is not None for v in parsed), 'timezone_required')
            times=[v.timestamp() for v in parsed]
        require(all(np.isfinite(v) for v in times), 'nonfinite_physical_time')
    require(policy == "iid" or (item.get("group_index") if policy == "group" else item.get("time_index")) != target,
            "target_cannot_define_physical_blocks")
    # Entity/time identifiers cannot become predictor features.
    excluded = {item.get("group_index"), item.get("time_index")}
    order = [j for j in order if j == target or j not in excluded]
    result = dict(rows=[[row[j] for j in order] for row in rows], kinds=[kinds[j] for j in order],
                slots=[j+1 if j != target else 0 for j in order], row_ids=row_ids,
                task=item["task"], family_id=item["family_id"],
                roles=role_indices(row_ids, item["split_seed"], policy=policy, groups=groups, times=times),
                evidence="unqualified_real_development")
    result['landmarks'] = read_landmarks(item, result) if include_landmarks else ([0.]*23, [False]*23)
    return result


LANDMARK_OPTIONS = dict(panel='landmarks-native-v1', folds=5, ridge_rows=8192,
                        small_rows=4096, knn_reference=2048, spectrum_features=64,
                        work_budget=2000000000, seed=0)


def landmark_input(item, data):
    """A new, private fit-only input; never reinterpret a whole-TRAIN report."""
    require(item['split_policy'] == 'iid', 'scoped_landmark_block_policy_unsupported')
    indices = sorted(data['roles']['fit'])
    require(indices and not set(indices).intersection(
        data['roles']['tune']+data['roles']['calibration']+data['roles']['confirmation']), 'landmark_role_overlap')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow([f'c{i}' for i in range(len(data['kinds']))])
    writer.writerows(data['rows'][i] for i in indices)
    raw = stream.getvalue().encode('utf-8')
    scope = dict(format='dope-fit-landmark-scope-v1', source_sha256=item['train_sha256'],
                 dataset_id=item['dataset_id'], family_id=item['family_id'], source_role='official_train',
                 split_policy=item['split_policy'], split_seed=item['split_seed'], allowed_roles=['fit'],
                 row_count=len(indices), row_ids_sha256=digest([data['row_ids'][i] for i in indices]),
                 input_sha256=hashlib.sha256(raw).hexdigest(), source_target_index=item['target_index'],
                 column_kinds=data['kinds'], slots=data['slots'], task=data['task'])
    return raw, scope


def read_landmarks(item, data):
    declared = item.get("landmarks")
    if declared is None: return ([0.]*23, [False]*23)
    manifest_raw = Path(declared["manifest_path"]).read_bytes()
    require(hashlib.sha256(manifest_raw).hexdigest() == declared["manifest_sha256"], "landmark_manifest_digest")
    manifest = json.loads(manifest_raw)
    require(manifest.get('format') == 'dope-role-landmarks-v1', 'role_scoped_landmarks_required')
    _, scope = landmark_input(item, data)
    binding = manifest['binding']
    require(binding['scope'] == scope and binding['options'] == LANDMARK_OPTIONS
            and binding['engine_sha256'] == declared['engine_sha256']
            and manifest['key'] == digest(binding), 'landmark_role_binding')
    raw = Path(declared["report_path"]).read_bytes()
    require(len(raw) == manifest["report_bytes"] and hashlib.sha256(raw).hexdigest() == manifest["report_sha256"],
            "landmark_report_digest")
    report = json.loads(raw)
    require(report["options"] == binding["options"] and report["panel"] == "landmarks-native-v1"
            and report["kind"] == "real" and report["privacy"] == "private"
            and report['task'] == scope['task'] and report['rows'] == scope['row_count']
            and [c["id"] for c in report["cells"]] == list(range(1,24)), "landmark_recipe_binding")
    values, mask = [], []
    for cell in report["cells"]:
        valid = cell["state"] == "complete" and cell.get("gain") is not None
        value = cell["gain"] if valid else 0.
        require(np.isfinite(value), "landmark_nonfinite")
        values.append(float(value)); mask.append(valid)
    return values, mask


def prepare_landmarks(item, engine, expected_engine_sha256, out, *, wall_seconds=300, threads=4):
    """Run the pinned native panel on admitted IID fit rows and publish the manifest last."""
    require(type(wall_seconds) is int and 1 <= wall_seconds <= 1800
            and type(threads) is int and 1 <= threads <= 4, 'landmark_allocation_unsupported')
    engine, out = Path(engine).resolve(), Path(out)
    require(engine.is_file() and file_sha(engine) == expected_engine_sha256, 'landmark_engine_digest')
    data = read_dataset(item, include_landmarks=False)
    raw, scope = landmark_input(item, data)
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    def write(name, content):
        with (out/name).open('xb') as stream:
            os.chmod(out/name, 0o600); stream.write(content); stream.flush(); os.fsync(stream.fileno())
    write('fit.csv', raw)
    binding = dict(scope=scope, engine_sha256=expected_engine_sha256, options=LANDMARK_OPTIONS)
    plan = canonical(dict(binding=binding, threads=threads, wall_seconds=wall_seconds))
    write('plan.json', plan)
    command = [str(engine), '--threads', str(threads), 'landmarks', 'real', str((out/'fit.csv').resolve()),
               '--target', str(len(data['kinds'])-1), '--task', data['task'], '--json']
    for name, value in LANDMARK_OPTIONS.items():command += ['--'+name.replace('_','-'), str(value)]
    try:
        with (out/'report.json').open('xb') as stream:
            os.chmod(out/'report.json', 0o600)
            result = subprocess.run(command, stdout=stream, stderr=subprocess.DEVNULL,
                                    timeout=wall_seconds, check=False)
            stream.flush(); os.fsync(stream.fileno())
        require(result.returncode == 0, 'native_scoped_landmark_failed')
        report_raw = (out/'report.json').read_bytes()
        report = json.loads(report_raw)
        require(report['options'] == LANDMARK_OPTIONS and report['kind'] == 'real'
                and report['privacy'] == 'private' and report['panel'] == LANDMARK_OPTIONS['panel']
                and report['task'] == scope['task'] and report['rows'] == scope['row_count']
                and [c['id'] for c in report['cells']] == list(range(1,24)), 'scoped_landmark_report_invalid')
        manifest = dict(format='dope-role-landmarks-v1', binding=binding, key=digest(binding),
                        report_bytes=len(report_raw), report_sha256=hashlib.sha256(report_raw).hexdigest(),
                        plan_sha256=hashlib.sha256(plan).hexdigest(),
                        producer_sha256=file_sha(__file__), platform=platform.platform(),
                        python_version=platform.python_version())
        # Validate every label before publication. Failed/unsupported cells retain their masks.
        for cell in report['cells']:
            require(cell.get('gain') is None or np.isfinite(cell['gain']), 'landmark_nonfinite')
        write('manifest.json', canonical(manifest))
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        write('failure.json', canonical(dict(state='failed', error_type=type(error).__name__)))
        require(False, 'native_scoped_landmark_failed')
    return dict(manifest_path=str((out/'manifest.json').resolve()),
                manifest_sha256=file_sha(out/'manifest.json'), report_path=str((out/'report.json').resolve()),
                engine_sha256=expected_engine_sha256)


class Corpus:
    def __init__(self, manifest=None, *, seed=20261023, prior_only=False):
        self.seed, self.prior_only, self.loaded = seed, prior_only, OrderedDict()
        self.manifest = validate_corpus(manifest) if manifest is not None else None
        require(prior_only or self.manifest is not None, "real_corpus_manifest_required")
        self.identity = digest(dict(manifest=self.manifest, prior_recipe="owned-mechanisms-v1",
                                    prior_only=prior_only, seed=seed))

    def get(self, role, cursor):
        require(role in ("train", "development"), "final_access_forbidden")
        # Exactly half real/prior with a real corpus; all-prior runs are explicitly labeled.
        use_prior = self.prior_only or cursor % 2 == 0
        if use_prior:
            families = list(range(7)) if role == "train" else [7,8]
            family = families[(cursor//2) % len(families)]
            result = table(family, self.seed + cursor//(2*len(families)))
            result["landmarks"] = ([0.]*23, [False]*23)
            return result
        entries = [d for d in self.manifest["datasets"] if d["meta_role"] == role]
        families = sorted({d["family_id"] for d in entries})
        require(families, "empty_corpus_role")
        family = families[(cursor//2) % len(families)]
        members = sorted([d for d in entries if d["family_id"] == family], key=lambda d:d["train_sha256"])
        chosen = members[(cursor//(2*len(families))) % len(members)]
        key = chosen["train_sha256"]
        if key not in self.loaded:
            self.loaded[key] = read_dataset(chosen)
            if len(self.loaded)>2:self.loaded.popitem(last=False)
        self.loaded.move_to_end(key)
        return self.loaded[key]
