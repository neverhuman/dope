"""Private TRAIN-only corpus loading; final families are never opened by this module."""
import csv
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from .contracts import canonical, digest, require, validate_corpus
from .data import role_indices
from .priors import table


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def read_dataset(item):
    require(item['meta_role']in('train','development') and item['source_role']=='official_train'
            and item['rights']['admitted']is True,'source_role_not_allowed')
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
    row_ids = [item["train_sha256"]+f":{i}" for i in range(len(rows))]
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
    return dict(rows=[[row[j] for j in order] for row in rows], kinds=[kinds[j] for j in order],
                slots=[j+1 if j != target else 0 for j in order], row_ids=row_ids,
                task=item["task"], family_id=item["family_id"],
                roles=role_indices(row_ids, item["split_seed"], policy=policy, groups=groups, times=times),
                landmarks=read_landmarks(item), evidence="unqualified_real_development")


def read_landmarks(item):
    declared = item.get("landmarks")
    if declared is None: return ([0.]*23, [False]*23)
    manifest_raw = Path(declared["manifest_path"]).read_bytes()
    require(hashlib.sha256(manifest_raw).hexdigest() == declared["manifest_sha256"], "landmark_manifest_digest")
    manifest = json.loads(manifest_raw)
    binding = manifest["binding"]
    require(manifest["format"] == "dope-landmark-cache-cell-v1"
            and binding["train_sha256"] == item["train_sha256"]
            and binding["source_role"] == "official_train"
            and binding["target"]["index"] in (item["target_index"], -1)
            and manifest["key"] == digest(binding), "landmark_source_binding")
    require(binding["target"]["index"] != -1 or item["target_index"] == len(item["column_kinds"])-1,
            "landmark_target_binding")
    raw = Path(declared["report_path"]).read_bytes()
    require(len(raw) == manifest["report_bytes"] and hashlib.sha256(raw).hexdigest() == manifest["report_sha256"],
            "landmark_report_digest")
    report = json.loads(raw)
    require(report["options"] == binding["options"] and report["panel"] == "landmarks-native-v1"
            and report["kind"] == "real" and report["privacy"] == "private"
            and [c["id"] for c in report["cells"]] == list(range(1,24)), "landmark_recipe_binding")
    values, mask = [], []
    for cell in report["cells"]:
        valid = cell["state"] == "complete" and cell.get("gain") is not None
        value = cell["gain"] if valid else 0.
        require(np.isfinite(value), "landmark_nonfinite")
        values.append(float(value)); mask.append(valid)
    return values, mask


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
