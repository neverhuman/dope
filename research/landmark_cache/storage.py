"""Private, conditional, content-verified S3 publication; no raw table uploads."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import os
import uuid
from pathlib import Path

from .plan import canonical, require


class S3Store:
    def __init__(self, client, bucket, prefix):
        require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9/_-]{0,180}", prefix)
                and ".." not in prefix and not prefix.endswith("/"), "invalid_output_prefix")
        self.client, self.bucket, self.prefix = client, bucket, prefix

    def get(self, key):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.prefix + "/" + key)
        except self.client.exceptions.ClientError as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                return None
            raise
        stream = response["Body"]
        try:
            raw = stream.read(32_000_001)
        finally:
            stream.close()
        require(len(raw) <= 32_000_000, "cache_object_too_large")
        return raw

    def put_once(self, key, raw):
        checksum = hashlib.sha256(raw).digest()
        try:
            self.client.put_object(Bucket=self.bucket, Key=self.prefix + "/" + key,
                                   Body=raw, IfNoneMatch="*", ServerSideEncryption="AES256",
                                   ChecksumSHA256=base64.b64encode(checksum).decode(),
                                   ContentType="application/json",
                                   Metadata={"sha256": checksum.hex(), "privacy": "private-measurement"})
        except self.client.exceptions.ClientError as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code not in ("PreconditionFailed", "412"):
                raise
            require(self.get(key) == raw, "immutable_cache_collision")

    def lookup(self, cell):
        raw = self.get("cells/" + cell["key"] + "/manifest.json")
        if raw is None:
            return None
        manifest = json.loads(raw)
        require(manifest["binding"] == cell["binding"] and manifest["key"] == cell["key"]
                and manifest["format"] == "dope-landmark-cache-cell-v1", "cache_binding_mismatch")
        require(re.fullmatch(r"[0-9a-f]{64}", manifest["report_sha256"])
                and type(manifest["report_bytes"]) is int and 0 < manifest["report_bytes"] <= 32_000_000,
                "cache_report_identity_invalid")
        body = self.get("objects/sha256/" + manifest["report_sha256"] + ".json")
        require(body is not None and hashlib.sha256(body).hexdigest() == manifest["report_sha256"]
                and len(body) == manifest["report_bytes"], "cached_report_corrupt")
        return body

    def publish(self, cell, report):
        raw = canonical(report)
        checksum = hashlib.sha256(raw).hexdigest()
        self.put_once("objects/sha256/" + checksum + ".json", raw)
        manifest = dict(format="dope-landmark-cache-cell-v1", key=cell["key"],
                        binding=cell["binding"], report_sha256=checksum, report_bytes=len(raw),
                        privacy="private", official_test_access=False, model_training=False)
        self.put_once("cells/" + cell["key"] + "/manifest.json", canonical(manifest))
        return manifest


class LocalStore(S3Store):
    """Same immutable namespace for hosts without AWS credentials; courier publishes later."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def path(self, key):
        require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9/_.-]*", key) and ".." not in key,
                "invalid_local_cache_key")
        return self.root / key

    def get(self, key):
        path = self.path(key)
        if not path.exists():
            return None
        require(not path.is_symlink() and path.stat().st_size <= 32_000_000, "invalid_local_cache_object")
        return path.read_bytes()

    def put_once(self, key, raw):
        path = self.path(key)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = path.with_name("." + uuid.uuid4().hex + ".partial")
        descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                require(self.get(key) == raw, "immutable_cache_collision")
        finally:
            tmp.unlink(missing_ok=True)


class LocalTrainSource:
    def __init__(self, root, plan):
        self.root = Path(root).resolve()
        self.allowed = {(c["binding"]["source_bucket"], c["binding"]["source_key"]):
                        c["binding"]["train_sha256"] for c in plan["cells"]}

    def get_object(self, *, Bucket, Key):
        require((Bucket, Key) in self.allowed, "source_object_not_in_TRAIN_schedule")
        path = self.root / (self.allowed[(Bucket, Key)] + ".csv")
        require(not path.is_symlink(), "source_link_refused")
        return {"Body": path.open("rb"), "ContentLength": path.stat().st_size}
