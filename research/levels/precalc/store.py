"""Private immutable publication. Only JSON reports/manifests, never source CSVs."""
import base64
import hashlib
import re
from pathlib import Path
from ..contracts import require


def locator(value):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9/_.-]{0,900}',value)and'..'not in value,'precalc_object_locator')
    return value


class PrivateStore:
    def __init__(self,client,bucket,prefix):
        self.client,self.bucket,self.prefix=client,bucket,locator(prefix)

    def get(self,key):
        try:r=self.client.get_object(Bucket=self.bucket,Key=self.prefix+'/'+locator(key))
        except Exception as error:
            if getattr(error,'response',{}).get('Error',{}).get('Code')in('NoSuchKey','404'):return None
            raise
        stream=r['Body']
        try:raw=stream.read(64000001)
        finally:stream.close()
        require(len(raw)<=64000000,'precalc_cache_object_envelope');return raw

    def put(self,key,raw):
        import json
        require(len(raw)<=64000000,'precalc_cache_object_envelope');from ..contracts import canonical
        canonical(json.loads(raw))
        checksum=hashlib.sha256(raw).digest()
        try:self.client.put_object(Bucket=self.bucket,Key=self.prefix+'/'+locator(key),Body=raw,IfNoneMatch='*',
                ServerSideEncryption='AES256',ChecksumSHA256=base64.b64encode(checksum).decode(),
                ContentType='application/json',Metadata={'sha256':checksum.hex(),'privacy':'private-measurement'})
        except Exception as error:
            require(getattr(error,'response',{}).get('Error',{}).get('Code')in('PreconditionFailed','412'),
                    'precalc_publication_failed')
        require(self.get(key)==raw,'precalc_immutable_collision_or_readback')
        return checksum.hex()


def _fetch_source(client,bucket,cell,root):
    """Only the plan's exact official TRAIN object is addressable through this helper."""
    s=cell['scope'];require(s['source_role']=='official_train'and s['source_key']=='blobs/sha256/'+s['train_sha256'],
                            'precalc_non_TRAIN_object')
    root=Path(root);root.mkdir(parents=True,exist_ok=True,mode=0o700);path=root/(s['train_sha256']+'.csv')
    if path.exists():
        require(not path.is_symlink()and path.stat().st_size==s['train_bytes']
                and hashlib.sha256(path.read_bytes()).hexdigest()==s['train_sha256'],'precalc_source_cache_corrupt');return path
    response=client.get_object(Bucket=bucket,Key=s['source_key']);stream=response['Body'];partial=path.with_suffix('.partial')
    import os
    try:
        require(response['ContentLength']==s['train_bytes']<=200000000,'precalc_source_transport_size')
        checksum=hashlib.sha256();size=0
        with partial.open('xb')as out:
            os.chmod(partial,0o600)
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                size+=len(chunk);require(size<=s['train_bytes'],'precalc_source_overflow');checksum.update(chunk);out.write(chunk)
            out.flush();os.fsync(out.fileno())
        require(size==s['train_bytes']and checksum.hexdigest()==s['train_sha256'],'precalc_source_transport_digest')
        os.link(partial,path);partial.unlink();return path
    finally:stream.close()


def train_source(client,bucket,cell,root):
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with(root/(cell['scope']['train_sha256']+'.lock')).open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _fetch_source(client,bucket,cell,root)


class LocalStore:
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
    def get(self,key):
        p=self.root/locator(key)
        if not p.exists():return None
        require(not p.is_symlink()and p.stat().st_size<=64000000,'precalc_local_object_envelope');return p.read_bytes()
    def put(self,key,raw):
        import os,uuid,json
        from ..contracts import canonical
        canonical(json.loads(raw));p=self.root/locator(key);p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        tmp=p.with_name('.'+uuid.uuid4().hex+'.partial')
        try:
            with tmp.open('xb')as f:os.chmod(tmp,0o600);f.write(raw);f.flush();os.fsync(f.fileno())
            try:os.link(tmp,p)
            except FileExistsError:require(p.read_bytes()==raw,'precalc_immutable_collision_or_readback')
        finally:tmp.unlink(missing_ok=True)
        return hashlib.sha256(raw).hexdigest()


class TrainURLSource:
    """Short-lived, exact-object TRAIN delegation; URLs stay private and are never published."""
    def __init__(self,urls,bucket,plan):
        self.urls,self.bucket=urls,bucket
        self.allowed={c['scope']['source_key']for c in plan['cells']if c['state']=='pending'}
        require(set(urls)<=self.allowed,'precalc_delegation_non_TRAIN_object')
    def get_object(self,*,Bucket,Key):
        from urllib.request import urlopen
        from urllib.parse import urlsplit
        require(Bucket==self.bucket and Key in self.allowed and Key in self.urls,'precalc_delegation_scope')
        parsed=urlsplit(self.urls[Key]);require(parsed.scheme=='https'and parsed.hostname is not None and parsed.hostname.endswith('.amazonaws.com')
            and parsed.path.endswith('/'+Key),'precalc_delegation_url')
        stream=urlopen(self.urls[Key],timeout=60)
        return dict(Body=stream,ContentLength=int(stream.headers['Content-Length']))


def materialize(store,key,expected_sha,output):
    """Retrieve selected measurements for a consumer; never discover/read raw source data."""
    import json,os
    from ..contracts import canonical
    from .worker import lookup
    raw=store.get(key);require(raw is not None and hashlib.sha256(raw).hexdigest()==expected_sha,'precalc_selected_index_digest')
    index=json.loads(raw);manifest_raw=store.get('objects/sha256/'+index['members']['manifest.json']+'.json')
    require(manifest_raw is not None and hashlib.sha256(manifest_raw).hexdigest()==index['members']['manifest.json'],'precalc_selected_manifest_digest')
    manifest=json.loads(manifest_raw);plan=dict(options=manifest['options'],sha256=index['plan_sha256'])
    checked=lookup(index['cell'],plan,index['producer'],store);require(checked is not None,'precalc_selected_entry_missing')
    out=Path(output);out.mkdir(mode=0o700)
    for name,checksum in index['members'].items():
        body=store.get('objects/sha256/'+checksum+'.json');require(hashlib.sha256(body).hexdigest()==checksum,'precalc_selected_member_digest')
        with(out/name).open('xb')as stream:os.chmod(out/name,0o600);stream.write(body)
    return dict(physical_table_id=manifest['scope']['physical_table_id'],directory=str(out.resolve()),
                manifest_sha256=index['members']['manifest.json'],engine_sha256=index['producer']['engine_sha256'])
