"""Bounded Gaussian-copula and compressed-row controls for public Bikeshare smoke."""
import gzip
import json

SEED = 1729


def gaussian_copula_artifact(rows, byte_cap, seed=SEED):
    """Fit a deterministic development-only Gaussian copula under a stored-byte cap."""
    import numpy as np
    from scipy.special import ndtri
    from scipy.stats import rankdata
    values=np.asarray(rows,dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("copula_fit_rows_invalid")
    normal=np.column_stack([ndtri((rankdata(values[:,j],method="average")-.5)/len(values)) for j in range(values.shape[1])])
    corr=np.corrcoef(normal,rowvar=False)
    corr=np.nan_to_num(corr,nan=0.0,posinf=0.0,neginf=0.0)
    np.fill_diagonal(corr,1.0)
    corr=(corr+corr.T)/2
    eig,vec=np.linalg.eigh(corr)
    corr=vec @ np.diag(np.maximum(eig,1e-8)) @ vec.T
    scale=np.sqrt(np.diag(corr)); corr=corr/np.outer(scale,scale)
    probs=np.linspace(0.0,1.0,256)
    marginals=[np.quantile(values[:,j],probs).tolist() for j in range(values.shape[1])]
    document={"format":"dope-bikeshare-gaussian-copula-v1","seed":seed,"fit_rows":len(values),
              "correlation":corr.tolist(),"marginal_quantiles":marginals,"quantile_probabilities":probs.tolist()}
    packed=gzip.compress(json.dumps(document,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),mtime=0)
    if len(packed)>byte_cap:
        raise ValueError("copula_artifact_exceeds_cap")
    return packed


def gaussian_copula_sample(artifact, count, seed=SEED+1):
    import numpy as np
    from scipy.special import ndtr
    document=json.loads(gzip.decompress(artifact))
    if document.get("format")!="dope-bikeshare-gaussian-copula-v1":
        raise ValueError("copula_artifact_invalid")
    rng=np.random.default_rng(seed)
    latent=rng.multivariate_normal(np.zeros(len(document["correlation"])),document["correlation"],size=count,check_valid="raise")
    uniforms=ndtr(latent)
    probs=np.asarray(document["quantile_probabilities"],dtype=np.float64)
    marginals=np.asarray(document["marginal_quantiles"],dtype=np.float64)
    return np.column_stack([np.interp(uniforms[:,j],probs,marginals[j]) for j in range(marginals.shape[0])])


def compressed_row_artifact(rows, byte_cap, seed=SEED):
    """Store a deterministic fit-only row subsample in a bounded gzip JSON artifact."""
    import numpy as np
    values=np.asarray(rows,dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("compressed_rows_invalid")
    order=np.random.default_rng(seed).permutation(len(values))
    def pack(k):
        body={"format":"dope-bikeshare-compressed-row-subsample-v1","seed":seed,
              "source_rows":len(values),"rows":values[order[:k]].tolist()}
        return gzip.compress(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),mtime=0)
    low,high,best=1,len(values),None
    while low<=high:
        mid=(low+high)//2; candidate=pack(mid)
        if len(candidate)<=byte_cap:
            best=candidate; low=mid+1
        else:
            high=mid-1
    if best is None:
        raise ValueError("compressed_row_artifact_exceeds_cap")
    return best


def compressed_row_sample(artifact, count, seed=SEED+1):
    import numpy as np
    body=json.loads(gzip.decompress(artifact))
    if body.get("format")!="dope-bikeshare-compressed-row-subsample-v1":
        raise ValueError("compressed_row_artifact_invalid")
    rows=np.asarray(body["rows"],dtype=np.float64)
    return rows[np.random.default_rng(seed).integers(0,len(rows),size=count)]

