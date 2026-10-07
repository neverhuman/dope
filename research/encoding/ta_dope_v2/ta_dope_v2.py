"""TA_Dope_V2: a learned, explicitly target-aware successor to the Dope dataset sketch encoder.

Dope v1 (dope-kernel embed-regression-corpus, router attention-small): a fixed 856-d hand-built sketch of the train table
(target moments; per-feature 12 stats incl. |corr(x, y)|; canonical top-64 features; pairwise |corr|) pushed through a
frozen 3-layer quantised router -> 4168-d vector (sketch + 24 candidate hidden states/actions). Its only label signal is
marginal (target moments and per-feature |Pearson|); it never models HOW y depends on X, and nothing is trained on it.

TA_Dope_V2 changes:
  1. Per-feature tokens (permutation-invariant over features, no canonical sort): Dope v1's 12 per-feature stats computed on
     the support rows, PLUS 9 target-aware stats (signed Pearson, Spearman, correlation ratio eta^2 over 8 x-quantile bins,
     corr(x^2, y), univariate R^2, in-context multivariate ridge coefficient and |coef|, ridge drop-one gain proxy, rank-y
     slope), PLUS a LEARNED joint (x, y) sketch: DeepSets phi([x, y, x*y, |x|]) mean-pooled over support rows (64-d).
  2. Global token: Dope v1 target block (moments/quantiles/rare share, log n, log p) + X->y learnability statistics: in-context
     ridge LOO R^2, 5-NN regression LOO R^2 (nonlinear smoothness), residual skew/kurtosis, distribution of |corr| features.
  3. 4-layer set transformer over [CLS, global, feature tokens] -> label-conditioned pooling (CLS + 4 PMA seeds) -> 1024-d.
  4. Row-level head: each query row's cells e_qj = MLP([x_qj, x_qj^2, token_j]) pooled (mean+max over features) -> 192-d row
     embedding, combined with the context embedding to predict the masked y.
Objective: masked-y regression of query rows (their y never enters any statistic) + 0.1 * two-view InfoNCE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mstats(v, m, n):
    mu = (v * m).sum(1) / n
    c = (v - mu[:, None]) * m
    var = (c ** 2).sum(1) / n
    sd = torch.sqrt(var + 1e-8)
    return mu, var, sd, c


def featurize(X, y, sup, fvalid, miss):
    """X [B,N,P] raw (train-only min-max scaled JTT1 values), y [B,N] standardised, sup [B,N], fvalid [B,P], miss [B,P].
    Returns per-feature stats [B,P,21], global stats [B,G], standardised X [B,N,P]. Uses support rows only."""
    B, N, P = X.shape
    m = sup.float()[..., None]                                   # [B,N,1]
    n = m.sum(1).clamp(min=2)                                    # [B,1]
    mu = (X * m).sum(1) / n; c = (X - mu[:, None]) * m; var = (c ** 2).sum(1) / n; sd = torch.sqrt(var + 1e-8)
    skew = (c ** 3).sum(1) / n / sd ** 3
    Xs = ((X - mu[:, None]) / sd[:, None]).clamp(-6, 6) * fvalid[:, None].float()
    # quantiles over support rows (non-support rows pushed to +inf then ignored by index)
    big = torch.where(sup[..., None], X, torch.full_like(X, float("inf")))
    srt = big.sort(1).values
    ns = sup.sum(1)                                              # [B]
    def q(p):
        i = ((ns - 1).clamp(min=0).float() * p).long()
        return srt.gather(1, i[:, None, None].expand(B, 1, P)).squeeze(1)
    q10, q25, q50, q75, q90 = q(.10), q(.25), q(.50), q(.75), q(.90)
    srt_f = torch.where(torch.isinf(srt), torch.zeros_like(srt), srt)
    diff = (srt_f[:, 1:] != srt_f[:, :-1]).float() * (torch.arange(1, N, device=X.device)[None, :, None] < ns[:, None, None]).float()
    uniq = (diff.sum(1) + 1) / n
    tail = (q90 - q10) / (q75 - q25).abs().clamp(min=1e-6)
    ym = y * sup; yc = (y - (ym.sum(1) / n[:, 0])[:, None]) * sup
    ysd = torch.sqrt((yc ** 2).sum(1) / n[:, 0] + 1e-8)
    pear = (c * yc[..., None]).sum(1) / n / (sd * ysd[:, None])
    # Spearman via ranks within support
    def ranks(v):  # v [B,N,(P)] with non-support at +inf
        return v.argsort(1).argsort(1).float()
    rx = ranks(big); rx = torch.where(sup[..., None], rx, torch.zeros_like(rx))
    yb = torch.where(sup, y, torch.full_like(y, float("inf")))
    ry = ranks(yb); ry = torch.where(sup, ry, torch.zeros_like(ry))
    _, _, rxs, rxc = _mstats(rx, m, n); _, _, rys, ryc = _mstats(ry[..., None], m, n)
    spear = (rxc * ryc).sum(1) / n / (rxs * rys)
    # correlation ratio eta^2 over 8 quantile bins of x (rank-based bins)
    nb = 8
    bins = (rx / ns.clamp(min=1)[:, None, None].float() * nb).long().clamp(max=nb - 1)
    oh = F.one_hot(bins, nb).float() * m[..., None]              # [B,N,P,nb]
    cnt = oh.sum(1).clamp(min=1)
    bmean = (oh * y[:, :, None, None]).sum(1) / cnt              # [B,P,nb]
    ybar = (ym.sum(1) / n[:, 0])
    eta2 = ((cnt * (bmean - ybar[:, None, None]) ** 2).sum(-1)) / ((yc ** 2).sum(1)[:, None] + 1e-8)
    x2 = Xs ** 2; _, _, x2s, x2c = _mstats(x2, m, n)
    c2 = (x2c * yc[..., None]).sum(1) / n / (x2s * ysd[:, None])
    # in-context ridge (primal, P<=256) on standardised support rows
    Xm = Xs * m
    G = Xm.transpose(1, 2) @ Xm / n[:, :, None] + 0.1 * torch.eye(P, device=X.device)[None]
    rhs = (Xm.transpose(1, 2) @ yc[..., None]).squeeze(-1) / n
    beta = torch.linalg.solve(G.float(), rhs.float()).to(X.dtype) * fvalid.float()
    fit = (Xs @ beta[..., None]).squeeze(-1)
    res = (yc - fit) * sup
    # LOO R^2 via hat diagonal h_i = x_i^T G^-1 x_i / n
    Ginv = torch.linalg.inv(G.float()).to(X.dtype)
    hdiag = ((Xs @ Ginv) * Xs).sum(-1) / n
    loo = res / (1 - hdiag).clamp(min=0.05)
    ridge_loo = 1 - ((loo * sup) ** 2).sum(1) / ((yc ** 2).sum(1) + 1e-8)
    gain = beta ** 2 / torch.diagonal(Ginv, dim1=1, dim2=2).clamp(min=1e-6)   # drop-one SSE increase proxy
    rslope = spear * ysd[:, None]
    feat = torch.stack([miss, mu, var, skew.clamp(-20, 20), q10, q25, q50, q75, q90, uniq, tail.clamp(0, 50), pear.abs(),
                        pear, spear, eta2, c2, pear ** 2, beta.clamp(-10, 10), beta.abs().clamp(0, 10), gain.clamp(0, 10), rslope.clamp(-10, 10)], -1)
    feat = torch.nan_to_num(feat) * fvalid[..., None].float()
    # 5-NN LOO regression R^2 on standardised support rows (nonlinear learnability)
    D = torch.cdist(Xm.float(), Xm.float()); BIG = 1e9
    D = D + torch.eye(N, device=X.device)[None] * BIG + (~sup)[:, None, :].float() * BIG
    k = 5; idx = D.topk(k, -1, largest=False).indices
    knn = torch.gather(yc[:, None, :].expand(B, N, N), 2, idx).mean(-1)
    knn_r2 = 1 - (((yc - knn) * sup) ** 2).sum(1) / ((yc ** 2).sum(1) + 1e-8)
    rsd = torch.sqrt((res ** 2).sum(1) / n[:, 0] + 1e-8)
    rsk = ((res / rsd[:, None]) ** 3 * sup).sum(1) / n[:, 0]; rku = ((res / rsd[:, None]) ** 4 * sup).sum(1) / n[:, 0]
    fv = fvalid.float(); pcount = fv.sum(1).clamp(min=1)
    ap = (pear.abs() * fv)
    apm = ap.sum(1) / pcount; apx = ap.max(1).values; e2m = (eta2 * fv).sum(1) / pcount; e2x = (eta2 * fv).max(1).values
    glob = torch.stack([torch.log1p(n[:, 0]), torch.log1p(pcount), ridge_loo.clamp(-1, 1), knn_r2.clamp(-1, 1), rsk.clamp(-20, 20),
                        rku.clamp(0, 100).log1p(), apm, apx, e2m, e2x, (e2x - apx ** 2)], -1)
    return feat, torch.nan_to_num(glob), Xs, yc / ysd[:, None]


class TADopeV2(nn.Module):
    NF, NG, YG = 21, 11, 8

    def __init__(self, d=192, layers=4, heads=4, seeds=4, out_dim=1024):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64))
        self.ftok = nn.Sequential(nn.Linear(self.NF + 64, d), nn.GELU(), nn.Linear(d, d))
        self.gtok = nn.Sequential(nn.Linear(self.NG + self.YG, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False); self.ln = nn.LayerNorm(d)
        self.seeds = nn.Parameter(torch.randn(seeds, d) * 0.02); self.pma = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear((seeds + 1) * d, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.cell = nn.Sequential(nn.Linear(2 + 64, 96), nn.GELU(), nn.Linear(96, 96)); self.tok64 = nn.Linear(d, 64)
        self.rowh = nn.Sequential(nn.Linear(192 + 128, 192), nn.GELU(), nn.Linear(192, 192))
        self.ctx128 = nn.Linear(out_dim, 128)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(192, 1))
        self.cproj = nn.Sequential(nn.Linear(out_dim, 256), nn.GELU(), nn.Linear(256, 128))
        self.out_dim = out_dim

    def forward(self, X, y, sup, valid, fvalid, miss, yraw_stats):
        """X [B,N,P]; y [B,N] standardised on support; sup/valid [B,N]; fvalid [B,P]; miss [B,P]; yraw_stats [B,8] (Dope v1
        target block on support rows). Returns yhat [B,N], ctx [B,out], rows [B,N,192]."""
        sup = sup & valid
        with torch.autocast("cuda", enabled=False):
            feat, glob, Xs, ys = featurize(X.float(), y.float(), sup, fvalid, miss.float())
        m = sup.float()[..., None, None]
        pin = torch.stack([Xs, ys[..., None].expand_as(Xs), Xs * ys[..., None], Xs.abs()], -1)   # [B,N,P,4]
        ph = (self.phi(pin) * m).sum(1) / m.sum(1).clamp(min=1)                                  # [B,P,64]
        ft = self.ftok(torch.cat([feat, ph], -1))
        gt = self.gtok(torch.cat([glob, yraw_stats], -1))[:, None]
        tok = torch.cat([self.cls.expand(X.shape[0], -1, -1), gt, ft], 1)
        pad = torch.cat([torch.zeros(X.shape[0], 2, dtype=torch.bool, device=X.device), ~fvalid], 1)
        h = self.ln(self.enc(tok, src_key_padding_mask=pad))
        pooled, _ = self.pma(self.seeds[None].expand(X.shape[0], -1, -1), h, h, key_padding_mask=pad)
        ctx = self.proj(torch.cat([h[:, 0], pooled.flatten(1)], 1))
        t64 = self.tok64(h[:, 2:])                                                               # [B,P,64]
        cin = torch.cat([Xs[..., None], (Xs ** 2)[..., None], t64[:, None].expand(-1, X.shape[1], -1, -1)], -1)
        e = self.cell(cin) * fvalid[:, None, :, None].float()
        emean = e.sum(2) / fvalid.float().sum(1).clamp(min=1)[:, None, None]
        emax = e.masked_fill(~fvalid[:, None, :, None], -1e4).max(2).values
        rows = self.rowh(torch.cat([emean, emax, self.ctx128(ctx)[:, None].expand(-1, X.shape[1], -1)], -1))
        return self.head(rows).squeeze(-1), ctx, rows
