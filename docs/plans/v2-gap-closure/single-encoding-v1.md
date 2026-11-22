# `dope.single-encoding/v1` — stat packet v0 (kind 0)

Status: stage SP1 of plan milestone M2. Implemented in
`engine/dope-arena/src/statpacket/` and the binary `dope_statpacket`.
This stage fixes the container, the scale card, the marginal and dependence
sections and the feature sampler. The target sections (S3, S8, S10 and the two
noise bytes of S1) have their byte layout fixed here and are written as zeros;
the target stage defines how they are fitted and decoded.

The packet is a zero-raw-row encoding of one table: explicit, quantised
sufficient statistics of the FIT rows, no training, no shared decoder weights.
Synthetic rows are sampled from the packet bytes alone.

Source of this text: the `encoding-ladder` design of the 2026-11-22 gap study.
Where this document and that design differ, section 11 lists the difference
and the reason; this document is the one to implement.

## 1. Conventions

- All multi-byte integers are little-endian.
- A *b-bit code list* packs `n` unsigned codes of `b` bits each, least
  significant bit first: bit `i` of code `j` is bit `(b*j + i) mod 8` of byte
  `(b*j + i) div 8`. The list takes `ceil(b*n/8)` bytes and unused high bits of
  the last byte are zero. This is the bit order of `dope_core::quant::pack`.
- A *column bitmap* has `B = ceil(P/8)` bytes; column `j` is bit `j mod 8` of
  byte `j div 8`; unused bits are zero.
- `round` is round-half-away-from-zero (`f64::round`). binary16 conversion
  rounds to nearest, ties to even, and saturates at the largest finite value.
- `Phi` is the standard normal CDF as computed by `dope_core::stats::norm_cdf`
  (absolute error below 1.2e-7), `ppf` its inverse `norm_ppf` (AS241), `pdf`
  the density. All arithmetic is IEEE float64 in the order written.
- A reader rejects every non-canonical encoding (reserved bits, padding bits,
  orderings, counts), so `pack(unpack(bytes)) == bytes` for every accepted file.
- Error messages are fixed text plus a section number. They never contain a
  data value, a header, a file name or an extremum.

Dimensions used below:

| Symbol | Meaning |
|---|---|
| `P` | feature columns, 1..65535 |
| `P_c`, `P_d`, `P_k` | continuous, discrete, constant columns (`P_c + P_d + P_k = P`) |
| `A_j` | atoms of discrete column `j`, 2..16 |
| `K` | header class count: 0 regression, 2 binary, classes (3..255) for multiclass, labels (1..255) for multilabel |
| `O` | target outputs: 1 regression and binary, `K-1` multiclass, `K` multilabel |
| `M` | regimes with factor means: `K` binary and multiclass, 0 otherwise |
| `Q` | prior bytes: 0 regression, `K-1` binary and multiclass, `K` multilabel |
| `B` | `ceil(P/8)` |
| `I` | pair index bytes: 1 if `P <= 256`, else 2 |

## 2. Unit space and the scale card

The packet lives in *unit space*. The owner-side scale card maps raw table
units to unit space and back. **The card is never part of the packet** and its
bytes are reported separately when rows are returned in raw units.

Per column, over the finite FIT cells: `median` (mean of the two middle values
for an even count), `scale = max(IQR/1.349, population sd, 1e-6)` with the
quartiles by linear interpolation at `q*(n-1)`. Then

```
to_unit(x): z = (x - median)/scale
            u = z                               if |z| <= 3
            u = sign(z) * (3 + asinh(|z| - 3))  otherwise
            return clip(u, -6, 6);  a non-finite x returns 0
to_raw(u):  u = clip(u, -6, 6)
            z = u                               if |u| <= 3
            z = sign(u) * (3 + sinh(|u| - 3))   otherwise
            x = median + scale*z;  if grid > 0: x = round(x/grid)*grid
```

This is `_fit_only_nc2_scale` of `research/levels/s3_dynamic_augment.py`
(parity is tested against numpy output; the float32 casts agree bit for bit on
the reference table). Additions on the owner side only:

- `constant`: every finite FIT cell is equal. `to_raw` returns `median`
  whatever the unit value is, so a FIT-constant column decodes to its constant.
  `to_unit` keeps the Python behaviour (`scale = 1e-6`).
- A column with fewer than two finite FIT cells has `scale = 0`, `to_unit`
  returns 0 everywhere (as in Python) and `to_raw` returns the single finite
  value, or 0 if there is none.
- `grid`: the coarsest step of `1000, 100, 10, 1, 0.5, 0.25, 0.1, 0.05, 0.01,
  0.001` such that every finite FIT cell is a multiple of it (to within half a
  float32 unit in the last place) and the step is at least 8 float32 ulps of
  the largest magnitude; 0 if none. `to_raw` snaps to it, so integer, decimal
  and atom columns come back on their exact raw levels.
- Target: regression `mean` and `sd = max(population sd, 1e-5)`,
  `unit = clip((y - mean)/sd, -6, 6)`; or the class map `labels[c]`.

JSON form (`dope.scale-card/v1`): `{schema, columns: [{median, scale,
constant, grid}], target: {kind: none | regression | classes, ...}}`. Every
float is a string holding the shortest decimal that parses back to the same
bits, because the JSON number parser in use is not exact.

Consequence for the fit: a column's FIT median is at unit 0. The continuous
marginal relies on this (section 5.2).

## 3. Container

```
header (24 bytes) | S1 | S2 | ... | S_n | CRC-32 (4 bytes)
```

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 4 | magic | `DSE1` |
| 4 | 1 | version | 1 |
| 5 | 1 | kind | 0 stat (1 neural and 2 hybrid are reserved; this reader rejects them) |
| 6 | 1 | task | 0 regression, 1 binary, 2 multiclass, 3 multilabel |
| 7 | 1 | K | see section 1 |
| 8 | 2 | P | u16 |
| 10 | 1 | level | 0..4 = A..E |
| 11 | 1 | bit profile | 0 (the widths of this document) |
| 12 | 1 | section count | 3, 4, 6, 8, 10 for levels A..E |
| 13 | 1 | `e = floor(log2 n_fit)` | 4..62 |
| 14 | 1 | flags | bit 0: S3 is sparse; other bits 0 |
| 15 | 1 | reserved | 0 |
| 16 | 8 | decoder sha8 | first 8 bytes of SHA-256 of the recipe text |

The recipe text is the constant `RECIPE` in `codec.rs`; for this stage
`decoder sha8 = 95 42 a2 4b 5c e2 6f 91`. Any change to a decode rule changes
the text, so earlier readers refuse the packets. The trailer is CRC-32 (IEEE,
reflected, polynomial 0xEDB88320, initial and final XOR 0xFFFFFFFF) of all
preceding bytes.

There is no directory. Section lengths follow from the header and from the S2
flag plane. `unpack` checks, in this order: length at least 28, magic, version,
kind, header fields, decoder sha8, every section inside the file, the CRC at
the computed end, no bytes after it, then the structural invariants.

`truncate(bytes, level)` keeps the header and the first sections of the lower
level, rewrites bytes 10 and 12 and appends a new CRC. **Prefix property**:
`truncate(pack(p, E), k) == pack(p, k)` for every level `k`, so one encode per
table yields every size level.

## 4. Levels and bytes

| Level | Sections | Knots | Rank |
|---|---|---|---|
| A | S1-S3 | 5 | 0 (independent columns) |
| B | + S4 | 5 | 4 |
| C | + S5, S6 | 9 | 8 |
| D | + S7, S8 | 17 | 8 |
| E | + S9, S10 | 17 | 12 |

| Section | Content | Bytes |
|---|---|---|
| S1 | priors, noise | `Q + 2` |
| S2 | coarse marginals, atoms | `P + 5*P_c + sum_j atom_j` |
| S3 | linear target | `O*m3 + 4*O` (dense: `m3 = P`; sparse: `+ B`) |
| S4, S6, S9 | 4 loadings per column, 4 factor means per regime | `4*P + 4*M` each |
| S5 | knots 5 to 9 | `3*P_c` |
| S7 | knots 9 to 17 | `6*P_c` |
| S8 | 4-bin main effects | `B + 4*O*m8` (dense: `m8 = P`) |
| S10 | 16 pairs | `16*(2*I + O)` |

`atom_j = 2*A_j + ceil(6*A_j/8)` for explicit values and
`4 + A_j + ceil(6*A_j/8)` for a lattice. File length = 28 + the sum of the
sections present. A discrete column costs its flag byte plus `atom_j` and
skips S5 and S7; a constant column costs its flag byte only. Examples: a
binary column 7 B in all (6 B payload), 4 explicit atoms 11 B payload,
16 lattice atoms 32 B, 16 explicit atoms 44 B; a continuous column 6 B at
level A and 15 B from level D.

All-continuous tables with dense sections (measured by test, equal to the
design byte table):

| Task | Level | P=8 | P=24 | P=48 | P=64 |
|---|---|---|---|---|---|
| regression | A / B / C / D / E | 90 / 122 / 178 / 259 / 339 | 202 / 298 / 466 / 709 / 853 | 370 / 562 / 898 / 1384 / 1624 | 482 / 738 / 1186 / 1834 / 2138 |
| binary | A / B / C / D / E | 91 / 131 / 195 / 276 / 364 | 203 / 307 / 483 / 726 / 878 | 371 / 571 / 915 / 1401 / 1649 | 483 / 747 / 1203 / 1851 / 2163 |
| multiclass K=5 | A / B / C / D / E | 130 / 182 / 258 / 435 / 583 | 290 / 406 / 594 / 1125 / 1337 | 530 / 742 / 1098 / 2160 / 2468 | 690 / 966 / 1434 / 2850 / 3222 |
| multilabel K=8 | E | 767 | 1841 | 3452 | — |
| multilabel K=32 | E | — | 5225 | 9716 | — |

## 5. Sections

### 5.1 S1 — priors and noise

`Q` prior bytes, then `resid_sd` (u8), then `temperature` (u8).

- Binary and multiclass: codes `c_0 .. c_{K-2}`; `c_{K-1} = 255 - sum`;
  `pi_c = c_c / 255`. The sum of the stored codes is at most 255. Fit:
  largest remainder on `255 * count_c / n`, with every class that has rows
  given at least 1; while the total is short, add 1 to the class with the
  largest `exact - code`; while it is over, take 1 from the class (code above
  1) with the largest `code - exact`; ties go to the lower class index.
- Multilabel: one code per label, `round(255 * rate)`.
- `resid_sd` and `temperature` are 0 in this stage (target stage).

### 5.2 S2 — coarse marginals

A flag plane of `P` bytes, then the payload of every column in column order.

Flag byte: bits 0-1 kind (0 continuous, 1 discrete, 2 constant; 3 invalid),
bits 2-5 `A_j - 1` (discrete only, else 0), bit 6 lattice (discrete only,
else 0), bit 7 zero.

**Constant** (every FIT cell equal): no payload; decodes to unit 0.

**Continuous**, 5 bytes: a 24-bit little-endian word with `lo` in bits 0-11
and `hi` in bits 12-23, then `split_0`, `split_1` (u8 each). With

```
half(c) = 0                          if c = 0
half(c) = 6 * 2^((c - 4095)/128)     otherwise      (relative step 0.54 %)
```

the five coarse knots (grid indices 0, 4, 8, 12, 16 of 17) are

```
k_0 = -half(lo)    k_4 = -half(lo) * (1 - split_0/255)    k_8 = 0
k_12 = half(hi) * split_1/255                              k_16 = half(hi)
```

Knot 8 is pinned to the origin, the FIT median. Each side of the median has
its own log-coded extent, so a skewed column keeps its relative precision on
both sides.

**Discrete**: atom values ascending, then a 6-bit code list of `A_j` masses.

- explicit (lattice bit 0): `A_j` binary16 values;
- lattice (bit 1): `base` binary16, `step` binary16, then `A_j` index bytes,
  strictly ascending; `value_a = base + step * index_a`.

Values are unit-space values, clipped to [-6, 6] on decode. Mass codes are
1..63 and `p_a = code_a / sum(codes)`. A reader rejects a non-finite value, a
lattice step that is not positive and finite, and values or indices that are
not strictly ascending.

### 5.3 S3 — linear target (layout only in this stage)

If header flag bit 0 is set: a column bitmap of the columns that carry
weights. Then `O` pairs (`scale` binary16, `intercept` binary16). Then, for
each output, one int8 weight per carried column in ascending column order.
A dense S3 carries all `P` columns and has no bitmap. Written as zeros here.

### 5.4 S4, S6, S9 — dependence

Rank block `b` (0, 1, 2 for S4, S6, S9) holds factors `4b .. 4b+3`: for each
column the four int8 loadings, then for each of the `M` regimes the four int8
factor means. `l = code/127`, `m = code * 6/127`. A regression or multilabel
packet has `M = 0` and a single regime.

### 5.5 S5, S7 — knot refinement

For each continuous column in column order: S5 a 6-bit code list of 4 codes
(3 bytes), S7 one of 8 codes (6 bytes). A code `t` places a new knot at
`lo + (t/63) * (hi - lo)` inside the interval `[lo, hi]` of its two parents:

- S5 code `i` gives knot `4i + 2` between knots `4i` and `4i + 4`;
- S7 code `i` gives knot `2i + 1` between knots `2i` and `2i + 2`.

The 5, 9 and 17 knot sets are exactly nested: a lower level uses a subset of
the same dequantised values.

### 5.6 S8 — main effects (layout only)

A column bitmap, then for each set column in ascending order `4*O` int8 values
(output-major, 4 bins each). Dense means every bit set. Zeros here.

### 5.7 S10 — pairs (layout only)

Exactly 16 entries: column `a`, column `b` (`I` bytes each, both below `P`),
then `O` int8 coefficients. Zeros here.

## 6. Budget fitting

`fit_to_budget(packet, max_bytes)` is deterministic and never drops a column.
Importance of a column is the sum over outputs of `|S3 weight code|`; ties go
to the lower index (until the target stage fills S3, this is column order).

1. For level E, D, C, B, A in turn: if the dense packet fits, return it.
2. At level D only, before moving on to C: keep S8 for the `m` most important
   columns, `m` the largest count that fits, if `m >= 1`.
3. If dense level A does not fit: set the sparse flag and keep S3 weights for
   the `m` most important columns, `m >= 0` the largest count that fits.
4. If even `m = 0` does not fit: return that smallest packet with
   `fits = false` (`over_budget`). The table counts as over budget; it is not
   cut.

The result reports level, bytes, the rule used, and the S3 and S8 column
counts. At 2,048 B (all-continuous columns): regression and binary reach E up
to P=48, D at P=64, B at P=128, A at P=256; multiclass K=5 reaches E at P=24,
D with S8 on the top 41 columns at P=48 (exactly 2,048 B), B at P=128, sparse
A at P=256; multiclass K=32 at P=256 keeps 9 target columns (2,032 B).
Regression keeps a dense level A up to P=287 (`34 + 7P`), a sparse S3 up to
P=328 (`34 + 6P + B`), and is over budget from P=329.

## 7. Fit

Input: unit-space FIT rows `n x P` (every cell finite, in [-6, 6]),
`n >= 16`, the task and its labels. `e = floor(log2 n)`.

### 7.1 Knot plan (a function of `e` only)

```
k    = min(10, max(3, floor(2^e / 10)))      smoothing count
h    = ceil(k/2)                              window half-width
u_lo = max(0.01, k / 2^e)                     lowest knot probability
z_hi = -ppf(u_lo);   z_i = z_hi * (i - 8)/8;   u_i = Phi(z_i),  i = 0..16
```

| e | 4 | 5 | 6 | 7 | 8 | 9 | >= 10 |
|---|---|---|---|---|---|---|---|
| k / h | 3 / 2 | 3 / 2 | 6 / 3 | 10 / 5 | 10 / 5 | 10 / 5 | 10 / 5 |
| u_lo | 0.1875 | 0.09375 | 0.09375 | 0.078125 | 0.0390625 | 0.01953125 | 0.01 |

The decoder recomputes the plan from the header; it needs no row count.

### 7.2 Column kind

Sort the column. One distinct value: constant. At most 16 distinct values:
the levels with at least 10 rows survive; every other level adds its rows to
the nearest surviving level (the lower one on a tie); with at least two
survivors whose binary16 values are distinct and ascending the column is
discrete. Otherwise it is continuous.

### 7.3 Continuous: smoothed knots and codes

With the column sorted ascending (`s_0 .. s_{n-1}`), for `i = 0..16`:

```
c  = clamp(floor(u_i * (n - 1) + 0.5), 1, n - 2)
lo = max(c - h, 1);   hi = min(c + h, n - 2)
knot_i = max(knot_{i-1}, mean(s_lo .. s_hi))      summed in ascending order
```

A window never contains rank 0 or rank `n-1`. Codes (closed loop: each is
chosen against the dequantised values of its parents):

```
code(v) = 0 if v <= 0, else clamp(round(4095 + 128*log2(v/6)), 1, 4095)
lo = code(-knot_0);  hi = code(knot_16)
split_0 = round(255 * clamp((knot_4 + half(lo)) / half(lo), 0, 1))   (0 if half(lo) = 0)
split_1 = round(255 * clamp(knot_12 / half(hi), 0, 1))               (0 if half(hi) = 0)
S5, S7:  t = round(63 * clamp((knot - lo)/(hi - lo), 0, 1))          (0 if hi <= lo)
```

### 7.4 Discrete: atoms

Values: the surviving unit values. A lattice is used if there are at least 5
atoms and, for the smallest `d` in 1..8, `step = (smallest gap)/d` gives
integer indices `round((v_a - v_0)/step) <= 255`, strictly ascending, that
rebuild every value within 0.004 from the binary16 `base = v_0` and `step`.
Otherwise the values are explicit binary16.

Masses: 6-bit codes by the *simplex rule* on the merged row counts `w`: for
every multiplier `m = 1..63` take `c_a = clamp(round(m * w_a / max(w)), 1, 63)`
and keep the first `m` whose largest error of the cumulative shares
`|sum_{b<=a} c_b / sum(c) - sum_{b<=a} w_b / sum(w)|` is smallest (strictly
smaller by more than 1e-12 to replace an earlier `m`).

### 7.5 Dependence

Scores (latent-normal convention). Each column is read as a monotone function
of a standard normal latent. Sort the column; a block of tied rows at sorted
positions `i..j` covers the probability interval `(a, b) = (i/n, (j+1)/n)`:

```
score = ppf((i + 0.5)/n)                         untied row (the average-rank score)
score = (pdf(ppf(a)) - pdf(ppf(b))) / (b - a)    tied block (conditional mean of the latent)
```

with `pdf(ppf(0)) = pdf(ppf(1)) = 0`. Centre the scores, let `v_j` be their
variance and `v_ref` the variance of the scores of `n` untied rows;
`kappa_j = min(1, sqrt(v_j / v_ref))`; a column with `v_j <= 1e-12` is treated
as constant (`kappa_j = 0`, all scores 0). Standardise to unit variance: `t_j`.

Latent correlation: `rho_jk = clip(mean(t_j * t_k) / (kappa_j * kappa_k), -1, 1)`,
unit diagonal; a constant column has a zero row, column and diagonal. Tied
columns attenuate every correlation they enter by `kappa`; dividing restores
the latent value, so that the sampler, which discretises a unit-variance
latent, reproduces both the level masses and the observed association.

Classes (binary, multiclass): `share_c = n_c/n`;
`mu_cj = mean of t_j over class c / kappa_j` for a class with at least 10
rows, else 0. If `between_j = sum_c share_c * mu_cj^2 > 0.9`, scale the
column's means by `sqrt(0.9/between_j)`. `within_j = 1 - between_j`. Then

```
rho_jk  <-  clip((rho_jk - sum_c share_c*mu_cj*mu_ck) / sqrt(within_j*within_k), -1, 1)
mu_cj   <-  mu_cj / sqrt(within_j)              class means in within-class SD units
```

Loadings. `r = min(12, P)`. If `P <= 12`: eigen-decompose `rho` (principal
components reproduce the matrix). Otherwise principal axes: communalities
start at each column's largest absolute off-diagonal correlation; repeat up to
10 times: put `min(h_j, 0.98)` on the diagonal, eigen-decompose, set `h_j` to
`min(0.98, sum_{f<r} max(lambda_f, 0) * v_jf^2)`, stop when no `h_j` moved by
1e-3 or more; the last decomposition is used. Eigenvalues descend. Factor `f`:
`l_jf = sign_f * sqrt(max(lambda_f, 0)) * v_jf`, where `sign_f` makes the
entry of largest magnitude positive (lowest column on a tie);
`code = clamp(round(127 * l_jf), -127, 127)`. Ranks 4, 8, 12 are prefixes.
The eigen routine is cyclic Jacobi (`dope_core::stats::sym_eigen`); an
implementation with another routine can differ by one code step.

Factor means, one factor at a time against the quantised loadings `L`:
`m_cf = (sum_j L_jf * mu_cj) / (sum_j L_jf^2 + 1e-3)`,
`code = clamp(round(m_cf * 127/6), -127, 127)`. Nested by construction.

The factor fit costs up to 10 eigen-decompositions of a `P x P` matrix. It is
skipped when the caller's budget cannot hold S4 (`fit_features_for_budget`;
the packed bytes are identical) and for `P > 1024`.

## 8. Decode (features)

Row `i` depends only on `(packet, seed, i)`. Draws:
`U(stream) = ((hash3(seed, stream, i) >> 11) + 0.5) / 2^53`, `N(stream) =
ppf(U(stream))`, with `hash3` of `dope_core::rng` and `stream = purpose << 32
| draw`: purpose 1 regime (draw 0), 2 factor (draw = factor), 3 column noise
(draw = column). Below, `U(p, d)` and `N(p, d)` name purpose and draw.
Purposes 4 and up are free for the target stage.

With `r` the rank of the level, `l_j` the first `r` loadings of column `j`:

```
psi_j = max(0.02, 1 - |l_j|^2);   s_j = sqrt(|l_j|^2 + psi_j)
f_k = N(2, k), k < r
regimes (M > 0 and r > 0):  c = first class with cumulative pi > U(1, 0), else the last;  f <- f + m_c
z_j = l_j . f + sqrt(psi_j) * N(3, j)
q_j = z_j / s_j                                            single regime
q_j = ppf(clip(sum_c pi_c * Phi((z_j - l_j . m_c)/s_j)))   regimes, clip to [1e-12, 1 - 1e-12]
x_j = Q_j(q_j)
```

`sum_c pi_c Phi(.)` is the exact CDF of `z_j` under the mixture, so `q_j` is
standard normal for every column: no batch statistic is used anywhere.

`Q_j` by kind:

- constant: 0.
- discrete: thresholds `tau_a = ppf(sum_{b<=a} p_b)` for `a < A_j - 1`; the
  value is atom number `#{a : tau_a <= q}`.
- continuous, with `m` knots `k_0..k_{m-1}` (5, 9 or 17) at scores
  `-z_hi + a*dz`, `dz = 2*z_hi/(m-1)`: `t = (q + z_hi)/dz`;
  inside, linear in `t` between neighbouring knots; outside,

```
below:  k_0     - tail(k_1 - k_0,         k_2 - k_1,         -t)
above:  k_{m-1} + tail(k_{m-1} - k_{m-2}, k_{m-2} - k_{m-3}, t - (m-1))
tail(d0, d1, t) = 0                             if d0 <= 0
                = d0 * t                        if d1 <= d0   (even or widening end)
                = d0 * r*(1 - r^t)/(1 - r),  r = d0/d1        (narrowing end)
```

  and the result is clipped to [-6, 6]. A narrowing end is how a bounded
  column looks in this coordinate; its extension is a geometric series that
  never passes `d0*r/(1 - r)`.

Output: unit values as float32, or raw values `to_raw(x_j)` as float32. Rows
`[0, n)` equal rows `[0, a)` followed by `[a, n)` bit for bit.

## 9. What the packet can and cannot contain

- No string, name or header.
- No column minimum or maximum: a knot is a mean over a window of `2h + 1`
  order statistics (11 from 128 rows, 7 from 64, 5 below) that never includes
  the smallest or largest row; the lowest and highest knots sit at the clipped
  probabilities `u_lo` and `1 - u_lo`. The packet bytes are identical when
  the extreme cells of every column are pushed to the clip (tested).
- No level with fewer than 10 rows: such a level is merged before the atom
  table is built and can never be sampled.
- No class factor mean from fewer than 10 rows.
- Removing one row moves a smoothed knot by at most the largest gap between
  neighbouring inner order statistics (tested for every row of a fixture).
- The header reveals `floor(log2 n_fit)`, `P`, `K` and the task; S1 reveals
  class shares to 1/255.

This is privacy by construction of the stored statistics. It is not a formal
differential-privacy or de-identification claim.

## 10. Command line

```
dope_statpacket fit      --input A.mat --task regression|binary|multiclass|multilabel
                         [--classes K] --packet P.dse --card card.json
                         [--max-bytes 2048] [--level A..E]
dope_statpacket info     --packet P.dse
dope_statpacket truncate --packet P.dse --level A..E --output Q.dse
dope_statpacket decode   --packet P.dse [--card card.json] --rows N [--row0 R] [--seed S] --output S.mat
```

Tables are `DOPEMAT1` files (formats.md section 5) in raw units, features
first, target last. `decode` writes features only (`y_cols = 0`) in this
stage. Each command prints one JSON object (level, bytes, ledger, sha256);
failures print one fixed line on stderr and exit with code 2.

Two packets pinned by test (`golden_packets_match_the_specification`):

- 16 rows, one constant feature, regression, level A, 36 bytes:
  `44534531 01 00 00 00 0100 00 00 03 04 00 00 9542a24b5ce26f91 | 0000 | 02 |
  0000000000 | f46dd66e` (header, S1, S2, S3, CRC).
- 32 rows, features `i` and `i mod 2` for `i = 0..31`, regression, level E,
  139 bytes, SHA-256 `23cfd3aba4f988e3dd696507c72a3ebcc1b02d7f1ad4121bbb73db2fb1f9c11d`.

## 11. Decisions and differences from the design

| # | Design text | This specification | Why |
|---|---|---|---|
| D1 | Knots at equal probability steps, interpolation linear in `u` | Knots equally spaced in normal score, interpolation linear in that score | Same bytes. Linear-in-`u` chords miss a Gaussian end segment by 0.16 unit and an exponential by about 1.0 at 5 knots; the score coordinate is exact for a Gaussian and gives 0.15 for the exponential at 5 knots |
| D2 | `u_lo = max(0.01, k/n)`, `k` from `n` | Both from `e = floor(log2 n)` | The decoder has no `n`; rounding `n` down only clips more |
| D3 | Location int8 over [-6, 6], log-span u8, 4 increments at 6 bits | Median-pinned: two 12-bit log half-spans and two 8-bit splits (same 5 bytes) | The unit scale is `max(IQR/1.349, sd)`; a heavy tail squeezes half of the mass into a few percent of the span. On a skewed fixture column the design layout gave KS 0.143 with 14 % of synthetic cells below the FIT minimum; this layout gives a worst column of 0.046 and under 1 % |
| D4 | Tails: linear extension in `u`, clipped | Extension in the score coordinate: linear for an even or widening end, geometric for a narrowing end | A bounded linear tail leaves every real cell beyond it identifiable (about 1.4 % of cells per Gaussian column); a narrowing end marks a bounded column and stays inside it |
| D5 | Atom payload outline | Exact bytes of section 5.2; at most 16 distinct values in the column; masses by the simplex rule with codes 1..63 | Completes the design; byte counts are in the ledger |
| D6 | Normal scores of average ranks | Conditional-mean scores for tied blocks and division by `kappa` | Average-rank scores of a binary column lose 20 % of a correlation once and again on decode; with the correction a mixed continuous / binary / ordinal table keeps its correlations within 0.034 |
| D7 | Principal-axis factor (design); principal components (stage brief) | Principal components for `P <= 12`, principal axes with iterated communalities above, eigen-ordered so that ranks are prefixes | Components of the full matrix spend factors 7..12 on unique variance. On a 48-column, 6-factor table the Spearman gap fell from mean 0.021 / max 0.146 to 0.012 / 0.060 |
| D8 | `m_c = (L'L + 1e-3 I)^-1 L' mu_c` | One factor at a time against the quantised loadings, class means in within-class SD units, zero for classes under 10 rows, within share at least 0.1 | Keeps ranks nested and the support floor for classes |
| D9 | Header `reserved u16` | `flags u8` (bit 0 sparse S3) and `reserved u8` | Needed by the many-class fallback; the prefix property holds because the flag belongs to the packet, not to the level |
| D10 | Pair entry `(2 + O)` bytes | `2*I + O` with 2-byte indices above 256 columns | `P` is a u16 |
| D11 | Multilabel "as multiclass with O = L" | `K = L`, `Q = L` label rates, `M = 0`, byte rows in section 4 | No budget row existed |
| D12 | Scale card `+4P` bytes | JSON card with constant flag and decimal grid; measured and reported separately | Exact raw levels for integer and atom columns |
| D13 | Lower level, then top-m target weights | Section 6: level D with partial S8 is tried before level C | It is the design's own answer for K=5, P=48, and reproduces its "top 41 columns" |
| D14 | — | Readers reject non-canonical encodings | One byte string per packet |
| D15 | — | `decode` writes `y_cols = 0` until the target stage | Features only |

## 12. Measured in this stage

All from `cargo test -p dope-arena --bin dope_statpacket -- --nocapture`
(39 unit tests) and `cargo test -p dope-arena --test statpacket_cli` (2
black-box tests) on deterministic fixtures; unit space unless stated.

| Check | Result |
|---|---|
| Byte table, formulas, prefix property, corruption, budget | exact (table of section 4) |
| Scale card vs numpy reference | identical float64 and float32 on 52 cells |
| Scale card round trip, 11,998 cells with \|unit\| < 6 (89 in the asinh tail), through a float32 unit value | worst relative error 1.4e-7 |
| Gaussian AR(1) table, P=12, n=2000, 50,000 decoded rows: max correlation gap vs FIT rows | rank 12: 0.019; rank 8: 0.080; rank 4: 0.145 |
| 3 factors in 30 columns, n=3000 | rank 12: 0.019; rank 8: 0.027; rank 4: 0.031 |
| Continuous / binary / ordinal table, n=4000: max correlation gap (raw units) | 0.034 |
| Binary class regimes, P=6, n=2000 | correlation gap 0.033, quantile gap 0.045 |
| Quantile gap, 5 column shapes, n=2000, 100,000 decoded rows, probabilities 0.02..0.98 | 5 knots 0.360; 9 knots 0.112; 17 knots 0.057 |
| Two separated modes (known weak case), 17 knots | 0.498 |
| Level frequencies of discrete columns, 50,000 decoded rows | 2.43 SE vs packet masses, 0.55 SE vs FIT rows; mass quantisation error 0.0008 |
| Canaries: unique extreme (3 kinds), level on 9 rows (2 kinds) | absent from packet bytes and from 100,000 decoded rows |
| Chunk invariance, determinism | exact, all levels, three tasks |
| Every single-bit flip of a packed file; 13,164 one-byte mutations with a recomputed CRC | all flips rejected; a mutation is rejected or is the canonical encoding of what it decodes to |
| Independent reader and sampler written from sections 3, 5 and 8 (a scratch script, not in the repository), 300 rows | regression levels A, C, E: identical float32 cells; binary and multiclass K=5 at level E: within 4.8e-7 (it used an exact `Phi`) |
| Leave one row out, n=2000 | smoothed knots move at most 0.051 (bound 0.185); stored knots at most 0.058 |

Command-line run on generated tables (n=1024, mixed skewed / integer / binary
columns, regression, debug build): fit 0.11 s at P=48, 0.79 s at P=128,
decode 100,000 rows of 48 columns in 0.5 s. Against the FIT rows at P=48,
level E (1,592 B): median per-column KS 0.012, worst 0.046; share of synthetic
cells outside the FIT range 0.13 %; every decoded cell of a discrete or
integer column is on a FIT level.

## 13. Open points

1. Target sections are zeros: no label, no target noise, importance order is
   column order. The target stage must fix their dequantisation and decode,
   which changes the recipe text and therefore `decoder sha8`.
2. A gap between two modes is filled by interpolation (0.5 unit at 17 knots).
3. The 5-knot levels are coarse (0.36). A monotone cubic in the score
   coordinate is a candidate that costs no bytes; not adopted without a
   measurement.
4. Atom masses have a 63:1 dynamic range; a rarer atom is raised to
   `1/sum(codes)`.
5. Class means are represented only inside the span of the stored loadings.
6. Wide tables: at 2,048 B regression gets level A only from P=184 (level B
   is `34 + 11P`) and is over budget from P=329; no narrower dependence block
   exists yet. Tables
   with many 16-level columns can exceed the budget on S2 alone.
7. Missing cells are read as the median (as the scaler does); missingness is
   not modelled.
8. Knot windows hold fewer than 10 rows when `n_fit < 128` (7 rows from 64,
   5 below); the 10-row floor is strict for atoms and class means only.
9. Bytes are reproducible on one platform. `exp`, `ln`, `log2`, `2^x`,
   `asinh` and `sinh` come from the platform math library; another platform
   can differ in the last bit of a decoded cell and, rarely, in a code.
10. The module is compiled into `dope_statpacket` by path; `lib.rs`, the owner
    map and the test map do not list it yet. No Python mirror, no shared golden
    fixtures and no embedding prefix exist yet.
