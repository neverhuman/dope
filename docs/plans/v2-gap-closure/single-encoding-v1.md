# `dope.single-encoding/v1` — stat packet v0 (kind 0)

Status: stages SP1 and SP2 of plan milestone M2. Implemented in
`engine/dope-arena/src/statpacket/` and the binary `dope_statpacket`.
SP1 fixed the container, the scale card, the marginal and dependence sections
and the feature sampler. SP2 (this revision) defines how the target sections
(S3, S8, S10 and the two noise bytes of S1) are fitted and decoded, how labels
are sampled, the embedding of a packet and the command line. The byte layout
of SP1 is unchanged; the decode recipe changed, so `decoder sha8` changed and
SP1 packets are refused.

The packet is a zero-raw-row encoding of one table: explicit, quantised
sufficient statistics of the FIT rows, no training, no shared decoder weights.
Synthetic rows are sampled from the packet bytes alone.

Source of this text: the `encoding-ladder` design of the 2026-11-22 gap study.
Where this document and that design differ, section 12 lists the difference
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
  `unit = clip((y - mean)/sd, -6, 6)`, `raw = mean + sd * unit`; or the class
  map `labels[c]` (the command line writes `labels = 0..K-1`). A multilabel
  target needs no card entry.

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

The recipe text is the constant `RECIPE` in `codec.rs`; since stage SP2
`decoder sha8 = 34 db 2b 6a 8d 18 46 ef` (stage SP1: `95 42 a2 4b 5c e2 6f 91`). Any change to a decode rule changes
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

`Q` prior bytes, then two noise bytes `n0`, `n1` (the fields `resid_sd` and
`temperature`).

- Binary and multiclass: codes `c_0 .. c_{K-2}`; `c_{K-1} = 255 - sum`;
  `pi_c = c_c / 255`. The sum of the stored codes is at most 255. **A class
  with code 0 is never drawn**, neither as a feature regime nor as a label.
  Fit: a class with fewer than 10 FIT rows counts as 0 rows (support floor);
  then largest remainder on `255 * count_c / n`, with every remaining class
  given at least 1; while the total is short, add 1 to the class with the
  largest `exact - code`; while it is over, take 1 from the class (code above
  1) with the largest `code - exact`; ties go to the lower class index. If no
  class reaches the floor every code is 0 and the last class takes the mass.
- Multilabel: one code per label: 0 if the label is set on fewer than 10 FIT
  rows (never emitted), 255 if it is clear on fewer than 10 (always emitted),
  else `clamp(round(255 * rate), 1, 254)`.
- Regression: `n0` is the residual SD code,
  `sd_A = 0 if n0 = 0 else 2^((n0 - 240)/32)` (2.2 % steps, 0.0056 .. 1.38);
  `n1 = 16 d + e` holds two ratio nibbles: `sd_D = sd_A * 2^(-d/5)`,
  `sd_E = sd_D * 2^(-e/5)`. Levels A, B, C use `sd_A`, level D `sd_D`, level E
  `sd_E`: a level that adds target sections has less left to explain.
- Binary, multiclass, multilabel: `n0 = 0` (a reader rejects anything else);
  `n1` is the temperature code, `T = 2^((n1 - 128)/32)`.

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

### 5.3 S3 — linear target

If header flag bit 0 is set: a column bitmap of the columns that carry
weights. Then `O` pairs (`scale` binary16, `intercept` binary16). Then, for
each output, one int8 weight per carried column in ascending column order.
A dense S3 carries all `P` columns and has no bitmap.

```
w_kj = scale_k * code_kj / 127        b_k = intercept_k
```

A reader rejects a scale that is negative, infinite or NaN and an intercept
that is infinite or NaN. Outputs: regression one (the standardised target);
binary one, the logit of class 1; multiclass `K - 1`, output `c - 1` being the
logit of class `c` against class 0, whose logit is 0; multilabel one logit per
label.

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

### 5.6 S8 — main effects

A column bitmap, then for each set column in ascending order `4*O` int8 values
(output-major, 4 bins each). The value of a code is

```
g(c) = 4 * sign(c) * (c/127)^2        (-4 .. 4; step 0.0005 at 0, 0.063 at 4)
```

The bin of column `j` is `#{m : t_jm <= x_j}` (0..3) with the thresholds
`t_j1, t_j2, t_j3` = the column's own decoded quantiles at 1/4, 1/2, 3/4:
`Q_j(-0.6744897501960817)`, `Q_j(0)`, `Q_j(0.6744897501960817)` with `Q_j` of
section 8 at 17 knots (S8 exists from level D on, where all knots are present)
or the atom table. Nothing is stored for the thresholds. The fitter sets a
column's bit only if at least one of its codes is non-zero.

### 5.7 S10 — pairs

Exactly 16 entries: column `a`, column `b` (`I` bytes each, both below `P`),
then `O` int8 coefficients with the value `g(c)` of section 5.6. Entries
0..14 are products `x_a * x_b` of decoded unit values (`a = b` is a square).
**Entry 15 is the constant term**: its `a` and `b` are 0 (a reader rejects
anything else) and its coefficients are added to `eta` as they are. An unused
entry is all zero.

## 6. Budget fitting

`fit_to_budget(packet, max_bytes)` is deterministic and never drops a column.
Importance of a column is its summed absolute weight over outputs,
`sum_k scale_k * |code_kj|`; ties go to the lower index.

1. For level E, D, C, B, A in turn: if the dense packet fits, return it.
2. At level D only, before moving on to C: keep S8 for the `m` most important
   columns, `m` the largest count that fits, if `m >= 1`.
3. If dense level A does not fit: set the sparse flag and keep S3 weights for
   the `m` most important columns, `m >= 0` the largest count that fits.
4. If even `m = 0` does not fit: return that smallest packet with
   `fits = false` (`over_budget`). The table counts as over budget; it is not
   cut.

`fit_packet` applies it after the target fit and then recomputes the
regression noise codes for the shape that was kept (section 7.10), so a packet
with a cut S8 or S3 does not understate its noise.

Byte counts depend on the data since stage SP2: S8 lists only the columns
whose main effects passed the gate of section 7.8, so the "dense" lengths of
section 4 are upper bounds. Measured: a 48-column, 5-class table without
non-linear effects packs level E in 1,703 B (dense: 2,468 B); the same shape
with 36 gated S8 columns packs level D in 1,968 B.

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

Mean carriers (binary, multiclass). The within-class axes below span at most
12 directions, and the class means need not lie in them. Let
`B = sum_c share_c (mu_c - mean)(mu_c - mean)'` with `mean = sum_c share_c mu_c`;
its eigenvalues `beta_g > 1e-6` (at most 12, descending) and unit eigenvectors
`d_g` are the between-class directions (computed through the `K x K` Gram
matrix of `sqrt(share_c) (mu_c - mean)`). Carriers are made if `P > 12`, or
if `P` plus their number is at most 12; otherwise (the axes of a table of at
most 12 columns already span everything and a carrier would push one out)
none is made. A carrier is the loading column

```
l_g = size_g * d_g,   size_g = clamp(max_c |mu_c . d_g| / 5.5, 0.2, 0.7)
```

so that the factor means it needs stay inside the code range, and the
within-class variance it adds is taken out of the matrix the axes are fitted
to: `rho_jk <- rho_jk - l_gj * l_gk` for all `j, k` (diagonal included) and
the communality cap of column `j` drops from 0.98 by `l_gj^2`.

Loadings. `r = min(12, P)`. If `P <= 12`: eigen-decompose `rho` (principal
components reproduce the matrix). Otherwise principal axes: communalities
start at each column's largest absolute off-diagonal correlation; repeat up to
10 times: put `min(h_j, cap_j)` on the diagonal, eigen-decompose, set `h_j` to
`min(cap_j, sum_{f<r} max(lambda_f, 0) * v_jf^2)`, stop when no `h_j` moved by
1e-3 or more; the last decomposition is used. Eigenvalues descend. Axis `f`:
`l_jf = sqrt(max(lambda_f, 0)) * v_jf`. The candidate factors are the `r` axes
with variance `max(lambda_f, 0)` followed by the carriers with variance
`beta_g`; they are sorted by decreasing variance (stable: an axis stays ahead
of a carrier on a tie) and the first 12 are kept. Every kept column is given
the sign that makes its entry of largest magnitude positive (lowest column on
a tie); `code = clamp(round(127 * l_jf), -127, 127)`. Ranks 4, 8, 12 are
prefixes. The eigen routine is cyclic Jacobi (`dope_core::stats::sym_eigen`);
an implementation with another routine can differ by one code step.

Factor means, one factor at a time against the quantised loadings `L`:
`m_cf = (sum_j L_jf * mu_cj) / (sum_j L_jf^2 + 1e-3)`,
`code = clamp(round(m_cf * 127/6), -127, 127)`. With carriers, only the
carriers get means (they are mutually orthogonal and carry the class means
between them) and the axes get 0; without carriers every axis gets its
projection. Nested by construction.

The factor fit costs up to 10 eigen-decompositions of a `P x P` matrix. It is
skipped when the caller's budget cannot hold S4 (`fit_features_for_budget`;
the packed bytes are identical) and for `P > 1024`.

### 7.6 Target: response and support floor

The target is fitted on the unit-space FIT rows `x` after the feature
sections. Regression uses the standardised target of section 2. A class with
fewer than 10 FIT rows is *unsupported*: its rows take no part in any target
fit, it gets no weights, and its prior code is 0 (section 5.1). Binary needs
both classes supported, a multilabel label needs 10 rows on each side;
multiclass needs two supported classes; otherwise all target codes are zero.

Every block below is fitted against the dequantised blocks before it (closed
loop). Solvers (`glm.rs`), all with a fixed step count, a fixed halving line
search (up to 8 halvings, a step is accepted if the penalised likelihood does
not fall) and an early stop when no coefficient moves by 1e-9:

- least squares: one solve of `(A'A + D) c = A'y`;
- logistic: 12 Newton steps (the Hessian `A'WA + D` changes every step and is
  factorised again each time);
- multinomial: symmetric model, one coefficient vector per supported class,
  penalty `sum_k c_k' D c_k / 2`; 40 steps, each solving one `dim x dim`
  system per class with the weights `2 p (1 - p)` (these blocks bound the full
  Hessian, so no `classes * dim` system is formed) and then centring the step
  over classes. The wire outputs are `c_class - c_ref` with `ref` the lowest
  supported class (class 0 when it is supported).

### 7.7 S3 and the temperature

`s_j` = population SD of unit column `j`. Columns with `s_j <= 1e-6` get no
weight. If more than 512 columns remain, the 512 with the largest
`sum_k (sum_i x_ij r_ik)^2 / s_j^2` are kept (`r` the response residuals at
the starting intercepts: the standardised target, `y` minus the FIT share,
or `1[y = c]` minus one over the number of supported classes; ties to the
lower index). Design `[1 | x]`, penalty
`D = diag(0, s_j^2)`: the intercept is free and the weights are penalised as
`Ridge(alpha = 1)` (regression) and `LogisticRegression(C = 1)` (binary: one
weight vector; multiclass: the symmetric model) penalise them on standardised
inputs. Binary starts at the log-odds of the FIT share.

Quantisation per output: `scale = binary16(max_j |w_j|)`,
`code = clamp(round(127 * w_j / scale), -127, 127)`. Then the intercept alone
is refitted against the quantised weights (the same solver on a design of
ones with `w . x` as offset) and stored as binary16; the mean fitted
probability on the FIT rows therefore equals the FIT share.

Temperature (classification): with `eta` of the stored S3 model and
`q = sum_j s_j^2 |w_j|^2` (for multiclass, `|w_j|^2` centred over the supported
classes), six Newton steps from `T = 1` on
`loglik(T * eta) - T^2 q / 2`, clamped to [0.25, 4], then the nearest code.
`T = 1` is the stationary point of the unquantised fit, so the code is 128
unless the int8 rounding moved the scale of the weights.

### 7.8 S8

Candidates: the 96 most important non-constant columns (section 6). For a
candidate, the FIT rows are binned by its thresholds (section 5.6); a bin with
fewer than 10 rows has no coefficient, and the column is skipped unless two
bins have one. Gate: with `r` the response residuals at `T * eta_S3`
(regression `y - eta`, logistic `y - sigmoid`, multiclass `1[y = c] - p_c` for
supported classes and rows, 0 elsewhere), `v_k = mean(r_k^2)`, and the groups
= the bins with a coefficient plus one group for all other rows,

```
stat = sum_{k: v_k > 1e-12} (sum_g S_kg^2 / n_g - (sum_g S_kg)^2 / n) / v_k
df   = (groups - 1) * #{k: v_k > 1e-12}
keep the column if stat >= gate(df, candidates)
gate(df, M) = df * (1 - 2/(9 df) + z * sqrt(2/(9 df)))^3,   z = ppf(1 - 0.05/M)
```

(`S_kg`, `n_g`: residual sum and rows of group `g`; the gate is the
Wilson-Hilferty upper `0.05/M` point of a chi-square.) The kept columns are
fitted jointly: indicator design, ridge 1 per coefficient, offset
`T * eta_S3`, solver of the task; coefficient `/ T`, code = nearest `g`.

### 7.9 S10

Candidate columns: the 48 most important non-constant ones; candidates are
all products `a <= b` of them. Start from the residuals at `T * eta` of S3 and
S8. Up to 15 rounds: for every unused candidate with centred product `z`
(`spread = sum z^2 > 1e-9 n`), `score = sum_k (z . r_k)^2 / (spread * v_k)`
over the outputs with `v_k > 1e-12` (`v_k` of the current residual); take the
largest (first in `(a, b)` order on a tie); stop if it is below
`gate(#outputs, #candidates)`; else keep it and deflate
`r_k <- r_k - z (z . r_k)/spread`. The kept products are fitted jointly with a
free constant (design `[1 | x_a x_b ...]`, penalty the variance of each
product, offset `T * eta` of S3 and S8); coefficients `/ T`, code = nearest
`g`; a product whose codes are all zero is dropped. Last, the constant alone
is refitted against the quantised products and stored in entry 15.

### 7.10 Noise codes (regression)

For the packet as it stands (its level and masks), tiers A (S3), D (+ S8),
E (+ S10): `sd_tier^2 = RSS_tier / max(n - df_tier, n/2)` with `df` = 1 + the
number of non-zero codes the tier uses. `n0` = nearest code to `sd_A`
(`clamp(round(240 + 32 log2 sd), 1, 255)`, 0 for 0); `d` = nearest nibble to
`sd_D / sd_A(n0)`; `e` = nearest nibble to `sd_E / (sd_A(n0) * 2^(-d/5))`.
`fit_packet` runs this once after the target fit and again after the budget
step.

## 8. Decode

Row `i` depends only on `(packet, seed, i)`. Draws:
`U(stream) = ((hash3(seed, stream, i) >> 11) + 0.5) / 2^53`, `N(stream) =
ppf(U(stream))`, with `hash3` of `dope_core::rng` and `stream = purpose << 32
| draw`: purpose 1 regime (draw 0), 2 factor (draw = factor), 3 column noise
(draw = column), 4 target (draw 0, or the label index for multilabel). Below,
`U(p, d)` and `N(p, d)` name purpose and draw.

### 8.1 Features

With `r` the rank of the level, `l_j` the first `r` loadings of column `j`:

```
psi_j = max(0.02, 1 - |l_j|^2);   s_j = sqrt(|l_j|^2 + psi_j)
f_k = N(2, k), k < r
regimes (M > 0 and r > 0):  c = first class with cumulative pi > U(1, 0), else the last class with pi > 0;  f <- f + m_c
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

### 8.2 Target

On the decoded unit values `x` (float64, before any rounding):

```
eta_k = b_k + sum_j w_kj x_j                                  S3
      + sum_{j in S8} g(code_{j,k,bin_j(x_j)})                from level D
      + sum_{t<15} g(code_tk) x_a x_b + g(code_15,k)          at level E
```

- Regression: `y = clip(eta + sd_level * N(4, 0), -6, 6)` in standardised
  units; raw `mean + sd * y` through the scale card.
- Binary, multiclass: logits `l_0 = 0`, `l_c = T * eta_{c-1}`; over the classes
  with `pi_c > 0`, weights `exp(l_c - max l)`; the label is the first such
  class, in ascending order, whose cumulative weight exceeds
  `U(4, 0) * total`, else the last one. **A label is always a draw, never an
  arg-max.** The draw is separate from the feature regime of section 8.1.
- Multilabel: label `l` is 0 for rate code 0, 1 for 255, else
  `1[U(4, l) < sigmoid(T * eta_l)]`.

Alternative label rule `regime` (not the default; `sample --labels regime`):
with regimes (binary or multiclass, rank above 0) the label is the regime `c`
of section 8.1; without regimes it is the rule above. Section 13 has the
measured comparison that keeps the separate draw as the default.

Output: features then target, as float32. With a scale card: `to_raw(x_j)`
and the raw target (regression value, class label, or 0/1 cells); without:
unit values, the standardised target or the class index. Rows `[0, n)` equal
rows `[0, a)` followed by `[a, n)` bit for bit.

## 9. Embedding

`embed(packet)` is 128 int8 values; the first 64 are the 64-D embedding. It is
a function of the packet bytes alone. A summary `s` of 128 numbers is split
into a primary half (slots 0..63) and a secondary half; half `b` is rotated by
the fixed orthonormal matrix `R_b` and quantised:

```
e = clamp(round(48 * R_b s_b), -127, 127)
R_b: 64 x 64; row r starts as N(seed = 0x44534531454d4244, stream = 64 b + r, index = column),
     rows orthonormalised in order by modified Gram-Schmidt
```

Because each half has its own rotation, the 64-D embedding is exactly the
prefix of the 128-D one and loses none of the primary half. Every entry of `s`
is a symmetric function of the columns (lists are sorted before they are
summed, sums of loading products are exact integers), so permuting the
columns of a packet leaves the embedding unchanged bit for bit.

Summary (`weight x value`; "moments" = mean, SD, quartiles 25/50/75 across
columns; `top n` = the `n` largest, descending, zero padded):

| Slots | Primary half |
|---|---|
| 0-3 | 0.3 x one-hot task |
| 4, 5 | `ln(1 + K)/ln 256`; `(log2 P - 4)/8` |
| 6-8 | shares of columns that are not continuous, discrete, constant |
| 9-16 | 1.5 x class shares, top 8, each minus `1/K` (multilabel: label rates, top 8) |
| 17 | entropy of the class shares over `ln K` (0 for multilabel) |
| 18-44 | for each score `z` in -2, -1.5, .., 2: mean, SD, median across varying columns of `Q_j(z) - z` (at the knots of the level) |
| 45-49 | 1.5 x moments of the uniqueness `max(0, 1 - h_j)`, `h_j = |l_j|^2` at the rank of the level |
| 50-57 | 2 x `(lambda_f - mean h)/P`, top 8 eigenvalues of `L'L` |
| 58-61 | importance shares, top 3; participation `1/(P sum share^2)` |
| 62 | regression: noise SD of the level; else `min(2, mean_k |b_k| / 4)` |
| 63 | 3 x root mean square of the modelled correlations between different columns, `sqrt((|L'L|_F^2 - sum h_j^2)/(P (P - 1)))` |

Secondary half (slots 64 + ..): 0-17 quartiles 25 and 75 of `Q_j(z) - z` per
score; 18-21 atoms (mean and SD of `A_j/16`, mean entropy of the masses,
lattice share); 22-29 norms of the class factor means over 6, top 8; 30-33
eigenvalues 9..12 as above; 34 mean `ln(1 + scale_k)`; 35 SD of the intercepts
(capped at 4) over 4; 36 mean `|main effect|`; 37 mean `|pair coefficient|`;
38 mean `|constant|`; 39 `T - 1`; 40, 41 mean and SD of
`ln(half(hi)/half(lo))`; 42 share of columns with main effects; the rest 0.

A packet cut to a lower level has other sections and so another embedding;
embeddings are comparable between packets of the same level.

## 10. What the packet can and cannot contain

- No string, name or header.
- No column minimum or maximum: a knot is a mean over a window of `2h + 1`
  order statistics (11 from 128 rows, 7 from 64, 5 below) that never includes
  the smallest or largest row; the lowest and highest knots sit at the clipped
  probabilities `u_lo` and `1 - u_lo`. The packet bytes are identical when
  the extreme cells of every column are pushed to the clip (tested).
- No level with fewer than 10 rows: such a level is merged before the atom
  table is built and can never be sampled.
- No class factor mean from fewer than 10 rows.
- No class or multilabel side with fewer than 10 rows: it has prior code 0
  (or 255), no weights, and is never sampled, under either label rule. A table
  whose minority class is below the floor yields a single-class target.
- No S8 coefficient from a bin with fewer than 10 rows.
- The regression target is clipped at 6 standardised units before any fit and
  after sampling, so a planted extreme target cannot be reproduced; its mean
  and SD live on the owner-side scale card only.
- Removing one row moves a smoothed knot by at most the largest gap between
  neighbouring inner order statistics (tested for every row of a fixture).
- The header reveals `floor(log2 n_fit)`, `P`, `K` and the task; S1 reveals
  the shares of the supported classes to 1/255; the S8 bitmap and the S10
  indices reveal which columns have non-linear or pairwise effects.
- `inspect` prints header fields, byte lengths and counts only; `embed`
  prints a function of the packet.

This is privacy by construction of the stored statistics. It is not a formal
differential-privacy or de-identification claim.

## 11. Command line

```
dope_statpacket fit      --in A.mat --task regression|binary|multiclass|multilabel
                         [--classes K] [--budget 2048] [--level A..E]
                         --out P.dse --card card.json
dope_statpacket sample   --packet P.dse [--card card.json] --rows N [--row0 R] [--seed S]
                         [--labels sampled|regime] --out S.mat
dope_statpacket truncate --packet P.dse --level A..E --out Q.dse
dope_statpacket inspect  --packet P.dse
dope_statpacket embed    --packet P.dse
```

Tables are `DOPEMAT1` files (formats.md section 5) in raw units, features
first, target last (one target column; `K` for multilabel). Class labels are
the integers `0..K-1`. `sample` writes the target too; without `--card` the
rows stay in unit space. Each command prints one JSON object: `fit` the status
(`ok` or `over_budget`), level, budget step, kept S3 and S8 columns, header
and byte ledger; `inspect` the header and ledger (integers and names only);
`embed` the two int8 lists. Failures print one fixed line on stderr and exit
with code 2. The names of stage SP1 remain as aliases: `info`, `decode`,
`--input`, `--packet` (of `fit`), `--max-bytes`, `--output`.

Library entry point: `fit::fit_packet(x, y, task, k, size, n_fit)` with `x`
the raw FIT rows (row-major), `y` a `Target`, `size` a level cap and an
optional byte budget; it returns the packet bytes, the scale card and the
budget outcome.

Three packets pinned by test (`golden_packets_match_the_specification`):

- 16 rows, one constant feature, regression, feature sections only, level A,
  36 bytes: `44534531 01 00 00 00 0100 00 00 03 04 00 00 34db2b6a8d1846ef |
  0000 | 02 | 0000000000 | 93e67a93` (header, S1, S2, S3, CRC).
- 32 rows, features `i` and `i mod 2` for `i = 0..31`, regression, feature
  sections only, level E, 139 bytes, SHA-256
  `52af756efbbfa3d127dff841542414dc42aed32406c314fb18ba91270b26a85e`.
- 64 rows, features `i` and `i mod 2`, target
  `2 i + 5 (i mod 2) + (0, 3, -2, 1)[i mod 4]`, through `fit_packet` at level
  E, 135 bytes, starting `4453453101000000020004000a060000 34db2b6a8d1846ef
  5110 ...` (S1: residual code 0x51, ratio nibbles 1 and 0); the test holds
  all the bytes.

## 12. Decisions and differences from the design

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
| D15 | — | `decode` writes `y_cols = 0` until the target stage | Features only (stage SP1); superseded by section 8.2 |
| D16 | "residual-SD u8 log scale, temperature u8" | Regression: SD code and two ratio nibbles; classification: temperature in the second byte, the first is 0 | One SD cannot serve levels A-C, D and E: on a table with one interaction the residual is 0.55 without S10 and 0.25 with it |
| D17 | Temperature unspecified | The maximiser of the penalised likelihood of the stored model; 1 at the unquantised optimum | A free scale would undo the ridge; this only absorbs rounding |
| D18 | S8 "4-bin main effects", S10 "top-16 pairs (2 + O) B each", no scale | One fixed companded code `g(c) = 4 sign(c) (c/127)^2`; bins at the column's decoded quartiles; S10 = 15 products and a constant | No bytes for a scale or a threshold. A product has a non-zero mean; without the constant, level E would shift the target mean by `sum coef * E[x_a x_b]` |
| D19 | S8 and S10 always filled | Stored only when a chi-square gate is passed | On linear tables, ungated S8/S10 cost retention (binary 0.97 against 0.99, multiclass 0.96 against 1.00 at level E) and bytes |
| D20 | Class priors for every class | A class under 10 rows has code 0, no weights, and is never drawn | The support floor of the features, applied to the target; a declared class label is otherwise a value carried by fewer than 10 rows |
| D21 | `m_c = (L'L + 1e-3 I)^-1 L' mu_c` on within-class axes | Mean carriers (section 7.5) | Class means outside the span of 12 within-class axes were lost: largest correlation gap at level E 0.151 without, 0.041 with, on a 20-column 5-class table |
| D22 | Importance = `|w|` | `sum_k scale_k |code_kj|` | Outputs have different scales |
| D23 | Open: label = regime or softmax | Separate draw from `softmax(T eta)`; `regime` is an option | Measured in section 13 |
| D24 | 64-D projection of one summary | Two 64-D halves, each with its own rotation | The 64-D embedding is then an exact prefix and loses nothing of the primary summary |
| D25 | — | Regime fallback is the last class with a non-zero prior | A class with code 0 must never be drawn |

## 13. Measured

From `cargo test -p dope-arena --locked -- --nocapture` (54 unit tests in the
module, 8 black-box tests of the binary) on deterministic fixtures; unit space
unless stated. Section 13.1 is the feature part (stage SP1, run again at this
revision), 13.2 the target stage.

### 13.1 Container, marginals, dependence

| Check | Result |
|---|---|
| Byte table, formulas, prefix property, corruption, budget | exact (table of section 4) |
| Scale card vs numpy reference | identical float64 and float32 on 52 cells |
| Scale card round trip, 11,998 cells with \|unit\| < 6 (89 in the asinh tail), through a float32 unit value | worst relative error 1.4e-7 |
| Gaussian AR(1) table, P=12, n=2000, 50,000 decoded rows: max correlation gap vs FIT rows | rank 12: 0.019; rank 8: 0.080; rank 4: 0.145 |
| 3 factors in 30 columns, n=3000 | rank 12: 0.019; rank 8: 0.027; rank 4: 0.031 |
| Continuous / binary / ordinal table, n=4000: max correlation gap (raw units) | 0.034 |
| Binary class regimes, P=6, n=2000 | correlation gap 0.019 (0.033 before the carriers), quantile gap 0.040 |
| Quantile gap, 5 column shapes, n=2000, 100,000 decoded rows, probabilities 0.02..0.98 | 5 knots 0.360; 9 knots 0.112; 17 knots 0.057 |
| Two separated modes (known weak case), 17 knots | 0.498 |
| Level frequencies of discrete columns, 50,000 decoded rows | 2.43 SE vs packet masses, 0.55 SE vs FIT rows; mass quantisation error 0.0008 |
| Canaries: unique extreme (3 kinds), level on 9 rows (2 kinds) | absent from packet bytes and from 100,000 decoded rows |
| Chunk invariance, determinism | exact, all levels, three tasks, with and without the target |
| Every single-bit flip of a packed file; 13,164 one-byte mutations with a recomputed CRC | all flips rejected; 10,573 mutations accepted as the canonical encoding of what they decode to, 2,591 rejected |
| Leave one row out, n=2000 | smoothed knots move at most 0.051 (bound 0.185); stored knots at most 0.058 |

An independent reader and sampler was written from this text at stage SP1 for
the feature sections (identical float32 cells for regression). It has not been
repeated for the target sections and the new recipe.

### 13.2 Target, labels, embedding

Solvers and the stored model:

| Check | Result |
|---|---|
| Largest gradient entry at the solution, n=500 | least squares 4e-13; logistic 5e-8; multinomial (3 classes, skipped rows) 2e-7, weights centred to 1e-16 |
| Stored S3 weights vs the plain model on standardised inputs (worst gap in int8 steps; rounding bound 0.5) | ridge 0.43; logistic 0.46; multinomial K=4 0.45 |
| Mean fitted class probability on the FIT rows vs FIT share | binary within 2e-3; multiclass 9e-6 |
| Temperature code on these fits | 128 (T = 1) |
| Regression table with a planted U-shaped effect (n=1200) | S8 keeps that column only; residual RMS A / D / E 0.732 / 0.589 / 0.404, stored SD 0.738 / 0.560 / 0.424 |
| Regression table with a planted interaction | first pair is the planted one; residual RMS 0.548 / 0.548 / 0.247, stored SD 0.545 / 0.545 / 0.237 |
| Linear table (regression and binary) | S8 and S10 empty; stored SD 0.279 vs RMS 0.276 |
| Level D with S8 cut to 5 of 6 columns by the budget | noise SD 0.732 after the refresh, 0.637 without it |
| Binary labels, 100,000 rows, ten probability bins | share of label 1 follows the model probability within 1.3 SE in every bin (an arg-max would give 0 or 1) |
| Regression target, 100,000 rows | mean 50.19 (FIT 50.15), SD 10.74 (FIT 10.72) |
| Canaries: class on 9 rows, binary with 9 positives, multilabel sides on 9 rows, unique extreme target | no prior mass and no weights; absent from 100,000 sampled rows under both label rules |

End to end through the command line (`statpacket_e2e.rs`): three tables,
`n_fit = 600`, budget 2,048 B, `4 n_fit` sampled rows, a plain
`Ridge(alpha = 1)` / `LogisticRegression(C = 1)` (written in the test, on
inputs standardised by its own training rows) trained on the real FIT rows and
on the sampled rows, loss on 2,000 real test rows.
Retention `R = (L0 - L_S)/(L0 - L_A)`. Every packet reached level E at the
budget; lower levels are `truncate` of it.

| Table | Level | Bytes | Retention | Max correlation gap | Label gap (mean of 8 samples, SE) |
|---|---|---|---|---|---|
| regression, P=16, linear + one interaction (`L0` 654.3, `L_A` 165.6) | A | 155 | 0.999 | 0.729 | — |
| | B | 219 | 0.998 | 0.079 | — |
| | C | 325 | 0.999 | 0.065 | — |
| | D | 411 | 0.999 | 0.071 | — |
| | E | 523 | 1.001 | 0.077 | — |
| binary, P=12, 17 % positives (`L0` 0.479, `L_A` 0.344) | A | 120 | 1.022 | 0.634 | 0.97 |
| | B | 176 | 1.035 | 0.276 | 0.49 |
| | C | 265 | 1.010 | 0.201 | 0.91 |
| | D | 333 | 1.025 | 0.197 | 1.54 |
| | E | 437 | 0.990 | 0.061 | 0.17 |
| multiclass K=5, P=20, class-separated means (`L0` 1.543, `L_A` 0.749) | A | 260 | 0.991 | 0.360 | 2.45 |
| | B | 360 | 1.003 | 0.155 | 3.15 |
| | C | 517 | 0.978 | 0.085 | 1.23 |
| | D | 634 | 0.979 | 0.090 | 1.28 |
| | E | 830 | 0.981 | 0.071 | 0.59 |

Notes on the table. Retention is one sample at one seed; its seed-to-seed SD
is about 0.03, and with sampled labels its expected value at `4 n_fit` rows is
about 0.975 (the refit on sampled labels adds estimation noise of that size).
Level A has independent columns, hence its correlation gap. The correlation
gap is against the 600 FIT rows, whose own sampling error is about 0.04. The
label gap is the largest gap over classes between the mean label share of 8
samples and the FIT share, in standard errors of one sample of 2,400 rows. At
level E, 8 of 8 (binary) and 7 of 8 (multiclass) single samples are within
3 SE; the one outside (3.13 SE) is what 5 classes give by chance about once in
fifty (30 further seeds: mean largest gap 1.59 SE, 1 above 3). At levels A and
B of the multiclass table the gap is systematic (2.5 and 3.2 SE): with no or
only one of the four class directions in the regimes, the distribution of
`eta` over decoded rows differs from the FIT rows.

Label rule (mean retention of 8 samples per cell):

| Table | Rule | Level B | Level D | Level E |
|---|---|---|---|---|
| binary | sampled (default) | 0.996 | 0.970 | 0.974 |
| | regime | 0.690 | 0.702 | 1.001 |
| multiclass, class-separated | sampled (default) | 0.978 | 0.976 | 0.976 |
| | regime | 0.346 | 0.825 | 0.993 |

Tying the label to the regime is ahead by 0.02 to 0.03 at level E, where every
class mean is represented, and behind by 0.15 to 0.63 at lower levels; it also
ignores S3, S8 and S10. The separate draw therefore stays the default.

Budget fallback through the command line (`n_fit = 600`, budget 2,048 B):

| Table | Result |
|---|---|
| multiclass K=5, P=48, linear class signal | level E, 1,703 B (S8 empty after the gate) |
| multiclass K=5, P=48, class signal in the spread of every column | level D, 1,968 B, S8 on 36 columns; at a budget of 1,900 B level D with S8 on the 31 leading columns, 1,888 B |
| multiclass K=5, P=200 | level A, sparse S3 on 190 columns, 2,045 B |
| regression P=200 / P=320 | level A dense, 1,434 B / level A, sparse S3 on 54 columns, 2,048 B |
| regression P=400 | `over_budget` (reported, not cut) |

Embedding (cosine similarity of int8 codes):

| Check | 64-D | 128-D |
|---|---|---|
| Two disjoint halves (600 rows each) of 5 tables, lowest | 0.982 | 0.968 |
| Unrelated tables among those 5, mean / highest | 0.448 / 0.752 | 0.463 / 0.728 |
| Halves (300 rows each) of the three end-to-end tables, lowest | 0.961 | 0.933 |
| Different end-to-end tables, highest | 0.790 | 0.700 |
| Same table with permuted columns, refitted (8 tables) | cosine 1.00000, no code changed | same |

Command-line times on the end-to-end tables (development profile, loaded
host): fit under 0.1 s (P up to 20), 0.2 s (K=5, P=48), 1.4 s (K=5, P=200);
sampling 100,000 rows 0.2 to 0.6 s. Scale card: 1.7 to 2.9 kB of JSON for 12
to 20 columns, reported by `fit` as `card_bytes` and never part of the packet.

## 14. Open points

1. The retention figures are for plain linear evaluators on synthetic tables
   whose signal is mostly linear. Nothing here measures a gradient-boosted
   evaluator, real corpus tables, or realism (C2ST); the scoreboard does.
2. A gap between two modes is filled by interpolation (0.5 unit at 17 knots).
3. The 5-knot levels are coarse (0.36). A monotone cubic in the score
   coordinate is a candidate that costs no bytes; not adopted without a
   measurement.
4. Atom masses have a 63:1 dynamic range; a rarer atom is raised to
   `1/sum(codes)`.
5. Class structure at low rank: carriers and axes share the 12 slots by
   variance, so levels B to D can hold only some of the class directions; on
   the 5-class table the label shares are then off by about 3 SE (level B).
   Between-class directions beyond the 12 slots (many classes) are dropped.
6. Rare classes disappear: a class under 10 FIT rows is never sampled, and a
   binary table with fewer than 10 minority rows yields a single-class target.
   This follows from the support floor; its cost on small tables with many
   classes is not measured.
7. The regime label rule is ahead at level E by 0.02 to 0.03 on two tables. A
   rule that combines the regime posterior with `softmax(T eta)` is untested.
8. S8 uses quartile bins and S10 products of unit values; both are gated at a
   family-wise 5 %. Effects below the gate are dropped, and S10 scans pairs of
   the 48 most important columns only, so an interaction between two columns
   without linear weight is missed on wider tables.
9. A sparse S3 (budget fallback) keeps the weights of the full fit; they are
   not refitted on the kept columns and the intercept is not adjusted. Noise
   codes are refreshed.
10. Fit cost grows as `steps * classes * n * P^2` (multinomial: up to 40
    steps); the S3 fit is capped at 512 columns by screening, the dependence
    fit at 1024. Very long tables are not subsampled.
11. Wide tables: at 2,048 B regression gets level A only from P=184 (level B
    is `34 + 11P`) and is over budget from P=329; no narrower dependence block
    exists yet. Tables with many 16-level columns can exceed the budget on S2
    alone.
12. Missing cells are read as the median (as the scaler does); missingness is
    not modelled.
13. Knot windows hold fewer than 10 rows when `n_fit < 128` (7 rows from 64,
    5 below); the 10-row floor is strict for atoms, classes, class means and
    S8 bins only.
14. Bytes are reproducible on one platform. `exp`, `ln`, `log2`, `2^x`,
    `asinh` and `sinh` come from the platform math library; another platform
    can differ in the last bit of a decoded cell and, rarely, in a code.
15. The embedding has no learned part. Two unrelated tables of the same task
    and width reach a cosine of 0.75; family retrieval on the corpus is not
    measured here.
16. The module is compiled into `dope_statpacket` by path; `lib.rs`, the owner
    map and the test map do not list it yet. No Python mirror and no shared
    golden fixtures exist yet; the second implementation of stage SP1 has not
    been repeated for the target sections.
