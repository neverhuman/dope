# Scoreboard data-plane formats (`dope-kpi/v2`)

These formats are the boundary between the Python side (PyTorch checkpoints,
training) and the Rust scoreboard (`engine/dope-arena`, binary `dope_kpi`).
All integers are little-endian. All paths are relative to a run directory.
Nothing here may contain column names, category strings or other source
metadata beyond what the corpus manifest already holds.

## 1. Flat corpus arena — `dope.flat-corpus/v1`

Written once per corpus by `python -m research.levels.s3_flat_corpus pack`,
read by training workers (memory-mapped) and by `dope_kpi`.

```
<arena>/META.json     identity and checksums
<arena>/index.jsonl   one JSON object per table, in table_id order
<arena>/X.f32         float32, row-major, tables concatenated in table_id order
<arena>/y.f32         float32 target per row, tables concatenated
<arena>/dup.i32       int32 duplicate-group id per row, tables concatenated
```

`index.jsonl` fields (all required):

| Field | Type | Meaning |
|---|---|---|
| `table_id` | int | 0-based position in the arena |
| `dataset_hash` | string | manifest `dataset_hash` |
| `canonical_sha256` | string | manifest `canonical_sha256` |
| `family_stem` | string | manifest `family_stem` (file-name stem; not the lineage family) |
| `task` | string | `regression`, `binary` or `multiclass` |
| `n_classes` | int | 0 for regression, 2 for binary, K for multiclass |
| `n_rows`, `n_cols` | int | table shape |
| `split` | string | manifest `split` |
| `x_off` | int | offset into `X.f32` in float32 elements |
| `row_off` | int | offset into `y.f32` and `dup.i32` in elements |

Cell (r, c) of table t is element `x_off + r * n_cols + c` of `X.f32`. NaN is
stored as float32 NaN. Targets: regression value; binary 0/1; multiclass class
index 0..K-1 exactly as stored in the corpus.

`META.json`: `schema`, `source_manifest_sha256`, `tables`, `rows`, `cells`,
`x_sha256`, `y_sha256`, `dup_sha256`, `index_sha256`.

## 2. Lineage families — `families.jsonl`

Written by `dope_kpi build-panels`. One object per table:
`dataset_hash`, `family_id` (lineage cluster id, stable string), `canon_stem`,
`evidence` (`name`, `content` or `singleton`), `split`, `task`.
A family never spans the training split and a panel. Generic stems (`rows`,
`export`, `unknown`, `en`, `csv_layers`, `data`, `train`, `test`, `dataset`)
never merge tables by name; they merge only on content evidence.

## 3. Panel manifest

The existing `dope_arena::panel::PanelManifest` JSON (protocol
`dope-fidelity/v3`), with `dataset_id = dataset_hash`, `family_id` from
`families.jsonl`, `target_column = "__target__"`, `source_sha256 =
canonical_sha256`. Panels: `legacy-v250` (current validation families with no
training sibling), later `dev-select`, `dev-confirm`, `sealed`.

## 4. Cell schedule — `dope.kpi-schedule/v1`

`schedule.jsonl`, written by `dope_kpi plan`, read by the emitter and by
`dope_kpi score`. One object per (table, view, data seed), sorted by `cell_id`:

| Field | Type | Meaning |
|---|---|---|
| `cell_id` | string | `<panel_id>:<dataset_hash>:<view>:d<data_seed>` |
| `panel_sha256` | string | canonical sha of the panel manifest |
| `dataset_hash`, `table_id` | string, int | table identity in the arena |
| `family_id`, `task`, `n_classes` | | from panel / arena |
| `view` | string | `full` (all columns) or `p48` (seeded 48-column subset of a wider table) |
| `cols` | int[] | column indices of the view, ascending |
| `data_seed` | int | split seed |
| `stratum` | string | `headline` (n_A >= 200 and n_T >= 200) or `small_n` |
| `a_idx`, `b_idx`, `c_idx`, `t_idx` | int[] | disjoint row indices; no duplicate group straddles roles; `len(a_idx) == len(b_idx)`; `c_idx` may be empty |
| `sample_seeds` | int[] | generator seeds to emit for this cell |
| `multipliers` | int[] | synthetic row multiples of `len(a_idx)` to emit (`[4, 1]`) |

Role sizes: `n_T = clamp(round(0.30 * G), 200, 2000)` duplicate groups when the
table allows, `n_A = n_B = min(1024, floor(rest / 2))`, `C` takes up to `n_A`
of what is left. Every random stream is derived from
`sha256("dope-kpi-v2|" + panel_sha256 + "|" + dataset_hash + "|" + data_seed +
"|" + sample_seed + "|" + purpose)`; split and column streams ignore
`sample_seed`.

## 5. Synthetic rows — `DOPEMAT1`

One file per (arm, cell, sample seed, multiplier), written by the emitter or by
a Rust control arm:

```
bytes 0..8    magic "DOPEMAT1"
u32           rows
u32           x_cols          (== len(cols) of the cell)
u32           y_cols          (1)
u32           reserved (0)
float32[rows * (x_cols + y_cols)]   row-major; features first, target last
```

Units are the **raw table units** of the arena restricted to `cols`, in that
order. Target: regression value in raw units; binary 0/1; multiclass class
index. Labels are sampled from the model's conditional, never thresholded or
arg-maxed. File name: `<arm_id>/<cell_id with ':' replaced by '__'>.s<sample_seed>.m<multiplier>.mat`.

## 6. Emission log — `emit.jsonl`

One object per attempted file, including failures (a failure is a row, never a
skip): `cell_id`, `sample_seed`, `multiplier`, `arm_id`, `arch_id`,
`weights_sha256`, `artifact_class` (`source_free`, `rows_at_sampler`,
`control`), `packet_bytes` (length of the serialized encoding the rows were
decoded from; 0 for `rows_at_sampler` and controls), `packet_sha256`,
`cold_bytes` (shared decoder closure, 0 if none), `rows`, `file`, `status`
(`ok`, `failed`, `over_budget`, `unsupported`), `reason`, `encode_ms`,
`decode_ms`.

For `source_free` arms the rows must be produced by
`decode(unpack(pack(encode(A))))`: the encoder sees only `A` rows of the cell,
and the decoder receives only the packet bytes.

## 7. Score outputs

`dope_kpi score` writes, per arm:

- `ledger.jsonl`: `dope_core::dvs::v3::FidelityCellRequest` objects (one per
  cell x auditor x sample seed x multiplier), sorted by `cell_id`, reducible by
  `evaluate_v3`.
- `rows.jsonl`: one compact object per ledger entry: `cell_id`, `family_id`,
  `task`, `view`, `stratum`, `auditor`, `data_seed`, `sample_seed`,
  `multiplier`, `l0`, `l_a`, `l_b`, `l_s`, `r_s`, `r_b`, `informative`,
  `status`, `packet_bytes`, `artifact_class`, `c2st_auc`, `c2st_real_ref`.

Losses: mean squared error in A-standardised target units for regression,
log-loss (probabilities clipped at 1e-12 and renormalised) for classification.
