# External-command arms — `dope_kpi emit-cli`

`emit-cli` scores any command-line generator on exactly the cells of a frozen
schedule: it hands the generator the A rows of each cell, lets it fit a packet,
asks it for synthetic rows and stores them as `DOPEMAT1` files with one
`emit.jsonl` row per file ([formats.md](formats.md) sections 4 to 6). The
previous release binary is scored this way, unchanged.

```
dope_kpi emit-cli --arena <dir> --schedule <schedule.jsonl> --out <run_dir>
                  --arm-file <arms.json> --arm <arm_id>
                  [--threads N] [--timeout-s 300] [--resume] [--limit-cells N]
```

| Option | Meaning |
|---|---|
| `--threads N` | cells run in parallel (default 1); a cell runs one command at a time on one CPU thread |
| `--timeout-s S` | wall-clock limit of each fit and each sample command, in seconds (default 300, fractions allowed) |
| `--resume` | keep the cells whose rows are all `ok` in the existing log and run the others again |
| `--limit-cells N` | run only the first N cells of the schedule file; the summary records the limit |

Arena, arm file and schedule are checked before the run directory is touched.
A generator that fails never fails the run: the exit status is 0 and every
failure is a row. The run itself fails (non-zero exit, one `error:` line, no
summary) only for bad inputs or when its own files cannot be written.

## 1. Arm file — `dope.kpi-ext-arms/v1`

```json
{
  "schema": "dope.kpi-ext-arms/v1",
  "arms": [
    {
      "arm_id": "my-generator",
      "artifact_class": "source_free",
      "io": "csv",
      "binary": "bin/my-generator",
      "fit": ["{bin}", "fit", "{fit_rows}", "-o", "{packet}"],
      "sample": ["{bin}", "sample", "{packet}", "-n", "{rows}", "--seed", "{seed}", "-o", "{out}"]
    }
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `arm_id` | string | letters, digits, `.`, `_`, `-`; starts with a letter or digit; at most 80 bytes; unique in the file. It is a directory name |
| `artifact_class` | string | `source_free`, `rows_at_sampler` or `control`; copied to every row. A generator whose packet holds fit rows is not `source_free` |
| `io` | string | `csv` or `mat`: the format of `{fit_rows}` and `{out}` (section 4) |
| `binary` | string | path of the executable. A relative path is resolved against the directory of the arm file. The file is hashed once per run: its SHA-256 is `weights_sha256` and its length is `cold_bytes` |
| `binary_sha256` | string, optional | 64 lowercase hex digits; the run is refused when the executable does not match |
| `fit`, `sample` | string[] | argument vectors, element 0 is the program (section 2) |
| `budget_bytes` | int, optional | value of `{budget}`. It is only passed through; the scorer applies budgets |
| `note` | string, optional | free text, ignored |

Unknown fields are refused. Every arm of the file is checked, including the
ones that are not selected. [`ext-arms.example.json`](ext-arms.example.json)
holds the arms of the previous signed release (`dope-1.3.0-small`, `-medium`,
`-large`: whole-file caps of 2, 4 and 16 KiB) and `packet-2k` for
`dope_statpacket`. Its `binary` paths are relative: copy the file next to the
unpacked release directory (or link the directory next to the copy). The
release arms are pinned to the signed binary by `binary_sha256`.

## 2. Commands and placeholders

A command is the argument vector with placeholders replaced inside each
argument. It is executed directly, never through a shell, so no value can
split or join arguments. `{name}` is a placeholder when `name` consists of
letters, digits and `_`; any other use of braces is literal text.

| Placeholder | fit | sample | Value |
|---|---|---|---|
| `{bin}` | yes | yes | absolute path of `binary` |
| `{fit_rows}` | yes | no | file with the A rows of the cell (section 4) |
| `{packet}` | yes | yes | `<run_dir>/<arm_id>/packets/<cell>.packet`; the fit must create it |
| `{card}` | yes | yes | `<run_dir>/<arm_id>/packets/<cell>.card`; the fit may create it |
| `{out}` | no | yes | file the sample command must write (section 4) |
| `{rows}` | no | yes | rows to generate: `multiplier * len(a_idx)` |
| `{seed}` | no | yes | 31-bit sample seed (below) |
| `{fit_seed}` | yes | yes | 31-bit fit seed (below) |
| `{task}` | yes | yes | `regression`, `binary` or `multiclass` |
| `{task2}` | yes | yes | `regression` or `classification` |
| `{classes}` | yes | yes | 0 for regression, else the number of classes K of the table |
| `{budget}` | yes | yes | `budget_bytes`; using it without `budget_bytes` is an error |

Refused when the arm file is loaded, before anything runs: an unknown
placeholder, a placeholder that has no value in its phase, an empty command,
a fit command without `{packet}`, a sample command without `{out}`.

Seeds use the stream formula of formats.md section 4 and keep the low 31 bits:
`{seed} = crn_seed(panel_sha256, dataset_hash, data_seed, sample_seed, "ext-sample") mod 2^31`
and `{fit_seed} = crn_seed(panel_sha256, dataset_hash, data_seed, 0, "ext-fit") mod 2^31`.
Neither depends on the view or the multiplier, so the two files of a sample
seed share their seed.

Every command runs with:

- stdin closed; stdout and stderr appended to `<run_dir>/<arm_id>/_logs/<cell>.log`,
  between marker lines that name the command and its exit status. The log is
  never parsed and never copied; it is replaced when the cell is run again;
- the environment `PATH` and `HOME` (inherited when set), `LANG=C`,
  `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=` (empty)
  and nothing else;
- a scratch working directory that is deleted after the cell, with whatever
  the commands left in it;
- its own process group. At the timeout the whole group is killed.

## 3. What happens per cell

`<cell>` is the `cell_id` with `:` replaced by `__`. For a cell with schedule
status `ok`:

1. Earlier results of the cell (log, packet, card, matrices) are removed.
2. The A rows are written to `{fit_rows}` under
   `<run_dir>/<arm_id>/_work/<cell>/`, readable by the current user only.
3. The fit command runs; `encode_ms` is its wall-clock time. `{fit_rows}` is
   deleted as soon as the command returns, whatever its outcome.
4. The fit succeeded when it exited with status 0 and `{packet}` is a regular
   file. Otherwise the packet and the card are removed and every file of the
   cell gets a `failed` row.
5. For a `source_free` arm, `packet_bytes` is the length of the packet plus
   the length of the card when the fit created one, and `packet_sha256` is
   the SHA-256 of the packet bytes followed by the card bytes. For
   `rows_at_sampler` and `control` arms both stay `0` and `""`, as formats.md
   section 6 requires: their state is not an encoding. Packet and card are
   kept under `packets/` in every case.
6. For each sample seed and each multiplier, in schedule order, the sample
   command runs (`decode_ms` is its wall-clock time), `{out}` is checked
   (section 4), written as
   `<run_dir>/<arm_id>/<cell>.s<sample_seed>.m<multiplier>.mat` and deleted.
   A file that fails does not stop the other files of the cell.

A cell with schedule status `unsupported` runs no command and gets
`unsupported` rows. Both times cover the whole process, including its start-up
and its own reading and writing of files; short commands are timed to about a
millisecond, long ones to about 10 ms.

## 4. Row files

Columns are the cell's `cols` in order, then the target. Units are the raw
table units of the arena.

**`io = "csv"`**

- Header `c0,c1,...,c{p-1},target`; `c<j>` is the j-th column of the view,
  not an arena column index. Lines end with `\n`; nothing is quoted.
- A float is printed in the shortest decimal form that reads back to the same
  float32 (`1`, `-0`, `0.1`, `123456.79`, `340282350000000000000000000000000000000`);
  no exponent is used. NaN is an empty field; infinities are `inf` and `-inf`.
- A regression target is printed as a float; a classification target as the
  integer class index `0..K-1`. Not every class has to occur in the A rows.

Reading `{out}` back:

- A header is required and columns are matched **by name**, in any order.
  Every name `c0..c{p-1}` and `target` must occur exactly once; any other
  name is refused. Names and fields are trimmed; quoted fields are accepted.
- An empty feature field is NaN. Other fields must parse as a decimal float
  (`nan`, `inf` and exponents are accepted; `NA`, `null` and the like are not).
  Every NaN reads back as the same quiet NaN.
- The target must not be empty, NaN or infinite. A classification target must
  be an integer in `0..K-1` (`1` and `1.0` are both accepted).
- The file must hold exactly `{rows}` records.

**`io = "mat"`**: `{fit_rows}` and `{out}` are `DOPEMAT1` files (formats.md
section 5) with `x_cols = p`, `y_cols = 1` and class indices stored as floats.
The same row count and target rules apply; values are copied bit for bit.

## 5. Outputs

```
<run_dir>/<arm_id>/emit.jsonl           one row per scheduled (cell, sample seed, multiplier)
<run_dir>/<arm_id>/emit-summary.json    the summary that is also printed on stdout
<run_dir>/<arm_id>/<cell>.s<S>.m<M>.mat DOPEMAT1, only for ok rows
<run_dir>/<arm_id>/packets/<cell>.packet, .card
<run_dir>/<arm_id>/_logs/<cell>.log
```

`emit.jsonl` holds this arm only, sorted by (`cell_id`, `sample_seed`,
`multiplier`) with sorted keys. Its row count always equals the sum over the
cells of the run of `len(sample_seeds) * len(multipliers)`. Arms do not share
files, so several arms can be emitted into one run directory at the same time.
Rows are self-describing (`arm_id`, and `file` relative to the run directory),
so a reader that wants one log for the whole run directory can concatenate the
per-arm files. Rows carry every field of formats.md section 6:

| Field | Value for an external arm |
|---|---|
| `arch_id` | `ext-cli/v1` |
| `weights_sha256`, `cold_bytes` | SHA-256 and length of `binary` |
| `artifact_class` | from the arm file |
| `packet_bytes`, `packet_sha256` | section 3 step 5; `0` and `""` when the fit produced no packet |
| `status` | `ok`, `failed` or `unsupported` (`over_budget` is left to the scorer) |
| `reason` | `""` for `ok`, else a code of section 6 |
| `rows`, `file` | rows in the matrix and its path relative to the run directory; `0` and `""` unless `ok` |
| `encode_ms`, `decode_ms` | wall-clock milliseconds; the only fields that differ between identical runs |

For identical inputs and identical generator output, every matrix, every
packet and every other field of `emit.jsonl` is byte-identical for any
`--threads`.

The summary counts cells and rows (`cells_ok`, `cells_failed`,
`cells_unsupported`, `rows_ok`, `rows_failed`, `rows_unsupported`), failures
per reason (`rows_failed_by_reason`; `cells_failed_by_reason` by the first
failed file of a cell), `packet_bytes_median` and `packet_bytes_max` over
cells with a packet, `encode_ms_median` over those cells and
`decode_ms_median` over `ok` rows. `schedule_cells`, `cells` and `limit_cells`
show whether the run covered the whole schedule; `cells_kept` and `cells_run`
show what `--resume` did.

## 6. Reason codes

A reason is one of these constants. It never contains a cell value, a column
name, a path or anything a command printed.

| Code | Status | Meaning |
|---|---|---|
| `schedule_unsupported` | unsupported | the schedule marks the cell `unsupported` |
| `run_interrupted` | failed | the run stopped before the cell was finished (section 7); never present after a completed run |
| `fit_spawn` | failed | the fit program could not be started |
| `fit_exit` | failed | the fit exited with a non-zero status or was killed by a signal |
| `fit_timeout` | failed | the fit was still running at the timeout and was killed |
| `packet_missing` | failed | the fit exited with 0 but `{packet}` is not a regular file |
| `sample_spawn` | failed | the sample program could not be started |
| `sample_exit` | failed | the sample exited with a non-zero status or was killed by a signal |
| `sample_timeout` | failed | the sample was still running at the timeout and was killed |
| `output_missing` | failed | the sample exited with 0 but `{out}` is not a readable regular file |
| `output_too_large` | failed | `{out}` is longer than `4096 + 256 * rows * (p + 1)` bytes; it is not read |
| `mat_invalid` | failed | `io = "mat"`: `{out}` is not a well-formed `DOPEMAT1` file |
| `header_mismatch` | failed | CSV header with a missing, extra or repeated column; `DOPEMAT1` with `x_cols != p` |
| `parse_error` | failed | malformed CSV, a record of the wrong width, or a non-empty field that is not a number |
| `row_count_mismatch` | failed | the output does not hold exactly `{rows}` rows |
| `non_finite_target` | failed | a target is empty, NaN or infinite |
| `label_out_of_range` | failed | a classification target is not an integer in `0..K-1` |

`run_interrupted` and the four fit codes apply to every file of the cell; the
others to one file. When an output has several defects, the header is judged first,
then the first defective record in file order.

## 7. Resume

A plain run refuses a run directory in which the arm already has an
`emit.jsonl`. With `--resume` a cell is kept, and none of its commands run,
when all of its rows in the existing log are `ok`, were written by this arm
and this binary (`weights_sha256`), and every matrix is still on disk with the
right size. Every other cell is run again in full, fit included, so all files
of a cell always come from one packet. The log is rewritten for exactly the
cells of the current run; files of cells outside it (after a smaller
`--limit-cells`, or another schedule) stay on disk and are not listed.

The log is complete at every moment. Before the first command of a run it is
written with the rows of the kept and unsupported cells and, for every cell
that still has to run, `failed` rows with reason `run_interrupted`; the
finished log replaces it in one rename and the summary is written last. A run
that is killed therefore leaves a log that lists its unfinished cells as
`run_interrupted` and no summary; `--resume` runs those cells again. Cells
that finished inside a killed run are run again too.

## 8. Real rows

The only file with real rows that `emit-cli` writes is `{fit_rows}`. It exists
under `<run_dir>/<arm_id>/_work/` while the fit command runs and is removed
when the command returns, also after a failure or a timeout. `_work` is
emptied before the first command of a run, so rows left by a killed run do
not survive the next one; until then they are on disk and should be deleted
by hand when the arm is not run again. Messages, `emit.jsonl` and the summary
hold no cell values. Two things remain under the generator's control: what it
prints (kept in `_logs/`) and what it stores in its packet; a generator that
copies rows there must be declared `rows_at_sampler`.

## 9. First measurement of the previous release (2026-11-22)

Arm `dope-1.3.0-small` (signed binary, sha256 `9f8562a1…e84d`) on the first 40
cells of the lite schedule of panel `baseline-v250` (337 cells; schedule
sha256 `5d292ea5…`), 8 cells in parallel on a shared, loaded host:

- 35 cells `ok`, 4 `failed` with `fit_exit`, 1 `unsupported`; 70, 8 and 2 rows.
- All four failures are `full` views with 132, 153, 256 and 256 columns: the
  1.3.0 binary refuses to fit them into its 2 KiB cap, with or without its
  `--anonymize` option. The `p48` views of the same four tables are `ok`.
  The widest `full` view that fitted has 57 columns; 14 of the 328 supported
  lite cells are wider than that.
- Packets of the 35 fitted cells: median 1241 bytes, maximum 2045, all within
  2048.
- Fit time median about 1.0 s, 90th percentile about 3.7 s; sample time
  median about 11 ms per file. These are wall-clock times on a busy host and
  not a speed claim.
- Three runs gave byte-identical matrices and packets and the same log apart
  from the two time fields.

This is a check of the arm, not a score: nothing here measures fidelity.
