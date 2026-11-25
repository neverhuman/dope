# UCI Airfoil Self-Noise Dataset — License & Provenance

- **Dataset:** UCI Airfoil Self-Noise (Dataset ID 291)
- **Authors:** Thomas F. Brooks, D. Stuart Pope, and Michael A. Marcolini (NASA, 1989)
- **DOI:** `10.24432/C5VW2C`
- **Source URL:** `https://archive.ics.uci.edu/dataset/291/airfoil+self+noise`
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Official archive SHA-256 (`airfoil+self+noise.zip`):** `5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c`
- **Official data file SHA-256 (`airfoil_self_noise.dat`, 59,984 bytes, 1,503 rows):** `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3`

## Preprocessing & Group-Disjoint Split

`airfoil_train.csv` (1,190 rows) and `airfoil_test.csv` (313 rows) are prepared from the
official 1,503-row `.dat` file under the DOPE target-last contract (`docs/data-contract.md`):

- **Headers (target last):** `frequency_hz,attack_angle_deg,chord_length_m,free_stream_velocity_ms,displacement_thickness_m,sound_pressure_db`
- **Configuration-grouped 80/20 split (seed `1729`):** Rows are grouped by the four physical
  airfoil configuration parameters (`attack_angle_deg`, `chord_length_m`,
  `free_stream_velocity_ms`, `displacement_thickness_m` — 106 unique configurations),
  excluding `frequency_hz` so repeated frequency sweeps of the same airfoil configuration
  never leak across train and test splits.
- **Scope:** Public engineering demonstration only; not part of the benchmark confirmation panel.
