# Contract files

The active contract source is `production/kpi-contract.json`. Change that
source and its typed validator together. `build.rs` generates tier constants
and embeds its digest; never edit Cargo `OUT_DIR` output. Keep the v1 historical
fixture read-only and run `cargo test --locked production::tests::embedded_contract_is_canonical_and_bound`
after a contract change. `production/kpi-contract-v3.json` pre-registers
MFS-v3 beside that pin. It is not the active embedded contract, and a v3
scalar stays null until its components and hard gates are measured.
