# Contract files

The active contract source is `production/kpi-contract.json`. Change that
source and its typed validator together. `build.rs` generates tier constants
and embeds its digest; never edit Cargo `OUT_DIR` output. Keep the v1 historical
fixture read-only and run `cargo test --locked production::tests::embedded_contract_is_canonical_and_bound`
after a contract change.
