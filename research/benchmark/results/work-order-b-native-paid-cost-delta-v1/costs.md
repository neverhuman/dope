# Closed native operation costs

| Operation | Status | Parent seconds | Charged bytes | Co-tenant | Observed own GPU MiB |
| --- | --- | ---: | ---: | --- | ---: |
| paid_operation_1 | infrastructure_failure | 215.101700 | unknown | true | 0 |
| paid_operation_2 | infrastructure_failure | 121.132676 | unknown | unknown | unknown |
| paid_operation_3 | ok | 107.422030 | 2910 | false | 430 |
| paid_operation_4 | ok | 107.947028 | 2903 | false | 430 |
| paid_operation_5 | infrastructure_failure | 148.388628 | unknown | true | 0 |

Five paid operations add **699.992063 s** to the fixed PR177 cutoff: **1624.422149 s**, **116 paid trials**.

Released sampler metadata hold: **1670.322090 s**, zero science and zero paid trials; excluded from native time.

Parent time includes guarded operation overhead. Nested timers are nonadditive. Residency is an observed maximum, not a true peak. Logical progress and gated scientific claims remain unavailable.
