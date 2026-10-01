# News DOPE target-weight confirmation

A frozen seed-23 GPU fit tested target weight 4 after a seed-11 validation lead. The earlier target-weight-2 q10 fit used the same candidate, binary, projection, and fit seed. The two model files have the same SHA-256 digest. Their raw charged artifacts are both 13,713 bytes; the existing lossless packed q10 artifact is 9,205 bytes.

| Target weight | Fit seconds | Peak GPU MiB | Raw charged bytes |
|---:|---:|---:|---:|
| 2 | 10.23 | 616 | 13,713 |
| 4 | 12.40 | 460 | 13,713 |

The first confirmation attempt was blocked at admission before GPU fitting; its receipt and the corrected round source are retained. Six duplicate n/4n samples were skipped because the model bytes are identical. The [existing q10 validation comparison](pilot24-news-q10-packed.md) supplies the shared outcomes; this confirmation adds no independent DOPE result. Official tests remain sealed, and MFS-v2, PTF-v1, and production certification remain null.
