# Ravel JVM vs NumPy access-pattern benchmark

Semantic parity passed for every reported row. Timings compare public, allocating operations except for zero-copy view creation and scalar reads.

NumPy 2.4.3 on Python 3.14.3; Ravel timings are from the supplied JMH result.

| side | family | case | work | Ravel median ns/op | NumPy median ns/op | Ravel speed vs NumPy | Ravel ns/unit | NumPy ns/unit |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 256 | reduction | `full_sum_contiguous` | 65,536 element | 21,559.9 | 10,692.6 | 0.496x | 0.329 | 0.163 |
| 256 | reduction | `full_sum_inner_stride` | 32,768 element | 20,532.5 | 7,144.1 | 0.348x | 0.627 | 0.218 |
| 256 | reduction | `axis0_sum` | 65,536 element | 21,142.5 | 12,188.3 | 0.576x | 0.323 | 0.186 |
| 256 | reduction | `axis1_sum` | 65,536 element | 29,992.6 | 10,627.6 | 0.354x | 0.458 | 0.162 |
| 1024 | reduction | `full_sum_contiguous` | 1,048,576 element | 354,586.3 | 159,179.7 | 0.449x | 0.338 | 0.152 |
| 1024 | reduction | `full_sum_inner_stride` | 524,288 element | 315,816.0 | 104,842.6 | 0.332x | 0.602 | 0.200 |
| 1024 | reduction | `axis0_sum` | 1,048,576 element | 338,080.8 | 190,584.9 | 0.564x | 0.322 | 0.182 |
| 1024 | reduction | `axis1_sum` | 1,048,576 element | 490,082.4 | 167,906.0 | 0.343x | 0.467 | 0.160 |

A speed value above 1.0x favors Ravel; below 1.0x favors NumPy. Scalar-access rows include each host language's call/indexing overhead and are not native-kernel comparisons.

These cross-process timings are descriptive. Treat changes as regressions only against same-host baselines collected with the same runtimes and benchmark settings.
