# Ravel JVM vs NumPy access-pattern benchmark

Semantic parity passed for every reported row. Timings compare public, allocating operations except for zero-copy view creation and scalar reads.

NumPy 2.4.3 on Python 3.14.3; Ravel timings are from the supplied JMH result.

| side | family | case | work | Ravel median ns/op | NumPy median ns/op | Ravel speed vs NumPy | Ravel ns/unit | NumPy ns/unit |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 256 | elementwise | `contiguous_add` | 65,536 element | 19,924.0 | 10,688.2 | 0.536x | 0.304 | 0.163 |
| 256 | elementwise | `inner_stride_add` | 32,768 element | 17,100.4 | 10,480.2 | 0.613x | 0.522 | 0.320 |
| 256 | elementwise | `outer_stride_add` | 32,768 element | 20,031.2 | 18,476.1 | 0.922x | 0.611 | 0.564 |
| 256 | elementwise | `reverse_add` | 65,536 element | 38,763.1 | 34,239.6 | 0.883x | 0.591 | 0.522 |
| 256 | elementwise | `transpose_add` | 65,536 element | 76,837.4 | 89,133.2 | 1.160x | 1.172 | 1.360 |
| 256 | broadcast | `broadcast_row_add` | 65,536 element | 39,013.7 | 23,498.1 | 0.602x | 0.595 | 0.359 |
| 256 | reduction | `full_sum_contiguous` | 65,536 element | 20,141.4 | 9,113.4 | 0.452x | 0.307 | 0.139 |
| 256 | reduction | `full_sum_inner_stride` | 32,768 element | 18,341.4 | 6,000.4 | 0.327x | 0.560 | 0.183 |
| 256 | reduction | `axis0_sum` | 65,536 element | 19,625.2 | 10,688.0 | 0.545x | 0.299 | 0.163 |
| 256 | reduction | `axis1_sum` | 65,536 element | 28,878.7 | 11,154.0 | 0.386x | 0.441 | 0.170 |
| 256 | copy | `copy_inner_stride` | 32,768 element | 11,277.4 | 9,696.8 | 0.860x | 0.344 | 0.296 |
| 256 | copy | `copy_transpose` | 65,536 element | 26,789.0 | 41,461.2 | 1.548x | 0.409 | 0.633 |
| 256 | scalar access | `scalar_read_row_major` | 65,536 element access | 45,832.2 | 4,430,377.7 | 96.665x | 0.699 | 67.602 |
| 256 | scalar access | `scalar_read_column_major` | 65,536 element access | 45,870.0 | 4,432,472.5 | 96.631x | 0.700 | 67.634 |
| 256 | view creation | `view_inner_stride_create` | 1 view | 81.0 | 121.7 | 1.502x | 80.994 | 121.660 |
| 256 | view creation | `view_transpose_create` | 1 view | 60.5 | 67.4 | 1.115x | 60.480 | 67.416 |
| 1024 | elementwise | `contiguous_add` | 1,048,576 element | 307,329.8 | 762,727.0 | 2.482x | 0.293 | 0.727 |
| 1024 | elementwise | `inner_stride_add` | 524,288 element | 276,077.1 | 439,014.4 | 1.590x | 0.527 | 0.837 |
| 1024 | elementwise | `outer_stride_add` | 524,288 element | 331,128.0 | 564,560.8 | 1.705x | 0.632 | 1.077 |
| 1024 | elementwise | `reverse_add` | 1,048,576 element | 653,625.2 | 1,233,744.4 | 1.888x | 0.623 | 1.177 |
| 1024 | elementwise | `transpose_add` | 1,048,576 element | 3,119,249.9 | 3,053,291.7 | 0.979x | 2.975 | 2.912 |
| 1024 | broadcast | `broadcast_row_add` | 1,048,576 element | 621,813.8 | 854,125.0 | 1.374x | 0.593 | 0.815 |
| 1024 | reduction | `full_sum_contiguous` | 1,048,576 element | 325,873.2 | 138,095.2 | 0.424x | 0.311 | 0.132 |
| 1024 | reduction | `full_sum_inner_stride` | 524,288 element | 296,209.5 | 86,804.0 | 0.293x | 0.565 | 0.166 |
| 1024 | reduction | `axis0_sum` | 1,048,576 element | 324,593.9 | 154,332.0 | 0.475x | 0.310 | 0.147 |
| 1024 | reduction | `axis1_sum` | 1,048,576 element | 465,456.8 | 141,567.5 | 0.304x | 0.444 | 0.135 |
| 1024 | copy | `copy_inner_stride` | 524,288 element | 231,781.1 | 380,292.3 | 1.641x | 0.442 | 0.725 |
| 1024 | copy | `copy_transpose` | 1,048,576 element | 623,331.7 | 1,530,174.9 | 2.455x | 0.594 | 1.459 |
| 1024 | scalar access | `scalar_read_row_major` | 1,048,576 element access | 740,433.2 | 74,855,791.7 | 101.097x | 0.706 | 71.388 |
| 1024 | scalar access | `scalar_read_column_major` | 1,048,576 element access | 972,770.0 | 79,476,194.3 | 81.701x | 0.928 | 75.794 |
| 1024 | view creation | `view_inner_stride_create` | 1 view | 81.2 | 126.3 | 1.555x | 81.247 | 126.308 |
| 1024 | view creation | `view_transpose_create` | 1 view | 60.8 | 66.8 | 1.099x | 60.811 | 66.801 |

A speed value above 1.0x favors Ravel; below 1.0x favors NumPy. Scalar-access rows include each host language's call/indexing overhead and are not native-kernel comparisons.

These cross-process timings are descriptive. Treat changes as regressions only against same-host baselines collected with the same runtimes and benchmark settings.
