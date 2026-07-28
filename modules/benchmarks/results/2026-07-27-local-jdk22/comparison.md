# Ravel JVM vs NumPy access-pattern benchmark

Semantic parity passed for every reported row. Timings compare public, allocating operations except for zero-copy view creation and scalar reads.

NumPy 2.4.3 on Python 3.14.3; Ravel timings are from the supplied JMH result.

| side | family | case | work | Ravel median ns/op | NumPy median ns/op | Ravel speed vs NumPy | Ravel ns/unit | NumPy ns/unit |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 256 | elementwise | `contiguous_add` | 65,536 element | 126,441.1 | 10,836.8 | 0.086x | 1.929 | 0.165 |
| 256 | elementwise | `inner_stride_add` | 32,768 element | 62,940.3 | 10,412.3 | 0.165x | 1.921 | 0.318 |
| 256 | elementwise | `outer_stride_add` | 32,768 element | 71,407.7 | 18,606.6 | 0.261x | 2.179 | 0.568 |
| 256 | elementwise | `reverse_add` | 65,536 element | 143,572.4 | 34,288.4 | 0.239x | 2.191 | 0.523 |
| 256 | elementwise | `transpose_add` | 65,536 element | 148,524.1 | 90,049.2 | 0.606x | 2.266 | 1.374 |
| 256 | broadcast | `broadcast_row_add` | 65,536 element | 143,975.7 | 22,199.0 | 0.154x | 2.197 | 0.339 |
| 256 | reduction | `full_sum_contiguous` | 65,536 element | 130,500.0 | 9,080.1 | 0.070x | 1.991 | 0.139 |
| 256 | reduction | `full_sum_inner_stride` | 32,768 element | 60,844.9 | 6,026.8 | 0.099x | 1.857 | 0.184 |
| 256 | reduction | `axis0_sum` | 65,536 element | 46,147.4 | 10,459.4 | 0.227x | 0.704 | 0.160 |
| 256 | reduction | `axis1_sum` | 65,536 element | 32,188.4 | 10,800.4 | 0.336x | 0.491 | 0.165 |
| 256 | copy | `copy_inner_stride` | 32,768 element | 361,560.1 | 9,318.1 | 0.026x | 11.034 | 0.284 |
| 256 | copy | `copy_transpose` | 65,536 element | 622,651.6 | 36,343.6 | 0.058x | 9.501 | 0.555 |
| 256 | scalar access | `scalar_read_row_major` | 65,536 element access | 1,001,573.3 | 4,326,854.6 | 4.320x | 15.283 | 66.023 |
| 256 | scalar access | `scalar_read_column_major` | 65,536 element access | 990,052.2 | 4,353,827.1 | 4.398x | 15.107 | 66.434 |
| 256 | view creation | `view_inner_stride_create` | 1 view | 84.9 | 120.4 | 1.418x | 84.892 | 120.350 |
| 256 | view creation | `view_transpose_create` | 1 view | 68.7 | 66.6 | 0.970x | 68.685 | 66.644 |
| 1024 | elementwise | `contiguous_add` | 1,048,576 element | 1,757,802.0 | 679,472.3 | 0.387x | 1.676 | 0.648 |
| 1024 | elementwise | `inner_stride_add` | 524,288 element | 1,065,280.7 | 347,285.3 | 0.326x | 2.032 | 0.662 |
| 1024 | elementwise | `outer_stride_add` | 524,288 element | 1,174,086.9 | 541,215.6 | 0.461x | 2.239 | 1.032 |
| 1024 | elementwise | `reverse_add` | 1,048,576 element | 2,327,328.3 | 1,140,082.1 | 0.490x | 2.220 | 1.087 |
| 1024 | elementwise | `transpose_add` | 1,048,576 element | 4,164,676.9 | 2,719,081.1 | 0.653x | 3.972 | 2.593 |
| 1024 | broadcast | `broadcast_row_add` | 1,048,576 element | 2,317,229.5 | 844,063.6 | 0.364x | 2.210 | 0.805 |
| 1024 | reduction | `full_sum_contiguous` | 1,048,576 element | 2,059,110.8 | 131,939.5 | 0.064x | 1.964 | 0.126 |
| 1024 | reduction | `full_sum_inner_stride` | 524,288 element | 1,039,037.5 | 84,256.7 | 0.081x | 1.982 | 0.161 |
| 1024 | reduction | `axis0_sum` | 1,048,576 element | 957,822.9 | 149,302.3 | 0.156x | 0.913 | 0.142 |
| 1024 | reduction | `axis1_sum` | 1,048,576 element | 537,457.2 | 132,005.5 | 0.246x | 0.513 | 0.126 |
| 1024 | copy | `copy_inner_stride` | 524,288 element | 6,534,450.1 | 299,427.3 | 0.046x | 12.463 | 0.571 |
| 1024 | copy | `copy_transpose` | 1,048,576 element | 7,225,723.2 | 1,386,241.4 | 0.192x | 6.891 | 1.322 |
| 1024 | scalar access | `scalar_read_row_major` | 1,048,576 element access | 15,719,743.7 | 73,582,319.7 | 4.681x | 14.992 | 70.174 |
| 1024 | scalar access | `scalar_read_column_major` | 1,048,576 element access | 16,057,336.6 | 77,439,653.0 | 4.823x | 15.313 | 73.852 |
| 1024 | view creation | `view_inner_stride_create` | 1 view | 84.7 | 121.8 | 1.439x | 84.676 | 121.838 |
| 1024 | view creation | `view_transpose_create` | 1 view | 66.9 | 65.9 | 0.985x | 66.921 | 65.912 |

A speed value above 1.0x favors Ravel; below 1.0x favors NumPy. Scalar-access rows include each host language's call/indexing overhead and are not native-kernel comparisons.

These cross-process timings are descriptive. Treat changes as regressions only against same-host baselines collected with the same runtimes and benchmark settings.
