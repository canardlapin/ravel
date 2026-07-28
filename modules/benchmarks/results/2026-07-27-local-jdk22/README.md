# Local JDK 22 access-pattern baseline

This directory preserves the pre-optimization development baseline collected on
2026-07-27. It is suitable for same-host optimization decisions, but it is not
the JDK 21 release gate and was not collected on a dedicated quiet runner.

## Environment and protocol

- source revision: `473c33bb1e53ec83256935861639581b92d3fbcd`
- host: Apple arm64, macOS 14.3
- JVM: OpenJDK 22+36-2370
- JMH: 1.37, average time, 2 forks, 5 x 500 ms warmup, 7 x 500 ms measurement
- Python: 3.14.3
- NumPy: 2.4.3
- sizes: side 256 and 1024
- dtype: `float64` for cross-runtime cases

`ravel-signatures.json` and `numpy.json` establish semantic and C-layout
parity for every row in `comparison.md`. `ravel-jmh.json` contains the public
allocating-operation timings. `controls-jmh.json` contains allocation, raw
monomorphic, reusable-output, inline-dispatch, opaque-alias, copy, and
exact-schedule reduction controls with the JMH GC profiler enabled.

The 24 kernel rows, excluding scalar access and view creation, have a 0.176x
geometric-mean speed relative to NumPy. The worst row is side-256
`copy_inner_stride` at 0.026x; the best is side-1024 `transpose_add` at 0.653x.
These values are descriptive cross-process comparisons, not regression gates.

## Public benchmark variability

Coefficient of variation is the sample standard deviation divided by the mean
over all 14 measured JMH iterations.

| case | side 256 CV | side 1024 CV |
|---|---:|---:|
| `axis0_sum` | 0.64% | 2.97% |
| `axis1_sum` | 0.69% | 0.69% |
| `broadcast_row_add` | 0.61% | 0.53% |
| `contiguous_add` | 11.06% | 2.61% |
| `copy_inner_stride` | 1.92% | 12.27% |
| `copy_transpose` | 5.99% | 1.72% |
| `full_sum_contiguous` | 0.53% | 0.44% |
| `full_sum_inner_stride` | 1.16% | 0.83% |
| `inner_stride_add` | 3.59% | 1.71% |
| `outer_stride_add` | 0.75% | 2.02% |
| `reverse_add` | 0.96% | 1.23% |
| `scalar_read_column_major` | 0.87% | 1.56% |
| `scalar_read_row_major` | 1.75% | 1.51% |
| `transpose_add` | 0.63% | 3.25% |
| `view_inner_stride_create` | 1.04% | 0.91% |
| `view_transpose_create` | 2.94% | 1.31% |

The side-256 contiguous-add and side-1024 inner-stride-copy rows exceed 10% CV.
They remain useful directional baselines, but a performance claim involving
either row requires a repeated same-host run.

## Control findings

Times are JMH means. Allocation is normalized bytes per operation.

| control | side 256 time | side 256 CV | side 256 B/op | side 1024 time | side 1024 CV | side 1024 B/op |
|---|---:|---:|---:|---:|---:|---:|
| raw add, reuse | 10.72 us | 4.77% | 0.1 | 197.24 us | 3.65% | 2.7 |
| storage-dispatched add, reuse | 10.23 us | 1.03% | 0.1 | 191.65 us | 3.41% | 2.6 |
| inline storage add, reuse | 10.55 us | 4.12% | 0.1 | 211.19 us | 10.70% | 2.9 |
| opaque-buffer add, reuse | 12.41 us | 16.08% | 0.2 | 190.67 us | 3.54% | 2.6 |
| public `addInto`, reuse | 104.66 us | 0.72% | 953.4 | 2.129 ms | 1.60% | 2,022.5 |
| output allocation, full | 2.97 us | 1.63% | 524,320 | 41.59 us | 0.99% | 8,388,641 |
| raw add, allocate | 14.01 us | 1.01% | 524,304 | 243.92 us | 1.38% | 8,388,627 |
| raw inner-stride copy, allocate | 9.56 us | 0.61% | 262,160 | 217.40 us | 1.71% | 4,194,323 |
| raw transpose copy, allocate | 46.15 us | 0.79% | 524,305 | 1.010 ms | 1.60% | 8,388,638 |
| raw exact sum, reuse | 19.81 us | 1.66% | 0.3 | 316.77 us | 0.86% | 4.3 |
| raw exact sum, allocate scratch | 19.77 us | 1.96% | 4,112 | 313.95 us | 1.08% | 65,556 |
| raw axis-0 sum, reuse | 41.98 us | 0.72% | 0.6 | 912.23 us | 1.34% | 12.4 |
| raw axis-1 sum, reuse | 25.55 us | 1.14% | 0.3 | 347.25 us | 1.83% | 4.7 |
| raw float add, reuse | 7.09 us | 0.66% | 0.1 | 117.64 us | 0.74% | 1.6 |
| raw int add, reuse | 7.16 us | 1.43% | 0.1 | 119.51 us | 2.31% | 1.6 |

One-time inline matching reaches the existing storage-dispatched lower bound.
The opaque alias does not improve throughput and has high side-256 variance.
Neither should be treated as a substitute for dispatch-once monomorphic
kernels.

Output allocation is a small fraction of public add time. Public reusable
`addInto` remains roughly ten times slower than the raw/storage controls, so
planning and general traversal are the dominant elementwise costs.

Copy is the clearest first target. At side 256, raw inner-stride copy including
allocation is 9.56 us versus 362.56 us through the public path and 9.32 us in
NumPy. The gap is therefore generic traversal and per-element storage dispatch,
not raw JVM memory throughput.

The exact 128-element reduction schedule has a lower bound of 19.81 us at side
256 and 316.77 us at side 1024. Scratch allocation is not material. Further
full-reduction gains require instruction-level parallelism that preserves each
block's addition order, while axis-0 needs cache tiling that preserves each
fiber's order.
