# Stage 2 exact-ILP and mutable-kernel receipt (JDK 25)

This is a same-host diagnostic receipt for the second post-closeout
performance stage. It is not a cross-host release baseline. The run used
OpenJDK 25.0.1 on macOS/aarch64, JMH 1.37, and the NumPy 2.4.3 timings and
semantic signatures from the adjacent JDK 25 operation-matrix receipt.

The public numerical contract is unchanged. `ReductionLawsSuite` compares the
new full and axis-1 paths to the fixed 128-value block-pairwise reference by
raw floating-point bits. `numpy-comparison.md` was rendered only after the
saved Ravel and NumPy signatures passed the operation-matrix parity check.

## Exact reduction controls

The ILP control preserves order inside every 128-value block and preserves the
same scratch-block merge tree. Ratios below use median JMH iteration time:

| input side | full serial / exact ILP | axis-1 serial / exact ILP |
|---:|---:|---:|
| 256 | 1.99x | 2.25x |
| 1024 | 1.59x | 1.83x |

Against the earlier same-host public receipt, the exact public full sum is
1.78x faster at side 256 and 1.71x faster at side 1024. Public axis-1 sum is
1.53x and 1.88x faster, respectively.

## Mutable dispatch removal

The final targeted public timings use primitive dtype/operation dispatch
outside the element loop:

| case, side 1024 | before | after | speedup | speed vs NumPy |
|---|---:|---:|---:|---:|
| contiguous `Double` add | 4,787,767 ns | 134,956 ns | 35.5x | 1.110x |
| reversed `Double` add | 13,583,049 ns | 293,983 ns | 46.2x | 0.518x |
| inner-stride `Double` multiply | 6,354,125 ns | 154,429 ns | 41.1x | 1.360x |

Across all 18 targeted in-place rows, the Ravel/NumPy geometric mean moved
from 0.027x to 0.990x. Reversed traversal is the remaining layout-specific
loss; contiguous and inner-stride rows are at or above parity except
contiguous `Double` division at side 1024.

The GC receipt records approximately 24-64 bytes per operation for the
selected mutable cases, with no collection and no element-count-proportional
growth. The pre-change side-256 `Double` add allocated about 3.15 MiB/op,
approximately 48 bytes per element.

## Artifacts

- `reduction-controls.json`: serial and four-way exact-schedule controls.
- `public-reductions.json`: public full and axis-1 exact sums.
- `operation-matrix.json`: targeted contiguous dtype/operation cases.
- `rank2-final.json`: uncontaminated final rank-2 mutable cases.
- `final-operation-matrix.json`: mechanical merge used for comparison.
- `mutable-gc.json`: selected post-change allocation receipt.
- `numpy-comparison.md`: 20-row parity-gated timing report.
- `rank2-operation-matrix.json`: retained noisy exploratory run, superseded by
  `rank2-final.json` for headline timing.

The JMH commands used average-time mode and wrote JSON with `-rf json -rff`.
The reduction controls selected
`AccessPatternControls.(raw_exact_sum_reuse|raw_exact_sum_ilp_reuse|raw_axis1_sum_reuse|raw_axis1_sum_ilp_reuse)`;
the public run selected
`AccessPatternBenchmarks.(full_sum_contiguous|axis1_sum)`. The targeted matrix
selected the cases named in `final-operation-matrix.json` at sides 256 and
1024. See `docs/numpy-benchmarks.md` for the reproducible parity and comparison
commands.
