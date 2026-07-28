# Elementwise optimization results

This directory records the final local JDK 22 measurements for the
elementwise-add optimization tranche. The implementation dispatches once by
storage type, uses monomorphic loops for contiguous and coalesced rank-one
layouts, and retains the general broadcast and strided fallback.

These are same-host development results. They are not the JDK 21 release gate
and were not collected on a dedicated quiet runner.

## Protocol

- Host: Apple arm64, macOS 14.3
- JVM: OpenJDK 22+36-2370
- JMH: 1.37, average time, 2 forks
- Warmup: 5 iterations of 500 ms
- Measurement: 7 iterations of 500 ms
- Python: 3.14.3
- NumPy: 2.4.3
- Data type: `float64`

`ravel-jmh-final.json` contains the final public-operation timings.
`ravel-jmh-gc.json` contains the contiguous-add and reusable-output allocation
profiles. `transpose-repeat.json` records the isolated repeat of the noisy
1024-side transpose row. The earlier `ravel-jmh.json` file is an intermediate
measurement taken before the final zero-offset contiguous fast path.

## Public add results

The ratio is NumPy time divided by Ravel time, so values above 1.0 favor Ravel.
Across all 12 rows, the geometric-mean ratio is 1.107x. The worst row is the
side-256 broadcast add at 0.578x. The best row is the side-1024 contiguous add
at 2.749x.

| case | side | Ravel | NumPy | ratio | Ravel CV |
|---|---:|---:|---:|---:|---:|
| broadcast row | 256 | 38.40 us | 22.20 us | 0.578x | 0.75% |
| broadcast row | 1024 | 625.10 us | 844.06 us | 1.350x | 0.71% |
| contiguous | 256 | 14.54 us | 10.84 us | 0.745x | 0.84% |
| contiguous | 1024 | 247.13 us | 679.47 us | 2.749x | 1.43% |
| inner stride | 256 | 16.85 us | 10.41 us | 0.618x | 1.17% |
| inner stride | 1024 | 265.50 us | 347.29 us | 1.308x | 1.18% |
| outer stride | 256 | 19.42 us | 18.61 us | 0.958x | 1.37% |
| outer stride | 1024 | 316.67 us | 541.22 us | 1.709x | 0.85% |
| reverse | 256 | 38.29 us | 34.29 us | 0.895x | 1.53% |
| reverse | 1024 | 634.38 us | 1140.08 us | 1.797x | 0.99% |
| transpose | 256 | 75.59 us | 90.05 us | 1.191x | 0.46% |
| transpose | 1024 | 3276.94 us | 2719.08 us | 0.830x | 24.35% |

The side-1024 transpose row is not stable enough for a precise regression
claim. Its isolated repeat also split by fork and had 21.18% CV. Both runs
remain above the tranche's 0.30x worst-row threshold, but a quiet-runner
release gate must replace this local evidence.

Relative to the original same-host baseline, the 12-row geometric-mean
speedup is 3.68x. Individual improvements range from 1.27x for the noisy
side-1024 transpose row to 8.63x for side-256 contiguous add.

## Allocation and raw controls

At side 256, public contiguous add allocates 525,112 B/op. At side 1024, it
allocates 8,389,435 B/op. Both are about 808 bytes above the raw output-array
control, within the 2,048-byte acceptance bound.

Reusable `addInto` takes 10.49 us at side 256 and 209.13 us at side 1024. The
corresponding raw reusable control from the immediately preceding same-host run
takes 10.97 us and 228.14 us. The remaining 152-155 B/op in the `addInto`
benchmark comes from reading its rank-two result through the scalar indexing
API; that API is measured and optimized in the next tranche.

## Scala.js structure

The optimized Scala.js output contains direct `Float64Array` loops for
contiguous, offset-contiguous, and strided addition. The generated loops use
indexed loads and stores and do not call collection `map` or `forEach`
callbacks.
