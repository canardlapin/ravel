# Residual public-operation optimization receipt

This directory records a same-host before/after court for the residual
operation-matrix bottlenecks tracked by `bd-01KYNW7FBJ1ZQJRQZ771J9V4HW`.
The baseline is commit `942ce831979d3d487bc569ed5c24b02ce65fc922`;
the candidate is the uncommitted working-tree implementation based on that
commit.

- Host: macOS 14.3, Apple silicon (`arm64`)
- JVM: Homebrew OpenJDK 25.0.1
- Build: Scala 3.7.4, JMH 1.37
- Workload: public operations over side-1024 arrays
- Protocol: two forks, five 500 ms warmups, seven 500 ms measurements
- Unit: microseconds per public operation, including required result allocation

The baseline was built from a Git archive of the exact commit so sbt-git did
not confuse a linked worktree with a bare repository. Both sides used explicit
JMH protocol overrides. Lower is better.

| Case | Baseline us/op | Candidate us/op | Speedup | Candidate relative error |
|---|---:|---:|---:|---:|
| `full_min_double` | 587.7 | 219.1 | **2.68x** | 0.9% |
| `full_max_double` | 594.8 | 236.8 | **2.51x** | 13.9% |
| `full_sum_int` | 295.3 | 135.8 | **2.17x** | 0.4% |
| `full_product_int` | 891.7 | 147.7 | **6.04x** | 0.7% |
| `full_sum_long` | 297.8 | 139.4 | **2.14x** | 0.4% |
| `full_product_long` | 890.0 | 149.4 | **5.96x** | 0.6% |
| `axis0_product_double` | 1471.3 | 298.6 | **4.93x** | 1.3% |
| `axis0_min_double` | 1656.2 | 302.9 | **5.47x** | 8.2% |
| `axis1_max_double` | 560.4 | 242.0 | **2.32x** | 2.2% |
| `axis0_mean_double` | 1360.2 | 313.1 | **4.34x** | 0.7% |
| `axis1_mean_double` | 478.2 | 262.0 | **1.83x** | 1.2% |
| `transpose_max_double` | 1410.3 | 224.5 | **6.28x** | 1.0% |
| `full_mean_float` | 531.0 | 197.3 | **2.69x** | 1.2% |
| `sum_as_double_float` | 470.7 | 202.3 | **2.33x** | 0.5% |
| `cast_double_int` | 1237.0 | 213.7 | **5.79x** | 3.3% |

The geometric-mean speedup across these 15 targeted rows is 3.47x. The
`full_max_double` candidate still contains one interrupted iteration and has a
wide interval; it is reported, not hidden. The baseline `axis0_min_double`,
`full_mean_float`, and `cast_double_int` rows also have wide intervals. Their
raw samples and confidence intervals remain in the JSON artifacts.

## Allocation evidence

The GC-profiled side-1024 controls show:

| Case | Time | Normalized allocation | GC during measurement |
|---|---:|---:|---:|
| `axis0_mean_double` | 312.0 us/op | 8,938 B/op | 0 |
| `full_mean_float` | 201.7 us/op | 18.8 B/op | 0 |
| `full_product_int` | 232.3 us/op | 3.2 B/op | 0 |
| `cast_double_int` | 199.2 us/op | 4,195,041 B/op | 27 collections |

Axis-0 mean now allocates approximately its required 8 KiB result plus fixed
overhead instead of allocating size-dependent pairwise scratch on every call.
The cast allocation is its required one-million-element Int result. The product
GC row predates the retained eight-lane refinement; that refinement changes
only scalar accumulators and does not introduce an allocation site.

## Correctness court

- NumPy parity: 32/32 access-pattern signatures and 158/158 operation-matrix
  signatures matched.
- JVM core: 128/128 tests passed.
- Scala.js core: 128/128 tests passed.
- JVM and Scala.js reusable laws: 2/2 on each platform.
- Full Scala.js optimized linking passed for core and laws.
- Formatting passed; MiMa ran as configured and reported no previous artifacts
  to compare.
- Added raw-bit mean-schedule checks, NaN/signed-zero extrema checks, and
  integer tail/wraparound checks through length 35.

This is a developer-workstation optimization receipt, not quiet-runner or
release evidence.

## Artifacts

- `baseline-jmh.json`: pre-change side-1024 receipt
- `focused-jmh.json`: candidate sides 256 and 1024 court
- `rerun-noisy-jmh.json`: focused rerun of two noisy candidate rows
- `eight-lane-sums-jmh.json`: retained final Int/Long sum refinement
- `eight-lane-products-jmh.json`: retained final Int/Long product refinement
- `focused-gc.json`: selected allocation controls
