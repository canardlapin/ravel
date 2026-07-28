# Stage 2 exact-ILP and mutable-kernel receipt (JDK 21)

This same-host baseline check used Eclipse Temurin 21.0.11+10-LTS on
macOS/aarch64 with JMH 1.37. It repeats the targeted Stage 2 controls and
public cases on Ravel's supported JDK baseline. NumPy 2.4.3 timings and the
JDK 21 semantic signatures come from the adjacent operation-matrix receipt.
The report is diagnostic rather than a cross-host release threshold.

## Exact reduction controls

The four-way controls retain the fixed 128-value block order and identical
merge tree:

| input side | full serial / exact ILP | axis-1 serial / exact ILP |
|---:|---:|---:|
| 256 | 1.73x | 2.34x |
| 1024 | 1.52x | 1.79x |

The public full sum measured 10,888 ns at side 256 and 180,206 ns at side
1024. Public axis-1 sum measured 18,646 ns and 251,370 ns, respectively.

## Mutable dispatch removal

| case, side 1024 | before | after | speedup | speed vs NumPy |
|---|---:|---:|---:|---:|
| contiguous `Double` add | 4,478,230 ns | 139,098 ns | 32.2x | 1.020x |
| reversed `Double` add | 13,430,405 ns | 329,432 ns | 40.8x | 0.428x |
| inner-stride `Double` multiply | 6,468,848 ns | 162,409 ns | 39.8x | 1.115x |

The 18-row in-place Ravel/NumPy geometric mean moved from 0.026x to 0.967x.
As on JDK 25, reversed traversal is the remaining layout-specific loss.

## Artifacts

- `reduction-controls.json`: serial and exact-ILP raw controls.
- `public-reductions.json`: public full and axis-1 exact sums.
- `operation-matrix.json`: targeted public mutable and `Float` sum cases.
- `numpy-comparison.md`: 20-row parity-gated timing report.

The JDK 21 commands used the Temurin installation as both `JAVA_HOME` and the
first `PATH` entry. JMH used average-time mode, two forks, and JSON output.
The benchmark selections match the JDK 25 receipt. See
`docs/numpy-benchmarks.md` for the semantic-gate and comparison commands.
