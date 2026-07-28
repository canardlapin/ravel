# Copy-kernel optimization result

This directory contains the two-fork JDK 22 validation of the dispatch-once
logical copy kernel against the frozen
`../2026-07-27-local-jdk22/ravel-jmh.json` baseline. The host and JMH protocol
match that baseline.

| case | side | baseline | optimized | speedup | Ravel speed vs NumPy | CV |
|---|---:|---:|---:|---:|---:|---:|
| `copy_inner_stride` | 256 | 362.56 us | 10.99 us | 33.00x | 0.848x | 1.20% |
| `copy_inner_stride` | 1024 | 6.487 ms | 226.39 us | 28.65x | 1.323x | 0.72% |
| `copy_transpose` | 256 | 625.00 us | 26.09 us | 23.96x | 1.393x | 2.91% |
| `copy_transpose` | 1024 | 7.277 ms | 572.72 us | 12.71x | 2.420x | 2.71% |

`ravel-jmh-gc.json` verifies normalized allocation:

| case | side | output bytes | measured B/op | overhead |
|---|---:|---:|---:|---:|
| `copy_inner_stride` | 256 | 262,144 | 262,928.2 | 784.2 |
| `copy_inner_stride` | 1024 | 4,194,304 | 4,195,091.1 | 787.1 |
| `copy_transpose` | 256 | 524,288 | 525,072.4 | 784.4 |
| `copy_transpose` | 1024 | 8,388,608 | 8,389,818.2 | 1,210.2 |

Every case is below the output-buffer-plus-2,048-byte allocation budget. Every
case exceeds the 10x same-host speedup requirement and the 0.25x NumPy floor.

The implementation matches the sealed storage family once and calls a small
monomorphic dtype method. Canonical layouts use the platform bulk-copy
primitive. Rank-one and ordinary rank-two layouts use direct loops.
Transpose-like rank-two layouts use an unrolled 8x8 cache tile, and higher
ranks use one counter array with direct typed reads and writes.

Code-generation checks were made against the compiled artifacts:

- JVM `javap -c -p ravel.internal.CopyKernels$` contains primitive array
  load/store instructions and no `Function`, `BoxesRunTime`, callback
  `invokeinterface`, or Scala reference-wrapper calls in the copy paths.
- optimized Scala.js test output retains separate dtype methods; the Double
  path is direct `Float64Array` indexing and contains no per-element function
  allocation or callback invocation.

Correctness was checked by `CopyKernelsSuite` on JVM and Scala.js for all seven
storage dtypes and for contiguous, transposed, reversed, sliced, broadcast,
scalar, empty, mutable, and general-rank layouts.
