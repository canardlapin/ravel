# General packed-layout mutable traversal

This receipt covers a production optimization in `ravel-core`, not a
benchmark-specific branch.

`Layout` now computes a rank-general physical-density invariant once at
construction. A layout is physically dense when its reachable addresses form
one packed interval, independent of logical axis order or stride direction.
This includes canonical C order, Fortran order, axis permutations, reversals,
and singleton-axis variants. It excludes gaps, overlapping strides, and
broadcast axes.

Element-independent scalar mutation may traverse that interval in ascending
physical order. Paired assignment retains logical-order traversal because
source/destination alias semantics can depend on visit order.

## Correctness gate

The JVM and Scala.js suites passed the same checks:

- exact examples for packed permutations, reversals, and singleton axes;
- counterexamples for gaps, overlap, and broadcast;
- 250 generated transpose, reversal, and offset cases that verify each
  reachable element is visited exactly once; and
- all four scalar in-place operations for `Int`, `Long`, `Float`, and
  `Double`, including exact floating-point bits.

The complete core suites passed 131 tests on each platform, both reduction-law
suites passed, and optimized Scala.js linking passed.

## Timing and allocation

The focused reversed side-1024 public row used the same Homebrew OpenJDK
25.0.1 AArch64 host before and after the change:

| Receipt | ns/op | Relative to before |
|---|---:|---:|
| before | 337,757.501 | 1.00x |
| after | 142,125.585 | 2.38x |

The short allocation controls measured 327,064.624 ns/op and 35.678 B/op
before, versus 140,839.713 ns/op and 25.945 B/op after. Neither run recorded a
collection. The small normalized byte counts are constant per operation, not
proportional to the 1,048,576 elements.

The full two-fork, five-warmup, seven-measurement side-1024 witness exercises
six equivalent packed layouts:

| Layout | ns/op |
|---|---:|
| contiguous | 137,695.071 |
| transpose | 137,702.770 |
| reverse axis 0 | 137,783.131 |
| reverse axis 1 | 137,766.434 |
| reverse both axes | 138,057.595 |
| transpose and reverse | 138,217.249 |

The slowest is 0.38 percent above the fastest. This is the primary generality
witness: the optimization follows a reusable layout property rather than the
name or shape of one benchmark case.

Files named `reverse-*` are the focused before/after and GC controls.
`dense-layouts-after.json` is the cross-layout JMH receipt.

## Reuse by exact full reductions

The same invariant now selects physical-order loops for operations whose
contracts permit regrouping:

- `Int` and `Long` sum and product, which are associative modulo their fixed
  width; and
- full `Double` minimum and maximum, whose NaN propagation and signed-zero
  result are invariant under regrouping. NaN payload identity is not public
  API.

Floating sums and products remain on their documented logical schedules.

The rank-3 full-protocol witnesses use 1,048,576 elements:

| Layout | `Double.maximum` ns/op | `Int.sum` ns/op |
|---|---:|---:|
| contiguous | 172,028.849 | 140,683.116 |
| permute | 172,270.382 | 136,231.583 |
| reverse | 172,210.656 | 136,307.531 |
| permute and reverse | 174,435.473 | 135,777.192 |

Maximum spans 1.4 percent across all four layouts. The three transformed
`Int.sum` rows span 0.4 percent. The contiguous sum row retained one 165
microsecond outlier and has an 8.7 microsecond 99.9 percent error, so it is not
evidence that transformed layouts are faster; its interval overlaps the other
rows.

`dense-reductions-maximum.json` and `dense-reductions-sum-int.json` contain the
complete fork and iteration data. Rank-3 JVM and Scala.js laws separately
cover minimum, maximum, both fixed-width dtypes, sum and product, permutations,
reversals, NaN, signed zero, overflow, and tails.
