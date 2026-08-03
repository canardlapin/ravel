# Failures and recovery

Ravel uses two failure styles. Programming errors and invalid operations on an
already constructed core array throw typed exceptions. Boundary validations
that callers commonly need to branch on return `Either`.

## Prefer checked boundary constructors

Dynamic shapes and rank refinement return their error values:

```scala mdoc:silent
import ravel.*

val shapeResult: Either[InvalidShape, Shape[AnyRank]] =
  Shape.from(Seq(2, -1))

val dynamic: AnyNDArray[Double] = NDArray.zeros[Double](2, 3)
val rankResult: Either[RankMismatch, Array3[Double]] =
  dynamic.requireRank[3]
```

```scala mdoc
shapeResult
rankResult
```

Checked numerical conversion also returns `Either` and validates the complete
logical input before allocating an output buffer when overflow is rejected:

```scala mdoc:silent
val measurements =
  NDArray.fromSeq(Shape(3), Seq(1.2, Double.NaN, 300.0))
```

```scala mdoc
measurements.convert[Byte]()
measurements.convert[Byte](
  ConversionPolicy(Rounding.NearestEven, Overflow.Clamp)
)
```

Use `cast` only when Scala/JVM primitive conversion behavior—including
truncation and clamping before narrow integer conversion—is the intended
contract. Use `convert` when rounding and overflow policy are part of the
scientific meaning.

## Recognize thrown core failures

| Failure | Typical cause | Recovery |
|---|---|---|
| `InvalidShape` | negative dimension or portable `Int` element-count overflow | validate with `Shape.from` at external boundaries |
| `ShapeMismatch` | wrong constructor value count or exact-shape operation | compare shapes and required element counts |
| `BroadcastMismatch` | trailing dimensions are neither equal nor `1` | insert an axis or reshape the intended operand |
| `InvalidAxis` | axis is outside the rank after negative-axis normalization | inspect `rank`; use axes in `[-rank, rank)` |
| `InvalidIndex` | wrong index arity or coordinate outside an axis | use fixed-rank `apply` or supply exactly `rank` dynamic indices |
| `InvalidSlice` | zero step or a resolved endpoint outside the axis | use `Slice` helpers or `narrow` for strict ranges |
| `NonContiguousLayout` | `reshapeView` would require data movement | choose `reshape` or `reshapeCopy` explicitly |
| `EmptyReduction` | min, max, or arg reduction over an empty domain | handle emptiness before reducing |
| `BuilderClosed` | retained `ArrayBuilder` used after its callback | keep all writes inside `NDArray.build` |

These classes extend `IllegalArgumentException` except `BuilderClosed`, which
extends `IllegalStateException`. Catch a specific Ravel type only when the
caller can make a specific recovery; otherwise prevent the invalid operation at
the boundary.

## Account for module-specific failures

`ravel-packed` returns `Either[PackedError, ...]` from constructors, views, and
set algebra. Its direct indexing methods use `require` and therefore throw
`IllegalArgumentException` for bad indices. Keep packed construction in the
typed `Either` path and validate indices before hot loops.

`ravel-stencil` validates `NeighborhoodSpec`, ranks, destination shapes, and
prepared-executor compatibility by throwing `IllegalArgumentException`. It
does not yet expose a typed stencil-error family. Treat specification creation
as a checked pipeline boundary rather than rebuilding it for every sample.

For allocation behavior after recovery, see
[Copy and view behavior](copy-view-table.md).
