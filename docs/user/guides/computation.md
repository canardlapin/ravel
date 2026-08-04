# Numerical computation

## Choose a constructor

Use `zeros`, `fill`, `fromSeq`, or `tabulate` when the rank is at most four:

```scala mdoc:silent
import ravel.*

val zeros: Array2[Float] = NDArray.zeros[Float](2, 3)
val labels: Array1[Int] = NDArray.fromSeq(Shape(4), Seq(2, 4, 6, 8))
val grid: Array2[Double] =
  NDArray.tabulate(2, 3)((row, column) => row.toDouble + column / 10.0)
```

`fromSeq` requires exactly `shape.size` values. For dynamic or higher ranks,
construct a `Shape` explicitly and pass it to the shape-taking overloads.

## Broadcast trailing axes

Binary array operations align trailing axes. A dimension is compatible when
it is equal on both sides or one side is `1`.

```scala mdoc
val offsets: Array1[Double] =
  NDArray.fromSeq(Shape(3), Seq(10.0, 20.0, 30.0))

val shifted: Array2[Double] = grid + offsets
shifted
```

Use `newAxis` when the intended alignment is not a trailing axis:

```scala mdoc
val rowOffsets: Array1[Double] =
  NDArray.fromSeq(Shape(2), Seq(100.0, 200.0))

val shiftedRows: Array2[Double] = grid + rowOffsets.newAxis(1)
shiftedRows
```

`BroadcastMismatch` reports incompatible shapes. `zipMapExact` is the callback
variant that rejects broadcasting and requires equal shapes.

## Apply numerical operations

Arithmetic stays in the input dtype:

```scala mdoc
val scaled = (grid * 2.0) - 1.0
val bounded = scaled.clip(0.0, 2.0)
val positive = bounded >= 1.0

scaled
positive
```

The floating API also provides `sqrt`, `exp`, `log`, trigonometric functions,
rounding, `isNaN`, and `isFinite`. Casts are explicit:

```scala mdoc
val integral: Array2[Int] = bounded.cast[Int]
integral
```

## Reduce all elements or selected axes

Scalar reductions consume every logical element. Axis reductions remove that
axis from the static rank; `*Keep`/`*KeepDims` variants retain a singleton
axis for later broadcasting.

```scala mdoc
grid.sum
grid.mean
grid.sum(axis = 0)
grid.meanKeep(axis = 1).shape
```

Validate a dynamic axis set once with `Axes`. Sum, product, min, max, and mean
all accept the same value; their `*Keep`/`*KeepDims` overloads preserve the
input rank with every selected extent replaced by one.

```scala mdoc
val bothAxes = Axes.from(grid.rank, 0, -1).toOption.get
grid.sum(bothAxes)
grid.sumKeepDims(bothAxes).shape
```

Negative axes are normalized once and duplicates are rejected as pure
`AxesError` values. Execution plans the complete reduction once, without a
sequential intermediate for each axis. An empty `Axes` value creates an owned
copy. The `sumAxes`, `productAxes`, `minAxes`, `maxAxes`, and `meanAxes`
varargs methods are throwing conveniences around the same validation.

Empty min/max/arg reductions throw `EmptyReduction`. Keep code explicit:

```scala mdoc:reset:silent
import ravel.*

val grid: Array2[Double] =
  NDArray.tabulate(2, 3)((row, column) => row.toDouble + column / 10.0)
```

```scala mdoc
grid.argMax
grid.argMax(axis = 1)
```

`argMin` and `argMax` support a scalar result or one axis. Multi-axis arg
semantics are intentionally deferred because a stable result needs an explicit
coordinate or flattened-index contract.

For multiple axes, each floating output fiber traverses selected source axes in
ascending source-axis order, with the last selected axis varying fastest. The
usual 128-element blocks and pairwise merge are then applied to that direct
fiber. Caller axis order therefore cannot alter floating bits. See
[Dtypes and ranks](../reference/dtype-rank.md) for empty, NaN, widening, and
integer-division behavior.
