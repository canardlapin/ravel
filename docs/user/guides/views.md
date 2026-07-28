# Indexing and views

## Read a fixed rank

Ranks one through four have direct `apply` overloads:

```scala mdoc:silent
import ravel.*

val cube: Array3[Int] =
  NDArray.tabulate(2, 3, 4)((i, j, k) => i * 100 + j * 10 + k)
```

```scala mdoc
cube(1, 2, 3)
cube(-1, -1, -1)
```

Use `at(IArray(...))` when rank is dynamic or greater than four. Wrong arity,
out-of-bounds coordinates, and invalid axes produce Ravel's typed exception
classes rather than silently clipping.

## Select and slice

`select` removes an axis; `slice` preserves rank:

```scala mdoc
val plane: Array2[Int] = cube.select(axis = 0, index = 1)
val alternating: Array3[Int] = cube.slice(axis = 2, Slice.every(2))
val reversed: Array3[Int] = cube.reverse(axis = -1)

plane.shape
alternating.shape
reversed(0, 0, 0)
```

Helpers include `Slice.all`, `from`, `until`, `between`, `every`, and
`reverse`. Scala `Range` is accepted as convenience syntax. `narrow` is the
strict exact-bounds operation.

## Move and insert axes

```scala mdoc
val transposed: Array2[Int] = plane.transpose
val batched: Array3[Int] = plane.newAxis(0)

transposed.shape
batched.shape
```

`swapAxes` and `permuteAxes` cover general reordering. Existing-axis APIs
accept negative axes; `newAxis` accepts every insertion position.

## Reshape deliberately

`reshapeView` succeeds only when the current logical traversal already has the
physical address sequence required by the target:

```scala mdoc
val flatView: Array1[Int] = plane.reshapeView(Shape(12))
flatView.shape
```

For a non-contiguous value, choose the behavior:

- `reshape(target)` uses a view when legal, otherwise copies;
- `reshapeCopy(target)` always materializes;
- `reshapeView(target)` never copies and throws `NonContiguousLayout` when the
  layout cannot be reinterpreted.

This naming is intentional: code that must control allocation should not rely
on a hidden fallback.

## Know when data moves

All operations on this page are views except `reshape` when it falls back and
the explicitly named copying operations. Use
[Copy and view behavior](../reference/copy-view-table.md) as the compact
reference.
