# Moving array code from NumPy to Ravel

Ravel uses NumPy-style trailing-axis broadcasting and row-major logical
iteration, but it makes copying, viewing, and mutation explicit. It does not
reproduce NumPy's full indexing language or dtype promotion table.

```scala
import ravel.*
```

| NumPy | Ravel |
|---|---|
| `np.zeros((3, 4))` | `NDArray.zeros[Double](3, 4)` |
| `x.shape` | `x.shape` |
| `x[1, 2]` | `x(1, 2)` |
| `x[:, ::2]` | `x.slice(1, Slice.every(2))` |
| `x[1, :]` | `x.select(0, 1)` |
| `x[-1]` | `x(-1)` or `x.select(0, -1)` |
| `x.T` | `x.transpose` |
| `x.reshape((2, 6))` | `x.reshape(Shape(2, 6))` |
| `x.copy()` | `x.copy` |
| `x.astype(np.float64)` | `x.cast[Double]` |
| `x.mean(axis=0)` | `x.mean(axis = 0)` |
| `np.expand_dims(x, 0)` | `x.newAxis(0)` |
| `np.broadcast_to(x, shape)` | `x.broadcastTo(shape)` |
| `np.maximum(x, 0)` | `x.maximum(0)` |
| `x / 2` (ints) | `x.quot(2)` — `/` is float-only |

## Index one axis at a time

Use `select` to remove one axis and `Slice` to keep it:

```scala
val x = NDArray.tabulate[Int](3, 4)((row, column) => row * 10 + column)
val secondRow: Array1[Int] = x.select(axis = 0, index = 1)
val evenColumns: Array2[Int] = x.slice(axis = 1, Slice.every(2))
val reversedRows: Array2[Int] = x.reverse(axis = 0)
```

Axes and element indices may be negative. `-1` denotes the last axis or last
index. `Slice` helpers (`all`, `from`, `until`, `between`, `every`, `reverse`)
normalize endpoints against the axis length. `narrow` remains strict.

## Reshape may copy unless you ask for a view

`reshape` returns a view when strides allow it and copies otherwise.
`reshapeView` never copies and throws `NonContiguousLayout` when a view is
impossible. `reshapeCopy` always copies.

## Broadcasting

Ravel aligns axes on the right. Two dimensions are compatible when they are
equal or either one is 1. Broadcasting dimensions 0 and 1 produces 0.
Incompatible operations throw `BroadcastMismatch` and report both shapes and
the failing aligned axis.

Ravel does not promote mixed dtypes:

```scala
val integers = NDArray.zeros[Int](3)
val doubles = NDArray.zeros[Double](3)
val result = integers.cast[Double] + doubles
```

`Byte` and `Short` are storage and comparison dtypes in 1.0. Cast them to
`Int` before arithmetic.

## Mutation is visible in the type

An ordinary `NDArray` has no update method:

```scala
val mutable = x.mutableCopy
mutable(0, 0) = 42
mutable.addInPlace(1)
val result = mutable.freezeCopy()
```

Both transitions copy. Mutable broadcast views are unavailable because
broadcast strides can alias logical positions.

## Equality is explicit

Scala `==` retains reference semantics. Use:

- `sameElements` for logical value equality;
- `sameElementsBits` for bit-sensitive `Float` and `Double` equality;
- `allClose(other, relativeTolerance, absoluteTolerance)` for floating
  tolerance checks.

`allClose` requires explicit nonnegative tolerances. It treats matching
infinities as equal and NaNs as unequal.
