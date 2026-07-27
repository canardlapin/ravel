# Ravel

Ravel provides dense, rectangular multidimensional arrays for Scala 3 on the
JVM and Scala.js. Use it when data has a runtime shape and numerical operations
should run over primitive platform storage.

```scala
import ravel.*
import ravel.DType.given

val x: Array2[Double] =
  NDArray.tabulate(3, 4)((row, column) => row * 10.0 + column)

val everyOtherColumn = x.slice(axis = 1, Slice(0, 4, 2))
val columnMeans = x.mean(axis = 0)
val centered = x - columnMeans.newAxis(0)
```

The array value consists of one flat primitive buffer plus a shape, strides,
and an offset. Slicing, reversal, transposition, axis insertion, and
broadcasting create views. Numerical operations and reductions execute eagerly.
Ravel does not implement matrix multiplication, decompositions, sparse arrays,
automatic differentiation, chunked storage, or I/O. Those features belong in
[Gale](https://github.com/canardlapin/gale) or another layer.

## Types and storage

Dimension sizes are runtime `Int` values. Rank can remain dynamic as
`AnyNDArray[A]` or be refined:

```scala
val matrix: Array2[Double] = NDArray.zeros[Double](3, 4)
val dynamic: AnyNDArray[Double] = matrix
val checked: Either[RankMismatch, Array2[Double]] = dynamic.requireRank[2]
```

Ravel 1.0 supports `Boolean`, `Byte`, `Short`, `Int`, `Long`, `Float`, and
`Double`. Arithmetic is available for `Int`, `Long`, `Float`, and `Double`.
`Byte` and `Short` support storage, casts, comparisons, minimum, and maximum,
but not same-dtype arithmetic.

Owned `NDArray` values do not expose mutable backing storage. Use
`mutableCopy` for explicit mutation and `freezeCopy` to obtain another owned
value. Platform zero-copy input returns `BorrowedNDArray`, which is a separate
type because later external mutation remains observable.

## Copy and view rules

| Operation | Result |
|---|---|
| `select`, `slice`, `narrow`, `reverse` | view |
| `swapAxes`, `permuteAxes`, `transpose` | view |
| `newAxis`, `squeeze`, `broadcastTo` | view |
| `reshapeView` | view or `NonContiguousLayout` |
| `contiguous` on an owned contiguous array | same value |
| `copy`, `flattenCopy` | owned copy |
| `map`, arithmetic, comparisons | owned allocation |
| array-valued reductions | owned allocation |
| `BorrowedNDArray.contiguous` | owned copy |

See [ownership and interop](docs/ownership-and-interop.md) for the aliasing
contract and [NumPy migration](docs/numpy-migration.md) for operation mappings.
The complete 1.0 guarantees are in the
[release contract](docs/release-contract.md).

## Artifacts

The cross-published artifacts are:

```scala
"io.github.canardlapin" %%% "ravel-core" % "1.0.0"
"io.github.canardlapin" %%% "ravel-laws" % "1.0.0" // test support
```

## Developer commands

```sh
sbt compileAll
sbt testAll
sbt browserTests/test
sbt testAllFull
sbt representationProof
```

`testAll` runs the core and reusable laws suites on the JVM and Node.
`browserTests/test` runs the browser-specific Scala.js suite in headless
Chromium.
