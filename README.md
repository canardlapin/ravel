# Ravel

> [!WARNING]
> **Ravel is an early development project. It is not ready for production use.**
> No release has been published, no Maven Central artifacts are available, and
> the API and numerical contract may change without source or binary
> compatibility. Use the current code only for development and experimentation.

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

The current development code supports `Boolean`, `Byte`, `Short`, `Int`, `Long`,
`Float`, and `Double`. Arithmetic is available for `Int`, `Long`, `Float`, and
`Double`. `Byte` and `Short` support storage, casts, comparisons, minimum, and
maximum, but not same-dtype arithmetic. On Scala.js, `Long` uses a Scala
`Array[Long]` fallback and is outside the native typed-array fast path.

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
The proposed 1.0 guarantees are in the
[release contract](docs/release-contract.md).

## Availability

Ravel is currently available only as source code. There is no released version
to add to an sbt build. The planned cross-published artifact names are
`ravel-core` and `ravel-laws`; dependency coordinates will be documented only
after those artifacts have been published.

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
Chromium. The current build uses Scala 3.7.4. Continuous integration runs on
Temurin JDK 21 and Node 22; other compiler and runtime combinations are not yet
part of the compatibility contract.
