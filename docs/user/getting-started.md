# Getting started

Ravel currently has no released artifact. To evaluate it, clone the repository
and run:

```sh
sbt testAll
sbt publishLocalSnapshot
```

The second command publishes development-only `1.0.0-SNAPSHOT` artifacts to
your local Ivy repository. Choose the smallest module that owns your task:

| Module | Use it for |
|---|---|
| `ravel-core` | Dense arrays, views, computation, mutation, and platform interop |
| `ravel-packed` | Compact one-, two-, and four-bit codes |
| `ravel-stencil` | Neighborhood execution over core arrays |
| `ravel-laws` | Reusable conformance laws in tests |

A separate Scala 3 project can then depend on the JVM core module:

```scala
libraryDependencies +=
  "io.github.canardlapin" %% "ravel-core" % "1.0.0-SNAPSHOT"
```

For Scala.js, use the same `%%%` dependency syntax you use for other
cross-published libraries. Do not publish or redistribute these local
snapshots as a released Ravel version.

## Complete one array workflow

Import the public package and construct a rank-two array. Dimension sizes are
runtime values; the `Array2` alias records only the rank in the type.

```scala mdoc:silent
import ravel.*

val samples: Array2[Double] =
  NDArray.tabulate(3, 4)((row, column) => row * 10.0 + column)

val columnMeans: Array1[Double] = samples.mean(axis = 0)
val centered: Array2[Double] = samples - columnMeans.newAxis(0)

centered.mean(axis = 0)
```

`Array2` records the rank in the type while the dimensions remain runtime
values. The subtraction broadcasts `columnMeans` across rows and returns a new
owned array; it does not mutate `samples`.

Inspect shape and elements with fixed-rank indexing:

```scala mdoc
samples.shape
samples.rank
samples(2, 3)
samples(-1, -1)
```

Negative indices count from the end of an axis.

## Transform without copying

Structural operations produce views over the same storage:

```scala mdoc
val everyOtherColumn = samples.slice(axis = 1, Slice.every(2))
everyOtherColumn.shape
everyOtherColumn(2, 1)

val transposed = samples.transpose
transposed.shape
transposed(3, 2)
```

Call `copy` or `reshapeCopy` when you want materialization. The
[copy/view table](reference/copy-view-table.md) lists the full contract.

## Interpret the result

Array/scalar arithmetic allocates one owned result. Array/array arithmetic
uses NumPy-style trailing-axis broadcasting:

```scala mdoc
columnMeans
centered.mean(axis = 0)
```

Ravel is an eager array layer, not a matrix algebra system: it does not provide
matrix multiplication, decompositions, sparse arrays, I/O, or automatic
differentiation.

## Refine a dynamic rank

When a boundary loses static rank information, validate it once:

```scala mdoc
val dynamic: AnyNDArray[Double] = samples
val checked: Either[RankMismatch, Array2[Double]] =
  dynamic.requireRank[2]

checked.map(_(1, 2))
```

Continue with [Core concepts](core-concepts.md), go directly to
[Indexing and views](guides/views.md), or choose a different module from the
[guide index](guides/index.md).
