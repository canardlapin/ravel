# Ravel

[User guide](https://canardlapin.github.io/ravel/) ·
[Core API](https://canardlapin.github.io/ravel/api/) ·
[Source](https://github.com/canardlapin/ravel)

Ravel is a Scala 3 library for eager, dense multidimensional arrays on the JVM
and Scala.js. It gives numerical and scientific libraries dtype-specific
storage, checked shapes, allocation-visible views, and explicit ownership
boundaries without taking on matrix algebra, image meaning, or I/O.

> **Status:** Early, pre-release, and source-only. No Maven Central artifacts
> are published, and the API and numerical contract may change.

## Quick start

```scala
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

Ravel is not released yet. To run this example from another project, clone this
repository and publish the core development snapshots locally:

```sh
sbt testAll
sbt publishLocalSnapshot
```

Then add the JVM dependency:

```scala
libraryDependencies +=
  "io.github.canardlapin" %% "ravel-core" % "1.0.0-SNAPSHOT"
```

Use `%%%` for a Scala.js project. The version above is a local integration
coordinate, not a public release.

## What Ravel covers

- Build rank-aware arrays from values or functions, then inspect them through
  fixed-rank or dynamic-rank indexing.
- Slice, reverse, transpose, reshape, and broadcast with explicit copy/view
  behavior.
- Run eager elementwise operations, comparisons, checked conversions, and
  deterministic reductions over supported primitive dtypes.
- Cross JVM-array and Scala.js typed-array boundaries by copying or by
  returning an explicitly borrowed value.
- Reuse mutable destinations and low-level kernels when measured code needs to
  control output allocation.

Two experimental source modules serve library authors:

- `ravel-packed` stores one-, two-, or four-bit codes and performs wordwise
  set algebra over one-bit arrays. It is parallel to `NDArray`, not another
  `DType`.
- `ravel-stencil` runs generic N-dimensional neighborhoods over Ravel arrays.
  It owns offset traversal and border mapping, not image geometry, units,
  colour, or display.

See [Store compact codes](https://canardlapin.github.io/ravel/guides/packed-codes.html)
and [Run neighborhood computations](https://canardlapin.github.io/ravel/guides/neighborhoods.html)
for executable workflows.

## Choose a module

| Module | Use it for | Relationship |
|---|---|---|
| `ravel-core` | Dense arrays, views, computation, mutation, and platform interop | Ordinary entry point |
| `ravel-packed` | Compact one-, two-, and four-bit codes | Independent packed representation |
| `ravel-stencil` | Neighborhood traversal over `NDArray` values | Depends on `ravel-core` |
| `ravel-laws` | Internal generated repository tests | Unpublished test harness with no main-source API or downstream coordinate |

All four are cross-built for the JVM and Scala.js in the current source tree.
The 1.0 publication matrix contains only `ravel-core` for the JVM and
Scala.js. `ravel-packed` and `ravel-stencil` remain unpublished experimental
source modules. `ravel-laws` is an internal test-only harness: its generated
properties carry no binary-compatibility or downstream conformance promise.

## Mental model

An `NDArray` is a flat primitive buffer plus a shape, strides, and an offset.
Logical traversal is always C-style row-major order, even when a view uses
negative, permuted, stepped, or broadcast strides.

Dimension sizes are runtime `Int` values. Rank may remain dynamic as
`AnyNDArray[A]` or be refined:

```scala
val matrix: Array2[Double] = NDArray.zeros[Double](3, 4)
val dynamic: AnyNDArray[Double] = matrix
val checked: Either[RankMismatch, Array2[Double]] = dynamic.requireRank[2]
```

The supported `NDArray` dtypes are `Boolean`, `Byte`, `UInt8`, `Short`,
`UInt16`, `Int`, `Long`, `Float`, and `Double`. Same-dtype arithmetic is
available for `Int`, `Long`, `Float`, and `Double`; the narrower integral
types support storage, ordering, comparisons, casts, minimum, and maximum.

`NDArray` owns immutable storage, `MutableNDArray` owns explicitly mutable
storage, and `BorrowedNDArray` records that an external owner can still mutate
the backing buffer. Numerical operations return new owned arrays.

## Copy and view rules

| Operation | Result |
|---|---|
| `select`, `slice`, `narrow`, `reverse` | view |
| `swapAxes`, `permuteAxes`, `transpose` | view |
| `newAxis`, `squeeze`, `broadcastTo` | view |
| `reshapeView` | view or `NonContiguousLayout` |
| `reshape` | view when legal, otherwise owned copy |
| owned `contiguous` | same value or owned copy |
| `copy`, `reshapeCopy`, `flattenCopy` | owned copy |
| `map`, arithmetic, comparisons, array reductions | owned allocation |
| `BorrowedNDArray.contiguous` | owned copy |

The guide’s [copy/view table](https://canardlapin.github.io/ravel/reference/copy-view-table.html)
covers mutation, builders, and platform boundaries as well.

## Fit and boundaries

Ravel is the dense rectangular storage and elementwise-computation layer. It
does not implement matrix multiplication, decompositions, sparse arrays,
automatic differentiation, chunked storage, random-number policy, GPU
execution, or file formats. Mathematical vectors, matrices, and decompositions
belong in [Gale](https://github.com/canardlapin/gale); image geometry and value
meaning belong in an image layer.

On Scala.js, `Boolean`, the fixed-width integers other than `Long`, `Float`, and
`Double` use matching typed-array representations. `Long` uses a Scala
`Array[Long]` fallback and has no typed-array borrowing API.

## Documentation

- [Getting started](https://canardlapin.github.io/ravel/getting-started.html) —
  local installation and the first complete array workflow.
- [Core concepts](https://canardlapin.github.io/ravel/core-concepts.html) —
  dtype, rank, layout, ownership, and eager execution.
- [Guides](https://canardlapin.github.io/ravel/guides/) — computation, views,
  mutation, packed codes, neighborhoods, and platform interop.
- [Failure reference](https://canardlapin.github.io/ravel/reference/failures.html) —
  thrown exceptions, `Either` results, and recovery choices.
- [Core API](https://canardlapin.github.io/ravel/api/) — generated Scaladoc;
  the guide’s reference index links the other module APIs.
- [Performance methodology](docs/numpy-benchmarks.md) — parity fixtures,
  benchmark scope, and why timing receipts are not a release claim.

## Verify the source

```sh
bash scripts/release-gate.sh
```

The release gate is the same entry point used by CI and tag publication. It
checks all source modules and both core platforms, then verifies NumPy parity,
documentation, artifact contents, and fresh JVM and Scala.js consumers. It
writes receipts under `target/release-gate/<commit>/`.

The current build uses Scala 3.7.4. CI is configured for Temurin JDK 21 and
Node 22; other compiler and runtime combinations are outside the current
compatibility evidence.

## License

Apache License 2.0. See [LICENSE](LICENSE).
