# Ravel

[User guide](https://canardlapin.github.io/ravel/) ·
[API reference](https://canardlapin.github.io/ravel/api/) ·
[Source](https://github.com/canardlapin/ravel)

> [!WARNING]
> **Ravel is an early 0.1-level GitHub project, not a 1.0 release candidate.**
> No Maven Central artifacts are published. The API and numerical contract may
> change without source or binary compatibility. Use the current code only for
> development and experimentation.
>
> The critical development gate is **semantic and performance parity against
> NumPy** on the public access-pattern suite
> ([docs/numpy-benchmarks.md](docs/numpy-benchmarks.md)). Run
> `bash scripts/numpy-parity-gate.sh` locally; CI runs the same correctness
> check. Timing reports are diagnostic until same-host baselines stabilize.

Ravel provides dense, rectangular multidimensional arrays for Scala 3 on the
JVM and Scala.js. Use it when data has a runtime shape and numerical operations
should run over primitive platform storage.

```scala
import ravel.*

val x: Array2[Double] =
  NDArray.tabulate(3, 4)((row, column) => row * 10.0 + column)

val everyOtherColumn = x.slice(axis = 1, Slice.every(2))
val columnMeans = x.mean(axis = 0)
val centered = x - columnMeans.newAxis(0)
```

The array value consists of one flat primitive buffer plus a shape, strides,
and an offset. Slicing, reversal, transposition, axis insertion, and
broadcasting create views. Numerical operations and reductions execute eagerly.
Ravel does not implement image semantics, matrix multiplication, decompositions,
sparse arrays, automatic differentiation, chunked storage, or I/O. Those
features belong in [Gale](https://github.com/canardlapin/gale) or another
layer.

## Neighborhood execution

`ravel-stencil` is a generic N-dimensional neighborhood substrate for sibling
libraries. It owns offset traversal, border-index mapping, and reference/direct
execution over `NDArray` views; it deliberately does not know image roles,
physical units, geometry, colour, or display.

```scala
import ravel.stencil.*

val spec = NeighborhoodSpec(
  spatialAxes = 2,
  offsets = Vector(Vector(-1, 0), Vector(0, 0), Vector(1, 0)),
  border = BorderMode.ReflectWithoutEdge,
  outputOrigin = Vector(0, 0),
  outputSpatialShape = Vector(width, height)
)
```

Trailing axes are batch axes: a spatial neighborhood keeps their coordinates
fixed. `ReferenceNeighborhoodExecutor` is the clarity-first conformance oracle.
`DirectNeighborhoodExecutor` supports arbitrary Ravel source views without
per-sample index-array allocation. Prepare a
`PreparedDirectNeighborhoodExecutor` when repeating a pass over the same
logical shape: its `runByte`, `runShort`, `runFloat`, and `runDouble` methods
reuse private workspace and keep their reads, writes, and accumulators
primitive.

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

`NDArray.cast` exposes the host's direct primitive conversion behavior.
Scientific code should use `NDArray.convert`, whose default policy is
nearest-even rounding with rejected overflow. `ConversionPolicy` can instead
request toward-zero, floor, or ceiling rounding and reject, clamp, or explicit
low-level wrap behavior. Checked conversion validates before allocating an
output buffer and never creates an output-sized boxed intermediate.

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

Start with the [user guide](https://canardlapin.github.io/ravel/), then use its
[copy/view table](https://canardlapin.github.io/ravel/reference/copy-view-table.html)
and [NumPy mapping](https://canardlapin.github.io/ravel/reference/numpy-map.html)
for quick lookups. The proposed 1.0 guarantees remain an engineering contract
in [docs/release-contract.md](docs/release-contract.md).

## Availability

Ravel is currently available only as source code. There is no released version
to add to an sbt build. Artifact names `ravel-core`, `ravel-laws`, and
`ravel-stencil` are the intended modules when a first tagged release eventually
happens; treat that as future packaging work, not current project maturity.

## Developer commands

```sh
sbt compileAll
sbt testAll
sbt browserTests/test
sbt testAllFull
sbt representationProof
bash scripts/numpy-parity-gate.sh   # critical: Ravel vs NumPy correctness
sbt docsCheck                       # Scaladoc + executable mdoc/Laika guide
sbt fmtCheck
sbt mimaCheck
sbt verifyPublishArtifacts
sbt coverageReportJvm               # diagnostic; no threshold
```

`testAll` runs the core, reusable laws, and stencil suites on the JVM and Node.
`browserTests/test` runs the browser-specific Scala.js suite in headless
Chromium. `scripts/numpy-parity-gate.sh` compares public access-pattern
signatures to NumPy without JMH timings. Full timed comparisons are documented
in [NumPy benchmarks](docs/numpy-benchmarks.md). The current build uses
Scala 3.7.4. Continuous integration runs on Temurin JDK 21 and Node 22; other
compiler and runtime combinations are not yet part of any compatibility
contract. Formatting, MiMa scaffolding, and publishable-coordinate checks are
documented in [release engineering](docs/release-engineering.md).

The public guide source is isolated under `docs/user/`; benchmark receipts,
audits, and release evidence elsewhere in `docs/` are intentionally excluded
from the published site.
