# Ravel 1.0 contract

This page separates public guarantees from current implementation details.

## Public guarantees

- Shapes contain nonnegative `Int` dimensions. Size products, canonical
  strides, offset transformations, and reachable-address bounds are calculated
  with checked `Long` arithmetic before narrowing to `Int`.
- A rank-zero shape contains one scalar. A shape with any zero dimension
  contains no elements.
- Negative axes are accepted and normalized against the current rank.
- Structural operations document whether they return a view or a copy.
  `reshapeView` never copies silently.
- Owned arrays do not leak their mutable backing buffer.
- Borrowed external storage remains visible as `BorrowedNDArray` through every
  structural view.
- Built-in numerical operations execute eagerly and allocate one contiguous
  owned output.
- Generic callbacks execute once per logical output element in logical
  row-major order. Ravel does not fuse, reorder, or promise unboxed callback
  execution.
- Same-dtype arithmetic supports `Int`, `Long`, `Float`, and `Double`.
  Integer operations use ordinary two's-complement overflow. `Byte` and
  `Short` arithmetic requires an explicit cast to `Int`.
- `/` is available only for `Float` and `Double`. Integral truncation uses
  `quot` or `truncDiv`.
- Numeric casts follow Scala primitive conversion behavior: integral narrowing
  keeps low bits; floating-to-integral conversion truncates toward zero,
  maps NaN to zero, and clamps infinities and out-of-range values before any
  final `Byte` or `Short` narrowing.
- `sum` preserves the input arithmetic dtype. The supported widened sums are
  `Int` to `Long` and `Float` to `Double`.
- Floating sum and mean use logical row-major blocks of 128 values and a fixed
  adjacent-pair merge schedule on both platforms.
- Empty sum is positive zero and empty product is one. Empty floating mean is
  NaN. Empty minimum, maximum, and arg reductions throw `EmptyReduction`.
- NaNs propagate through minimum, maximum, sum, and mean. Arg reductions choose
  the first logical NaN; ordinary ties choose the first logical value.
  Minimum chooses negative zero and maximum chooses positive zero.
- `equals` is reference equality. Extensional comparisons are explicit.

## Current implementation details

JVM storage uses primitive `Array[T]`. Scala.js uses `Uint8Array`, `Int8Array`,
`Int16Array`, `Int32Array`, `Float32Array`, and `Float64Array` for its fast
dtypes. Scala.js `Long` uses `Array[Long]` and is outside the JavaScript
fast-path performance contract.

The loop planner aligns broadcast axes, assigns zero strides, removes
length-one axes, and coalesces adjacent compatible axes. Same-shape
C-contiguous binary and unary plans skip that work and run as linear
contiguous kernels. General loops use counters and offsets rather than
division of a linear index.

`kernel` exposes reusable-output Into forms for binary arithmetic, min/max,
unary negate/abs, scalar arithmetic, and callback `mapInto` / `zipMapInto`.
Destinations must be whole contiguous buffers that do not alias inputs.
`NDArray.build` fills a fresh contiguous array through a consuming
`ArrayBuilder` sealed when the body returns.
The supported external-kernel contract, including fixed Rank1-4 indexing,
builder ownership, failure semantics, and allocation evidence, is documented
in [allocation-free kernel APIs](allocation-free-kernels.md).

Numeric `cast` dispatches once on the source and target storage families.

These details may change without changing the public array semantics or
interop types.

## Deliberate exclusions

`ravel-core` contains no matrix multiplication, adjoint, decomposition, sparse
format, BLAS/backend selection, FFT, random-number policy, named axis, I/O,
chunked execution, GPU execution, automatic differentiation, or lazy
expression graph. A rank-two Ravel array is a dense array, not a mathematical
matrix. Gale owns matrix and vector semantics.

## Verification

While Ravel is a 0.1-level project, the critical numerical gate is NumPy
semantic parity. The full local checklist is:

```sh
bash scripts/numpy-parity-gate.sh
sbt fmtCheck
sbt mimaCheck
sbt testAll
sbt browserTests/test
sbt testAllFull
sbt representationProof
sbt verifyPublishArtifacts
bash scripts/verify-publish-artifacts.sh
```

Diagnostic JVM coverage (`sbt coverageReportJvm`) and the Gale sibling
`interopRavelTest` gate are documented in
[release engineering](release-engineering.md). NumPy timing aspirations live in
[benchmark baselines](benchmark-baselines.md).

The Node and browser suites are separate. The browser suite verifies that it is
running with `window` and `document` in real Chromium before testing typed-array
borrowing and numerical behavior.

There is no published binary-compatibility baseline yet. If a first tagged
release is published later, that API becomes the baseline for subsequent
compatible releases; incompatible public changes would then require a new major
version.

See [benchmark baselines](benchmark-baselines.md) for the performance
regression budget and [Gale integration](gale-integration.md) for the adapter
boundary.
