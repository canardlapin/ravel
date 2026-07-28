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
length-one axes, and coalesces adjacent compatible axes. General loops use
counters and offsets rather than division of a linear index.

The supported external-kernel contract, including fixed Rank1-4 indexing,
builder ownership, failure semantics, and allocation evidence, is documented
in [allocation-free kernel APIs](allocation-free-kernels.md).

These details may change without changing the public array semantics or
interop types.

## Deliberate exclusions

`ravel-core` contains no matrix multiplication, adjoint, decomposition, sparse
format, BLAS/backend selection, FFT, random-number policy, named axis, I/O,
chunked execution, GPU execution, automatic differentiation, or lazy
expression graph. A rank-two Ravel array is a dense array, not a mathematical
matrix. Gale owns matrix and vector semantics.

## Verification

The release gate runs:

```sh
sbt testAll
sbt browserTests/test
sbt testAllFull
sbt representationProof
```

The Node and browser suites are separate. The browser suite verifies that it is
running with `window` and `document` in real Chromium before testing typed-array
borrowing and numerical behavior.

Ravel has no binary-compatibility baseline before the first 1.0 publication.
The published 1.0 API becomes the baseline for the 1.x series. Subsequent 1.x
releases must add an automated MiMa or equivalent binary-compatibility gate
against the latest released 1.x artifact; incompatible public changes require a
new major version.

See [benchmark baselines](benchmark-baselines.md) for the performance
regression budget and [Gale integration](gale-integration.md) for the adapter
boundary.
