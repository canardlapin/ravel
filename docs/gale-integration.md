# Gale integration boundary

Ravel represents dense rectangular storage. Gale represents mathematical
vectors and matrices and owns multiplication, factorizations, solvers, sparse
formats, and numerical backend selection.

The `gale-interop-ravel` artifact belongs in the Gale repository and depends on
both `gale-core` and `ravel-core`. Neither core artifact depends on the other.
Conversions are explicit and copy logical values:

```scala
import gale.interop.ravel.*

val vector: gale.linalg.DVec = fromRavelCopy(ravelVector)
val matrix: gale.linalg.DMat = fromRavelCopy(ravelArray2)
val array1: ravel.Array1[Double] = toRavelCopy(vector)
val array2: ravel.Array2[Double] = toRavelCopy(matrix)
```

Copy-only conversion preserves both libraries' ordinary ownership contracts.
It also handles sliced, reversed, transposed, and broadcast Ravel inputs
without exposing either library's private platform storage.

Ravel documentation calls `Array2[A]` a rank-two array, not a matrix. Matrix
orientation, multiplication, adjoints, structural properties, and algorithms
begin only after conversion to Gale.

The libraries intentionally retain domain-specific semantics where uniform
spelling would be misleading:

- Ravel coordinate indices may be negative and count from the end of an axis;
  Gale vector and matrix indices are non-negative.
- Ravel `*` is elementwise with broadcasting; Gale `DMat.*` is matrix
  multiplication and Gale's elementwise product is explicit through
  `pointwise`.
- Ravel rank-two values remain arrays; Gale values gain matrix semantics only
  after explicit conversion.

## Deferred substrate convergence

The copy-only boundary is the current contract, not necessarily the final
representation strategy. Once Ravel is mature, revisit whether Gale's `DVec`
and `DMat` should remain Gale mathematical types backed by Ravel
`Array1[Double]` and `Array2[Double]` values. That could eliminate boundary
copies, unify ownership and view semantics, and concentrate primitive storage
optimization in Ravel without moving linear algebra into Ravel.

Do not begin that migration until Ravel provides:

- a stable monomorphic `Double` kernel-view SPI that dispatches once and gives
  trusted Gale kernels buffer, offset, shape, and stride access;
- consuming ownership-transfer builders so Gale destinations do not acquire a
  final full-array copy;
- explicit compatibility rules for positive, negative, broadcast, and mutable
  layouts; and
- JVM and Scala.js differential benchmarks showing that Gale's indexing,
  vector, matrix, builder, and backend hot paths are performance-neutral or
  faster.

Fixed-rank indexing (`physicalIndex1`–`4`) is already allocation-free in
`ravel-core`. Consuming `NDArray.build` / `ArrayBuilder` landed for destination
fills; the monomorphic `Double` kernel-view SPI remains deferred.

If those gates are met, preserve `DVec` and `DMat` as Gale's public
mathematical vocabulary, keep the dependency direction `Gale -> Ravel`, and
replace copy conversions with explicit zero-copy views only where ownership and
layout constraints make them honest.

## Panama and native linear algebra

Foreign Function and Memory API work remains wholly Gale-owned. The committed
Gale tree at `d55fe2f97196a76ab7879e1a12f1e92403aeba06` already contains the
appropriate optional boundary:

- `gale-backend-jvm-native` provides JDK 22+ `NativeDMat` storage with explicit
  `Arena` lifetime, row/column-major layout, leading dimension, and checked
  heap/native conversion;
- `gale-backend-jvm-blas-ffm` owns runtime library discovery, CBLAS/LAPACK ABI
  descriptors and downcalls, thread-count policy, typed Gale errors, and
  provider-specific routing thresholds; and
- `benchmarks/jvm-ffm` measures the public Gale facades including heap-to-native
  copy-in, native work, native-to-heap copy-out, and result allocation, with a
  separate long-lived `NativeDMat` control.

Neither Gale core nor Ravel core imports `java.lang.foreign`. Native access is
an application opt-in (`--enable-native-access=ALL-UNNAMED`), and the minimum
runtime for these optional artifacts is JDK 22, where the FFM API was
[finalized by JEP 454](https://openjdk.org/jeps/454). The bindings do not use
`Linker.Option.critical(true)` or pass heap segments to long-running BLAS/LAPACK
calls; they copy to arena-owned native segments before ordinary downcalls.
There are no upcall stubs or native-to-Java callbacks in this boundary.

The checked-in Apple AArch64 / Accelerate receipts produce deliberately narrow
defaults:

| Route | Copy-inclusive decision |
|---|---|
| square GEMM | enable from `n = 512`; `n = 256` was a measured 0.25x loss |
| GEMV | disabled; no crossover through `n = 2048` |
| LU and solve | enable from `n = 128` |
| symmetric eigen | enable from `n = 128` |
| Cholesky and QR/least-squares | disabled; results were non-monotone or losing |
| OpenBLAS, MKL, unknown BLAS | automatic routing disabled pending provider-specific sweeps |

This is the Stage 5 decision: use the existing Gale optional backend for
coarse, typed BLAS/LAPACK calls and keep explicit copy costs in its dispatch
policy. Do not add FFM storage or downcalls to Ravel, do not route elementwise
or reduction microkernels through native code, and do not infer one vendor's
thresholds for another. A clean JDK 25 audit of the committed Gale snapshot
passed all three native-storage tests, all 21 BLAS/LAPACK tests, and FFM JMH
compilation; the authoritative crossover timings remain the checked-in JDK 22
two-fork receipts.

## CI verification today

Until Central publication, verify the copy-only boundary from a Ravel snapshot:

```sh
# ravel
sbt publishLocalSnapshot
# gale
sbt interopRavelTest
```

Ravel CI does not clone Gale; the sibling `interopRavelTest` gate remains the
authoritative interop check.
