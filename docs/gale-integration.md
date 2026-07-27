# Gale integration boundary

Ravel represents dense rectangular storage. Gale represents mathematical
vectors and matrices and owns multiplication, factorizations, solvers, sparse
formats, and numerical backend selection.

The `gale-interop-ravel` artifact belongs in the Gale repository and depends on
both `gale-core` and `ravel-core`. Neither core artifact depends on the other.
Conversions are explicit and copy logical values:

```scala
import gale.interop.ravel.*

val vector: gale.linalg.DVec = vectorFromRavelCopy(ravelVector)
val matrix: gale.linalg.DMat = matrixFromRavelCopy(ravelArray2)
val array1: ravel.Array1[Double] = toRavelCopy(vector)
val array2: ravel.Array2[Double] = toRavelCopy(matrix)
```

Copy-only conversion preserves both libraries' ordinary ownership contracts.
It also handles sliced, reversed, transposed, and broadcast Ravel inputs
without exposing either library's private platform storage.

Ravel documentation calls `Array2[A]` a rank-two array, not a matrix. Matrix
orientation, multiplication, adjoints, structural properties, and algorithms
begin only after conversion to Gale.

## Deferred substrate convergence

The copy-only boundary is the current contract, not necessarily the final
representation strategy. Once Ravel is mature, revisit whether Gale's `DVec`
and `DMat` should remain Gale mathematical types backed by Ravel
`Array1[Double]` and `Array2[Double]` values. That could eliminate boundary
copies, unify ownership and view semantics, and concentrate primitive storage
optimization in Ravel without moving linear algebra into Ravel.

Do not begin that migration until Ravel provides:

- allocation-free fixed-rank indexing;
- a stable monomorphic `Double` kernel-view SPI that dispatches once and gives
  trusted Gale kernels buffer, offset, shape, and stride access;
- consuming ownership-transfer builders so Gale destinations do not acquire a
  final full-array copy;
- explicit compatibility rules for positive, negative, broadcast, and mutable
  layouts; and
- JVM and Scala.js differential benchmarks showing that Gale's indexing,
  vector, matrix, builder, and backend hot paths are performance-neutral or
  faster.

If those gates are met, preserve `DVec` and `DMat` as Gale's public
mathematical vocabulary, keep the dependency direction `Gale -> Ravel`, and
replace copy conversions with explicit zero-copy views only where ownership and
layout constraints make them honest.
