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

### Kernel-view SPI decision: no-go for core 1.0

The design gate rejects a public raw-`Double` kernel view in `ravel-core` 1.0.
This is an ownership decision, not a claim that raw buffers are technically
difficult to expose.

`CanonicalArray[Double, R]` and `MutableCanonicalArray[Double, R]` are the
minimal safe prototype. Refinement returns the original whole canonical array
without a wrapper, dispatches primitive linear reads or writes at the call
site, and exposes neither the platform buffer nor a physical address. The
cross-platform `CanonicalAccessSuite` verifies canonical acceptance,
noncanonical rejection, primitive access, bounds, and mutable locality. This
surface is suitable for allocation-free scalar kernels, but it does not give
Gale the raw substrate required by its current array loops or native backend
handoff.

A callback such as `withDoubleStorage { Array[Double] => ... }` is not a safe
substitute. Scala cannot prevent the callback from retaining the JVM array or
Scala.js `Float64Array`; retaining it would permit later mutation of an
immutable `NDArray`. A nominal read-only wrapper would avoid that leak but
would return to per-element method access and would not serve BLAS, typed-array,
or same-storage alias checks. The mutable case is stricter: Ravel cannot prove
that no `MutableNDArray` view or caller alias remains while Gale holds an
“exclusive” raw destination. Borrowed input adds an external writer that Ravel
does not control.

Storage identity, physical offset, and validated reachable bounds therefore
remain private kernel facts. Adding public descriptor copies of shape and
strides without the raw buffer would not accelerate Gale; adding the raw buffer
would weaken the ownership contract that core 1.0 is intended to stabilize.

If this decision is revisited, the candidate belongs in a separately versioned,
explicitly unsafe and platform-specific experimental artifact—not stable core.
It must define non-retention or linear-lifetime enforcement, exclusive mutable
borrowing, JVM `Array[Double]` and Scala.js `Float64Array` parity, storage
identity, checked address bounds, and backend handoff before Gale prototypes
against it. Until then, the supported boundary remains copy-only.

### Layout and ownership compatibility

The current adapter accepts logical rank-one and rank-two values, not physical
Ravel layouts. It reads through Ravel's public checked indexing API and fills a
fresh consuming `DVecBuilder` or row-major `DMatBuilder`. The result therefore
owns Gale storage and retains no Ravel or external alias. Conversion in the
other direction likewise tabulates a fresh canonical Ravel owner from the
logical Gale view.

| Ravel source or destination case | Current copy adapter | Zero-copy raw adapter under the 1.0 contract |
|---|---|---|
| Whole canonical rank-one or rank-two owner | Copy logical values | Reject: immutable Ravel storage must not become externally mutable or retainable |
| Positive-stride slice or narrow view | Copy logical values in coordinate order | Reject: Gale has no public Ravel storage descriptor, offset, or checked reachable interval |
| Negative-stride reverse view | Copy in reversed logical order | Reject: a raw backend cannot assume positive or unit strides |
| Zero-stride broadcast view | Copy and expand the repeated logical values | Reject, especially as a destination: multiple logical coordinates alias one physical cell |
| Rank-two transpose or permutation | Copy in Ravel logical row/column order into a row-major `DMat` | Reject: interpreting the buffer as another major order would be backend-specific and would change `DMat`'s public coordinate semantics |
| Empty rank-one or rank-two shape, including `0 x n` and `n x 0` | Copy to a legal empty `DVec` or `DMat`; no element address is dereferenced | Reject: there is no useful buffer handoff and ownership/lifetime questions remain |
| Borrowed Ravel input | Copy; later external mutation cannot affect Gale | Reject: the external owner may write at any time |
| Mutable Ravel input | No direct overload; make ownership explicit with an owned copy before conversion | Reject: Ravel cannot prove that no mutable view or caller alias survives |
| Gale `DVec` or `DMat` view to Ravel | Copy logical values into a canonical owned Ravel result | Reject: Gale storage and view lifetime remain Gale-owned |
| Mutable Gale/Ravel destination | Fill that library's consuming builder through its public API | Reject raw handoff: neither library exposes a cross-library exclusive-borrow token |

Copies make alias analysis trivial: the destination is distinct by
construction. A zero-copy implementation would instead need public storage
identity for every source and destination, checked physical intervals, and a
rule for partially overlapping strided layouts. Those facts remain private.
Ravel's `Layout` validation proves its own reachable minimum and maximum
addresses against storage length, so the current public-indexing copy inherits
checked access. A future descriptor would have to carry and revalidate the raw
buffer length, base offset, shape, strides, and reachable interval with checked
`Long` arithmetic before Gale could dereference it.

This matrix deliberately does not say that Gale supports arbitrary Ravel
layouts natively. It says that the copy adapter can normalize their logical
values while preserving `DVec` and `DMat` indexing, orientation, ownership, and
row-major construction semantics. With the stable-core SPI rejected, there is
no safe zero-copy layout class to prototype or benchmark for 1.0.

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
