# Core concepts

Ravel keeps five concerns separate: element dtype, runtime shape, optional
static rank, physical layout, and ownership. Most API behavior follows from
those boundaries.

## Dtype

The closed `DType[A]` family supports `Boolean`, `Byte`, `UInt8`, `Short`,
`UInt16`, `Int`, `Long`, `Float`, and `Double`. The compiler exposes operations
only when the element type has the required capability:

- `Int`, `Long`, `Float`, and `Double` support same-dtype arithmetic.
- `Byte`, `UInt8`, `Short`, and `UInt16` support storage, casts, ordering,
  min/max, and comparison; cast or widen to `Int` before arithmetic.
- `/` is floating-point division. `Int` and `Long` use the explicit `quot` or
  `truncDiv` operation.
- `Boolean` supports storage and elementwise equality.

These constraints are type evidence, not runtime feature flags.

## Shape and rank

A `Shape[R]` contains nonnegative runtime dimensions. `R` describes rank:

- `Array0[A]` through `Array4[A]` are convenient fixed-rank aliases.
- `NDArray[A, Rank[N]]` works for any statically known rank.
- `AnyNDArray[A]` retains a runtime rank only.

Rank-aware operations preserve or transform this type information. For
example, selecting one axis from `Array2[A]` yields `Array1[A]`, while
inserting an axis does the reverse. `requireRank[N]` validates a dynamic value
and returns `Either[RankMismatch, ...]`.

## Layout and logical order

An array is one primitive storage buffer plus shape, strides, and an offset.
Logical traversal is always C-style row-major order, even when physical
strides are negative, permuted, stepped, or zero because of broadcasting.

`isContiguous`, `isCanonicalLayout`, and `isWholeBuffer` answer different
questions. A contiguous view may still start at a nonzero offset or cover only
part of its backing buffer. Use the named copy operations when an API needs
owned, canonical whole-buffer storage.

## Ownership

`NDArray` is an immutable owned value. It never exposes mutable backing
storage. `MutableNDArray` is the explicit mutation surface, and
`BorrowedNDArray` records that external mutation may still be observed.

Structural operations retain ownership:

- an owned view remains `NDArray`;
- a borrowed view remains `BorrowedNDArray`;
- a mutable view remains `MutableNDArray`.

Numerical operations over owned or borrowed inputs return a new owned
`NDArray`.

## Eager computation

Elementwise operations and reductions execute immediately. Built-in kernels
may specialize dtype and layout, but generic callbacks run once per logical
output element in row-major order. Ravel does not promise fusion, laziness, or
an unboxed generic callback.

Next, see [Numerical computation](guides/computation.md) or the
[copy/view table](reference/copy-view-table.md).
