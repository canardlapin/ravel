# Structural views

Every structural operation in Ravel preserves the same storage object and
changes only shape, strides, and offset. The exceptions are deliberately named:
`copy`, `flattenCopy`, and `contiguous` when materialization is required.

`Slice(start, stopExclusive, step)` is the resolved form. Open helpers
(`Slice.all`, `Slice.from`, `Slice.until`, `Slice.between`, `Slice.every`,
`Slice.reverse`) normalize endpoints against the axis length when applied.
Negative element indices are accepted on indexing and `select`. Fully specified
negative-step slices still treat stop `-1` as “before the first element”.
`narrow` remains the strict exact-bounds operation. Scala `Range` is convenience
syntax converted through checked exclusive-end arithmetic.

All existing-axis APIs accept axes in `[-rank, rank)`. `newAxis` accepts
insertion positions from `-(rank + 1)` through `rank`.

## `reshapeView`

`reshapeView` never copies. It succeeds only when the source's logical
row-major traversal already has the physical address sequence required by the
target shape:

1. Check equal element counts.
2. For zero elements, construct a checked canonical empty layout.
3. Ignore singleton source axes.
4. Reject every non-singleton zero-stride source axis.
5. Coalesce adjacent non-singleton source axes only when
   `outerStride == innerSize * innerStride`.
6. Partition the target's non-singleton axes, in order, into groups whose
   products exactly equal the source contiguous-block products.
7. Derive each target group's strides from its source block's innermost stride,
   using checked `Long` products before narrowing to `Int`.

If any partition or checked derivation fails, `reshapeView` throws
`NonContiguousLayout` and does not allocate.
