# Guides

These pages begin with a task and make allocation and ownership consequences
visible.

- [Numerical computation](computation.md): construct values, broadcast,
  transform, compare, cast, and reduce.
- [Indexing and views](views.md): choose fixed-rank indexing, dynamic indexing,
  slicing, axis operations, and reshape behavior.
- [Mutation and builders](mutation-and-builders.md): use mutable destinations
  or construct one immutable result without a second output-sized copy.
- [Packed codes](packed-codes.md): store compact categorical or mask values,
  take views, round-trip canonical words, and combine one-bit sets.
- [Neighborhood computations](neighborhoods.md): define offsets and border
  behavior, then run a reducer into an explicit destination.
- [JVM and Scala.js interop](interop.md): copy at a platform boundary or borrow
  with an explicit provenance type.

For terse lookup and recovery material, use the
[Reference](../reference/index.md) section.
