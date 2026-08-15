# Allocation-free kernel APIs

Ravel's public fixed-rank scalar reads and consuming builder are the supported
surface for external resampling and stencil kernels that need primitive storage
without depending on `ravel.internal`.

## Fixed-rank reads and mutable writes

`NDArray` and `MutableNDArray` provide `apply` overloads for ranks 1 through 4.
`MutableNDArray` provides matching `update` overloads. These calls calculate the
physical address directly from the validated layout. For statically known
primitive element types, their inline dispatch does not create an index
collection, box elements, or allocate per element in steady-state JVM
traversal. Code abstract in the element type uses the ordinary boxed JVM
fallback.

The overloads preserve the logical coordinates of:

- owned contiguous arrays;
- non-zero-offset and stepped slices;
- negative-stride views;
- permuted views; and
- broadcast immutable views.

Negative element indices count from the end of their axis. An index below
`-dimension` or at least `dimension` throws `InvalidIndex.OutOfBounds`. A
fixed-rank call with the wrong number of coordinates is rejected at compile
time.
`at(IArray[Int])` remains the arbitrary-rank alternative.

Mutable views write the same underlying logical element selected by their
layout. They do not expose their primitive storage publicly.

## Consuming immutable construction

`NDArray.build(shape) { builder => ... }` allocates one primitive destination
buffer. `ArrayBuilder.writeLinear` writes directly into that buffer by C-order
linear index. When the callback returns, the immutable `NDArray` owns that same
buffer; Ravel does not copy it into a second destination.

Writes may arrive in any order. Repeated writes use the last value, and
unwritten elements retain the primitive dtype's zero value. Invalid linear
indices throw `InvalidIndex.LinearOutOfBounds`.

The builder is scoped to the synchronous callback. It is sealed whether the
callback returns normally or throws, and every later write throws
`BuilderClosed`. If the callback throws, `NDArray.build` rethrows that exception
and publishes no partially built array.

`ArrayBuilder` is mutable and not thread-safe. Do not share it between threads
or retain it for later work. After construction, the returned `NDArray` is
immutable and does not expose a mutable alias to its storage.

## Canonical linear access

`CanonicalArray.require(array)` and `MutableCanonicalArray.require(array)`
refine a whole canonical array without creating a wrapper. The refinement
preserves the complete `NDArray` or `MutableNDArray` coordinate API and adds
explicit `readLinear`; mutable refinements also add `writeLinear`.

Coordinate access continues to normalize negative element indices by axis.
Linear access is deliberately distinct and accepts only C-order indices in
`[0, size)`. Structural operations such as `transpose` return ordinary arrays;
callers must establish the canonical refinement again before using linear
access on their result.

## Allocation evidence

The committed JMH probe covers Rank3 and Rank4 coordinate traversal, canonical
linear reads and writes, canonical refinement, a Rank3 output, and Rank4 reads
with linear output writes:

```sh
sbt 'representationProbeJVM/Jmh/run -prof gc -wi 5 -i 7 -f 2 -p edge=8,16 .*KernelApiBenchmarks.*'
```

The checksum methods consume their results so the loops cannot be eliminated;
normalized allocation must not scale with element count. Canonical refinement
must not allocate a wrapper. The two builder cases may allocate one primitive
output buffer plus constant-sized Ravel objects. Their normalized allocation
must not contain a second output-sized buffer or per-element objects.

The clean external-consumer probe publishes the current candidate to the local
Ivy repository, generates an unrelated temporary sbt build, and uses only
public `ravel-core` APIs:

```sh
bash scripts/verify-external-kernel-consumer.sh
```

Reusable-output kernels accept owned, borrowed, and mutable readable sources.
They validate result shape, whole contiguous destination layout, and storage
nonaliasing before mutation. Mutable `reshapeCopy` and noncontiguous `reshape`
allocate exactly one primitive destination and bounded metadata.
