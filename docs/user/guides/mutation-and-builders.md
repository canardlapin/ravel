# Mutation and builders

Ravel offers two deliberate ways to fill storage: mutate an isolated
`MutableNDArray`, or consume an `ArrayBuilder` while constructing an immutable
array.

## Mutate an isolated copy

```scala mdoc:silent
import ravel.*

val source: Array2[Double] =
  NDArray.tabulate(2, 3)((row, column) => row * 10.0 + column)
val work = source.mutableCopy
```

Fixed-rank reads and writes address logical coordinates directly:

```scala mdoc
work(0, 1) = 99.0
work.select(axis = 0, index = 1).addInPlace(0.5)

val result: Array2[Double] = work.freezeCopy()
source(0, 1)
result(0, 1)
result(1, 2)
```

Both `mutableCopy` and `freezeCopy` copy. Mutable slices, reversals,
permutations, and legal no-copy reshapes remain views over the same mutable
storage. Broadcasting is intentionally unavailable for mutable arrays because
a zero stride would alias multiple logical cells.

Mutable arrays are also read operands for eager arithmetic, comparisons,
reductions, callbacks, and expert kernels. Those operations do not freeze the
source; eager results own new storage, while `kernel.*Into` rejects a
destination that aliases any mutable input before writing.

`reshapeCopy` and the copying fallback of `reshape` traverse the mutable source
directly into one new primitive destination. They do not freeze and then copy
that intermediate a second time.

## Construct one immutable destination

`NDArray.build` allocates one zero-initialized destination. The callback writes
by C-order linear index, and the returned immutable array takes ownership of
that destination without a second output-sized copy.

```scala mdoc
val diagonal: Array2[Double] = NDArray.build(Shape(4, 4)) { builder =>
  var i = 0
  while i < 4 do
    builder.writeLinear(i * 4 + i, 1.0)
    i += 1
}

diagonal
```

Writes may arrive in any order. Repeated writes are last-write-wins, and
unwritten cells retain the dtype's zero.

The builder is valid only during the synchronous callback. It is sealed on
normal return or failure, and later writes throw `BuilderClosed`. Do not retain
or share it.

## Reuse an output in a kernel

The expert `ravel.kernel` API writes into caller-owned, whole, contiguous
mutable destinations:

```scala mdoc
val left = NDArray.fill(Shape(2, 3), 2.0)
val right = NDArray.fill(Shape(2, 3), 3.0)
val out = MutableNDArray.zeros[Double, Rank[2]](Shape(2, 3))

kernel.addInto(left, right, out)
out.freezeCopy()
```

Use this surface for measured loops that need output reuse. Inputs may be
owned, borrowed, or mutable readable arrays. Destinations must not alias any
input; ordinary arithmetic remains the clearer default.
