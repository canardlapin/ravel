# Store compact codes

Use `ravel-packed` when every sample is a small non-negative code and an
`NDArray[Byte]` would spend more storage than the values require. The module
stores one-, two-, or four-bit codes in 32-bit words on both the JVM and
Scala.js.

Packed arrays are not `NDArray` values and packed widths are not `DType`s. They
have their own shape, error, view, mutation, and serialization contracts.

## Construct validated codes

Choose `B1` for `0`–`1`, `B2` for `0`–`3`, or `B4` for `0`–`15`:

```scala mdoc:silent
import ravel.packed.*

val labels =
  PackedArray.fromCodes(
    shape = Vector(2, 4),
    bits = PackedBits.B2,
    codes = Vector(0, 1, 2, 3, 3, 2, 1, 0)
  ) match
    case Right(value) => value
    case Left(error)  => throw new IllegalArgumentException(error.message)
```

```scala mdoc
labels.shape
labels.bits
labels(1, 2)
labels.sumCodes
```

`fromCodes` returns `Either[PackedError, PackedArray]`. It rejects an empty
shape, non-positive extents, the wrong number of codes, and values outside the
selected width. Packed arrays currently require at least one non-empty axis;
unlike core `Shape`, they do not represent scalars or zero-length dimensions.

`tabulate` has a different contract: it masks each generated value to the
selected width. Use `fromCodes` when an out-of-range value should be reported
rather than truncated.

## Take a view, then materialize deliberately

`slice(axis, index)` fixes one axis and drops it. `narrow` preserves rank and
restricts an axis to a half-open range:

```scala mdoc
val secondRow = labels.slice(axis = 0, index = 1)
secondRow.map(row => (row.shape, row.isCanonical, row.codeVector))

val middleColumns = labels.narrow(axis = 1, start = 1, length = 2)
middleColumns.map(_.codeVector)
```

Both operations return `Either` and, on success, share the original words.
Packed indexing is non-negative. Call `copy` to re-pack a view into minimal
canonical storage with zeroed unused tail bits:

```scala mdoc
secondRow.map(row => (row.copy.isCanonical, row.copy.wordVector))
```

The name `slice` here differs from core `NDArray.slice`, which preserves rank;
the packed operation behaves like core `select`. Keep that difference visible
when code uses both representations.

## Round-trip canonical words

`wordVector` returns canonical row-major words. Code zero occupies the least
significant bits of the first word, and unused bits in the final word are zero.
The logical word sequence has the same meaning on the JVM and Scala.js.

```scala mdoc
val words = labels.wordVector
val restored = PackedArray.fromWords(labels.shape, labels.bits, words)

words
restored.map(_.codeVector)
```

`fromWords` copies and validates the input. It rejects an incorrect word count
or a nonzero unused tail, so accepted words are canonical serialization input.
The module does not choose a byte-level file format; a caller serializing
32-bit words must still define byte order for that outer format.

## Combine one-bit masks

`PackedBitOps` performs union, intersection, difference, symmetric difference,
complement, and population count over `B1` arrays:

```scala mdoc:silent
val selected = PackedArray.fromCodes(
  Vector(8),
  PackedBits.B1,
  Vector(1, 0, 1, 0, 1, 0, 0, 0)
)
val reviewed = PackedArray.fromCodes(
  Vector(8),
  PackedBits.B1,
  Vector(0, 1, 1, 0, 0, 0, 0, 1)
)

val combined =
  for
    left <- selected
    right <- reviewed
    union <- PackedBitOps.union(left, right)
    count <- PackedBitOps.countTrue(union)
  yield (union.codeVector, count)
```

```scala mdoc
combined
```

Set operations require equal shapes and `B1` widths. Views are canonicalized
before wordwise work, so a view may allocate an intermediate copy as well as
the result.

## Preserve the ownership boundary

| Operation | Storage consequence | Failure surface |
|---|---|---|
| `fromCodes`, `fromWords`, `zeros` | new canonical immutable storage | `Either[PackedError, ...]` |
| `slice`, `narrow` | shared immutable view | `Either[PackedError, ...]` |
| `copy` | new canonical immutable storage | none after a valid source exists |
| `wordVector`, `codeVector` | copied Scala collection | none after a valid source exists |
| `PackedBitOps.*` | new canonical result; views may first be copied | `Either[PackedError, ...]` |
| `MutablePackedArray.freezeCopy` | immutable copy; workspace remains writable | none |
| `MutablePackedArray.freeze` | transfers the backing words without copying | caller must retire the workspace |

`freeze` does not enforce retirement at runtime. Reading or writing the mutable
workspace after the transfer can violate the immutable result's ownership
assumption. Use `freezeCopy` unless the producer can make the transfer final by
construction.

Continue with [Failures and recovery](../reference/failures.md), or use the
[Packed API](https://canardlapin.github.io/ravel/api/packed/) for signatures.
