# Store compact codes

> `ravel-packed` is an experimental source module and is not part of the core
> 1.0 artifact or compatibility promise.

Use `ravel-packed` when every sample is a small non-negative code and an
`NDArray[Byte]` would spend more storage than the values require. The module
stores one-, two-, or four-bit codes in 32-bit words on both the JVM and
Scala.js.

Packed arrays are not `NDArray` values and packed widths are not `DType`s. They
have their own shape, error, view, mutation, and serialization contracts.

## Construct validated codes

Choose `B1` for `0`–`1`, `B2` for `0`–`3`, or `B4` for `0`–`15`:

```scala mdoc:silent
import ravel.*
import ravel.packed.*

val labels =
  PackedArray.fromCodes(
    shape = Shape(2, 4),
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

`fromCodes` returns `Either[PackedError, PackedArray]`. The supplied core
`Shape` has already checked negative dimensions and portable element-count
overflow. The packed constructor checks the number and range of codes.

Packed arrays use the same scalar and empty shapes as core. `Shape.scalar`
contains one code and has rank zero; shapes such as `Shape(2, 0, 3)` contain no
codes and preserve their zero extent through views.

`tabulate` has a different contract: it masks each generated value to the
selected width. Use `fromCodes` when an out-of-range value should be reported
rather than truncated.

## Take a view, then materialize deliberately

`select(axis, index)` fixes one axis and drops it. `slice(axis, Slice)` and
`narrow` preserve rank:

```scala mdoc
val secondRow = labels.select(axis = 0, index = 1)
(secondRow.shape, secondRow.isCanonical, secondRow.codeVector)

val middleColumns = labels.slice(axis = 1, Slice.between(1, 3))
middleColumns.codeVector

val finalColumn = labels.select(axis = -1, index = -1)
finalColumn.codeVector
```

Structural views share the original words. Axes and element coordinates accept
the same negative normalization as core. `narrowChecked` returns
`Either[InvalidNarrow, ...]`; the `narrow` convenience throws
`InvalidNarrowException`. `permuteAxesChecked` and `permuteAxes` use the same
checked/throwing split for permutation errors. Call `copy` to re-pack a view
into minimal canonical storage with zeroed unused tail bits:

```scala mdoc
(secondRow.copy.isCanonical, secondRow.copy.wordVector)
```

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
  Shape(8),
  PackedBits.B1,
  Vector(1, 0, 1, 0, 1, 0, 0, 0)
)
val reviewed = PackedArray.fromCodes(
  Shape(8),
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
| `select`, `slice`, `reverse` | shared immutable view | typed core exception for invalid programmer input |
| `narrowChecked`, `permuteAxesChecked` | shared immutable view | checked core error value |
| `copy` | new canonical immutable storage | none after a valid source exists |
| `wordVector`, `codeVector` | copied Scala collection | none after a valid source exists |
| `PackedBitOps.*` | new canonical result; views may first be copied | `Either[PackedError, ...]` |
| `MutablePackedArray.freezeCopy` | immutable copy; workspace remains writable | none |
| `MutablePackedArray.freeze` | transfers the backing words without copying and consumes the workspace | later data access throws `PackedWorkspaceConsumedException` |

`freeze` enforces retirement at runtime. `freezeCopy` is the alternative when
the producer must keep reading or writing the workspace.

Continue with [Failures and recovery](../reference/failures.md), or use the
[Packed API](https://canardlapin.github.io/ravel/api/packed/) for signatures.
