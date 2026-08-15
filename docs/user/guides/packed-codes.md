# Store compact codes

> `ravel-packed` is an experimental source module and is not part of the core
> 1.0 artifact or compatibility promise.

Use `ravel-packed` when every sample is a small non-negative code and an
`NDArray[Byte]` would spend more storage than the values require. The module
stores one-, two-, or four-bit codes in 32-bit words on both the JVM and
Scala.js.

Packed arrays are not `NDArray` values and packed widths are not `DType`s. They
reuse core `Shape`, `Slice`, indexing, and view semantics while retaining
storage-specific construction, mutation, and serialization contracts.

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

`tabulate` applies the same range check to every generated code. Use the
explicitly named `tabulateMasked` only when low-bit truncation is the intended
operation:

```scala mdoc
PackedArray.tabulate(Shape(3), PackedBits.B2)(index => index * 2)
PackedArray.tabulateMasked(Shape(3), PackedBits.B2)(index => index * 2).codeVector
```

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

## Round-trip words or portable bytes

`wordVector` returns canonical row-major words. Code zero occupies the least
significant bits of the first word, and unused bits in the final word are zero.
The logical word sequence has the same meaning on the JVM and Scala.js.

```scala mdoc
val words = labels.wordVector
val restored = PackedArray.fromWords(labels.shape, labels.bits, words)

words
restored.map(_.codeVector)
```

`fromWords` copies and validates the input. It consumes no more than the
expected number of words plus one, even when given an unbounded iterator. It
rejects an incorrect word count or a nonzero unused tail, so accepted words
are canonical serialization input.

For storage or interchange, `toBytes` and `fromBytes` implement the version 1
portable format:

```scala mdoc
val encoded = labels.toBytes
val decoded = encoded.flatMap(PackedArray.fromBytes)
decoded.map(value => (value.shape, value.bits, value.codeVector))
```

| Bytes | Version 1 meaning |
|---|---|
| `0..3` | ASCII magic `RVPK` |
| `4` | format version, currently `1` |
| `5` | code width: `1`, `2`, or `4` |
| `6..7` | reserved and required to be zero |
| `8..11` | non-negative rank as a big-endian 32-bit integer |
| next `4 * rank` | non-negative dimensions as big-endian 32-bit integers |
| remainder | canonical words in logical order, each big-endian |

Within each decoded word, the earliest logical code occupies the
least-significant slot. Later codes occupy progressively higher slots. Unused
high bits in the final word must be zero, and the payload must contain exactly
the word count implied by shape and width. Decoders reject unknown versions,
nonzero reserved bytes, invalid shapes or widths, wrong byte lengths, and
noncanonical tails. A future incompatible interpretation therefore requires a
new version rather than silently changing version 1.

## Combine one-bit masks

`PackedBitOps` performs union, intersection, difference, `xor` (also named
`symmetricDifference`), complement, and population count over `B1` arrays:

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
| `fromCodes`, `fromWords`, `fromBytes`, `tabulate`, `zeros` | new canonical immutable storage | `Either[PackedError, ...]`, except validated-shape `zeros` |
| `tabulateMasked` | new canonical immutable storage with explicit low-bit truncation | none after a valid shape exists |
| `toBytes` | copied portable byte representation; views first become canonical | `Either[PackedError, ...]` for unrepresentable byte length |
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
