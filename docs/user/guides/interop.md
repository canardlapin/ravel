# JVM and Scala.js interop

Interop starts with an ownership choice: copy when the Ravel value should be
independent, or borrow when later external mutation must remain visible.

## JVM arrays

`JvmInterop.copyToArray` always returns a new primitive JVM array:

```scala mdoc
import ravel.*
import ravel.jvm.JvmInterop

val owned: Array2[Double] =
  NDArray.fromSeq(Shape(2, 2), Seq(1.0, 2.0, 3.0, 4.0))
val copied: Array[Double] = JvmInterop.copyToArray(owned)

copied.toSeq
```

`unsafeBorrow` avoids the copy and returns `BorrowedNDArray`:

```scala mdoc
val external = Array(1.0, 2.0, 3.0, 4.0)
val borrowed = JvmInterop.unsafeBorrow(external, Shape(2, 2))

external(0) = 42.0
borrowed(0, 0)

val detached: Array2[Double] = borrowed.copy
external(0) = -1.0
detached(0, 0)
```

Borrowed structural views remain borrowed. Numerical operations and `copy`
produce owned values.

## Scala.js typed arrays

Scala.js uses dtype-specific typed arrays:

```scala
import ravel.*
import ravel.js.JsInterop
import scala.scalajs.js.typedarray.Float64Array

val values = new Float64Array(4)
val borrowed = JsInterop.unsafeBorrow(values, Shape(2, 2))
val copied = JsInterop.copyToFloat64Array(borrowed)
```

`Boolean` uses `Uint8Array` values exactly equal to `0` or `1`. `Long` is
semantically supported through Scala `Array[Long]`, but it has no JavaScript
typed-array borrowing API and is outside the Scala.js typed-array fast path.

Descriptor methods such as `describeDouble` expose the borrowed buffer plus
offset, shape, and strides for JavaScript consumers. They do not turn an owned
`NDArray` into a borrowed value.

## Boundary checklist

Before crossing a platform boundary, decide:

1. Must later external writes be visible? If no, copy.
2. Can the consumer handle nonzero offsets or arbitrary strides? If no,
   materialize a contiguous owned value first.
3. Does the boundary retain data beyond the current call? If yes, prefer an
   owned copy unless shared lifetime and mutation are part of the contract.
