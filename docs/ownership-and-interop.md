# Ownership and platform interop

An `NDArray` owns its storage. Public constructors allocate a new primitive
buffer and copy caller data when necessary. No public method returns that
mutable buffer.

`BorrowedNDArray` describes a different contract. It reads a caller-owned JVM
array or JavaScript typed array without copying. If the caller later changes
that storage, the borrowed value changes too. `BorrowedNDArray` is not a
subtype of `NDArray`, and there is no implicit conversion between them.

## Borrowed views and owned computations

Structural operations preserve the marker:

```scala
val borrowed: BorrowedNDArray[Double, Rank[2]] = ...
val row: BorrowedNDArray[Double, Rank[1]] = borrowed.select(0, 1)
val reversed: BorrowedNDArray[Double, Rank[2]] = borrowed.reverse(1)
```

Operations that calculate or materialize elements return owned values:

```scala
val copied: Array2[Double] = borrowed.copy
val transformed: Array2[Double] = borrowed.map(_ * 2.0)
val reduced: Array1[Double] = borrowed.sum(0)
```

Changing the external buffer after these calls can change `borrowed`, `row`,
and `reversed`. It cannot change `copied`, `transformed`, or `reduced`.
`BorrowedNDArray.contiguous` also copies, even when the borrowed layout is
already contiguous, because returning an ordinary `NDArray` must establish
ownership.

## JVM arrays

JVM interop lives under `ravel.jvm`:

```scala
import ravel.jvm.JvmInterop

val values = Array(1.0, 2.0, 3.0, 4.0)
val borrowed = JvmInterop.unsafeBorrow(values, Shape(2, 2))
val copy: Array[Double] = JvmInterop.copyToArray(borrowed.transpose)
```

`unsafeBorrow` requires the array length to equal the shape size.
`copyToArray` always returns a new array in logical row-major order, including
for sliced, reversed, or transposed inputs.

## JavaScript typed arrays

Scala.js interop lives under `ravel.js`. Fast-path dtypes use their matching
typed arrays:

| Ravel dtype | JavaScript input or output |
|---|---|
| `Boolean` | `Uint8Array` containing only 0 and 1 (`unsafeBorrow` / `copyToUint8Array`) |
| `UInt8` | `Uint8Array` (`unsafeBorrowUInt8` / `copyToUInt8Array`) |
| `Byte` | `Int8Array` |
| `UInt16` | `Uint16Array` (`unsafeBorrowUInt16` / `copyToUInt16Array`) |
| `Short` | `Int16Array` |
| `Int` | `Int32Array` |
| `Float` | `Float32Array` |
| `Double` | `Float64Array` |

On the JVM, `JvmInterop.unsafeBorrowUInt8` / `unsafeBorrowUInt16` borrow
`Array[Byte]` / `Array[Short]` as unsigned magnitude storage for NIfTI-style
buffers without copying.

```scala
import ravel.js.JsInterop
import scala.scalajs.js.typedarray.Float64Array

val values = new Float64Array(6)
val borrowed = JsInterop.unsafeBorrow(values, Shape(2, 3))
val copy = JsInterop.copyToFloat64Array(borrowed.transpose)
val descriptor = JsInterop.describeDouble(borrowed.reverse(1))
```

Copy functions always materialize logical row-major order. Descriptor
functions are available only for borrowed values. A descriptor exposes the
original typed array, the view offset, and metadata copies of its shape and
strides. A strided view remains strided in the descriptor. Mutating the
descriptor's metadata arrays cannot alter the Ravel layout.

Scala.js `Long` uses a Scala `Array[Long]` fallback. Ravel 1.0 therefore
provides neither a native typed-array borrowing method nor a zero-copy `Long`
descriptor.

## Mutable arrays

`MutableNDArray` owns its storage and supports only layouts known to be
one-to-one. Mutable slicing, reversal, transposition, and legal reshape views
are supported. Broadcasting is not: a non-singleton zero stride would make
several logical indices name the same cell.

`freezeCopy` always copies. Later mutation through the mutable value or one of
its views cannot change the frozen result.
