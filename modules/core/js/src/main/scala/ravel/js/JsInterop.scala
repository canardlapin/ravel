package ravel.js

import ravel.*
import ravel.internal.*
import scala.annotation.targetName
import scala.scalajs.js.typedarray.*

final case class BooleanDescriptor(
    buffer: Uint8Array,
    offset: Int,
    shape: Int32Array,
    strides: Int32Array
)

final case class ByteDescriptor(
    buffer: Int8Array,
    offset: Int,
    shape: Int32Array,
    strides: Int32Array
)

final case class UInt8Descriptor(
    buffer: Uint8Array,
    offset: Int,
    shape: Int32Array,
    strides: Int32Array
)

final case class ShortDescriptor(
    buffer: Int16Array,
    offset: Int,
    shape: Int32Array,
    strides: Int32Array
)

final case class UInt16Descriptor(
    buffer: Uint16Array,
    offset: Int,
    shape: Int32Array,
    strides: Int32Array
)

final case class IntDescriptor(
    buffer: Int32Array,
    offset: Int,
    shape: Int32Array,
    strides: Int32Array
)

final case class FloatDescriptor(
    buffer: Float32Array,
    offset: Int,
    shape: Int32Array,
    strides: Int32Array
)

final case class DoubleDescriptor(
    buffer: Float64Array,
    offset: Int,
    shape: Int32Array,
    strides: Int32Array
)

object JsInterop:
  def copyToUint8Array(array: NDArray[Boolean, ?]): Uint8Array =
    copyBoolean(array)

  def copyToUint8Array(array: BorrowedNDArray[Boolean, ?]): Uint8Array =
    copyBoolean(array.underlying)

  def copyToInt8Array(array: NDArray[Byte, ?]): Int8Array =
    copyByte(array)

  def copyToInt8Array(array: BorrowedNDArray[Byte, ?]): Int8Array =
    copyByte(array.underlying)

  /** Logical row-major copy of unsigned 8-bit values into a `Uint8Array`.
    *
    * Distinct from [[copyToUint8Array]], which encodes `Boolean` as 0/1.
    */
  def copyToUInt8Array(array: NDArray[UInt8, ?]): Uint8Array =
    copyUInt8(array)

  def copyToUInt8Array(array: BorrowedNDArray[UInt8, ?]): Uint8Array =
    copyUInt8(array.underlying)

  def copyToInt16Array(array: NDArray[Short, ?]): Int16Array =
    copyShort(array)

  def copyToInt16Array(array: BorrowedNDArray[Short, ?]): Int16Array =
    copyShort(array.underlying)

  def copyToUInt16Array(array: NDArray[UInt16, ?]): Uint16Array =
    copyUInt16(array)

  def copyToUInt16Array(array: BorrowedNDArray[UInt16, ?]): Uint16Array =
    copyUInt16(array.underlying)

  def copyToInt32Array(array: NDArray[Int, ?]): Int32Array =
    copyInt(array)

  def copyToInt32Array(array: BorrowedNDArray[Int, ?]): Int32Array =
    copyInt(array.underlying)

  def copyToFloat32Array(array: NDArray[Float, ?]): Float32Array =
    copyFloat(array)

  def copyToFloat32Array(array: BorrowedNDArray[Float, ?]): Float32Array =
    copyFloat(array.underlying)

  def copyToFloat64Array(array: NDArray[Double, ?]): Float64Array =
    copyDouble(array)

  def copyToFloat64Array(array: BorrowedNDArray[Double, ?]): Float64Array =
    copyDouble(array.underlying)

  @targetName("unsafeBorrowBoolean")
  def unsafeBorrow[R <: AnyRank](
      values: Uint8Array,
      shape: Shape[R]
  ): BorrowedNDArray[Boolean, R] =
    var i = 0
    while i < values.length do
      val value = values(i)
      if value != 0 && value != 1 then
        throw new IllegalArgumentException(
          s"Boolean Uint8Array contains $value at index $i; expected 0 or 1"
        )
      i += 1
    borrowed(new BooleanStorage(values), shape, DType.booleanDType)

  @targetName("unsafeBorrowByte")
  def unsafeBorrow[R <: AnyRank](
      values: Int8Array,
      shape: Shape[R]
  ): BorrowedNDArray[Byte, R] =
    borrowed(new ByteStorage(values), shape, DType.byteDType)

  /** Borrows a `Uint8Array` as unsigned 8-bit storage.
    *
    * Named distinctly from Boolean [[unsafeBorrow]] on `Uint8Array`, which validates a 0/1
    * encoding.
    */
  def unsafeBorrowUInt8[R <: AnyRank](
      values: Uint8Array,
      shape: Shape[R]
  ): BorrowedNDArray[UInt8, R] =
    borrowed(new UInt8Storage(values), shape, DType.uint8DType)

  @targetName("unsafeBorrowShort")
  def unsafeBorrow[R <: AnyRank](
      values: Int16Array,
      shape: Shape[R]
  ): BorrowedNDArray[Short, R] =
    borrowed(new ShortStorage(values), shape, DType.shortDType)

  def unsafeBorrowUInt16[R <: AnyRank](
      values: Uint16Array,
      shape: Shape[R]
  ): BorrowedNDArray[UInt16, R] =
    borrowed(new UInt16Storage(values), shape, DType.uint16DType)

  @targetName("unsafeBorrowInt")
  def unsafeBorrow[R <: AnyRank](
      values: Int32Array,
      shape: Shape[R]
  ): BorrowedNDArray[Int, R] =
    borrowed(new IntStorage(values), shape, DType.intDType)

  @targetName("unsafeBorrowFloat")
  def unsafeBorrow[R <: AnyRank](
      values: Float32Array,
      shape: Shape[R]
  ): BorrowedNDArray[Float, R] =
    borrowed(new FloatStorage(values), shape, DType.floatDType)

  @targetName("unsafeBorrowDouble")
  def unsafeBorrow[R <: AnyRank](
      values: Float64Array,
      shape: Shape[R]
  ): BorrowedNDArray[Double, R] =
    borrowed(new DoubleStorage(values), shape, DType.doubleDType)

  def describeBoolean(array: BorrowedNDArray[Boolean, ?]): BooleanDescriptor =
    val layout = array.underlying.layout
    val storage = array.underlying.storage.asInstanceOf[BooleanStorage]
    BooleanDescriptor(storage.raw, layout.offset, metadata(layout.shape), metadata(layout.strides))

  def describeByte(array: BorrowedNDArray[Byte, ?]): ByteDescriptor =
    val layout = array.underlying.layout
    val storage = array.underlying.storage.asInstanceOf[ByteStorage]
    ByteDescriptor(storage.raw, layout.offset, metadata(layout.shape), metadata(layout.strides))

  def describeUInt8(array: BorrowedNDArray[UInt8, ?]): UInt8Descriptor =
    val layout = array.underlying.layout
    val storage = array.underlying.storage.asInstanceOf[UInt8Storage]
    UInt8Descriptor(storage.raw, layout.offset, metadata(layout.shape), metadata(layout.strides))

  def describeShort(array: BorrowedNDArray[Short, ?]): ShortDescriptor =
    val layout = array.underlying.layout
    val storage = array.underlying.storage.asInstanceOf[ShortStorage]
    ShortDescriptor(storage.raw, layout.offset, metadata(layout.shape), metadata(layout.strides))

  def describeUInt16(array: BorrowedNDArray[UInt16, ?]): UInt16Descriptor =
    val layout = array.underlying.layout
    val storage = array.underlying.storage.asInstanceOf[UInt16Storage]
    UInt16Descriptor(storage.raw, layout.offset, metadata(layout.shape), metadata(layout.strides))

  def describeInt(array: BorrowedNDArray[Int, ?]): IntDescriptor =
    val layout = array.underlying.layout
    val storage = array.underlying.storage.asInstanceOf[IntStorage]
    IntDescriptor(storage.raw, layout.offset, metadata(layout.shape), metadata(layout.strides))

  def describeFloat(array: BorrowedNDArray[Float, ?]): FloatDescriptor =
    val layout = array.underlying.layout
    val storage = array.underlying.storage.asInstanceOf[FloatStorage]
    FloatDescriptor(storage.raw, layout.offset, metadata(layout.shape), metadata(layout.strides))

  def describeDouble(array: BorrowedNDArray[Double, ?]): DoubleDescriptor =
    val layout = array.underlying.layout
    val storage = array.underlying.storage.asInstanceOf[DoubleStorage]
    DoubleDescriptor(storage.raw, layout.offset, metadata(layout.shape), metadata(layout.strides))

  private def borrowed[A, R <: AnyRank](
      storage: Storage[A],
      shape: Shape[R],
      dtype: DType[A]
  ): BorrowedNDArray[A, R] =
    if storage.length != shape.size then
      throw ShapeMismatch(shape.toString, s"typed array(length = ${storage.length})")
    val view = new NDArray(storage, Layout.contiguous(shape, storage.length), dtype)
    new BorrowedNDArray(view)

  private def copyBoolean(array: NDArray[Boolean, ?]): Uint8Array =
    val output = new Uint8Array(array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = (if value then 1 else 0)
      write += 1
    }
    output

  private def copyByte(array: NDArray[Byte, ?]): Int8Array =
    val output = new Int8Array(array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = value
      write += 1
    }
    output

  private def copyUInt8(array: NDArray[UInt8, ?]): Uint8Array =
    val output = new Uint8Array(array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = value.toInt.toShort
      write += 1
    }
    output

  private def copyShort(array: NDArray[Short, ?]): Int16Array =
    val output = new Int16Array(array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = value
      write += 1
    }
    output

  private def copyUInt16(array: NDArray[UInt16, ?]): Uint16Array =
    val output = new Uint16Array(array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = value.toInt
      write += 1
    }
    output

  private def copyInt(array: NDArray[Int, ?]): Int32Array =
    val output = new Int32Array(array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = value
      write += 1
    }
    output

  private def copyFloat(array: NDArray[Float, ?]): Float32Array =
    val output = new Float32Array(array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = value
      write += 1
    }
    output

  private def copyDouble(array: NDArray[Double, ?]): Float64Array =
    val output = new Float64Array(array.size)
    var write = 0
    array.foreachElement { value =>
      output(write) = value
      write += 1
    }
    output

  private def metadata(values: IArray[Int]): Int32Array =
    val output = new Int32Array(values.length)
    var i = 0
    while i < values.length do
      output(i) = values(i)
      i += 1
    output
