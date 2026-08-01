package ravel

import ravel.internal.*
import scala.annotation.unused
import scala.annotation.publicInBinary
import scala.compiletime.erasedValue

final class MutableNDArray[A, R <: AnyRank] private[ravel] (
    private[ravel] val storage: Storage[A],
    private[ravel] val mutableLayout: MutableLayout,
    val dtype: DType[A]
):
  private[ravel] def layout: Layout = mutableLayout.underlying

  val shape: Shape[R] =
    Shape.retag[R](layout.shapeValue)

  def rank: Int = layout.rank
  def size: Int = layout.size
  def isContiguous: Boolean = layout.isCContiguous
  def isCanonicalLayout: Boolean = layout.isCanonicalLayout
  def isWholeBuffer: Boolean = layout.isWholeBuffer(storage.length)

  inline def apply(i: Int): A =
    readPrimitive(physicalIndex1(i))

  inline def apply(i: Int, j: Int): A =
    readPrimitive(physicalIndex2(i, j))

  inline def apply(i: Int, j: Int, k: Int): A =
    readPrimitive(physicalIndex3(i, j, k))

  inline def apply(i: Int, j: Int, k: Int, l: Int): A =
    readPrimitive(physicalIndex4(i, j, k, l))

  def at(indices: IArray[Int]): A =
    ProbeApi.get(storage, layout.physicalIndex(indices))

  inline def update(i: Int, value: A): Unit =
    writePrimitive(physicalIndex1(i), value)

  inline def update(i: Int, j: Int, value: A): Unit =
    writePrimitive(physicalIndex2(i, j), value)

  inline def update(i: Int, j: Int, k: Int, value: A): Unit =
    writePrimitive(physicalIndex3(i, j, k), value)

  inline def update(i: Int, j: Int, k: Int, l: Int, value: A): Unit =
    writePrimitive(physicalIndex4(i, j, k, l), value)

  def updateAt(indices: IArray[Int], value: A): Unit =
    ProbeApi.set(storage, layout.physicalIndex(indices), value)

  private inline def readPrimitive(index: Int): A =
    // This match is reduced at the Scala call site. Each cast is guarded by its exact primitive
    // branch and routes to a primitive JVM/Scala.js method; abstract A uses the boxed fallback.
    inline erasedValue[A] match
      case _: Boolean => readBoolean(index).asInstanceOf[A]
      case _: Byte => readByte(index).asInstanceOf[A]
      case _: UInt8 => readUInt8(index).asInstanceOf[A]
      case _: Short => readShort(index).asInstanceOf[A]
      case _: UInt16 => readUInt16(index).asInstanceOf[A]
      case _: Int => readInt(index).asInstanceOf[A]
      case _: Long => readLong(index).asInstanceOf[A]
      case _: Float => readFloat(index).asInstanceOf[A]
      case _: Double => readDouble(index).asInstanceOf[A]
      case _ => readGeneric(index)

  private inline def writePrimitive(index: Int, value: A): Unit =
    inline erasedValue[A] match
      case _: Boolean => writeBoolean(index, value.asInstanceOf[Boolean])
      case _: Byte => writeByte(index, value.asInstanceOf[Byte])
      case _: UInt8 => writeUInt8(index, value.asInstanceOf[UInt8])
      case _: Short => writeShort(index, value.asInstanceOf[Short])
      case _: UInt16 => writeUInt16(index, value.asInstanceOf[UInt16])
      case _: Int => writeInt(index, value.asInstanceOf[Int])
      case _: Long => writeLong(index, value.asInstanceOf[Long])
      case _: Float => writeFloat(index, value.asInstanceOf[Float])
      case _: Double => writeDouble(index, value.asInstanceOf[Double])
      case _ => writeGeneric(index, value)

  @publicInBinary private[ravel] def physicalIndex1(i: Int): Int =
    layout.physicalIndex1(i)

  @publicInBinary private[ravel] def physicalIndex2(i: Int, j: Int): Int =
    layout.physicalIndex2(i, j)

  @publicInBinary private[ravel] def physicalIndex3(i: Int, j: Int, k: Int): Int =
    layout.physicalIndex3(i, j, k)

  @publicInBinary private[ravel] def physicalIndex4(i: Int, j: Int, k: Int, l: Int): Int =
    layout.physicalIndex4(i, j, k, l)

  @publicInBinary private[ravel] def readGeneric(index: Int): A =
    ProbeApi.get(storage, index)

  @publicInBinary private[ravel] def readBoolean(index: Int): Boolean =
    ProbeApi.getBoolean(storage.asInstanceOf[Storage[Boolean]], index)

  @publicInBinary private[ravel] def readByte(index: Int): Byte =
    ProbeApi.getByte(storage.asInstanceOf[Storage[Byte]], index)

  @publicInBinary private[ravel] def readShort(index: Int): Short =
    ProbeApi.getShort(storage.asInstanceOf[Storage[Short]], index)

  @publicInBinary private[ravel] def readUInt8(index: Int): UInt8 =
    ProbeApi.getUInt8(storage.asInstanceOf[Storage[UInt8]], index)

  @publicInBinary private[ravel] def readUInt16(index: Int): UInt16 =
    ProbeApi.getUInt16(storage.asInstanceOf[Storage[UInt16]], index)

  @publicInBinary private[ravel] def readInt(index: Int): Int =
    ProbeApi.getInt(storage.asInstanceOf[Storage[Int]], index)

  @publicInBinary private[ravel] def readLong(index: Int): Long =
    ProbeApi.getLong(storage.asInstanceOf[Storage[Long]], index)

  @publicInBinary private[ravel] def readFloat(index: Int): Float =
    ProbeApi.getFloat(storage.asInstanceOf[Storage[Float]], index)

  @publicInBinary private[ravel] def readDouble(index: Int): Double =
    ProbeApi.getDouble(storage.asInstanceOf[Storage[Double]], index)

  @publicInBinary private[ravel] def writeGeneric(index: Int, value: A): Unit =
    ProbeApi.set(storage, index, value)

  @publicInBinary private[ravel] def writeBoolean(index: Int, value: Boolean): Unit =
    ProbeApi.setBoolean(storage.asInstanceOf[Storage[Boolean]], index, value)

  @publicInBinary private[ravel] def writeByte(index: Int, value: Byte): Unit =
    ProbeApi.setByte(storage.asInstanceOf[Storage[Byte]], index, value)

  @publicInBinary private[ravel] def writeShort(index: Int, value: Short): Unit =
    ProbeApi.setShort(storage.asInstanceOf[Storage[Short]], index, value)

  @publicInBinary private[ravel] def writeUInt8(index: Int, value: UInt8): Unit =
    ProbeApi.setUInt8(storage.asInstanceOf[Storage[UInt8]], index, value)

  @publicInBinary private[ravel] def writeUInt16(index: Int, value: UInt16): Unit =
    ProbeApi.setUInt16(storage.asInstanceOf[Storage[UInt16]], index, value)

  @publicInBinary private[ravel] def writeInt(index: Int, value: Int): Unit =
    ProbeApi.setInt(storage.asInstanceOf[Storage[Int]], index, value)

  @publicInBinary private[ravel] def writeLong(index: Int, value: Long): Unit =
    ProbeApi.setLong(storage.asInstanceOf[Storage[Long]], index, value)

  @publicInBinary private[ravel] def writeFloat(index: Int, value: Float): Unit =
    ProbeApi.setFloat(storage.asInstanceOf[Storage[Float]], index, value)

  @publicInBinary private[ravel] def writeDouble(index: Int, value: Double): Unit =
    ProbeApi.setDouble(storage.asInstanceOf[Storage[Double]], index, value)

  def fill(value: A): Unit =
    if layout.isCContiguous && layout.offset == 0 && layout.size == storage.length then
      ProbeApi.fill(storage, value)
    else layout.foreachPhysicalIndex(index => ProbeApi.set(storage, index, value))

  def assign(source: NDArray[A, ?]): Unit =
    MutableNDArray.requireSameShape(layout.shape, source.layout.shape)
    MutableKernels.assign(source.storage, source.layout, storage, layout)

  def addInPlace(value: A)(using
      @unused arithmetic: ArithmeticDType[A]
  ): Unit =
    MutableKernels.scalarInPlace(KernelOp.Add, storage, layout, value)

  def subtractInPlace(value: A)(using
      @unused arithmetic: ArithmeticDType[A]
  ): Unit =
    MutableKernels.scalarInPlace(KernelOp.Subtract, storage, layout, value)

  def multiplyInPlace(value: A)(using
      @unused arithmetic: ArithmeticDType[A]
  ): Unit =
    MutableKernels.scalarInPlace(KernelOp.Multiply, storage, layout, value)

  def quotInPlace(value: A)(using
      @unused integral: IntegralArithmeticDType[A]
  ): Unit =
    MutableKernels.scalarInPlace(KernelOp.Divide, storage, layout, value)

  def divideInPlace(value: A)(using
      @unused floating: FloatingDType[A]
  ): Unit =
    MutableKernels.scalarInPlace(KernelOp.Divide, storage, layout, value)

  def select(axis: Int, index: Int)(using
      CanDropAxis[R]
  ): MutableNDArray[A, DropAxis[R]] =
    new MutableNDArray(
      storage,
      MutableLayout.select(mutableLayout, axis, index, storage.length),
      dtype
    )

  def slice(axis: Int, slice: Slice): MutableNDArray[A, R] =
    new MutableNDArray(
      storage,
      MutableLayout.slice(mutableLayout, axis, slice, storage.length),
      dtype
    )

  def slice(axis: Int, range: Range): MutableNDArray[A, R] =
    val canonical = Slice.from(range).fold(throw _, identity)
    slice(axis, canonical)

  def narrow(axis: Int, from: Int, length: Int): MutableNDArray[A, R] =
    if length < 0 then throw InvalidSlice(s"negative narrow length $length")
    val stop = Layout.checkedInt(
      Layout.checkedAdd(from.toLong, length.toLong, "mutable narrow endpoint"),
      "mutable narrow endpoint"
    )
    slice(axis, Slice(from, stop))

  def reverse(axis: Int): MutableNDArray[A, R] =
    new MutableNDArray(
      storage,
      MutableLayout.reverse(mutableLayout, axis, storage.length),
      dtype
    )

  def swapAxes(first: Int, second: Int): MutableNDArray[A, R] =
    val left = layout.normalizedAxis(first)
    val right = layout.normalizedAxis(second)
    val order = Array.tabulate(rank)(identity)
    val temporary = order(left)
    order(left) = order(right)
    order(right) = temporary
    permuteAxes(order.toSeq*)

  def permuteAxes(order: Int*): MutableNDArray[A, R] =
    new MutableNDArray(
      storage,
      MutableLayout.permute(mutableLayout, order, storage.length),
      dtype
    )

  def transpose: MutableNDArray[A, R] =
    if rank != 2 then throw InvalidAxis(2, rank)
    swapAxes(0, 1)

  def newAxis(axis: Int): MutableNDArray[A, AddAxis[R]] =
    new MutableNDArray(
      storage,
      MutableLayout.newAxis(mutableLayout, axis, storage.length),
      dtype
    )

  def squeeze(axis: Int)(using
      CanDropAxis[R]
  ): MutableNDArray[A, DropAxis[R]] =
    new MutableNDArray(
      storage,
      MutableLayout.squeeze(mutableLayout, axis, storage.length),
      dtype
    )

  def reshapeView[S <: AnyRank](
      target: Shape[S]
  ): MutableNDArray[A, S] =
    new MutableNDArray(
      storage,
      MutableLayout.reshape(mutableLayout, target, storage.length),
      dtype
    )

  def reshape[S <: AnyRank](target: Shape[S]): MutableNDArray[A, S] =
    try reshapeView(target)
    catch
      case _: NonContiguousLayout =>
        val owned = freezeCopy().reshapeCopy(target)
        new MutableNDArray(
          owned.storage,
          MutableLayout.owned(owned.shape, owned.size),
          dtype
        )

  def reshapeCopy[S <: AnyRank](target: Shape[S]): MutableNDArray[A, S] =
    val owned = freezeCopy().reshapeCopy(target)
    new MutableNDArray(
      owned.storage,
      MutableLayout.owned(owned.shape, owned.size),
      dtype
    )

  def freezeCopy(): NDArray[A, R] =
    val output = ProbeApi.allocate[A](size)(using dtype)
    CopyKernels.logical(storage, layout, output)
    new NDArray(output, Layout.contiguous(shape, size), dtype)

object MutableNDArray:
  def zeros[A, R <: AnyRank](shape: Shape[R])(using
      dtype: DType[A]
  ): MutableNDArray[A, R] =
    val storage = ProbeApi.allocate[A](shape.size)
    new MutableNDArray(storage, MutableLayout.owned(shape, shape.size), dtype)

  private[ravel] def requireSameShape(
      left: IArray[Int],
      right: IArray[Int]
  ): Unit =
    var same = left.length == right.length
    var i = 0
    while i < left.length && same do
      same = left(i) == right(i)
      i += 1
    if !same then
      throw ShapeMismatch(
        left.mkString("(", ", ", ")"),
        right.mkString("(", ", ", ")")
      )

extension [A, R <: AnyRank](array: NDArray[A, R])
  def mutableCopy: MutableNDArray[A, R] =
    val copied = array.copy
    new MutableNDArray(
      copied.storage,
      MutableLayout.owned(copied.shape, copied.size),
      copied.dtype
    )
