package ravel

import ravel.internal.*
import scala.annotation.unused
import scala.annotation.publicInBinary
import scala.compiletime.erasedValue

final class MutableNDArray[A, R <: AnyRank] private[ravel] (
    private[ravel] val storage: Storage[A],
    private[ravel] val mutableLayout: MutableLayout,
    val dtype: DType[A]
) extends ReadableArray[A, R]:
  private[ravel] def layout: Layout = mutableLayout.underlying

  val shape: Shape[R] =
    Shape.retag[R](layout.shapeValue)

  def rank: Int = layout.rank
  def size: Int = layout.size
  def isContiguous: Boolean = layout.isCContiguous
  def isCanonicalLayout: Boolean = layout.isCanonicalLayout
  def isWholeBuffer: Boolean = layout.isWholeBuffer(storage.length)

  inline def apply(i: Int)(using R <:< Rank[1]): A =
    readPrimitive(physicalIndex1(i))

  inline def apply(i: Int, j: Int)(using R <:< Rank[2]): A =
    readPrimitive(physicalIndex2(i, j))

  inline def apply(i: Int, j: Int, k: Int)(using R <:< Rank[3]): A =
    readPrimitive(physicalIndex3(i, j, k))

  inline def apply(i: Int, j: Int, k: Int, l: Int)(using R <:< Rank[4]): A =
    readPrimitive(physicalIndex4(i, j, k, l))

  def at(indices: IArray[Int]): A =
    ProbeApi.get(storage, layout.physicalIndex(indices))

  inline def update(i: Int, value: A)(using R <:< Rank[1]): Unit =
    writePrimitive(physicalIndex1(i), value)

  inline def update(i: Int, j: Int, value: A)(using R <:< Rank[2]): Unit =
    writePrimitive(physicalIndex2(i, j), value)

  inline def update(i: Int, j: Int, k: Int, value: A)(using R <:< Rank[3]): Unit =
    writePrimitive(physicalIndex3(i, j, k), value)

  inline def update(i: Int, j: Int, k: Int, l: Int, value: A)(using
      R <:< Rank[4]
  ): Unit =
    writePrimitive(physicalIndex4(i, j, k, l), value)

  def updateAt(indices: IArray[Int], value: A): Unit =
    ProbeApi.set(storage, layout.physicalIndex(indices), value)

  def requireRank[N <: Int](using
      expected: ValueOf[N]
  ): Either[RankMismatch, MutableNDArray[A, Rank[N]]] =
    if rank == expected.value then Right(this.asInstanceOf[MutableNDArray[A, Rank[N]]])
    else Left(RankMismatch(expected.value, rank))

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

  def cast[B](using source: NumericDType[A], target: NumericDType[B]): NDArray[B, R] =
    val output = ProbeApi.allocate[B](size)(using target)
    CastKernels.convert(storage, layout, output, source, target)
    new NDArray(output, Layout.contiguous(shape, size), target)

  def convert[B](
      policy: ConversionPolicy = ConversionPolicy()
  )(using
      source: NumericDType[A],
      target: NumericDType[B]
  ): Either[ConversionError, NDArray[B, R]] =
    PolicyCastKernels
      .convert(storage, layout, source, target, policy)
      .map { output =>
        new NDArray(output, Layout.contiguous(shape, size), target)
      }

  def fill(value: A): Unit =
    if layout.isCContiguous && layout.offset == 0 && layout.size == storage.length then
      ProbeApi.fill(storage, value)
    else layout.foreachPhysicalIndex(index => ProbeApi.set(storage, index, value))

  def assign(source: ReadableArray[A, ?]): Unit =
    MutableNDArray.requireSameShape(layout.shape, source.layout.shape)
    if storage.asInstanceOf[AnyRef] eq source.storage.asInstanceOf[AnyRef] then
      throw new IllegalArgumentException("mutable assignment source must not alias destination")
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
    val canonical = Slice
      .from(range)
      .fold(
        error => throw InvalidSlice(error.reason),
        identity
      )
    slice(axis, canonical)

  /** Checked exact, non-clipping narrow with negative-start normalization. */
  def narrowChecked(
      axis: Int,
      from: Int,
      length: Int
  ): Either[InvalidNarrow, MutableNDArray[A, R]] =
    NarrowPlan.from(layout.shape, axis, from, length).map { plan =>
      new MutableNDArray[A, R](
        storage,
        MutableLayout.narrow(mutableLayout, plan, storage.length),
        dtype
      )
    }

  def narrow(axis: Int, from: Int, length: Int): MutableNDArray[A, R] =
    narrowChecked(axis, from, length).fold(
      error => throw InvalidNarrowException(error),
      identity
    )

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

  /** Checked rank-preserving axis permutation. */
  def permuteAxesChecked(
      order: Int*
  ): Either[PermutationError, MutableNDArray[A, R]] =
    PermutationPlan.from(rank, order).map { plan =>
      new MutableNDArray(
        storage,
        MutableLayout.permute(mutableLayout, plan, storage.length),
        dtype
      )
    }

  def permuteAxes(order: Int*): MutableNDArray[A, R] =
    permuteAxesChecked(order*).fold(
      error => throw InvalidPermutationException(error),
      identity
    )

  def transpose(using R <:< Rank[2]): MutableNDArray[A, Rank[2]] =
    swapAxes(0, 1).asInstanceOf[MutableNDArray[A, Rank[2]]]

  /** Checked rank-two transpose for arrays whose rank is not statically refined. */
  def transpose2D: Either[RankMismatch, MutableNDArray[A, Rank[2]]] =
    if rank == 2 then Right(swapAxes(0, 1).asInstanceOf[MutableNDArray[A, Rank[2]]])
    else Left(RankMismatch(2, rank))

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
    catch case _: NonContiguousLayout => materializeReshape(target)

  def reshapeCopy[S <: AnyRank](target: Shape[S]): MutableNDArray[A, S] =
    materializeReshape(target)

  private def materializeReshape[S <: AnyRank](target: Shape[S]): MutableNDArray[A, S] =
    if target.size != size then
      throw InvalidShape(
        s"cannot reshape mutable array of size $size into shape $target"
      )
    val output = ProbeApi.allocate[A](size)(using dtype)
    CopyKernels.logical(storage, layout, output)
    new MutableNDArray(
      output,
      MutableLayout.owned(target, target.size),
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
