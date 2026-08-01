package ravel

import ravel.internal.*
import scala.annotation.publicInBinary
import scala.compiletime.erasedValue

/** Allocation-free capability view for logical linear access to a whole, canonical immutable array.
  *
  * The opaque value is the original [[NDArray]] reference: successful refinement allocates no
  * wrapper and exposes neither platform storage nor physical offsets. Linear indices are logical
  * C-order indices in `[0, size)`.
  */
opaque type CanonicalArray[A, R <: AnyRank] <: NDArray[A, R] =
  NDArray[A, R]

object CanonicalArray:
  def from[A, R <: AnyRank](
      array: NDArray[A, R]
  ): Either[NonContiguousLayout, CanonicalArray[A, R]] =
    if array.isCanonicalLayout && array.isWholeBuffer then Right(array)
    else
      Left(
        NonContiguousLayout(
          "canonical linear access requires a whole canonical array"
        )
      )

  def require[A, R <: AnyRank](
      array: NDArray[A, R]
  ): CanonicalArray[A, R] =
    if array.isCanonicalLayout && array.isWholeBuffer then array
    else
      throw NonContiguousLayout(
        "canonical linear access requires a whole canonical array"
      )

  extension [A, R <: AnyRank](array: CanonicalArray[A, R])
    /** Read a logical C-order linear index in `[0, size)`.
      *
      * Ordinary `apply` overloads remain coordinate access inherited from [[NDArray]], including
      * negative element-index normalization.
      */
    inline def readLinear(index: Int): A =
      inline erasedValue[A] match
        case _: Boolean =>
          readBoolean(
            array.asInstanceOf[CanonicalArray[Boolean, R]],
            index
          ).asInstanceOf[A]
        case _: Byte =>
          readByte(
            array.asInstanceOf[CanonicalArray[Byte, R]],
            index
          ).asInstanceOf[A]
        case _: UInt8 =>
          readUInt8(
            array.asInstanceOf[CanonicalArray[UInt8, R]],
            index
          ).asInstanceOf[A]
        case _: Short =>
          readShort(
            array.asInstanceOf[CanonicalArray[Short, R]],
            index
          ).asInstanceOf[A]
        case _: UInt16 =>
          readUInt16(
            array.asInstanceOf[CanonicalArray[UInt16, R]],
            index
          ).asInstanceOf[A]
        case _: Int =>
          readInt(
            array.asInstanceOf[CanonicalArray[Int, R]],
            index
          ).asInstanceOf[A]
        case _: Long =>
          readLong(
            array.asInstanceOf[CanonicalArray[Long, R]],
            index
          ).asInstanceOf[A]
        case _: Float =>
          readFloat(
            array.asInstanceOf[CanonicalArray[Float, R]],
            index
          ).asInstanceOf[A]
        case _: Double =>
          readDouble(
            array.asInstanceOf[CanonicalArray[Double, R]],
            index
          ).asInstanceOf[A]
        case _ =>
          readGeneric(array, index)

  @publicInBinary private[ravel] def readGeneric[A, R <: AnyRank](
      array: CanonicalArray[A, R],
      index: Int
  ): A =
    ProbeApi.get(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readBoolean[R <: AnyRank](
      array: CanonicalArray[Boolean, R],
      index: Int
  ): Boolean =
    ProbeApi.getBoolean(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readByte[R <: AnyRank](
      array: CanonicalArray[Byte, R],
      index: Int
  ): Byte =
    ProbeApi.getByte(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readShort[R <: AnyRank](
      array: CanonicalArray[Short, R],
      index: Int
  ): Short =
    ProbeApi.getShort(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readUInt8[R <: AnyRank](
      array: CanonicalArray[UInt8, R],
      index: Int
  ): UInt8 =
    ProbeApi.getUInt8(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readUInt16[R <: AnyRank](
      array: CanonicalArray[UInt16, R],
      index: Int
  ): UInt16 =
    ProbeApi.getUInt16(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readInt[R <: AnyRank](
      array: CanonicalArray[Int, R],
      index: Int
  ): Int =
    ProbeApi.getInt(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readLong[R <: AnyRank](
      array: CanonicalArray[Long, R],
      index: Int
  ): Long =
    ProbeApi.getLong(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readFloat[R <: AnyRank](
      array: CanonicalArray[Float, R],
      index: Int
  ): Float =
    ProbeApi.getFloat(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readDouble[R <: AnyRank](
      array: CanonicalArray[Double, R],
      index: Int
  ): Double =
    ProbeApi.getDouble(array.storage, checkedIndex(index, array.size))

  private def checkedIndex(index: Int, size: Int): Int =
    if index < 0 || index >= size then throw InvalidIndex.LinearOutOfBounds(index, size)
    index

/** Allocation-free capability view for logical linear access to a whole, canonical mutable array.
  *
  * Refinement returns the original [[MutableNDArray]] reference. It does not create another storage
  * owner or weaken the mutable array's aliasing rules.
  */
opaque type MutableCanonicalArray[A, R <: AnyRank] <: MutableNDArray[A, R] =
  MutableNDArray[A, R]

object MutableCanonicalArray:
  def from[A, R <: AnyRank](
      array: MutableNDArray[A, R]
  ): Either[NonContiguousLayout, MutableCanonicalArray[A, R]] =
    if array.isCanonicalLayout && array.isWholeBuffer then Right(array)
    else
      Left(
        NonContiguousLayout(
          "canonical linear access requires a whole canonical mutable array"
        )
      )

  def require[A, R <: AnyRank](
      array: MutableNDArray[A, R]
  ): MutableCanonicalArray[A, R] =
    if array.isCanonicalLayout && array.isWholeBuffer then array
    else
      throw NonContiguousLayout(
        "canonical linear access requires a whole canonical mutable array"
      )

  extension [A, R <: AnyRank](
      array: MutableCanonicalArray[A, R]
  )
    /** Read a logical C-order linear index in `[0, size)`.
      *
      * Ordinary `apply` overloads remain coordinate access inherited from [[MutableNDArray]],
      * including negative element-index normalization.
      */
    inline def readLinear(index: Int): A =
      inline erasedValue[A] match
        case _: Boolean =>
          readBoolean(
            array.asInstanceOf[MutableCanonicalArray[Boolean, R]],
            index
          ).asInstanceOf[A]
        case _: Byte =>
          readByte(
            array.asInstanceOf[MutableCanonicalArray[Byte, R]],
            index
          ).asInstanceOf[A]
        case _: UInt8 =>
          readUInt8(
            array.asInstanceOf[MutableCanonicalArray[UInt8, R]],
            index
          ).asInstanceOf[A]
        case _: Short =>
          readShort(
            array.asInstanceOf[MutableCanonicalArray[Short, R]],
            index
          ).asInstanceOf[A]
        case _: UInt16 =>
          readUInt16(
            array.asInstanceOf[MutableCanonicalArray[UInt16, R]],
            index
          ).asInstanceOf[A]
        case _: Int =>
          readInt(
            array.asInstanceOf[MutableCanonicalArray[Int, R]],
            index
          ).asInstanceOf[A]
        case _: Long =>
          readLong(
            array.asInstanceOf[MutableCanonicalArray[Long, R]],
            index
          ).asInstanceOf[A]
        case _: Float =>
          readFloat(
            array.asInstanceOf[MutableCanonicalArray[Float, R]],
            index
          ).asInstanceOf[A]
        case _: Double =>
          readDouble(
            array.asInstanceOf[MutableCanonicalArray[Double, R]],
            index
          ).asInstanceOf[A]
        case _ =>
          readGeneric(array, index)

    /** Write a logical C-order linear index in `[0, size)`.
      *
      * Ordinary `update` overloads remain coordinate writes inherited from [[MutableNDArray]],
      * including negative element-index normalization.
      */
    inline def writeLinear(index: Int, value: A): Unit =
      inline erasedValue[A] match
        case _: Boolean =>
          writeBoolean(
            array.asInstanceOf[MutableCanonicalArray[Boolean, R]],
            index,
            value.asInstanceOf[Boolean]
          )
        case _: Byte =>
          writeByte(
            array.asInstanceOf[MutableCanonicalArray[Byte, R]],
            index,
            value.asInstanceOf[Byte]
          )
        case _: UInt8 =>
          writeUInt8(
            array.asInstanceOf[MutableCanonicalArray[UInt8, R]],
            index,
            value.asInstanceOf[UInt8]
          )
        case _: Short =>
          writeShort(
            array.asInstanceOf[MutableCanonicalArray[Short, R]],
            index,
            value.asInstanceOf[Short]
          )
        case _: UInt16 =>
          writeUInt16(
            array.asInstanceOf[MutableCanonicalArray[UInt16, R]],
            index,
            value.asInstanceOf[UInt16]
          )
        case _: Int =>
          writeInt(
            array.asInstanceOf[MutableCanonicalArray[Int, R]],
            index,
            value.asInstanceOf[Int]
          )
        case _: Long =>
          writeLong(
            array.asInstanceOf[MutableCanonicalArray[Long, R]],
            index,
            value.asInstanceOf[Long]
          )
        case _: Float =>
          writeFloat(
            array.asInstanceOf[MutableCanonicalArray[Float, R]],
            index,
            value.asInstanceOf[Float]
          )
        case _: Double =>
          writeDouble(
            array.asInstanceOf[MutableCanonicalArray[Double, R]],
            index,
            value.asInstanceOf[Double]
          )
        case _ =>
          writeGeneric(array, index, value)

  @publicInBinary private[ravel] def readGeneric[A, R <: AnyRank](
      array: MutableCanonicalArray[A, R],
      index: Int
  ): A =
    ProbeApi.get(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readBoolean[R <: AnyRank](
      array: MutableCanonicalArray[Boolean, R],
      index: Int
  ): Boolean =
    ProbeApi.getBoolean(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readByte[R <: AnyRank](
      array: MutableCanonicalArray[Byte, R],
      index: Int
  ): Byte =
    ProbeApi.getByte(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readShort[R <: AnyRank](
      array: MutableCanonicalArray[Short, R],
      index: Int
  ): Short =
    ProbeApi.getShort(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readUInt8[R <: AnyRank](
      array: MutableCanonicalArray[UInt8, R],
      index: Int
  ): UInt8 =
    ProbeApi.getUInt8(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readUInt16[R <: AnyRank](
      array: MutableCanonicalArray[UInt16, R],
      index: Int
  ): UInt16 =
    ProbeApi.getUInt16(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readInt[R <: AnyRank](
      array: MutableCanonicalArray[Int, R],
      index: Int
  ): Int =
    ProbeApi.getInt(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readLong[R <: AnyRank](
      array: MutableCanonicalArray[Long, R],
      index: Int
  ): Long =
    ProbeApi.getLong(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readFloat[R <: AnyRank](
      array: MutableCanonicalArray[Float, R],
      index: Int
  ): Float =
    ProbeApi.getFloat(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def readDouble[R <: AnyRank](
      array: MutableCanonicalArray[Double, R],
      index: Int
  ): Double =
    ProbeApi.getDouble(array.storage, checkedIndex(index, array.size))

  @publicInBinary private[ravel] def writeGeneric[A, R <: AnyRank](
      array: MutableCanonicalArray[A, R],
      index: Int,
      value: A
  ): Unit =
    ProbeApi.set(array.storage, checkedIndex(index, array.size), value)

  @publicInBinary private[ravel] def writeBoolean[R <: AnyRank](
      array: MutableCanonicalArray[Boolean, R],
      index: Int,
      value: Boolean
  ): Unit =
    ProbeApi.setBoolean(
      array.storage,
      checkedIndex(index, array.size),
      value
    )

  @publicInBinary private[ravel] def writeByte[R <: AnyRank](
      array: MutableCanonicalArray[Byte, R],
      index: Int,
      value: Byte
  ): Unit =
    ProbeApi.setByte(array.storage, checkedIndex(index, array.size), value)

  @publicInBinary private[ravel] def writeShort[R <: AnyRank](
      array: MutableCanonicalArray[Short, R],
      index: Int,
      value: Short
  ): Unit =
    ProbeApi.setShort(array.storage, checkedIndex(index, array.size), value)

  @publicInBinary private[ravel] def writeUInt8[R <: AnyRank](
      array: MutableCanonicalArray[UInt8, R],
      index: Int,
      value: UInt8
  ): Unit =
    ProbeApi.setUInt8(array.storage, checkedIndex(index, array.size), value)

  @publicInBinary private[ravel] def writeUInt16[R <: AnyRank](
      array: MutableCanonicalArray[UInt16, R],
      index: Int,
      value: UInt16
  ): Unit =
    ProbeApi.setUInt16(array.storage, checkedIndex(index, array.size), value)

  @publicInBinary private[ravel] def writeInt[R <: AnyRank](
      array: MutableCanonicalArray[Int, R],
      index: Int,
      value: Int
  ): Unit =
    ProbeApi.setInt(array.storage, checkedIndex(index, array.size), value)

  @publicInBinary private[ravel] def writeLong[R <: AnyRank](
      array: MutableCanonicalArray[Long, R],
      index: Int,
      value: Long
  ): Unit =
    ProbeApi.setLong(array.storage, checkedIndex(index, array.size), value)

  @publicInBinary private[ravel] def writeFloat[R <: AnyRank](
      array: MutableCanonicalArray[Float, R],
      index: Int,
      value: Float
  ): Unit =
    ProbeApi.setFloat(array.storage, checkedIndex(index, array.size), value)

  @publicInBinary private[ravel] def writeDouble[R <: AnyRank](
      array: MutableCanonicalArray[Double, R],
      index: Int,
      value: Double
  ): Unit =
    ProbeApi.setDouble(
      array.storage,
      checkedIndex(index, array.size),
      value
    )

  private def checkedIndex(index: Int, size: Int): Int =
    if index < 0 || index >= size then throw InvalidIndex.LinearOutOfBounds(index, size)
    index
