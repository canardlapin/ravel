package ravel.internal

import ravel.DType

/**
 * Executable representation probe used to freeze the internal storage ABI.
 *
 * Production array operations call the same allocation and monomorphic kernel
 * entry points. `Storage[A]` is a sealed, platform-specific family.
 */
private[ravel] object ProbeApi:
  def allocate[A](size: Int)(using dtype: DType[A]): Storage[A] =
    PlatformStorage.allocate(size)

  def fill[A](storage: Storage[A], value: A): Unit =
    ProbeKernels.fill(storage, value)

  def copy[A](source: Storage[A], target: Storage[A], length: Int): Unit =
    ProbeKernels.copy(source, 0, target, 0, length)

  def get[A](storage: Storage[A], index: Int): A =
    ProbeKernels.get(storage, index)

  def set[A](storage: Storage[A], index: Int, value: A): Unit =
    ProbeKernels.set(storage, index, value)

  def getBoolean(storage: Storage[Boolean], index: Int): Boolean =
    ProbeKernels.getBoolean(storage, index)

  def getByte(storage: Storage[Byte], index: Int): Byte =
    ProbeKernels.getByte(storage, index)

  def getShort(storage: Storage[Short], index: Int): Short =
    ProbeKernels.getShort(storage, index)

  def getInt(storage: Storage[Int], index: Int): Int =
    ProbeKernels.getInt(storage, index)

  def getLong(storage: Storage[Long], index: Int): Long =
    ProbeKernels.getLong(storage, index)

  def getFloat(storage: Storage[Float], index: Int): Float =
    ProbeKernels.getFloat(storage, index)

  def getDouble(storage: Storage[Double], index: Int): Double =
    ProbeKernels.getDouble(storage, index)

  def setBoolean(storage: Storage[Boolean], index: Int, value: Boolean): Unit =
    ProbeKernels.setBoolean(storage, index, value)

  def setByte(storage: Storage[Byte], index: Int, value: Byte): Unit =
    ProbeKernels.setByte(storage, index, value)

  def setShort(storage: Storage[Short], index: Int, value: Short): Unit =
    ProbeKernels.setShort(storage, index, value)

  def setInt(storage: Storage[Int], index: Int, value: Int): Unit =
    ProbeKernels.setInt(storage, index, value)

  def setLong(storage: Storage[Long], index: Int, value: Long): Unit =
    ProbeKernels.setLong(storage, index, value)

  def setFloat(storage: Storage[Float], index: Int, value: Float): Unit =
    ProbeKernels.setFloat(storage, index, value)

  def setDouble(storage: Storage[Double], index: Int, value: Double): Unit =
    ProbeKernels.setDouble(storage, index, value)

  def add[A](left: Storage[A], right: Storage[A], out: Storage[A], size: Int): Unit =
    ProbeKernels.add(left, right, out, size)

  def negate[A](source: Storage[A], out: Storage[A], size: Int): Unit =
    ProbeKernels.negate(source, out, size)

  def addStrided[A](
      left: Storage[A],
      leftOffset: Int,
      leftStride: Int,
      right: Storage[A],
      rightOffset: Int,
      rightStride: Int,
      out: Storage[A],
      size: Int
  ): Unit =
    ProbeKernels.addStrided(
      left,
      leftOffset,
      leftStride,
      right,
      rightOffset,
      rightStride,
      out,
      size
    )
