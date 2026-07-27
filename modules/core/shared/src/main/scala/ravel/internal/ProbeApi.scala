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
