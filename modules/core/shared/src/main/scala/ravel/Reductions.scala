package ravel

import ravel.internal.*
import scala.annotation.unused

sealed trait SumAs[A, B]:
  private[ravel] def apply(array: ArraySource[A, ?]): B

object SumAs:
  given intToLong: SumAs[Int, Long] with
    private[ravel] def apply(array: ArraySource[Int, ?]): Long =
      ReductionKernels.sumAsLong(array.storage, array.layout)

  given floatToDouble: SumAs[Float, Double] with
    private[ravel] def apply(array: ArraySource[Float, ?]): Double =
      ReductionKernels.sumAsDouble(array.storage, array.layout)

extension [R <: AnyRank](array: ReadableArray[Boolean, R])
  def all: Boolean =
    ReductionKernels.all(array.storage, array.layout)

  def any: Boolean =
    ReductionKernels.any(array.storage, array.layout)

  def countTrue: Int =
    ReductionKernels.countTrue(array.storage, array.layout)

  def all(axis: Int)(using CanDropAxis[R]): NDArray[Boolean, DropAxis[R]] =
    ReductionApi.allAxis[R, DropAxis[R]](array, axis, keep = false)

  def allKeep(axis: Int): NDArray[Boolean, R] =
    ReductionApi.allAxis[R, R](array, axis, keep = true)

  def allKeepDims(axis: Int): NDArray[Boolean, R] = allKeep(axis)

  def any(axis: Int)(using CanDropAxis[R]): NDArray[Boolean, DropAxis[R]] =
    ReductionApi.anyAxis[R, DropAxis[R]](array, axis, keep = false)

  def anyKeep(axis: Int): NDArray[Boolean, R] =
    ReductionApi.anyAxis[R, R](array, axis, keep = true)

  def anyKeepDims(axis: Int): NDArray[Boolean, R] = anyKeep(axis)

  def countTrue(axis: Int)(using CanDropAxis[R]): NDArray[Int, DropAxis[R]] =
    ReductionApi.countTrueAxis[R, DropAxis[R]](array, axis, keep = false)

  def countTrueKeep(axis: Int): NDArray[Int, R] =
    ReductionApi.countTrueAxis[R, R](array, axis, keep = true)

  def countTrueKeepDims(axis: Int): NDArray[Int, R] = countTrueKeep(axis)

  def all(axes: Axes): AnyNDArray[Boolean] =
    ReductionApi.allAxes[R, AnyRank](array, axes, keep = false)

  def allKeep(axes: Axes): NDArray[Boolean, R] =
    ReductionApi.allAxes[R, R](array, axes, keep = true)

  def allKeepDims(axes: Axes): NDArray[Boolean, R] = allKeep(axes)

  def any(axes: Axes): AnyNDArray[Boolean] =
    ReductionApi.anyAxes[R, AnyRank](array, axes, keep = false)

  def anyKeep(axes: Axes): NDArray[Boolean, R] =
    ReductionApi.anyAxes[R, R](array, axes, keep = true)

  def anyKeepDims(axes: Axes): NDArray[Boolean, R] = anyKeep(axes)

  def countTrue(axes: Axes): AnyNDArray[Int] =
    ReductionApi.countTrueAxes[R, AnyRank](array, axes, keep = false)

  def countTrueKeep(axes: Axes): NDArray[Int, R] =
    ReductionApi.countTrueAxes[R, R](array, axes, keep = true)

  def countTrueKeepDims(axes: Axes): NDArray[Int, R] = countTrueKeep(axes)

  def allAxes(axes: Int*): AnyNDArray[Boolean] =
    all(Axes.require(array.rank, axes*))

  def anyAxes(axes: Int*): AnyNDArray[Boolean] =
    any(Axes.require(array.rank, axes*))

  def countTrueAxes(axes: Int*): AnyNDArray[Int] =
    countTrue(Axes.require(array.rank, axes*))

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused arithmetic: ArithmeticDType[A]
)
  def sum: A =
    ReductionKernels.sum(array.storage, array.layout)

  def product: A =
    ReductionKernels.product(array.storage, array.layout)

  def sumAs[B](using widening: SumAs[A, B]): B =
    widening(array)

  def sum(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.sumAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def sumKeep(axis: Int): NDArray[A, R] =
    ReductionApi.sumAxis[A, R, R](array, axis, keep = true)

  def sumKeepDims(axis: Int): NDArray[A, R] = sumKeep(axis)

  def product(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.productAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def productKeep(axis: Int): NDArray[A, R] =
    ReductionApi.productAxis[A, R, R](array, axis, keep = true)

  def productKeepDims(axis: Int): NDArray[A, R] = productKeep(axis)

  def sum(axes: Axes): AnyNDArray[A] =
    ReductionApi.sumAxes[A, R, AnyRank](array, axes, keep = false)

  def sumKeep(axes: Axes): NDArray[A, R] =
    ReductionApi.sumAxes[A, R, R](array, axes, keep = true)

  def sumKeepDims(axes: Axes): NDArray[A, R] = sumKeep(axes)

  def product(axes: Axes): AnyNDArray[A] =
    ReductionApi.productAxes[A, R, AnyRank](array, axes, keep = false)

  def productKeep(axes: Axes): NDArray[A, R] =
    ReductionApi.productAxes[A, R, R](array, axes, keep = true)

  def productKeepDims(axes: Axes): NDArray[A, R] = productKeep(axes)

  /** Convenience multi-axis sum. An empty axis list still creates an owned copy. */
  def sumAxes(axes: Int*): AnyNDArray[A] =
    sum(Axes.require(array.rank, axes*))

  def productAxes(axes: Int*): AnyNDArray[A] =
    product(Axes.require(array.rank, axes*))

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused ordered: OrderedDType[A]
)
  def min: A =
    ReductionKernels.minimum(array.storage, array.layout)

  def max: A =
    ReductionKernels.maximum(array.storage, array.layout)

  def argMin: Int =
    ReductionKernels.argMinimum(array.storage, array.layout)

  def argMax: Int =
    ReductionKernels.argMaximum(array.storage, array.layout)

  def min(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.minimumAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def minKeep(axis: Int): NDArray[A, R] =
    ReductionApi.minimumAxis[A, R, R](array, axis, keep = true)

  def minKeepDims(axis: Int): NDArray[A, R] = minKeep(axis)

  def max(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.maximumAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def maxKeep(axis: Int): NDArray[A, R] =
    ReductionApi.maximumAxis[A, R, R](array, axis, keep = true)

  def maxKeepDims(axis: Int): NDArray[A, R] = maxKeep(axis)

  def min(axes: Axes): AnyNDArray[A] =
    ReductionApi.minimumAxes[A, R, AnyRank](array, axes, keep = false)

  def minKeep(axes: Axes): NDArray[A, R] =
    ReductionApi.minimumAxes[A, R, R](array, axes, keep = true)

  def minKeepDims(axes: Axes): NDArray[A, R] = minKeep(axes)

  def max(axes: Axes): AnyNDArray[A] =
    ReductionApi.maximumAxes[A, R, AnyRank](array, axes, keep = false)

  def maxKeep(axes: Axes): NDArray[A, R] =
    ReductionApi.maximumAxes[A, R, R](array, axes, keep = true)

  def maxKeepDims(axes: Axes): NDArray[A, R] = maxKeep(axes)

  def minAxes(axes: Int*): AnyNDArray[A] =
    min(Axes.require(array.rank, axes*))

  def maxAxes(axes: Int*): AnyNDArray[A] =
    max(Axes.require(array.rank, axes*))

  def argMin(axis: Int)(using CanDropAxis[R]): NDArray[Int, DropAxis[R]] =
    ReductionApi.argMinimumAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def argMinKeep(axis: Int): NDArray[Int, R] =
    ReductionApi.argMinimumAxis[A, R, R](array, axis, keep = true)

  def argMinKeepDims(axis: Int): NDArray[Int, R] = argMinKeep(axis)

  def argMax(axis: Int)(using CanDropAxis[R]): NDArray[Int, DropAxis[R]] =
    ReductionApi.argMaximumAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def argMaxKeep(axis: Int): NDArray[Int, R] =
    ReductionApi.argMaximumAxis[A, R, R](array, axis, keep = true)

  def argMaxKeepDims(axis: Int): NDArray[Int, R] = argMaxKeep(axis)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused floating: FloatingDType[A]
)
  def mean: A =
    ReductionKernels.mean(array.storage, array.layout)

  def mean(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.meanAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def meanKeep(axis: Int): NDArray[A, R] =
    ReductionApi.meanAxis[A, R, R](array, axis, keep = true)

  def meanKeepDims(axis: Int): NDArray[A, R] = meanKeep(axis)

  def mean(axes: Axes): AnyNDArray[A] =
    ReductionApi.meanAxes[A, R, AnyRank](array, axes, keep = false)

  def meanKeep(axes: Axes): NDArray[A, R] =
    ReductionApi.meanAxes[A, R, R](array, axes, keep = true)

  def meanKeepDims(axes: Axes): NDArray[A, R] = meanKeep(axes)

  def meanAxes(axes: Int*): AnyNDArray[A] =
    mean(Axes.require(array.rank, axes*))
