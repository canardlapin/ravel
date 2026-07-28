package ravel

import ravel.internal.*
import scala.annotation.unused

sealed trait SumAs[A, B]:
  private[ravel] def apply(array: NDArray[A, ?]): B

object SumAs:
  given intToLong: SumAs[Int, Long] with
    private[ravel] def apply(array: NDArray[Int, ?]): Long =
      ReductionKernels.sumAsLong(array.storage, array.layout)

  given floatToDouble: SumAs[Float, Double] with
    private[ravel] def apply(array: NDArray[Float, ?]): Double =
      ReductionKernels.sumAsDouble(array.storage, array.layout)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused arithmetic: ArithmeticDType[A]
)
  def sum: A =
    val owned = array.toNDArray
    ReductionKernels.sum(owned.storage, owned.layout)

  def product: A =
    val owned = array.toNDArray
    ReductionKernels.product(owned.storage, owned.layout)

  def sumAs[B](using widening: SumAs[A, B]): B =
    widening(array.toNDArray)

  def sum(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.sumAxis[A, R, DropAxis[R]](array.toNDArray, axis, keep = false)

  def sumKeep(axis: Int): NDArray[A, R] =
    ReductionApi.sumAxis[A, R, R](array.toNDArray, axis, keep = true)

  def sumKeepDims(axis: Int): NDArray[A, R] = sumKeep(axis)

  def product(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.productAxis[A, R, DropAxis[R]](array.toNDArray, axis, keep = false)

  def productKeep(axis: Int): NDArray[A, R] =
    ReductionApi.productAxis[A, R, R](array.toNDArray, axis, keep = true)

  def productKeepDims(axis: Int): NDArray[A, R] = productKeep(axis)

  def sumAxes(axes: Int*): AnyNDArray[A] =
    if axes.isEmpty then array.toNDArray
    else
      val owned = array.toNDArray
      val normalized = axes.map(owned.layout.normalizedAxis)
      if normalized.distinct.size != normalized.size then throw InvalidAxis(axes.head, owned.rank)
      var current: AnyNDArray[A] = owned
      normalized.sorted.reverse.foreach { axis =>
        current = ReductionApi.sumAxis[A, AnyRank, AnyRank](
          current,
          axis,
          keep = false
        )
      }
      current

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused ordered: OrderedDType[A]
)
  def min: A =
    val owned = array.toNDArray
    ReductionKernels.minimum(owned.storage, owned.layout)

  def max: A =
    val owned = array.toNDArray
    ReductionKernels.maximum(owned.storage, owned.layout)

  def argMin: Int =
    val owned = array.toNDArray
    ReductionKernels.argMinimum(owned.storage, owned.layout)

  def argMax: Int =
    val owned = array.toNDArray
    ReductionKernels.argMaximum(owned.storage, owned.layout)

  def min(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.minimumAxis[A, R, DropAxis[R]](array.toNDArray, axis, keep = false)

  def minKeep(axis: Int): NDArray[A, R] =
    ReductionApi.minimumAxis[A, R, R](array.toNDArray, axis, keep = true)

  def minKeepDims(axis: Int): NDArray[A, R] = minKeep(axis)

  def max(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.maximumAxis[A, R, DropAxis[R]](array.toNDArray, axis, keep = false)

  def maxKeep(axis: Int): NDArray[A, R] =
    ReductionApi.maximumAxis[A, R, R](array.toNDArray, axis, keep = true)

  def maxKeepDims(axis: Int): NDArray[A, R] = maxKeep(axis)

  def argMin(axis: Int)(using CanDropAxis[R]): NDArray[Int, DropAxis[R]] =
    ReductionApi.argMinimumAxis[A, R, DropAxis[R]](array.toNDArray, axis, keep = false)

  def argMinKeep(axis: Int): NDArray[Int, R] =
    ReductionApi.argMinimumAxis[A, R, R](array.toNDArray, axis, keep = true)

  def argMinKeepDims(axis: Int): NDArray[Int, R] = argMinKeep(axis)

  def argMax(axis: Int)(using CanDropAxis[R]): NDArray[Int, DropAxis[R]] =
    ReductionApi.argMaximumAxis[A, R, DropAxis[R]](array.toNDArray, axis, keep = false)

  def argMaxKeep(axis: Int): NDArray[Int, R] =
    ReductionApi.argMaximumAxis[A, R, R](array.toNDArray, axis, keep = true)

  def argMaxKeepDims(axis: Int): NDArray[Int, R] = argMaxKeep(axis)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused floating: FloatingDType[A]
)
  def mean: A =
    val owned = array.toNDArray
    ReductionKernels.mean(owned.storage, owned.layout)

  def mean(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.meanAxis[A, R, DropAxis[R]](array.toNDArray, axis, keep = false)

  def meanKeep(axis: Int): NDArray[A, R] =
    ReductionApi.meanAxis[A, R, R](array.toNDArray, axis, keep = true)

  def meanKeepDims(axis: Int): NDArray[A, R] = meanKeep(axis)
