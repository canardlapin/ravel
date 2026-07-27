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

extension [A, R <: AnyRank](array: NDArray[A, R])(using
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

  def product(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.productAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def productKeep(axis: Int): NDArray[A, R] =
    ReductionApi.productAxis[A, R, R](array, axis, keep = true)

  def sumAxes(axes: Int*): AnyNDArray[A] =
    if axes.isEmpty then array
    else
      val normalized = axes.map(array.layout.normalizedAxis)
      if normalized.distinct.size != normalized.size then
        throw InvalidAxis(axes.head, array.rank)
      var current: AnyNDArray[A] = array
      normalized.sorted.reverse.foreach { axis =>
        current = ReductionApi.sumAxis[A, AnyRank, AnyRank](
          current,
          axis,
          keep = false
        )
      }
      current

extension [A, R <: AnyRank](array: NDArray[A, R])(using
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

  def max(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.maximumAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def maxKeep(axis: Int): NDArray[A, R] =
    ReductionApi.maximumAxis[A, R, R](array, axis, keep = true)

  def argMin(axis: Int)(using CanDropAxis[R]): NDArray[Int, DropAxis[R]] =
    ReductionApi.argMinimumAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def argMinKeep(axis: Int): NDArray[Int, R] =
    ReductionApi.argMinimumAxis[A, R, R](array, axis, keep = true)

  def argMax(axis: Int)(using CanDropAxis[R]): NDArray[Int, DropAxis[R]] =
    ReductionApi.argMaximumAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def argMaxKeep(axis: Int): NDArray[Int, R] =
    ReductionApi.argMaximumAxis[A, R, R](array, axis, keep = true)

extension [A, R <: AnyRank](array: NDArray[A, R])(using
    @unused floating: FloatingDType[A]
)
  def mean: A =
    ReductionKernels.mean(array.storage, array.layout)

  def mean(axis: Int)(using CanDropAxis[R]): NDArray[A, DropAxis[R]] =
    ReductionApi.meanAxis[A, R, DropAxis[R]](array, axis, keep = false)

  def meanKeep(axis: Int): NDArray[A, R] =
    ReductionApi.meanAxis[A, R, R](array, axis, keep = true)
