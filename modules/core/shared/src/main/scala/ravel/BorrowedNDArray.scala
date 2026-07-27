package ravel

import ravel.internal.*
import scala.annotation.unused

/**
 * A read-only array view whose storage remains externally mutable.
 *
 * This type is deliberately not a subtype of [[NDArray]]. Structural views
 * retain the borrowed marker; computations and materialization return owned
 * `NDArray` values.
 */
final class BorrowedNDArray[A, R <: AnyRank] private[ravel] (
    private[ravel] val underlying: NDArray[A, R]
):
  val dtype: DType[A] = underlying.dtype
  val shape: Shape[R] = underlying.shape
  def rank: Int = underlying.rank
  def size: Int = underlying.size
  def isContiguous: Boolean = underlying.isContiguous

  inline def apply(i: Int): A = underlying(i)
  inline def apply(i: Int, j: Int): A = underlying(i, j)
  inline def apply(i: Int, j: Int, k: Int): A = underlying(i, j, k)
  inline def apply(i: Int, j: Int, k: Int, l: Int): A = underlying(i, j, k, l)
  def at(indices: IArray[Int]): A = underlying.at(indices)
  def foreachElement(f: A => Unit): Unit = underlying.foreachElement(f)
  def foreachIndex(f: IArray[Int] => Unit): Unit = underlying.foreachIndex(f)
  def elementsIterator: Iterator[A] = underlying.elementsIterator

  def requireRank[N <: Int](using
      expected: ValueOf[N]
  ): Either[RankMismatch, BorrowedNDArray[A, Rank[N]]] =
    if rank == expected.value then
      Right(this.asInstanceOf[BorrowedNDArray[A, Rank[N]]])
    else Left(RankMismatch(expected.value, rank))

  def select(axis: Int, index: Int)(using
      CanDropAxis[R]
  ): BorrowedNDArray[A, DropAxis[R]] =
    new BorrowedNDArray(underlying.select(axis, index))

  def slice(axis: Int, slice: Slice): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.slice(axis, slice))

  def slice(axis: Int, range: Range): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.slice(axis, range))

  def narrow(axis: Int, from: Int, length: Int): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.narrow(axis, from, length))

  def reverse(axis: Int): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.reverse(axis))

  def swapAxes(first: Int, second: Int): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.swapAxes(first, second))

  def permuteAxes(order: Int*): BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.permuteAxes(order*))

  def transpose: BorrowedNDArray[A, R] =
    new BorrowedNDArray(underlying.transpose)

  def newAxis(axis: Int): BorrowedNDArray[A, AddAxis[R]] =
    new BorrowedNDArray(underlying.newAxis(axis))

  def squeeze(axis: Int)(using
      CanDropAxis[R]
  ): BorrowedNDArray[A, DropAxis[R]] =
    new BorrowedNDArray(underlying.squeeze(axis))

  def broadcastTo[S <: AnyRank](
      target: Shape[S]
  ): BorrowedNDArray[A, S] =
    new BorrowedNDArray(underlying.broadcastTo(target))

  def reshapeView[S <: AnyRank](
      target: Shape[S]
  ): BorrowedNDArray[A, S] =
    new BorrowedNDArray(underlying.reshapeView(target))

  /** Always materializes owned storage. */
  def copy: NDArray[A, R] = underlying.copy

  /** Borrowed contiguity cannot erase ownership provenance, so this copies. */
  def contiguous: NDArray[A, R] = underlying.copy

  def flattenCopy: Array1[A] = underlying.flattenCopy

  def cast[B](using
      source: NumericDType[A],
      target: NumericDType[B]
  ): NDArray[B, R] =
    underlying.cast[B]

  def map[B](f: A => B)(using DType[B]): NDArray[B, R] =
    underlying.map(f)

  def unary_+(using @unused arithmetic: ArithmeticDType[A]): NDArray[A, R] =
    underlying.copy

  def unary_-(using @unused arithmetic: ArithmeticDType[A]): NDArray[A, R] =
    KernelApi.unary(KernelOp.Negate, underlying)

  def +[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  )(using @unused arithmetic: ArithmeticDType[A]): NDArray[A, BorrowedOperandRank[A, R, B]] =
    borrowedOperand(KernelOp.Add, other)

  def -[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  )(using @unused arithmetic: ArithmeticDType[A]): NDArray[A, BorrowedOperandRank[A, R, B]] =
    borrowedOperand(KernelOp.Subtract, other)

  def *[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  )(using @unused arithmetic: ArithmeticDType[A]): NDArray[A, BorrowedOperandRank[A, R, B]] =
    borrowedOperand(KernelOp.Multiply, other)

  def /[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  )(using @unused arithmetic: ArithmeticDType[A]): NDArray[A, BorrowedOperandRank[A, R, B]] =
    borrowedOperand(KernelOp.Divide, other)

  def abs(using @unused arithmetic: ArithmeticDType[A]): NDArray[A, R] =
    KernelApi.unary(KernelOp.Absolute, underlying)

  def sum(using @unused arithmetic: ArithmeticDType[A]): A =
    ReductionKernels.sum(underlying.storage, underlying.layout)

  def product(using @unused arithmetic: ArithmeticDType[A]): A =
    ReductionKernels.product(underlying.storage, underlying.layout)

  def sumAs[B](using
      @unused arithmetic: ArithmeticDType[A],
      widening: SumAs[A, B]
  ): B =
    widening(underlying)

  def sum(axis: Int)(using
      @unused arithmetic: ArithmeticDType[A],
      @unused canDrop: CanDropAxis[R]
  ): NDArray[A, DropAxis[R]] =
    ReductionApi.sumAxis[A, R, DropAxis[R]](underlying, axis, keep = false)

  def sumKeep(axis: Int)(using
      @unused arithmetic: ArithmeticDType[A]
  ): NDArray[A, R] =
    ReductionApi.sumAxis[A, R, R](underlying, axis, keep = true)

  def product(axis: Int)(using
      @unused arithmetic: ArithmeticDType[A],
      @unused canDrop: CanDropAxis[R]
  ): NDArray[A, DropAxis[R]] =
    ReductionApi.productAxis[A, R, DropAxis[R]](underlying, axis, keep = false)

  def productKeep(axis: Int)(using
      @unused arithmetic: ArithmeticDType[A]
  ): NDArray[A, R] =
    ReductionApi.productAxis[A, R, R](underlying, axis, keep = true)

  def sumAxes(axes: Int*)(using
      @unused arithmetic: ArithmeticDType[A]
  ): AnyNDArray[A] =
    underlying.sumAxes(axes*)

  def min(using @unused ordered: OrderedDType[A]): A =
    ReductionKernels.minimum(underlying.storage, underlying.layout)

  def max(using @unused ordered: OrderedDType[A]): A =
    ReductionKernels.maximum(underlying.storage, underlying.layout)

  def min(axis: Int)(using
      @unused ordered: OrderedDType[A],
      @unused canDrop: CanDropAxis[R]
  ): NDArray[A, DropAxis[R]] =
    ReductionApi.minimumAxis[A, R, DropAxis[R]](underlying, axis, keep = false)

  def max(axis: Int)(using
      @unused ordered: OrderedDType[A],
      @unused canDrop: CanDropAxis[R]
  ): NDArray[A, DropAxis[R]] =
    ReductionApi.maximumAxis[A, R, DropAxis[R]](underlying, axis, keep = false)

  def minKeep(axis: Int)(using
      @unused ordered: OrderedDType[A]
  ): NDArray[A, R] =
    ReductionApi.minimumAxis[A, R, R](underlying, axis, keep = true)

  def maxKeep(axis: Int)(using
      @unused ordered: OrderedDType[A]
  ): NDArray[A, R] =
    ReductionApi.maximumAxis[A, R, R](underlying, axis, keep = true)

  def argMin(using @unused ordered: OrderedDType[A]): Int =
    ReductionKernels.argMinimum(underlying.storage, underlying.layout)

  def argMax(using @unused ordered: OrderedDType[A]): Int =
    ReductionKernels.argMaximum(underlying.storage, underlying.layout)

  def argMin(axis: Int)(using
      @unused ordered: OrderedDType[A],
      @unused canDrop: CanDropAxis[R]
  ): NDArray[Int, DropAxis[R]] =
    ReductionApi.argMinimumAxis[A, R, DropAxis[R]](underlying, axis, keep = false)

  def argMinKeep(axis: Int)(using
      @unused ordered: OrderedDType[A]
  ): NDArray[Int, R] =
    ReductionApi.argMinimumAxis[A, R, R](underlying, axis, keep = true)

  def argMax(axis: Int)(using
      @unused ordered: OrderedDType[A],
      @unused canDrop: CanDropAxis[R]
  ): NDArray[Int, DropAxis[R]] =
    ReductionApi.argMaximumAxis[A, R, DropAxis[R]](underlying, axis, keep = false)

  def argMaxKeep(axis: Int)(using
      @unused ordered: OrderedDType[A]
  ): NDArray[Int, R] =
    ReductionApi.argMaximumAxis[A, R, R](underlying, axis, keep = true)

  def mean(using @unused floating: FloatingDType[A]): A =
    ReductionKernels.mean(underlying.storage, underlying.layout)

  def mean(axis: Int)(using
      @unused floating: FloatingDType[A],
      @unused canDrop: CanDropAxis[R]
  ): NDArray[A, DropAxis[R]] =
    ReductionApi.meanAxis[A, R, DropAxis[R]](underlying, axis, keep = false)

  def meanKeep(axis: Int)(using
      @unused floating: FloatingDType[A]
  ): NDArray[A, R] =
    ReductionApi.meanAxis[A, R, R](underlying, axis, keep = true)

  def sameElements(other: NDArray[A, ?]): Boolean =
    EqualityApi.sameElements(underlying, other)

  def sameElements(other: BorrowedNDArray[A, ?]): Boolean =
    EqualityApi.sameElements(underlying, other.underlying)

  def sameElementsBits(other: NDArray[A, ?]): Boolean =
    EqualityApi.sameElementsBits(underlying, other)

  def sameElementsBits(other: BorrowedNDArray[A, ?]): Boolean =
    EqualityApi.sameElementsBits(underlying, other.underlying)

  def allClose(
      other: NDArray[A, ?],
      relativeTolerance: Double,
      absoluteTolerance: Double
  )(using @unused floating: FloatingDType[A]): Boolean =
    EqualityApi.allClose(
      underlying,
      other,
      relativeTolerance,
      absoluteTolerance
    )

  def allClose(
      other: BorrowedNDArray[A, ?],
      relativeTolerance: Double,
      absoluteTolerance: Double
  )(using @unused floating: FloatingDType[A]): Boolean =
    EqualityApi.allClose(
      underlying,
      other.underlying,
      relativeTolerance,
      absoluteTolerance
    )

  private def borrowedOperand[
      B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]
  ](
      operation: Byte,
      other: B
  ): NDArray[A, BorrowedOperandRank[A, R, B]] =
    val result =
      other match
        case borrowed: BorrowedNDArray[?, ?] =>
          KernelApi.binary(
            operation,
            underlying,
            borrowed.underlying.asInstanceOf[NDArray[A, AnyRank]]
          )
        case owned: NDArray[?, ?] =>
          KernelApi.binary(
            operation,
            underlying,
            owned.asInstanceOf[NDArray[A, AnyRank]]
          )
        case scalar =>
          KernelApi.scalar(operation, underlying, scalar.asInstanceOf[A])
    result.asInstanceOf[NDArray[A, BorrowedOperandRank[A, R, B]]]

  override def toString: String =
    s"Borrowed${underlying.toString}"
