package ravel

import ravel.internal.*
import scala.annotation.unused

extension [A, R <: AnyRank](array: ReadableArray[A, R])
  def map[B](f: A => B)(using DType[B]): NDArray[B, R] =
    KernelApi.map(array.toNDArray, f)

  def zipMap[B, RY <: AnyRank](
      other: ReadableArray[A, RY]
  )(f: (A, A) => B)(using DType[B]): NDArray[B, BroadcastRank[R, RY]] =
    KernelApi.zipMap(array.toNDArray, other.toNDArray, exact = false, f)

  def zipMapExact[B, RY <: AnyRank](
      other: ReadableArray[A, RY]
  )(f: (A, A) => B)(using DType[B]): NDArray[B, BroadcastRank[R, RY]] =
    KernelApi.zipMap(array.toNDArray, other.toNDArray, exact = true, f)

  def elementwiseEqual[RY <: AnyRank](
      other: ReadableArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    KernelApi.compare(KernelOp.Equal, array.toNDArray, other.toNDArray)

  def elementwiseNotEqual[RY <: AnyRank](
      other: ReadableArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    KernelApi.compare(KernelOp.NotEqual, array.toNDArray, other.toNDArray)

  def ===[RY <: AnyRank](
      other: ReadableArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    elementwiseEqual(other)

  def =!=[RY <: AnyRank](
      other: ReadableArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    elementwiseNotEqual(other)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused arithmetic: ArithmeticDType[A]
)
  def unary_+ : NDArray[A, R] =
    array.toNDArray.copy

  def unary_- : NDArray[A, R] =
    KernelApi.unary(KernelOp.Negate, array.toNDArray)

  def +[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Add, array, other)

  def -[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Subtract, array, other)

  def *[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Multiply, array, other)

  def abs: NDArray[A, R] =
    KernelApi.unary(KernelOp.Absolute, array.toNDArray)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused integral: IntegralArithmeticDType[A]
)
  /** Truncating integer division. Prefer floating `/` for scientific code. */
  def quot[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Divide, array, other)

  def truncDiv[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    quot(other)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused floating: FloatingDType[A]
)
  def /[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Divide, array, other)

  def sqrt: NDArray[A, R] = KernelApi.unary(KernelOp.Sqrt, array.toNDArray)
  def exp: NDArray[A, R] = KernelApi.unary(KernelOp.Exp, array.toNDArray)
  def log: NDArray[A, R] = KernelApi.unary(KernelOp.Log, array.toNDArray)
  def sin: NDArray[A, R] = KernelApi.unary(KernelOp.Sin, array.toNDArray)
  def cos: NDArray[A, R] = KernelApi.unary(KernelOp.Cos, array.toNDArray)
  def tan: NDArray[A, R] = KernelApi.unary(KernelOp.Tan, array.toNDArray)
  def floor: NDArray[A, R] = KernelApi.unary(KernelOp.Floor, array.toNDArray)
  def ceil: NDArray[A, R] = KernelApi.unary(KernelOp.Ceil, array.toNDArray)
  def isNaN: NDArray[Boolean, R] =
    KernelApi.floatingPredicate(KernelOp.IsNaN, array.toNDArray)
  def isFinite: NDArray[Boolean, R] =
    KernelApi.floatingPredicate(KernelOp.IsFinite, array.toNDArray)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused ordered: OrderedDType[A]
)
  def minimum[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Minimum, array, other)

  def maximum[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Maximum, array, other)

  def clip(lower: A, upper: A): NDArray[A, R] =
    KernelApi.clip(array.toNDArray, lower, upper)

  def <[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompare(KernelOp.Less, array, other)

  def <=[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompare(KernelOp.LessEqual, array, other)

  def >[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompareSwapped(KernelOp.Less, array, other)

  def >=[B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompareSwapped(KernelOp.LessEqual, array, other)

  /** Compare with `Ordering.compare` semantics rather than primitive IEEE
    * semantics. This is primarily useful when an image operation promises to
    * honor Scala's default total ordering for floating values.
    */
  def orderedLessThan[
      B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]
  ](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompare(KernelOp.OrderedLess, array, other)

  def orderedLessOrEqual[
      B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]
  ](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompare(KernelOp.OrderedLessEqual, array, other)

  def orderedGreaterThan[
      B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]
  ](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompareSwapped(KernelOp.OrderedLess, array, other)

  def orderedGreaterOrEqual[
      B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]
  ](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompareSwapped(KernelOp.OrderedLessEqual, array, other)

private def readableOperand[
    A,
    R <: AnyRank,
    B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]
](
    operation: Byte,
    left: ReadableArray[A, R],
    right: B
): NDArray[A, OperandRank[A, R, B]] =
  val result =
    right match
      case borrowed: BorrowedNDArray[?, ?] =>
        KernelApi.binary(
          operation,
          left.toNDArray,
          borrowed.underlying.asInstanceOf[NDArray[A, AnyRank]]
        )
      case owned: NDArray[?, ?] =>
        KernelApi.binary(
          operation,
          left.toNDArray,
          owned.asInstanceOf[NDArray[A, AnyRank]]
        )
      case scalar =>
        KernelApi.scalar(operation, left.toNDArray, scalar.asInstanceOf[A])
  result.asInstanceOf[NDArray[A, OperandRank[A, R, B]]]

private def readableCompare[
    A,
    R <: AnyRank,
    B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]
](
    operation: Byte,
    left: ReadableArray[A, R],
    right: B
): NDArray[Boolean, OperandRank[A, R, B]] =
  val result =
    right match
      case borrowed: BorrowedNDArray[?, ?] =>
        KernelApi.compare(
          operation,
          left.toNDArray,
          borrowed.underlying.asInstanceOf[NDArray[A, AnyRank]]
        )
      case owned: NDArray[?, ?] =>
        KernelApi.compare(
          operation,
          left.toNDArray,
          owned.asInstanceOf[NDArray[A, AnyRank]]
        )
      case scalar if isOrderedComparison(operation) =>
        KernelApi.orderedCompareScalar(
          operation,
          left.toNDArray,
          scalar.asInstanceOf[A],
          scalarLeft = false
        )
      case scalar =>
        KernelApi.compare(
          operation,
          left.toNDArray,
          NDArray.scalar(scalar.asInstanceOf[A])(using left.dtype)
        )
  result.asInstanceOf[NDArray[Boolean, OperandRank[A, R, B]]]

private def readableCompareSwapped[
    A,
    R <: AnyRank,
    B <: A | NDArray[A, ? <: AnyRank] | BorrowedNDArray[A, ? <: AnyRank]
](
    operation: Byte,
    left: ReadableArray[A, R],
    right: B
): NDArray[Boolean, OperandRank[A, R, B]] =
  val result =
    right match
      case borrowed: BorrowedNDArray[?, ?] =>
        KernelApi.compare(
          operation,
          borrowed.underlying.asInstanceOf[NDArray[A, AnyRank]],
          left.toNDArray
        )
      case owned: NDArray[?, ?] =>
        KernelApi.compare(
          operation,
          owned.asInstanceOf[NDArray[A, AnyRank]],
          left.toNDArray
        )
      case scalar if isOrderedComparison(operation) =>
        KernelApi.orderedCompareScalar(
          operation,
          left.toNDArray,
          scalar.asInstanceOf[A],
          scalarLeft = true
        )
      case scalar =>
        KernelApi.compare(
          operation,
          NDArray.scalar(scalar.asInstanceOf[A])(using left.dtype),
          left.toNDArray
        )
  result.asInstanceOf[NDArray[Boolean, OperandRank[A, R, B]]]

private def isOrderedComparison(operation: Byte): Boolean =
  operation == KernelOp.OrderedLess || operation == KernelOp.OrderedLessEqual
