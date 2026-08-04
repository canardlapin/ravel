package ravel

import ravel.internal.*
import scala.annotation.unused

extension [A, R <: AnyRank](array: ReadableArray[A, R])
  def map[B](f: A => B)(using DType[B]): NDArray[B, R] =
    KernelApi.map(array, f)

  def zipMap[B, RY <: AnyRank](
      other: ReadableArray[A, RY]
  )(f: (A, A) => B)(using DType[B]): NDArray[B, BroadcastRank[R, RY]] =
    KernelApi.zipMap(array, other, exact = false, f)

  def zipMapExact[B, RY <: AnyRank](
      other: ReadableArray[A, RY]
  )(f: (A, A) => B)(using DType[B]): NDArray[B, BroadcastRank[R, RY]] =
    KernelApi.zipMap(array, other, exact = true, f)

  def elementwiseEqual[RY <: AnyRank](
      other: ReadableArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    KernelApi.compare(KernelOp.Equal, array, other)

  def elementwiseNotEqual[RY <: AnyRank](
      other: ReadableArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    KernelApi.compare(KernelOp.NotEqual, array, other)

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
    KernelApi.copy(array)

  def unary_- : NDArray[A, R] =
    KernelApi.unary(KernelOp.Negate, array)

  def +[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Add, array, other)

  def -[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Subtract, array, other)

  def *[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Multiply, array, other)

  def abs: NDArray[A, R] =
    KernelApi.unary(KernelOp.Absolute, array)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused integral: IntegralArithmeticDType[A]
)
  /** Truncating integer division. Prefer floating `/` for scientific code. */
  def quot[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Divide, array, other)

  def truncDiv[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    quot(other)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused floating: FloatingDType[A]
)
  def /[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Divide, array, other)

  def sqrt: NDArray[A, R] = KernelApi.unary(KernelOp.Sqrt, array)
  def exp: NDArray[A, R] = KernelApi.unary(KernelOp.Exp, array)
  def log: NDArray[A, R] = KernelApi.unary(KernelOp.Log, array)
  def sin: NDArray[A, R] = KernelApi.unary(KernelOp.Sin, array)
  def cos: NDArray[A, R] = KernelApi.unary(KernelOp.Cos, array)
  def tan: NDArray[A, R] = KernelApi.unary(KernelOp.Tan, array)
  def floor: NDArray[A, R] = KernelApi.unary(KernelOp.Floor, array)
  def ceil: NDArray[A, R] = KernelApi.unary(KernelOp.Ceil, array)
  def isNaN: NDArray[Boolean, R] =
    KernelApi.floatingPredicate(KernelOp.IsNaN, array)
  def isFinite: NDArray[Boolean, R] =
    KernelApi.floatingPredicate(KernelOp.IsFinite, array)

extension [A, R <: AnyRank](array: ReadableArray[A, R])(using
    @unused ordered: OrderedDType[A]
)
  def minimum[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Minimum, array, other)

  def maximum[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    readableOperand(KernelOp.Maximum, array, other)

  def clip(lower: A, upper: A): NDArray[A, R] =
    KernelApi.clip(array, lower, upper)

  def <[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompare(KernelOp.Less, array, other)

  def <=[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompare(KernelOp.LessEqual, array, other)

  def >[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompareSwapped(KernelOp.Less, array, other)

  def >=[B <: A | ReadableArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompareSwapped(KernelOp.LessEqual, array, other)

  /** Compare with `Ordering.compare` semantics rather than primitive IEEE semantics. This is
    * primarily useful when an image operation promises to honor Scala's default total ordering for
    * floating values.
    */
  def orderedLessThan[
      B <: A | ReadableArray[A, ? <: AnyRank]
  ](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompare(KernelOp.OrderedLess, array, other)

  def orderedLessOrEqual[
      B <: A | ReadableArray[A, ? <: AnyRank]
  ](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompare(KernelOp.OrderedLessEqual, array, other)

  def orderedGreaterThan[
      B <: A | ReadableArray[A, ? <: AnyRank]
  ](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompareSwapped(KernelOp.OrderedLess, array, other)

  def orderedGreaterOrEqual[
      B <: A | ReadableArray[A, ? <: AnyRank]
  ](
      other: B
  ): NDArray[Boolean, OperandRank[A, R, B]] =
    readableCompareSwapped(KernelOp.OrderedLessEqual, array, other)

private def readableOperand[
    A,
    R <: AnyRank,
    B <: A | ReadableArray[A, ? <: AnyRank]
](
    operation: Byte,
    left: ReadableArray[A, R],
    right: B
): NDArray[A, OperandRank[A, R, B]] =
  val result =
    right match
      case readable: ReadableArray[?, ?] =>
        KernelApi.binary(
          operation,
          left,
          readable.asInstanceOf[ArraySource[A, AnyRank]]
        )
      case scalar =>
        KernelApi.scalar(operation, left, scalar.asInstanceOf[A])
  result.asInstanceOf[NDArray[A, OperandRank[A, R, B]]]

private def readableCompare[
    A,
    R <: AnyRank,
    B <: A | ReadableArray[A, ? <: AnyRank]
](
    operation: Byte,
    left: ReadableArray[A, R],
    right: B
): NDArray[Boolean, OperandRank[A, R, B]] =
  val result =
    right match
      case readable: ReadableArray[?, ?] =>
        KernelApi.compare(
          operation,
          left,
          readable.asInstanceOf[ArraySource[A, AnyRank]]
        )
      case scalar if isOrderedComparison(operation) =>
        KernelApi.orderedCompareScalar(
          operation,
          left,
          scalar.asInstanceOf[A],
          scalarLeft = false
        )
      case scalar =>
        KernelApi.compare(
          operation,
          left,
          NDArray.scalar(scalar.asInstanceOf[A])(using left.dtype)
        )
  result.asInstanceOf[NDArray[Boolean, OperandRank[A, R, B]]]

private def readableCompareSwapped[
    A,
    R <: AnyRank,
    B <: A | ReadableArray[A, ? <: AnyRank]
](
    operation: Byte,
    left: ReadableArray[A, R],
    right: B
): NDArray[Boolean, OperandRank[A, R, B]] =
  val result =
    right match
      case readable: ReadableArray[?, ?] =>
        KernelApi.compare(
          operation,
          readable.asInstanceOf[ArraySource[A, AnyRank]],
          left
        )
      case scalar if isOrderedComparison(operation) =>
        KernelApi.orderedCompareScalar(
          operation,
          left,
          scalar.asInstanceOf[A],
          scalarLeft = true
        )
      case scalar =>
        KernelApi.compare(
          operation,
          NDArray.scalar(scalar.asInstanceOf[A])(using left.dtype),
          left
        )
  result.asInstanceOf[NDArray[Boolean, OperandRank[A, R, B]]]

private def isOrderedComparison(operation: Byte): Boolean =
  operation == KernelOp.OrderedLess || operation == KernelOp.OrderedLessEqual
