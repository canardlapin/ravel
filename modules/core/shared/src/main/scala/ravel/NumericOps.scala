package ravel

import ravel.internal.*
import scala.annotation.unused

extension [A, R <: AnyRank](array: NDArray[A, R])
  def map[B](f: A => B)(using DType[B]): NDArray[B, R] =
    KernelApi.map(array, f)

  def zipMap[B, RY <: AnyRank](
      other: NDArray[A, RY]
  )(f: (A, A) => B)(using DType[B]): NDArray[B, BroadcastRank[R, RY]] =
    KernelApi.zipMap(array, other, exact = false, f)

  def zipMapExact[B, RY <: AnyRank](
      other: NDArray[A, RY]
  )(f: (A, A) => B)(using DType[B]): NDArray[B, BroadcastRank[R, RY]] =
    KernelApi.zipMap(array, other, exact = true, f)

  def elementwiseEqual[RY <: AnyRank](
      other: NDArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    KernelApi.compare(KernelOp.Equal, array, other)

  def ===[RY <: AnyRank](
      other: NDArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    elementwiseEqual(other)

  def =!=[RY <: AnyRank](
      other: NDArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    KernelApi.compare(KernelOp.NotEqual, array, other)

extension [A, R <: AnyRank](array: NDArray[A, R])(using
    @unused arithmetic: ArithmeticDType[A]
)
  def unary_+ : NDArray[A, R] =
    array.copy

  def unary_- : NDArray[A, R] =
    KernelApi.unary(KernelOp.Negate, array)

  def +[B <: A | NDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    KernelApi.operand(KernelOp.Add, array, other)

  def -[B <: A | NDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    KernelApi.operand(KernelOp.Subtract, array, other)

  def *[B <: A | NDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    KernelApi.operand(KernelOp.Multiply, array, other)

  def /[B <: A | NDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    KernelApi.operand(KernelOp.Divide, array, other)

  def abs: NDArray[A, R] =
    KernelApi.unary(KernelOp.Absolute, array)

extension [A, R <: AnyRank](array: NDArray[A, R])(using
    @unused ordered: OrderedDType[A]
)
  def minimum[B <: A | NDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    KernelApi.operand(KernelOp.Minimum, array, other)

  def maximum[B <: A | NDArray[A, ? <: AnyRank]](
      other: B
  ): NDArray[A, OperandRank[A, R, B]] =
    KernelApi.operand(KernelOp.Maximum, array, other)

  def clip(lower: A, upper: A): NDArray[A, R] =
    KernelApi.clip(array, lower, upper)

  def <[RY <: AnyRank](
      other: NDArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    KernelApi.compare(KernelOp.Less, array, other)

  def <=[RY <: AnyRank](
      other: NDArray[A, RY]
  ): NDArray[Boolean, BroadcastRank[R, RY]] =
    KernelApi.compare(KernelOp.LessEqual, array, other)

extension [A, R <: AnyRank](array: NDArray[A, R])(using
    @unused floating: FloatingDType[A]
)
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
