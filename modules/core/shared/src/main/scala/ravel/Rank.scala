package ravel

import scala.compiletime.ops.int.-
import scala.compiletime.ops.int.>
import scala.compiletime.ops.int.+
import scala.compiletime.ops.int.Max

/** Dynamic rank marker and supertype of all statically refined ranks. */
sealed trait AnyRank

/** A rank known statically. Dimension sizes remain runtime values. */
sealed trait Rank[N <: Int] extends AnyRank

type Array0[A] = NDArray[A, Rank[0]]
type Array1[A] = NDArray[A, Rank[1]]
type Array2[A] = NDArray[A, Rank[2]]
type Array3[A] = NDArray[A, Rank[3]]
type Array4[A] = NDArray[A, Rank[4]]
type AnyNDArray[A] = NDArray[A, AnyRank]

type DropAxis[R <: AnyRank] <: AnyRank = R match
  case Rank[0] => Nothing
  case Rank[n] => Rank[n - 1]
  case _ => AnyRank

type AddAxis[R <: AnyRank] <: AnyRank = R match
  case Rank[n] => Rank[n + 1]
  case _ => AnyRank

type BroadcastRank[X <: AnyRank, Y <: AnyRank] <: AnyRank = (X, Y) match
  case (Rank[x], Rank[y]) => Rank[Max[x, y]]
  case _ => AnyRank

/** Result rank for scalar-or-array operands accepted by readable arithmetic. */
type OperandRank[A, R <: AnyRank, B] <: AnyRank = B match
  case BorrowedNDArray[A, rank] => BroadcastRank[R, rank]
  case NDArray[A, rank] => BroadcastRank[R, rank]
  case _ => R

/** Evidence that a rank-lowering operation cannot produce a negative rank. */
sealed trait CanDropAxis[R <: AnyRank]

object CanDropAxis:
  given dynamic: CanDropAxis[AnyRank] with {}

  given knownPositive[N <: Int](using (N > 0) =:= true): CanDropAxis[Rank[N]] with {}
