package ravel.internal

import ravel.*

/** Checked, normalized source axes for a rank-preserving permutation. */
private[ravel] final class PermutationPlan private (
    val normalizedAxes: IArray[Int]
)

private[ravel] object PermutationPlan:
  def from(rank: Int, order: Seq[Int]): Either[PermutationError, PermutationPlan] =
    if order.size != rank then return Left(InvalidPermutation(rank, order.size))

    val normalized = new Array[Int](rank)
    val seen = new Array[Boolean](rank)
    var position = 0
    while position < rank do
      val supplied = order(position)
      val resolvedLong =
        if supplied < 0 then supplied.toLong + rank.toLong else supplied.toLong
      if resolvedLong < 0L || resolvedLong >= rank.toLong then
        return Left(InvalidPermutationAxis(supplied, rank))
      val resolved = resolvedLong.toInt
      if seen(resolved) then return Left(DuplicateAxis(supplied, resolved))
      normalized(position) = resolved
      seen(resolved) = true
      position += 1

    Right(new PermutationPlan(IArray.unsafeFromArray(normalized)))
