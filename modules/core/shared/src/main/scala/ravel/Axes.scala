package ravel

/** Validated axes for one multi-axis reduction.
  *
  * Negative axes are normalized once. `normalized` is stored in ascending source-axis order so
  * equivalent axis sets have one value representation and one deterministic numerical schedule.
  */
final class Axes private (
    val rank: Int,
    val normalized: IArray[Int]
):
  def isEmpty: Boolean = normalized.isEmpty
  def size: Int = normalized.length

  override def equals(other: Any): Boolean =
    other match
      case that: Axes =>
        rank == that.rank &&
        IArray
          .genericWrapArray(normalized)
          .sameElements(
            IArray.genericWrapArray(that.normalized)
          )
      case _ => false

  override def hashCode(): Int =
    var result = rank
    var index = 0
    while index < normalized.length do
      result = 31 * result + normalized(index)
      index += 1
    result

  override def toString: String = normalized.mkString("Axes(", ", ", ")")

object Axes:
  def from(rank: Int, axes: Int*): Either[AxesError, Axes] =
    if rank < 0 then return Left(InvalidAxesRank(rank))
    val normalized = new Array[Int](axes.length)
    val seen = new Array[Boolean](rank)
    var index = 0
    while index < axes.length do
      val supplied = axes(index)
      val resolvedLong =
        if supplied < 0 then supplied.toLong + rank.toLong else supplied.toLong
      if resolvedLong < 0L || resolvedLong >= rank.toLong then
        return Left(InvalidReductionAxis(supplied, rank))
      val resolved = resolvedLong.toInt
      if seen(resolved) then return Left(DuplicateReductionAxis(supplied, resolved))
      normalized(index) = resolved
      seen(resolved) = true
      index += 1
    Right(new Axes(rank, IArray.unsafeFromArray(normalized.sorted)))

  def all(rank: Int): Either[AxesError, Axes] =
    from(rank, 0.until(rank)*)

  def require(rank: Int, axes: Int*): Axes =
    from(rank, axes*).fold(error => throw InvalidAxesException(error), identity)
