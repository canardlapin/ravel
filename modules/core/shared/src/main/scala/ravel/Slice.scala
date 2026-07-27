package ravel

final case class Slice private (start: Int, stopExclusive: Int, step: Int)

object Slice:
  def apply(start: Int, stopExclusive: Int, step: Int = 1): Slice =
    from(start, stopExclusive, step).fold(throw _, identity)

  def from(
      start: Int,
      stopExclusive: Int,
      step: Int = 1
  ): Either[InvalidSlice, Slice] =
    if step == 0 then Left(InvalidSlice("step must not be zero"))
    else Right(new Slice(start, stopExclusive, step))

  def from(range: Range): Either[InvalidSlice, Slice] =
    val stop =
      if range.isInclusive then
        val shifted = range.end.toLong + range.step.sign.toLong
        if shifted < Int.MinValue.toLong || shifted > Int.MaxValue.toLong then
          return Left(InvalidSlice("inclusive range endpoint cannot be represented exclusively"))
        shifted.toInt
      else range.end
    from(range.start, stop, range.step)
