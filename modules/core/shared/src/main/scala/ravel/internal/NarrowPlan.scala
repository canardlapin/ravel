package ravel.internal

import ravel.*

/** Checked, normalized plan for an exact rank-preserving narrow operation. */
private[ravel] final case class NarrowPlan(
    axis: Int,
    start: Int,
    stopExclusive: Int,
    length: Int
)

private[ravel] object NarrowPlan:
  def apply(shape: IArray[Int], axis: Int, startIndex: Int, length: Int): NarrowPlan =
    from(shape, axis, startIndex, length).fold(
      error => throw InvalidNarrowException(error),
      identity
    )

  def from(
      shape: IArray[Int],
      axis: Int,
      startIndex: Int,
      length: Int
  ): Either[InvalidNarrow, NarrowPlan] =
    def invalid(reason: String): Left[InvalidNarrow, Nothing] =
      Left(InvalidNarrow(axis, startIndex, length, reason))

    if length < 0 then return invalid(s"length $length is negative")

    val normalizedAxisLong =
      if axis < 0 then axis.toLong + shape.length.toLong else axis.toLong
    if normalizedAxisLong < 0L || normalizedAxisLong >= shape.length.toLong then
      return invalid(s"axis $axis is outside rank ${shape.length}")
    val normalizedAxis = normalizedAxisLong.toInt
    val extent = shape(normalizedAxis)
    val start =
      if startIndex < 0 then startIndex.toLong + extent.toLong
      else startIndex.toLong

    if start < 0L || start > extent.toLong then
      return invalid(s"start $startIndex is outside [-$extent, $extent]")

    val stop = start + length.toLong
    if stop > extent.toLong then return invalid(s"interval [$start, $stop) exceeds extent $extent")

    Right(
      NarrowPlan(
        normalizedAxis,
        start.toInt,
        stop.toInt,
        length
      )
    )
