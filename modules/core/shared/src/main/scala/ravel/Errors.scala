package ravel

sealed abstract class NDArrayException(message: String) extends IllegalArgumentException(message)

final case class ShapeMismatch(left: String, right: String)
    extends NDArrayException(s"Shape mismatch: $left and $right")

final case class BroadcastMismatch(left: String, right: String, alignedAxis: Int)
    extends NDArrayException(
      s"Cannot broadcast shapes $left and $right: incompatible aligned axis $alignedAxis"
    )

final case class InvalidAxis(axis: Int, rank: Int)
    extends NDArrayException(s"Invalid axis $axis for rank $rank")

final case class InvalidShape(reason: String) extends NDArrayException(s"Invalid shape: $reason")

final case class InvalidSlice(reason: String) extends NDArrayException(s"Invalid slice: $reason")

final case class LayoutOverflow(reason: String)
    extends NDArrayException(s"Layout overflow: $reason")

final case class NonContiguousLayout(reason: String)
    extends NDArrayException(s"Non-contiguous layout: $reason")

final case class EmptyReduction(operation: String)
    extends NDArrayException(s"$operation is undefined for an empty reduction domain")

/** Soft rank refinement failure returned by [[NDArray.requireRank]]. */
final case class RankMismatch(expected: Int, actual: Int)
    extends NDArrayException(s"Rank mismatch: expected $expected but found $actual"):
  override def toString: String =
    s"RankMismatch(expected = $expected, actual = $actual)"

/** A scalar index is out of bounds, or its arity does not match the array rank. */
sealed abstract class InvalidIndex(reason: String)
    extends NDArrayException(s"Invalid index: $reason")

object InvalidIndex:
  final case class ArityMismatch(expected: Int, received: Int)
      extends InvalidIndex(s"expected $expected indices but received $received")

  final case class OutOfBounds(axis: Int, index: Int, dimension: Int)
      extends InvalidIndex(s"axis $axis index $index is outside [0, $dimension)")

  final case class LinearOutOfBounds(index: Int, size: Int)
      extends InvalidIndex(s"builder index $index is outside [0, $size)")

/** A consuming [[ArrayBuilder]] was used after its construction callback ended. */
final class BuilderClosed private[ravel] () extends IllegalStateException("array builder is closed")
