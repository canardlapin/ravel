package ravel

/** Pure validation failure returned by checked APIs. */
sealed trait RavelError derives CanEqual:
  def message: String

/** A dynamically supplied shape is invalid. */
final case class ShapeError(reason: String) extends RavelError:
  def message: String = s"Invalid shape: $reason"

/** A slice could not be constructed. */
final case class SliceError(reason: String) extends RavelError:
  def message: String = s"Invalid slice: $reason"

/** A canonical-access refinement failed. */
final case class CanonicalLayoutError(reason: String) extends RavelError:
  def message: String = s"Non-contiguous layout: $reason"

/** Soft rank refinement failure returned by `requireRank` and `transpose2D`. */
final case class RankMismatch(expected: Int, actual: Int) extends RavelError:
  def message: String = s"Rank mismatch: expected $expected but found $actual"

/** Checked axis-permutation failure. */
sealed trait PermutationError extends RavelError

/** A permutation supplies the wrong number of axes. */
final case class InvalidPermutation(expectedRank: Int, receivedAxes: Int) extends PermutationError:
  def message: String =
    s"Invalid permutation: rank $expectedRank requires $expectedRank axes but received $receivedAxes"

/** A permutation axis is outside the input rank. */
final case class InvalidPermutationAxis(axis: Int, rank: Int) extends PermutationError:
  def message: String = s"Invalid permutation axis $axis for rank $rank"

/** A permutation names the same normalized source axis more than once. */
final case class DuplicateAxis(axis: Int, normalizedAxis: Int) extends PermutationError:
  def message: String =
    s"Duplicate permutation axis $axis resolves to source axis $normalizedAxis"

/** Checked exact-narrow failure. */
final case class InvalidNarrow(axis: Int, from: Int, length: Int, reason: String)
    extends RavelError:
  def message: String =
    s"Invalid narrow(axis = $axis, from = $from, length = $length): $reason"

/** Checked multi-axis reduction failure. */
sealed trait AxesError extends RavelError

/** The rank supplied to [[Axes.from]] is negative. */
final case class InvalidAxesRank(rank: Int) extends AxesError:
  def message: String = s"Invalid reduction rank $rank"

/** A reduction axis is outside the input rank. */
final case class InvalidReductionAxis(axis: Int, rank: Int) extends AxesError:
  def message: String = s"Invalid reduction axis $axis for rank $rank"

/** A reduction names the same normalized source axis more than once. */
final case class DuplicateReductionAxis(axis: Int, normalizedAxis: Int) extends AxesError:
  def message: String =
    s"Duplicate reduction axis $axis resolves to source axis $normalizedAxis"

/** An [[Axes]] value was applied to an array of a different rank. */
final case class AxesRankMismatch(expectedRank: Int, axesRank: Int) extends AxesError:
  def message: String =
    s"Reduction axes were validated for rank $axesRank but the array has rank $expectedRank"

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

/** Throwing convenience failure for an invalid axis permutation. */
final case class InvalidPermutationException(error: PermutationError)
    extends NDArrayException(error.message)

/** Throwing convenience failure for an invalid exact narrow. */
final case class InvalidNarrowException(error: InvalidNarrow)
    extends NDArrayException(error.message)

/** Throwing convenience failure for invalid multi-axis reduction axes. */
final case class InvalidAxesException(error: AxesError) extends NDArrayException(error.message)

/** A scalar index is out of bounds, or its arity does not match the array rank. */
sealed abstract class InvalidIndex(reason: String)
    extends NDArrayException(s"Invalid index: $reason")

object InvalidIndex:
  final case class ArityMismatch(expected: Int, received: Int)
      extends InvalidIndex(s"expected $expected indices but received $received")

  final case class OutOfBounds(axis: Int, index: Int, dimension: Int)
      extends InvalidIndex(s"axis $axis index $index is outside [0, $dimension)")

  final case class LinearOutOfBounds(index: Int, size: Int)
      extends InvalidIndex(s"linear index $index is outside [0, $size)")

/** A consuming [[ArrayBuilder]] was used after its construction callback ended. */
final class BuilderClosed private[ravel] () extends IllegalStateException("array builder is closed")
