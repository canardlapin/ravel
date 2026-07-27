package ravel

sealed abstract class NDArrayException(message: String)
    extends IllegalArgumentException(message)

final case class ShapeMismatch(left: String, right: String)
    extends NDArrayException(s"Shape mismatch: $left and $right")

final case class BroadcastMismatch(left: String, right: String, alignedAxis: Int)
    extends NDArrayException(
      s"Cannot broadcast shapes $left and $right: incompatible aligned axis $alignedAxis"
    )

final case class InvalidAxis(axis: Int, rank: Int)
    extends NDArrayException(s"Invalid axis $axis for rank $rank")

final case class InvalidShape(reason: String)
    extends NDArrayException(s"Invalid shape: $reason")

final case class InvalidSlice(reason: String)
    extends NDArrayException(s"Invalid slice: $reason")

final case class LayoutOverflow(reason: String)
    extends NDArrayException(s"Layout overflow: $reason")

final case class NonContiguousLayout(reason: String)
    extends NDArrayException(s"Non-contiguous layout: $reason")

final case class EmptyReduction(operation: String)
    extends NDArrayException(s"$operation is undefined for an empty reduction domain")

private[ravel] final case class InvalidIndex(reason: String)
    extends NDArrayException(s"Invalid index: $reason")
