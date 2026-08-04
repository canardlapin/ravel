package ravel.packed

/** Checked logical shape and canonical sample-stride plan shared by packed representations. */
private[packed] final case class PackedLayoutPlan private (
    shape: Vector[Int],
    size: Int,
    rowMajorStrides: Vector[Int]
)

private[packed] object PackedLayoutPlan:
  /** Validate a positive-rank, positive-extent shape and plan it without allocating storage. */
  def from(shape: Vector[Int]): Either[PackedError, PackedLayoutPlan] =
    if shape.isEmpty then Left(PackedError.InvalidShape("shape must have at least one axis"))
    else
      shape.find(_ <= 0) match
        case Some(extent) =>
          Left(PackedError.InvalidShape(s"axis extents must be positive, got $extent"))
        case None =>
          val strides = new Array[Int](shape.length)
          var suffix = 1L
          var axis = shape.length - 1
          var error: Option[PackedError] = None
          while axis >= 0 && error.isEmpty do
            strides(axis) = suffix.toInt
            suffix *= shape(axis).toLong
            if suffix > Int.MaxValue.toLong then
              error = Some(
                PackedError.InvalidShape(
                  s"shape ${shape.mkString("(", ",", ")")} exceeds the portable Int storage limit"
                )
              )
            axis -= 1
          error match
            case Some(value) => Left(value)
            case None =>
              Right(PackedLayoutPlan(shape, suffix.toInt, strides.toVector))

  /** Ceiling division by codes per word, avoiding an overflowing rounded-up numerator. */
  def wordCount(samples: Int, bits: PackedBits): Either[PackedError, Int] =
    if samples < 0 then
      Left(PackedError.InvalidShape(s"sample count must be nonnegative, got $samples"))
    else if samples == 0 then Right(0)
    else Right(1 + (samples - 1) / bits.codesPerWord)

  /** Validate a narrow endpoint with widened arithmetic. */
  def narrowEnd(
      axis: Int,
      start: Int,
      length: Int,
      extent: Int
  ): Either[PackedError, Int] =
    val end = start.toLong + length.toLong
    if start < 0 || length <= 0 || end > extent.toLong then
      Left(PackedError.InvalidRange(axis, start, length, extent))
    else Right(end.toInt)

  /** Transform one logical index into a physical sample offset with widened arithmetic. */
  def sampleOffset(base: Int, index: Int, stride: Int): Either[PackedError, Int] =
    val transformed = base.toLong + index.toLong * stride.toLong
    if transformed < 0L || transformed > Int.MaxValue.toLong then
      Left(PackedError.AddressOverflow(base, index, stride))
    else Right(transformed.toInt)

  /** Internal form for already-validated array/index invariants. */
  def requireSampleOffset(base: Int, index: Int, stride: Int): Int =
    sampleOffset(base, index, stride) match
      case Right(value) => value
      case Left(error) => throw new IllegalStateException(error.message)
