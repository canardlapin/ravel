package ravel

import scala.util.control.NonFatal

final class Shape[+R <: AnyRank] private[ravel] (
    private[ravel] val unsafeDimensions: IArray[Int],
    val size: Int
):
  def rank: Int = unsafeDimensions.length

  def apply(axis: Int): Int =
    unsafeDimensions(Shape.normalizeAxis(axis, rank))

  def toIArray: IArray[Int] =
    IArray.unsafeFromArray(IArray.genericWrapArray(unsafeDimensions).toArray)

  override def equals(other: Any): Boolean =
    other match
      case that: Shape[?] =>
        Shape.dimensionsEqual(unsafeDimensions, that.unsafeDimensions)
      case _ => false

  override def hashCode(): Int =
    var hash = 1
    var axis = 0
    while axis < unsafeDimensions.length do
      hash = 31 * hash + unsafeDimensions(axis)
      axis += 1
    hash

  override def toString: String =
    unsafeDimensions.mkString("(", ", ", ")")

object Shape:
  val scalar: Shape[Rank[0]] =
    checked[Rank[0]](IArray.empty)

  def apply(d0: Int): Shape[Rank[1]] =
    checked[Rank[1]](IArray(d0))

  def apply(d0: Int, d1: Int): Shape[Rank[2]] =
    checked[Rank[2]](IArray(d0, d1))

  def apply(d0: Int, d1: Int, d2: Int): Shape[Rank[3]] =
    checked[Rank[3]](IArray(d0, d1, d2))

  def apply(d0: Int, d1: Int, d2: Int, d3: Int): Shape[Rank[4]] =
    checked[Rank[4]](IArray(d0, d1, d2, d3))

  def from(dimensions: IArray[Int]): Either[InvalidShape, Shape[AnyRank]] =
    try
      Right(
        checked[AnyRank](
          IArray.unsafeFromArray(IArray.genericWrapArray(dimensions).toArray)
        )
      )
    catch
      case error: InvalidShape => Left(error)
      case NonFatal(error) => Left(InvalidShape(error.getMessage))

  def from(dimensions: Seq[Int]): Either[InvalidShape, Shape[AnyRank]] =
    from(IArray.unsafeFromArray(dimensions.toArray))

  private[ravel] def unsafeRanked[R <: AnyRank](
      dimensions: IArray[Int]
  ): Shape[R] =
    checked[R](
      IArray.unsafeFromArray(IArray.genericWrapArray(dimensions).toArray)
    )

  /** Trust already-validated dimensions; does not copy or recheck. */
  private[ravel] def trusted[R <: AnyRank](
      dimensions: IArray[Int],
      size: Int
  ): Shape[R] =
    new Shape(dimensions, size)

  /** Validate owned dimensions without copying. */
  private[ravel] def validated[R <: AnyRank](
      dimensions: IArray[Int]
  ): Shape[R] =
    checked[R](dimensions)

  /** Retag a validated shape without copying dimensions. */
  private[ravel] def retag[R <: AnyRank](shape: Shape[?]): Shape[R] =
    shape.asInstanceOf[Shape[R]]

  private def checked[R <: AnyRank](dimensions: IArray[Int]): Shape[R] =
    var product = 1L
    var axis = 0
    while axis < dimensions.length do
      val dimension = dimensions(axis)
      if dimension < 0 then throw InvalidShape(s"axis $axis has negative dimension $dimension")
      product =
        try Math.multiplyExact(product, dimension.toLong)
        catch
          case _: ArithmeticException =>
            throw InvalidShape(s"dimension product exceeds Long at axis $axis")
      if product > Int.MaxValue.toLong then
        throw InvalidShape(s"element count $product exceeds portable Int buffer limit")
      axis += 1
    new Shape(dimensions, product.toInt)

  private[ravel] def normalizeAxis(axis: Int, rank: Int): Int =
    val normalized = if axis < 0 then axis + rank else axis
    if normalized < 0 || normalized >= rank then throw InvalidAxis(axis, rank)
    normalized

  private[ravel] def dimensionsEqual(left: IArray[Int], right: IArray[Int]): Boolean =
    if left.length != right.length then false
    else
      var axis = 0
      var same = true
      while axis < left.length && same do
        same = left(axis) == right(axis)
        axis += 1
      same
