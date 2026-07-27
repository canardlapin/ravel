package ravel.internal

import ravel.*

private[ravel] object EqualityApi:
  def sameElements[A](
      left: NDArray[A, ?],
      right: NDArray[A, ?]
  ): Boolean =
    sameShape(left.layout.shape, right.layout.shape) && {
      val leftValues = left.elementsIterator
      val rightValues = right.elementsIterator
      var same = true
      while leftValues.hasNext && same do
        same = leftValues.next() == rightValues.next()
      same
    }

  def sameElementsBits[A](
      left: NDArray[A, ?],
      right: NDArray[A, ?]
  ): Boolean =
    sameShape(left.layout.shape, right.layout.shape) && {
      val leftValues = left.elementsIterator
      val rightValues = right.elementsIterator
      var same = true
      left.dtype.tag match
        case DType.FloatTag =>
          while leftValues.hasNext && same do
            same =
              java.lang.Float.floatToRawIntBits(
                leftValues.next().asInstanceOf[Float]
              ) ==
                java.lang.Float.floatToRawIntBits(
                  rightValues.next().asInstanceOf[Float]
                )
        case DType.DoubleTag =>
          while leftValues.hasNext && same do
            same =
              java.lang.Double.doubleToRawLongBits(
                leftValues.next().asInstanceOf[Double]
              ) ==
                java.lang.Double.doubleToRawLongBits(
                  rightValues.next().asInstanceOf[Double]
                )
        case _ =>
          while leftValues.hasNext && same do
            same = leftValues.next() == rightValues.next()
      same
    }

  def allClose[A](
      left: NDArray[A, ?],
      right: NDArray[A, ?],
      relativeTolerance: Double,
      absoluteTolerance: Double
  ): Boolean =
    if relativeTolerance.isNaN ||
        absoluteTolerance.isNaN ||
        relativeTolerance < 0.0 ||
        absoluteTolerance < 0.0
    then
      throw new IllegalArgumentException("allClose tolerances must be nonnegative and not NaN")
    sameShape(left.layout.shape, right.layout.shape) && {
      val leftValues = left.elementsIterator
      val rightValues = right.elementsIterator
      var close = true
      while leftValues.hasNext && close do
        val x =
          left.dtype.tag match
            case DType.FloatTag  => leftValues.next().asInstanceOf[Float].toDouble
            case DType.DoubleTag => leftValues.next().asInstanceOf[Double]
            case _ => throw new UnsupportedOperationException("allClose requires Float or Double")
        val y =
          right.dtype.tag match
            case DType.FloatTag  => rightValues.next().asInstanceOf[Float].toDouble
            case DType.DoubleTag => rightValues.next().asInstanceOf[Double]
            case _ => throw new UnsupportedOperationException("allClose requires Float or Double")
        close =
          x == y ||
            (!x.isNaN &&
              !y.isNaN &&
              math.abs(x - y) <= absoluteTolerance + relativeTolerance * math.abs(y))
      close
    }

  private def sameShape(left: IArray[Int], right: IArray[Int]): Boolean =
    if left.length != right.length then false
    else
      var same = true
      var i = 0
      while i < left.length && same do
        same = left(i) == right(i)
        i += 1
      same
