package ravel.internal

import ravel.*

/** Extensional equality without materializing boxed snapshots.
  *
  * Walks both layouts in logical order and fails fast on the first disagreement.
  */
private[ravel] object EqualityApi:
  def sameElements[A](
      left: NDArray[A, ?],
      right: NDArray[A, ?]
  ): Boolean =
    sameShape(left.layout.shape, right.layout.shape) &&
      zipPhysical(left.layout, right.layout) { (li, ri) =>
        ProbeApi.get(left.storage, li) == ProbeApi.get(right.storage, ri)
      }

  def sameElementsBits[A](
      left: NDArray[A, ?],
      right: NDArray[A, ?]
  ): Boolean =
    sameShape(left.layout.shape, right.layout.shape) && {
      left.dtype.tag match
        case DType.FloatTag =>
          zipPhysical(left.layout, right.layout) { (li, ri) =>
            java.lang.Float.floatToRawIntBits(
              ProbeApi.get(left.storage, li).asInstanceOf[Float]
            ) ==
              java.lang.Float.floatToRawIntBits(
                ProbeApi.get(right.storage, ri).asInstanceOf[Float]
              )
          }
        case DType.DoubleTag =>
          zipPhysical(left.layout, right.layout) { (li, ri) =>
            java.lang.Double.doubleToRawLongBits(
              ProbeApi.get(left.storage, li).asInstanceOf[Double]
            ) ==
              java.lang.Double.doubleToRawLongBits(
                ProbeApi.get(right.storage, ri).asInstanceOf[Double]
              )
          }
        case _ =>
          sameElements(left, right)
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
    then throw new IllegalArgumentException("allClose tolerances must be nonnegative and not NaN")
    sameShape(left.layout.shape, right.layout.shape) && {
      left.dtype.tag match
        case DType.FloatTag | DType.DoubleTag =>
          zipPhysical(left.layout, right.layout) { (li, ri) =>
            val x = toDouble(left, li)
            val y = toDouble(right, ri)
            x == y ||
            (!x.isNaN &&
              !y.isNaN &&
              math.abs(x - y) <= absoluteTolerance + relativeTolerance * math.abs(y))
          }
        case _ =>
          throw new UnsupportedOperationException("allClose requires Float or Double")
    }

  private def toDouble[A](array: NDArray[A, ?], index: Int): Double =
    array.dtype.tag match
      case DType.FloatTag => ProbeApi.get(array.storage, index).asInstanceOf[Float].toDouble
      case DType.DoubleTag => ProbeApi.get(array.storage, index).asInstanceOf[Double]
      case _ => throw new UnsupportedOperationException("allClose requires Float or Double")

  private def zipPhysical(
      left: Layout,
      right: Layout
  )(agree: (Int, Int) => Boolean): Boolean =
    if left.size == 0 then true
    else if left.rank == 0 then agree(left.offset, right.offset)
    else
      val counters = new Array[Int](left.rank)
      var leftAddress = left.offset.toLong
      var rightAddress = right.offset.toLong
      var visited = 0
      var same = true
      while visited < left.size && same do
        same = agree(
          Layout.checkedInt(leftAddress, "equality left"),
          Layout.checkedInt(rightAddress, "equality right")
        )
        visited += 1
        if visited < left.size && same then
          var axis = left.rank - 1
          var advanced = false
          while axis >= 0 && !advanced do
            counters(axis) += 1
            leftAddress = Layout.checkedAdd(leftAddress, left.strides(axis).toLong, "equality left")
            rightAddress =
              Layout.checkedAdd(rightAddress, right.strides(axis).toLong, "equality right")
            if counters(axis) < left.shape(axis) then advanced = true
            else
              leftAddress = Layout.checkedAdd(
                leftAddress,
                -Layout.checkedMultiply(
                  counters(axis).toLong,
                  left.strides(axis).toLong,
                  "equality left rewind"
                ),
                "equality left rewind"
              )
              rightAddress = Layout.checkedAdd(
                rightAddress,
                -Layout.checkedMultiply(
                  counters(axis).toLong,
                  right.strides(axis).toLong,
                  "equality right rewind"
                ),
                "equality right rewind"
              )
              counters(axis) = 0
              axis -= 1
      same

  private def sameShape(left: IArray[Int], right: IArray[Int]): Boolean =
    Layout.sameShape(left, right)
