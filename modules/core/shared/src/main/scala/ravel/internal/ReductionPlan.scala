package ravel.internal

import ravel.*

private[ravel] final class ReductionPlan(
    val source: Layout,
    val axis: Int,
    val keepDimension: Boolean
):
  val reducedLength: Int = source.shape(axis)
  val reducedStride: Int = source.strides(axis)
  val outerShape: Array[Int] =
    val result = new Array[Int](source.rank - 1)
    var read = 0
    var write = 0
    while read < source.rank do
      if read != axis then
        result(write) = source.shape(read)
        write += 1
      read += 1
    result
  val outerStrides: Array[Int] =
    val result = new Array[Int](source.rank - 1)
    var read = 0
    var write = 0
    while read < source.rank do
      if read != axis then
        result(write) = source.strides(read)
        write += 1
      read += 1
    result
  val outputShape: IArray[Int] =
    if keepDimension then
      val result = IArray.genericWrapArray(source.shape).toArray
      result(axis) = 1
      IArray.unsafeFromArray(result)
    else IArray.unsafeFromArray(outerShape.clone())
  val outputSize: Int =
    var result = 1L
    var i = 0
    while i < outerShape.length do
      result = Layout.checkedMultiply(
        result,
        outerShape(i).toLong,
        s"reduction output size at axis $i"
      )
      i += 1
    Layout.checkedInt(result, "reduction output size")

private[ravel] object ReductionPlan:
  def apply(layout: Layout, axis: Int, keepDimension: Boolean): ReductionPlan =
    new ReductionPlan(layout, layout.normalizedAxis(axis), keepDimension)
