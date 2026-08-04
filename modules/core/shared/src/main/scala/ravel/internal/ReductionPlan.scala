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

/** One-pass plan for reducing any validated set of axes. */
private[ravel] final class MultiReductionPlan private (
    val source: Layout,
    val axes: Axes,
    val keepDimensions: Boolean
):
  private val reducedMask: Array[Boolean] =
    val result = new Array[Boolean](source.rank)
    var index = 0
    while index < axes.normalized.length do
      result(axes.normalized(index)) = true
      index += 1
    result

  val reducedShape: Array[Int] = new Array[Int](axes.normalized.length)
  val reducedStrides: Array[Int] = new Array[Int](axes.normalized.length)
  val outerShape: Array[Int] = new Array[Int](source.rank - axes.size)
  val outerStrides: Array[Int] = new Array[Int](source.rank - axes.size)

  private var sourceAxis = 0
  private var reducedPosition = 0
  private var outerPosition = 0
  while sourceAxis < source.rank do
    if reducedMask(sourceAxis) then
      reducedShape(reducedPosition) = source.shape(sourceAxis)
      reducedStrides(reducedPosition) = source.strides(sourceAxis)
      reducedPosition += 1
    else
      outerShape(outerPosition) = source.shape(sourceAxis)
      outerStrides(outerPosition) = source.strides(sourceAxis)
      outerPosition += 1
    sourceAxis += 1

  val reducedLength: Int = checkedProduct(reducedShape, "multi-axis reduction domain")
  val outputSize: Int = checkedProduct(outerShape, "multi-axis reduction output")

  val outputShape: IArray[Int] =
    if keepDimensions then
      val result = IArray.genericWrapArray(source.shape).toArray
      var index = 0
      while index < axes.normalized.length do
        result(axes.normalized(index)) = 1
        index += 1
      IArray.unsafeFromArray(result)
    else IArray.unsafeFromArray(outerShape.clone())

  private def checkedProduct(dimensions: Array[Int], context: String): Int =
    var result = 1L
    var index = 0
    while index < dimensions.length do
      result = Layout.checkedMultiply(result, dimensions(index).toLong, s"$context axis $index")
      index += 1
    Layout.checkedInt(result, context)

private[ravel] object MultiReductionPlan:
  def apply(layout: Layout, axes: Axes, keepDimensions: Boolean): MultiReductionPlan =
    if axes.rank != layout.rank then
      throw InvalidAxesException(AxesRankMismatch(layout.rank, axes.rank))
    new MultiReductionPlan(layout, axes, keepDimensions)
