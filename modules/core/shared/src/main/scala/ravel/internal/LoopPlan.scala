package ravel.internal

import ravel.*
import scala.collection.mutable.ArrayBuffer

private[ravel] enum LoopKind:
  case LinearContiguous
  case ScalarBroadcast
  case InnerStrided
  case GeneralStrided

private[ravel] final class LoopPlan private (
    val shape: Array[Int],
    val leftStrides: Array[Int],
    val rightStrides: Array[Int],
    val leftOffset: Int,
    val rightOffset: Int,
    val size: Int,
    val resultShape: IArray[Int],
    val kind: LoopKind
):
  def rank: Int = shape.length

  /** Logical row-major callback traversal for generic callback APIs only. */
  def foreachOffset(f: (Int, Int, Int) => Unit): Unit =
    if size == 0 then return
    if rank == 0 then
      f(leftOffset, rightOffset, 0)
      return
    val counters = new Array[Int](rank)
    var left = leftOffset.toLong
    var right = rightOffset.toLong
    var output = 0
    while output < size do
      f(
        Layout.checkedInt(left, "callback left offset"),
        Layout.checkedInt(right, "callback right offset"),
        output
      )
      output += 1
      if output < size then
        var axis = rank - 1
        var advanced = false
        while axis >= 0 && !advanced do
          if counters(axis) + 1 < shape(axis) then
            counters(axis) += 1
            left = Layout.checkedAdd(left, leftStrides(axis).toLong, s"callback left axis $axis")
            right =
              Layout.checkedAdd(right, rightStrides(axis).toLong, s"callback right axis $axis")
            advanced = true
          else
            left = Layout.checkedAdd(
              left,
              -Layout.checkedMultiply(
                counters(axis).toLong,
                leftStrides(axis).toLong,
                s"callback left rewind $axis"
              ),
              s"callback left rewind $axis"
            )
            right = Layout.checkedAdd(
              right,
              -Layout.checkedMultiply(
                counters(axis).toLong,
                rightStrides(axis).toLong,
                s"callback right rewind $axis"
              ),
              s"callback right rewind $axis"
            )
            counters(axis) = 0
            axis -= 1

private[ravel] object LoopPlan:
  def unary(layout: Layout): LoopPlan =
    if layout.isCContiguous then linearUnary(layout)
    else
      build(
        layout.shape,
        layout.strides,
        layout.offset,
        layout.shape,
        layout.strides,
        layout.offset,
        exact = true
      )

  def exact(left: Layout, right: Layout): LoopPlan =
    if left.isCContiguous &&
      right.isCContiguous &&
      Layout.sameShape(left.shape, right.shape)
    then linearSameShape(left, right)
    else
      build(
        left.shape,
        left.strides,
        left.offset,
        right.shape,
        right.strides,
        right.offset,
        exact = true
      )

  def broadcast(left: Layout, right: Layout): LoopPlan =
    if left.isCContiguous &&
      right.isCContiguous &&
      Layout.sameShape(left.shape, right.shape)
    then linearSameShape(left, right)
    else
      build(
        left.shape,
        left.strides,
        left.offset,
        right.shape,
        right.strides,
        right.offset,
        exact = false
      )

  /** Same-shape C-contiguous operands: no axis alignment or coalescing. */
  def linearSameShape(left: Layout, right: Layout): LoopPlan =
    if !Layout.sameShape(left.shape, right.shape) then
      throw ShapeMismatch(
        left.shape.mkString("(", ", ", ")"),
        right.shape.mkString("(", ", ", ")")
      )
    val size = left.size
    if size == 0 then
      new LoopPlan(
        Array.emptyIntArray,
        Array.emptyIntArray,
        Array.emptyIntArray,
        left.offset,
        right.offset,
        0,
        left.shape,
        LoopKind.LinearContiguous
      )
    else
      new LoopPlan(
        Array(size),
        Array(1),
        Array(1),
        left.offset,
        right.offset,
        size,
        left.shape,
        LoopKind.LinearContiguous
      )

  /** C-contiguous unary/scalar plans: no axis coalescing. */
  def linearUnary(layout: Layout): LoopPlan =
    val size = layout.size
    if size == 0 then
      new LoopPlan(
        Array.emptyIntArray,
        Array.emptyIntArray,
        Array.emptyIntArray,
        layout.offset,
        0,
        0,
        layout.shape,
        LoopKind.LinearContiguous
      )
    else
      new LoopPlan(
        Array(size),
        Array(1),
        Array(1),
        layout.offset,
        0,
        size,
        layout.shape,
        LoopKind.LinearContiguous
      )

  private def build(
      leftShape: IArray[Int],
      leftOriginalStrides: IArray[Int],
      leftOffset: Int,
      rightShape: IArray[Int],
      rightOriginalStrides: IArray[Int],
      rightOffset: Int,
      exact: Boolean
  ): LoopPlan =
    if exact && !sameShape(leftShape, rightShape) then
      throw ShapeMismatch(
        leftShape.mkString("(", ", ", ")"),
        rightShape.mkString("(", ", ", ")")
      )
    val rank = if exact then leftShape.length else math.max(leftShape.length, rightShape.length)
    val result = new Array[Int](rank)
    val left = new Array[Int](rank)
    val right = new Array[Int](rank)
    var aligned = rank - 1
    while aligned >= 0 do
      val leftAxis = aligned - (rank - leftShape.length)
      val rightAxis = aligned - (rank - rightShape.length)
      val leftDimension = if leftAxis < 0 then 1 else leftShape(leftAxis)
      val rightDimension = if rightAxis < 0 then 1 else rightShape(rightAxis)
      val dimension =
        if leftDimension == rightDimension then leftDimension
        else if leftDimension == 1 then rightDimension
        else if rightDimension == 1 then leftDimension
        else
          throw BroadcastMismatch(
            leftShape.mkString("(", ", ", ")"),
            rightShape.mkString("(", ", ", ")"),
            aligned - rank
          )
      result(aligned) = dimension
      left(aligned) =
        if leftAxis < 0 || (leftDimension == 1 && dimension != 1) then 0
        else leftOriginalStrides(leftAxis)
      right(aligned) =
        if rightAxis < 0 || (rightDimension == 1 && dimension != 1) then 0
        else rightOriginalStrides(rightAxis)
      aligned -= 1

    val dimensions = ArrayBuffer.empty[Int]
    val leftStrides = ArrayBuffer.empty[Int]
    val rightStrides = ArrayBuffer.empty[Int]
    var axis = 0
    while axis < rank do
      if result(axis) != 1 then
        dimensions += result(axis)
        leftStrides += left(axis)
        rightStrides += right(axis)
      axis += 1

    axis = dimensions.length - 2
    while axis >= 0 do
      val innerSize = dimensions(axis + 1)
      val leftCompatible = coalescible(leftStrides(axis), leftStrides(axis + 1), innerSize)
      val rightCompatible = coalescible(rightStrides(axis), rightStrides(axis + 1), innerSize)
      if leftCompatible && rightCompatible then
        dimensions(axis) = Layout.checkedInt(
          Layout.checkedMultiply(
            dimensions(axis).toLong,
            innerSize.toLong,
            s"coalesced loop dimension at axis $axis"
          ),
          s"coalesced loop dimension at axis $axis"
        )
        leftStrides(axis) = leftStrides(axis + 1)
        rightStrides(axis) = rightStrides(axis + 1)
        val _ = dimensions.remove(axis + 1)
        val _ = leftStrides.remove(axis + 1)
        val _ = rightStrides.remove(axis + 1)
      axis -= 1

    var size = 1L
    axis = 0
    while axis < result.length do
      size = Layout.checkedMultiply(size, result(axis).toLong, s"loop size at axis $axis")
      axis += 1
    val intSize = Layout.checkedInt(size, "loop size")
    val shapeArray = dimensions.toArray
    val leftArray = leftStrides.toArray
    val rightArray = rightStrides.toArray
    val leftScalar = leftArray.forall(_ == 0)
    val rightScalar = rightArray.forall(_ == 0)
    val kind =
      if intSize == 0 ||
        (shapeArray.length <= 1 && leftArray.forall(_ == 1) && rightArray.forall(_ == 1))
      then LoopKind.LinearContiguous
      else if leftScalar || rightScalar then LoopKind.ScalarBroadcast
      else if shapeArray.length <= 2 then LoopKind.InnerStrided
      else LoopKind.GeneralStrided
    new LoopPlan(
      shapeArray,
      leftArray,
      rightArray,
      leftOffset,
      rightOffset,
      intSize,
      IArray.unsafeFromArray(result),
      kind
    )

  private def sameShape(left: IArray[Int], right: IArray[Int]): Boolean =
    if left.length != right.length then false
    else
      var i = 0
      var same = true
      while i < left.length && same do
        same = left(i) == right(i)
        i += 1
      same

  private def coalescible(outer: Int, inner: Int, innerSize: Int): Boolean =
    (outer == 0 && inner == 0) ||
      outer.toLong == Layout.checkedMultiply(
        innerSize.toLong,
        inner.toLong,
        "loop coalescing stride"
      )
