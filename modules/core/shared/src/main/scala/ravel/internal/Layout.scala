package ravel.internal

import ravel.*

private[ravel] final class Layout private (
    val shape: IArray[Int],
    val strides: IArray[Int],
    val offset: Int,
    val size: Int,
    val flags: Byte
):
  def rank: Int = shape.length

  def isCContiguous: Boolean = (flags & Layout.CContiguous) != 0
  def hasNegativeStride: Boolean = (flags & Layout.HasNegativeStride) != 0
  def hasBroadcastStride: Boolean = (flags & Layout.HasBroadcastStride) != 0

  def normalizedAxis(axis: Int): Int =
    Shape.normalizeAxis(axis, rank)

  def physicalIndex(indices: IArray[Int]): Int =
    if indices.length != rank then
      throw InvalidIndex(s"expected $rank indices but received ${indices.length}")
    var address = offset.toLong
    var axis = 0
    while axis < rank do
      val index = indices(axis)
      val dimension = shape(axis)
      if index < 0 || index >= dimension then
        throw InvalidIndex(s"axis $axis index $index is outside [0, $dimension)")
      address = Layout.checkedAdd(
        address,
        Layout.checkedMultiply(index.toLong, strides(axis).toLong, s"index axis $axis"),
        s"index axis $axis"
      )
      axis += 1
    Layout.checkedInt(address, "physical index")

  def foreachPhysicalIndex(f: Int => Unit): Unit =
    if size == 0 then return
    if rank == 0 then
      f(offset)
      return
    val counters = new Array[Int](rank)
    var address = offset.toLong
    var visited = 0
    while visited < size do
      f(Layout.checkedInt(address, "iteration address"))
      visited += 1
      if visited < size then
        var axis = rank - 1
        var advanced = false
        while axis >= 0 && !advanced do
          counters(axis) += 1
          address = Layout.checkedAdd(address, strides(axis).toLong, s"advance axis $axis")
          if counters(axis) < shape(axis) then advanced = true
          else
            address = Layout.checkedAdd(
              address,
              -Layout.checkedMultiply(
                counters(axis).toLong,
                strides(axis).toLong,
                s"rewind axis $axis"
              ),
              s"rewind axis $axis"
            )
            counters(axis) = 0
            axis -= 1

private[ravel] object Layout:
  val CContiguous: Byte = 1
  val HasNegativeStride: Byte = 2
  val HasBroadcastStride: Byte = 4

  def contiguous[R <: AnyRank](shape: Shape[R], bufferLength: Int): Layout =
    val dimensions = shape.unsafeDimensions
    val strides = new Array[Int](dimensions.length)
    var running = 1L
    var axis = dimensions.length - 1
    while axis >= 0 do
      strides(axis) = checkedInt(running, s"canonical stride at axis $axis")
      running = checkedMultiply(
        running,
        dimensions(axis).toLong,
        s"canonical stride product at axis $axis"
      )
      if running > Int.MaxValue.toLong then
        throw LayoutOverflow(s"canonical stride product $running exceeds Int at axis $axis")
      axis -= 1
    create(
      dimensions,
      IArray.unsafeFromArray(strides),
      0,
      bufferLength,
      requireContiguous = true
    )

  def view(
      shape: IArray[Int],
      strides: IArray[Int],
      offset: Int,
      bufferLength: Int
  ): Layout =
    create(shape, strides, offset, bufferLength, requireContiguous = false)

  private def create(
      dimensions: IArray[Int],
      strides: IArray[Int],
      offset: Int,
      bufferLength: Int,
      requireContiguous: Boolean
  ): Layout =
    if bufferLength < 0 then throw LayoutOverflow(s"negative buffer length $bufferLength")
    if dimensions.length != strides.length then
      throw InvalidShape(
        s"shape rank ${dimensions.length} differs from stride rank ${strides.length}"
      )
    val shape = Shape.unsafeRanked[AnyRank](dimensions)
    var minimum = offset.toLong
    var maximum = offset.toLong
    var negative = false
    var broadcast = false
    var axis = 0
    while axis < dimensions.length do
      val extent = checkedMultiply(
        (dimensions(axis) - 1).toLong,
        strides(axis).toLong,
        s"reachable extent at axis $axis"
      )
      if strides(axis) < 0 then
        negative = true
        minimum = checkedAdd(minimum, extent, s"minimum address at axis $axis")
      else
        maximum = checkedAdd(maximum, extent, s"maximum address at axis $axis")
      if strides(axis) == 0 && dimensions(axis) > 1 then broadcast = true
      axis += 1

    if shape.size > 0 then
      if minimum < 0L || maximum >= bufferLength.toLong then
        throw LayoutOverflow(
          s"reachable addresses [$minimum, $maximum] are outside buffer length $bufferLength"
        )
    else if offset < 0 || offset > bufferLength then
      throw LayoutOverflow(s"empty-layout offset $offset is outside [0, $bufferLength]")

    val contiguous = isCanonical(dimensions, strides)
    if requireContiguous && !contiguous then
      throw NonContiguousLayout("canonical layout construction failed")
    var flags: Byte = 0
    if contiguous then flags = (flags | CContiguous).toByte
    if negative then flags = (flags | HasNegativeStride).toByte
    if broadcast then flags = (flags | HasBroadcastStride).toByte
    new Layout(
      IArray.unsafeFromArray(IArray.genericWrapArray(dimensions).toArray),
      IArray.unsafeFromArray(IArray.genericWrapArray(strides).toArray),
      offset,
      shape.size,
      flags
    )

  private def isCanonical(shape: IArray[Int], strides: IArray[Int]): Boolean =
    var expected = 1L
    var axis = shape.length - 1
    var result = true
    while axis >= 0 && result do
      if strides(axis).toLong != expected then result = false
      expected = checkedMultiply(expected, shape(axis).toLong, s"contiguity axis $axis")
      axis -= 1
    result

  private[ravel] def checkedMultiply(left: Long, right: Long, context: String): Long =
    try Math.multiplyExact(left, right)
    catch
      case _: ArithmeticException =>
        throw LayoutOverflow(s"$context overflows Long: $left * $right")

  private[ravel] def checkedAdd(left: Long, right: Long, context: String): Long =
    try Math.addExact(left, right)
    catch
      case _: ArithmeticException =>
        throw LayoutOverflow(s"$context overflows Long: $left + $right")

  private[ravel] def checkedInt(value: Long, context: String): Int =
    if value < Int.MinValue.toLong || value > Int.MaxValue.toLong then
      throw LayoutOverflow(s"$context $value cannot be represented as Int")
    value.toInt
