package ravel.internal

import ravel.*

private[ravel] final class Layout private (
    val shape: IArray[Int],
    val strides: IArray[Int],
    val offset: Int,
    val size: Int,
    val minimumPhysicalAddress: Int,
    val flags: Byte,
    val shapeValue: Shape[AnyRank]
):
  def rank: Int = shape.length

  /** Logical C-order values occupy a contiguous memory interval. */
  def isCContiguous: Boolean = (flags & Layout.CContiguous) != 0

  /** Exact canonical C strides, including length-one axes. */
  def isCanonicalLayout: Boolean = (flags & Layout.CanonicalLayout) != 0

  /** The reachable addresses form one packed interval, independent of logical axis order or
    * direction.
    *
    * Singleton axes do not affect density. Thus C-order, Fortran-order, axis permutations,
    * reversals, and their combinations may all be physically dense.
    */
  def isPhysicallyDense: Boolean = (flags & Layout.PhysicallyDense) != 0

  def hasNegativeStride: Boolean = (flags & Layout.HasNegativeStride) != 0
  def hasBroadcastStride: Boolean = (flags & Layout.HasBroadcastStride) != 0

  /** Offset zero and logical size equals the whole storage length. */
  def isWholeBuffer(storageLength: Int): Boolean =
    offset == 0 && size == storageLength

  def normalizedAxis(axis: Int): Int =
    Shape.normalizeAxis(axis, rank)

  def physicalIndex(indices: IArray[Int]): Int =
    if indices.length != rank then throw InvalidIndex.ArityMismatch(rank, indices.length)
    var address = offset.toLong
    var axis = 0
    while axis < rank do
      val index = Layout.normalizeElementIndex(indices(axis), shape(axis), axis)
      address = Layout.checkedAdd(
        address,
        Layout.checkedMultiply(index.toLong, strides(axis).toLong, s"index axis $axis"),
        s"index axis $axis"
      )
      axis += 1
    Layout.checkedInt(address, "physical index")

  private[ravel] def physicalIndex1(i: Int): Int =
    if rank != 1 then invalidArity(1)
    val index0 = Layout.normalizeElementIndex(i, shape(0), 0)
    (offset.toLong + index0.toLong * strides(0).toLong).toInt

  private[ravel] def physicalIndex2(i: Int, j: Int): Int =
    if rank != 2 then invalidArity(2)
    val index0 = Layout.normalizeElementIndex(i, shape(0), 0)
    val index1 = Layout.normalizeElementIndex(j, shape(1), 1)
    (
      offset.toLong +
        index0.toLong * strides(0).toLong +
        index1.toLong * strides(1).toLong
    ).toInt

  private[ravel] def physicalIndex3(i: Int, j: Int, k: Int): Int =
    if rank != 3 then invalidArity(3)
    val index0 = Layout.normalizeElementIndex(i, shape(0), 0)
    val index1 = Layout.normalizeElementIndex(j, shape(1), 1)
    val index2 = Layout.normalizeElementIndex(k, shape(2), 2)
    (
      offset.toLong +
        index0.toLong * strides(0).toLong +
        index1.toLong * strides(1).toLong +
        index2.toLong * strides(2).toLong
    ).toInt

  private[ravel] def physicalIndex4(
      i: Int,
      j: Int,
      k: Int,
      l: Int
  ): Int =
    if rank != 4 then invalidArity(4)
    val index0 = Layout.normalizeElementIndex(i, shape(0), 0)
    val index1 = Layout.normalizeElementIndex(j, shape(1), 1)
    val index2 = Layout.normalizeElementIndex(k, shape(2), 2)
    val index3 = Layout.normalizeElementIndex(l, shape(3), 3)
    (
      offset.toLong +
        index0.toLong * strides(0).toLong +
        index1.toLong * strides(1).toLong +
        index2.toLong * strides(2).toLong +
        index3.toLong * strides(3).toLong
    ).toInt

  private def invalidArity(received: Int): Nothing =
    throw InvalidIndex.ArityMismatch(rank, received)

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
  val CanonicalLayout: Byte = 8
  val PhysicallyDense: Byte = 16

  private[ravel] def sameShape(left: IArray[Int], right: IArray[Int]): Boolean =
    if left.length != right.length then false
    else
      var axis = 0
      var same = true
      while axis < left.length && same do
        same = left(axis) == right(axis)
        axis += 1
      same

  private[ravel] def normalizeElementIndex(index: Int, dimension: Int, axis: Int): Int =
    val normalized = if index < 0 then index + dimension else index
    if normalized < 0 || normalized >= dimension then
      throw InvalidIndex.OutOfBounds(axis, index, dimension)
    normalized

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
    createOwned(
      shape.asInstanceOf[Shape[AnyRank]],
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
    val ownedDimensions =
      IArray.unsafeFromArray(IArray.genericWrapArray(dimensions).toArray)
    val ownedStrides =
      IArray.unsafeFromArray(IArray.genericWrapArray(strides).toArray)
    createOwned(
      Shape.validated[AnyRank](ownedDimensions),
      ownedStrides,
      offset,
      bufferLength,
      requireContiguous
    )

  /** Reuse a validated shape object; strides must already be owned. */
  private def createOwned(
      shapeValue: Shape[AnyRank],
      strides: IArray[Int],
      offset: Int,
      bufferLength: Int,
      requireContiguous: Boolean
  ): Layout =
    val dimensions = shapeValue.unsafeDimensions
    if bufferLength < 0 then throw LayoutOverflow(s"negative buffer length $bufferLength")
    if dimensions.length != strides.length then
      throw InvalidShape(
        s"shape rank ${dimensions.length} differs from stride rank ${strides.length}"
      )
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
      else maximum = checkedAdd(maximum, extent, s"maximum address at axis $axis")
      if strides(axis) == 0 && dimensions(axis) > 1 then broadcast = true
      axis += 1

    if shapeValue.size > 0 then
      if minimum < 0L || maximum >= bufferLength.toLong then
        throw LayoutOverflow(
          s"reachable addresses [$minimum, $maximum] are outside buffer length $bufferLength"
        )
    else if offset < 0 || offset > bufferLength then
      throw LayoutOverflow(s"empty-layout offset $offset is outside [0, $bufferLength]")

    val canonical = isCanonical(dimensions, strides)
    val contiguous = isLogicalContiguous(dimensions, strides)
    val physicallyDense = isPhysicallyDense(dimensions, strides, shapeValue.size)
    if requireContiguous && !canonical then
      throw NonContiguousLayout("canonical layout construction failed")
    var flags: Byte = 0
    if contiguous then flags = (flags | CContiguous).toByte
    if canonical then flags = (flags | CanonicalLayout).toByte
    if physicallyDense then flags = (flags | PhysicallyDense).toByte
    if negative then flags = (flags | HasNegativeStride).toByte
    if broadcast then flags = (flags | HasBroadcastStride).toByte
    new Layout(
      dimensions,
      strides,
      offset,
      shapeValue.size,
      if shapeValue.size == 0 then offset
      else checkedInt(minimum, "minimum physical address"),
      flags,
      shapeValue
    )

  /** Exact canonical strides, including length-one axes. */
  private[ravel] def isCanonical(shape: IArray[Int], strides: IArray[Int]): Boolean =
    var expected = 1L
    var axis = shape.length - 1
    var result = true
    while axis >= 0 && result do
      if strides(axis).toLong != expected then result = false
      expected = checkedMultiply(expected, shape(axis).toLong, s"contiguity axis $axis")
      axis -= 1
    result

  /** Logical C-contiguity: non-singleton axes must have the packed C stride. Length-one axes may
    * carry any stride (including zero from newAxis). A zero-length axis zeros the expected outer
    * stride product.
    */
  private[ravel] def isLogicalContiguous(
      shape: IArray[Int],
      strides: IArray[Int]
  ): Boolean =
    var expected = 1L
    var axis = shape.length - 1
    var result = true
    while axis >= 0 && result do
      val dimension = shape(axis)
      if dimension > 1 then
        if strides(axis).toLong != expected then result = false
        expected = checkedMultiply(expected, dimension.toLong, s"logical contiguity axis $axis")
      else if dimension == 0 then expected = 0L
      axis -= 1
    result

  /** Physical density ignores logical axis order and direction. Non-singleton axes, ordered by
    * absolute stride, must exactly tile the packed interval.
    */
  private def isPhysicallyDense(
      shape: IArray[Int],
      strides: IArray[Int],
      size: Int
  ): Boolean =
    if size == 0 then false
    else if size == 1 then true
    else
      val axes = new Array[Int](shape.length)
      var count = 0
      var axis = 0
      while axis < shape.length do
        if shape(axis) > 1 then
          var position = count
          val stride = Math.abs(strides(axis).toLong)
          while position > 0 &&
            Math.abs(strides(axes(position - 1)).toLong) > stride
          do
            axes(position) = axes(position - 1)
            position -= 1
          axes(position) = axis
          count += 1
        axis += 1

      var expected = 1L
      var index = 0
      var dense = true
      while index < count && dense do
        axis = axes(index)
        if Math.abs(strides(axis).toLong) != expected then dense = false
        else
          expected = checkedMultiply(
            expected,
            shape(axis).toLong,
            s"physical density axis $axis"
          )
        index += 1
      dense && expected == size.toLong

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
