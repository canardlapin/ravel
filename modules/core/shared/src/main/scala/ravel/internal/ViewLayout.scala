package ravel.internal

import ravel.*

private[ravel] object ViewLayout:
  def select(layout: Layout, axis: Int, index: Int, bufferLength: Int): Layout =
    val selectedAxis = layout.normalizedAxis(axis)
    val dimension = layout.shape(selectedAxis)
    val normalizedIndex = Layout.normalizeElementIndex(index, dimension, selectedAxis)
    val shape = new Array[Int](layout.rank - 1)
    val strides = new Array[Int](layout.rank - 1)
    var source = 0
    var target = 0
    while source < layout.rank do
      if source != selectedAxis then
        shape(target) = layout.shape(source)
        strides(target) = layout.strides(source)
        target += 1
      source += 1
    val offset = Layout.checkedInt(
      Layout.checkedAdd(
        layout.offset.toLong,
        Layout.checkedMultiply(
          normalizedIndex.toLong,
          layout.strides(selectedAxis).toLong,
          s"select axis $selectedAxis"
        ),
        s"select axis $selectedAxis"
      ),
      "selected offset"
    )
    Layout.view(
      IArray.unsafeFromArray(shape),
      IArray.unsafeFromArray(strides),
      offset,
      bufferLength
    )

  def slice(layout: Layout, axis: Int, slice: Slice, bufferLength: Int): Layout =
    val slicedAxis = layout.normalizedAxis(axis)
    val dimension = layout.shape(slicedAxis)
    val normalized = slice.normalize(dimension)
    val length = sliceLength(normalized)
    val shape = IArray.genericWrapArray(layout.shape).toArray
    val strides = IArray.genericWrapArray(layout.strides).toArray
    shape(slicedAxis) = length
    strides(slicedAxis) = Layout.checkedInt(
      Layout.checkedMultiply(
        layout.strides(slicedAxis).toLong,
        normalized.step.toLong,
        s"slice stride at axis $slicedAxis"
      ),
      s"slice stride at axis $slicedAxis"
    )
    val offset =
      if layout.size == 0 || length == 0 then 0
      else
        Layout.checkedInt(
          Layout.checkedAdd(
            layout.offset.toLong,
            Layout.checkedMultiply(
              normalized.start.toLong,
              layout.strides(slicedAxis).toLong,
              s"slice offset at axis $slicedAxis"
            ),
            s"slice offset at axis $slicedAxis"
          ),
          s"slice offset at axis $slicedAxis"
        )
    Layout.view(
      IArray.unsafeFromArray(shape),
      IArray.unsafeFromArray(strides),
      offset,
      bufferLength
    )

  def narrow(layout: Layout, plan: NarrowPlan, bufferLength: Int): Layout =
    val shape = IArray.genericWrapArray(layout.shape).toArray
    shape(plan.axis) = plan.length
    val offset =
      if layout.size == 0 || plan.length == 0 then 0
      else
        Layout.checkedInt(
          Layout.checkedAdd(
            layout.offset.toLong,
            Layout.checkedMultiply(
              plan.start.toLong,
              layout.strides(plan.axis).toLong,
              s"narrow offset at axis ${plan.axis}"
            ),
            s"narrow offset at axis ${plan.axis}"
          ),
          s"narrow offset at axis ${plan.axis}"
        )
    Layout.view(
      IArray.unsafeFromArray(shape),
      layout.strides,
      offset,
      bufferLength
    )

  def reverse(layout: Layout, axis: Int, bufferLength: Int): Layout =
    val reversedAxis = layout.normalizedAxis(axis)
    val dimension = layout.shape(reversedAxis)
    if dimension == 0 then
      val strides = IArray.genericWrapArray(layout.strides).toArray
      strides(reversedAxis) = Layout.checkedInt(
        -strides(reversedAxis).toLong,
        s"reverse stride at axis $reversedAxis"
      )
      Layout.view(
        layout.shape,
        IArray.unsafeFromArray(strides),
        0,
        bufferLength
      )
    else
      slice(
        layout,
        reversedAxis,
        Slice(dimension - 1, -1, -1),
        bufferLength
      )

  def permute(layout: Layout, plan: PermutationPlan, bufferLength: Int): Layout =
    val shape = new Array[Int](layout.rank)
    val strides = new Array[Int](layout.rank)
    var i = 0
    while i < layout.rank do
      val sourceAxis = plan.normalizedAxes(i)
      shape(i) = layout.shape(sourceAxis)
      strides(i) = layout.strides(sourceAxis)
      i += 1
    Layout.view(
      IArray.unsafeFromArray(shape),
      IArray.unsafeFromArray(strides),
      layout.offset,
      bufferLength
    )

  def newAxis(layout: Layout, axis: Int, bufferLength: Int): Layout =
    val insertedAxis = normalizeInsertionAxis(axis, layout.rank)
    val shape = new Array[Int](layout.rank + 1)
    val strides = new Array[Int](layout.rank + 1)
    var source = 0
    var target = 0
    while target < shape.length do
      if target == insertedAxis then
        shape(target) = 1
        strides(target) = 0
      else
        shape(target) = layout.shape(source)
        strides(target) = layout.strides(source)
        source += 1
      target += 1
    Layout.view(
      IArray.unsafeFromArray(shape),
      IArray.unsafeFromArray(strides),
      layout.offset,
      bufferLength
    )

  def squeeze(layout: Layout, axis: Int, bufferLength: Int): Layout =
    val squeezedAxis = layout.normalizedAxis(axis)
    if layout.shape(squeezedAxis) != 1 then
      throw InvalidShape(
        s"cannot squeeze axis $squeezedAxis of length ${layout.shape(squeezedAxis)}"
      )
    val shape = new Array[Int](layout.rank - 1)
    val strides = new Array[Int](layout.rank - 1)
    var source = 0
    var target = 0
    while source < layout.rank do
      if source != squeezedAxis then
        shape(target) = layout.shape(source)
        strides(target) = layout.strides(source)
        target += 1
      source += 1
    Layout.view(
      IArray.unsafeFromArray(shape),
      IArray.unsafeFromArray(strides),
      layout.offset,
      bufferLength
    )

  def broadcastTo[S <: AnyRank](
      layout: Layout,
      target: Shape[S],
      bufferLength: Int
  ): Layout =
    if target.rank < layout.rank then
      throw BroadcastMismatch(
        layout.shape.mkString("(", ", ", ")"),
        target.toString,
        -target.rank - 1
      )
    val strides = new Array[Int](target.rank)
    var targetAxis = target.rank - 1
    var sourceAxis = layout.rank - 1
    while targetAxis >= 0 do
      val targetDimension = target.unsafeDimensions(targetAxis)
      if sourceAxis < 0 then strides(targetAxis) = 0
      else
        val sourceDimension = layout.shape(sourceAxis)
        if sourceDimension == targetDimension then strides(targetAxis) = layout.strides(sourceAxis)
        else if sourceDimension == 1 then strides(targetAxis) = 0
        else
          throw BroadcastMismatch(
            layout.shape.mkString("(", ", ", ")"),
            target.toString,
            targetAxis - target.rank
          )
        sourceAxis -= 1
      targetAxis -= 1
    Layout.view(
      target.unsafeDimensions,
      IArray.unsafeFromArray(strides),
      layout.offset,
      bufferLength
    )

  def reshape[S <: AnyRank](
      layout: Layout,
      target: Shape[S],
      bufferLength: Int
  ): Layout =
    if layout.size != target.size then
      throw ShapeMismatch(
        layout.shape.mkString("(", ", ", ")"),
        target.toString
      )
    if target.size == 0 then return Layout.contiguous(target, bufferLength)

    val sourceAxes = Vector.newBuilder[(Int, Int)]
    var axis = 0
    while axis < layout.rank do
      val dimension = layout.shape(axis)
      val stride = layout.strides(axis)
      if dimension > 1 then
        if stride == 0 then
          throw NonContiguousLayout(
            s"cannot reshape broadcast axis $axis of length $dimension"
          )
        sourceAxes += ((dimension, stride))
      axis += 1
    val axes = sourceAxes.result()

    val targetNonSingleton = Vector.newBuilder[Int]
    axis = 0
    while axis < target.rank do
      if target.unsafeDimensions(axis) > 1 then targetNonSingleton += axis
      axis += 1
    val targetAxes = targetNonSingleton.result()
    val resultStrides = Array.fill(target.rank)(0)

    if axes.isEmpty then
      return Layout.view(
        target.unsafeDimensions,
        IArray.unsafeFromArray(resultStrides),
        layout.offset,
        bufferLength
      )

    final case class Block(size: Long, innerStride: Int)
    val blocks = Vector.newBuilder[Block]
    var blockSize = axes.head._1.toLong
    var index = 0
    while index < axes.length - 1 do
      val (_, outerStride) = axes(index)
      val (innerSize, innerStride) = axes(index + 1)
      val coalescedStride = Layout.checkedMultiply(
        innerSize.toLong,
        innerStride.toLong,
        s"reshape source axes $index/${index + 1}"
      )
      if outerStride.toLong == coalescedStride then
        blockSize = Layout.checkedMultiply(
          blockSize,
          innerSize.toLong,
          s"reshape block at axis $index"
        )
      else
        blocks += Block(blockSize, outerStride)
        blockSize = innerSize.toLong
      index += 1
    blocks += Block(blockSize, axes.last._2)
    val sourceBlocks = blocks.result()

    var targetPosition = 0
    var blockPosition = 0
    while blockPosition < sourceBlocks.length do
      val block = sourceBlocks(blockPosition)
      val groupStart = targetPosition
      var product = 1L
      while targetPosition < targetAxes.length && product < block.size do
        product = Layout.checkedMultiply(
          product,
          target.unsafeDimensions(targetAxes(targetPosition)).toLong,
          s"reshape target block $blockPosition"
        )
        targetPosition += 1
      if product != block.size then
        throw NonContiguousLayout(
          s"target dimensions cannot partition source contiguous block ${block.size}"
        )
      var running = block.innerStride.toLong
      var group = targetPosition - 1
      while group >= groupStart do
        val targetAxis = targetAxes(group)
        resultStrides(targetAxis) =
          Layout.checkedInt(running, s"reshape target stride at axis $targetAxis")
        running = Layout.checkedMultiply(
          running,
          target.unsafeDimensions(targetAxis).toLong,
          s"reshape target stride product at axis $targetAxis"
        )
        group -= 1
      blockPosition += 1
    if targetPosition != targetAxes.length then
      throw NonContiguousLayout("target has dimensions left after source blocks")

    Layout.view(
      target.unsafeDimensions,
      IArray.unsafeFromArray(resultStrides),
      layout.offset,
      bufferLength
    )

  private def sliceLength(slice: Slice): Int =
    val length =
      if slice.step > 0 then
        if slice.start >= slice.stopExclusive then 0L
        else 1L + (slice.stopExclusive.toLong - 1L - slice.start.toLong) / slice.step.toLong
      else if slice.start <= slice.stopExclusive then 0L
      else 1L + (slice.start.toLong - 1L - slice.stopExclusive.toLong) / -slice.step.toLong
    Layout.checkedInt(length, "slice length")

  private def normalizeInsertionAxis(axis: Int, rank: Int): Int =
    val normalized = if axis < 0 then axis + rank + 1 else axis
    if normalized < 0 || normalized > rank then throw InvalidAxis(axis, rank + 1)
    normalized
