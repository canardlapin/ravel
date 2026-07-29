package ravel.internal

import scala.scalajs.js.typedarray.{Float32Array, Float64Array}

/** Scala.js pairwise reductions over typed-array storage. */
private[ravel] object PlatformReduction:
  def pairwiseContiguousDouble(
      storage: DoubleStorage,
      offset: Int,
      size: Int,
      scratch: Array[Double]
  ): Double =
    val raw = storage.raw
    var block = 0
    var index = 0
    var physical = offset
    while index < size do
      var sum = 0.0
      val until = math.min(index + ReductionKernels.PairwiseBlockSize, size)
      while index < until do
        sum += raw(physical)
        physical += 1
        index += 1
      scratch(block) = sum
      block += 1
    mergeDouble(scratch, block)

  def pairwiseContiguousFloat(
      storage: FloatStorage,
      offset: Int,
      size: Int,
      scratch: Array[Float]
  ): Float =
    val raw = storage.raw
    var block = 0
    var index = 0
    var physical = offset
    while index < size do
      var sum = 0.0f
      val until = math.min(index + ReductionKernels.PairwiseBlockSize, size)
      while index < until do
        sum = (sum + raw(physical).toFloat).toFloat
        physical += 1
        index += 1
      scratch(block) = sum
      block += 1
    mergeFloat(scratch, block)

  def pairwiseContiguousDoubleFromFloat(
      storage: FloatStorage,
      offset: Int,
      size: Int,
      scratch: Array[Double]
  ): Double =
    // Compute independent complete blocks in parallel lanes. Each block and
    // the subsequent merge retain the documented fixed pairwise schedule.
    val raw = storage.raw
    var block = 0
    val blockSize = ReductionKernels.PairwiseBlockSize
    val completeBlocks = size / blockSize
    while block + 3 < completeBlocks do
      val base = offset + block * blockSize
      var sum0 = 0.0
      var sum1 = 0.0
      var sum2 = 0.0
      var sum3 = 0.0
      var index = 0
      while index < blockSize do
        sum0 += raw(base + index).toDouble
        sum1 += raw(base + blockSize + index).toDouble
        sum2 += raw(base + 2 * blockSize + index).toDouble
        sum3 += raw(base + 3 * blockSize + index).toDouble
        index += 1
      scratch(block) = sum0
      scratch(block + 1) = sum1
      scratch(block + 2) = sum2
      scratch(block + 3) = sum3
      block += 4
    var index = block * blockSize
    var physical = offset + index
    while index < size do
      var sum = 0.0
      val until = math.min(index + blockSize, size)
      while index < until do
        sum += raw(physical).toDouble
        physical += 1
        index += 1
      scratch(block) = sum
      block += 1
    mergeDouble(scratch, block)

  def axis0PairwiseDouble(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan,
      blockValues: Array[Double],
      workspace: Array[Double]
  ): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    val raw = source.raw
    val out = output.raw
    if rows == 0 then
      var col = 0
      while col < cols do
        out(col) = 0.0
        col += 1
    else
      val blockCount =
        (rows + ReductionKernels.PairwiseBlockSize - 1) / ReductionKernels.PairwiseBlockSize
      var block = 0
      var rowStart = 0
      while rowStart < rows do
        val rowUntil = math.min(rowStart + ReductionKernels.PairwiseBlockSize, rows)
        var col = 0
        while col < cols do
          workspace(col) = 0.0
          col += 1
        var row = rowStart
        while row < rowUntil do
          var physical = offset + row * rowStride
          col = 0
          while col < cols do
            workspace(col) += raw(physical)
            physical += colStride
            col += 1
          row += 1
        col = 0
        while col < cols do
          blockValues(block * cols + col) = workspace(col)
          col += 1
        block += 1
        rowStart = rowUntil
      var col = 0
      while col < cols do
        var b = 0
        while b < blockCount do
          workspace(b) = blockValues(b * cols + col)
          b += 1
        out(col) = mergeDouble(workspace, blockCount)
        col += 1

  def axis1PairwiseDouble(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  ): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    val raw = source.raw
    val out = output.raw
    val blockCount =
      (cols + ReductionKernels.PairwiseBlockSize - 1) / ReductionKernels.PairwiseBlockSize
    val blocks = new Array[Double](math.max(blockCount, 1))
    var row = 0
    while row < rows do
      if cols == 0 then out(row) = 0.0
      else
        fillStridedDouble(raw, offset + row * rowStride, colStride, cols, blocks)
        out(row) = mergeDouble(blocks, blockCount)
      row += 1

  def axis0PairwiseFloat(
      source: FloatStorage,
      output: FloatStorage,
      plan: ReductionPlan,
      blockValues: Array[Float],
      workspace: Array[Float]
  ): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    val raw = source.raw
    val out = output.raw
    if rows == 0 then
      var col = 0
      while col < cols do
        out(col) = 0.0f
        col += 1
    else
      val blockCount =
        (rows + ReductionKernels.PairwiseBlockSize - 1) / ReductionKernels.PairwiseBlockSize
      var block = 0
      var rowStart = 0
      while rowStart < rows do
        val rowUntil = math.min(rowStart + ReductionKernels.PairwiseBlockSize, rows)
        var col = 0
        while col < cols do
          workspace(col) = 0.0f
          col += 1
        var row = rowStart
        while row < rowUntil do
          var physical = offset + row * rowStride
          col = 0
          while col < cols do
            workspace(col) = (workspace(col) + raw(physical).toFloat).toFloat
            physical += colStride
            col += 1
          row += 1
        col = 0
        while col < cols do
          blockValues(block * cols + col) = workspace(col)
          col += 1
        block += 1
        rowStart = rowUntil
      var col = 0
      while col < cols do
        var b = 0
        while b < blockCount do
          workspace(b) = blockValues(b * cols + col)
          b += 1
        out(col) = mergeFloat(workspace, blockCount)
        col += 1

  def axis1PairwiseFloat(
      source: FloatStorage,
      output: FloatStorage,
      plan: ReductionPlan
  ): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    val raw = source.raw
    val out = output.raw
    val blockCount =
      (cols + ReductionKernels.PairwiseBlockSize - 1) / ReductionKernels.PairwiseBlockSize
    val blocks = new Array[Float](math.max(blockCount, 1))
    var row = 0
    while row < rows do
      if cols == 0 then out(row) = 0.0f
      else
        fillStridedFloat(raw, offset + row * rowStride, colStride, cols, blocks)
        out(row) = mergeFloat(blocks, blockCount)
      row += 1

  private def fillStridedDouble(
      raw: Float64Array,
      offset: Int,
      stride: Int,
      length: Int,
      blocks: Array[Double]
  ): Unit =
    var block = 0
    var index = 0
    var physical = offset
    while index < length do
      var sum = 0.0
      val until = math.min(index + ReductionKernels.PairwiseBlockSize, length)
      while index < until do
        sum += raw(physical)
        physical += stride
        index += 1
      blocks(block) = sum
      block += 1

  private def fillStridedFloat(
      raw: Float32Array,
      offset: Int,
      stride: Int,
      length: Int,
      blocks: Array[Float]
  ): Unit =
    var block = 0
    var index = 0
    var physical = offset
    while index < length do
      var sum = 0.0f
      val until = math.min(index + ReductionKernels.PairwiseBlockSize, length)
      while index < until do
        sum = (sum + raw(physical).toFloat).toFloat
        physical += stride
        index += 1
      blocks(block) = sum
      block += 1

  private def mergeDouble(blocks: Array[Double], initialCount: Int): Double =
    var count = initialCount
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        blocks(write) = blocks(read) + blocks(read + 1)
        read += 2
        write += 1
      if read < count then
        blocks(write) = blocks(read)
        write += 1
      count = write
    if initialCount == 0 then 0.0 else blocks(0)

  private def mergeFloat(blocks: Array[Float], initialCount: Int): Float =
    var count = initialCount
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        blocks(write) = (blocks(read) + blocks(read + 1)).toFloat
        read += 2
        write += 1
      if read < count then
        blocks(write) = blocks(read)
        write += 1
      count = write
    if initialCount == 0 then 0.0f else blocks(0)
