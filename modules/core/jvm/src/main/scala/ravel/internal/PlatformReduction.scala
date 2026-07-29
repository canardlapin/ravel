package ravel.internal

/** JVM monomorphic pairwise reductions over owned storage arrays. */
private[ravel] object PlatformReduction:
  def pairwiseContiguousDouble(
      storage: DoubleStorage,
      offset: Int,
      size: Int,
      scratch: Array[Double]
  ): Double =
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
        sum0 += raw(base + index)
        sum1 += raw(base + blockSize + index)
        sum2 += raw(base + 2 * blockSize + index)
        sum3 += raw(base + 3 * blockSize + index)
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
    val blockSize = ReductionKernels.PairwiseBlockSize
    val completeBlocks = size / blockSize
    while block + 3 < completeBlocks do
      val base = offset + block * blockSize
      var sum0 = 0.0f
      var sum1 = 0.0f
      var sum2 = 0.0f
      var sum3 = 0.0f
      var index = 0
      while index < blockSize do
        sum0 = (sum0 + raw(base + index)).toFloat
        sum1 = (sum1 + raw(base + blockSize + index)).toFloat
        sum2 = (sum2 + raw(base + 2 * blockSize + index)).toFloat
        sum3 = (sum3 + raw(base + 3 * blockSize + index)).toFloat
        index += 1
      scratch(block) = sum0
      scratch(block + 1) = sum1
      scratch(block + 2) = sum2
      scratch(block + 3) = sum3
      block += 4
    var index = block * blockSize
    var physical = offset + index
    while index < size do
      var sum = 0.0f
      val until = math.min(index + blockSize, size)
      while index < until do
        sum = (sum + raw(physical)).toFloat
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
    val scratchSize = math.max(blockCount, 1)
    val blocks0 = new Array[Double](scratchSize)
    val blocks1 = new Array[Double](scratchSize)
    val blocks2 = new Array[Double](scratchSize)
    val blocks3 = new Array[Double](scratchSize)
    var row = 0
    while row + 3 < rows do
      if cols == 0 then
        out(row) = 0.0
        out(row + 1) = 0.0
        out(row + 2) = 0.0
        out(row + 3) = 0.0
      else
        fillFourRowsDouble(
          raw,
          offset + row * rowStride,
          rowStride,
          colStride,
          cols,
          blocks0,
          blocks1,
          blocks2,
          blocks3
        )
        out(row) = mergeDouble(blocks0, blockCount)
        out(row + 1) = mergeDouble(blocks1, blockCount)
        out(row + 2) = mergeDouble(blocks2, blockCount)
        out(row + 3) = mergeDouble(blocks3, blockCount)
      row += 4
    while row < rows do
      if cols == 0 then out(row) = 0.0
      else
        fillStridedDouble(raw, offset + row * rowStride, colStride, cols, blocks0)
        out(row) = mergeDouble(blocks0, blockCount)
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
            workspace(col) = (workspace(col) + raw(physical)).toFloat
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
    val scratchSize = math.max(blockCount, 1)
    val blocks0 = new Array[Float](scratchSize)
    val blocks1 = new Array[Float](scratchSize)
    val blocks2 = new Array[Float](scratchSize)
    val blocks3 = new Array[Float](scratchSize)
    var row = 0
    while row + 3 < rows do
      if cols == 0 then
        out(row) = 0.0f
        out(row + 1) = 0.0f
        out(row + 2) = 0.0f
        out(row + 3) = 0.0f
      else
        fillFourRowsFloat(
          raw,
          offset + row * rowStride,
          rowStride,
          colStride,
          cols,
          blocks0,
          blocks1,
          blocks2,
          blocks3
        )
        out(row) = mergeFloat(blocks0, blockCount)
        out(row + 1) = mergeFloat(blocks1, blockCount)
        out(row + 2) = mergeFloat(blocks2, blockCount)
        out(row + 3) = mergeFloat(blocks3, blockCount)
      row += 4
    while row < rows do
      if cols == 0 then out(row) = 0.0f
      else
        fillStridedFloat(raw, offset + row * rowStride, colStride, cols, blocks0)
        out(row) = mergeFloat(blocks0, blockCount)
      row += 1

  private def fillFourRowsDouble(
      raw: Array[Double],
      offset: Int,
      rowStride: Int,
      colStride: Int,
      length: Int,
      blocks0: Array[Double],
      blocks1: Array[Double],
      blocks2: Array[Double],
      blocks3: Array[Double]
  ): Unit =
    var block = 0
    var index = 0
    var physical0 = offset
    var physical1 = offset + rowStride
    var physical2 = offset + 2 * rowStride
    var physical3 = offset + 3 * rowStride
    while index < length do
      var sum0 = 0.0
      var sum1 = 0.0
      var sum2 = 0.0
      var sum3 = 0.0
      val until = math.min(index + ReductionKernels.PairwiseBlockSize, length)
      while index < until do
        sum0 += raw(physical0)
        sum1 += raw(physical1)
        sum2 += raw(physical2)
        sum3 += raw(physical3)
        physical0 += colStride
        physical1 += colStride
        physical2 += colStride
        physical3 += colStride
        index += 1
      blocks0(block) = sum0
      blocks1(block) = sum1
      blocks2(block) = sum2
      blocks3(block) = sum3
      block += 1

  private def fillFourRowsFloat(
      raw: Array[Float],
      offset: Int,
      rowStride: Int,
      colStride: Int,
      length: Int,
      blocks0: Array[Float],
      blocks1: Array[Float],
      blocks2: Array[Float],
      blocks3: Array[Float]
  ): Unit =
    var block = 0
    var index = 0
    var physical0 = offset
    var physical1 = offset + rowStride
    var physical2 = offset + 2 * rowStride
    var physical3 = offset + 3 * rowStride
    while index < length do
      var sum0 = 0.0f
      var sum1 = 0.0f
      var sum2 = 0.0f
      var sum3 = 0.0f
      val until = math.min(index + ReductionKernels.PairwiseBlockSize, length)
      while index < until do
        sum0 = (sum0 + raw(physical0)).toFloat
        sum1 = (sum1 + raw(physical1)).toFloat
        sum2 = (sum2 + raw(physical2)).toFloat
        sum3 = (sum3 + raw(physical3)).toFloat
        physical0 += colStride
        physical1 += colStride
        physical2 += colStride
        physical3 += colStride
        index += 1
      blocks0(block) = sum0
      blocks1(block) = sum1
      blocks2(block) = sum2
      blocks3(block) = sum3
      block += 1

  private def fillStridedDouble(
      raw: Array[Double],
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
      raw: Array[Float],
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
        sum = (sum + raw(physical)).toFloat
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
