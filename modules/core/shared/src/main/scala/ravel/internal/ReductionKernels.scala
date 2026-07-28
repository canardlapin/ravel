package ravel.internal

import ravel.*

private[ravel] object ReductionKernels:
  val PairwiseBlockSize: Int = 128

  /** Per-thread scratch for pairwise block vectors. Avoids allocating on every public sum. */
  private final class FloatScratchPad:
    var primary: Array[Float] = Array.emptyFloatArray
    var secondary: Array[Float] = Array.emptyFloatArray
    def ensurePrimary(n: Int): Array[Float] =
      if primary.length < n then primary = new Array[Float](n)
      primary
    def ensureSecondary(n: Int): Array[Float] =
      if secondary.length < n then secondary = new Array[Float](n)
      secondary

  private final class DoubleScratchPad:
    var primary: Array[Double] = Array.emptyDoubleArray
    var secondary: Array[Double] = Array.emptyDoubleArray
    def ensurePrimary(n: Int): Array[Double] =
      if primary.length < n then primary = new Array[Double](n)
      primary
    def ensureSecondary(n: Int): Array[Double] =
      if secondary.length < n then secondary = new Array[Double](n)
      secondary

  private val floatScratchPads = new java.lang.ThreadLocal[FloatScratchPad]:
    override def initialValue(): FloatScratchPad = new FloatScratchPad
  private val doubleScratchPads = new java.lang.ThreadLocal[DoubleScratchPad]:
    override def initialValue(): DoubleScratchPad = new DoubleScratchPad

  private def floatScratch(n: Int): Array[Float] =
    floatScratchPads.get().ensurePrimary(math.max(n, 1))
  private def floatScratch2(n: Int): Array[Float] =
    floatScratchPads.get().ensureSecondary(math.max(n, 1))
  private def doubleScratch(n: Int): Array[Double] =
    doubleScratchPads.get().ensurePrimary(math.max(n, 1))
  private def doubleScratch2(n: Int): Array[Double] =
    doubleScratchPads.get().ensureSecondary(math.max(n, 1))

  def sum[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: IntStorage => logicalFold(layout, x.raw.apply, 0)(_ + _)
      case x: LongStorage => logicalFold(layout, x.raw.apply, 0L)(_ + _)
      case x: FloatStorage => pairwiseFloatStorage(x, layout)
      case x: DoubleStorage => pairwiseDoubleStorage(x, layout)
      case _ => throw new UnsupportedOperationException("sum requires Int, Long, Float, or Double")
    ).asInstanceOf[A]

  def product[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: IntStorage => logicalFold(layout, x.raw.apply, 1)(_ * _)
      case x: LongStorage => logicalFold(layout, x.raw.apply, 1L)(_ * _)
      case x: FloatStorage => logicalFold(layout, x.raw.apply, 1.0f)((a, b) => (a * b).toFloat)
      case x: DoubleStorage => logicalFold(layout, x.raw.apply, 1.0)(_ * _)
      case _ =>
        throw new UnsupportedOperationException("product requires Int, Long, Float, or Double")
    ).asInstanceOf[A]

  def minimum[A](source: Storage[A], layout: Layout): A =
    if layout.size == 0 then throw EmptyReduction("min")
    (source match
      case x: ByteStorage => logicalExtremum(layout, x.raw.apply)(math.min(_, _).toByte)
      case x: ShortStorage => logicalExtremum(layout, x.raw.apply)(math.min(_, _).toShort)
      case x: IntStorage => logicalExtremum(layout, x.raw.apply)(math.min)
      case x: LongStorage => logicalExtremum(layout, x.raw.apply)(math.min)
      case x: FloatStorage => logicalExtremum(layout, x.raw.apply)((a, b) => math.min(a, b).toFloat)
      case x: DoubleStorage => logicalExtremum(layout, x.raw.apply)(math.min)
      case _ => throw new UnsupportedOperationException("min requires an ordered numeric dtype")
    ).asInstanceOf[A]

  def maximum[A](source: Storage[A], layout: Layout): A =
    if layout.size == 0 then throw EmptyReduction("max")
    (source match
      case x: ByteStorage => logicalExtremum(layout, x.raw.apply)(math.max(_, _).toByte)
      case x: ShortStorage => logicalExtremum(layout, x.raw.apply)(math.max(_, _).toShort)
      case x: IntStorage => logicalExtremum(layout, x.raw.apply)(math.max)
      case x: LongStorage => logicalExtremum(layout, x.raw.apply)(math.max)
      case x: FloatStorage => logicalExtremum(layout, x.raw.apply)((a, b) => math.max(a, b).toFloat)
      case x: DoubleStorage => logicalExtremum(layout, x.raw.apply)(math.max)
      case _ => throw new UnsupportedOperationException("max requires an ordered numeric dtype")
    ).asInstanceOf[A]

  def argMinimum[A](source: Storage[A], layout: Layout): Int =
    if layout.size == 0 then throw EmptyReduction("argMin")
    source match
      case x: ByteStorage => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: ShortStorage => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: IntStorage => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: LongStorage => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: FloatStorage => logicalArgFloat(layout, x.raw.apply, minimum = true)
      case x: DoubleStorage => logicalArgDouble(layout, x.raw.apply, minimum = true)
      case _ => throw new UnsupportedOperationException("argMin requires an ordered numeric dtype")

  def argMaximum[A](source: Storage[A], layout: Layout): Int =
    if layout.size == 0 then throw EmptyReduction("argMax")
    source match
      case x: ByteStorage => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: ShortStorage => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: IntStorage => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: LongStorage => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: FloatStorage => logicalArgFloat(layout, x.raw.apply, minimum = false)
      case x: DoubleStorage => logicalArgDouble(layout, x.raw.apply, minimum = false)
      case _ => throw new UnsupportedOperationException("argMax requires an ordered numeric dtype")

  def mean[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: FloatStorage =>
        if layout.size == 0 then Float.NaN
        else (pairwiseDoubleFromFloat(layout, x.raw.apply) / layout.size.toDouble).toFloat
      case x: DoubleStorage =>
        if layout.size == 0 then Double.NaN
        else pairwiseDouble(layout, x.raw.apply) / layout.size.toDouble
      case _ => throw new UnsupportedOperationException("mean requires Float or Double")
    ).asInstanceOf[A]

  def sumAsLong(source: Storage[Int], layout: Layout): Long =
    val raw = source.asInstanceOf[IntStorage].raw
    logicalFold(layout, raw.apply, 0L)((acc, value) => acc + value.toLong)

  def sumAsDouble(source: Storage[Float], layout: Layout): Double =
    pairwiseDoubleFromFloat(layout, source.asInstanceOf[FloatStorage].raw.apply)

  def sumAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    (source, output) match
      case (x: IntStorage, z: IntStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 0)(_ + _)
      case (x: LongStorage, z: LongStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 0L)(_ + _)
      case (x: FloatStorage, z: FloatStorage) =>
        axisPairwiseFloatStorage(x, z, plan)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisPairwiseDoubleStorage(x, z, plan)
      case _ => throw new UnsupportedOperationException("sum requires matching arithmetic storage")

  def productAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    (source, output) match
      case (x: IntStorage, z: IntStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 1)(_ * _)
      case (x: LongStorage, z: LongStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 1L)(_ * _)
      case (x: FloatStorage, z: FloatStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 1.0f)((a, b) => (a * b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisFold(plan, x.raw.apply, z.raw.update, 1.0)(_ * _)
      case _ =>
        throw new UnsupportedOperationException("product requires matching arithmetic storage")

  def minimumAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then throw EmptyReduction("min")
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min(_, _).toByte)
      case (x: ShortStorage, z: ShortStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min(_, _).toShort)
      case (x: IntStorage, z: IntStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case (x: LongStorage, z: LongStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case (x: FloatStorage, z: FloatStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)((a, b) => math.min(a, b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case _ => throw new UnsupportedOperationException("min requires matching ordered storage")

  def maximumAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then throw EmptyReduction("max")
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max(_, _).toByte)
      case (x: ShortStorage, z: ShortStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max(_, _).toShort)
      case (x: IntStorage, z: IntStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case (x: LongStorage, z: LongStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case (x: FloatStorage, z: FloatStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)((a, b) => math.max(a, b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case _ => throw new UnsupportedOperationException("max requires matching ordered storage")

  def meanAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    (source, output) match
      case (x: FloatStorage, z: FloatStorage) =>
        axisPairwiseMeanFloat(plan, x.raw.apply, z.raw.update)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisPairwiseMeanDouble(plan, x.raw.apply, z.raw.update)
      case _ => throw new UnsupportedOperationException("mean requires matching floating storage")

  def argMinimumAxis[A](
      source: Storage[A],
      output: Storage[Int],
      plan: ReductionPlan
  ): Unit =
    axisArgDispatch(source, output, plan, minimum = true)

  def argMaximumAxis[A](
      source: Storage[A],
      output: Storage[Int],
      plan: ReductionPlan
  ): Unit =
    axisArgDispatch(source, output, plan, minimum = false)

  private def axisArgDispatch[A](
      source: Storage[A],
      output: Storage[Int],
      plan: ReductionPlan,
      minimum: Boolean
  ): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then
      throw EmptyReduction(if minimum then "argMin" else "argMax")
    val target = output.asInstanceOf[IntStorage]
    source match
      case x: ByteStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: ShortStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: IntStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: LongStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: FloatStorage =>
        axisArgFloat(plan, x.raw.apply, target.raw.update, minimum)
      case x: DoubleStorage =>
        axisArgDouble(plan, x.raw.apply, target.raw.update, minimum)
      case _ => throw new UnsupportedOperationException("arg reduction requires ordered storage")

  private inline def logicalFold[T, U](
      layout: Layout,
      inline read: Int => T,
      initial: U
  )(inline combine: (U, T) => U): U =
    var result = initial
    foreachLogical(layout) { physical =>
      result = combine(result, read(physical))
    }
    result

  private inline def logicalExtremum[T](
      layout: Layout,
      inline read: Int => T
  )(inline combine: (T, T) => T): T =
    var initialized = false
    var result: T = null.asInstanceOf[T]
    foreachLogical(layout) { physical =>
      val value = read(physical)
      if !initialized then
        result = value
        initialized = true
      else result = combine(result, value)
    }
    result

  private inline def logicalArg[T](
      layout: Layout,
      inline read: Int => T,
      inline less: (T, T) => Boolean
  ): Int =
    var initialized = false
    var best: T = null.asInstanceOf[T]
    var bestIndex = 0
    var logical = 0
    foreachLogical(layout) { physical =>
      val value = read(physical)
      if !initialized then
        best = value
        initialized = true
      else if less(value, best) then
        best = value
        bestIndex = logical
      logical += 1
    }
    bestIndex

  private inline def logicalArgFloat(
      layout: Layout,
      inline read: Int => Float,
      minimum: Boolean
  ): Int =
    var initialized = false
    var best = 0.0f
    var bestIndex = 0
    var logical = 0
    var foundNaN = false
    foreachLogical(layout) { physical =>
      val value = read(physical)
      if !initialized then
        best = value
        initialized = true
        foundNaN = value.isNaN
      else if !foundNaN && value.isNaN then
        best = value
        bestIndex = logical
        foundNaN = true
      else if !foundNaN && (if minimum then value < best else value > best) then
        best = value
        bestIndex = logical
      logical += 1
    }
    bestIndex

  private inline def logicalArgDouble(
      layout: Layout,
      inline read: Int => Double,
      minimum: Boolean
  ): Int =
    var initialized = false
    var best = 0.0
    var bestIndex = 0
    var logical = 0
    var foundNaN = false
    foreachLogical(layout) { physical =>
      val value = read(physical)
      if !initialized then
        best = value
        initialized = true
        foundNaN = value.isNaN
      else if !foundNaN && value.isNaN then
        best = value
        bestIndex = logical
        foundNaN = true
      else if !foundNaN && (if minimum then value < best else value > best) then
        best = value
        bestIndex = logical
      logical += 1
    }
    bestIndex

  private def pairwiseFloatStorage(storage: FloatStorage, layout: Layout): Float =
    if layout.size == 0 then 0.0f
    else if layout.isCContiguous then
      PlatformReduction.pairwiseContiguousFloat(
        storage,
        layout.offset,
        layout.size,
        floatScratch((layout.size + PairwiseBlockSize - 1) / PairwiseBlockSize)
      )
    else pairwiseFloat(layout, storage.raw.apply)

  private def pairwiseDoubleStorage(storage: DoubleStorage, layout: Layout): Double =
    if layout.size == 0 then 0.0
    else if layout.isCContiguous then
      PlatformReduction.pairwiseContiguousDouble(
        storage,
        layout.offset,
        layout.size,
        doubleScratch((layout.size + PairwiseBlockSize - 1) / PairwiseBlockSize)
      )
    else pairwiseDouble(layout, storage.raw.apply)

  private inline def pairwiseFloat(layout: Layout, inline read: Int => Float): Float =
    if layout.size == 0 then 0.0f
    else if layout.isCContiguous then pairwiseFloatUnit(layout.offset, layout.size, read)
    else if layout.rank == 1 then
      pairwiseFloatStrided(layout.offset, layout.strides(0), layout.shape(0), read)
    else if layout.rank == 2 then pairwiseFloatRank2(layout, read)
    else pairwiseFloatGeneral(layout, read)

  private inline def pairwiseDouble(layout: Layout, inline read: Int => Double): Double =
    if layout.size == 0 then 0.0
    else if layout.isCContiguous then pairwiseDoubleUnit(layout.offset, layout.size, read)
    else if layout.rank == 1 then
      pairwiseDoubleStrided(layout.offset, layout.strides(0), layout.shape(0), read)
    else if layout.rank == 2 then pairwiseDoubleRank2(layout, read)
    else pairwiseDoubleGeneral(layout, read)

  private inline def pairwiseDoubleFromFloat(
      layout: Layout,
      inline read: Int => Float
  ): Double =
    if layout.size == 0 then 0.0
    else if layout.isCContiguous then pairwiseDoubleUnitFromFloat(layout.offset, layout.size, read)
    else if layout.rank == 1 then
      pairwiseDoubleStridedFromFloat(layout.offset, layout.strides(0), layout.shape(0), read)
    else if layout.rank == 2 then pairwiseDoubleRank2FromFloat(layout, read)
    else pairwiseDoubleGeneralFromFloat(layout, read)

  private inline def pairwiseFloatUnit(
      offset: Int,
      size: Int,
      inline read: Int => Float
  ): Float =
    val blocks = floatScratch((size + PairwiseBlockSize - 1) / PairwiseBlockSize)
    var block = 0
    var index = 0
    var physical = offset
    while index < size do
      var sum = 0.0f
      val until = math.min(index + PairwiseBlockSize, size)
      while index < until do
        sum = (sum + read(physical)).toFloat
        physical += 1
        index += 1
      blocks(block) = sum
      block += 1
    mergeFloatBlocks(blocks, block)

  private inline def pairwiseDoubleUnit(
      offset: Int,
      size: Int,
      inline read: Int => Double
  ): Double =
    val blocks = doubleScratch((size + PairwiseBlockSize - 1) / PairwiseBlockSize)
    var block = 0
    var index = 0
    var physical = offset
    while index < size do
      var sum = 0.0
      val until = math.min(index + PairwiseBlockSize, size)
      while index < until do
        sum += read(physical)
        physical += 1
        index += 1
      blocks(block) = sum
      block += 1
    mergeDoubleBlocks(blocks, block)

  private inline def pairwiseDoubleUnitFromFloat(
      offset: Int,
      size: Int,
      inline read: Int => Float
  ): Double =
    val blocks = doubleScratch((size + PairwiseBlockSize - 1) / PairwiseBlockSize)
    var block = 0
    var index = 0
    var physical = offset
    while index < size do
      var sum = 0.0
      val until = math.min(index + PairwiseBlockSize, size)
      while index < until do
        sum += read(physical).toDouble
        physical += 1
        index += 1
      blocks(block) = sum
      block += 1
    mergeDoubleBlocks(blocks, block)

  private inline def pairwiseFloatStrided(
      offset: Int,
      stride: Int,
      length: Int,
      inline read: Int => Float
  ): Float =
    val blockCount = (length + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = floatScratch(blockCount)
    fillFloatStridedBlocks(offset, stride, length, blocks, read)
    mergeFloatBlocks(blocks, blockCount)

  private inline def pairwiseDoubleStrided(
      offset: Int,
      stride: Int,
      length: Int,
      inline read: Int => Double
  ): Double =
    val blockCount = (length + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = doubleScratch(blockCount)
    fillDoubleStridedBlocks(offset, stride, length, blocks, read)
    mergeDoubleBlocks(blocks, blockCount)

  private inline def pairwiseDoubleStridedFromFloat(
      offset: Int,
      stride: Int,
      length: Int,
      inline read: Int => Float
  ): Double =
    val blockCount = (length + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = doubleScratch(blockCount)
    fillDoubleStridedBlocksFromFloat(offset, stride, length, blocks, read)
    mergeDoubleBlocks(blocks, blockCount)

  private inline def fillFloatStridedBlocks(
      offset: Int,
      stride: Int,
      length: Int,
      blocks: Array[Float],
      inline read: Int => Float
  ): Unit =
    var block = 0
    var index = 0
    var physical = offset
    while index < length do
      var sum = 0.0f
      val until = math.min(index + PairwiseBlockSize, length)
      while index < until do
        sum = (sum + read(physical)).toFloat
        physical += stride
        index += 1
      blocks(block) = sum
      block += 1

  private inline def fillDoubleStridedBlocks(
      offset: Int,
      stride: Int,
      length: Int,
      blocks: Array[Double],
      inline read: Int => Double
  ): Unit =
    var block = 0
    var index = 0
    var physical = offset
    while index < length do
      var sum = 0.0
      val until = math.min(index + PairwiseBlockSize, length)
      while index < until do
        sum += read(physical)
        physical += stride
        index += 1
      blocks(block) = sum
      block += 1

  private inline def fillDoubleStridedBlocksFromFloat(
      offset: Int,
      stride: Int,
      length: Int,
      blocks: Array[Double],
      inline read: Int => Float
  ): Unit =
    var block = 0
    var index = 0
    var physical = offset
    while index < length do
      var sum = 0.0
      val until = math.min(index + PairwiseBlockSize, length)
      while index < until do
        sum += read(physical).toDouble
        physical += stride
        index += 1
      blocks(block) = sum
      block += 1

  private inline def pairwiseFloatRank2(layout: Layout, inline read: Int => Float): Float =
    pairwiseFloatGeneral(layout, read)

  private inline def pairwiseDoubleRank2(layout: Layout, inline read: Int => Double): Double =
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    val size = layout.size
    val blocks = doubleScratch((size + PairwiseBlockSize - 1) / PairwiseBlockSize)
    var block = 0
    var within = 0
    var sum = 0.0
    var row = 0
    while row < rows do
      var physical = offset + row * rowStride
      var col = 0
      while col < cols do
        sum += read(physical)
        within += 1
        if within == PairwiseBlockSize then
          blocks(block) = sum
          block += 1
          within = 0
          sum = 0.0
        physical += colStride
        col += 1
      row += 1
    if within > 0 then
      blocks(block) = sum
      block += 1
    mergeDoubleBlocks(blocks, block)

  private inline def pairwiseDoubleRank2FromFloat(
      layout: Layout,
      inline read: Int => Float
  ): Double =
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    val size = layout.size
    val blocks = doubleScratch((size + PairwiseBlockSize - 1) / PairwiseBlockSize)
    var block = 0
    var within = 0
    var sum = 0.0
    var row = 0
    while row < rows do
      var physical = offset + row * rowStride
      var col = 0
      while col < cols do
        sum += read(physical).toDouble
        within += 1
        if within == PairwiseBlockSize then
          blocks(block) = sum
          block += 1
          within = 0
          sum = 0.0
        physical += colStride
        col += 1
      row += 1
    if within > 0 then
      blocks(block) = sum
      block += 1
    mergeDoubleBlocks(blocks, block)

  private inline def pairwiseFloatGeneral(layout: Layout, inline read: Int => Float): Float =
    val blocks = floatScratch((layout.size + PairwiseBlockSize - 1) / PairwiseBlockSize)
    var block = 0
    var within = 0
    var sum = 0.0f
    foreachLogical(layout) { physical =>
      sum = (sum + read(physical)).toFloat
      within += 1
      if within == PairwiseBlockSize then
        blocks(block) = sum
        block += 1
        within = 0
        sum = 0.0f
    }
    if within > 0 then
      blocks(block) = sum
      block += 1
    mergeFloatBlocks(blocks, block)

  private inline def pairwiseDoubleGeneral(layout: Layout, inline read: Int => Double): Double =
    val blocks = doubleScratch((layout.size + PairwiseBlockSize - 1) / PairwiseBlockSize)
    var block = 0
    var within = 0
    var sum = 0.0
    foreachLogical(layout) { physical =>
      sum += read(physical)
      within += 1
      if within == PairwiseBlockSize then
        blocks(block) = sum
        block += 1
        within = 0
        sum = 0.0
    }
    if within > 0 then
      blocks(block) = sum
      block += 1
    mergeDoubleBlocks(blocks, block)

  private inline def pairwiseDoubleGeneralFromFloat(
      layout: Layout,
      inline read: Int => Float
  ): Double =
    val blocks = doubleScratch((layout.size + PairwiseBlockSize - 1) / PairwiseBlockSize)
    var block = 0
    var within = 0
    var sum = 0.0
    foreachLogical(layout) { physical =>
      sum += read(physical).toDouble
      within += 1
      if within == PairwiseBlockSize then
        blocks(block) = sum
        block += 1
        within = 0
        sum = 0.0
    }
    if within > 0 then
      blocks(block) = sum
      block += 1
    mergeDoubleBlocks(blocks, block)

  private def mergeFloatBlocks(blocks: Array[Float], initialCount: Int): Float =
    mergeFloatBlocksAt(blocks, 0, initialCount)

  private def mergeFloatBlocksAt(blocks: Array[Float], offset: Int, initialCount: Int): Float =
    var count = initialCount
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        blocks(offset + write) = (blocks(offset + read) + blocks(offset + read + 1)).toFloat
        read += 2
        write += 1
      if read < count then
        blocks(offset + write) = blocks(offset + read)
        write += 1
      count = write
    if initialCount == 0 then 0.0f else blocks(offset)

  private def mergeDoubleBlocks(blocks: Array[Double], initialCount: Int): Double =
    mergeDoubleBlocksAt(blocks, 0, initialCount)

  private def mergeDoubleBlocksAt(blocks: Array[Double], offset: Int, initialCount: Int): Double =
    var count = initialCount
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        blocks(offset + write) = blocks(offset + read) + blocks(offset + read + 1)
        read += 2
        write += 1
      if read < count then
        blocks(offset + write) = blocks(offset + read)
        write += 1
      count = write
    if initialCount == 0 then 0.0 else blocks(offset)

  private inline def axisFold[T](
      plan: ReductionPlan,
      inline read: Int => T,
      inline write: (Int, T) => Unit,
      identity: T
  )(inline combine: (T, T) => T): Unit =
    foreachAxisBase(plan) { (base, output) =>
      var result = identity
      var index = 0
      var physical = base
      while index < plan.reducedLength do
        result = combine(result, read(physical))
        physical += plan.reducedStride
        index += 1
      write(output, result)
    }

  private inline def axisExtremum[T](
      plan: ReductionPlan,
      inline read: Int => T,
      inline write: (Int, T) => Unit
  )(inline combine: (T, T) => T): Unit =
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength > 0 then
        var result = read(base)
        var index = 1
        var physical = base + plan.reducedStride
        while index < plan.reducedLength do
          result = combine(result, read(physical))
          physical += plan.reducedStride
          index += 1
        write(output, result)
    }

  private inline def axisArg[T](
      plan: ReductionPlan,
      inline read: Int => T,
      inline write: (Int, Int) => Unit
  )(inline better: (T, T) => Boolean): Unit =
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength > 0 then
        var best = read(base)
        var bestIndex = 0
        var index = 1
        var physical = base + plan.reducedStride
        while index < plan.reducedLength do
          val value = read(physical)
          if better(value, best) then
            best = value
            bestIndex = index
          physical += plan.reducedStride
          index += 1
        write(output, bestIndex)
    }

  private inline def axisArgFloat(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Int) => Unit,
      minimum: Boolean
  ): Unit =
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength > 0 then
        var best = read(base)
        var bestIndex = 0
        var foundNaN = best.isNaN
        var index = 1
        var physical = base + plan.reducedStride
        while index < plan.reducedLength do
          val value = read(physical)
          if !foundNaN && value.isNaN then
            best = value
            bestIndex = index
            foundNaN = true
          else if !foundNaN && (if minimum then value < best else value > best) then
            best = value
            bestIndex = index
          physical += plan.reducedStride
          index += 1
        write(output, bestIndex)
    }

  private inline def axisArgDouble(
      plan: ReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Int) => Unit,
      minimum: Boolean
  ): Unit =
    foreachAxisBase(plan) { (base, output) =>
      if plan.reducedLength > 0 then
        var best = read(base)
        var bestIndex = 0
        var foundNaN = best.isNaN
        var index = 1
        var physical = base + plan.reducedStride
        while index < plan.reducedLength do
          val value = read(physical)
          if !foundNaN && value.isNaN then
            best = value
            bestIndex = index
            foundNaN = true
          else if !foundNaN && (if minimum then value < best else value > best) then
            best = value
            bestIndex = index
          physical += plan.reducedStride
          index += 1
        write(output, bestIndex)
    }

  private def axisPairwiseFloatStorage(
      source: FloatStorage,
      output: FloatStorage,
      plan: ReductionPlan
  ): Unit =
    if plan.source.rank == 2 then
      if plan.axis == 0 then PlatformReduction.axis0PairwiseFloat(source, output, plan)
      else PlatformReduction.axis1PairwiseFloat(source, output, plan)
    else axisPairwiseFloat(plan, source.raw.apply, output.raw.update)

  private def axisPairwiseDoubleStorage(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  ): Unit =
    if plan.source.rank == 2 then
      if plan.axis == 0 then PlatformReduction.axis0PairwiseDouble(source, output, plan)
      else PlatformReduction.axis1PairwiseDouble(source, output, plan)
    else axisPairwiseDouble(plan, source.raw.apply, output.raw.update)

  private inline def axisPairwiseFloat(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    if plan.source.rank == 2 then
      if plan.axis == 0 then axis0PairwiseFloatRank2(plan, read, write)
      else axis1PairwiseFloatRank2(plan, read, write)
    else
      val blockCount = (plan.reducedLength + PairwiseBlockSize - 1) / PairwiseBlockSize
      val blocks = floatScratch(blockCount)
      foreachAxisBase(plan) { (base, out) =>
        fillFloatStridedBlocks(base, plan.reducedStride, plan.reducedLength, blocks, read)
        write(out, mergeFloatBlocks(blocks, blockCount))
      }

  private inline def axisPairwiseDouble(
      plan: ReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Double) => Unit
  ): Unit =
    if plan.source.rank == 2 then
      if plan.axis == 0 then axis0PairwiseDoubleRank2(plan, read, write)
      else axis1PairwiseDoubleRank2(plan, read, write)
    else
      val blockCount = (plan.reducedLength + PairwiseBlockSize - 1) / PairwiseBlockSize
      val blocks = doubleScratch(blockCount)
      foreachAxisBase(plan) { (base, out) =>
        fillDoubleStridedBlocks(base, plan.reducedStride, plan.reducedLength, blocks, read)
        write(out, mergeDoubleBlocks(blocks, blockCount))
      }

  private inline def axis0PairwiseDoubleRank2(
      plan: ReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Double) => Unit
  ): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    if rows == 0 then
      var col = 0
      while col < cols do
        write(col, 0.0)
        col += 1
    else
      val blockCount = (rows + PairwiseBlockSize - 1) / PairwiseBlockSize
      // Primary holds row-major block tiles (hot write path); secondary holds per-column merge vector.
      val blockValues = doubleScratch(blockCount * cols)
      val partials = doubleScratch2(cols)
      var block = 0
      var rowStart = 0
      var col = 0
      while rowStart < rows do
        val rowUntil = math.min(rowStart + PairwiseBlockSize, rows)
        col = 0
        while col < cols do
          partials(col) = 0.0
          col += 1
        var row = rowStart
        while row < rowUntil do
          var physical = offset + row * rowStride
          col = 0
          while col < cols do
            partials(col) += read(physical)
            physical += colStride
            col += 1
          row += 1
        col = 0
        while col < cols do
          blockValues(block * cols + col) = partials(col)
          col += 1
        block += 1
        rowStart = rowUntil
      val scratch = partials // reuse secondary; length >= cols >= blockCount for typical matrices
      val mergeScratch =
        if scratch.length >= blockCount then scratch else doubleScratch2(blockCount)
      col = 0
      while col < cols do
        var b = 0
        while b < blockCount do
          mergeScratch(b) = blockValues(b * cols + col)
          b += 1
        write(col, mergeDoubleBlocks(mergeScratch, blockCount))
        col += 1

  private inline def axis1PairwiseDoubleRank2(
      plan: ReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Double) => Unit
  ): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    val blockCount = (cols + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = doubleScratch(blockCount)
    var row = 0
    while row < rows do
      if cols == 0 then write(row, 0.0)
      else
        fillDoubleStridedBlocks(offset + row * rowStride, colStride, cols, blocks, read)
        write(row, mergeDoubleBlocks(blocks, blockCount))
      row += 1

  private inline def axis0PairwiseFloatRank2(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    if rows == 0 then
      var col = 0
      while col < cols do
        write(col, 0.0f)
        col += 1
    else
      val blockCount = (rows + PairwiseBlockSize - 1) / PairwiseBlockSize
      val blockValues = floatScratch(blockCount * cols)
      val partials = floatScratch2(cols)
      var block = 0
      var rowStart = 0
      var col = 0
      while rowStart < rows do
        val rowUntil = math.min(rowStart + PairwiseBlockSize, rows)
        col = 0
        while col < cols do
          partials(col) = 0.0f
          col += 1
        var row = rowStart
        while row < rowUntil do
          var physical = offset + row * rowStride
          col = 0
          while col < cols do
            partials(col) = (partials(col) + read(physical)).toFloat
            physical += colStride
            col += 1
          row += 1
        col = 0
        while col < cols do
          blockValues(block * cols + col) = partials(col)
          col += 1
        block += 1
        rowStart = rowUntil
      val mergeScratch =
        if partials.length >= blockCount then partials else floatScratch2(blockCount)
      col = 0
      while col < cols do
        var b = 0
        while b < blockCount do
          mergeScratch(b) = blockValues(b * cols + col)
          b += 1
        write(col, mergeFloatBlocks(mergeScratch, blockCount))
        col += 1

  private inline def axis1PairwiseFloatRank2(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val offset = layout.offset
    val blockCount = (cols + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = floatScratch(blockCount)
    var row = 0
    while row < rows do
      if cols == 0 then write(row, 0.0f)
      else
        fillFloatStridedBlocks(offset + row * rowStride, colStride, cols, blocks, read)
        write(row, mergeFloatBlocks(blocks, blockCount))
      row += 1

  private inline def axisPairwiseMeanFloat(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    val blockCount = (plan.reducedLength + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = doubleScratch(blockCount)
    foreachAxisBase(plan) { (base, out) =>
      if plan.reducedLength == 0 then write(out, Float.NaN)
      else
        fillDoubleStridedBlocksFromFloat(
          base,
          plan.reducedStride,
          plan.reducedLength,
          blocks,
          read
        )
        write(out, (mergeDoubleBlocks(blocks, blockCount) / plan.reducedLength.toDouble).toFloat)
    }

  private inline def axisPairwiseMeanDouble(
      plan: ReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Double) => Unit
  ): Unit =
    val blockCount = (plan.reducedLength + PairwiseBlockSize - 1) / PairwiseBlockSize
    val blocks = doubleScratch(blockCount)
    foreachAxisBase(plan) { (base, out) =>
      if plan.reducedLength == 0 then write(out, Double.NaN)
      else
        fillDoubleStridedBlocks(base, plan.reducedStride, plan.reducedLength, blocks, read)
        write(out, mergeDoubleBlocks(blocks, blockCount) / plan.reducedLength.toDouble)
    }

  private inline def foreachLogical(
      layout: Layout
  )(inline body: Int => Unit): Unit =
    if layout.size == 0 then ()
    else if layout.rank == 0 then body(layout.offset)
    else if layout.isCContiguous then
      var physical = layout.offset
      val end = physical + layout.size
      while physical < end do
        body(physical)
        physical += 1
    else if layout.rank == 1 then
      var index = 0
      var physical = layout.offset
      val stride = layout.strides(0)
      val length = layout.shape(0)
      while index < length do
        body(physical)
        physical += stride
        index += 1
    else if layout.rank == 2 then
      val rows = layout.shape(0)
      val cols = layout.shape(1)
      val rowStride = layout.strides(0)
      val colStride = layout.strides(1)
      val offset = layout.offset
      var row = 0
      while row < rows do
        var physical = offset + row * rowStride
        var col = 0
        while col < cols do
          body(physical)
          physical += colStride
          col += 1
        row += 1
    else
      val counters = new Array[Int](layout.rank)
      var physical = layout.offset
      var visited = 0
      while visited < layout.size do
        body(physical)
        visited += 1
        if visited < layout.size then
          var axis = layout.rank - 1
          var advanced = false
          while axis >= 0 && !advanced do
            if counters(axis) + 1 < layout.shape(axis) then
              counters(axis) += 1
              physical += layout.strides(axis)
              advanced = true
            else
              physical -= counters(axis) * layout.strides(axis)
              counters(axis) = 0
              axis -= 1

  private inline def foreachAxisBase(
      plan: ReductionPlan
  )(inline body: (Int, Int) => Unit): Unit =
    if plan.outputSize > 0 then
      if plan.outerShape.isEmpty then body(plan.source.offset, 0)
      else
        val counters = new Array[Int](plan.outerShape.length)
        var base = plan.source.offset
        var output = 0
        while output < plan.outputSize do
          body(base, output)
          output += 1
          if output < plan.outputSize then
            var axis = plan.outerShape.length - 1
            var advanced = false
            while axis >= 0 && !advanced do
              if counters(axis) + 1 < plan.outerShape(axis) then
                counters(axis) += 1
                base += plan.outerStrides(axis)
                advanced = true
              else
                base -= counters(axis) * plan.outerStrides(axis)
                counters(axis) = 0
                axis -= 1
