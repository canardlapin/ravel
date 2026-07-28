package ravel.bench

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations.*
import ravel.*
import ravel.DType.given
import ravel.internal.*
import scala.compiletime.uninitialized

/** Lower-bound controls for the public access-pattern suite.
  *
  * These are deliberately not alternative product implementations. They separate output allocation,
  * loop mechanics, public planning, and the fixed pairwise reduction schedule so optimization work
  * has a falsifiable target.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 7, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(2)
@State(Scope.Thread)
class AccessPatternControls:
  @Param(Array("256", "1024"))
  var side: Int = 0

  private var fixture: AccessPatternControlFixture = uninitialized

  @Setup(Level.Trial)
  def setup(): Unit =
    fixture = new AccessPatternControlFixture(side)

  @Benchmark
  def ravel_output_allocation_full(): AnyRef =
    ProbeApi.allocate[Double](fixture.fullSize).asInstanceOf[AnyRef]

  @Benchmark
  def ravel_output_allocation_half(): AnyRef =
    ProbeApi.allocate[Double](fixture.halfSize).asInstanceOf[AnyRef]

  @Benchmark
  def raw_contiguous_add_reuse(): Double =
    AccessPatternControlLoops.add(
      fixture.left,
      fixture.right,
      fixture.fullOutput
    )
    fixture.fullOutput(fixture.fullSize - 1)

  @Benchmark
  def probe_storage_add_reuse(): Double =
    ProbeApi.add(
      fixture.ravel.left.storage,
      fixture.ravel.right.storage,
      fixture.storageOutput,
      fixture.fullSize
    )
    ProbeApi.get(fixture.storageOutput, fixture.fullSize - 1)

  @Benchmark
  def inline_storage_add_reuse(): Double =
    InlineStoragePrototype.add(
      fixture.ravel.left.storage,
      fixture.ravel.right.storage,
      fixture.storageOutput,
      fixture.fullSize
    )
    ProbeApi.get(fixture.storageOutput, fixture.fullSize - 1)

  @Benchmark
  def opaque_contiguous_add_reuse(): Double =
    OpaqueDoubleBuffer.add(
      fixture.opaqueLeft,
      fixture.opaqueRight,
      fixture.opaqueOutput,
      fixture.fullSize
    )
    OpaqueDoubleBuffer.get(fixture.opaqueOutput, fixture.fullSize - 1)

  @Benchmark
  def raw_contiguous_add_allocate(): Array[Double] =
    val output = new Array[Double](fixture.fullSize)
    AccessPatternControlLoops.add(fixture.left, fixture.right, output)
    output

  @Benchmark
  def ravel_add_into_reuse(): Double =
    kernel.addInto(fixture.ravel.left, fixture.ravel.right, fixture.mutableOutput)
    fixture.mutableOutput(side - 1, side - 1)

  @Benchmark
  def raw_inner_stride_copy_allocate(): Array[Double] =
    AccessPatternControlLoops.copyInnerStride(fixture.left, side)

  @Benchmark
  def raw_transpose_copy_allocate(): Array[Double] =
    AccessPatternControlLoops.copyTranspose(fixture.left, side)

  @Benchmark
  def raw_exact_sum_reuse(): Double =
    AccessPatternControlLoops.pairwiseSum(
      fixture.left,
      offset = 0,
      stride = 1,
      length = fixture.fullSize,
      fixture.fullSumScratch
    )

  @Benchmark
  def raw_exact_sum_ilp_reuse(): Double =
    AccessPatternControlLoops.pairwiseSumIlp(
      fixture.left,
      offset = 0,
      stride = 1,
      length = fixture.fullSize,
      fixture.fullSumScratch
    )

  @Benchmark
  def raw_exact_sum_allocate(): Double =
    val scratch =
      new Array[Double](AccessPatternControlLoops.blockCount(fixture.fullSize))
    AccessPatternControlLoops.pairwiseSum(
      fixture.left,
      offset = 0,
      stride = 1,
      length = fixture.fullSize,
      scratch
    )

  @Benchmark
  def raw_axis0_sum_reuse(): Double =
    AccessPatternControlLoops.axis0Sum(
      fixture.left,
      side,
      fixture.axisOutput,
      fixture.axisScratch
    )
    fixture.axisOutput(side - 1)

  @Benchmark
  def raw_axis1_sum_reuse(): Double =
    AccessPatternControlLoops.axis1Sum(
      fixture.left,
      side,
      fixture.axisOutput,
      fixture.axisScratch
    )
    fixture.axisOutput(side - 1)

  @Benchmark
  def raw_axis1_sum_ilp_reuse(): Double =
    AccessPatternControlLoops.axis1SumIlp(
      fixture.left,
      side,
      fixture.axisOutput,
      fixture.axisScratch,
      fixture.axisScratch1,
      fixture.axisScratch2,
      fixture.axisScratch3
    )
    fixture.axisOutput(side - 1)

  @Benchmark
  def raw_float_contiguous_add_reuse(): Float =
    AccessPatternControlLoops.addFloat(
      fixture.floatLeft,
      fixture.floatRight,
      fixture.floatOutput
    )
    fixture.floatOutput(fixture.fullSize - 1)

  @Benchmark
  def raw_int_contiguous_add_reuse(): Int =
    AccessPatternControlLoops.addInt(
      fixture.intLeft,
      fixture.intRight,
      fixture.intOutput
    )
    fixture.intOutput(fixture.fullSize - 1)

private final class AccessPatternControlFixture(val side: Int):
  require(side > 0 && side % 2 == 0, s"side must be a positive even integer, got $side")

  val ravel = new AccessPatternFixture(side)
  val fullSize: Int = Math.multiplyExact(side, side)
  val halfSize: Int = fullSize / 2
  val left: Array[Double] =
    ravel.left.storage.asInstanceOf[DoubleStorage].raw
  val right: Array[Double] =
    ravel.right.storage.asInstanceOf[DoubleStorage].raw
  val fullOutput = new Array[Double](fullSize)
  val storageOutput: Storage[Double] = ProbeApi.allocate[Double](fullSize)
  val opaqueLeft: OpaqueDoubleBuffer.Buffer = OpaqueDoubleBuffer.wrap(left)
  val opaqueRight: OpaqueDoubleBuffer.Buffer = OpaqueDoubleBuffer.wrap(right)
  val opaqueOutput: OpaqueDoubleBuffer.Buffer =
    OpaqueDoubleBuffer.wrap(new Array[Double](fullSize))
  val axisOutput = new Array[Double](side)
  val fullSumScratch =
    new Array[Double](AccessPatternControlLoops.blockCount(fullSize))
  val axisScratch =
    new Array[Double](AccessPatternControlLoops.blockCount(side))
  val axisScratch1 =
    new Array[Double](AccessPatternControlLoops.blockCount(side))
  val axisScratch2 =
    new Array[Double](AccessPatternControlLoops.blockCount(side))
  val axisScratch3 =
    new Array[Double](AccessPatternControlLoops.blockCount(side))
  val mutableOutput: MutableNDArray[Double, Rank[2]] =
    MutableNDArray.zeros[Double, Rank[2]](Shape(side, side))

  val floatLeft: Array[Float] =
    Array.tabulate(fullSize)(index => left(index).toFloat)
  val floatRight: Array[Float] =
    Array.tabulate(fullSize)(index => right(index).toFloat)
  val floatOutput = new Array[Float](fullSize)

  val intLeft: Array[Int] =
    Array.tabulate(fullSize)(index => (left(index) * 16.0).toInt)
  val intRight: Array[Int] =
    Array.tabulate(fullSize)(index => (right(index) * 32.0).toInt)
  val intOutput = new Array[Int](fullSize)

  validateControls()

  private def validateControls(): Unit =
    AccessPatternControlLoops.add(left, right, fullOutput)
    requireSameElements(
      AccessPatternWorkloads.contiguousAdd(ravel),
      fullOutput,
      "raw contiguous add"
    )

    requireSameElements(
      ravel.innerStrideLeft,
      AccessPatternControlLoops.copyInnerStride(left, side),
      "raw inner-stride copy"
    )
    requireSameElements(
      ravel.transposedLeft,
      AccessPatternControlLoops.copyTranspose(left, side),
      "raw transpose copy"
    )

    val rawSum =
      AccessPatternControlLoops.pairwiseSum(
        left,
        offset = 0,
        stride = 1,
        length = fullSize,
        fullSumScratch
      )
    require(
      java.lang.Double.doubleToRawLongBits(rawSum) ==
        java.lang.Double.doubleToRawLongBits(ravel.left.sum),
      "raw exact-schedule sum differs from Ravel"
    )
    val rawIlpSum =
      AccessPatternControlLoops.pairwiseSumIlp(
        left,
        offset = 0,
        stride = 1,
        length = fullSize,
        fullSumScratch
      )
    require(
      java.lang.Double.doubleToRawLongBits(rawIlpSum) ==
        java.lang.Double.doubleToRawLongBits(rawSum),
      "raw exact ILP sum differs from the serial exact schedule"
    )

    AccessPatternControlLoops.axis0Sum(left, side, axisOutput, axisScratch)
    requireSameElements(ravel.left.sum(axis = 0), axisOutput, "raw axis-0 sum")
    AccessPatternControlLoops.axis1Sum(left, side, axisOutput, axisScratch)
    requireSameElements(ravel.left.sum(axis = 1), axisOutput, "raw axis-1 sum")
    AccessPatternControlLoops.axis1SumIlp(
      left,
      side,
      axisOutput,
      axisScratch,
      axisScratch1,
      axisScratch2,
      axisScratch3
    )
    requireSameElements(ravel.left.sum(axis = 1), axisOutput, "raw axis-1 exact ILP sum")

  private def requireSameElements(
      expected: NDArray[Double, ?],
      observed: Array[Double],
      label: String
  ): Unit =
    require(expected.size == observed.length, s"$label size differs")
    val iterator = expected.elementsIterator
    var index = 0
    while iterator.hasNext do
      require(
        java.lang.Double.doubleToRawLongBits(iterator.next()) ==
          java.lang.Double.doubleToRawLongBits(observed(index)),
        s"$label differs at logical index $index"
      )
      index += 1

private object InlineStoragePrototype:
  inline def add(
      left: Storage[?],
      right: Storage[?],
      output: Storage[?],
      size: Int
  ): Unit =
    (left, right, output) match
      case (x: DoubleStorage, y: DoubleStorage, z: DoubleStorage) =>
        addRaw(x.raw, y.raw, z.raw, size)
      case _ =>
        throw new IllegalArgumentException("inline storage prototype requires Double")

  private inline def addRaw(
      left: Array[Double],
      right: Array[Double],
      output: Array[Double],
      size: Int
  ): Unit =
    var index = 0
    while index < size do
      output(index) = left(index) + right(index)
      index += 1

private object OpaqueDoubleBuffer:
  opaque type Buffer = Array[Double]

  def wrap(array: Array[Double]): Buffer = array

  inline def get(buffer: Buffer, index: Int): Double =
    buffer(index)

  inline def add(
      left: Buffer,
      right: Buffer,
      output: Buffer,
      size: Int
  ): Unit =
    var index = 0
    while index < size do
      output(index) = left(index) + right(index)
      index += 1

private object AccessPatternControlLoops:
  private val PairwiseBlockSize = 128

  def blockCount(length: Int): Int =
    (length + PairwiseBlockSize - 1) / PairwiseBlockSize

  def add(
      left: Array[Double],
      right: Array[Double],
      output: Array[Double]
  ): Unit =
    var index = 0
    while index < output.length do
      output(index) = left(index) + right(index)
      index += 1

  def addFloat(
      left: Array[Float],
      right: Array[Float],
      output: Array[Float]
  ): Unit =
    var index = 0
    while index < output.length do
      output(index) = (left(index) + right(index)).toFloat
      index += 1

  def addInt(
      left: Array[Int],
      right: Array[Int],
      output: Array[Int]
  ): Unit =
    var index = 0
    while index < output.length do
      output(index) = left(index) + right(index)
      index += 1

  def copyInnerStride(source: Array[Double], side: Int): Array[Double] =
    val columns = side / 2
    val output = new Array[Double](Math.multiplyExact(side, columns))
    var row = 0
    var write = 0
    while row < side do
      var column = 0
      var read = row * side
      while column < columns do
        output(write) = source(read)
        write += 1
        read += 2
        column += 1
      row += 1
    output

  def copyTranspose(source: Array[Double], side: Int): Array[Double] =
    val output = new Array[Double](Math.multiplyExact(side, side))
    var row = 0
    var write = 0
    while row < side do
      var column = 0
      var read = row
      while column < side do
        output(write) = source(read)
        write += 1
        read += side
        column += 1
      row += 1
    output

  def pairwiseSum(
      source: Array[Double],
      offset: Int,
      stride: Int,
      length: Int,
      scratch: Array[Double]
  ): Double =
    var block = 0
    var consumed = 0
    var physical = offset
    while consumed < length do
      var sum = 0.0
      val until = math.min(consumed + PairwiseBlockSize, length)
      while consumed < until do
        sum += source(physical)
        physical += stride
        consumed += 1
      scratch(block) = sum
      block += 1
    merge(scratch, block)

  def pairwiseSumIlp(
      source: Array[Double],
      offset: Int,
      stride: Int,
      length: Int,
      scratch: Array[Double]
  ): Double =
    var block = 0
    val completeBlocks = length / PairwiseBlockSize
    while block + 3 < completeBlocks do
      val base = offset + block * PairwiseBlockSize * stride
      var sum0 = 0.0
      var sum1 = 0.0
      var sum2 = 0.0
      var sum3 = 0.0
      var index = 0
      var physical0 = base
      var physical1 = base + PairwiseBlockSize * stride
      var physical2 = base + 2 * PairwiseBlockSize * stride
      var physical3 = base + 3 * PairwiseBlockSize * stride
      while index < PairwiseBlockSize do
        sum0 += source(physical0)
        sum1 += source(physical1)
        sum2 += source(physical2)
        sum3 += source(physical3)
        physical0 += stride
        physical1 += stride
        physical2 += stride
        physical3 += stride
        index += 1
      scratch(block) = sum0
      scratch(block + 1) = sum1
      scratch(block + 2) = sum2
      scratch(block + 3) = sum3
      block += 4
    var consumed = block * PairwiseBlockSize
    var physical = offset + consumed * stride
    while consumed < length do
      var sum = 0.0
      val until = math.min(consumed + PairwiseBlockSize, length)
      while consumed < until do
        sum += source(physical)
        physical += stride
        consumed += 1
      scratch(block) = sum
      block += 1
    merge(scratch, block)

  def axis0Sum(
      source: Array[Double],
      side: Int,
      output: Array[Double],
      scratch: Array[Double]
  ): Unit =
    var column = 0
    while column < side do
      output(column) = pairwiseSum(source, column, side, side, scratch)
      column += 1

  def axis1Sum(
      source: Array[Double],
      side: Int,
      output: Array[Double],
      scratch: Array[Double]
  ): Unit =
    var row = 0
    while row < side do
      output(row) = pairwiseSum(source, row * side, 1, side, scratch)
      row += 1

  def axis1SumIlp(
      source: Array[Double],
      side: Int,
      output: Array[Double],
      scratch0: Array[Double],
      scratch1: Array[Double],
      scratch2: Array[Double],
      scratch3: Array[Double]
  ): Unit =
    val blockCount = AccessPatternControlLoops.blockCount(side)
    var row = 0
    while row + 3 < side do
      fillFourRows(
        source,
        row * side,
        side,
        scratch0,
        scratch1,
        scratch2,
        scratch3
      )
      output(row) = merge(scratch0, blockCount)
      output(row + 1) = merge(scratch1, blockCount)
      output(row + 2) = merge(scratch2, blockCount)
      output(row + 3) = merge(scratch3, blockCount)
      row += 4
    while row < side do
      output(row) = pairwiseSum(source, row * side, 1, side, scratch0)
      row += 1

  private def fillFourRows(
      source: Array[Double],
      offset: Int,
      rowStride: Int,
      scratch0: Array[Double],
      scratch1: Array[Double],
      scratch2: Array[Double],
      scratch3: Array[Double]
  ): Unit =
    var block = 0
    var index = 0
    var physical0 = offset
    var physical1 = offset + rowStride
    var physical2 = offset + 2 * rowStride
    var physical3 = offset + 3 * rowStride
    while index < rowStride do
      var sum0 = 0.0
      var sum1 = 0.0
      var sum2 = 0.0
      var sum3 = 0.0
      val until = math.min(index + PairwiseBlockSize, rowStride)
      while index < until do
        sum0 += source(physical0)
        sum1 += source(physical1)
        sum2 += source(physical2)
        sum3 += source(physical3)
        physical0 += 1
        physical1 += 1
        physical2 += 1
        physical3 += 1
        index += 1
      scratch0(block) = sum0
      scratch1(block) = sum1
      scratch2(block) = sum2
      scratch3(block) = sum3
      block += 1

  private def merge(scratch: Array[Double], initialCount: Int): Double =
    var count = initialCount
    while count > 1 do
      var read = 0
      var write = 0
      while read + 1 < count do
        scratch(write) = scratch(read) + scratch(read + 1)
        read += 2
        write += 1
      if read < count then
        scratch(write) = scratch(read)
        write += 1
      count = write
    if initialCount == 0 then 0.0 else scratch(0)
