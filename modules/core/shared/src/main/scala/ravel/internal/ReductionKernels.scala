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

  private inline def pairwiseBlockCount(length: Int): Int =
    if length == 0 then 0 else 1 + (length - 1) / PairwiseBlockSize

  def all(source: Storage[Boolean], layout: Layout): Boolean =
    booleanFold(source.asInstanceOf[BooleanStorage], layout, identity = true)(_ && _)

  def any(source: Storage[Boolean], layout: Layout): Boolean =
    booleanFold(source.asInstanceOf[BooleanStorage], layout, identity = false)(_ || _)

  def countTrue(source: Storage[Boolean], layout: Layout): Int =
    val boolean = source.asInstanceOf[BooleanStorage]
    var count = 0
    if layout.isPhysicallyDense then
      val offset = layout.minimumPhysicalAddress
      var index = 0
      while index < layout.size do
        if PlatformBoolean.get(boolean, offset + index) then count += 1
        index += 1
    else
      foreachLogical(layout) { physical =>
        if PlatformBoolean.get(boolean, physical) then count += 1
      }
    count

  def sum[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: IntStorage => sumIntStorage(x, layout)
      case x: LongStorage => sumLongStorage(x, layout)
      case x: FloatStorage => pairwiseFloatStorage(x, layout)
      case x: DoubleStorage => pairwiseDoubleStorage(x, layout)
      case _ => throw new UnsupportedOperationException("sum requires Int, Long, Float, or Double")
    ).asInstanceOf[A]

  def product[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: IntStorage => productIntStorage(x, layout)
      case x: LongStorage => productLongStorage(x, layout)
      case x: FloatStorage => logicalFold(layout, x.raw.apply, 1.0f)((a, b) => (a * b).toFloat)
      case x: DoubleStorage => logicalFold(layout, x.raw.apply, 1.0)(_ * _)
      case _ =>
        throw new UnsupportedOperationException("product requires Int, Long, Float, or Double")
    ).asInstanceOf[A]

  def minimum[A](source: Storage[A], layout: Layout): A =
    if layout.size == 0 then throw EmptyReduction("min")
    (source match
      case x: ByteStorage => logicalExtremum(layout, x.raw.apply)(math.min(_, _).toByte)
      case x: UInt8Storage => logicalExtremum(layout, x.getRaw)(minUInt8)
      case x: ShortStorage => logicalExtremum(layout, x.raw.apply)(math.min(_, _).toShort)
      case x: UInt16Storage => logicalExtremum(layout, x.getRaw)(minUInt16)
      case x: IntStorage => logicalExtremum(layout, x.raw.apply)(math.min)
      case x: LongStorage => logicalExtremum(layout, x.raw.apply)(math.min)
      case x: FloatStorage => logicalExtremum(layout, x.raw.apply)((a, b) => math.min(a, b).toFloat)
      case x: DoubleStorage => optimizedDoubleExtremum(layout, x.raw.apply)(math.min)
      case _ => throw new UnsupportedOperationException("min requires an ordered numeric dtype")
    ).asInstanceOf[A]

  def maximum[A](source: Storage[A], layout: Layout): A =
    if layout.size == 0 then throw EmptyReduction("max")
    (source match
      case x: ByteStorage => logicalExtremum(layout, x.raw.apply)(math.max(_, _).toByte)
      case x: UInt8Storage => logicalExtremum(layout, x.getRaw)(maxUInt8)
      case x: ShortStorage => logicalExtremum(layout, x.raw.apply)(math.max(_, _).toShort)
      case x: UInt16Storage => logicalExtremum(layout, x.getRaw)(maxUInt16)
      case x: IntStorage => logicalExtremum(layout, x.raw.apply)(math.max)
      case x: LongStorage => logicalExtremum(layout, x.raw.apply)(math.max)
      case x: FloatStorage => logicalExtremum(layout, x.raw.apply)((a, b) => math.max(a, b).toFloat)
      case x: DoubleStorage => optimizedDoubleExtremum(layout, x.raw.apply)(math.max)
      case _ => throw new UnsupportedOperationException("max requires an ordered numeric dtype")
    ).asInstanceOf[A]

  def argMinimum[A](source: Storage[A], layout: Layout): Int =
    if layout.size == 0 then throw EmptyReduction("argMin")
    source match
      case x: ByteStorage => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: UInt8Storage => logicalArg(layout, x.getRaw, less = unsignedByteLess)
      case x: ShortStorage => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: UInt16Storage => logicalArg(layout, x.getRaw, less = unsignedShortLess)
      case x: IntStorage => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: LongStorage => logicalArg(layout, x.raw.apply, less = _ < _)
      case x: FloatStorage => logicalArgFloat(layout, x.raw.apply, minimum = true)
      case x: DoubleStorage => logicalArgDouble(layout, x.raw.apply, minimum = true)
      case _ => throw new UnsupportedOperationException("argMin requires an ordered numeric dtype")

  def argMaximum[A](source: Storage[A], layout: Layout): Int =
    if layout.size == 0 then throw EmptyReduction("argMax")
    source match
      case x: ByteStorage => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: UInt8Storage => logicalArg(layout, x.getRaw, less = unsignedByteGreater)
      case x: ShortStorage => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: UInt16Storage => logicalArg(layout, x.getRaw, less = unsignedShortGreater)
      case x: IntStorage => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: LongStorage => logicalArg(layout, x.raw.apply, less = _ > _)
      case x: FloatStorage => logicalArgFloat(layout, x.raw.apply, minimum = false)
      case x: DoubleStorage => logicalArgDouble(layout, x.raw.apply, minimum = false)
      case _ => throw new UnsupportedOperationException("argMax requires an ordered numeric dtype")

  def mean[A](source: Storage[A], layout: Layout): A =
    (source match
      case x: FloatStorage =>
        if layout.size == 0 then Float.NaN
        else (pairwiseDoubleFromFloatStorage(x, layout) / layout.size.toDouble).toFloat
      case x: DoubleStorage =>
        if layout.size == 0 then Double.NaN
        else pairwiseDoubleStorage(x, layout) / layout.size.toDouble
      case _ => throw new UnsupportedOperationException("mean requires Float or Double")
    ).asInstanceOf[A]

  def sumAsLong(source: Storage[Int], layout: Layout): Long =
    val raw = source.asInstanceOf[IntStorage].raw
    logicalFold(layout, raw.apply, 0L)((acc, value) => acc + value.toLong)

  def sumAsDouble(source: Storage[Float], layout: Layout): Double =
    pairwiseDoubleFromFloatStorage(source.asInstanceOf[FloatStorage], layout)

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
        axisProductDoubleStorage(x, z, plan)
      case _ =>
        throw new UnsupportedOperationException("product requires matching arithmetic storage")

  def minimumAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then throw EmptyReduction("min")
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min(_, _).toByte)
      case (x: UInt8Storage, z: UInt8Storage) =>
        axisExtremum(plan, x.getRaw, z.setRaw)(minUInt8)
      case (x: ShortStorage, z: ShortStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min(_, _).toShort)
      case (x: UInt16Storage, z: UInt16Storage) =>
        axisExtremum(plan, x.getRaw, z.setRaw)(minUInt16)
      case (x: IntStorage, z: IntStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case (x: LongStorage, z: LongStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case (x: FloatStorage, z: FloatStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)((a, b) => math.min(a, b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisExtremumDoubleStorage(x, z, plan)(math.min)
      case _ => throw new UnsupportedOperationException("min requires matching ordered storage")

  def maximumAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then throw EmptyReduction("max")
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max(_, _).toByte)
      case (x: UInt8Storage, z: UInt8Storage) =>
        axisExtremum(plan, x.getRaw, z.setRaw)(maxUInt8)
      case (x: ShortStorage, z: ShortStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max(_, _).toShort)
      case (x: UInt16Storage, z: UInt16Storage) =>
        axisExtremum(plan, x.getRaw, z.setRaw)(maxUInt16)
      case (x: IntStorage, z: IntStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case (x: LongStorage, z: LongStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case (x: FloatStorage, z: FloatStorage) =>
        axisExtremum(plan, x.raw.apply, z.raw.update)((a, b) => math.max(a, b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisExtremumDoubleStorage(x, z, plan)(math.max)
      case _ => throw new UnsupportedOperationException("max requires matching ordered storage")

  def meanAxis[A](source: Storage[A], output: Storage[A], plan: ReductionPlan): Unit =
    (source, output) match
      case (x: FloatStorage, z: FloatStorage) =>
        axisPairwiseMeanFloat(plan, x.raw.apply, z.raw.update)
      case (x: DoubleStorage, z: DoubleStorage) =>
        axisPairwiseMeanDoubleStorage(x, z, plan)
      case _ => throw new UnsupportedOperationException("mean requires matching floating storage")

  def allAxis(
      source: Storage[Boolean],
      output: Storage[Boolean],
      plan: ReductionPlan
  ): Unit =
    val input = source.asInstanceOf[BooleanStorage]
    val target = output.asInstanceOf[BooleanStorage]
    axisFold(
      plan,
      physical => PlatformBoolean.get(input, physical),
      (index, value) => PlatformBoolean.set(target, index, value),
      identity = true
    )(_ && _)

  def anyAxis(
      source: Storage[Boolean],
      output: Storage[Boolean],
      plan: ReductionPlan
  ): Unit =
    val input = source.asInstanceOf[BooleanStorage]
    val target = output.asInstanceOf[BooleanStorage]
    axisFold(
      plan,
      physical => PlatformBoolean.get(input, physical),
      (index, value) => PlatformBoolean.set(target, index, value),
      identity = false
    )(_ || _)

  def countTrueAxis(
      source: Storage[Boolean],
      output: Storage[Int],
      plan: ReductionPlan
  ): Unit =
    axisCountTrue(
      source.asInstanceOf[BooleanStorage],
      output.asInstanceOf[IntStorage],
      plan
    )

  def sumAxes[A](source: Storage[A], output: Storage[A], plan: MultiReductionPlan): Unit =
    (source, output) match
      case (x: IntStorage, z: IntStorage) =>
        multiAxisFold(plan, x.raw.apply, z.raw.update, 0)(_ + _)
      case (x: LongStorage, z: LongStorage) =>
        multiAxisFold(plan, x.raw.apply, z.raw.update, 0L)(_ + _)
      case (x: FloatStorage, z: FloatStorage) =>
        multiAxisPairwiseFloat(plan, x.raw.apply, z.raw.update)
      case (x: DoubleStorage, z: DoubleStorage) =>
        multiAxisPairwiseDouble(plan, x.raw.apply, z.raw.update)
      case _ => throw new UnsupportedOperationException("sum requires matching arithmetic storage")

  def productAxes[A](source: Storage[A], output: Storage[A], plan: MultiReductionPlan): Unit =
    (source, output) match
      case (x: IntStorage, z: IntStorage) =>
        multiAxisFold(plan, x.raw.apply, z.raw.update, 1)(_ * _)
      case (x: LongStorage, z: LongStorage) =>
        multiAxisFold(plan, x.raw.apply, z.raw.update, 1L)(_ * _)
      case (x: FloatStorage, z: FloatStorage) =>
        multiAxisFold(plan, x.raw.apply, z.raw.update, 1.0f)((a, b) => (a * b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        multiAxisFold(plan, x.raw.apply, z.raw.update, 1.0)(_ * _)
      case _ =>
        throw new UnsupportedOperationException("product requires matching arithmetic storage")

  def minimumAxes[A](source: Storage[A], output: Storage[A], plan: MultiReductionPlan): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then throw EmptyReduction("min")
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.min(_, _).toByte)
      case (x: UInt8Storage, z: UInt8Storage) =>
        multiAxisExtremum(plan, x.getRaw, z.setRaw)(minUInt8)
      case (x: ShortStorage, z: ShortStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.min(_, _).toShort)
      case (x: UInt16Storage, z: UInt16Storage) =>
        multiAxisExtremum(plan, x.getRaw, z.setRaw)(minUInt16)
      case (x: IntStorage, z: IntStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case (x: LongStorage, z: LongStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case (x: FloatStorage, z: FloatStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)((a, b) => math.min(a, b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.min)
      case _ => throw new UnsupportedOperationException("min requires matching ordered storage")

  def maximumAxes[A](source: Storage[A], output: Storage[A], plan: MultiReductionPlan): Unit =
    if plan.reducedLength == 0 && plan.outputSize > 0 then throw EmptyReduction("max")
    (source, output) match
      case (x: ByteStorage, z: ByteStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.max(_, _).toByte)
      case (x: UInt8Storage, z: UInt8Storage) =>
        multiAxisExtremum(plan, x.getRaw, z.setRaw)(maxUInt8)
      case (x: ShortStorage, z: ShortStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.max(_, _).toShort)
      case (x: UInt16Storage, z: UInt16Storage) =>
        multiAxisExtremum(plan, x.getRaw, z.setRaw)(maxUInt16)
      case (x: IntStorage, z: IntStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case (x: LongStorage, z: LongStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case (x: FloatStorage, z: FloatStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)((a, b) => math.max(a, b).toFloat)
      case (x: DoubleStorage, z: DoubleStorage) =>
        multiAxisExtremum(plan, x.raw.apply, z.raw.update)(math.max)
      case _ => throw new UnsupportedOperationException("max requires matching ordered storage")

  def meanAxes[A](source: Storage[A], output: Storage[A], plan: MultiReductionPlan): Unit =
    (source, output) match
      case (x: FloatStorage, z: FloatStorage) =>
        multiAxisPairwiseMeanFloat(plan, x.raw.apply, z.raw.update)
      case (x: DoubleStorage, z: DoubleStorage) =>
        multiAxisPairwiseMeanDouble(plan, x.raw.apply, z.raw.update)
      case _ => throw new UnsupportedOperationException("mean requires matching floating storage")

  def allAxes(
      source: Storage[Boolean],
      output: Storage[Boolean],
      plan: MultiReductionPlan
  ): Unit =
    val input = source.asInstanceOf[BooleanStorage]
    val target = output.asInstanceOf[BooleanStorage]
    multiAxisFold(
      plan,
      physical => PlatformBoolean.get(input, physical),
      (index, value) => PlatformBoolean.set(target, index, value),
      identity = true
    )(_ && _)

  def anyAxes(
      source: Storage[Boolean],
      output: Storage[Boolean],
      plan: MultiReductionPlan
  ): Unit =
    val input = source.asInstanceOf[BooleanStorage]
    val target = output.asInstanceOf[BooleanStorage]
    multiAxisFold(
      plan,
      physical => PlatformBoolean.get(input, physical),
      (index, value) => PlatformBoolean.set(target, index, value),
      identity = false
    )(_ || _)

  def countTrueAxes(
      source: Storage[Boolean],
      output: Storage[Int],
      plan: MultiReductionPlan
  ): Unit =
    multiAxisCountTrue(
      source.asInstanceOf[BooleanStorage],
      output.asInstanceOf[IntStorage],
      plan
    )

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
      case x: UInt8Storage =>
        if minimum then axisArg(plan, x.getRaw, target.raw.update)(unsignedByteLess)
        else axisArg(plan, x.getRaw, target.raw.update)(unsignedByteGreater)
      case x: ShortStorage =>
        if minimum then axisArg(plan, x.raw.apply, target.raw.update)(_ < _)
        else axisArg(plan, x.raw.apply, target.raw.update)(_ > _)
      case x: UInt16Storage =>
        if minimum then axisArg(plan, x.getRaw, target.raw.update)(unsignedShortLess)
        else axisArg(plan, x.getRaw, target.raw.update)(unsignedShortGreater)
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

  private inline def unsignedByteLess(left: Byte, right: Byte): Boolean =
    (left & 0xff) < (right & 0xff)

  private inline def unsignedByteGreater(left: Byte, right: Byte): Boolean =
    (left & 0xff) > (right & 0xff)

  private inline def unsignedShortLess(left: Short, right: Short): Boolean =
    (left & 0xffff) < (right & 0xffff)

  private inline def unsignedShortGreater(left: Short, right: Short): Boolean =
    (left & 0xffff) > (right & 0xffff)

  private inline def minUInt8(left: Byte, right: Byte): Byte =
    if unsignedByteLess(left, right) then left else right

  private inline def maxUInt8(left: Byte, right: Byte): Byte =
    if unsignedByteGreater(left, right) then left else right

  private inline def minUInt16(left: Short, right: Short): Short =
    if unsignedShortLess(left, right) then left else right

  private inline def maxUInt16(left: Short, right: Short): Short =
    if unsignedShortGreater(left, right) then left else right

  // Fixed-width arithmetic is associative modulo 2^N, so independent lanes
  // preserve the public overflow result while breaking the dependency chain.
  private def sumIntStorage(storage: IntStorage, layout: Layout): Int =
    if layout.isPhysicallyDense then
      val raw = storage.raw
      val offset = layout.minimumPhysicalAddress
      val size = layout.size
      var sum0 = 0
      var sum1 = 0
      var sum2 = 0
      var sum3 = 0
      var sum4 = 0
      var sum5 = 0
      var sum6 = 0
      var sum7 = 0
      var index = 0
      while index + 7 < size do
        sum0 += raw(offset + index)
        sum1 += raw(offset + index + 1)
        sum2 += raw(offset + index + 2)
        sum3 += raw(offset + index + 3)
        sum4 += raw(offset + index + 4)
        sum5 += raw(offset + index + 5)
        sum6 += raw(offset + index + 6)
        sum7 += raw(offset + index + 7)
        index += 8
      var result =
        ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7))
      while index < size do
        result += raw(offset + index)
        index += 1
      result
    else logicalFold(layout, storage.raw.apply, 0)(_ + _)

  private def sumLongStorage(storage: LongStorage, layout: Layout): Long =
    if layout.isPhysicallyDense then
      val raw = storage.raw
      val offset = layout.minimumPhysicalAddress
      val size = layout.size
      var sum0 = 0L
      var sum1 = 0L
      var sum2 = 0L
      var sum3 = 0L
      var sum4 = 0L
      var sum5 = 0L
      var sum6 = 0L
      var sum7 = 0L
      var index = 0
      while index + 7 < size do
        sum0 += raw(offset + index)
        sum1 += raw(offset + index + 1)
        sum2 += raw(offset + index + 2)
        sum3 += raw(offset + index + 3)
        sum4 += raw(offset + index + 4)
        sum5 += raw(offset + index + 5)
        sum6 += raw(offset + index + 6)
        sum7 += raw(offset + index + 7)
        index += 8
      var result =
        ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7))
      while index < size do
        result += raw(offset + index)
        index += 1
      result
    else logicalFold(layout, storage.raw.apply, 0L)(_ + _)

  private def productIntStorage(storage: IntStorage, layout: Layout): Int =
    if layout.isPhysicallyDense then
      val raw = storage.raw
      val offset = layout.minimumPhysicalAddress
      val size = layout.size
      var product0 = 1
      var product1 = 1
      var product2 = 1
      var product3 = 1
      var product4 = 1
      var product5 = 1
      var product6 = 1
      var product7 = 1
      var index = 0
      while index + 7 < size do
        product0 *= raw(offset + index)
        product1 *= raw(offset + index + 1)
        product2 *= raw(offset + index + 2)
        product3 *= raw(offset + index + 3)
        product4 *= raw(offset + index + 4)
        product5 *= raw(offset + index + 5)
        product6 *= raw(offset + index + 6)
        product7 *= raw(offset + index + 7)
        index += 8
      var result =
        ((product0 * product1) * (product2 * product3)) *
          ((product4 * product5) * (product6 * product7))
      while index < size do
        result *= raw(offset + index)
        index += 1
      result
    else logicalFold(layout, storage.raw.apply, 1)(_ * _)

  private def productLongStorage(storage: LongStorage, layout: Layout): Long =
    if layout.isPhysicallyDense then
      val raw = storage.raw
      val offset = layout.minimumPhysicalAddress
      val size = layout.size
      var product0 = 1L
      var product1 = 1L
      var product2 = 1L
      var product3 = 1L
      var product4 = 1L
      var product5 = 1L
      var product6 = 1L
      var product7 = 1L
      var index = 0
      while index + 7 < size do
        product0 *= raw(offset + index)
        product1 *= raw(offset + index + 1)
        product2 *= raw(offset + index + 2)
        product3 *= raw(offset + index + 3)
        product4 *= raw(offset + index + 4)
        product5 *= raw(offset + index + 5)
        product6 *= raw(offset + index + 6)
        product7 *= raw(offset + index + 7)
        index += 8
      var result =
        ((product0 * product1) * (product2 * product3)) *
          ((product4 * product5) * (product6 * product7))
      while index < size do
        result *= raw(offset + index)
        index += 1
      result
    else logicalFold(layout, storage.raw.apply, 1L)(_ * _)

  private inline def optimizedDoubleExtremum(
      layout: Layout,
      inline read: Int => Double
  )(inline combine: (Double, Double) => Double): Double =
    // The extrema contract observes NaN propagation and signed zero, both of
    // which are invariant under this regrouping. NaN payload order is not API.
    if layout.isPhysicallyDense then
      val offset = layout.minimumPhysicalAddress
      val size = layout.size
      if size < 4 then logicalExtremum(layout, read)(combine)
      else
        var value0 = read(offset)
        var value1 = read(offset + 1)
        var value2 = read(offset + 2)
        var value3 = read(offset + 3)
        var index = 4
        while index + 3 < size do
          value0 = combine(value0, read(offset + index))
          value1 = combine(value1, read(offset + index + 1))
          value2 = combine(value2, read(offset + index + 2))
          value3 = combine(value3, read(offset + index + 3))
          index += 4
        var result = combine(combine(value0, value1), combine(value2, value3))
        while index < size do
          result = combine(result, read(offset + index))
          index += 1
        result
    else if layout.rank == 2 then
      val rows = layout.shape(0)
      val cols = layout.shape(1)
      val rowStride = layout.strides(0)
      val colStride = layout.strides(1)
      val seed = read(layout.offset)
      var value0 = seed
      var value1 = seed
      var value2 = seed
      var value3 = seed
      if math.abs(colStride.toLong) <= math.abs(rowStride.toLong) then
        var row = 0
        while row < rows do
          var physical = layout.offset + row * rowStride
          var col = 0
          while col + 3 < cols do
            value0 = combine(value0, read(physical))
            value1 = combine(value1, read(physical + colStride))
            value2 = combine(value2, read(physical + 2 * colStride))
            value3 = combine(value3, read(physical + 3 * colStride))
            physical += 4 * colStride
            col += 4
          while col < cols do
            value0 = combine(value0, read(physical))
            physical += colStride
            col += 1
          row += 1
      else
        var col = 0
        while col < cols do
          var physical = layout.offset + col * colStride
          var row = 0
          while row + 3 < rows do
            value0 = combine(value0, read(physical))
            value1 = combine(value1, read(physical + rowStride))
            value2 = combine(value2, read(physical + 2 * rowStride))
            value3 = combine(value3, read(physical + 3 * rowStride))
            physical += 4 * rowStride
            row += 4
          while row < rows do
            value0 = combine(value0, read(physical))
            physical += rowStride
            row += 1
          col += 1
      combine(combine(value0, value1), combine(value2, value3))
    else logicalExtremum(layout, read)(combine)

  private def axisProductDoubleStorage(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  ): Unit =
    // Stream rank-2 axis 0 in the friendliest physical direction without
    // changing the row order of multiplication inside any output fiber.
    if plan.source.rank == 2 && plan.axis == 0 then
      val layout = plan.source
      val rows = layout.shape(0)
      val cols = layout.shape(1)
      val rowStride = layout.strides(0)
      val colStride = layout.strides(1)
      val raw = source.raw
      val out = output.raw
      var col = 0
      while col < cols do
        out(col) = 1.0
        col += 1
      if math.abs(colStride.toLong) <= math.abs(rowStride.toLong) then
        var row = 0
        while row < rows do
          var physical = layout.offset + row * rowStride
          col = 0
          while col < cols do
            out(col) *= raw(physical)
            physical += colStride
            col += 1
          row += 1
      else
        col = 0
        while col < cols do
          var product = 1.0
          var physical = layout.offset + col * colStride
          var row = 0
          while row < rows do
            product *= raw(physical)
            physical += rowStride
            row += 1
          out(col) = product
          col += 1
    else axisFold(plan, source.raw.apply, output.raw.update, 1.0)(_ * _)

  private inline def axisExtremumDoubleStorage(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  )(inline combine: (Double, Double) => Double): Unit =
    if plan.source.rank != 2 then axisExtremum(plan, source.raw.apply, output.raw.update)(combine)
    else if plan.axis == 0 then axis0ExtremumDoubleRank2(source, output, plan)(combine)
    else axis1ExtremumDoubleRank2(source, output, plan)(combine)

  private inline def axis0ExtremumDoubleRank2(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  )(inline combine: (Double, Double) => Double): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val raw = source.raw
    val out = output.raw
    if math.abs(colStride.toLong) <= math.abs(rowStride.toLong) then
      var physical = layout.offset
      var col = 0
      while col < cols do
        out(col) = raw(physical)
        physical += colStride
        col += 1
      var row = 1
      while row + 3 < rows do
        var physical0 = layout.offset + row * rowStride
        var physical1 = physical0 + rowStride
        var physical2 = physical1 + rowStride
        var physical3 = physical2 + rowStride
        col = 0
        while col < cols do
          val pair0 = combine(out(col), raw(physical0))
          val pair1 = combine(raw(physical1), raw(physical2))
          out(col) = combine(combine(pair0, pair1), raw(physical3))
          physical0 += colStride
          physical1 += colStride
          physical2 += colStride
          physical3 += colStride
          col += 1
        row += 4
      while row < rows do
        physical = layout.offset + row * rowStride
        col = 0
        while col < cols do
          out(col) = combine(out(col), raw(physical))
          physical += colStride
          col += 1
        row += 1
    else
      var col = 0
      while col < cols do
        var physical = layout.offset + col * colStride
        var result = raw(physical)
        var row = 1
        physical += rowStride
        while row < rows do
          result = combine(result, raw(physical))
          physical += rowStride
          row += 1
        out(col) = result
        col += 1

  private inline def axis1ExtremumDoubleRank2(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  )(inline combine: (Double, Double) => Double): Unit =
    val layout = plan.source
    val rows = layout.shape(0)
    val cols = layout.shape(1)
    val rowStride = layout.strides(0)
    val colStride = layout.strides(1)
    val raw = source.raw
    val out = output.raw
    var row = 0
    while row + 3 < rows do
      var physical0 = layout.offset + row * rowStride
      var physical1 = physical0 + rowStride
      var physical2 = physical1 + rowStride
      var physical3 = physical2 + rowStride
      var result0 = raw(physical0)
      var result1 = raw(physical1)
      var result2 = raw(physical2)
      var result3 = raw(physical3)
      var col = 1
      physical0 += colStride
      physical1 += colStride
      physical2 += colStride
      physical3 += colStride
      while col < cols do
        result0 = combine(result0, raw(physical0))
        result1 = combine(result1, raw(physical1))
        result2 = combine(result2, raw(physical2))
        result3 = combine(result3, raw(physical3))
        physical0 += colStride
        physical1 += colStride
        physical2 += colStride
        physical3 += colStride
        col += 1
      out(row) = result0
      out(row + 1) = result1
      out(row + 2) = result2
      out(row + 3) = result3
      row += 4
    while row < rows do
      var physical = layout.offset + row * rowStride
      var result = raw(physical)
      var col = 1
      physical += colStride
      while col < cols do
        result = combine(result, raw(physical))
        physical += colStride
        col += 1
      out(row) = result
      row += 1

  private def axisPairwiseMeanDoubleStorage(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  ): Unit =
    if plan.source.rank == 2 then
      if plan.axis == 0 then platformAxis0PairwiseDouble(source, output, plan)
      else PlatformReduction.axis1PairwiseDouble(source, output, plan)
      val divisor = plan.reducedLength.toDouble
      val out = output.raw
      var index = 0
      while index < plan.outputSize do
        out(index) = out(index) / divisor
        index += 1
    else axisPairwiseMeanDouble(plan, source.raw.apply, output.raw.update)

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

  private inline def booleanFold(
      storage: BooleanStorage,
      layout: Layout,
      identity: Boolean
  )(inline combine: (Boolean, Boolean) => Boolean): Boolean =
    var result = identity
    if layout.isPhysicallyDense then
      val offset = layout.minimumPhysicalAddress
      var index = 0
      while index < layout.size do
        result = combine(result, PlatformBoolean.get(storage, offset + index))
        index += 1
    else
      foreachLogical(layout) { physical =>
        result = combine(result, PlatformBoolean.get(storage, physical))
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
        floatScratch(pairwiseBlockCount(layout.size))
      )
    else pairwiseFloat(layout, storage.raw.apply)

  private def pairwiseDoubleStorage(storage: DoubleStorage, layout: Layout): Double =
    if layout.size == 0 then 0.0
    else if layout.isCContiguous then
      PlatformReduction.pairwiseContiguousDouble(
        storage,
        layout.offset,
        layout.size,
        doubleScratch(pairwiseBlockCount(layout.size))
      )
    else pairwiseDouble(layout, storage.raw.apply)

  private def pairwiseDoubleFromFloatStorage(storage: FloatStorage, layout: Layout): Double =
    if layout.size == 0 then 0.0
    else if layout.isCContiguous then
      PlatformReduction.pairwiseContiguousDoubleFromFloat(
        storage,
        layout.offset,
        layout.size,
        doubleScratch(pairwiseBlockCount(layout.size))
      )
    else pairwiseDoubleFromFloat(layout, storage.raw.apply)

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
    val blocks = floatScratch(pairwiseBlockCount(size))
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
    val blocks = doubleScratch(pairwiseBlockCount(size))
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
    val blocks = doubleScratch(pairwiseBlockCount(size))
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
    val blockCount = pairwiseBlockCount(length)
    val blocks = floatScratch(blockCount)
    fillFloatStridedBlocks(offset, stride, length, blocks, read)
    mergeFloatBlocks(blocks, blockCount)

  private inline def pairwiseDoubleStrided(
      offset: Int,
      stride: Int,
      length: Int,
      inline read: Int => Double
  ): Double =
    val blockCount = pairwiseBlockCount(length)
    val blocks = doubleScratch(blockCount)
    fillDoubleStridedBlocks(offset, stride, length, blocks, read)
    mergeDoubleBlocks(blocks, blockCount)

  private inline def pairwiseDoubleStridedFromFloat(
      offset: Int,
      stride: Int,
      length: Int,
      inline read: Int => Float
  ): Double =
    val blockCount = pairwiseBlockCount(length)
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
    val blocks = doubleScratch(pairwiseBlockCount(size))
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
    val blocks = doubleScratch(pairwiseBlockCount(size))
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
    val blocks = floatScratch(pairwiseBlockCount(layout.size))
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
    val blocks = doubleScratch(pairwiseBlockCount(layout.size))
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
    val blocks = doubleScratch(pairwiseBlockCount(layout.size))
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

  private inline def multiAxisFold[T](
      plan: MultiReductionPlan,
      inline read: Int => T,
      inline write: (Int, T) => Unit,
      identity: T
  )(inline combine: (T, T) => T): Unit =
    val reducedCounters = new Array[Int](plan.reducedShape.length)
    foreachMultiAxisBase(plan) { (base, output) =>
      var result = identity
      foreachReducedPhysical(plan, base, reducedCounters) { physical =>
        result = combine(result, read(physical))
      }
      write(output, result)
    }

  private def multiAxisCountTrue(
      source: BooleanStorage,
      output: IntStorage,
      plan: MultiReductionPlan
  ): Unit =
    val reducedCounters = new Array[Int](plan.reducedShape.length)
    foreachMultiAxisBase(plan) { (base, target) =>
      var count = 0
      foreachReducedPhysical(plan, base, reducedCounters) { physical =>
        if PlatformBoolean.get(source, physical) then count += 1
      }
      output.raw(target) = count
    }

  private inline def multiAxisExtremum[T](
      plan: MultiReductionPlan,
      inline read: Int => T,
      inline write: (Int, T) => Unit
  )(inline combine: (T, T) => T): Unit =
    val reducedCounters = new Array[Int](plan.reducedShape.length)
    foreachMultiAxisBase(plan) { (base, output) =>
      var initialized = false
      var result: T = null.asInstanceOf[T]
      foreachReducedPhysical(plan, base, reducedCounters) { physical =>
        val value = read(physical)
        if initialized then result = combine(result, value)
        else
          result = value
          initialized = true
      }
      if initialized then write(output, result)
    }

  private inline def multiAxisPairwiseFloat(
      plan: MultiReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    val blockCount = pairwiseBlockCount(plan.reducedLength)
    val blocks = floatScratch(blockCount)
    val reducedCounters = new Array[Int](plan.reducedShape.length)
    foreachMultiAxisBase(plan) { (base, output) =>
      var block = 0
      var within = 0
      var sum = 0.0f
      foreachReducedPhysical(plan, base, reducedCounters) { physical =>
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
      write(output, mergeFloatBlocks(blocks, block))
    }

  private inline def multiAxisPairwiseDouble(
      plan: MultiReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Double) => Unit
  ): Unit =
    val blockCount = pairwiseBlockCount(plan.reducedLength)
    val blocks = doubleScratch(blockCount)
    val reducedCounters = new Array[Int](plan.reducedShape.length)
    foreachMultiAxisBase(plan) { (base, output) =>
      var block = 0
      var within = 0
      var sum = 0.0
      foreachReducedPhysical(plan, base, reducedCounters) { physical =>
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
      write(output, mergeDoubleBlocks(blocks, block))
    }

  private inline def multiAxisPairwiseMeanFloat(
      plan: MultiReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    val blockCount = pairwiseBlockCount(plan.reducedLength)
    val blocks = doubleScratch(blockCount)
    val reducedCounters = new Array[Int](plan.reducedShape.length)
    foreachMultiAxisBase(plan) { (base, output) =>
      if plan.reducedLength == 0 then write(output, Float.NaN)
      else
        var block = 0
        var within = 0
        var sum = 0.0
        foreachReducedPhysical(plan, base, reducedCounters) { physical =>
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
        write(
          output,
          (mergeDoubleBlocks(blocks, block) / plan.reducedLength.toDouble).toFloat
        )
    }

  private inline def multiAxisPairwiseMeanDouble(
      plan: MultiReductionPlan,
      inline read: Int => Double,
      inline write: (Int, Double) => Unit
  ): Unit =
    val blockCount = pairwiseBlockCount(plan.reducedLength)
    val blocks = doubleScratch(blockCount)
    val reducedCounters = new Array[Int](plan.reducedShape.length)
    foreachMultiAxisBase(plan) { (base, output) =>
      if plan.reducedLength == 0 then write(output, Double.NaN)
      else
        var block = 0
        var within = 0
        var sum = 0.0
        foreachReducedPhysical(plan, base, reducedCounters) { physical =>
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
        write(output, mergeDoubleBlocks(blocks, block) / plan.reducedLength.toDouble)
    }

  private inline def foreachReducedPhysical(
      plan: MultiReductionPlan,
      base: Int,
      counters: Array[Int]
  )(inline body: Int => Unit): Unit =
    java.util.Arrays.fill(counters, 0)
    var physical = base
    var visited = 0
    while visited < plan.reducedLength do
      body(physical)
      visited += 1
      if visited < plan.reducedLength then
        var axis = plan.reducedShape.length - 1
        var advanced = false
        while axis >= 0 && !advanced do
          if counters(axis) + 1 < plan.reducedShape(axis) then
            counters(axis) += 1
            physical += plan.reducedStrides(axis)
            advanced = true
          else
            physical -= counters(axis) * plan.reducedStrides(axis)
            counters(axis) = 0
            axis -= 1

  private inline def foreachMultiAxisBase(
      plan: MultiReductionPlan
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

  private def axisCountTrue(
      source: BooleanStorage,
      output: IntStorage,
      plan: ReductionPlan
  ): Unit =
    foreachAxisBase(plan) { (base, target) =>
      var count = 0
      var index = 0
      var physical = base
      while index < plan.reducedLength do
        if PlatformBoolean.get(source, physical) then count += 1
        physical += plan.reducedStride
        index += 1
      output.raw(target) = count
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
      if plan.axis == 0 then platformAxis0PairwiseFloat(source, output, plan)
      else PlatformReduction.axis1PairwiseFloat(source, output, plan)
    else axisPairwiseFloat(plan, source.raw.apply, output.raw.update)

  private def axisPairwiseDoubleStorage(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  ): Unit =
    if plan.source.rank == 2 then
      if plan.axis == 0 then platformAxis0PairwiseDouble(source, output, plan)
      else PlatformReduction.axis1PairwiseDouble(source, output, plan)
    else axisPairwiseDouble(plan, source.raw.apply, output.raw.update)

  private def platformAxis0PairwiseFloat(
      source: FloatStorage,
      output: FloatStorage,
      plan: ReductionPlan
  ): Unit =
    val rows = plan.source.shape(0)
    val cols = plan.source.shape(1)
    val blockCount = pairwiseBlockCount(rows)
    PlatformReduction.axis0PairwiseFloat(
      source,
      output,
      plan,
      floatScratch(blockCount * cols),
      floatScratch2(math.max(cols, blockCount))
    )

  private def platformAxis0PairwiseDouble(
      source: DoubleStorage,
      output: DoubleStorage,
      plan: ReductionPlan
  ): Unit =
    val rows = plan.source.shape(0)
    val cols = plan.source.shape(1)
    val blockCount = pairwiseBlockCount(rows)
    PlatformReduction.axis0PairwiseDouble(
      source,
      output,
      plan,
      doubleScratch(blockCount * cols),
      doubleScratch2(math.max(cols, blockCount))
    )

  private inline def axisPairwiseFloat(
      plan: ReductionPlan,
      inline read: Int => Float,
      inline write: (Int, Float) => Unit
  ): Unit =
    if plan.source.rank == 2 then
      if plan.axis == 0 then axis0PairwiseFloatRank2(plan, read, write)
      else axis1PairwiseFloatRank2(plan, read, write)
    else
      val blockCount = pairwiseBlockCount(plan.reducedLength)
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
      val blockCount = pairwiseBlockCount(plan.reducedLength)
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
      val blockCount = pairwiseBlockCount(rows)
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
    val blockCount = pairwiseBlockCount(cols)
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
      val blockCount = pairwiseBlockCount(rows)
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
    val blockCount = pairwiseBlockCount(cols)
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
    val blockCount = pairwiseBlockCount(plan.reducedLength)
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
    val blockCount = pairwiseBlockCount(plan.reducedLength)
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
